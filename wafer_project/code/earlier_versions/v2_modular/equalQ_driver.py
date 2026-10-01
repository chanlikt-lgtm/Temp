#!/usr/bin/env python3
"""Reproducibility driver for the validated equal-Q wafer-pitch CFD study.

No fitting.  For each inlet speed trial, the 2-D incompressible Navier-Stokes
field is advanced to a steady state.  A safeguarded secant/bisection scalar
root search then adjusts inlet speed until Q_rack equals the requested target.

Examples
--------
python equalQ_driver.py --case Medium
python equalQ_driver.py --case all --out results_reproduced
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from mac_navier_stokes_core import Solver, NU

CASES = {
    "Small":  {"pitch": 0.040, "bracket": (0.62, 0.76)},
    "Medium": {"pitch": 0.060, "bracket": (0.34, 0.44)},
    "Large":  {"pitch": 0.080, "bracket": (0.21, 0.27)},
}
Q_TARGET = 0.0055463


def metrics(s: Solver):
    g=s.g; j=g["jcut"]; dx=g["dx"]
    qr=float(np.sum(s.v[j,g["gapmask"]]*dx[g["gapmask"]]))
    qin=float(np.sum(s.v[0,:]*dx))
    central=g["central"]
    cmean=float(np.sum(s.v[j,central]*dx[central])/np.sum(dx[central]))
    cpeak=float(np.max(s.v[j,central]))
    pbar=[]
    for yy in (0.10,0.28):
        jj=int(np.argmin(np.abs(g["yc"]-yy)))
        m=g["rack"] & g["fluid"][jj,:]
        pbar.append(float(np.sum(s.p[jj,m]*dx[m])/np.sum(dx[m])))
    dp=pbar[0]-pbar[1]
    mass=[]
    for yy in (0.05,0.10,0.20,0.30,0.40):
        jf=int(np.argmin(np.abs(g["yf"]-yy)))
        q=float(np.sum(s.v[jf,:]*dx))
        mass.append((yy,float(g["yf"][jf]),q,100*(q/qin-1)))
    return {
        "Q_rack": qr, "Q_in": qin,
        "mean_gap_v": qr/float(np.sum(dx[g["gapmask"]])),
        "central_mean_v": cmean, "central_peak_v": cpeak,
        "delta_p_rack": dp, "R_h": dp/qr, "mass_flux": mass,
    }


def state(s,U):
    return {"U":float(U),"u":s.u.copy(),"v":s.v.copy(),"p":s.p.copy()}


def load_scaled_state(s, st, U):
    if st is None:
        s.u.fill(0); s.v.fill(0); s.p.fill(0); return
    f=U/st["U"]
    s.u[:]=st["u"]*f; s.v[:]=st["v"]*f; s.p[:]=st["p"]*f


def converge(s,U,dt,st=None,min_steps=2000,max_steps=12000,chunk=500):
    load_scaled_state(s,st,U)
    hist=[]; total=0
    while total < max_steps:
        n=min(chunk,max_steps-total)
        r=s.solve(U,dt=dt,max_steps=n,tol=1e-14,warm=True)
        total += n
        m=metrics(s)
        hist.append((total,float(r["res"]),m["Q_rack"],m["central_peak_v"],m["delta_p_rack"]))
        if total>=min_steps and len(hist)>=3:
            a,b,c=hist[-3:]
            def rc(x,y): return abs(x-y)/(abs(y)+1e-14)
            stable=max(rc(c[2],b[2]),rc(c[3],b[3]),rc(c[4],b[4]),rc(b[2],a[2]))
            if r["res"]<3e-5 and stable<1e-3:
                break
    return metrics(s), state(s,U), float(r["res"]), total


def root_match(label, q_target=Q_TARGET, n_gap=20, qtol=7.5e-4):
    cfg=CASES[label]; s=Solver(cfg["pitch"],n_gap)
    h=min(float(s.g["dx"].min()),float(s.g["dy"].min()))
    dt=0.90*h*h/(4*NU)
    lo,hi=cfg["bracket"]
    root_history=[]

    mlo,slo,rlo,nlo=converge(s,lo,dt,None,min_steps=2500,max_steps=14000)
    root_history.append(("lo",lo,mlo["Q_rack"],rlo,nlo))
    mhi,shi,rhi,nhi=converge(s,hi,dt,slo,min_steps=1800,max_steps=8000)
    root_history.append(("hi",hi,mhi["Q_rack"],rhi,nhi))
    if not (mlo["Q_rack"] <= q_target <= mhi["Q_rack"]):
        raise RuntimeError(f"Initial bracket failed for {label}: {mlo['Q_rack']}, {mhi['Q_rack']}")

    final=None
    for it in range(8):
        ql,qh=mlo["Q_rack"],mhi["Q_rack"]
        us=lo+(q_target-ql)*(hi-lo)/(qh-ql)
        span=hi-lo
        u=min(max(us,lo+0.15*span),hi-0.15*span)  # safeguard toward bisection
        base=slo if abs(u-lo)<abs(hi-u) else shi
        mm,sm,rr,nn=converge(s,u,dt,base,min_steps=1500,max_steps=7000)
        root_history.append((f"root{it+1}",u,mm["Q_rack"],rr,nn))
        final=(u,mm,sm,rr)
        if abs(mm["Q_rack"]/q_target-1) <= qtol:
            break
        if mm["Q_rack"]<q_target:
            lo,mlo,slo=u,mm,sm
        else:
            hi,mhi,shi=u,mm,sm

    U,mf,sf,rf=final
    # final steady continuation; if root drift is appreciable, one bisection polish
    mf,sf,rf,nf=converge(s,U,dt,sf,min_steps=1500,max_steps=5000)
    if abs(mf["Q_rack"]/q_target-1)>2*qtol:
        if mf["Q_rack"]<q_target: lo,mlo,slo=U,mf,sf
        else: hi,mhi,shi=U,mf,sf
        U=.5*(lo+hi); base=slo if abs(U-lo)<abs(hi-U) else shi
        mf,sf,rf,nf=converge(s,U,dt,base,min_steps=1800,max_steps=7000)
        root_history.append(("bisection_polish",U,mf["Q_rack"],rf,nf))
    return s,U,dt,mf,rf,root_history


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--case",choices=["Small","Medium","Large","all"],default="all")
    ap.add_argument("--out",default="reproduced_equalQ")
    ap.add_argument("--q",type=float,default=Q_TARGET)
    ap.add_argument("--gap-cells",type=int,default=20)
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    labels=list(CASES) if args.case=="all" else [args.case]
    rows=[]
    for label in labels:
        s,U,dt,m,rr,hist=root_match(label,args.q,args.gap_cells)
        g=s.g
        rows.append({"case":label,"pitch":CASES[label]["pitch"],"n_gap_cells":args.gap_cells,
            "Nx":g["Nx"],"Ny":g["Ny"],"dt":dt,"U_in":U,"Q_target":args.q,"Q_rack":m["Q_rack"],
            "Q_error_pct":100*(m["Q_rack"]/args.q-1),"Q_in":m["Q_in"],"mean_gap_v":m["mean_gap_v"],
            "central_mean_v":m["central_mean_v"],"central_peak_v":m["central_peak_v"],
            "delta_p_rack":m["delta_p_rack"],"R_h":m["R_h"],"final_residual":rr,
            "max_mass_error_pct":max(abs(z[3]) for z in m["mass_flux"])})
        pd.DataFrame(hist,columns=["stage","U_in","Q_rack","residual","steps"]).to_csv(os.path.join(args.out,f"{label.lower()}_root_history.csv"),index=False)
        np.savez_compressed(os.path.join(args.out,f"{label.lower()}_field.npz"),u=s.u,v=s.v,p=s.p,xf=g["xf"],yf=g["yf"],xc=g["xc"],yc=g["yc"],solid=g["solid"],U_in=U,dt=dt)
    pd.DataFrame(rows).to_csv(os.path.join(args.out,"production_summary.csv"),index=False)
    with open(os.path.join(args.out,"settings.json"),"w") as f:
        json.dump({"Q_target":args.q,"n_gap_cells":args.gap_cells,"nu":NU,"no_fitting":True},f,indent=2)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
