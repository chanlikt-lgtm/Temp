import sys, os, time, json
import numpy as np
# Portable paths: resolve the audited driver relative to this file, and write
# outputs to $WAFER_OUT (default: this directory).  No hard-coded temp paths.
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('WAFER_OUT', _HERE)
os.makedirs(BASE, exist_ok=True)
sys.path.insert(0, os.path.join(_HERE, '..', 'code', 'audited_final'))
# Reuse the audited driver's EXACT converge/metrics/root logic (no re-implementation).
# NB: this is a REPRODUCIBILITY check (same solver implementation), not an
# independent CFD verification -- it cannot expose an error shared by the solver.
from equalQ_root_driver import (Solver, NU, Q_TARGET, N_GAP, ROOT_TOL,
                                nominal_dt, converge, metrics, snapshot)

# Generous step caps for the tightened (enforced) steady-state gate.  Exploratory
# bracket/secant evaluations run best-effort (require_converged=False) since they
# only steer the root; the FINAL polish is convergence-enforced.
CASES = {
 'Small':  {'pitch':0.040,'bracket':(0.55,0.82),'max0':30000,'min0':4000,'maxf':30000},
 'Medium': {'pitch':0.060,'bracket':(0.30,0.52),'max0':18000,'min0':3000,'maxf':18000},
 'Large':  {'pitch':0.080,'bracket':(0.16,0.32),'max0':14000,'min0':3000,'maxf':14000},
}

def run(label):
    cfg=CASES[label]; pitch=cfg['pitch']; lo,hi=cfg['bracket']
    t0=time.time()
    s=Solver(pitch,N_GAP); dt=nominal_dt(s)
    print(f'[{label}] grid Nx={s.g["Nx"]} Ny={s.g["Ny"]} unknowns={len(s.coords)} dt={dt:.3e}',flush=True)
    mlo,slo,hlo,rlo,nlo=converge(s,lo,dt,None,min_steps=cfg['min0'],max_steps=cfg['max0'],require_converged=False)
    mhi,shi,hhi,rhi,nhi=converge(s,hi,dt,slo,min_steps=cfg['min0'],max_steps=cfg['max0'],require_converged=False)
    print(f'[{label}] bracket Q: lo(U={lo})={mlo["Q_rack"]:.5e}  hi(U={hi})={mhi["Q_rack"]:.5e}',flush=True)
    final=None
    for it in range(8):
        ql,qh=mlo['Q_rack'],mhi['Q_rack']
        usec=lo+(Q_TARGET-ql)*(hi-lo)/(qh-ql); span=hi-lo
        umid=min(max(usec,lo+0.20*span),hi-0.20*span)
        state=slo if abs(umid-lo)<=abs(hi-umid) else shi
        mm,sm,hh,rr,nn=converge(s,umid,dt,state,min_steps=cfg['min0'],max_steps=cfg['max0'],require_converged=False)
        err=mm['Q_rack']/Q_TARGET-1
        print(f'[{label}] root{it+1} U={umid:.5f} Q_rack={mm["Q_rack"]:.6e} err%={100*err:+.4f}',flush=True)
        final=(umid,mm,sm,rr)
        if abs(err)<=ROOT_TOL: break
        if mm['Q_rack']<Q_TARGET: lo,mlo,slo=umid,mm,sm
        else: hi,mhi,shi=umid,mm,sm
    U,mm,sm,rr=final
    # final polish: convergence ENFORCED (raises if the cap is hit)
    mf,sf,hf,rf,nf=converge(s,U,dt,sm,min_steps=cfg['min0'],max_steps=cfg['maxf'],require_converged=True)
    errf=mf['Q_rack']/Q_TARGET-1
    # Enforced secant/bisection correction: a fully-settled solve can drift Q
    # past the cheap root step, so re-root on convergence-gated evaluations until
    # the accepted steady-state Q is within tolerance.
    tries=0
    while abs(errf)>ROOT_TOL and tries<4:
        tries+=1
        if mf['Q_rack']<Q_TARGET: lo,mlo,slo=U,mf,sf
        else:                     hi,mhi,shi=U,mf,sf
        ql,qh=mlo['Q_rack'],mhi['Q_rack']
        U=lo+(Q_TARGET-ql)*(hi-lo)/(qh-ql)
        U=min(max(U,lo+0.10*(hi-lo)),hi-0.10*(hi-lo))
        state=slo if abs(U-lo)<=abs(hi-U) else shi
        mf,sf,hf,rf,nf=converge(s,U,dt,state,min_steps=cfg['min0'],max_steps=cfg['maxf'],require_converged=True)
        errf=mf['Q_rack']/Q_TARGET-1
        print(f'[{label}] correct{tries} U={U:.5f} Q_rack={mf["Q_rack"]:.6e} err%={100*errf:+.4f}',flush=True)
    max_mass=max(abs(v['err_pct']) for v in mf['mass_flux'].values())
    g=s.g
    inlet_w=float(np.sum(g['dx'][g['inlet']]))
    gap_frac=float(mf['Q_rack']/mf['Q_in']) if mf['Q_in'] else float('nan')
    res={'case':label,'pitch':pitch,'U_in':U,'Q_rack':mf['Q_rack'],'Q_target':Q_TARGET,
         'Q_err_pct':100*errf,'max_mass_err_pct':max_mass,'delta_p_rack':mf['delta_p_rack'],
         'R_h':mf['R_h'],'central_peak_v':mf['central_peak_v'],'Q_in':mf['Q_in'],
         'inlet_width_total':inlet_w,'gap_inlet_fraction':gap_frac,
         'y_p_below':mf['y_p_below'],'y_p_above':mf['y_p_above'],
         'final_res':rf,'final_converged':True,'elapsed_s':time.time()-t0,
         'mass_flux':{k:v['err_pct'] for k,v in mf['mass_flux'].items()}}
    np.savez_compressed(os.path.join(BASE,f'{label.lower()}_verify.npz'),
        u=s.u,v=s.v,p=s.p,xf=g['xf'],yf=g['yf'],xc=g['xc'],yc=g['yc'],dx=g['dx'],dy=g['dy'],
        solid=g['solid'],left=g['left'],right=g['right'],centers=g['centers'],
        U_in=U,pitch=pitch,Q_target=Q_TARGET,Q_rack=mf['Q_rack'])
    with open(os.path.join(BASE,f'{label.lower()}_verify.json'),'w') as fjson:
        json.dump(res,fjson,indent=2)
    print(f'[{label}] RESULT '+json.dumps({k:res[k] for k in
        ['U_in','Q_rack','Q_err_pct','max_mass_err_pct','delta_p_rack','R_h','elapsed_s']},default=float),flush=True)
    return res

if __name__=='__main__':
    run(sys.argv[1])
