import sys, os, time, json
import numpy as np
BASE='/tmp/claude-0/-home-user-Temp/36b846dc-25a1-5185-8f16-2e39764cc5de/scratchpad/wafer'
sys.path.insert(0, BASE+'/wafer_project_all_python_code/audited_final')
# Reuse the audited driver's EXACT converge/metrics/root logic (no re-implementation).
from equalQ_root_driver import (Solver, NU, Q_TARGET, N_GAP, ROOT_TOL,
                                nominal_dt, converge, metrics, snapshot)

CASES = {
 'Small':  {'pitch':0.040,'bracket':(0.55,0.82),'max0':14500,'min0':3500},
 'Medium': {'pitch':0.060,'bracket':(0.30,0.52),'max0':9500, 'min0':2500},
 'Large':  {'pitch':0.080,'bracket':(0.16,0.32),'max0':7500, 'min0':2500},
}

def run(label):
    cfg=CASES[label]; pitch=cfg['pitch']; lo,hi=cfg['bracket']
    t0=time.time()
    s=Solver(pitch,N_GAP); dt=nominal_dt(s)
    print(f'[{label}] grid Nx={s.g["Nx"]} Ny={s.g["Ny"]} unknowns={len(s.coords)} dt={dt:.3e}',flush=True)
    mlo,slo,hlo,rlo,nlo=converge(s,lo,dt,None,min_steps=cfg['min0'],max_steps=cfg['max0'])
    mhi,shi,hhi,rhi,nhi=converge(s,hi,dt,slo,min_steps=1800,max_steps=7000)
    print(f'[{label}] bracket Q: lo(U={lo})={mlo["Q_rack"]:.5e}  hi(U={hi})={mhi["Q_rack"]:.5e}',flush=True)
    final=None
    for it in range(7):
        ql,qh=mlo['Q_rack'],mhi['Q_rack']
        usec=lo+(Q_TARGET-ql)*(hi-lo)/(qh-ql); span=hi-lo
        umid=min(max(usec,lo+0.20*span),hi-0.20*span)
        state=slo if abs(umid-lo)<=abs(hi-umid) else shi
        mm,sm,hh,rr,nn=converge(s,umid,dt,state,min_steps=1600,max_steps=6500)
        err=mm['Q_rack']/Q_TARGET-1
        print(f'[{label}] root{it+1} U={umid:.5f} Q_rack={mm["Q_rack"]:.6e} err%={100*err:+.4f}',flush=True)
        final=(umid,mm,sm,rr)
        if abs(err)<=ROOT_TOL: break
        if mm['Q_rack']<Q_TARGET: lo,mlo,slo=umid,mm,sm
        else: hi,mhi,shi=umid,mm,sm
    U,mm,sm,rr=final
    mf,sf,hf,rf,nf=converge(s,U,dt,sm,min_steps=1500,max_steps=5000)
    errf=mf['Q_rack']/Q_TARGET-1
    max_mass=max(abs(v['err_pct']) for v in mf['mass_flux'].values())
    res={'case':label,'pitch':pitch,'U_in':U,'Q_rack':mf['Q_rack'],'Q_target':Q_TARGET,
         'Q_err_pct':100*errf,'max_mass_err_pct':max_mass,'delta_p_rack':mf['delta_p_rack'],
         'R_h':mf['R_h'],'central_peak_v':mf['central_peak_v'],'Q_in':mf['Q_in'],
         'final_res':rf,'elapsed_s':time.time()-t0,
         'mass_flux':{k:v['err_pct'] for k,v in mf['mass_flux'].items()}}
    g=s.g
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
