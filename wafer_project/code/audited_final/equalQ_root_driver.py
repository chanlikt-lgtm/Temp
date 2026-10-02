import os, sys, json, time, multiprocessing as mp
import numpy as np
import pandas as pd
# Import the solver from this file's own directory, independent of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mac_cfd_adaptive import Solver, NU

# Output directory is configurable via $WAFER_OUT; defaults to a folder in the
# current working directory.  Created lazily in __main__ so that importing this
# module (to reuse metrics/converge) has no filesystem side effects.
OUT=os.environ.get('WAFER_OUT', os.path.join(os.getcwd(), 'wafer_mac_equalQ_v2'))
Q_TARGET=0.0055463
CASES={
 'Small': {'pitch':0.040,'bracket':(0.55,0.82)},
 'Medium':{'pitch':0.060,'bracket':(0.30,0.52)},
 'Large': {'pitch':0.080,'bracket':(0.16,0.32)},
}
N_GAP=20
ROOT_TOL=7.5e-4   # 0.075% on Q


def nominal_dt(s):
    h=min(float(s.g['dx'].min()),float(s.g['dy'].min()))
    return 0.90*h*h/(4*NU)

def snapshot(s,U):
    return {'U':float(U),'u':s.u.copy(),'v':s.v.copy(),'p':s.p.copy()}

def restore_scaled(s,state,U):
    if state is None:
        s.u.fill(0.0); s.v.fill(0.0); s.p.fill(0.0)
        return
    f=float(U/state['U'])
    s.u[:]=state['u']*f
    s.v[:]=state['v']*f
    # kinematic pressure has viscous and inertial contributions; linear scaling is a benign warm start only.
    s.p[:]=state['p']*f

def metrics(s):
    g=s.g; j=g['jcut']; dx=g['dx']
    q_rack=float(np.sum(s.v[j,g['gapmask']]*dx[g['gapmask']]))
    q_total=float(np.sum(s.v[j,:]*dx))
    q_in=float(np.sum(s.v[0,:]*dx))
    central=g['central']
    vcent=s.v[j,central]
    mean_c=float(np.sum(vcent*dx[central])/np.sum(dx[central]))
    peak_c=float(np.max(vcent))
    mean_all=float(q_rack/np.sum(dx[g['gapmask']]))
    # rack pressure drop: weighted pressure averages below and above wafers
    pvals=[]; yrows=[]
    for yy in (0.10,0.28):
        jj=int(np.argmin(np.abs(g['yc']-yy)))
        m=g['rack'] & g['fluid'][jj,:]
        w=dx[m]
        pavg=float(np.sum(s.p[jj,m]*w)/np.sum(w))
        pvals.append(pavg); yrows.append(float(g['yc'][jj]))
    dp=float(pvals[0]-pvals[1])
    Rh=float(dp/q_rack)
    # mass flux at requested horizontal faces
    qs={}
    for yy in (0.05,0.10,0.20,0.30,0.40):
        jf=int(np.argmin(np.abs(g['yf']-yy)))
        q=float(np.sum(s.v[jf,:]*dx))
        qs[f'{yy:.2f}']={'y_face':float(g['yf'][jf]),'Q':q,'err_pct':float((q/q_in-1)*100 if q_in else np.nan)}
    return {
      'Q_rack':q_rack,'Q_total_ycut':q_total,'Q_in':q_in,
      'mean_gap_v':mean_all,'central_mean_v':mean_c,'central_peak_v':peak_c,
      'p_below':pvals[0],'p_above':pvals[1],'y_p_below':yrows[0],'y_p_above':yrows[1],
      'delta_p_rack':dp,'R_h':Rh,'mass_flux':qs,
    }

def rel_change(a,b):
    return abs(a-b)/(abs(b)+1e-14)

# Steady-state acceptance gate (tightened relative to the original bundle, which
# used RES_TOL=3e-5, DRIFT_TOL=1e-3 and returned silently at the step cap).
RES_TOL   = 1.0e-5    # per-step relative velocity change (field residual)
DRIFT_TOL = 2.0e-4    # chunk-to-chunk relative drift of observables (Q, peak v, dp)

def converge(s,U,dt,init_state=None,min_steps=2000,max_steps=12000,chunk=500,tag='',
             require_converged=True,res_tol=RES_TOL,drift_tol=DRIFT_TOL):
    """Advance to an approximate steady state.

    Acceptance requires BOTH a small field residual (res_tol) AND small
    chunk-to-chunk drift (drift_tol) of the acceptance observables Q_rack,
    central peak v and delta_p_rack.  If the step cap is reached without meeting
    the gate, a RuntimeError is raised (unless require_converged=False) rather
    than returning a silently under-converged state.
    """
    restore_scaled(s,init_state,U)
    hist=[]; total=0; converged=False; last_stable=np.inf
    # First call must not reset the state.
    while total < max_steps:
        n=min(chunk,max_steps-total)
        r=s.solve(U,dt=dt,max_steps=n,tol=1e-14,warm=True)
        total += n
        m=metrics(s)
        hist.append({'steps':total,'residual':float(r['res']),**{k:m[k] for k in ['Q_rack','central_peak_v','delta_p_rack']}})
        if total>=min_steps and len(hist)>=3:
            h0,h1,h2=hist[-3:]
            last_stable=max(
                rel_change(h2['Q_rack'],h1['Q_rack']),
                rel_change(h2['central_peak_v'],h1['central_peak_v']),
                rel_change(h2['delta_p_rack'],h1['delta_p_rack']),
                rel_change(h1['Q_rack'],h0['Q_rack']),
                rel_change(h1['delta_p_rack'],h0['delta_p_rack']),
            )
            if r['res'] < res_tol and last_stable < drift_tol:
                converged=True
                break
    m=metrics(s)
    if not converged and require_converged:
        raise RuntimeError(
            f'converge: step cap {max_steps} reached without meeting the gate '
            f'(res={float(r["res"]):.2e} > {res_tol:.1e} or drift={last_stable:.2e} '
            f'> {drift_tol:.1e}) at U={U:.5f}, pitch={s.g.get("n_gap","?")}. '
            f'Raise max_steps or relax the gate deliberately.')
    return m,snapshot(s,U),hist,float(r['res']),total

def run_case(item):
    label,cfg=item
    pitch=cfg['pitch']; lo,hi=cfg['bracket']
    s=Solver(pitch,N_GAP)
    dt=nominal_dt(s)
    root_hist=[]
    # small pitch is slowest to settle; allow generous steps for the enforced gate.
    # Exploratory evaluations run best-effort (require_converged=False); only the
    # final polish enforces convergence.
    max0=30000 if label=='Small' else (18000 if label=='Medium' else 14000)
    min0=4000 if label=='Small' else 3000
    mlo,slo,hlo,rlo,nlo=converge(s,lo,dt,None,min_steps=min0,max_steps=max0,tag='lo',require_converged=False)
    root_hist.append({'eval':'lo','U_in':lo,'Q_rack':mlo['Q_rack'],'err_pct':100*(mlo['Q_rack']/Q_TARGET-1),'steps':nlo,'residual':rlo})
    mhi,shi,hhi,rhi,nhi=converge(s,hi,dt,slo,min_steps=min0,max_steps=max0,tag='hi',require_converged=False)
    root_hist.append({'eval':'hi','U_in':hi,'Q_rack':mhi['Q_rack'],'err_pct':100*(mhi['Q_rack']/Q_TARGET-1),'steps':nhi,'residual':rhi})
    # expand bracket if needed
    expand=0
    while not (mlo['Q_rack'] <= Q_TARGET <= mhi['Q_rack']) and expand<3:
        expand+=1
        if mlo['Q_rack']>Q_TARGET:
            hi, mhi, shi = lo, mlo, slo
            lo*=0.75
            mlo,slo,hh,rr,nn=converge(s,lo,dt,slo,min_steps=min0,max_steps=max0,require_converged=False)
            root_hist.append({'eval':f'expand_lo{expand}','U_in':lo,'Q_rack':mlo['Q_rack'],'err_pct':100*(mlo['Q_rack']/Q_TARGET-1),'steps':nn,'residual':rr})
        elif mhi['Q_rack']<Q_TARGET:
            lo,mlo,slo=hi,mhi,shi
            hi*=1.30
            mhi,shi,hh,rr,nn=converge(s,hi,dt,shi,min_steps=min0,max_steps=max0,require_converged=False)
            root_hist.append({'eval':f'expand_hi{expand}','U_in':hi,'Q_rack':mhi['Q_rack'],'err_pct':100*(mhi['Q_rack']/Q_TARGET-1),'steps':nn,'residual':rr})
    if not (mlo['Q_rack'] <= Q_TARGET <= mhi['Q_rack']):
        raise RuntimeError(f'{label}: failed to bracket Q target: {mlo["Q_rack"]}, {mhi["Q_rack"]}')
    final=None
    for it in range(7):
        # bracketed secant, clipped away from endpoints; falls back toward bisection
        ql,qh=mlo['Q_rack'],mhi['Q_rack']
        usec=lo+(Q_TARGET-ql)*(hi-lo)/(qh-ql)
        span=hi-lo
        umid=min(max(usec,lo+0.20*span),hi-0.20*span)
        # warm start from closer endpoint
        state=slo if abs(umid-lo)<=abs(hi-umid) else shi
        mm,sm,hh,rr,nn=converge(s,umid,dt,state,min_steps=min0,max_steps=max0,require_converged=False)
        err=(mm['Q_rack']/Q_TARGET-1)
        root_hist.append({'eval':f'root{it+1}','U_in':umid,'Q_rack':mm['Q_rack'],'err_pct':100*err,'steps':nn,'residual':rr})
        if abs(err) <= ROOT_TOL:
            final=(umid,mm,sm,rr,nn,hh); break
        if mm['Q_rack']<Q_TARGET:
            lo,mlo,slo=umid,mm,sm
        else:
            hi,mhi,shi=umid,mm,sm
        final=(umid,mm,sm,rr,nn,hh)
    U,mm,sm,rr,nn,hh=final
    # one final polishing convergence from final state
    mf,sf,hf,rf,nf=converge(s,U,dt,sm,min_steps=min0,max_steps=max0,require_converged=True)
    errf=mf['Q_rack']/Q_TARGET-1
    # if polishing drifted beyond tolerance, do one bracket correction using current bracket
    if abs(errf)>ROOT_TOL:
        if mf['Q_rack']<Q_TARGET:
            lo,mlo,slo=U,mf,sf
        else:
            hi,mhi,shi=U,mf,sf
        U2=0.5*(lo+hi)
        state=slo if abs(U2-lo)<=abs(hi-U2) else shi
        mf,sf,hf,rf,nf=converge(s,U2,dt,state,min_steps=min0,max_steps=max0,require_converged=True)
        U=U2; errf=mf['Q_rack']/Q_TARGET-1
        root_hist.append({'eval':'final_bisect','U_in':U,'Q_rack':mf['Q_rack'],'err_pct':100*errf,'steps':nf,'residual':rf})
    # save fields
    g=s.g
    np.savez_compressed(os.path.join(OUT,f'{label.lower()}_production.npz'),
        u=s.u,v=s.v,p=s.p,xf=g['xf'],yf=g['yf'],xc=g['xc'],yc=g['yc'],dx=g['dx'],dy=g['dy'],
        solid=g['solid'],left=g['left'],right=g['right'],centers=g['centers'],gapmask=g['gapmask'],central=g['central'],
        U_in=U,dt=dt,pitch=pitch,Q_target=Q_TARGET)
    pd.DataFrame(root_hist).to_csv(os.path.join(OUT,f'{label.lower()}_root_history.csv'),index=False)
    # mass table
    massrows=[]
    for key,val in mf['mass_flux'].items(): massrows.append({'case':label,'y_requested':float(key),**val})
    pd.DataFrame(massrows).to_csv(os.path.join(OUT,f'{label.lower()}_mass_flux.csv'),index=False)
    return {
      'case':label,'pitch':pitch,'n_gap_cells':N_GAP,'Nx':g['Nx'],'Ny':g['Ny'],
      'dx_min':float(g['dx'].min()),'dy_min':float(g['dy'].min()),'dt':dt,'U_in':U,
      'Q_target':Q_TARGET,'Q_rack':mf['Q_rack'],'Q_error_pct':100*errf,
      'Q_in':mf['Q_in'],'Q_total_ycut':mf['Q_total_ycut'],
      'mean_gap_v':mf['mean_gap_v'],'central_mean_v':mf['central_mean_v'],'central_peak_v':mf['central_peak_v'],
      'delta_p_rack':mf['delta_p_rack'],'R_h':mf['R_h'],
      'p_below':mf['p_below'],'p_above':mf['p_above'],'y_p_below':mf['y_p_below'],'y_p_above':mf['y_p_above'],
      'final_residual':rf,'final_polish_steps':nf,
      'max_mass_error_pct':max(abs(x['err_pct']) for x in mf['mass_flux'].values()),
    }

if __name__=='__main__':
    t=time.time()
    os.makedirs(OUT,exist_ok=True)
    ctx=mp.get_context('fork')
    with ctx.Pool(3) as pool:
        results=pool.map(run_case,list(CASES.items()))
    df=pd.DataFrame(results).sort_values('pitch')
    df.to_csv(os.path.join(OUT,'production_summary.csv'),index=False)
    meta={
      'method':'2-D incompressible Navier-Stokes, MAC staggered-grid finite-volume/finite-difference projection',
      'advection':'first-order donor-cell upwind on staggered faces',
      'diffusion':'second-order nonuniform finite differences, explicit',
      'pressure_projection':'compatible discrete D/G operators; sparse direct pressure-Poisson solve',
      'pressure_outlet':'p=0 at open top boundary',
      'walls':'no-slip on tank side/bottom walls and wafer surfaces; prescribed bottom vertical inlet slots',
      'equal_Q_method':'bracketed secant with bisection safeguard; each function evaluation is a converged CFD solve',
      'no_fitting':True,
      'nu':NU,'Q_target':Q_TARGET,'n_gap_cells_production':N_GAP,'root_tolerance_fraction':ROOT_TOL,
      'elapsed_seconds':time.time()-t,
    }
    with open(os.path.join(OUT,'run_metadata.json'),'w') as f: json.dump(meta,f,indent=2)
    print(df.to_string(index=False))
    print('elapsed',time.time()-t)
