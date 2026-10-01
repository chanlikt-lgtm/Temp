import os,sys,json,time
import numpy as np,pandas as pd
# Import solver + the single shared metrics/snapshot/restore implementation from
# the audited driver, so this lighter single-case runner cannot drift from it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'audited_final'))
from mac_cfd_adaptive import Solver, NU
from equalQ_root_driver import metrics, snapshot, restore_scaled as restore

OUT=os.environ.get('WAFER_OUT', os.path.join(os.getcwd(), 'wafer_mac_equalQ_v2'))
os.makedirs(OUT,exist_ok=True)
Q=0.0055463
CFG={
 'Small':(.04,.69,0.00047,9000,3500,2500),
 'Medium':(.06,.41,None,5500,2500,1800),
 'Large':(.08,.245,None,4500,2200,1500),
}

def run_segment(s,U,dt,steps,state=None):
 restore(s,state,U)
 r=s.solve(U,dt=dt,max_steps=steps,tol=1e-12,warm=True)
 return metrics(s),snapshot(s,U),r

def main(label):
 pitch,U0,dt0,n0,n1,n2=CFG[label]
 s=Solver(pitch,20)
 if dt0 is None:
  h=min(s.g['dx'].min(),s.g['dy'].min()); dt0=0.90*h*h/(4*NU)
 hist=[]
 m0,st0,r0=run_segment(s,U0,dt0,n0,None)
 hist.append((U0,m0['Q_rack'],r0['res'],n0,'initial'))
 # first root correction; limited to 15% so warm-start remains close
 ratio=Q/m0['Q_rack']; U1=U0*np.clip(ratio,0.85,1.15)
 m1,st1,r1=run_segment(s,U1,dt0,n1,st0)
 hist.append((U1,m1['Q_rack'],r1['res'],n1,'controller1'))
 # if not bracketed, move once in the required direction by 8%
 def sign(m): return np.sign(m['Q_rack']-Q)
 if sign(m0)==sign(m1):
  if m1['Q_rack']<Q: Ux=U1*1.08
  else: Ux=U1*0.92
  mx,stx,rx=run_segment(s,Ux,dt0,n1,st1)
  hist.append((Ux,mx['Q_rack'],rx['res'],n1,'bracket_expand'))
  # retain two latest different-sign points if possible
  pts=[(U0,m0,st0),(U1,m1,st1),(Ux,mx,stx)]
 else:
  pts=[(U0,m0,st0),(U1,m1,st1)]
 # find bracket pair
 br=None
 for a in range(len(pts)):
  for b in range(a+1,len(pts)):
   if (pts[a][1]['Q_rack']-Q)*(pts[b][1]['Q_rack']-Q)<=0:
    br=(pts[a],pts[b]);break
  if br:break
 if br:
  (Ua,ma,sta),(Ub,mb,stb)=br
  if Ua>Ub: Ua,Ub,ma,mb,sta,stb=Ub,Ua,mb,ma,stb,sta
  # safeguarded secant inside bracket
  Usec=Ua+(Q-ma['Q_rack'])*(Ub-Ua)/(mb['Q_rack']-ma['Q_rack'])
  Usec=min(max(Usec,Ua+0.15*(Ub-Ua)),Ub-0.15*(Ub-Ua))
  base=sta if abs(Usec-Ua)<abs(Ub-Usec) else stb
  m2,st2,r2=run_segment(s,Usec,dt0,n2,base)
  hist.append((Usec,m2['Q_rack'],r2['res'],n2,'bracketed_secant'))
  Ufinal,mfinal,stfinal,rfinal=Usec,m2,st2,r2
  # one bisection polish if needed
  if abs(m2['Q_rack']/Q-1)>0.002:
   if (ma['Q_rack']-Q)*(m2['Q_rack']-Q)<=0: Ub,mb,stb=Usec,m2,st2
   else: Ua,ma,sta=Usec,m2,st2
   Umid=.5*(Ua+Ub);base=sta if abs(Umid-Ua)<abs(Ub-Umid) else stb
   m3,st3,r3=run_segment(s,Umid,dt0,n2,base)
   hist.append((Umid,m3['Q_rack'],r3['res'],n2,'bisection_polish'))
   Ufinal,mfinal,stfinal,rfinal=Umid,m3,st3,r3
 else:
  # fallback safeguarded proportional root iteration
  U2=U1*np.clip(Q/m1['Q_rack'],0.90,1.10)
  m2,st2,r2=run_segment(s,U2,dt0,n2,st1)
  hist.append((U2,m2['Q_rack'],r2['res'],n2,'fallback_controller'))
  Ufinal,mfinal,stfinal,rfinal=U2,m2,st2,r2
 # final steady polish at the selected U
 mF,stF,rF=run_segment(s,Ufinal,dt0,1800,stfinal)
 hist.append((Ufinal,mF['Q_rack'],rF['res'],1800,'steady_polish'))
 g=s.g
 np.savez_compressed(os.path.join(OUT,f'{label.lower()}_production.npz'),u=s.u,v=s.v,p=s.p,xf=g['xf'],yf=g['yf'],xc=g['xc'],yc=g['yc'],dx=g['dx'],dy=g['dy'],solid=g['solid'],left=g['left'],right=g['right'],centers=g['centers'],gapmask=g['gapmask'],central=g['central'],U_in=Ufinal,dt=dt0,pitch=pitch,Q_target=Q)
 pd.DataFrame(hist,columns=['U_in','Q_rack','residual','steps','stage']).assign(err_pct=lambda d:100*(d.Q_rack/Q-1)).to_csv(os.path.join(OUT,f'{label.lower()}_root_history.csv'),index=False)
 pd.DataFrame([{'case':label,'y_requested':float(k),'y_face':v['y_face'],'Q':v['Q'],'err_pct':v['err_pct']} for k,v in mF['mass_flux'].items()]).to_csv(os.path.join(OUT,f'{label.lower()}_mass_flux.csv'),index=False)
 row={'case':label,'pitch':pitch,'n_gap_cells':20,'Nx':g['Nx'],'Ny':g['Ny'],'dx_min':g['dx'].min(),'dy_min':g['dy'].min(),'dt':dt0,'U_in':Ufinal,'Q_target':Q,'Q_rack':mF['Q_rack'],'Q_error_pct':100*(mF['Q_rack']/Q-1),'Q_in':mF['Q_in'],'Q_total_ycut':mF['Q_total_ycut'],'mean_gap_v':mF['mean_gap_v'],'central_mean_v':mF['central_mean_v'],'central_peak_v':mF['central_peak_v'],'delta_p_rack':mF['delta_p_rack'],'R_h':mF['R_h'],'p_below':mF['p_below'],'p_above':mF['p_above'],'y_p_below':mF['y_p_below'],'y_p_above':mF['y_p_above'],'final_residual':rF['res'],'max_mass_error_pct':max(abs(v['err_pct']) for v in mF['mass_flux'].values())}
 pd.DataFrame([row]).to_csv(os.path.join(OUT,f'{label.lower()}_summary.csv'),index=False)
 print(pd.DataFrame(hist,columns=['U','Q','res','steps','stage']).to_string(index=False))
 print('FINAL',row)

if __name__=='__main__':
 main(sys.argv[1])
