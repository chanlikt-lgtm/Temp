import sys,os,time,multiprocessing as mp
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
from mac_cfd_adaptive import Solver,NU
Q=0.0055463
OUT='/mnt/data/wafer_mac_validation';os.makedirs(OUT,exist_ok=True)

def metrics(s):
 g=s.g;j=g['jcut'];dx=g['dx'];qr=float(np.sum(s.v[j,g['gapmask']]*dx[g['gapmask']]))
 cm=g['central'];pk=float(np.max(s.v[j,cm]));mean=float(qr/np.sum(dx[g['gapmask']]))
 ps=[]
 for yy in (.10,.28):
  jj=int(np.argmin(abs(g['yc']-yy)));m=g['rack']&g['fluid'][jj];w=dx[m];ps.append(float(np.sum(s.p[jj,m]*w)/np.sum(w)))
 dp=ps[0]-ps[1]
 return qr,mean,pk,dp,dp/qr

def root_case(n_gap,dt_factor=1.0,U0=.389):
 s=Solver(.06,n_gap);g=s.g;h=min(g['dx'].min(),g['dy'].min());dt=dt_factor*0.90*h*h/(4*NU)
 hist=[];U=U0
 # independent, well-settled evaluations; 3 secant/proportional updates max
 for it in range(4):
  r=s.solve(U,dt=dt,max_steps=6500 if it==0 else 5000,tol=4e-6,warm=False)
  qr,mean,pk,dp,R=metrics(s);err=qr/Q-1
  hist.append((it,U,qr,100*err,r['res'],r['steps'],mean,pk,dp,R))
  if abs(err)<0.0015: break
  U*=Q/qr
 # final polish at U from zero to reduce warm-history dependence
 r=s.solve(U,dt=dt,max_steps=8000,tol=2e-6,warm=False)
 qr,mean,pk,dp,R=metrics(s);err=qr/Q-1
 hist.append((99,U,qr,100*err,r['res'],r['steps'],mean,pk,dp,R))
 pd.DataFrame(hist,columns=['iter','U_in','Q_rack','Q_err_pct','residual','steps','mean_gap_v','central_peak_v','delta_p','R_h']).to_csv(os.path.join(OUT,f'medium_ngap{n_gap}_dtf{dt_factor:.2f}_history.csv'),index=False)
 return {'n_gap':n_gap,'Nx':g['Nx'],'Ny':g['Ny'],'dx_min':g['dx'].min(),'dy_min':g['dy'].min(),'dt_factor':dt_factor,'dt':dt,'U_in':U,'Q_rack':qr,'Q_err_pct':100*err,'mean_gap_v':mean,'central_peak_v':pk,'delta_p':dp,'R_h':R,'residual':r['res'],'steps':r['steps']}

if __name__=='__main__':
 tasks=[(12,1.0,.389),(16,1.0,.389),(20,0.5,.389)]
 with mp.get_context('fork').Pool(3) as pool:
  rows=pool.starmap(root_case,tasks)
 # append production 20x nominal from existing summary
 prod=pd.read_csv('/mnt/data/wafer_mac_equalQ_v2/medium_summary.csv').iloc[0]
 rows.append({'n_gap':20,'Nx':int(prod.Nx),'Ny':int(prod.Ny),'dx_min':prod.dx_min,'dy_min':prod.dy_min,'dt_factor':1.0,'dt':prod.dt,'U_in':prod.U_in,'Q_rack':prod.Q_rack,'Q_err_pct':prod.Q_error_pct,'mean_gap_v':prod.mean_gap_v,'central_peak_v':prod.central_peak_v,'delta_p':prod.delta_p_rack,'R_h':prod.R_h,'residual':prod.final_residual,'steps':np.nan})
 df=pd.DataFrame(rows).sort_values(['dt_factor','n_gap'])
 df.to_csv(os.path.join(OUT,'validation_summary.csv'),index=False)
 print(df.to_string(index=False))
