import sys,os,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
from mac_cfd_adaptive import Solver,NU
Q=.0055463; OUT='/mnt/data/wafer_mac_equalQ_v2'
prod=np.load(os.path.join(OUT,'small_production.npz'))
U=float(prod['U_in']);dt0=float(prod['dt'])
rows=[]
for factor in [1.0,0.5]:
 s=Solver(.04,20);g=s.g;dt=dt0*factor
 # initialize from production when shapes match
 s.u[:]=prod['u'];s.v[:]=prod['v'];s.p[:]=prod['p']
 # run to steady at same U
 r=s.solve(U,dt=dt,max_steps=4000 if factor<1 else 1500,tol=7e-7,warm=True)
 j=g['jcut'];dx=g['dx'];qr=float(np.sum(s.v[j,g['gapmask']]*dx[g['gapmask']]))
 # one equal-Q correction if needed
 err=qr/Q-1
 Uuse=U
 if abs(err)>.001:
  Uuse=U*(Q/qr)
  f=Uuse/U;s.u*=f;s.v*=f;s.p*=f
  r=s.solve(Uuse,dt=dt,max_steps=3500,tol=7e-7,warm=True)
  qr=float(np.sum(s.v[j,g['gapmask']]*dx[g['gapmask']]))
 cm=g['central'];pk=float(np.max(s.v[j,cm]));cmean=float(np.sum(s.v[j,cm]*dx[cm])/np.sum(dx[cm]))
 ps=[]
 for yy in (.10,.28):
  jj=int(np.argmin(abs(g['yc']-yy)));m=g['rack']&g['fluid'][jj];w=dx[m];ps.append(float(np.sum(s.p[jj,m]*w)/np.sum(w)))
 dp=ps[0]-ps[1]
 rows.append({'dt_factor':factor,'dt':dt,'U_in':Uuse,'Q_rack':qr,'Q_err_pct':100*(qr/Q-1),'central_mean_v':cmean,'central_peak_v':pk,'delta_p_rack':dp,'R_h':dp/qr,'residual':r['res']})
pd.DataFrame(rows).to_csv(os.path.join(OUT,'small_timestep_study.csv'),index=False)
print(pd.DataFrame(rows).to_string(index=False))
