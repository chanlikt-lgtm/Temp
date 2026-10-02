import os,sys,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
from mac_cfd_adaptive import Solver
from run_one_case_equalq import metrics
OUT='/mnt/data/wafer_mac_equalQ_v2'; z=np.load(os.path.join(OUT,'small_production.npz'))
U=float(z['U_in']); dt0=float(z['dt']); Q=float(z['Q_target'])
rows=[]
for factor,steps in [(1.0,1800),(0.5,3600)]:
 s=Solver(.04,20); s.u[:]=z['u']; s.v[:]=z['v']; s.p[:]=z['p']
 dt=dt0*factor
 r=s.solve(U,dt=dt,max_steps=steps,tol=1e-12,warm=True);m=metrics(s)
 rows.append({'dt_factor':factor,'dt':dt,'steps':steps,'U_in':U,'Q_rack':m['Q_rack'],'Q_error_pct':100*(m['Q_rack']/Q-1),'central_peak_v':m['central_peak_v'],'central_mean_v':m['central_mean_v'],'delta_p_rack':m['delta_p_rack'],'R_h':m['R_h'],'residual':r['res'],'max_mass_error_pct':max(abs(x[3]) for x in m['mass_flux'])})
df=pd.DataFrame(rows)
base=df.iloc[0]
for col in ['Q_rack','central_peak_v','delta_p_rack']:
 df[col+'_change_vs_dt_pct']=100*(df[col]/base[col]-1)
df.to_csv(os.path.join(OUT,'timestep_study_small.csv'),index=False)
print(df.to_string(index=False))
