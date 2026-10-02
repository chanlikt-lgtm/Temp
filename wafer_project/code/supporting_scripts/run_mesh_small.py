import os,sys,json
import numpy as np,pandas as pd
_HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE,'..','audited_final'))
sys.path.insert(0, _HERE)
from mac_cfd_adaptive import Solver,NU
from run_one_case_equalq import metrics,snapshot,restore
OUT=os.environ.get('WAFER_OUT', os.path.join(os.getcwd(),'wafer_mac_equalQ_v2'));os.makedirs(OUT,exist_ok=True);Q=.0055463
n=int(sys.argv[1]);U0=float(sys.argv[2]) if len(sys.argv)>2 else .69
s=Solver(.04,n);h=min(s.g['dx'].min(),s.g['dy'].min());dt=.9*h*h/(4*NU)
# steps scale roughly inverse dt but cap
if n<=10: steps=(4500,1800,1400)
elif n<=15: steps=(6500,2400,1800)
else: steps=(9000,3200,2200)
rows=[]
def seg(U,ns,st=None,stage=''):
 restore(s,st,U);r=s.solve(U,dt=dt,max_steps=ns,tol=1e-12,warm=True);m=metrics(s);ss=snapshot(s,U);rows.append({'stage':stage,'U_in':U,'Q_rack':m['Q_rack'],'residual':r['res'],'steps':ns,'err_pct':100*(m['Q_rack']/Q-1)});return m,ss,r
m0,st0,r0=seg(U0,steps[0],None,'initial')
U1=U0*np.clip(Q/m0['Q_rack'],.88,1.12);m1,st1,r1=seg(U1,steps[1],st0,'controller')
# bracketed secant if bracket, else second controller
if (m0['Q_rack']-Q)*(m1['Q_rack']-Q)<=0:
 U2=U0+(Q-m0['Q_rack'])*(U1-U0)/(m1['Q_rack']-m0['Q_rack']);lo=min(U0,U1);hi=max(U0,U1);U2=min(max(U2,lo+.1*(hi-lo)),hi-.1*(hi-lo));stage='bracketed_secant'
else:
 U2=U1*np.clip(Q/m1['Q_rack'],.94,1.06);stage='controller2'
m2,st2,r2=seg(U2,steps[2],st1,stage)
# one final correction if >0.2%
if abs(m2['Q_rack']/Q-1)>.002:
 pts=[(U0,m0,st0),(U1,m1,st1),(U2,m2,st2)];br=None
 for i in range(3):
  for j in range(i+1,3):
   if (pts[i][1]['Q_rack']-Q)*(pts[j][1]['Q_rack']-Q)<=0:br=(pts[i],pts[j]);break
  if br:break
 if br: U3=.5*(br[0][0]+br[1][0]);base=br[0][2]
 else: U3=U2*np.clip(Q/m2['Q_rack'],.97,1.03);base=st2
 m3,st3,r3=seg(U3,steps[2],base,'polish');U=U3;m=m3;r=r3
else: U=U2;m=m2;r=r2
# save compact mesh result
row={'n_gap_cells':n,'Nx':s.g['Nx'],'Ny':s.g['Ny'],'dx_min':s.g['dx'].min(),'dy_min':s.g['dy'].min(),'dt':dt,'U_in':U,'Q_target':Q,'Q_rack':m['Q_rack'],'Q_error_pct':100*(m['Q_rack']/Q-1),'central_peak_v':m['central_peak_v'],'central_mean_v':m['central_mean_v'],'delta_p_rack':m['delta_p_rack'],'R_h':m['R_h'],'residual':r['res'],'max_mass_error_pct':max(abs(v['err_pct']) for v in m['mass_flux'].values())}
pd.DataFrame(rows).to_csv(os.path.join(OUT,f'mesh_small_n{n}_root_history.csv'),index=False)
pd.DataFrame([row]).to_csv(os.path.join(OUT,f'mesh_small_n{n}_summary.csv'),index=False)
print(pd.DataFrame(rows).to_string(index=False));print('RESULT',row)
