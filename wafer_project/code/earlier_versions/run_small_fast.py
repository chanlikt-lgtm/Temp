import sys,os,time,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
from mac_cfd_adaptive import Solver,NU
OUT='/mnt/data/wafer_mac_equalQ_v2';Q=0.0055463
s=Solver(.04,20);g=s.g;h=min(g['dx'].min(),g['dy'].min());dt=.9*h*h/(4*NU)

def qrack():
 return float(np.sum(s.v[g['jcut'],g['gapmask']]*g['dx'][g['gapmask']]))
def metrics():
 j=g['jcut'];dx=g['dx']; qr=qrack();qin=float(np.sum(s.v[0]*dx));qt=float(np.sum(s.v[j]*dx)); cm=g['central'];
 cmean=float(np.sum(s.v[j,cm]*dx[cm])/np.sum(dx[cm])); cpeak=float(np.max(s.v[j,cm])); mean=float(qr/np.sum(dx[g['gapmask']]))
 ps=[];yrs=[]
 for yy in (.10,.28):
  jj=int(np.argmin(abs(g['yc']-yy)));m=g['rack']&g['fluid'][jj];w=dx[m];ps.append(float(np.sum(s.p[jj,m]*w)/np.sum(w)));yrs.append(float(g['yc'][jj]))
 dp=ps[0]-ps[1]; mf=[]
 for yy in (.05,.10,.20,.30,.40):
  jf=int(np.argmin(abs(g['yf']-yy)));q=float(np.sum(s.v[jf]*dx));mf.append((yy,float(g['yf'][jf]),q,100*(q/qin-1)))
 return dict(Q_rack=qr,Q_in=qin,Q_total_ycut=qt,mean_gap_v=mean,central_mean_v=cmean,central_peak_v=cpeak,p_below=ps[0],p_above=ps[1],y_p_below=yrs[0],y_p_above=yrs[1],delta_p_rack=dp,R_h=dp/qr,mass_flux=mf)

def scale_state(Uold,Unew):
 f=Unew/Uold;s.u*=f;s.v*=f;s.p*=f

def segment(U,steps,stage,hist):
 r=s.solve(U,dt=dt,max_steps=steps,tol=1e-12,warm=True); q=qrack(); hist.append((U,q,r['res'],steps,stage,100*(q/Q-1)));print(stage,U,q,100*(q/Q-1),r['res'],flush=True);return r,q
hist=[]
U=.69
r,q=segment(U,6000,'initial',hist)
for k in range(3):
 Unew=U*(Q/q)
 # damp to 12% max
 Unew=U*np.clip(Unew/U,0.88,1.12)
 scale_state(U,Unew);U=Unew
 r,q=segment(U,2800,f'controller{k+1}',hist)
 if abs(q/Q-1)<0.001: break
# final polish, then one correction if drift
r,q=segment(U,3000,'steady_polish',hist)
if abs(q/Q-1)>0.001:
 Unew=U*(Q/q);scale_state(U,Unew);U=Unew
 r,q=segment(U,2500,'final_correction',hist)
m=metrics()
np.savez_compressed(os.path.join(OUT,'small_production.npz'),u=s.u,v=s.v,p=s.p,xf=g['xf'],yf=g['yf'],xc=g['xc'],yc=g['yc'],dx=g['dx'],dy=g['dy'],solid=g['solid'],left=g['left'],right=g['right'],centers=g['centers'],gapmask=g['gapmask'],central=g['central'],U_in=U,dt=dt,pitch=.04,Q_target=Q)
pd.DataFrame(hist,columns=['U_in','Q_rack','residual','steps','stage','err_pct']).to_csv(os.path.join(OUT,'small_root_history.csv'),index=False)
pd.DataFrame([{'case':'Small','y_requested':a,'y_face':b,'Q':c,'err_pct':d} for a,b,c,d in m['mass_flux']]).to_csv(os.path.join(OUT,'small_mass_flux.csv'),index=False)
row={'case':'Small','pitch':.04,'n_gap_cells':20,'Nx':g['Nx'],'Ny':g['Ny'],'dx_min':g['dx'].min(),'dy_min':g['dy'].min(),'dt':dt,'U_in':U,'Q_target':Q,'Q_rack':m['Q_rack'],'Q_error_pct':100*(m['Q_rack']/Q-1),'Q_in':m['Q_in'],'Q_total_ycut':m['Q_total_ycut'],'mean_gap_v':m['mean_gap_v'],'central_mean_v':m['central_mean_v'],'central_peak_v':m['central_peak_v'],'delta_p_rack':m['delta_p_rack'],'R_h':m['R_h'],'p_below':m['p_below'],'p_above':m['p_above'],'y_p_below':m['y_p_below'],'y_p_above':m['y_p_above'],'final_residual':r['res'],'max_mass_error_pct':max(abs(z[3]) for z in m['mass_flux'])}
pd.DataFrame([row]).to_csv(os.path.join(OUT,'small_summary.csv'),index=False)
print('FINAL',row)
