import os,sys,json,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
from mac_cfd_adaptive import Solver
from run_one_case_equalq import metrics
OUT='/mnt/data/wafer_mac_equalQ_v2';Q=.0055463;dt=.00047
a=json.load(open(os.path.join(OUT,'small_stage3.json')));b=json.load(open(os.path.join(OUT,'small_final.json')));z=np.load(os.path.join(OUT,'small_production.npz'))
Ua,qa=a['U'],a['Q'];Ub,qb=b['U'],b['Q']
U=Ua+(Q-qa)*(Ub-Ua)/(qb-qa)
lo=min(Ua,Ub);hi=max(Ua,Ub);U=min(max(U,lo+.1*(hi-lo)),hi-.1*(hi-lo))
s=Solver(.04,20);f=U/Ub;s.u[:]=z['u']*f;s.v[:]=z['v']*f;s.p[:]=z['p']*f
r=s.solve(U,dt=dt,max_steps=2600,tol=1e-12,warm=True);m=metrics(s);g=s.g
np.savez_compressed(os.path.join(OUT,'small_production.npz'),u=s.u,v=s.v,p=s.p,xf=g['xf'],yf=g['yf'],xc=g['xc'],yc=g['yc'],dx=g['dx'],dy=g['dy'],solid=g['solid'],left=g['left'],right=g['right'],centers=g['centers'],gapmask=g['gapmask'],central=g['central'],U_in=U,dt=dt,pitch=.04,Q_target=Q)
# histories combined from json stages + stage5
rows=[]
for name in ['small_stage1.json','small_stage2.json','small_stage3.json','small_final.json']:
 d=json.load(open(os.path.join(OUT,name)));rows.append({'stage':name.replace('.json',''),'U_in':d['U'],'Q_rack':d['Q'],'residual':d['res']})
rows.append({'stage':'secant_final','U_in':U,'Q_rack':m['Q_rack'],'residual':float(r['res'])})
pd.DataFrame(rows).assign(err_pct=lambda d:100*(d.Q_rack/Q-1)).to_csv(os.path.join(OUT,'small_root_history.csv'),index=False)
pd.DataFrame([{'case':'Small','y_requested':x,'y_face':yf,'Q':q,'err_pct':e} for x,yf,q,e in m['mass_flux']]).to_csv(os.path.join(OUT,'small_mass_flux.csv'),index=False)
row={'case':'Small','pitch':.04,'n_gap_cells':20,'Nx':g['Nx'],'Ny':g['Ny'],'dx_min':g['dx'].min(),'dy_min':g['dy'].min(),'dt':dt,'U_in':U,'Q_target':Q,'Q_rack':m['Q_rack'],'Q_error_pct':100*(m['Q_rack']/Q-1),'Q_in':m['Q_in'],'Q_total_ycut':m['Q_total_ycut'],'mean_gap_v':m['mean_gap_v'],'central_mean_v':m['central_mean_v'],'central_peak_v':m['central_peak_v'],'delta_p_rack':m['delta_p_rack'],'R_h':m['R_h'],'p_below':m['p_below'],'p_above':m['p_above'],'y_p_below':m['y_p_below'],'y_p_above':m['y_p_above'],'final_residual':r['res'],'max_mass_error_pct':max(abs(z[3]) for z in m['mass_flux'])}
pd.DataFrame([row]).to_csv(os.path.join(OUT,'small_summary.csv'),index=False)
print('U',U,'Q',m['Q_rack'],'err%',100*(m['Q_rack']/Q-1),'res',r['res']);print(row)
