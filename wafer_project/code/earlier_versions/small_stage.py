import os,sys,json
import numpy as np
sys.path.insert(0,'/mnt/data')
from mac_cfd_adaptive import Solver
from run_one_case_equalq import metrics
OUT='/mnt/data/wafer_mac_equalQ_v2'; os.makedirs(OUT,exist_ok=True)
stage=sys.argv[1]
Q=0.0055463;dt=.00047
if stage=='1':
 s=Solver(.04,20);U=.70
 r=s.solve(U,dt=dt,max_steps=9000,tol=1e-12,warm=False);m=metrics(s)
 np.savez_compressed(os.path.join(OUT,'small_stage1.npz'),u=s.u,v=s.v,p=s.p,U=U)
 json.dump({'U':U,'Q':m['Q_rack'],'res':float(r['res']),'metrics':m},open(os.path.join(OUT,'small_stage1.json'),'w'),indent=2)
 print(U,m['Q_rack'],r['res'])
elif stage=='2':
 z=np.load(os.path.join(OUT,'small_stage1.npz')); base=json.load(open(os.path.join(OUT,'small_stage1.json')))
 s=Solver(.04,20); U0=float(z['U']); q0=base['Q']; U=U0*np.clip(Q/q0,.95,1.05)
 f=U/U0;s.u[:]=z['u']*f;s.v[:]=z['v']*f;s.p[:]=z['p']*f
 r=s.solve(U,dt=dt,max_steps=3500,tol=1e-12,warm=True);m=metrics(s)
 np.savez_compressed(os.path.join(OUT,'small_stage2.npz'),u=s.u,v=s.v,p=s.p,U=U)
 json.dump({'U':U,'Q':m['Q_rack'],'res':float(r['res']),'metrics':m},open(os.path.join(OUT,'small_stage2.json'),'w'),indent=2)
 print(U,m['Q_rack'],r['res'])
elif stage=='3':
 a=json.load(open(os.path.join(OUT,'small_stage1.json')));b=json.load(open(os.path.join(OUT,'small_stage2.json'))); z=np.load(os.path.join(OUT,'small_stage2.npz'))
 U0,q0=a['U'],a['Q'];U1,q1=b['U'],b['Q']
 if (q0-Q)*(q1-Q)<=0 and abs(q1-q0)>1e-12:
  U=U0+(Q-q0)*(U1-U0)/(q1-q0);lo=min(U0,U1);hi=max(U0,U1);U=min(max(U,lo+.1*(hi-lo)),hi-.1*(hi-lo))
 else: U=U1*np.clip(Q/q1,.97,1.03)
 s=Solver(.04,20); f=U/U1;s.u[:]=z['u']*f;s.v[:]=z['v']*f;s.p[:]=z['p']*f
 r=s.solve(U,dt=dt,max_steps=3000,tol=1e-12,warm=True);m=metrics(s)
 np.savez_compressed(os.path.join(OUT,'small_stage3.npz'),u=s.u,v=s.v,p=s.p,U=U)
 json.dump({'U':U,'Q':m['Q_rack'],'res':float(r['res']),'metrics':m},open(os.path.join(OUT,'small_stage3.json'),'w'),indent=2)
 print(U,m['Q_rack'],r['res'])
elif stage=='4':
 a=json.load(open(os.path.join(OUT,'small_stage1.json')));b=json.load(open(os.path.join(OUT,'small_stage2.json')));c=json.load(open(os.path.join(OUT,'small_stage3.json'))); z=np.load(os.path.join(OUT,'small_stage3.npz'))
 pts=[a,b,c]; br=None
 for i in range(3):
  for j in range(i+1,3):
   if (pts[i]['Q']-Q)*(pts[j]['Q']-Q)<=0: br=(pts[i],pts[j]);break
  if br: break
 if abs(c['Q']/Q-1)<=.002: U=c['U']
 elif br: U=.5*(br[0]['U']+br[1]['U'])
 else: U=c['U']*np.clip(Q/c['Q'],.985,1.015)
 s=Solver(.04,20); f=U/c['U'];s.u[:]=z['u']*f;s.v[:]=z['v']*f;s.p[:]=z['p']*f
 r=s.solve(U,dt=dt,max_steps=2800,tol=1e-12,warm=True);m=metrics(s)
 # final save
 g=s.g
 np.savez_compressed(os.path.join(OUT,'small_production.npz'),u=s.u,v=s.v,p=s.p,xf=g['xf'],yf=g['yf'],xc=g['xc'],yc=g['yc'],dx=g['dx'],dy=g['dy'],solid=g['solid'],left=g['left'],right=g['right'],centers=g['centers'],gapmask=g['gapmask'],central=g['central'],U_in=U,dt=dt,pitch=.04,Q_target=Q)
 json.dump({'U':U,'Q':m['Q_rack'],'res':float(r['res']),'metrics':m},open(os.path.join(OUT,'small_final.json'),'w'),indent=2)
 print(U,m['Q_rack'],r['res'])
