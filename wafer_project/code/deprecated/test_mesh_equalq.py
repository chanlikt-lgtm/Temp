import sys, numpy as np, time
sys.path.insert(0,'/mnt/data')
from mac_cfd_prototype import Solver, NU
Q=0.0055463

def metrics(s):
 g=s.g; j=g['jcut']; mask=np.zeros(g['Nx'],bool)
 for k in range(5): mask |= (g['xc']>g['right'][k])&(g['xc']<g['left'][k+1])
 qr=np.sum(s.v[j,mask]*g['dx'][mask])
 # central gap
 cm=(g['xc']>g['right'][2])&(g['xc']<g['left'][3])
 pk=np.max(s.v[j,cm])
 # pressure averages at nearest y center 0.10 and 0.28, rack x between outer faces
 rack=(g['xc']>=g['left'][0])&(g['xc']<=g['right'][-1])
 rows=[]
 for yy in [0.10,0.28]:
  jj=int(np.argmin(abs(g['yc']-yy))); m=rack & g['fluid'][jj]
  rows.append(np.sum(s.p[jj,m]*g['dx'][m])/np.sum(g['dx'][m]))
 dp=rows[0]-rows[1]
 return qr,pk,dp

for h in [0.0025,0.001667,0.00125]:
 s=Solver(0.06,h)
 mind=min(s.g['dx'].min(),s.g['dy'].min()); dt=0.82*mind*mind/(4*NU)
 U=.38
 n0=3000 if h>=0.002 else 3500
 print('h',h,'grid',s.g['Nx'],s.g['Ny'],'dt',dt,'U0',U)
 r=s.solve(U,dt=dt,max_steps=n0,tol=1e-12,warm=False)
 q,pk,dp=metrics(s); print(' first',r,'m',q,pk,dp)
 fac=Q/q; U*=fac; s.u*=fac; s.v*=fac; s.p*=fac
 r=s.solve(U,dt=dt,max_steps=1200,tol=1e-12,warm=True)
 q,pk,dp=metrics(s); print(' second U',U,r,'m',q,pk,dp,'err%',100*(q/Q-1))
 fac=Q/q; U*=fac; s.u*=fac; s.v*=fac; s.p*=fac
 r=s.solve(U,dt=dt,max_steps=800,tol=1e-12,warm=True)
 q,pk,dp=metrics(s); print(' final U',U,r,'m',q,pk,dp,'err%',100*(q/Q-1))
