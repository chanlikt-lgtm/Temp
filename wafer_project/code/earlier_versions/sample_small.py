import sys,numpy as np,multiprocessing as mp,time
sys.path.insert(0,'/mnt/data')
from mac_cfd_adaptive import Solver,NU
Q=.0055463

def f(U):
 s=Solver(.04,20);g=s.g;h=min(g['dx'].min(),g['dy'].min());dt=.9*h*h/(4*NU)
 r=s.solve(U,dt=dt,max_steps=7000,tol=2e-5,warm=False)
 q=float(np.sum(s.v[g['jcut'],g['gapmask']]*g['dx'][g['gapmask']]))
 return U,q,100*(q/Q-1),r['res'],r['steps']
if __name__=='__main__':
 with mp.get_context('fork').Pool(2) as p: print(p.map(f,[.72,.75]))
