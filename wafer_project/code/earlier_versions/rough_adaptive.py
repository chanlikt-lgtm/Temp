import sys,multiprocessing as mp, numpy as np
sys.path.insert(0,'/mnt/data')
from mac_cfd_adaptive import Solver, NU

def run(args):
 label,pitch,U=args
 s=Solver(pitch,20)
 mind=min(s.g['dx'].min(),s.g['dy'].min()); dt=.85*mind*mind/(4*NU)
 r=s.solve(U,dt=dt,max_steps=7000,tol=2e-5,warm=False)
 return label,U,dt,s.g['Nx'],s.g['Ny'],r
if __name__=='__main__':
 args=[('Small',.04,.9),('Medium',.06,.36),('Large',.08,.20)]
 with mp.Pool(3) as p:
  for z in p.map(run,args): print(z,flush=True)
