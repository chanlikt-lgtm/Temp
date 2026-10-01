import sys,multiprocessing as mp
sys.path.insert(0,'/mnt/data')
from mac_cfd_prototype import Solver

def run(args):
 label,pitch,U=args
 s=Solver(pitch,0.0025)
 r=s.solve(U,dt=.0013,max_steps=5500,tol=1e-12,warm=False)
 return label,U,r['qr'],r['res'],r['time']
if __name__=='__main__':
 args=[('Small',.04,.55),('Medium',.06,.34),('Large',.08,.22)]
 with mp.Pool(3) as p:
  for z in p.map(run,args): print(z,flush=True)
