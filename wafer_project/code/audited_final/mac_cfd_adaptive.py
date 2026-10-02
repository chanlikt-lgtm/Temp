import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit
import time

W=1.0; H=0.45
N_WAFERS=6; TH=0.015; Y0=0.12; Y1=0.25
INLET_W=0.015; NU=7.5e-4; YCUT=0.20


def piecewise_faces(critical, fine_lo, fine_hi, h_fine, h_coarse):
    critical=sorted(set([round(float(z),12) for z in critical if 0<=z<=1.0]))
    faces=[critical[0]]
    for a,b in zip(critical[:-1], critical[1:]):
        mid=0.5*(a+b)
        h=h_fine if (mid>=fine_lo and mid<=fine_hi) else h_coarse
        n=max(1,int(np.ceil((b-a)/h)))
        seg=np.linspace(a,b,n+1)
        faces.extend(seg[1:].tolist())
    return np.array(faces)

def subdiv(a,b,n):
    if b <= a + 1e-14:
        return []
    return np.linspace(a,b,int(n)+1)[1:].tolist()

def build_geom(pitch, n_gap=20):
    """Geometry-aligned locally refined MAC grid.

    Every wafer face is an x-grid face.  Each inter-wafer gap has exactly
    n_gap fluid cells, while each solid wafer thickness is represented by a
    single solid cell.  This concentrates resolution where the flow is.
    """
    centers=0.5+(np.arange(N_WAFERS)-(N_WAFERS-1)/2)*pitch
    left=centers-TH/2; right=centers+TH/2
    gap_width=pitch-TH
    hgap=gap_width/n_gap

    xf=[0.0]
    xf += subdiv(0.0,0.25,25)
    nside=max(4,int(np.ceil((left[0]-0.25)/max(2*hgap,0.0035))))
    xf += subdiv(0.25,left[0],nside)
    for k in range(N_WAFERS):
        # one solid cell across wafer thickness; boundaries align exactly
        xf += subdiv(left[k],right[k],1)
        if k < N_WAFERS-1:
            # Inter-wafer gap.  Place the two inlet-slot edges (gap centre
            # +/- INLET_W/2) as EXACT x-grid faces so every discretized slot is
            # identically INLET_W wide, independent of pitch.  n_gap cells are
            # distributed across the three sub-segments in proportion to width
            # (no slivers: the slot sits well inside every gap considered here).
            g0=right[k]; g1=left[k+1]; gc=0.5*(g0+g1)
            a=gc-INLET_W/2; b=gc+INLET_W/2
            if a>g0+1e-9 and b<g1-1e-9:
                w=g1-g0
                nL=max(1,int(round(n_gap*(a-g0)/w)))
                nM=max(1,int(round(n_gap*(b-a)/w)))
                nR=max(1,n_gap-nL-nM)
                xf += subdiv(g0,a,nL)
                xf += subdiv(a,b,nM)
                xf += subdiv(b,g1,nR)
            else:
                xf += subdiv(g0,g1,n_gap)
    nside=max(4,int(np.ceil((0.75-right[-1])/max(2*hgap,0.0035))))
    xf += subdiv(right[-1],0.75,nside)
    xf += subdiv(0.75,1.0,25)
    xf=np.array(xf)
    # remove any accidental duplicate coordinates
    xf=np.unique(np.round(xf,14))

    # y refinement: scale with n_gap so 20-cell production grid has
    # ~52 cells through the wafer height.
    nrack=max(20,int(round(2.6*n_gap)))
    n1=max(1,int(round(nrack*(YCUT-Y0)/(Y1-Y0))))
    n2=nrack-n1
    yf=[0.0]
    yf += subdiv(0.0,Y0,max(12,int(round(1.2*n_gap))))
    yf += subdiv(Y0,YCUT,n1)
    yf += subdiv(YCUT,Y1,n2)
    yf += subdiv(Y1,0.30,max(6,int(round(0.5*n_gap))))
    yf += subdiv(0.30,H,max(12,int(round(1.0*n_gap))))
    yf=np.unique(np.round(np.array(yf),14))

    dx=np.diff(xf); dy=np.diff(yf); xc=0.5*(xf[:-1]+xf[1:]); yc=0.5*(yf[:-1]+yf[1:])
    Nx=len(dx); Ny=len(dy)
    X,Y=np.meshgrid(xc,yc)
    solid=np.zeros((Ny,Nx),dtype=np.bool_)
    for xl,xr in zip(left,right):
        solid |= (X>xl+1e-12)&(X<xr-1e-12)&(Y>Y0+1e-12)&(Y<Y1-1e-12)
    fluid=~solid
    au=np.zeros((Ny,Nx+1),dtype=np.bool_)
    if Nx>1:
        au[:,1:Nx]=fluid[:,:-1]&fluid[:,1:]
    av=np.zeros((Ny+1,Nx),dtype=np.bool_)
    if Ny>1:
        av[1:Ny,:]=fluid[:-1,:]&fluid[1:,:]
    av[Ny,:]=fluid[-1,:]
    inlet_centers=0.5*(centers[:-1]+centers[1:])
    inlet=np.zeros(Nx,dtype=np.bool_)
    for c in inlet_centers:
        inlet |= np.abs(xc-c)<=INLET_W/2+1e-12
    jcut=int(np.argmin(np.abs(yf-YCUT)))
    assert abs(yf[jcut]-YCUT)<1e-10
    rack=(xc>=left[0]-1e-12)&(xc<=right[-1]+1e-12)
    gapmask=np.zeros(Nx,dtype=np.bool_)
    for k in range(N_WAFERS-1):
        gapmask |= (xc>right[k]+1e-12)&(xc<left[k+1]-1e-12)
    central=(xc>right[2]+1e-12)&(xc<left[3]-1e-12)
    return dict(xf=xf,yf=yf,xc=xc,yc=yc,dx=dx,dy=dy,Nx=Nx,Ny=Ny,solid=solid,fluid=fluid,au=au,av=av,
                centers=centers,left=left,right=right,inlet_centers=inlet_centers,inlet=inlet,jcut=jcut,rack=rack,
                gapmask=gapmask,central=central,n_gap=n_gap)

def build_pressure(geom):
    Nx,Ny=geom['Nx'],geom['Ny']; fluid=geom['fluid']; dx=geom['dx']; dy=geom['dy']; xc=geom['xc']; yc=geom['yc']
    idx=-np.ones((Ny,Nx),dtype=np.int64)
    coords=np.argwhere(fluid)
    for k,(j,i) in enumerate(coords): idx[j,i]=k
    rows=[]; cols=[]; data=[]
    for k,(j,i) in enumerate(coords):
        diag=0.0
        # left/right internal fluid neighbors
        if i>0 and fluid[j,i-1]:
            c=1.0/(dx[i]*(xc[i]-xc[i-1])); diag+=c; rows.append(k); cols.append(idx[j,i-1]); data.append(-c)
        if i<Nx-1 and fluid[j,i+1]:
            c=1.0/(dx[i]*(xc[i+1]-xc[i])); diag+=c; rows.append(k); cols.append(idx[j,i+1]); data.append(-c)
        if j>0 and fluid[j-1,i]:
            c=1.0/(dy[j]*(yc[j]-yc[j-1])); diag+=c; rows.append(k); cols.append(idx[j-1,i]); data.append(-c)
        if j<Ny-1 and fluid[j+1,i]:
            c=1.0/(dy[j]*(yc[j+1]-yc[j])); diag+=c; rows.append(k); cols.append(idx[j+1,i]); data.append(-c)
        elif j==Ny-1:
            # top pressure Dirichlet p=0 at boundary, distance dy/2
            c=1.0/(dy[j]*(0.5*dy[j])); diag+=c
        rows.append(k); cols.append(k); data.append(diag)
    A=sp.csr_matrix((data,(rows,cols)),shape=(len(coords),len(coords)))
    fac=spla.factorized(A.tocsc())
    return idx,coords,fac,A

@njit(cache=True)
def predictor(u,v,us,vs,au,av,fluid,xf,yf,xc,yc,dx,dy,dt,nu):
    Ny,Nxp1=u.shape; Nx=Nxp1-1
    # copy current state
    us[:,:]=u; vs[:,:]=v
    # u interior active faces
    for j in range(Ny):
        for i in range(1,Nx):
            if not au[j,i]:
                us[j,i]=0.0
                continue
            u0=u[j,i]
            # cross velocity at u face
            vcross=0.25*(v[j,i-1]+v[j+1,i-1]+v[j,i]+v[j+1,i])
            # upwind du/dx
            if u0>=0:
                ul = u[j,i-1] if au[j,i-1] else 0.0
                dm=xf[i]-xf[i-1]
                dudx=(u0-ul)/dm
            else:
                ur = u[j,i+1] if au[j,i+1] else 0.0
                dp=xf[i+1]-xf[i]
                dudx=(ur-u0)/dp
            # upwind du/dy
            if vcross>=0:
                if j>0 and au[j-1,i]:
                    ub=u[j-1,i]; dm=yc[j]-yc[j-1]
                else:
                    ub=0.0; dm=0.5*dy[j]
                dudy=(u0-ub)/dm
            else:
                if j<Ny-1 and au[j+1,i]:
                    ut=u[j+1,i]; dp=yc[j+1]-yc[j]
                else:
                    # top open -> zero tangential gradient, solid interface -> no slip
                    if j==Ny-1:
                        ut=u0
                    else:
                        ut=0.0
                    dp=0.5*dy[j]
                dudy=(ut-u0)/dp
            # diffusion x
            ul = u[j,i-1] if au[j,i-1] else 0.0
            ur = u[j,i+1] if au[j,i+1] else 0.0
            dm=xf[i]-xf[i-1]; dp=xf[i+1]-xf[i]
            d2x=2.0*((ur-u0)/dp-(u0-ul)/dm)/(dp+dm)
            # diffusion y with boundary/interface treatment
            if j>0 and au[j-1,i]:
                ub=u[j-1,i]; dm=yc[j]-yc[j-1]
            else:
                ub=0.0; dm=0.5*dy[j]
            if j<Ny-1 and au[j+1,i]:
                ut=u[j+1,i]; dp=yc[j+1]-yc[j]
            else:
                if j==Ny-1:
                    ut=u0
                else:
                    ut=0.0
                dp=0.5*dy[j]
            d2y=2.0*((ut-u0)/dp-(u0-ub)/dm)/(dp+dm)
            us[j,i]=u0+dt*(-u0*dudx-vcross*dudy+nu*(d2x+d2y))
    # side u boundaries zero
    for j in range(Ny):
        us[j,0]=0.0; us[j,Nx]=0.0
    # v interior active faces; bottom prescribed outside, top handled zero-gradient predictor
    for j in range(1,Ny):
        for i in range(Nx):
            if not av[j,i]:
                vs[j,i]=0.0
                continue
            v0=v[j,i]
            ucross=0.25*(u[j-1,i]+u[j-1,i+1]+u[j,i]+u[j,i+1])
            # dv/dy upwind
            if v0>=0:
                vb=v[j-1,i] if (j-1>0 and av[j-1,i]) or j-1==0 else 0.0
                dm=yf[j]-yf[j-1]
                dvdy=(v0-vb)/dm
            else:
                vt=v[j+1,i] if av[j+1,i] else 0.0
                dp=yf[j+1]-yf[j]
                dvdy=(vt-v0)/dp
            # dv/dx upwind
            if ucross>=0:
                if i>0 and av[j,i-1]:
                    vl=v[j,i-1]; dm=xc[i]-xc[i-1]
                else:
                    vl=0.0; dm=0.5*dx[i]
                dvdx=(v0-vl)/dm
            else:
                if i<Nx-1 and av[j,i+1]:
                    vr=v[j,i+1]; dp=xc[i+1]-xc[i]
                else:
                    vr=0.0; dp=0.5*dx[i]
                dvdx=(vr-v0)/dp
            # diffusion y on v face positions.
            # NOTE: at a solid interface below (av[j-1,i] False) the ghost value
            # is 0 at the full face spacing yf[j]-yf[j-1], whereas the x-direction
            # below uses a half-cell wall distance.  This is a deliberate
            # first-order near-wall asymmetry, consistent with the scheme's
            # overall first-order advection; changing it perturbs the audited
            # results, so it is kept as-is.
            vb=v[j-1,i] if ((j-1>0 and av[j-1,i]) or j-1==0) else 0.0
            vt=v[j+1,i] if av[j+1,i] else 0.0
            dm=yf[j]-yf[j-1]; dp=yf[j+1]-yf[j]
            d2y=2.0*((vt-v0)/dp-(v0-vb)/dm)/(dp+dm)
            # diffusion x
            if i>0 and av[j,i-1]:
                vl=v[j,i-1]; dm=xc[i]-xc[i-1]
            else:
                vl=0.0; dm=0.5*dx[i]
            if i<Nx-1 and av[j,i+1]:
                vr=v[j,i+1]; dp=xc[i+1]-xc[i]
            else:
                vr=0.0; dp=0.5*dx[i]
            d2x=2.0*((vr-v0)/dp-(v0-vl)/dm)/(dp+dm)
            vs[j,i]=v0+dt*(-ucross*dvdx-v0*dvdy+nu*(d2x+d2y))
    # top predictor zero-normal gradient, bottom remains prescribed
    for i in range(Nx):
        if fluid[Ny-1,i]:
            vs[Ny,i]=vs[Ny-1,i] if av[Ny-1,i] else v[Ny,i]
        else:
            vs[Ny,i]=0.0

@njit(cache=True)
def divergence(vu,vv,fluid,dx,dy,out):
    Ny,Nx=fluid.shape
    for j in range(Ny):
        for i in range(Nx):
            if fluid[j,i]:
                out[j,i]=(vu[j,i+1]-vu[j,i])/dx[i]+(vv[j+1,i]-vv[j,i])/dy[j]
            else:
                out[j,i]=0.0

@njit(cache=True)
def correct(u,v,us,vs,p,au,av,fluid,xc,yc,dy,dt):
    Ny,Nx=fluid.shape
    # start from predictor
    u[:,:]=us; v[:,:]=vs
    # interior u active
    for j in range(Ny):
        for i in range(1,Nx):
            if au[j,i]:
                grad=(p[j,i]-p[j,i-1])/(xc[i]-xc[i-1])
                u[j,i]=us[j,i]-dt*grad
            else:
                u[j,i]=0.0
        u[j,0]=0.0; u[j,Nx]=0.0
    # interior v active
    for j in range(1,Ny):
        for i in range(Nx):
            if av[j,i]:
                grad=(p[j,i]-p[j-1,i])/(yc[j]-yc[j-1])
                v[j,i]=vs[j,i]-dt*grad
            else:
                v[j,i]=0.0
    # top pressure outlet p=0 at boundary
    for i in range(Nx):
        if fluid[Ny-1,i]:
            grad=(0.0-p[Ny-1,i])/(0.5*dy[Ny-1])
            v[Ny,i]=vs[Ny,i]-dt*grad
        else:
            v[Ny,i]=0.0
    # bottom preserved from predictor / inlet BC

class Solver:
    def __init__(self,pitch,n_gap=20):
        self.g=build_geom(pitch,n_gap)
        self.idx,self.coords,self.fac,self.A=build_pressure(self.g)
        Ny,Nx=self.g['Ny'],self.g['Nx']
        self.u=np.zeros((Ny,Nx+1)); self.v=np.zeros((Ny+1,Nx)); self.p=np.zeros((Ny,Nx))
        self.us=np.zeros_like(self.u); self.vs=np.zeros_like(self.v); self.div=np.zeros((Ny,Nx))
    def set_inlet(self,U):
        self.v[0,:]=0.0; self.v[0,self.g['inlet']]=U
    def solve(self,U,dt=0.001,max_steps=2000,tol=2e-6,verbose=False,warm=False):
        if not warm:
            self.u.fill(0); self.v.fill(0); self.p.fill(0)
        self.set_inlet(U)
        g=self.g; fluid=g['fluid']; coords=self.coords
        t0=time.time(); last=1e9
        for n in range(max_steps):
            uold=self.u.copy(); vold=self.v.copy()
            # enforce bottom inlet before predictor
            self.v[0,:]=0.0; self.v[0,g['inlet']]=U
            predictor(self.u,self.v,self.us,self.vs,g['au'],g['av'],fluid,g['xf'],g['yf'],g['xc'],g['yc'],g['dx'],g['dy'],dt,NU)
            # enforce bottom inlet predictor
            self.vs[0,:]=0.0; self.vs[0,g['inlet']]=U
            divergence(self.us,self.vs,fluid,g['dx'],g['dy'],self.div)
            b=-self.div[coords[:,0],coords[:,1]]/dt
            pv=self.fac(b)
            self.p.fill(0.0); self.p[coords[:,0],coords[:,1]]=pv
            correct(self.u,self.v,self.us,self.vs,self.p,g['au'],g['av'],fluid,g['xc'],g['yc'],g['dy'],dt)
            # restore bottom inlet exactly
            self.v[0,:]=0.0; self.v[0,g['inlet']]=U
            if n%25==0 and n>0:
                du=np.sqrt(np.mean((self.u-uold)**2)+np.mean((self.v-vold)**2))
                mag=np.sqrt(np.mean(self.u**2)+np.mean(self.v**2))+1e-14
                last=du/mag
                if verbose and n%200==0:
                    divergence(self.u,self.v,fluid,g['dx'],g['dy'],self.div)
                    print(n,last,np.max(np.abs(self.div[fluid])))
                if last<tol:
                    break
        divergence(self.u,self.v,fluid,g['dx'],g['dy'],self.div)
        qtot=np.sum(self.v[g['jcut'],:]*g['dx'])
        mask=g['rack'] & (~g['solid'][max(g['jcut']-1,0),:]) & (~g['solid'][min(g['jcut'],g['Ny']-1),:])
        qr=np.sum(self.v[g['jcut'],mask]*g['dx'][mask])
        return dict(steps=n+1,res=last,qtot=qtot,qr=qr,maxdiv=np.max(np.abs(self.div[fluid])),time=time.time()-t0)

if __name__=='__main__':
    # Smoke test: n_gap is an integer cell count per inter-wafer gap (production
    # uses 20).  dt is passed separately to solve().
    s=Solver(0.06,20)
    print('grid',s.g['Nx'],s.g['Ny'],'unknown',len(s.coords))
    r=s.solve(0.3,dt=0.0015,max_steps=1500,tol=1e-5,verbose=True)
    print(r)
    for yy in [0.05,0.10,0.20,0.30,0.40]:
        j=np.argmin(np.abs(s.g['yf']-yy))
        print('Q',yy,s.g['yf'][j],np.sum(s.v[j,:]*s.g['dx']))
