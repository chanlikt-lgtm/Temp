import os, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
W=1.0; H=0.45; N_WAFERS=6; WAFER_THICKNESS=0.015; WAFER_Y0=0.12; WAFER_Y1=0.25
INLET_WIDTH=0.015; NU=7.5e-4; NX=201; NY=91; DT=0.0012
Y_CUT=0.20

def setup(pitch):
    x=np.linspace(0,W,NX); y=np.linspace(0,H,NY); dx=x[1]-x[0]; dy=y[1]-y[0]
    X,Y=np.meshgrid(x,y)
    wc=0.5+(np.arange(N_WAFERS)-(N_WAFERS-1)/2)*pitch
    solid=np.zeros((NY,NX),bool)
    for xc in wc:
        solid |= ((np.abs(X-xc)<=WAFER_THICKNESS/2+1e-12)&(Y>=WAFER_Y0-1e-12)&(Y<=WAFER_Y1+1e-12))
    ic=0.5*(wc[:-1]+wc[1:]); inlet=np.zeros(NX,bool)
    for xc in ic: inlet |= np.abs(x-xc)<=INLET_WIDTH/2+1e-12
    fluid=~solid; idx=-np.ones((NY,NX),int); coords=np.argwhere(fluid[1:-1,1:-1]); coords[:,0]+=1; coords[:,1]+=1
    for k,(j,i) in enumerate(coords): idx[j,i]=k
    rows=[]; cols=[]; data=[]; idx2=1/dx**2; idy2=1/dy**2
    for k,(j,i) in enumerate(coords):
        diag=0.0
        for jj,ii,c in [(j,i+1,idx2),(j,i-1,idx2),(j+1,i,idy2),(j-1,i,idy2)]:
            if jj==NY-1:
                if fluid[jj,ii]: diag+=c
            elif ii==0 or ii==NX-1 or jj==0: continue
            elif fluid[jj,ii]:
                kk=idx[jj,ii]
                if kk>=0: diag+=c; rows.append(k); cols.append(kk); data.append(-c)
        rows.append(k); cols.append(k); data.append(diag)
    A=sp.csr_matrix((data,(rows,cols)),shape=(len(coords),len(coords)))
    solver=spla.factorized(A.tocsc())
    iy=int(np.argmin(np.abs(y-Y_CUT)))
    gaps=np.zeros(NX,bool)
    for a,b in zip(wc[:-1],wc[1:]):
        left=a+WAFER_THICKNESS/2; right=b-WAFER_THICKNESS/2
        gaps |= (x>left+1e-12)&(x<right-1e-12)
    return dict(pitch=pitch,x=x,y=y,dx=dx,dy=dy,solid=solid,fluid=fluid,Fi=fluid[1:-1,1:-1],wc=wc,ic=ic,inlet=inlet,coords=coords,solver=solver,iy=iy,gaps=gaps)

def apply(u,v,c,uin):
    solid=c['solid']; inlet=c['inlet']; dx=c['dx']; qin=np.sum(inlet)*dx*uin
    u[:,0]=0;v[:,0]=0;u[:,-1]=0;v[:,-1]=0
    u[0,:]=0;v[0,:]=0;v[0,inlet]=uin
    u[-1,:]=u[-2,:];v[-1,:]=v[-2,:]
    u[solid]=0;v[solid]=0
    qout=np.sum(v[-1,:])*dx
    if qout>1e-14: v[-1,:]*=qin/qout
    v[-1,0]=0;v[-1,-1]=0

def solve(c,uin,max_steps=2200,init=None):
    dx=c['dx'];dy=c['dy'];solid=c['solid'];fluid=c['fluid'];Fi=c['Fi'];coords=c['coords'];solver=c['solver'];inlet=c['inlet']
    if init is None:
        u=np.zeros((NY,NX));v=np.zeros_like(u);p=np.zeros_like(u)
    else:
        u=init[0].copy();v=init[1].copy();p=np.zeros_like(u)
    apply(u,v,c,uin); res=np.nan
    for n in range(max_steps):
        un=u.copy();vn=v.copy();uc=un[1:-1,1:-1];vc=vn[1:-1,1:-1]
        dudx=np.where(uc>=0,(uc-un[1:-1,:-2])/dx,(un[1:-1,2:]-uc)/dx)
        dudy=np.where(vc>=0,(uc-un[:-2,1:-1])/dy,(un[2:,1:-1]-uc)/dy)
        dvdx=np.where(uc>=0,(vc-vn[1:-1,:-2])/dx,(vn[1:-1,2:]-vc)/dx)
        dvdy=np.where(vc>=0,(vc-vn[:-2,1:-1])/dy,(vn[2:,1:-1]-vc)/dy)
        lapu=(un[1:-1,2:]-2*uc+un[1:-1,:-2])/dx**2+(un[2:,1:-1]-2*uc+un[:-2,1:-1])/dy**2
        lapv=(vn[1:-1,2:]-2*vc+vn[1:-1,:-2])/dx**2+(vn[2:,1:-1]-2*vc+vn[:-2,1:-1])/dy**2
        us=un.copy();vs=vn.copy();us[1:-1,1:-1]=np.where(Fi,uc+DT*(-uc*dudx-vc*dudy+NU*lapu),0);vs[1:-1,1:-1]=np.where(Fi,vc+DT*(-uc*dvdx-vc*dvdy+NU*lapv),0)
        apply(us,vs,c,uin)
        div=(us[1:-1,2:]-us[1:-1,:-2])/(2*dx)+(vs[2:,1:-1]-vs[:-2,1:-1])/(2*dy)
        rhs=np.zeros_like(p);rhs[1:-1,1:-1]=np.where(Fi,div/DT,0);b=-rhs[coords[:,0],coords[:,1]];pv=solver(b)
        p.fill(0);p[coords[:,0],coords[:,1]]=pv;p[:,0]=p[:,1];p[:,-1]=p[:,-2];p[0,:]=p[1,:];p[-1,:]=0;p[solid]=0
        pc=p[1:-1,1:-1];pe=np.where(fluid[1:-1,2:],p[1:-1,2:],pc);pw=np.where(fluid[1:-1,:-2],p[1:-1,:-2],pc);pn=np.where(fluid[2:,1:-1],p[2:,1:-1],pc);ps=np.where(fluid[:-2,1:-1],p[:-2,1:-1],pc)
        u=us.copy();v=vs.copy();u[1:-1,1:-1]=np.where(Fi,us[1:-1,1:-1]-DT*(pe-pw)/(2*dx),0);v[1:-1,1:-1]=np.where(Fi,vs[1:-1,1:-1]-DT*(pn-ps)/(2*dy),0);apply(u,v,c,uin)
        if n%100==0 and n>0:
            du=np.sqrt(np.mean((u-un)**2+(v-vn)**2));um=np.sqrt(np.mean(u*u+v*v))+1e-14;res=du/um
            if res<2e-7 and n>800: break
    iy=c['iy'];g=c['gaps'];qr=np.sum(v[iy,g])*dx; spd=np.hypot(u,v);spd[solid]=np.nan
    return u,v,p,spd,qr,res,n+1

if __name__=='__main__':
    c=setup(0.06)
    r=solve(c,0.3)
    print('q rack',r[4],'res',r[5],'steps',r[6],'gap points',np.sum(c['gaps']),'gap width numeric',np.sum(c['gaps'])*c['dx'])
