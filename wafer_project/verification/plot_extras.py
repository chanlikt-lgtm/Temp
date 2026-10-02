import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
BASE=os.environ.get('WAFER_OUT', os.path.dirname(os.path.abspath(__file__)))
LBL=['Small','Medium','Large']; colors={'Small':'tab:blue','Medium':'tab:orange','Large':'tab:green'}
def load(label):
    z=np.load(f'{BASE}/{label.lower()}_verify.npz')
    return z

# ---------- 1) signed v(x) overlay at y=0.20 ----------
fig,ax=plt.subplots(figsize=(11,5.2))
for label in LBL:
    z=load(label); v=z['v']; xc=z['xc']; yf=z['yf']
    left=z['left']; right=z['right']; U=float(z['U_in']); pitch=float(z['pitch'])
    j=int(np.argmin(np.abs(yf-0.20))); vv=v[j,:]
    # plot each of the 5 interior clear gaps with imposed no-slip zeros at wafer faces
    first=True
    for k in range(len(left)-1):
        m=(xc>right[k])&(xc<left[k+1])
        xx=np.r_[right[k],xc[m],left[k+1]]; yy=np.r_[0.0,vv[m],0.0]
        ax.plot(xx,yy,lw=1.9,color=colors[label],
                label=(f'{label} P={pitch:.3f}, U_in={U:.4f}' if first else None)); first=False
for xl,xr in zip(load('Small')['left'],load('Small')['right']):
    pass
ax.axhline(0,color='0.3',lw=0.7)
ax.set_xlim(0.27,0.73); ax.set_xlabel('x'); ax.set_ylabel(r'$v(x,\,0.20)$  (vertical velocity)')
ax.set_title('Overlaid y=0.20 cut: signed vertical velocity v(x) - equal rack flow')
ax.grid(alpha=0.25); ax.legend()
fig.tight_layout(); fig.savefig(f'{BASE}/ycut_v_overlay.png',dpi=175,bbox_inches='tight'); plt.close(fig)
print('wrote ycut_v_overlay.png')

# ---------- 2) streamlines over speed (per case, one figure) ----------
fig,axes=plt.subplots(3,1,figsize=(12,13.5))
for ax,label in zip(axes,LBL):
    z=load(label); u,v=z['u'],z['v']; xc,yc,xf,yf=z['xc'],z['yc'],z['xf'],z['yf']
    solid=z['solid'].astype(bool); left=z['left']; right=z['right']
    U=float(z['U_in']); pitch=float(z['pitch'])
    uc=0.5*(u[:,:-1]+u[:,1:]); vc=0.5*(v[:-1,:]+v[1:,:])
    spd=np.where(solid,np.nan,np.sqrt(uc**2+vc**2))
    UC=np.where(solid,0.0,uc); VC=np.where(solid,0.0,vc)
    pcm=ax.pcolormesh(xf,yf,spd,shading='flat',cmap='viridis')
    # streamplot needs a regular grid; xc/yc are nonuniform -> interpolate to uniform grid
    nx,ny=240,120
    xg=np.linspace(0,1,nx); yg=np.linspace(0,0.45,ny)
    Xg,Yg=np.meshgrid(xg,yg)
    from scipy.interpolate import RegularGridInterpolator
    fu=RegularGridInterpolator((yc,xc),UC,bounds_error=False,fill_value=0.0)
    fv=RegularGridInterpolator((yc,xc),VC,bounds_error=False,fill_value=0.0)
    pts=np.column_stack([Yg.ravel(),Xg.ravel()])
    Ug=fu(pts).reshape(ny,nx); Vg=fv(pts).reshape(ny,nx)
    ax.streamplot(xg,yg,Ug,Vg,color='white',density=1.5,linewidth=0.7,arrowsize=0.8)
    for xl,xr in zip(left,right):
        ax.add_patch(Rectangle((xl,0.12),xr-xl,0.13,facecolor='0.85',edgecolor='black',lw=0.9,zorder=5))
    ax.set(xlim=(0,1),ylim=(0,0.45),ylabel='y',
           title=f'{label} P={pitch:.3f}  streamlines over speed  (U_in={U:.4f})')
    fig.colorbar(pcm,ax=ax,pad=0.012).set_label(r'$|\mathbf{u}|$')
axes[-1].set_xlabel('x')
fig.tight_layout(); fig.savefig(f'{BASE}/streamlines_all.png',dpi=150,bbox_inches='tight'); plt.close(fig)
print('wrote streamlines_all.png')

# ---------- 3) pressure contours (per case, one figure) ----------
fig,axes=plt.subplots(3,1,figsize=(12,13.5))
for ax,label in zip(axes,LBL):
    z=load(label); p=z['p']; xf,yf,xc,yc=z['xf'],z['yf'],z['xc'],z['yc']
    solid=z['solid'].astype(bool); left=z['left']; right=z['right']
    U=float(z['U_in']); pitch=float(z['pitch'])
    pp=np.where(solid,np.nan,p)
    vmax=np.nanmax(np.abs(pp))
    pcm=ax.pcolormesh(xf,yf,pp,shading='flat',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    X,Y=np.meshgrid(xc,yc)
    cs=ax.contour(X,Y,pp,levels=12,colors='k',linewidths=0.5,alpha=0.6)
    for xl,xr in zip(left,right):
        ax.add_patch(Rectangle((xl,0.12),xr-xl,0.13,facecolor='0.85',edgecolor='black',lw=0.9,zorder=5))
    ax.axhline(0.20,ls='--',lw=1.0,color='0.3',alpha=0.8)
    ax.set(xlim=(0,1),ylim=(0,0.45),ylabel='y',
           title=f'{label} P={pitch:.3f}  pressure field  (U_in={U:.4f})')
    fig.colorbar(pcm,ax=ax,pad=0.012).set_label('p (kinematic)')
axes[-1].set_xlabel('x')
fig.tight_layout(); fig.savefig(f'{BASE}/pressure_all.png',dpi=150,bbox_inches='tight'); plt.close(fig)
print('wrote pressure_all.png')
