import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE=os.environ.get('WAFER_OUT', os.path.dirname(os.path.abspath(__file__)))
z=np.load(BASE+'/large_verify.npz')
u,v,p=z['u'],z['v'],z['p']
xc,yc,xf,yf=z['xc'],z['yc'],z['xf'],z['yf']
solid=z['solid'].astype(bool); left=z['left']; right=z['right']
pitch=float(z['pitch']); U=float(z['U_in']); Q=float(z['Q_rack'])

# cell-centred velocity from staggered faces
uc=0.5*(u[:,:-1]+u[:,1:]); vc=0.5*(v[:-1,:]+v[1:,:])
spd=np.sqrt(uc**2+vc**2); spd=np.where(solid,np.nan,spd)

fig,ax=plt.subplots(figsize=(12,5.3))
pcm=ax.pcolormesh(xf,yf,spd,shading='flat',cmap='viridis')
# quiver: subsample so arrows are legible
skipx=max(1,len(xc)//40); skipy=max(1,len(yc)//22)
X,Y=np.meshgrid(xc[::skipx],yc[::skipy])
UQ=np.ma.array(uc[::skipy,::skipx],mask=solid[::skipy,::skipx])
VQ=np.ma.array(vc[::skipy,::skipx],mask=solid[::skipy,::skipx])
ax.quiver(X,Y,UQ,VQ,color='white',scale=2.2,width=0.0022,headwidth=3.2,headlength=3.6)
# draw wafers
for xl,xr in zip(left,right):
    ax.add_patch(Rectangle((xl,0.12),xr-xl,0.13,facecolor='0.85',edgecolor='black',lw=0.9,zorder=5))
ax.axhline(0.20,ls='--',lw=1.2,color='white',alpha=0.9)
ax.set(xlim=(0,1),ylim=(0,0.45),xlabel='x',ylabel='y',
       title=f'Large pitch P={pitch:.3f}  |  MAC Navier-Stokes equal rack flow  |  '
             f'Q_rack={Q:.6f}, U_in={U:.5f}')
cb=fig.colorbar(pcm,ax=ax,pad=0.015); cb.set_label(r'speed  $|\mathbf{u}|$')
fig.tight_layout()
out=BASE+'/large_vector_field.png'
fig.savefig(out,dpi=170,bbox_inches='tight'); plt.close(fig)
print('wrote',out)

# zoomed view on the rack region
fig,ax=plt.subplots(figsize=(11,5.0))
pcm=ax.pcolormesh(xf,yf,spd,shading='flat',cmap='viridis')
skipx=max(1,len(xc)//70); skipy=max(1,len(yc)//30)
X,Y=np.meshgrid(xc[::skipx],yc[::skipy])
UQ=np.ma.array(uc[::skipy,::skipx],mask=solid[::skipy,::skipx])
VQ=np.ma.array(vc[::skipy,::skipx],mask=solid[::skipy,::skipx])
ax.quiver(X,Y,UQ,VQ,color='white',scale=1.6,width=0.0025,headwidth=3.2)
for xl,xr in zip(left,right):
    ax.add_patch(Rectangle((xl,0.12),xr-xl,0.13,facecolor='0.85',edgecolor='black',lw=0.9,zorder=5))
ax.axhline(0.20,ls='--',lw=1.1,color='white',alpha=0.9)
ax.set(xlim=(left[0]-0.06,right[-1]+0.06),ylim=(0.02,0.34),xlabel='x',ylabel='y',
       title='Large pitch - zoom on wafer rack (dashed line = y=0.20 flux cut)')
cb=fig.colorbar(pcm,ax=ax,pad=0.015); cb.set_label(r'speed  $|\mathbf{u}|$')
fig.tight_layout()
out2=BASE+'/large_vector_field_zoom.png'
fig.savefig(out2,dpi=170,bbox_inches='tight'); plt.close(fig)
print('wrote',out2)
