import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
BASE='/tmp/claude-0/-home-user-Temp/36b846dc-25a1-5185-8f16-2e39764cc5de/scratchpad/wafer'

for label in ['Small','Medium','Large']:
    z=np.load(f'{BASE}/{label.lower()}_verify.npz')
    u,v=z['u'],z['v']; xc,yc,xf,yf=z['xc'],z['yc'],z['xf'],z['yf']
    solid=z['solid'].astype(bool); left=z['left']; right=z['right']
    pitch=float(z['pitch']); U=float(z['U_in']); Q=float(z['Q_rack'])
    uc=0.5*(u[:,:-1]+u[:,1:]); vc=0.5*(v[:-1,:]+v[1:,:])
    spd=np.where(solid,np.nan,np.sqrt(uc**2+vc**2))
    fig,ax=plt.subplots(figsize=(12,5.3))
    pcm=ax.pcolormesh(xf,yf,spd,shading='flat',cmap='viridis')
    skipx=max(1,len(xc)//40); skipy=max(1,len(yc)//22)
    X,Y=np.meshgrid(xc[::skipx],yc[::skipy])
    UQ=np.ma.array(uc[::skipy,::skipx],mask=solid[::skipy,::skipx])
    VQ=np.ma.array(vc[::skipy,::skipx],mask=solid[::skipy,::skipx])
    ax.quiver(X,Y,UQ,VQ,color='white',scale=2.4,width=0.0022,headwidth=3.2,headlength=3.6)
    for xl,xr in zip(left,right):
        ax.add_patch(Rectangle((xl,0.12),xr-xl,0.13,facecolor='0.85',edgecolor='black',lw=0.9,zorder=5))
    ax.axhline(0.20,ls='--',lw=1.2,color='white',alpha=0.9)
    ax.set(xlim=(0,1),ylim=(0,0.45),xlabel='x',ylabel='y',
           title=f'{label} pitch P={pitch:.3f}  |  MAC Navier-Stokes equal rack flow  |  '
                 f'Q_rack={Q:.6f}, U_in={U:.5f}')
    cb=fig.colorbar(pcm,ax=ax,pad=0.015); cb.set_label(r'speed  $|\mathbf{u}|$')
    fig.tight_layout()
    out=f'{BASE}/field_{label.lower()}_2d.png'
    fig.savefig(out,dpi=170,bbox_inches='tight'); plt.close(fig)
    print('wrote',out)
