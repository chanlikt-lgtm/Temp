import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE='/tmp/claude-0/-home-user-Temp/36b846dc-25a1-5185-8f16-2e39764cc5de/scratchpad/wafer'
YCUT=0.20
colors={'Small':'tab:blue','Medium':'tab:orange','Large':'tab:green'}

fig,ax=plt.subplots(figsize=(11,5.2))
for label in ['Small','Medium','Large']:
    z=np.load(f'{BASE}/{label.lower()}_verify.npz')
    u,v=z['u'],z['v']; xc,yc=z['xc'],z['yc']; solid=z['solid'].astype(bool)
    U=float(z['U_in']); pitch=float(z['pitch'])
    # cell-centred speed on the row nearest y=0.20
    uc=0.5*(u[:,:-1]+u[:,1:]); vc=0.5*(v[:-1,:]+v[1:,:])
    spd=np.sqrt(uc**2+vc**2)
    j=int(np.argmin(np.abs(yc-YCUT)))
    s=spd[j,:].copy(); s[solid[j,:]]=np.nan   # blank out wafer interiors
    ax.plot(xc,s,lw=1.9,color=colors[label],
            label=f'{label}  P={pitch:.3f},  U_in={U:.4f}')
ax.set_xlabel('x'); ax.set_ylabel(r'flow speed $|\mathbf{u}|$ at y=0.20')
ax.set_title('Overlaid horizontal (y=0.20) cut of flow speed - equal rack flow, all pitches')
ax.set_xlim(0.20,0.80); ax.grid(alpha=0.25); ax.legend()
fig.tight_layout()
out=f'{BASE}/ycut_speed_overlay.png'
fig.savefig(out,dpi=175,bbox_inches='tight'); plt.close(fig)
print('wrote',out)
