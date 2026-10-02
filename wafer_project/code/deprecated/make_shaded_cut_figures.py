import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

base='/mnt/data/wafer_pitch_equalQ_report'
figdir=os.path.join(base,'figures')
os.makedirs(figdir, exist_ok=True)

df=pd.read_csv(os.path.join(base,'horizontal_cut_equalQ_y_0p20.csv'))
WAFER_THICKNESS=0.015
N=6
cases=[('Small',0.040,'small_v'),('Medium',0.060,'medium_v'),('Large',0.080,'large_v')]

# Three-panel exact equal-Q horizontal cut with each case's own wafer bands.
fig, axes = plt.subplots(3,1,figsize=(10.0,8.0),sharex=True,sharey=False)
for ax,(name,pitch,col) in zip(axes,cases):
    x=df['x'].to_numpy(); v=df[col].to_numpy()
    wc=0.5+(np.arange(N)-(N-1)/2)*pitch
    left=wc-WAFER_THICKNESS/2; right=wc+WAFER_THICKNESS/2
    rack_lo=left[0]-0.02; rack_hi=right[-1]+0.02
    # Shade exact solid wafer locations.
    for xl,xr in zip(left,right):
        ax.axvspan(xl,xr,facecolor='0.82',edgecolor='0.45',linewidth=0.6,zorder=0)
    # Plot only fluid values. NaNs in the original CSV already remove solids.
    ax.plot(x,v,linewidth=1.8,label=f'{name}: P={pitch:.3f}')
    ax.axhline(0.0,linewidth=0.8,color='0.35')
    ax.set_xlim(rack_lo,rack_hi)
    ax.grid(True,alpha=0.25)
    ax.set_ylabel(r'$v(x,y=0.20)$')
    ax.legend(loc='upper right',frameon=True)
axes[-1].set_xlabel('x')
fig.suptitle('Equal-$Q_{rack}$ horizontal cut: solid wafers shaded; $v(x)$ shown only in fluid gaps',y=0.995,fontsize=13)
fig.tight_layout(rect=[0,0,1,0.975])
out=os.path.join(figdir,'horizontal_cut_equalQ_v_shaded_panels.png')
fig.savefig(out,dpi=220,bbox_inches='tight')
plt.close(fig)

# Single medium-pitch close-up emphasizing no-slip wall locations.
name,pitch,col=cases[1]
x=df['x'].to_numpy(); v=df[col].to_numpy(); wc=0.5+(np.arange(N)-(N-1)/2)*pitch
left=wc-WAFER_THICKNESS/2; right=wc+WAFER_THICKNESS/2
fig,ax=plt.subplots(figsize=(9.6,4.4))
for xl,xr in zip(left,right):
    ax.axvspan(xl,xr,facecolor='0.82',edgecolor='0.45',linewidth=0.7,zorder=0)
ax.plot(x,v,linewidth=2.0,label='Medium pitch direct CFD nodes')
ax.axhline(0.0,linewidth=0.8,color='0.35')
ax.set_xlim(left[0]-0.02,right[-1]+0.02)
ax.set_xlabel('x'); ax.set_ylabel(r'$v(x,y=0.20)$')
ax.set_title('Medium pitch: wafer solids made explicit on the horizontal cut')
ax.grid(True,alpha=0.25); ax.legend(loc='upper right')
fig.tight_layout()
out2=os.path.join(figdir,'horizontal_cut_medium_equalQ_v_shaded.png')
fig.savefig(out2,dpi=220,bbox_inches='tight')
plt.close(fig)

print(out)
print(out2)
