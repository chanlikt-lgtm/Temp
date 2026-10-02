import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
base='/mnt/data/wafer_pitch_equalQ_report'
figdir=os.path.join(base,'figures')
WAFER_THICKNESS=0.015; N=6
cases=[('Small',0.040,'small_v'),('Medium',0.060,'medium_v'),('Large',0.080,'large_v')]
df=pd.read_csv(os.path.join(base,'horizontal_cut_equalQ_y_0p20.csv'))

fig,axes=plt.subplots(3,1,figsize=(10.2,7.8),sharex=False)
for ax,(name,pitch,col) in zip(axes,cases):
    x=df['x'].to_numpy(); v=df[col].to_numpy()
    wc=0.5+(np.arange(N)-(N-1)/2)*pitch
    lefts=wc-WAFER_THICKNESS/2; rights=wc+WAFER_THICKNESS/2
    for xl,xr in zip(lefts,rights):
        ax.axvspan(xl,xr,facecolor='0.82',edgecolor='0.45',linewidth=0.7,zorder=0)
    # Plot only the five interior fluid gaps. Add exact no-slip BC endpoints v=0 at wafer faces.
    first=True
    for i in range(N-1):
        gl=rights[i]; gr=lefts[i+1]
        m=(x>gl+1e-12)&(x<gr-1e-12)&np.isfinite(v)
        xx=np.concatenate(([gl],x[m],[gr]))
        vv=np.concatenate(([0.0],v[m],[0.0]))
        ax.plot(xx,vv,linewidth=1.9,label=f'{name}: P={pitch:.3f}' if first else None)
        ax.scatter([gl,gr],[0,0],s=9,zorder=3)
        first=False
    ax.axhline(0,color='0.35',lw=0.8)
    ax.set_xlim(lefts[0]-0.006,rights[-1]+0.006)
    ax.set_ylabel(r'$v(x,y=0.20)$')
    ax.grid(True,alpha=0.25)
    ax.legend(loc='upper right')
axes[-1].set_xlabel('x')
fig.suptitle('Equal-$Q_{rack}$ cut through the five fluid gaps: wafer solids shaded and no-slip wall values shown',fontsize=13,y=0.995)
fig.tight_layout(rect=[0,0,1,0.975])
out=os.path.join(figdir,'horizontal_cut_equalQ_v_gap_bc_panels.png')
fig.savefig(out,dpi=220,bbox_inches='tight')
plt.close(fig)

# Refined-grid representative medium-pitch wall-resolution diagnostic.
rdf=pd.read_csv('/mnt/data/wafer_pitch_equalQ_refined_fast/horizontal_cut_v_y_0p20_refined.csv')
x=rdf['x'].to_numpy(); v=rdf['v_medium'].to_numpy(); pitch=0.060
wc=0.5+(np.arange(N)-(N-1)/2)*pitch
lefts=wc-WAFER_THICKNESS/2; rights=wc+WAFER_THICKNESS/2
fig,ax=plt.subplots(figsize=(9.8,4.2))
for xl,xr in zip(lefts,rights):
    ax.axvspan(xl,xr,facecolor='0.82',edgecolor='0.45',linewidth=0.7,zorder=0)
for i in range(N-1):
    gl=rights[i]; gr=lefts[i+1]
    m=(x>gl+1e-12)&(x<gr-1e-12)&np.isfinite(v)
    xx=np.concatenate(([gl],x[m],[gr]))
    vv=np.concatenate(([0.0],v[m],[0.0]))
    ax.plot(xx,vv,color='C0',linewidth=1.9)
    ax.scatter([gl,gr],[0,0],color='C0',s=10,zorder=3)
ax.axhline(0,color='0.35',lw=0.8)
ax.set_xlim(lefts[0]-0.006,rights[-1]+0.006)
ax.set_xlabel('x'); ax.set_ylabel(r'$v(x,y=0.20)$')
ax.set_title('Refined-grid wall-resolution diagnostic, medium pitch (241 x 109)')
ax.grid(True,alpha=0.25)
fig.tight_layout()
out2=os.path.join(figdir,'horizontal_cut_medium_refined_gap_bc.png')
fig.savefig(out2,dpi=220,bbox_inches='tight')
plt.close(fig)
print(out); print(out2)
