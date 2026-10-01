import os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

OUT='/mnt/data/wafer_mac_equalQ_v2'
FIG=os.path.join(OUT,'figures'); os.makedirs(FIG,exist_ok=True)
CASES=['Small','Medium','Large']

prod=pd.read_csv(os.path.join(OUT,'production_summary.csv'))
mesh=pd.read_csv(os.path.join(OUT,'mesh_study_small.csv'))
time=pd.read_csv(os.path.join(OUT,'timestep_study_small.csv'))
mass=pd.read_csv(os.path.join(OUT,'mass_conservation_table.csv'))

meta={
  'study':'Equal rack flow-rate wafer pitch study',
  'governing_equations':'2-D incompressible Navier-Stokes',
  'discretization':'MAC staggered grid; pressure at cell centers, u/v at cell faces',
  'grid':'geometry-aligned locally refined nonuniform grid',
  'advection':'first-order donor-cell upwind',
  'diffusion':'second-order nonuniform finite difference',
  'projection':'compatible discrete divergence and gradient; sparse pressure-Poisson solve',
  'pressure_boundary':'p=0 at open top outlet',
  'velocity_boundaries':'no-slip tank side/bottom walls and wafer surfaces; prescribed vertical bottom inlet slots',
  'equal_Q_constraint':'sum of vertical face fluxes through the five inter-wafer gaps at y=0.20',
  'Q_target':float(prod.Q_target.iloc[0]),
  'nu':7.5e-4,
  'wafer_count':6,
  'wafer_thickness':0.015,
  'wafer_y0':0.12,
  'wafer_y1':0.25,
  'horizontal_cut_y':0.20,
  'production_gap_cells':20,
  'no_fitting':True,
  'plot_note':'streamlines are visualization interpolants only; all reported cut data and integrals are native staggered-grid values',
  'production_cases':prod.to_dict(orient='records'),
  'mesh_study':mesh.to_dict(orient='records'),
  'timestep_study':time.to_dict(orient='records'),
  'mass_conservation_max_abs_error_pct':mass.groupby('case').err_pct.apply(lambda x: float(x.abs().max())).to_dict(),
}
with open(os.path.join(OUT,'run_metadata.json'),'w') as f: json.dump(meta,f,indent=2)

# common max speed for field color scale
speeds={}; arrays={}
for case in CASES:
    z=np.load(os.path.join(OUT,f'{case.lower()}_production.npz'))
    u=z['u'];v=z['v'];solid=z['solid']
    uc=.5*(u[:,:-1]+u[:,1:]); vc=.5*(v[:-1,:]+v[1:,:])
    sp=np.hypot(uc,vc);sp=np.where(solid,np.nan,sp)
    arrays[case]=(z,uc,vc,sp)
    speeds[case]=float(np.nanmax(sp))
Vmax=max(speeds.values())

for case in CASES:
    z,uc,vc,sp=arrays[case]
    xf=z['xf'];yf=z['yf'];xc=z['xc'];yc=z['yc'];solid=z['solid'];left=z['left'];right=z['right'];pitch=float(z['pitch'])
    fig,ax=plt.subplots(figsize=(12,5.4))
    pcm=ax.pcolormesh(xf,yf,sp,shading='flat',cmap='viridis',vmin=0,vmax=Vmax)
    # regular-grid visualization for streamlines only
    xr=np.linspace(0,1,301); yr=np.linspace(0,.45,136); Xr,Yr=np.meshgrid(xr,yr)
    Fu=RegularGridInterpolator((yc,xc),uc,bounds_error=False,fill_value=0.0)
    Fv=RegularGridInterpolator((yc,xc),vc,bounds_error=False,fill_value=0.0)
    pts=np.column_stack([Yr.ravel(),Xr.ravel()]); ur=Fu(pts).reshape(Yr.shape);vr=Fv(pts).reshape(Yr.shape)
    mask=np.zeros_like(ur,dtype=bool)
    for xl,xr0 in zip(left,right): mask |= (Xr>=xl)&(Xr<=xr0)&(Yr>=.12)&(Yr<=.25)
    ur=np.ma.array(ur,mask=mask);vr=np.ma.array(vr,mask=mask)
    ax.streamplot(np.linspace(0,1,301),np.linspace(0,.45,136),ur,vr,density=1.4,color='white',linewidth=.75,arrowsize=.8)
    for xl,xr0 in zip(left,right):
        ax.axvspan(xl,xr0,ymin=.12/.45,ymax=.25/.45,facecolor='white',edgecolor='black',linewidth=1.0,zorder=5)
    centers=.5*(left+right); inlet_centers=.5*(centers[:-1]+centers[1:])
    ax.scatter(inlet_centers,np.zeros_like(inlet_centers),marker='^',s=42,color='red',zorder=6,label='inlets')
    ax.axhline(.20,ls='--',lw=1.2,color='white',alpha=.9,label='horizontal cut y=0.20')
    ax.set(xlim=(0,1),ylim=(0,.45),xlabel='x',ylabel='y',title=f'{case} pitch P={pitch:.3f}: MAC Navier-Stokes |u| and streamlines')
    cb=fig.colorbar(pcm,ax=ax);cb.set_label('|u|')
    ax.legend(loc='upper right');fig.tight_layout();fig.savefig(os.path.join(FIG,f'field_{case.lower()}_MAC_equalQ.png'),dpi=180,bbox_inches='tight');plt.close(fig)

# Horizontal-cut panels, raw gap values with exact no-slip wall boundary markers
fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=False)
for ax,case in zip(axes,CASES):
    z=np.load(os.path.join(OUT,f'{case.lower()}_production.npz'))
    xc=z['xc'];dx=z['dx'];v=z['v'];yf=z['yf'];left=z['left'];right=z['right'];pitch=float(z['pitch'])
    j=int(np.argmin(np.abs(yf-.20)))
    for xl,xr0 in zip(left,right): ax.axvspan(xl,xr0,color='0.82',alpha=.75,zorder=0)
    for k in range(5):
        m=(xc>right[k])&(xc<left[k+1])
        xx=np.r_[right[k],xc[m],left[k+1]]; yy=np.r_[0.0,v[j,m],0.0]
        ax.plot(xx,yy,lw=1.8)
    ax.axhline(0,color='black',lw=.7)
    ax.set_ylabel('v')
    ax.set_title(f'{case}: P={pitch:.3f}; gray = wafer solids; endpoints = imposed no-slip BC')
    ax.grid(True,alpha=.25)
axes[-1].set_xlabel('x at y=0.20')
fig.suptitle('Equal-Q horizontal cut - native staggered-grid values in the five wafer gaps',y=.995)
fig.tight_layout(rect=(0,0,1,.985));fig.savefig(os.path.join(FIG,'horizontal_cut_v_shaded_panels_MAC.png'),dpi=180,bbox_inches='tight');plt.close(fig)

# central-gap normalized coordinate comparison
fig,ax=plt.subplots(figsize=(8,5.5))
for case in CASES:
    z=np.load(os.path.join(OUT,f'{case.lower()}_production.npz'))
    xc=z['xc'];v=z['v'];yf=z['yf'];left=z['left'];right=z['right'];j=int(np.argmin(abs(yf-.20)))
    a=right[2];b=left[3];m=(xc>a)&(xc<b);eta=(xc[m]-a)/(b-a)
    ax.plot(np.r_[0,eta,1],np.r_[0,v[j,m],0],marker='o',ms=2.6,lw=1.5,label=case)
ax.set(xlabel='normalized central-gap coordinate, eta',ylabel='v(x,y=0.20)',title='Central-gap horizontal velocity profile at equal rack flow rate')
ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();fig.savefig(os.path.join(FIG,'central_gap_profiles_equalQ_MAC.png'),dpi=180,bbox_inches='tight');plt.close(fig)

# Mean/peak velocity and hydraulic resistance plots
fig,ax=plt.subplots(figsize=(7.5,5.2))
x=np.arange(len(prod));w=.34
ax.bar(x-w/2,prod['central_mean_v'],w,label='central-gap mean v')
ax.bar(x+w/2,prod['central_peak_v'],w,label='central-gap peak v')
ax.set_xticks(x,prod['case']);ax.set_ylabel('velocity');ax.set_title('Equal-Q gap velocities');ax.grid(axis='y',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(os.path.join(FIG,'gap_velocity_summary.png'),dpi=180,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.5,5.2));ax.bar(prod['case'],prod['R_h']);ax.set_ylabel('R_h = Delta p / Q_rack');ax.set_title('Hydraulic resistance at equal rack flow');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(os.path.join(FIG,'hydraulic_resistance_equalQ.png'),dpi=180,bbox_inches='tight');plt.close(fig)

# validation plots
fig,ax=plt.subplots(figsize=(7.5,5.2));ax.plot(mesh['n_gap_cells'],mesh['central_peak_v'],marker='o',label='central peak v');ax.set_xlabel('cells across smallest gap');ax.set_ylabel('central peak v');ax.set_title('Spatial-grid convergence - small-pitch case');ax.grid(True,alpha=.3);fig.tight_layout();fig.savefig(os.path.join(FIG,'mesh_convergence_peak_v.png'),dpi=180,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.5,5.2));ax.plot(mesh['n_gap_cells'],mesh['delta_p_rack'],marker='o');ax.set_xlabel('cells across smallest gap');ax.set_ylabel('Delta p_rack');ax.set_title('Spatial-grid convergence - pressure drop');ax.grid(True,alpha=.3);fig.tight_layout();fig.savefig(os.path.join(FIG,'mesh_convergence_dp.png'),dpi=180,bbox_inches='tight');plt.close(fig)

print('Wrote metadata and figures to',OUT)
