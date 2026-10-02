import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as patches, zipfile
BASE='/mnt/data/wafer_pitch_equalQ'; W=1.0;H=0.45;WAFER_THICKNESS=0.015;WAFER_Y0=0.12;WAFER_Y1=0.25;Y_CUT=0.20
labels=['Small','Medium','Large']; names=['small','medium','large']
D={}
for L,n in zip(labels,names):
    z=np.load(os.path.join(BASE,f'field_{n}_equalQ.npz'))
    D[L]={k:z[k] for k in z.files}
Qtarget=float(D['Medium']['rack_flux'])
x=D['Medium']['x']; y=D['Medium']['y']; iy=int(np.argmin(np.abs(y-Y_CUT))); dx=float(x[1]-x[0])
cut=pd.DataFrame({'x':x})
rows=[]
for L,n in zip(labels,names):
    d=D[L]; p=float(d['pitch']); u=d['u'];v=d['v'];spd=d['speed'];solid=d['solid'].astype(bool);g=d['gaps'].astype(bool); wc=d['wafer_centers']
    s=spd[iy,:].copy(); vv=v[iy,:].copy(); uu=u[iy,:].copy(); s[solid[iy,:]]=np.nan;vv[solid[iy,:]]=np.nan;uu[solid[iy,:]]=np.nan
    cut[f'{n}_speed']=s;cut[f'{n}_v']=vv;cut[f'{n}_u']=uu
    a=float(wc[2])+WAFER_THICKNESS/2;b=float(wc[3])-WAFER_THICKNESS/2
    cg=(x>a)&(x<b)&(~solid[iy,:])
    q=float(d['rack_flux']); numerical_gap_width=float(np.sum(g)*dx); analytic_gap_width=5*(p-WAFER_THICKNESS)
    # Pressure metric just below wafer bottoms, inside span between outer wafer centers.
    jp=int(np.argmin(np.abs(y-0.10))); footprint=(x>=wc[0])&(x<=wc[-1]); pbelow=float(np.nanmean(d['p'][jp,footprint]))
    rows.append({
        'case':L,'pitch':p,'clear_gap':p-WAFER_THICKNESS,
        'analytic_total_5_gap_width':analytic_gap_width,'numerical_gap_width_used':numerical_gap_width,
        'inlet_speed_required':float(d['inlet_speed']),'target_rack_Q':Qtarget,'achieved_rack_Q':q,
        'Q_error_percent':100*(q-Qtarget)/Qtarget,
        'mean_vertical_v_from_flux':q/numerical_gap_width,
        'mean_speed_magnitude_in_gaps':float(np.nanmean(s[g])),
        'central_gap_mean_speed':float(np.nanmean(s[cg])),
        'central_gap_peak_speed':float(np.nanmax(s[cg])),
        'full_cut_peak_speed':float(np.nanmax(s)),
        'mean_pressure_y_0p10_in_rack_footprint':pbelow,
    })
summary=pd.DataFrame(rows)
cut.to_csv(os.path.join(BASE,'horizontal_cut_equalQ_y_0p20.csv'),index=False)
summary.to_csv(os.path.join(BASE,'equalQ_summary.csv'),index=False)

# Speed plot
fig,ax=plt.subplots(figsize=(10,5.4))
for L,n in zip(labels,names):
    d=D[L];ax.plot(x,cut[f'{n}_speed'],lw=1.8,label=f'{L}: P={float(d["pitch"]):.3f}')
ax.set_xlim(0,W);ax.set_xlabel('x');ax.set_ylabel(r'$|\mathbf{u}|$ at $y=0.20$')
ax.set_title(r'Horizontal cut — same $Q_{rack}$ through five wafer gaps (no fitting)')
ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(os.path.join(BASE,'horizontal_cut_equalQ_speed.png'),dpi=200,bbox_inches='tight');plt.close(fig)

# Vertical velocity plot
fig,ax=plt.subplots(figsize=(10,5.4))
for L,n in zip(labels,names):
    d=D[L];ax.plot(x,cut[f'{n}_v'],lw=1.8,label=f'{L}: P={float(d["pitch"]):.3f}')
ax.set_xlim(0,W);ax.set_xlabel('x');ax.set_ylabel(r'$v(x,y=0.20)$')
ax.set_title(r'Normal velocity on horizontal cut — equal $Q_{rack}$')
ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(os.path.join(BASE,'horizontal_cut_equalQ_vertical_velocity.png'),dpi=200,bbox_inches='tight');plt.close(fig)

# Common field color limit. Include inlet but avoid single-pixel max dominance.
vmax=max(float(np.nanpercentile(D[L]['speed'],99.7)) for L in labels)
for L,n in zip(labels,names):
    d=D[L];u=d['u'];v=d['v'];spd=d['speed'];solid=d['solid'].astype(bool);wc=d['wafer_centers'];ic=d['inlet_centers'];p=float(d['pitch']);qin=float(d['rack_flux'])
    fig,ax=plt.subplots(figsize=(12,5.2));pcm=ax.pcolormesh(x,y,spd,shading='auto',cmap='viridis',vmin=0,vmax=vmax)
    um=np.ma.array(u,mask=solid);vm=np.ma.array(v,mask=solid)
    ax.streamplot(x,y,um,vm,density=1.5,linewidth=.75,arrowsize=.85,color='white')
    for xc in wc:
        ax.add_patch(patches.Rectangle((float(xc)-WAFER_THICKNESS/2,WAFER_Y0),WAFER_THICKNESS,WAFER_Y1-WAFER_Y0,facecolor='white',edgecolor='black',linewidth=.9,zorder=5))
    ax.scatter(ic,np.zeros_like(ic),marker='^',s=36,color='red',zorder=6,label='Bottom inlets');ax.axhline(Y_CUT,ls='--',lw=1.3,color='white',alpha=.95,label='Horizontal cut y=0.20')
    ax.set_xlim(0,W);ax.set_ylim(0,H);ax.set_xlabel('x');ax.set_ylabel('y');ax.set_title(f'{L} pitch P={p:.3f}: Qrack={qin:.6f}, Uin={float(d["inlet_speed"]):.4f}')
    cb=fig.colorbar(pcm,ax=ax);cb.set_label(r'$|\mathbf{u}|$');ax.legend(loc='upper right');fig.tight_layout();fig.savefig(os.path.join(BASE,f'field_{n}_equalQ.png'),dpi=200,bbox_inches='tight');plt.close(fig)

# Reproducible source: copy kernel plus explicit equal-Q procedure description and values.
# NOTE: this script and the inlet speeds below belong to the EARLIER finite-difference
# model (earlier_versions/original_fd/equal_rack_flow_navier_stokes.py), NOT to the
# audited MAC solver.  The legacy FD model defined Q_target as its medium-pitch case at
# U_in=0.30, giving legacy inlet speeds 0.5605 / 0.3000 / 0.1833.  The audited MAC solver
# reuses the same numeric Q_target (0.0055463) but recalibrates the inlet speeds, yielding
# U_in ~= 0.6937 / 0.3887 / 0.2374 (Small/Medium/Large).  Do not read the values below as
# the MAC production results; see audited_final/ + production_summary.csv for those.
with open('/mnt/data/equalQ_temp.py','r',encoding='utf-8') as f: kernel=f.read()
header='''#!/usr/bin/env python3\n"""\nEqual-rack-flow wafer pitch Navier-Stokes model (LEGACY finite-difference model).\n\nNO FITTING. NO REGRESSION. NO SMOOTHING.\n\nGoverning PDE:\n    du/dt + (u.grad)u = -grad(p) + nu*laplacian(u)\n    div(u) = 0\n\nRack-flow constraint at y=0.20:\n    Q_rack = integral over the five INTERIOR clear gaps of v(x,y) dx = constant.\n\nThe medium-pitch reference at U_in=0.30 defines Q_target. The inlet velocity\nfor small and large pitch is found by a scalar root solve on Q_rack-Q_target.\nThat operation enforces a physical integral boundary/control constraint; it is\nnot a fit to the velocity profile or to measured data. The plotted cut values\nare raw finite-difference grid values.\n\nLEGACY FD converged inlet speeds (NOT the audited MAC results):\n    Small  P=0.040 : U_in=0.5605\n    Medium P=0.060 : U_in=0.3000\n    Large  P=0.080 : U_in=0.1833\nThe audited MAC solver (audited_final/) gives U_in ~= 0.6937 / 0.3887 / 0.2374.\n"""\n\n'''
with open(os.path.join(BASE,'equal_rack_flow_navier_stokes.py'),'w',encoding='utf-8') as f:f.write(header+kernel)
with open(os.path.join(BASE,'README.txt'),'w',encoding='utf-8') as f:
    f.write(f'Equal rack-flow Navier-Stokes study\\nNO FITTING / NO SMOOTHING\\nHorizontal cut y=0.20\\nQ_target={Qtarget:.12g} normalized 2-D flow per unit depth.\\n')
zip_path='/mnt/data/wafer_pitch_equalQ_results.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(BASE)): z.write(os.path.join(BASE,fn),arcname=fn)
print(summary.to_string(index=False))
print('zip',zip_path)
