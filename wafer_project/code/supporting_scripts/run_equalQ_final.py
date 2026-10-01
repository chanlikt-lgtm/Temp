import os, importlib.util, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as patches, zipfile, shutil
BASE='/mnt/data/wafer_pitch_equalQ'
os.makedirs(BASE,exist_ok=True)
# Load the direct Navier-Stokes numerical kernel prepared in this session.
spec=importlib.util.spec_from_file_location('eq','/mnt/data/equalQ_temp.py')
eq=importlib.util.module_from_spec(spec);spec.loader.exec_module(eq)
eq.DT=0.0025
pitches={'Small':0.04,'Medium':0.06,'Large':0.08}
# Flux-control inlet speeds obtained by solving the integral constraint Q_rack=Q_target.
uin={'Small':0.5605,'Medium':0.3000,'Large':0.1833}
results={}
for label,p in pitches.items():
    c=eq.setup(p)
    r=eq.solve(c,uin[label],max_steps=5000)
    u,v,pres,spd,qr,res,steps=r
    results[label]=(c,u,v,pres,spd,qr,res,steps)
    np.savez_compressed(os.path.join(BASE,f'field_{label.lower()}_equalQ.npz'),
        x=c['x'],y=c['y'],u=u,v=v,p=pres,speed=spd,solid=c['solid'],gaps=c['gaps'],
        wafer_centers=c['wc'],inlet_centers=c['ic'],pitch=p,inlet_speed=uin[label],rack_flux=qr)

Qtarget=results['Medium'][5]
rows=[]
cut=pd.DataFrame({'x':results['Medium'][0]['x']})
for label in ['Small','Medium','Large']:
    c,u,v,pres,spd,qr,res,steps=results[label]
    x=c['x']; y=c['y']; iy=c['iy']; g=c['gaps']; dx=c['dx']
    s=spd[iy,:].copy(); vv=v[iy,:].copy(); uu=u[iy,:].copy()
    s[c['solid'][iy,:]]=np.nan; vv[c['solid'][iy,:]]=np.nan; uu[c['solid'][iy,:]]=np.nan
    cut[f'{label.lower()}_speed']=s
    cut[f'{label.lower()}_v']=vv
    cut[f'{label.lower()}_u']=uu
    # Central interior gap between wafers 3 and 4
    a=c['wc'][2]+eq.WAFER_THICKNESS/2
    b=c['wc'][3]-eq.WAFER_THICKNESS/2
    cg=(x>a)&(x<b)&(~c['solid'][iy,:])
    rack_min=c['wc'][0]+eq.WAFER_THICKNESS/2
    rack_max=c['wc'][-1]-eq.WAFER_THICKNESS/2
    rackmask=(x>=rack_min)&(x<=rack_max)&g
    total_gap_width=5*(pitches[label]-eq.WAFER_THICKNESS)
    rows.append({
        'case':label,
        'pitch':pitches[label],
        'clear_gap':pitches[label]-eq.WAFER_THICKNESS,
        'total_5_gap_width':total_gap_width,
        'inlet_speed_required':uin[label],
        'target_rack_Q':Qtarget,
        'achieved_rack_Q':qr,
        'Q_error_percent':100*(qr-Qtarget)/Qtarget,
        'continuity_Q_over_gap_width':Qtarget/total_gap_width,
        'numerical_mean_vertical_v_in_gaps':float(np.nanmean(vv[g])),
        'numerical_mean_speed_in_gaps':float(np.nanmean(s[g])),
        'central_gap_mean_speed':float(np.nanmean(s[cg])),
        'central_gap_peak_speed':float(np.nanmax(s[cg])),
        'full_cut_peak_speed':float(np.nanmax(s)),
        'residual_indicator':res,
        'steps':steps,
    })
summary=pd.DataFrame(rows)
cut.to_csv(os.path.join(BASE,'horizontal_cut_equalQ_y_0p20.csv'),index=False)
summary.to_csv(os.path.join(BASE,'equalQ_summary.csv'),index=False)

# Overlay speed magnitude horizontal cut
fig,ax=plt.subplots(figsize=(10,5.3))
for label in ['Small','Medium','Large']:
    ax.plot(cut['x'],cut[f'{label.lower()}_speed'],lw=1.8,label=f'{label}: P={pitches[label]:.3f}, Uin={uin[label]:.4f}')
ax.set_xlim(0,eq.W); ax.set_xlabel('x'); ax.set_ylabel(r'$|\mathbf{u}|$ at $y=0.20$')
ax.set_title(r'Horizontal cut with equal rack flow $Q$ — direct Navier–Stokes, no fitting')
ax.grid(True,alpha=.25); ax.legend(ncol=1); fig.tight_layout()
fig.savefig(os.path.join(BASE,'horizontal_cut_equalQ_speed.png'),dpi=200,bbox_inches='tight'); plt.close(fig)

# Overlay normal/vertical component used in rack-flux integral
fig,ax=plt.subplots(figsize=(10,5.3))
for label in ['Small','Medium','Large']:
    ax.plot(cut['x'],cut[f'{label.lower()}_v'],lw=1.8,label=f'{label}: P={pitches[label]:.3f}')
ax.set_xlim(0,eq.W); ax.set_xlabel('x'); ax.set_ylabel(r'$v(x,y=0.20)$')
ax.set_title(r'Horizontal cut: vertical velocity component, equal $Q_{rack}$')
ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(BASE,'horizontal_cut_equalQ_vertical_velocity.png'),dpi=200,bbox_inches='tight'); plt.close(fig)

# Field plots with common speed range.
vmax=max(np.nanpercentile(results[k][4],99.7) for k in results)
for label in ['Small','Medium','Large']:
    c,u,v,pres,spd,qr,res,steps=results[label]
    fig,ax=plt.subplots(figsize=(12,5.2))
    pcm=ax.pcolormesh(c['x'],c['y'],spd,shading='auto',cmap='viridis',vmin=0,vmax=vmax)
    um=np.ma.array(u,mask=c['solid']); vm=np.ma.array(v,mask=c['solid'])
    ax.streamplot(c['x'],c['y'],um,vm,density=1.5,linewidth=.75,arrowsize=.85,color='white')
    for xc in c['wc']:
        ax.add_patch(patches.Rectangle((xc-eq.WAFER_THICKNESS/2,eq.WAFER_Y0),eq.WAFER_THICKNESS,eq.WAFER_Y1-eq.WAFER_Y0,
                                       facecolor='white',edgecolor='black',linewidth=.9,zorder=5))
    ax.scatter(c['ic'],np.zeros_like(c['ic']),marker='^',s=36,color='red',zorder=6,label='Bottom inlets')
    ax.axhline(eq.Y_CUT,ls='--',lw=1.3,color='white',alpha=.95,label='Horizontal cut y=0.20')
    ax.set_xlim(0,eq.W);ax.set_ylim(0,eq.H);ax.set_xlabel('x');ax.set_ylabel('y')
    ax.set_title(f'{label} pitch P={pitches[label]:.3f}: equal rack Q={Qtarget:.6f}, Uin={uin[label]:.4f}')
    cb=fig.colorbar(pcm,ax=ax);cb.set_label(r'$|\mathbf{u}|$')
    ax.legend(loc='upper right');fig.tight_layout()
    fig.savefig(os.path.join(BASE,f'field_{label.lower()}_equalQ.png'),dpi=200,bbox_inches='tight');plt.close(fig)

# Write standalone reproducible solver by combining explanatory header with the numerical kernel source.
with open('/mnt/data/equalQ_temp.py','r',encoding='utf-8') as f: kernel=f.read()
source='''#!/usr/bin/env python3\n"""\nEqual-rack-flow wafer pitch comparison.\n\nNO FITTING / NO SMOOTHING.\nThe code directly solves the 2-D incompressible Navier-Stokes equations with a\nfinite-difference pressure-projection method.  The constraint is\n\n    Q_rack = integral_over_the_five_interior_gaps v(x,y_cut) dx = constant.\n\nThe medium-pitch case at U_in=0.30 defines the normalized target Q.  For the\nother pitch cases the scalar inlet boundary velocity is changed until the\nintegral flow constraint is satisfied.  This is a boundary-condition/root\nconstraint, not a regression or fit to velocity data.  Horizontal-cut curves\nare raw grid values.\n"""\n\n'''+kernel+'''\n\n# Final equal-Q values used for the delivered comparison:\n# Small:  U_in=0.5605\n# Medium: U_in=0.3000\n# Large:  U_in=0.1833\n'''
with open(os.path.join(BASE,'equal_rack_flow_navier_stokes.py'),'w',encoding='utf-8') as f:f.write(source)

with open(os.path.join(BASE,'README.txt'),'w',encoding='utf-8') as f:
    f.write(f'''Equal-Q wafer pitch Navier-Stokes comparison\n\nNo fitting. No regression. No smoothing.\nHorizontal cut: y=0.20\nRack flow definition: Q_rack = integral of vertical velocity across the five interior wafer gaps.\nTarget Q_rack = {Qtarget:.10f} (normalized 2-D flow per unit depth), taken from the medium-pitch reference at Uin=0.30.\nThe small and large cases use flux-constrained inlet velocities so the same Q crosses the interior wafer gaps.\n\nSee equalQ_summary.csv and horizontal_cut_equalQ_y_0p20.csv.\n''')

zip_path='/mnt/data/wafer_pitch_equalQ_results.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(BASE)):
        z.write(os.path.join(BASE,fn),arcname=fn)

print(summary.to_string(index=False))
print('Q target',Qtarget)
print('OUT',BASE)
print('ZIP',zip_path)
