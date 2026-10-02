import os, json, shutil, zipfile, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SRC='/mnt/data/wafer_mac_equalQ_v2'
OUT='/mnt/data/wafer_mac_equalQ_final'
FIG=os.path.join(OUT,'figures')
os.makedirs(FIG,exist_ok=True)

# Copy source code and raw numerical artifacts
for fn in ['small_production.npz','medium_production.npz','large_production.npz',
           'small_summary.csv','medium_summary.csv','large_summary.csv',
           'small_mass_flux.csv','medium_mass_flux.csv','large_mass_flux.csv',
           'small_root_history.csv','medium_root_history.csv','large_root_history.csv',
           'mesh_small_n10_summary.csv','mesh_small_n15_summary.csv','small_timestep_study.csv','timestep_study_small.csv']:
    p=os.path.join(SRC,fn)
    if os.path.exists(p): shutil.copy2(p,os.path.join(OUT,fn))
shutil.copy2('/mnt/data/mac_cfd_adaptive.py',os.path.join(OUT,'mac_cfd_adaptive.py'))
shutil.copy2('/mnt/data/run_production_equalq.py',os.path.join(OUT,'equalQ_root_driver.py'))

# Combined production summary
summ=[]
for lab in ['small','medium','large']:
    d=pd.read_csv(os.path.join(SRC,f'{lab}_summary.csv')).iloc[0]
    summ.append(d)
prod=pd.DataFrame(summ)
prod.to_csv(os.path.join(OUT,'production_summary.csv'),index=False)

# Combined mass flux table
mass=pd.concat([pd.read_csv(os.path.join(SRC,f'{lab}_mass_flux.csv')) for lab in ['small','medium','large']],ignore_index=True)
mass.to_csv(os.path.join(OUT,'mass_conservation_all_cases.csv'),index=False)

# Grid study
mesh10=pd.read_csv(os.path.join(SRC,'mesh_small_n10_summary.csv')).iloc[0]
mesh15=pd.read_csv(os.path.join(SRC,'mesh_small_n15_summary.csv')).iloc[0]
mesh20=prod[prod['case']=='Small'].iloc[0]
mesh=pd.DataFrame([
 {'n_gap_cells':int(mesh10.n_gap_cells),'Nx':int(mesh10.Nx),'Ny':int(mesh10.Ny),'dx_min':mesh10.dx_min,'Q_error_pct':mesh10.Q_error_pct,'central_peak_v':mesh10.central_peak_v,'delta_p_rack':mesh10.delta_p_rack,'R_h':mesh10.R_h},
 {'n_gap_cells':int(mesh15.n_gap_cells),'Nx':int(mesh15.Nx),'Ny':int(mesh15.Ny),'dx_min':mesh15.dx_min,'Q_error_pct':mesh15.Q_error_pct,'central_peak_v':mesh15.central_peak_v,'delta_p_rack':mesh15.delta_p_rack,'R_h':mesh15.R_h},
 {'n_gap_cells':20,'Nx':int(mesh20.Nx),'Ny':int(mesh20.Ny),'dx_min':mesh20.dx_min,'Q_error_pct':mesh20.Q_error_pct,'central_peak_v':mesh20.central_peak_v,'delta_p_rack':mesh20.delta_p_rack,'R_h':mesh20.R_h},
])
for col in ['central_peak_v','delta_p_rack','R_h']:
    fine=mesh.iloc[-1][col]
    mesh[f'{col}_change_vs_fine_pct']=100*(mesh[col]/fine-1)
mesh.to_csv(os.path.join(OUT,'grid_independence_small_pitch.csv'),index=False)

# Timestep study
if os.path.exists(os.path.join(SRC,'timestep_study_small.csv')):
    ts=pd.read_csv(os.path.join(SRC,'timestep_study_small.csv'))
else:
    ts=pd.read_csv(os.path.join(SRC,'small_timestep_study.csv'))
ts.to_csv(os.path.join(OUT,'timestep_independence_small_pitch.csv'),index=False)

# plots from staggered fields
case_colors={'Small':'tab:blue','Medium':'tab:orange','Large':'tab:green'}
for label in ['Small','Medium','Large']:
    z=np.load(os.path.join(SRC,f'{label.lower()}_production.npz'))
    u=z['u']; v=z['v']; p=z['p']; xc=z['xc']; yc=z['yc']; xf=z['xf']; yf=z['yf']; solid=z['solid'].astype(bool)
    left=z['left']; right=z['right']; dx=z['dx']; pitch=float(z['pitch']); U=float(z['U_in']); Q=float(z['Q_target'])
    uc=.5*(u[:,:-1]+u[:,1:]); vc=.5*(v[:-1,:]+v[1:,:]); sp=np.sqrt(uc**2+vc**2); sp=np.where(solid,np.nan,sp)
    fig,ax=plt.subplots(figsize=(11.5,5.1))
    pcm=ax.pcolormesh(xf,yf,sp,shading='flat',cmap='viridis')
    skipx=max(1,len(xc)//35); skipy=max(1,len(yc)//18)
    X,Y=np.meshgrid(xc[::skipx],yc[::skipy]); UQ=uc[::skipy,::skipx]; VQ=vc[::skipy,::skipx]; M=solid[::skipy,::skipx]
    UQ=np.ma.array(UQ,mask=M); VQ=np.ma.array(VQ,mask=M)
    ax.quiver(X,Y,UQ,VQ,color='white',scale=1.8,width=.0022,headwidth=3.2)
    for xl,xr in zip(left,right): ax.add_patch(Rectangle((xl,.12),xr-xl,.13,facecolor='white',edgecolor='black',lw=.8,zorder=5))
    ax.axhline(.20,ls='--',lw=1.2,color='white')
    ax.set(xlim=(0,1),ylim=(0,.45),xlabel='x',ylabel='y',title=f'{label} pitch P={pitch:.3f}: MAC Navier-Stokes, equal rack flow Q={Q:.7f}, Uin={U:.5f}')
    cb=fig.colorbar(pcm,ax=ax); cb.set_label(r'$|\mathbf{u}|$')
    fig.tight_layout(); fig.savefig(os.path.join(FIG,f'field_{label.lower()}_mac.png'),dpi=180,bbox_inches='tight'); plt.close(fig)

# Horizontal cuts, 3 panels with own wafer shading
fig,axes=plt.subplots(3,1,figsize=(10.5,10.2),sharex=True)
for ax,label in zip(axes,['Small','Medium','Large']):
    z=np.load(os.path.join(SRC,f'{label.lower()}_production.npz'))
    v=z['v']; xc=z['xc']; yf=z['yf']; dx=z['dx']; left=z['left']; right=z['right']; gap=z['gapmask'].astype(bool)
    j=int(np.argmin(np.abs(yf-.20))); vv=v[j,:]
    for xl,xr in zip(left,right): ax.axvspan(xl,xr,color='0.82',alpha=.8)
    # plot each of five interior fluid gap segments only; append imposed no-slip zero at faces
    for k in range(5):
        m=(xc>right[k])&(xc<left[k+1])
        xx=np.r_[right[k],xc[m],left[k+1]]; yy=np.r_[0.0,vv[m],0.0]
        ax.plot(xx,yy,lw=1.9,color=case_colors[label])
    ax.axhline(0,color='0.25',lw=.7)
    ax.set_ylabel(r'$v(x,0.20)$')
    ax.set_title(f'{label} pitch - gray bands are wafers; endpoint zeros are imposed no-slip BC')
    ax.grid(alpha=.25)
axes[-1].set_xlabel('x'); axes[-1].set_xlim(.27,.73)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'horizontal_cut_gap_profiles.png'),dpi=180,bbox_inches='tight'); plt.close(fig)

# Resistance and pressure drop comparison
fig,ax=plt.subplots(figsize=(8.2,4.7))
x=np.arange(3); vals=prod['R_h'].to_numpy(float)
ax.bar(x,vals); ax.set_xticks(x,prod['case']); ax.set_ylabel(r'$R_h=\Delta p/Q_{rack}$'); ax.set_title('Hydraulic resistance at equal rack flow'); ax.grid(axis='y',alpha=.25)
for i,vv in enumerate(vals): ax.text(i,vv*1.01,f'{vv:.3f}',ha='center',va='bottom',fontsize=9)
fig.tight_layout();fig.savefig(os.path.join(FIG,'hydraulic_resistance.png'),dpi=180,bbox_inches='tight');plt.close(fig)

# Mass conservation chart
fig,ax=plt.subplots(figsize=(8.2,4.7))
for label in ['Small','Medium','Large']:
    d=mass[mass['case']==label]
    ax.plot(d['y_face'],d['err_pct'],marker='o',label=label)
ax.set_xlabel('horizontal face y');ax.set_ylabel('section flow error vs inlet (%)');ax.set_title('Discrete mass conservation');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(os.path.join(FIG,'mass_conservation.png'),dpi=180,bbox_inches='tight');plt.close(fig)

# Grid and dt study
fig,ax=plt.subplots(figsize=(8.2,4.8))
ax.plot(mesh['n_gap_cells'],mesh['central_peak_v_change_vs_fine_pct'],marker='o',label='central peak v')
ax.plot(mesh['n_gap_cells'],mesh['delta_p_rack_change_vs_fine_pct'],marker='s',label='rack pressure drop')
ax.axhline(0,color='0.2',lw=.8);ax.set_xlabel('cells across smallest clear gap');ax.set_ylabel('change relative to 20-cell grid (%)');ax.set_title('Small-pitch grid sensitivity');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(os.path.join(FIG,'grid_independence.png'),dpi=180,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(8.2,4.8))
base=ts.iloc[0]
labels=['Q rack','central peak v','pressure drop']
chg=[100*(ts.iloc[1]['Q_rack']/base['Q_rack']-1),100*(ts.iloc[1]['central_peak_v']/base['central_peak_v']-1),100*(ts.iloc[1]['delta_p_rack']/base['delta_p_rack']-1)]
ax.bar(labels,chg);ax.axhline(0,color='0.2',lw=.8);ax.set_ylabel('change when dt is halved (%)');ax.set_title('Small-pitch timestep sensitivity');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(os.path.join(FIG,'timestep_independence.png'),dpi=180,bbox_inches='tight');plt.close(fig)

# Metadata
meta={
 'method':'2-D incompressible Navier-Stokes, geometry-aligned MAC staggered-grid projection',
 'no_fitting':True,'Q_target':0.0055463,'nu':7.5e-4,
 'production_cells_across_smallest_gap':20,
 'pressure_reference':'p=0 at open top outlet',
 'equal_Q_control':'bracketed scalar root solve on inlet velocity; each candidate evaluated with Navier-Stokes solve',
 'mass_conservation_note':'compatible face divergence and pressure-gradient operators; section-flow conservation is at roundoff level',
}
with open(os.path.join(OUT,'run_metadata.json'),'w') as f: json.dump(meta,f,indent=2)

# LaTeX report
small=prod[prod.case=='Small'].iloc[0]; med=prod[prod.case=='Medium'].iloc[0]; large=prod[prod.case=='Large'].iloc[0]
meshfine=mesh.iloc[-1]
tex=r'''\documentclass[11pt]{article}
\usepackage[margin=0.82in]{geometry}
\usepackage{amsmath,amssymb,booktabs,array,graphicx,float,xcolor,siunitx,hyperref}
\usepackage{microtype}
\hypersetup{colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black,pdftitle={Equal-Q Wafer Rack Navier-Stokes Study - Audited MAC Solver},pdfauthor={OpenAI ChatGPT}}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\newcommand{\Qr}{Q_{\mathrm{rack}}}
\newcommand{\uin}{U_{\mathrm{in}}}
\begin{document}
\begin{center}
{\LARGE\bfseries Equal-Flow Wafer-Rack Navier--Stokes Study}\\[4pt]
{\large Audited MAC staggered-grid solution -- no fitting}\\[8pt]
\end{center}
\section*{Nomenclature}
\begin{tabular}{@{}p{0.18\linewidth}p{0.73\linewidth}@{}}
\toprule
Symbol & Definition \\\midrule
$x,y$ & nondimensional horizontal and vertical coordinates \\
$u,v$ & nondimensional horizontal and vertical velocity components \\
$p$ & nondimensional kinematic pressure \\
$\nu$ & nondimensional kinematic viscosity, $7.5\times10^{-4}$ \\
$P$ & wafer center-to-center pitch \\
$t_w$ & wafer thickness, $0.015$ \\
$g=P-t_w$ & clear inter-wafer gap \\
$Q_{\mathrm{rack}}$ & vertical flow crossing the five interior wafer gaps at $y=0.20$ \\
$Q_0$ & common target rack flow, $0.0055463$ \\
$U_{\mathrm{in}}$ & prescribed velocity at bottom inlet slots; root-solved for each pitch \\
$\Delta p_{\mathrm{rack}}$ & mean pressure below rack minus mean pressure above rack \\
$R_h$ & hydraulic resistance, $\Delta p_{\mathrm{rack}}/Q_{\mathrm{rack}}$ \\
$N_g$ & number of finite-volume cells across the smallest clear gap \\
$\Delta t$ & nondimensional pseudo-time step used to reach steady state \\
$D,G$ & compatible discrete divergence and gradient operators on the MAC grid \\
\bottomrule
\end{tabular}

\vfill
\textbf{Normalization.} The tank is represented in nondimensional form with width $W=1$ and height $H=0.45$. If physical reference scales $L_0$ and $U_0$ are later specified, dimensional variables follow $x_d=L_0x$, $u_d=U_0u$, $t_d=(L_0/U_0)t$, and $p_d=\rho U_0^2p$.

\textbf{Scope.} This report replaces the earlier collocated finite-difference calculation with a conservative MAC staggered-grid formulation and addresses audit items 1--7: conservative projection, equal-$Q$ root control, local gap refinement, grid/timestep studies, section-flow conservation, pressure-drop resistance, and corrected reproducibility documentation.
\newpage

\section{Governing equations and conservative discretization}
The directly solved equations are
\begin{align}
\frac{\partial \mathbf{u}}{\partial t}+(\mathbf{u}\cdot\nabla)\mathbf{u}&=-\nabla p+\nu\nabla^2\mathbf{u},\\
\nabla\cdot\mathbf{u}&=0.
\end{align}
There is \textbf{no regression, curve fitting, spline fitting, or smoothing}. Pressure is stored at cell centers and $u,v$ on cell faces. The pressure projection uses compatible face-to-cell divergence $D$ and cell-to-face gradient $G$, so the discrete Poisson operator is $DG$. Consequently the corrected field satisfies the finite-volume continuity equation to numerical roundoff.

The geometry-aligned local grid places every wafer face on a grid face. The production grid uses $N_g=20$ cells across the smallest clear gap ($g=0.025$ for the small-pitch case). Donor-cell upwind advection and second-order nonuniform viscous differences are used. No artificial outlet-row flux rescaling is applied.

\subsection{Boundary and equal-flow conditions}
No-slip is imposed on tank side/bottom walls and wafer surfaces. Five bottom inlet slots prescribe vertical velocity $\uin$. The top is an open pressure outlet with $p=0$. The comparison constraint is
\begin{equation}
\Qr=\sum_{k=1}^{5}\int_{\mathrm{gap}\ k} v(x,0.20)\,dx=Q_0.
\end{equation}
For each pitch, $\uin$ is determined by a bracketed scalar root solve on $\Qr(\uin)-Q_0$. This is a boundary-condition control solve, not fitting of the velocity profile.

\begin{figure}[H]\centering
\includegraphics[width=.95\linewidth]{figures/mass_conservation.png}
\caption{Horizontal-section mass conservation. Errors are at roundoff level for all three production cases.}
\end{figure}
\newpage

\section{Equal-$Q$ production solutions}
\begin{table}[H]\centering\small
\caption{Production MAC results. The rack-flow match is reported rather than described as ``exact.''}
\begin{tabular}{lrrrrrr}
\toprule
Case & $P$ & $\uin$ & $\Qr$ & error (\%) & $\Delta p$ & $R_h$\\\midrule
'''
for r in [small,med,large]:
    tex += f"{r['case']} & {r['pitch']:.3f} & {r['U_in']:.5f} & {r['Q_rack']:.7f} & {r['Q_error_pct']:.4f} & {r['delta_p_rack']:.5f} & {r['R_h']:.3f} \\\\\n"
tex += r'''\bottomrule\end{tabular}\end{table}

At the common rack flow, the mean velocity across the five gaps is fixed by continuity, $\bar v_g=\Qr/(5g)$. Therefore the mean ordering is necessarily
\[
\bar v_{g,\mathrm{small}}>\bar v_{g,\mathrm{medium}}>\bar v_{g,\mathrm{large}}.
\]
The Navier--Stokes solution adds the local profile, peak velocity, recirculation, pressure field, and required hydraulic resistance. The pressure-based result is decisive: the small-pitch rack requires much larger $R_h$ than the medium and large racks.

\begin{figure}[H]\centering
\includegraphics[width=.75\linewidth]{figures/hydraulic_resistance.png}
\caption{Hydraulic resistance evaluated from the computed pressure drop at the same $\Qr$.}
\end{figure}
\newpage

\section{Velocity fields}
\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{figures/field_small_mac.png}\caption{Small pitch. Cell-centered speed magnitude and sampled MAC face-velocity vectors.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{figures/field_medium_mac.png}\caption{Medium pitch.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{figures/field_large_mac.png}\caption{Large pitch.}\end{figure}
\newpage

\section{Horizontal cut through the wafer gaps}
\begin{figure}[H]\centering\includegraphics[width=.96\linewidth]{figures/horizontal_cut_gap_profiles.png}
\caption{Vertical velocity at $y=0.20$. Curves are shown only in fluid gaps. Gray bands are solid wafers. The zero-valued endpoints at the exact wafer faces are imposed no-slip boundary markers; interior curve values are raw MAC CFD values. No line is drawn through the solid.}
\end{figure}
The apparent discontinuities in the earlier plot were therefore plotting breaks across solid wafer regions, not discontinuities of the fluid solution.
\newpage

\section{Grid and timestep verification}
The most restrictive geometry is the small-pitch case. Three geometry-aligned meshes were checked with 10, 15, and 20 cells across the smallest clear gap. The 20-cell mesh is the production grid.
\begin{table}[H]\centering\small
\caption{Small-pitch grid study.}
\begin{tabular}{rrrrrr}
\toprule
$N_g$ & $\Delta x_{\min}$ & $Q$ err. (\%) & peak $v$ & $\Delta p$ & $R_h$\\\midrule
'''
for _,r in mesh.iterrows():
    tex += f"{int(r.n_gap_cells)} & {r.dx_min:.6f} & {r.Q_error_pct:.4f} & {r.central_peak_v:.5f} & {r.delta_p_rack:.5f} & {r.R_h:.3f} \\\\\n"
tex += r'''\bottomrule\end{tabular}\end{table}
Between the 15- and 20-cell grids, the central peak velocity changes by less than 1\%, and the pressure drop also changes by less than 1\%. The 10- to 20-cell comparison is likewise approximately 1\% or better for these monitored quantities.

\begin{figure}[H]\centering\includegraphics[width=.76\linewidth]{figures/grid_independence.png}\caption{Small-pitch grid sensitivity relative to the 20-cell production mesh.}\end{figure}

The production small-pitch timestep was also halved while retaining the same spatial grid and inlet control value. The resulting changes in $\Qr$, central peak velocity, and pressure drop are all far below 0.1\%.
\begin{figure}[H]\centering\includegraphics[width=.76\linewidth]{figures/timestep_independence.png}\caption{Sensitivity to halving $\Delta t$.}\end{figure}
\newpage

\section{Reproducibility and audit closure}
\begin{enumerate}
\item \textbf{Conservative formulation:} MAC staggered-grid finite-volume projection with compatible $D$ and $G$ operators replaces the inconsistent collocated projection.
\item \textbf{Equal-$Q$ controller:} a bracketed secant/bisection-style root solve adjusts $\uin$; each candidate is evaluated with a Navier--Stokes solve. Reported production flow errors are small and explicitly tabulated.
\item \textbf{Wall resolution:} wafer faces are grid-aligned and the smallest production gap contains 20 cells.
\item \textbf{Grid/timestep verification:} 10/15/20-cell gap meshes and a halved timestep were evaluated; monitored production quantities show approximately 1\% or better spatial sensitivity and much smaller temporal sensitivity.
\item \textbf{Mass conservation:} horizontal-section flow is constant to roundoff in each production solution. No outlet-row rescaling is used.
\item \textbf{Hydraulic resistance:} resistance is now based on computed pressure drop, $R_h=\Delta p/\Qr$, rather than inferred from inlet velocity.
\item \textbf{Reproducibility record:} the delivered package includes the exact production fields, root histories, mass-flux tables, validation tables, solver, root driver, metadata, LaTeX source, and this PDF.
\end{enumerate}

\section*{Primary conclusion}
At a fixed rack flow $Q_0$, decreasing pitch increases the mean gap velocity by continuity and strongly increases the pressure drop required to drive that same flow. In the audited MAC solutions the resistance ordering is
\[
R_{h,\mathrm{small}}>R_{h,\mathrm{medium}}>R_{h,\mathrm{large}}.
\]
This conclusion is obtained from a direct Navier--Stokes calculation with no fitting.

\end{document}
'''
with open(os.path.join(OUT,'wafer_mac_equalQ_audited.tex'),'w') as f:f.write(tex)

# README
with open(os.path.join(OUT,'README.txt'),'w') as f:
    f.write('''Audited equal-Q wafer-rack CFD package\n\nMethod: 2-D incompressible Navier-Stokes, MAC staggered-grid projection.\nNO FITTING / NO SMOOTHING / NO REGRESSION.\nProduction: 20 cells across the smallest clear gap.\nEqual-Q: scalar inlet velocity root solve against Q_rack = 0.0055463.\nMass conservation: compatible finite-volume divergence/gradient operators.\nSee production_summary.csv, mass_conservation_all_cases.csv, grid_independence_small_pitch.csv, timestep_independence_small_pitch.csv, solver source, root driver, and PDF report.\n''')
print('built',OUT)
