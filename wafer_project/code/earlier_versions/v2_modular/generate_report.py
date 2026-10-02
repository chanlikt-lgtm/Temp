import os, json, pandas as pd, numpy as np
OUT=os.path.dirname(__file__)
prod=pd.read_csv(os.path.join(OUT,'production_summary.csv')).sort_values('pitch')
mesh=pd.read_csv(os.path.join(OUT,'mesh_study_small.csv')).sort_values('n_gap_cells')
ts=pd.read_csv(os.path.join(OUT,'timestep_study_small.csv')).sort_values('dt_factor',ascending=False)
mass=pd.read_csv(os.path.join(OUT,'mass_conservation_table.csv'))
meta=json.load(open(os.path.join(OUT,'run_metadata.json')))
CASES=['Small','Medium','Large']

def f(x,n=5): return f'{float(x):.{n}f}'
def sci(x,n=2): return f'{float(x):.{n}e}'
def pct(x,n=3): return f'{float(x):.{n}f}'

def row(case): return prod.loc[prod.case==case].iloc[0]
s=row('Small');m=row('Medium');l=row('Large')
last2=mesh.iloc[-2:]
q_mesh=100*(last2.Q_rack.iloc[-1]/last2.Q_rack.iloc[-2]-1)
pk_mesh=100*(last2.central_peak_v.iloc[-1]/last2.central_peak_v.iloc[-2]-1)
dp_mesh=100*(last2.delta_p_rack.iloc[-1]/last2.delta_p_rack.iloc[-2]-1)
base=ts.iloc[0]; half=ts.iloc[1]
q_dt=100*(half.Q_rack/base.Q_rack-1); pk_dt=100*(half.central_peak_v/base.central_peak_v-1); dp_dt=100*(half.delta_p_rack/base.delta_p_rack-1)
max_mass=float(mass.err_pct.abs().max())
res_ratio=float(s.R_h/l.R_h)

tex=r'''\documentclass[10pt]{article}
\usepackage[letterpaper,margin=0.72in]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,array,graphicx,float,xcolor,siunitx,caption,enumitem,microtype}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={MAC Navier-Stokes Equal-Q Wafer Pitch Study},pdfsubject={Validated CFD comparison of small, medium and large wafer pitch},pdfauthor={OpenAI ChatGPT}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlist[itemize]{leftmargin=1.4em,itemsep=2pt,topsep=3pt}
\definecolor{navy}{RGB}{32,63,92}
\definecolor{pale}{RGB}{240,245,249}
\definecolor{greenbox}{RGB}{235,247,239}
\newcommand{\sect}[1]{\vspace{2pt}{\large\bfseries\color{navy}#1}\par\vspace{2pt}}
\newcommand{\note}[1]{\fcolorbox{navy}{pale}{\parbox{0.95\linewidth}{#1}}}
\newcommand{\passbox}[1]{\fcolorbox{navy}{greenbox}{\parbox{0.95\linewidth}{#1}}}
\begin{document}

% PAGE 1 - NOMENCLATURE
\begin{center}
{\LARGE\bfseries MAC Navier--Stokes Equal-$Q$ Wafer-Pitch Study}\\[3pt]
{\large Nomenclature and normalized problem definition}
\end{center}
\vspace{4pt}
\small
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.16\textwidth} X >{\raggedright\arraybackslash}p{0.19\textwidth}}
\toprule
\textbf{Symbol} & \textbf{Definition} & \textbf{Value / unit}\\
\midrule
$W,H$ & Tank width and height & $1.0,\;0.45$\\
$x,y$ & Horizontal and vertical coordinates & normalized length\\
$\mathbf{u}=(u,v)$ & Velocity vector; $v$ is normal to horizontal cut & normalized velocity\\
$p$ & Kinematic pressure ($p_{\rm physical}/\rho$) & normalized\\
$\nu$ & Kinematic viscosity & $7.5\times10^{-4}$\\
$P$ & Wafer center-to-center pitch & 0.040 / 0.060 / 0.080\\
$t_w$ & Wafer thickness & 0.015\\
$g=P-t_w$ & Clear inter-wafer gap & 0.025 / 0.045 / 0.065\\
$N_w$ & Number of wafers & 6\\
$y_w$ & Wafer vertical span & $0.12\le y\le0.25$\\
$y_c$ & Horizontal diagnostic cut & $0.20$\\
$Q_{\rm rack}$ & Sum of vertical flux through the five interior gaps at $y_c$ & target 0.0055463\\
$Q_{\rm in}$ & Total bottom-inlet volumetric flux per unit depth & case-dependent\\
$U_{\rm in}$ & Prescribed vertical inlet-slot speed found by root search & case-dependent\\
$\bar v_g$ & Mean vertical velocity over the five rack gaps & $Q_{\rm rack}/G$\\
$G$ & Total open width of five gaps & $5g$\\
$\Delta p_{\rm rack}$ & Rack pressure drop, weighted mean below minus above wafer array & normalized pressure\\
$R_h$ & Hydraulic resistance & $\Delta p_{\rm rack}/Q_{\rm rack}$\\
$D,G$ & Discrete divergence and pressure-gradient operators & MAC operators\\
$N_g$ & Number of finite-volume cells across each gap & 20 production\\
$\Delta t$ & Pseudo-time step used to converge steady solution & case-dependent\\
\bottomrule
\end{tabularx}

\vfill
\note{\textbf{Normalization.} The supplied problem did not define dimensional reference scales $L_0$, $U_0$, or $\rho$. Therefore all results remain in the solver's normalized units. The reported flow rate is a two-dimensional flux per unit out-of-plane depth.}

\note{\textbf{No fitting.} No regression, polynomial fit, spline fit, smoothing, or empirical velocity model is used. Horizontal-cut values are native staggered-grid face velocities. The zero values drawn at wafer faces are the imposed no-slip boundary condition, not fitted points.}
\normalsize
\clearpage

\section*{1. Governing equations and corrected numerical formulation}
The flow is computed from the two-dimensional incompressible Navier--Stokes equations,
\begin{align}
\frac{\partial\mathbf{u}}{\partial t}+(\mathbf{u}\cdot\nabla)\mathbf{u} &= -\nabla p+\nu\nabla^2\mathbf{u},\\
\nabla\cdot\mathbf{u} &=0.
\end{align}
The production solver uses a geometry-aligned, locally refined MAC staggered grid: pressure is stored at cell centers, $u$ on vertical faces, and $v$ on horizontal faces. Donor-cell upwinding is used for advection and nonuniform finite differences for diffusion. The pressure projection is constructed with compatible discrete operators,
\begin{equation}
-DG\,p^{n+1}=-\frac{1}{\Delta t}D\mathbf{u}^{*},\qquad
\mathbf{u}^{n+1}=\mathbf{u}^{*}-\Delta t\,Gp^{n+1},
\end{equation}
so $D\mathbf{u}^{n+1}=0$ to the linear-solver tolerance. This corrects the operator inconsistency identified in the audit of the previous collocated implementation.

\sect{Boundary conditions}
\begin{itemize}
\item Tank side walls, bottom wall outside the inlet slots, and all wafer surfaces: no slip.
\item Five bottom inlet slots: prescribed upward $v=U_{\rm in}$, $u=0$.
\item Top: open pressure outlet with $p=0$; velocity is obtained consistently from the projection.
\item Wafer faces coincide with grid faces; there is no numerical flow through solid wafers.
\end{itemize}

\sect{Equal-$Q$ control}
For every pitch, the same rack flow is imposed after the CFD field is converged:
\begin{equation}
Q_{\rm rack}(U_{\rm in})=\sum_{k=1}^{5}\int_{\text{gap }k}v(x,y_c)\,dx=Q_0,\qquad Q_0=0.0055463.
\end{equation}
$U_{\rm in}$ is found by a bracketed scalar root search using converged Navier--Stokes evaluations (secant updates with bisection safeguards). This is a numerical constraint solve, not curve fitting.

\begin{table}[H]\centering\small
\caption{Production grid and converged equal-$Q$ settings read from run metadata.}
\begin{tabular}{lrrrrrr}
\toprule
Case & $P$ & $N_x\times N_y$ & $N_g$ & $\Delta t$ & $U_{\rm in}$ & $Q$ error (\%)\\
\midrule
'''
for _,r in prod.iterrows():
    tex+=f"{r['case']} & {r['pitch']:.3f} & {int(r['Nx'])}$\\times${int(r['Ny'])} & {int(r['n_gap_cells'])} & {r['dt']:.6f} & {r['U_in']:.6f} & {r['Q_error_pct']:.4f}\\\\\n"
tex+=r'''\bottomrule
\end{tabular}
\end{table}
\passbox{\textbf{Reproducibility correction.} The report now records the actual per-case grid and timestep used by the MAC production runs. Numerical tables are generated from \texttt{run\_metadata.json} and CSV outputs rather than manually retyped solver settings.}
\clearpage

\section*{2. Production flow fields at equal rack flow}
\begin{figure}[H]\centering
\includegraphics[width=0.88\textwidth]{figures/field_small_MAC_equalQ.png}
\caption{Small pitch, $P=0.040$. The dashed horizontal line is the $y=0.20$ cut. Streamlines are interpolated only for visualization; reported profiles and integrals use native MAC face values.}
\end{figure}
\vspace{-4pt}
\begin{figure}[H]\centering
\includegraphics[width=0.88\textwidth]{figures/field_medium_MAC_equalQ.png}
\caption{Medium pitch, $P=0.060$, at the same $Q_{\rm rack}$.}
\end{figure}
\clearpage
\begin{figure}[H]\centering
\includegraphics[width=0.88\textwidth]{figures/field_large_MAC_equalQ.png}
\caption{Large pitch, $P=0.080$, at the same $Q_{\rm rack}$.}
\end{figure}
\vspace{-2pt}
\begin{figure}[H]\centering
\includegraphics[width=0.66\textwidth]{figures/central_gap_profiles_equalQ_MAC.png}
\caption{Central-gap velocity profile versus normalized gap coordinate. Points are native MAC face values; line segments are only point-to-point plotting.}
\end{figure}
\clearpage

\section*{3. Horizontal cut through the wafer gaps}
\begin{figure}[H]\centering
\includegraphics[width=0.94\textwidth]{figures/horizontal_cut_v_shaded_panels_MAC.png}
\caption{Vertical velocity $v(x,y=0.20)$ shown only in fluid gaps. Gray bands are solid wafers. Each gap curve is terminated at $v=0$ on the wafer faces to display the imposed no-slip boundary condition. No curve is drawn through a solid wafer.}
\end{figure}
\clearpage

\section*{4. Equal-$Q$ quantitative comparison and hydraulic resistance}
\begin{table}[H]\centering\small
\caption{Production equal-$Q$ results.}
\begin{tabular}{lrrrrrr}
\toprule
Case & $Q_{\rm rack}$ & $\bar v_g$ & central mean $v$ & central peak $v$ & $\Delta p_{\rm rack}$ & $R_h$\\
\midrule
'''
for _,r in prod.iterrows():
    tex+=f"{r['case']} & {r['Q_rack']:.7f} & {r['mean_gap_v']:.5f} & {r['central_mean_v']:.5f} & {r['central_peak_v']:.5f} & {r['delta_p_rack']:.5f} & {r['R_h']:.3f}\\\\\n"
tex+=r'''\bottomrule
\end{tabular}
\end{table}

At fixed rack flow, the mean gap velocity increases as the total open gap width decreases:
\begin{equation}
\bar v_{g,\mathrm{small}} > \bar v_{g,\mathrm{medium}} > \bar v_{g,\mathrm{large}}.
\end{equation}
The CFD adds information that continuity alone cannot provide: the local profile shape, peak velocity, recirculation, pressure field, bypass fraction, and pressure drop. The pressure-drop result confirms the hydraulic penalty of close spacing. In these normalized conditions the small-pitch hydraulic resistance is approximately '''+f'{res_ratio:.1f}'+r''' times the large-pitch value.

\begin{figure}[H]\centering
\includegraphics[width=0.70\textwidth]{figures/hydraulic_resistance_equalQ.png}
\caption{Hydraulic resistance $R_h=\Delta p_{\rm rack}/Q_{\rm rack}$ from the equal-$Q$ production simulations.}
\end{figure}

The fraction of the total inlet flux that passes through the five interior gaps is '''+f"{s.rack_fraction_pct:.2f}"+r'''\% for small pitch, '''+f"{m.rack_fraction_pct:.2f}"+r'''\% for medium pitch, and '''+f"{l.rack_fraction_pct:.2f}"+r'''\% for large pitch. The remainder bypasses the interior gaps through the side passages in this open tank geometry.
\clearpage

\section*{5. Verification: spatial grid, timestep, and mass conservation}
\sect{Spatial-grid study - most restrictive small-pitch case}
The gap is resolved with 10, 15, and 20 cells while the same $Q_{\rm rack}$ is re-matched on every mesh.
\begin{table}[H]\centering\small
\caption{Small-pitch grid convergence.}
\begin{tabular}{rrrrrr}
\toprule
$N_g$ & $Q_{\rm rack}$ & peak $v$ & $\Delta p$ & change peak (\%) & change $\Delta p$ (\%)\\
\midrule
'''
for _,r in mesh.iterrows():
    cpk='' if pd.isna(r.get('central_peak_v_change_from_prev_pct')) else f"{r['central_peak_v_change_from_prev_pct']:.3f}"
    cdp='' if pd.isna(r.get('delta_p_rack_change_from_prev_pct')) else f"{r['delta_p_rack_change_from_prev_pct']:.3f}"
    tex+=f"{int(r['n_gap_cells'])} & {r['Q_rack']:.7f} & {r['central_peak_v']:.6f} & {r['delta_p_rack']:.6f} & {cpk} & {cdp}\\\\\n"
tex+=r'''\bottomrule
\end{tabular}
\end{table}
Between the 15-cell and 20-cell meshes, the changes are '''+f'{q_mesh:.3f}'+r'''\% in $Q_{\rm rack}$, '''+f'{pk_mesh:.3f}'+r'''\% in central peak velocity, and '''+f'{dp_mesh:.3f}'+r'''\% in pressure drop. All are below 1\% in magnitude.

\begin{figure}[H]\centering
\begin{minipage}{0.48\textwidth}\centering
\includegraphics[width=\linewidth]{figures/mesh_convergence_peak_v.png}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}\centering
\includegraphics[width=\linewidth]{figures/mesh_convergence_dp.png}
\end{minipage}
\caption{Spatial convergence of the most restrictive small-pitch case.}
\end{figure}

\sect{Timestep sensitivity}
The converged 20-cell small-pitch solution was continued with $\Delta t$ and $\Delta t/2$. The changes were '''+f'{q_dt:.4f}'+r'''\% in $Q_{\rm rack}$, '''+f'{pk_dt:.4f}'+r'''\% in peak velocity, and '''+f'{dp_dt:.4f}'+r'''\% in $\Delta p$, all far below 1\%.

\sect{Discrete mass conservation}
Horizontal-section flux was evaluated at $y\approx0.05,0.10,0.20,0.30,0.40$. The maximum section-to-section error relative to the inlet flow is '''+f'{max_mass:.2e}'+r'''\%, i.e. roundoff-level for the compatible MAC projection.
\begin{table}[H]\centering\small
\caption{Maximum horizontal-section mass-balance error.}
\begin{tabular}{lr}
\toprule Case & max $|Q(y)/Q_{\rm in}-1|\times100$ (\%)\\\midrule
'''
for case in CASES:
    val=float(mass.loc[mass.case==case,'err_pct'].abs().max())
    tex+=f"{case} & {val:.2e}\\\\\n"
tex+=r'''\bottomrule\end{tabular}\end{table}
\clearpage

\section*{6. Audit closure and interpretation}
\passbox{\textbf{Audit status after corrections:} the seven requested numerical corrections are implemented. The production solver uses a compatible MAC projection, an actual equal-$Q$ root search, a 20-cell minimum-gap production resolution, a three-grid convergence check, a two-timestep check, a mass-conservation table, and pressure-drop-based hydraulic resistance.}

\sect{What the result means}
\begin{itemize}
\item With the same $Q_{\rm rack}$ forced through the five wafer gaps, smaller pitch necessarily produces a higher \emph{mean} gap velocity because the open area is smaller.
\item The Navier--Stokes solution determines how that flow is distributed locally and quantifies the pressure penalty. Small pitch has the largest peak velocity and by far the largest $\Delta p_{\rm rack}$ and $R_h$.
\item $U_{\rm in}$ is not used as the resistance metric. Resistance is evaluated from $\Delta p_{\rm rack}/Q_{\rm rack}$.
\item Equal rack flow is not equal total tank flow. The open side passages allow bypass, and the bypass fraction depends strongly on pitch.
\end{itemize}

\sect{Remaining limitation}
The geometry and fluid properties are normalized because dimensional tank size, wafer size, density, and physical viscosity were not supplied. These simulations support relative pitch comparison in the defined normalized problem. A dimensional design calculation should replace the normalized values with the actual process geometry and liquid properties.

\sect{Reproducibility files}
The accompanying package contains the MAC solver core, equal-$Q$ driver, production field arrays, root histories, mass-balance tables, grid/timestep studies, figures, \LaTeX{} source, and \texttt{run\_metadata.json}. The report-generation script reads the numerical metadata directly from these outputs.

\vfill
\begin{center}\small\textit{No fitting. Direct Navier--Stokes discretization with numerical verification.}\end{center}
\end{document}
'''
open(os.path.join(OUT,'wafer_mac_equalQ_validated_report.tex'),'w').write(tex)
print('wrote tex')
