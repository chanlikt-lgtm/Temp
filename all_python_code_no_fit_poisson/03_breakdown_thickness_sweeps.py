import math
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# -------------------- constants; no fitting --------------------
q = 1.602176634e-19            # C
eps0 = 8.8541878128e-14       # F/cm
eps_s = 11.8 * eps0
eps_ox = 3.9 * eps0
k = eps_ox / eps_s
A = 1.8e-35                   # Fulop: alpha_eff = A E^7, E in V/cm
um = 1e-4                     # cm
L = 40.0 * um
t1 = 1.0 * um
tj = 2.0 * um
Vsb = 0.0
N2_default = 1.0e14

OUT = Path('/mnt/data/fig3_green_function')


def j7_closed(K, lam):
    # Stable enough for present lambda range (~4-10)
    a = K + math.exp(-lam)
    b = K + math.exp(lam)
    total = 0.0
    for m in range(8):
        p = 2*m - 7
        total += math.comb(7, m) * (a**m) * (b**(7-m)) * math.expm1(p*lam) / p
    return total / 128.0


def geom(ts_um=5.0, tb_um=2.0):
    ts = ts_um * um
    tb = tb_um * um
    t2 = ts - t1
    if t2 <= 0:
        raise ValueError('t_s must exceed t_1')
    lc = math.sqrt(ts * (ts/2.0 + tb/k))
    tc = t2/2.0 + tb/k
    lam = L/lc
    return ts, tb, t2, lc, tc, lam


def sigma_c(N1, N2, ts_um=5.0, tb_um=2.0):
    ts, tb, t2, lc, tc, lam = geom(ts_um, tb_um)
    return (q*N1/eps_s * (tc + ts/2.0)*t1
            + q*N2/eps_s * (tc*t2) + Vsb)


def sigma_from_K(K, ts_um=5.0, tb_um=2.0):
    ts, tb, t2, lc, tc, lam = geom(ts_um, tb_um)
    J = j7_closed(K, lam)
    return (lc**6 * math.sinh(lam)**7 / (A*J))**(1.0/7.0)


def sigma0(ts_um=5.0, tb_um=2.0):
    return sigma_from_K(0.0, ts_um, tb_um)


def sigma1(ts_um=5.0, tb_um=2.0):
    return sigma_from_K(1.0, ts_um, tb_um)


def bvh_from_K(K, ts_um=5.0, tb_um=2.0):
    s = sigma_from_K(K, ts_um, tb_um)
    return s*(1.0+K)


def K_from_sigma(sig, ts_um=5.0, tb_um=2.0):
    s0 = sigma0(ts_um, tb_um)
    if sig <= 0 or sig > s0*(1+1e-12):
        return math.nan
    # sigma decreases monotonically with K on K>=0
    lo, hi = 0.0, 1.0
    while sigma_from_K(hi, ts_um, tb_um) > sig:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError('K bracket failed')
    for _ in range(90):
        mid = 0.5*(lo+hi)
        if sigma_from_K(mid, ts_um, tb_um) > sig:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)


def bvh_from_N1(N1, N2, ts_um=5.0, tb_um=2.0):
    sig = sigma_c(N1, N2, ts_um, tb_um)
    if sig > sigma0(ts_um, tb_um):
        return math.nan
    K = K_from_sigma(sig, ts_um, tb_um)
    return sig*(1.0+K)


def vertical_breakdown_general(Nv, ts_um=5.0, tb_um=2.0, tj_um=2.0):
    """1-D vertical Poisson + Fulop, for a uniform depleted Si segment below drain."""
    ts = ts_um*um; tb = tb_um*um; tjc = tj_um*um
    d = ts - tjc
    if d <= 0:
        return math.nan, math.nan, math.nan
    g = q*Nv/eps_s
    if abs(g) < 1e-40:
        E = (1.0/(A*d))**(1.0/7.0)
        return E*(d+tb/k), E, E
    def F(E0):
        Eb = E0 + g*d
        return A*(Eb**8 - E0**8)/(8.0*g) - 1.0
    lo, hi = 0.0, 1e5
    while F(hi) < 0:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError('vertical field bracket failed')
    for _ in range(120):
        mid = 0.5*(lo+hi)
        if F(mid) >= 0:
            hi = mid
        else:
            lo = mid
    E0 = 0.5*(lo+hi)
    Eb = E0 + g*d
    V = E0*d + 0.5*g*d*d + Eb*tb/k
    return V, E0, Eb


def vertical_breakdown_lowcharge(ts_um=5.0, tb_um=2.0, tj_um=2.0):
    ts = ts_um*um; tb = tb_um*um; tjc=tj_um*um
    d = ts-tjc
    E = (1.0/(A*d))**(1.0/7.0)
    return E*(d+tb/k)


def N1_from_sigma(sig, N2, ts_um=5.0, tb_um=2.0):
    ts, tb, t2, lc, tc, lam = geom(ts_um, tb_um)
    denom = t1*(tc+ts/2.0)
    return ((eps_s/q)*(sig - Vsb) - N2*tc*t2)/denom


def crossover_K(BVv, ts_um=5.0, tb_um=2.0):
    B0 = bvh_from_K(0.0, ts_um, tb_um)
    B1 = bvh_from_K(1.0, ts_um, tb_um)
    if BVv <= B0:
        return None, None, 'vertical_all'
    if BVv >= B1:
        return None, None, 'horizontal_all'
    def f(K):
        return bvh_from_K(K, ts_um, tb_um) - BVv
    # low-K root in (0,1)
    lo, hi = 0.0, 1.0
    flo, fhi = f(lo), f(hi)
    for _ in range(100):
        mid=0.5*(lo+hi)
        fm=f(mid)
        if flo*fm <= 0:
            hi=mid; fhi=fm
        else:
            lo=mid; flo=fm
    Klo=0.5*(lo+hi)
    # symmetry gives reciprocal root, but solve independently as check
    lo, hi = 1.0, 2.0
    while f(hi) > 0:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError('high-K crossover bracket failed')
    flo, fhi = f(lo), f(hi)
    for _ in range(100):
        mid=0.5*(lo+hi)
        fm=f(mid)
        if flo*fm <= 0:
            hi=mid; fhi=fm
        else:
            lo=mid; flo=fm
    Khi=0.5*(lo+hi)
    return Klo, Khi, 'two_crossovers'


def overdoped_bv(N1, N2, ts_um=5.0):
    ts = ts_um*um
    t2 = ts-t1
    Nbar=(N1*t1+N2*t2)/ts
    return (1.0/(2.0*A))**0.25*(eps_s/(q*Nbar))**0.75


def curve(N2, ts_um, tb_um, npts=500):
    N1s=np.linspace(1e15,1.6e16,npts)
    BVv, E0, Eb = vertical_breakdown_general(N2,ts_um,tb_um,tj/um)
    s0=sigma0(ts_um,tb_um)
    vals=[]; valid=[]; post=[]
    for n in N1s:
        sig=sigma_c(n,N2,ts_um,tb_um)
        if sig <= s0:
            h=bvh_from_N1(n,N2,ts_um,tb_um)
            vals.append(min(h,BVv)); valid.append(True); post.append(False)
        else:
            vals.append(overdoped_bv(n,N2,ts_um)); valid.append(False); post.append(True)
    return N1s,np.array(vals),np.array(valid),BVv


def summary_row(label, N2, ts_um, tb_um):
    BVv,E0,Eb=vertical_breakdown_general(N2,ts_um,tb_um,tj/um)
    s0=sigma0(ts_um,tb_um); s1=sigma1(ts_um,tb_um)
    B0=bvh_from_K(0,ts_um,tb_um); B1=bvh_from_K(1,ts_um,tb_um)
    Ncrit=N1_from_sigma(s0,N2,ts_um,tb_um)
    Nopt=N1_from_sigma(s1,N2,ts_um,tb_um)
    Klo,Khi,reg=crossover_K(BVv,ts_um,tb_um)
    if reg=='two_crossovers':
        # low doping = high K; high doping = low K
        Nentry=N1_from_sigma(sigma_from_K(Khi,ts_um,tb_um),N2,ts_um,tb_um)
        Nexit=N1_from_sigma(sigma_from_K(Klo,ts_um,tb_um),N2,ts_um,tb_um)
    else:
        Nentry=math.nan; Nexit=math.nan
    return {
        'case':label,'ts_um':ts_um,'tb_um':tb_um,'BVv_V':BVv,
        'sigma0_V':s0,'BVHmax_V':B1,'N1_K1_cm-3':Nopt,
        'N1_plateau_entry_cm-3':Nentry,'N1_plateau_exit_cm-3':Nexit,
        'N1_crit_cm-3':Ncrit,'K_low':Klo if Klo is not None else math.nan,
        'K_high':Khi if Khi is not None else math.nan,'regime':reg,
        'Etop_Vcm':E0,'Ebottom_Vcm':Eb,
        'BVv_lowcharge_V':vertical_breakdown_lowcharge(ts_um,tb_um,tj/um)
    }


def plot_sweep(cases, title, outfile, N2=N2_default, ymax=530):
    fig,ax=plt.subplots(figsize=(8.4,5.8))
    for label,ts_um,tb_um in cases:
        N1s,vals,valid,BVv=curve(N2,ts_um,tb_um)
        x=N1s/1e15
        # valid branch solid; post-RESURF asymptote dashed
        ysolid=np.where(valid,vals,np.nan)
        ydash=np.where(~valid,vals,np.nan)
        line,=ax.plot(x,ysolid,label=label)
        ax.plot(x,ydash,linestyle='--',color=line.get_color(),alpha=0.85)
        row=summary_row(label,N2,ts_um,tb_um)
        # mark vertical-cap entry/exit and critical full-depletion limit
        for key,marker in [('N1_plateau_entry_cm-3','o'),('N1_plateau_exit_cm-3','o')]:
            n=row[key]
            if math.isfinite(n) and 1e15<=n<=1.6e16:
                ax.plot(n/1e15,row['BVv_V'],marker=marker,linestyle='None',color=line.get_color(),markersize=4)
        ncrit=row['N1_crit_cm-3']
        if 1e15<=ncrit<=1.6e16:
            ycrit=min(bvh_from_K(0,ts_um,tb_um),BVv)
            ax.plot(ncrit/1e15,ycrit,marker='x',linestyle='None',color=line.get_color(),markersize=6)
    ax.set_xlim(1,16)
    ax.set_ylim(80,ymax)
    ax.set_xlabel(r'Surface doping $N_1$ [$10^{15}$ cm$^{-3}$]')
    ax.set_ylabel(r'Breakdown voltage $V_{BD}$ [V]')
    ax.set_title(title)
    ax.grid(True,alpha=0.25)
    ax.legend(fontsize=9,loc='best')
    ax.text(0.01,0.02,'solid: fully depleted Green/Fulop branch; dashed: no-fit post-RESURF 1-D asymptote',
            transform=ax.transAxes,fontsize=8,va='bottom')
    fig.tight_layout()
    fig.savefig(outfile,dpi=220,bbox_inches='tight')
    plt.close(fig)


oxide_cases=[('low BOX: $t_b=1$ um',5.0,1.0),('medium BOX: $t_b=2$ um',5.0,2.0),('high BOX: $t_b=4$ um',5.0,4.0)]
si_cases=[('thin Si: $t_s=3$ um',3.0,2.0),('medium Si: $t_s=5$ um',5.0,2.0),('thick Si: $t_s=7$ um',7.0,2.0)]

plot_sweep(oxide_cases,'No-fit $V_{BD}(N_1)$: buried-oxide thickness sweep',OUT/'vbd_vs_N1_box_sweep.png',ymax=535)
plot_sweep(si_cases,'No-fit $V_{BD}(N_1)$: silicon-thickness sweep',OUT/'vbd_vs_N1_si_sweep.png',ymax=435)

rows=[]
for c in oxide_cases+si_cases:
    rows.append(summary_row(c[0],N2_default,c[1],c[2]))
with open(OUT/'vbd_turning_points.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    w.writeheader();w.writerows(rows)

# Curve data for reproducibility
with open(OUT/'vbd_sweep_curves.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['sweep','label','N1_cm-3','VBD_V','full_depletion_valid'])
    for sweep,cases in [('BOX',oxide_cases),('Si',si_cases)]:
        for label,ts_um,tb_um in cases:
            N1s,vals,valid,BVv=curve(N2_default,ts_um,tb_um)
            for n,v,ok in zip(N1s,vals,valid):
                w.writerow([sweep,label,n,v,int(ok)])

print('k=',k)
for r in rows:
    print(r)

# Turning-mechanism decomposition for the nominal geometry
N2=N2_default; ts_um=5.0; tb_um=2.0
N1s=np.linspace(1e15,1.6e16,600)
BVv,_,_=vertical_breakdown_general(N2,ts_um,tb_um,tj/um)
s0=sigma0(ts_um,tb_um)
bvh=[]; actual=[]; post=[]
for n in N1s:
    sig=sigma_c(n,N2,ts_um,tb_um)
    if sig<=s0:
        h=bvh_from_N1(n,N2,ts_um,tb_um)
        bvh.append(h); actual.append(min(h,BVv)); post.append(np.nan)
    else:
        bvh.append(np.nan); actual.append(np.nan); post.append(overdoped_bv(n,N2,ts_um))
row=summary_row('nominal',N2,ts_um,tb_um)
fig,ax=plt.subplots(figsize=(8.4,5.8))
ax.plot(N1s/1e15,bvh,label=r'horizontal avalanche $BV_H$')
ax.axhline(BVv,linestyle='--',label=r'vertical avalanche cap $BV_V$')
ax.plot(N1s/1e15,actual,linewidth=2.4,label=r'actual $V_{BD}=\min(BV_H,BV_V)$')
ax.plot(N1s/1e15,post,linestyle=':',linewidth=2.0,label='post-RESURF 1-D asymptote')
points=[
    ('plateau entry',row['N1_plateau_entry_cm-3'],BVv),
    ('$K=1$',row['N1_K1_cm-3'],bvh_from_K(1,ts_um,tb_um)),
    ('plateau exit',row['N1_plateau_exit_cm-3'],BVv),
    ('$K=0$',row['N1_crit_cm-3'],bvh_from_K(0,ts_um,tb_um)),
]
for label,n,y in points:
    if math.isfinite(n):
        ax.plot(n/1e15,y,marker='o',linestyle='None')
        dx = 0.18
        dy = 8 if label not in ('$K=1$',) else -28
        ax.annotate(label,(n/1e15,y),xytext=(n/1e15+dx,y+dy),fontsize=8,
                    arrowprops=dict(arrowstyle='->',lw=0.7))
ax.set_xlim(1,16); ax.set_ylim(110,435)
ax.set_xlabel(r'Surface doping $N_1$ [$10^{15}$ cm$^{-3}$]')
ax.set_ylabel(r'Breakdown voltage [V]')
ax.set_title('What sets the turning points of $V_{BD}(N_1)$?')
ax.grid(True,alpha=0.25); ax.legend(fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(OUT/'vbd_turning_mechanism.png',dpi=220,bbox_inches='tight'); plt.close(fig)
