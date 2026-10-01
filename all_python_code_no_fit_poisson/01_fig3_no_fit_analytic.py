"""
No-fit analytical reconstruction / extension of Fig. 3
S.-K. Chung, IEEE Trans. Electron Devices 47(5), 2000.

Physics:
1) Poisson/full-depletion RESURF model.
2) Fulop avalanche law alpha_eff = A*E^7.
3) Exact finite-L avalanche integral in closed form.
4) Explicit L/lc >> 1 closed-form BV.
5) Optional no-fit conventional 1-D junction continuation after RESURF is lost.
6) Drift-region on-resistance from parallel conductance.

No data-point fitting is used.
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# Constants (cgs-centimeter device units)
q = 1.602176634e-19
eps0 = 8.8541878128e-14       # F/cm
eps_s = 11.8*eps0             # F/cm
A = 1.8e-35                   # Fulop alpha_eff = A E^7, E in V/cm
um = 1e-4                     # cm

# Fig. 3 / paper geometry
L  = 40.0*um
t1 = 1.0*um
t2 = 4.0*um
ts = t1+t2
tb = 2.0*um
tj = 2.0*um
Vsb = 0.0

# Paper reports lc = 6.614 um. Infer k=eps_ox/eps_s from that report.
lc_paper = 6.614*um
k = tb/(lc_paper**2/ts - ts/2)
tc = t2/2 + tb/k
lc = math.sqrt(ts*(ts/2 + tb/k))
lam = L/lc

def sigma_c(N1, N2, Vsb=0.0):
    return (q*N1/eps_s*(tc+ts/2)*t1
            + q*N2/eps_s*(tc*t2)
            + Vsb)

def J7_closed(K, lam=lam):
    """
    Exact closed form:
      J7 = int_0^lam [K cosh u + cosh(lam-u)]^7 du.
    """
    a = K + math.exp(-lam)
    b = K + math.exp(lam)
    s = 0.0
    for m in range(8):
        p = 2*m - 7
        s += (math.comb(7,m) * a**m * b**(7-m)
              * math.expm1(p*lam)/p)
    return s/(2.0**7)

def sigma_from_K_exact(K):
    J = J7_closed(K)
    return (lc**6*math.sinh(lam)**7/(A*J))**(1/7)

sigma0_exact = sigma_from_K_exact(0.0)

def K_from_sigma_exact(sig):
    """
    Physical K>=0 root of sigma_from_K_exact(K)=sig.
    Bisection: equation solving, not curve fitting.
    """
    if not (0.0 < sig <= sigma0_exact):
        return float("nan")
    lo, hi = 0.0, 1.0
    while sigma_from_K_exact(hi) > sig:
        hi *= 2.0
        if hi > 1e9:
            raise RuntimeError("Could not bracket K.")
    for _ in range(100):
        mid = 0.5*(lo+hi)
        if sigma_from_K_exact(mid) > sig:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

def bv_lateral_exact(sig):
    K = K_from_sigma_exact(sig)
    return sig*(1.0+K)

# Explicit long-drift formula
sigma0 = (7.0*lc**6/A)**(1/7)
sigma1 = sigma0/(2.0**(1/7))

def bv_lateral_closed(sig):
    if not (0.0 < sig <= sigma0):
        return float("nan")
    K = ((sigma0/sig)**7 - 1.0)**(1/7)
    return sig*(1.0+K)

# Vertical cap
Ec = (1.0/(A*(ts-tj)))**(1/7)
BV_VF = Ec*((ts-tj)+tb/k)

# No-fit asymptotic continuation after RESURF loss
def Nbar(N1, N2):
    return (N1*t1+N2*t2)/ts

def bv_1d_overdoped(N1, N2):
    N = Nbar(N1,N2)
    return (1.0/(2.0*A))**0.25*(eps_s/(q*N))**0.75

def bv_general(N1, N2, use_exact_resurf=True):
    sig = sigma_c(N1,N2,Vsb)
    sigcrit = sigma0_exact if use_exact_resurf else sigma0
    if sig <= sigcrit:
        bvl = bv_lateral_exact(sig) if use_exact_resurf else bv_lateral_closed(sig)
        return min(bvl, BV_VF)
    return bv_1d_overdoped(N1,N2)

# Drift-region on resistance times width [ohm cm]
def ron_drift_W(N1, N2, mu1=1350.0, mu2=1350.0):
    return L/(q*(mu1*N1*t1 + mu2*N2*t2))

# Critical doses
def N1_for_sigma(sig_target, N2):
    return ((eps_s/q)*(sig_target-Vsb) - N2*tc*t2)/((tc+ts/2)*t1)

print(f"k = {k:.6f}")
print(f"lc = {lc/um:.4f} um, L/lc = {lam:.4f}")
print(f"sigma0 exact = {sigma0_exact:.4f} V")
print(f"sigma0 closed = {sigma0:.4f} V")
print(f"sigma1 closed = {sigma1:.4f} V")
print(f"BV_VF = {BV_VF:.3f} V")
for N2 in (1e14,1e15):
    print(f"N2={N2:.0e}: N1(K=1)={N1_for_sigma(sigma1,N2):.4e}, "
          f"N1crit={N1_for_sigma(sigma0_exact,N2):.4e} cm^-3")

# Fig.-3-like plot
N1 = np.linspace(1e15,1.6e16,500)
fig, ax1 = plt.subplots(figsize=(8.5,6.5))
ax2 = ax1.twinx()

for N2 in (1e14,1e15):
    x = N1/1e15
    bv = np.array([bv_general(n,N2,True) for n in N1])
    ron = np.array([ron_drift_W(n,N2) for n in N1])
    ax1.plot(x,bv,label=f"BV, N2={N2:.0e} cm^-3")
    ax2.plot(x,ron,linestyle="--",label=f"Rdrift*W, N2={N2:.0e} cm^-3")

ax1.set_xlim(0,16); ax1.set_ylim(50,300); ax2.set_ylim(0,120)
ax1.set_xlabel("Surface doping N1 [1e15 cm^-3]")
ax1.set_ylabel("Breakdown voltage [V]")
ax2.set_ylabel("Drift on-resistance x width [ohm cm]")
ax1.set_title("Fig. 3 no-fit analytical reconstruction")
ax1.grid(True,alpha=0.25)
h1,l1=ax1.get_legend_handles_labels()
h2,l2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2,l1+l2,fontsize=8,loc="best")
fig.tight_layout()
plt.show()
