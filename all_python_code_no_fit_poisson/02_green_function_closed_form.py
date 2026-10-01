import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ------------------------------------------------------------
# No-fit Green-function analytical reconstruction of Fig. 3
# Chung, IEEE TED 47(5), 2000
# ------------------------------------------------------------
q = 1.602176634e-19
eps0 = 8.8541878128e-14       # F/cm
eps_s = 11.8 * eps0           # F/cm
A = 1.8e-35                   # Fulop: alpha_eff = A E^7, E in V/cm
um = 1e-4                     # cm

# Paper geometry
L = 40.0 * um
t1 = 1.0 * um
t2 = 4.0 * um
ts = t1 + t2
tb = 2.0 * um
tj = 2.0 * um
Vsb = 0.0

# The paper explicitly reports lc=6.614 um for ts=5 um and tb=2 um.
# This is used as a stated analytical-model parameter, NOT fitted to Fig. 3.
lc = 6.614 * um
lam = L / lc

# Algebraic inversion of the paper's own lc definition, only to map sigma_c <-> N1.
k = tb / (lc**2 / ts - ts / 2.0)
tc = t2 / 2.0 + tb / k


def sigma_c(N1, N2, Vsb=0.0):
    return (q * N1 / eps_s * (tc + ts / 2.0) * t1
            + q * N2 / eps_s * tc * t2 + Vsb)


def J7(K, lam=lam):
    """Exact elementary finite-L integral of [K cosh u + cosh(lam-u)]^7."""
    a = K + math.exp(-lam)
    b = K + math.exp(lam)
    total = 0.0
    for m in range(8):
        p = 2 * m - 7
        total += (math.comb(7, m) * a**m * b**(7-m)
                  * math.expm1(p * lam) / p)
    return total / 2.0**7


def sigma_from_K(K):
    return (lc**6 * math.sinh(lam)**7 / (A * J7(K)))**(1.0 / 7.0)

# Exact finite-length special points
S = math.sinh(lam)
P0 = S + S**3 + (3.0/5.0)*S**5 + (1.0/7.0)*S**7
sigma0 = (lc**6 * S**7 / (A * P0))**(1.0/7.0)  # K=0
s = math.sinh(lam/2.0)
c = math.cosh(lam/2.0)
P1 = s + s**3 + (3.0/5.0)*s**5 + (1.0/7.0)*s**7
J1 = 2.0**8 * c**7 * P1
sigma1 = (lc**6 * S**7 / (A * J1))**(1.0/7.0)  # K=1


def K_from_sigma(sig):
    """Physical K>=0 root. Root solving is equation solving, not fitting."""
    if sig <= 0.0 or sig > sigma0:
        return math.nan
    if abs(sig - sigma0) < 1e-11 * sigma0:
        return 0.0
    f = lambda K: sigma_from_K(K) - sig
    lo, hi = 0.0, 1.0
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Failed to bracket K")
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)


def bv_lateral_green(N1, N2):
    sig = sigma_c(N1, N2, Vsb)
    if sig > sigma0:
        return math.nan
    K = K_from_sigma(sig)
    return sig * (1.0 + K)


def bv_vertical():
    Ec = (1.0 / (A * (ts - tj)))**(1.0 / 7.0)
    return Ec * ((ts - tj) + tb / k)

BV_VF = bv_vertical()


def nbar(N1, N2):
    return (N1 * t1 + N2 * t2) / ts


def bv_overdoped_1d(N1, N2):
    """No-fit post-RESURF asymptote, clearly separate from Chung's depleted model."""
    N = nbar(N1, N2)
    return (1.0 / (2.0 * A))**0.25 * (eps_s / (q * N))**0.75


def bv_general(N1, N2):
    sig = sigma_c(N1, N2, Vsb)
    if sig <= sigma0:
        return min(bv_lateral_green(N1, N2), BV_VF), "Green RESURF"
    return bv_overdoped_1d(N1, N2), "1-D post-RESURF asymptote"


def ron_drift_W(N1, N2, mu1=1350.0, mu2=1350.0):
    """Drift-region width-normalized resistance only, no fit to Fig. 3."""
    return L / (q * (mu1 * N1 * t1 + mu2 * N2 * t2))


def N1_for_sigma(sig, N2):
    slope = q / eps_s * (tc + ts / 2.0) * t1
    offset = q / eps_s * N2 * tc * t2 + Vsb
    return (sig - offset) / slope

# Generate data
N1_grid = np.linspace(1.0e15, 1.6e16, 600)
N2_values = [1.0e14, 1.0e15]
rows = []
for N2 in N2_values:
    for N1 in N1_grid:
        bv, regime = bv_general(N1, N2)
        rows.append({
            "N1_cm^-3": N1,
            "N2_cm^-3": N2,
            "sigma_c_V": sigma_c(N1, N2),
            "BV_V": bv,
            "Rdrift_times_W_ohm_cm": ron_drift_W(N1, N2),
            "regime": regime,
        })

df = pd.DataFrame(rows)
outdir = Path('/mnt/data/fig3_green_function')
df.to_csv(outdir / 'fig3_green_function_data.csv', index=False)

fig, ax1 = plt.subplots(figsize=(8.3, 6.1))
ax2 = ax1.twinx()
for N2 in N2_values:
    d = df[df['N2_cm^-3'] == N2]
    x = d['N1_cm^-3'].to_numpy() / 1e15
    ax1.plot(x, d['BV_V'].to_numpy(), label=f"BV, N2={N2:.0e} cm^-3")
    ax2.plot(x, d['Rdrift_times_W_ohm_cm'].to_numpy(), linestyle='--',
             label=f"Rdrift W, N2={N2:.0e} cm^-3")

ax1.axhline(BV_VF, linestyle=':', linewidth=1.0, label='Vertical avalanche cap')
ax1.set_xlim(0, 16)
ax1.set_ylim(50, 300)
ax2.set_ylim(0, 120)
ax1.set_xlabel(r"Surface doping $N_1$ [$10^{15}$ cm$^{-3}$]")
ax1.set_ylabel("Breakdown voltage [V]")
ax2.set_ylabel(r"Drift resistance times width [$\Omega$ cm]")
ax1.set_title("No-fit Green-function analytical reconstruction of Fig. 3")
ax1.grid(True, alpha=0.25)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='best', fontsize=8)
fig.tight_layout()
fig.savefig(outdir / 'fig3_green_function.png', dpi=220, bbox_inches='tight')

print(f"lambda = {lam:.9f}")
print(f"paper-reported lc = {lc/um:.6f} um")
print(f"implied k (from paper lc, not fit) = {k:.9f}")
print(f"sigma0 exact finite-L (K=0) = {sigma0:.6f} V")
print(f"sigma1 exact finite-L (K=1) = {sigma1:.6f} V")
print(f"max lateral BV at K=1 = {2*sigma1:.6f} V")
print(f"vertical cap = {BV_VF:.6f} V")
for N2 in N2_values:
    print(f"N2={N2:.0e}: N1(K=1)={N1_for_sigma(sigma1,N2):.6e}, N1crit={N1_for_sigma(sigma0,N2):.6e}")
