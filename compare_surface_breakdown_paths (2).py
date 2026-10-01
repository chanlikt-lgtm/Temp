#!/usr/bin/env python3
"""
compare_surface_breakdown_paths.py

No-fit comparison of:
  (1) parabolic quasi-2-D closed-form surface-breakdown model
  (2) direct 2-D finite-difference Poisson solution
plus the independent 1-D vertical avalanche cap.

Outputs:
  surface_breakdown_parabolic_vs_direct_fd.png
  surface_breakdown_comparison.csv

Dependencies:
  numpy, scipy, matplotlib
"""

from __future__ import annotations
import csv, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------
# 1. Physical/device parameters -- NO FITTING
# ---------------------------------------------------------------------
q = 1.602176634e-19
eps0 = 8.8541878128e-14       # F/cm
eps_s = 11.8 * eps0
eps_ox = 3.9 * eps0
k = eps_ox / eps_s
A_F = 1.8e-35                 # Fulop: alpha_eff = A_F E^7, E in V/cm
um = 1e-4                     # cm

L  = 40.0 * um
t1 = 1.0 * um
t2 = 4.0 * um
ts = t1 + t2
tb = 2.0 * um
tj = 2.0 * um
Vsb = 0.0
N2 = 1.0e14                   # cm^-3

# Direct-PDE mesh
NX, NY = 161, 51
V_SEARCH_MAX = 1200.0

# Sweep
N1_MIN = 1.0e14
N1_MAX = 1.6e16

OUT_PNG = "surface_breakdown_parabolic_vs_direct_fd.png"
OUT_CSV = "surface_breakdown_comparison.csv"

# ---------------------------------------------------------------------
# 2. Parabolic quasi-2-D model
# ---------------------------------------------------------------------
tc = t2 / 2.0 + tb / k
D = ts * (ts / 2.0 + tb / k)
lc = math.sqrt(D)
lam = L / lc

def sigma_c(N1):
    return (q/eps_s) * (
        N1*t1*(tc + ts/2.0) + N2*tc*t2
    ) + Vsb

def J7_closed(K):
    """Exact finite-L avalanche integral."""
    a = K + math.exp(-lam)
    b = K + math.exp(+lam)
    total = 0.0
    for m in range(8):
        p = 2*m - 7
        total += (
            math.comb(7, m) * a**m * b**(7-m)
            * math.expm1(p*lam) / p
        )
    return total / 2.0**7

def sigma_bd_from_K(K):
    return (
        lc**6 * math.sinh(lam)**7 /
        (A_F * J7_closed(K))
    )**(1.0/7.0)

sigma0 = sigma_bd_from_K(0.0)

def K_from_sigma(sig):
    """
    Physical K>=0 solution of sigma_bd(K)=sig.
    None means the fully depleted quasi-2-D branch has ended.
    """
    if sig <= 0.0 or sig > sigma0*(1+1e-12):
        return None
    if math.isclose(sig, sigma0, rel_tol=1e-12, abs_tol=1e-12):
        return 0.0

    lo, hi = 0.0, 1.0
    while sigma_bd_from_K(hi) > sig:
        hi *= 2.0
        if hi > 1e12:
            return None

    for _ in range(110):
        mid = 0.5*(lo+hi)
        if sigma_bd_from_K(mid) > sig:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

def BV_quasi(N1):
    sig = sigma_c(N1)
    K = K_from_sigma(sig)
    if K is None:
        return math.nan
    return sig*(1.0+K)

# ---------------------------------------------------------------------
# 3. Independent 1-D vertical cap
# ---------------------------------------------------------------------
dv = ts - tj
Ec_v = (1.0/(A_F*dv))**(1.0/7.0)
BV_vertical = Vsb + Ec_v*(dv + tb/k)

# ---------------------------------------------------------------------
# 4. Direct 2-D finite-difference Poisson solver
# ---------------------------------------------------------------------
x = np.linspace(0.0, L, NX)
y = np.linspace(0.0, ts, NY)
dx = x[1] - x[0]
dy = y[1] - y[0]
gamma_box = k/tb

j_int = int(np.argmin(np.abs(y-t1)))
if abs(y[j_int]-t1) > 1e-12:
    raise RuntimeError("Choose NY so that t1 is exactly a y-grid row.")

def node(j, i):
    return j*NX+i

def build_A():
    rr, cc, vv = [], [], []

    for j in range(NY):
        for i in range(NX):
            p = node(j, i)

            # Lateral Dirichlet sidewalls
            if i == 0 or i == NX-1:
                rr.append(p); cc.append(p); vv.append(1.0)
                continue

            # Top Neumann: dphi/dy = 0
            if j == 0:
                for c, v in [
                    (node(0,i), -3.0/(2*dy)),
                    (node(1,i),  4.0/(2*dy)),
                    (node(2,i), -1.0/(2*dy)),
                ]:
                    rr.append(p); cc.append(c); vv.append(v)
                continue

            # Bottom BOX Robin:
            # dphi/dy + gamma*phi = gamma*Vsb
            if j == NY-1:
                for c, v in [
                    (node(NY-1,i),  3.0/(2*dy) + gamma_box),
                    (node(NY-2,i), -4.0/(2*dy)),
                    (node(NY-3,i),  1.0/(2*dy)),
                ]:
                    rr.append(p); cc.append(c); vv.append(v)
                continue

            # Interior 5-point Laplacian
            stencil = [
                (node(j,i),   -2.0/dx**2 - 2.0/dy**2),
                (node(j,i-1),  1.0/dx**2),
                (node(j,i+1),  1.0/dx**2),
                (node(j-1,i),  1.0/dy**2),
                (node(j+1,i),  1.0/dy**2),
            ]
            for c, v in stencil:
                rr.append(p); cc.append(c); vv.append(v)

    return sparse.csr_matrix((vv,(rr,cc)), shape=(NX*NY,NX*NY))

A = build_A()
LU = splu(A.tocsc())

# Piecewise source basis. At the exact interface use half weight for each side.
w1 = np.zeros(NY)
w2 = np.zeros(NY)
for j in range(NY):
    if j < j_int:
        w1[j] = 1.0
    elif j > j_int:
        w2[j] = 1.0
    else:
        w1[j] = w2[j] = 0.5

N_SCALE = 1e15

def rhs(Vd=0.0, Vsb_value=0.0, N1_value=0.0, N2_value=0.0):
    b = np.zeros(NX*NY)
    for j in range(NY):
        for i in range(NX):
            p = node(j,i)

            if i == 0:
                b[p] = 0.0
            elif i == NX-1:
                b[p] = Vd
            elif j == 0:
                b[p] = 0.0
            elif j == NY-1:
                b[p] = gamma_box*Vsb_value
            else:
                Nd = N1_value*w1[j] + N2_value*w2[j]
                b[p] = -q*Nd/eps_s
    return b

def solve_basis(**kwargs):
    return LU.solve(rhs(**kwargs)).reshape(NY,NX)

# Solve basis PDEs exactly once.
phi_Vd  = solve_basis(Vd=1.0)
phi_Vsb = solve_basis(Vsb_value=1.0)
phi_N1  = solve_basis(N1_value=N_SCALE)
phi_N2  = solve_basis(N2_value=N_SCALE)

def Ex_of(phi):
    return -np.gradient(phi, dx, axis=1, edge_order=2)

Ex_Vd  = Ex_of(phi_Vd)
Ex_Vsb = Ex_of(phi_Vsb)
Ex_N1  = Ex_of(phi_N1)
Ex_N2  = Ex_of(phi_N2)

def surface_integral_fd(N1, Vd):
    r1 = N1/N_SCALE
    r2 = N2/N_SCALE
    Ex_top = (
        Vd*Ex_Vd[0,:]
        + Vsb*Ex_Vsb[0,:]
        + r1*Ex_N1[0,:]
        + r2*Ex_N2[0,:]
    )
    return A_F*np.trapezoid(np.abs(Ex_top)**7, x)

def BV_direct_fd(N1):
    """
    Find the high-voltage root of I_H(Vd,N1)=1.
    If the minimum ionization integral is already >1, the fully depleted
    direct-PDE branch has no subcritical state and is marked NaN.
    """
    f = lambda V: surface_integral_fd(N1, V)

    opt = minimize_scalar(
        f, bounds=(0.0, V_SEARCH_MAX), method="bounded",
        options={"xatol":1e-8}
    )
    Vmin = float(opt.x)
    Imin = float(opt.fun)

    if Imin > 1.0:
        return math.nan, Vmin, Imin

    lo = Vmin
    hi = max(Vmin+25.0, 50.0)
    while f(hi) < 1.0 and hi < 4*V_SEARCH_MAX:
        hi *= 1.5

    if f(hi) < 1.0:
        return math.nan, Vmin, Imin

    for _ in range(100):
        mid = 0.5*(lo+hi)
        if f(mid) <= 1.0:
            lo = mid
        else:
            hi = mid

    return 0.5*(lo+hi), Vmin, Imin

# ---------------------------------------------------------------------
# 5. Simulate both curves
# ---------------------------------------------------------------------
Nq = np.logspace(np.log10(2.5e14), np.log10(N1_MAX), 650)
BVq = np.array([BV_quasi(v) for v in Nq])

Nd = np.linspace(N1_MIN, 1.2e16, 61)  # uniform physical doping sweep
BVd = []
Vmins = []
Imins = []
for v in Nd:
    b, vm, im = BV_direct_fd(float(v))
    BVd.append(b); Vmins.append(vm); Imins.append(im)
BVd = np.asarray(BVd, float)
Vmins = np.asarray(Vmins, float)
Imins = np.asarray(Imins, float)

# Quasi characteristic points
sigma1 = sigma_bd_from_K(1.0)
N1_K1 = (
    (eps_s/q)*(sigma1-Vsb) - N2*tc*t2
) / (t1*(tc+ts/2.0))
N1_K0 = (
    (eps_s/q)*(sigma0-Vsb) - N2*tc*t2
) / (t1*(tc+ts/2.0))
BVq_max = 2.0*sigma1

# Direct maximum
valid = np.isfinite(BVd)
idx_max_fd = np.nanargmax(BVd)
N1_fd_max = Nd[idx_max_fd]
BV_fd_max = BVd[idx_max_fd]

# ---------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 6.5))

ax.plot(
    Nq, BVq,
    linewidth=2.3,
    label="parabolic quasi-2-D (closed form)"
)

ax.plot(
    Nd[valid], BVd[valid],
    marker="o",
    markersize=4.5,
    linewidth=1.7,
    label="direct 2-D finite difference"
)

ax.axhline(
    BV_vertical,
    linestyle="--",
    linewidth=1.4,
    label=f"1-D vertical cap = {BV_vertical:.1f} V"
)

ax.set_xscale("log")
ax.set_xlim(8e13, 1.75e16)
ax.set_ylim(0, 425)
ax.set_xlabel(r"Surface doping $N_1$ [cm$^{-3}$]")
ax.set_ylabel(r"Horizontal surface $V_{BD}$ [V]")
ax.set_title("Surface breakdown path: parabolic vs direct 2-D PDE")
ax.grid(True, alpha=0.28)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# 7. CSV
# ---------------------------------------------------------------------
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "model","N1_cm^-3","BV_horizontal_V",
        "Vmin_surface_integral_V","min_surface_integral"
    ])
    for n, b in zip(Nq, BVq):
        w.writerow(["quasi2D_closed_form", n, b, "", ""])
    for n, b, vm, im in zip(Nd, BVd, Vmins, Imins):
        w.writerow(["direct2D_finite_difference", n, b, vm, im])

print("="*78)
print("SURFACE BREAKDOWN COMPARISON -- NO FITTING")
print("="*78)
print(f"k = {k:.8f}")
print(f"lc = {lc/um:.6f} um")
print(f"L/lc = {lam:.6f}")
print(f"Vertical cap = {BV_vertical:.6f} V")
print()
print("Parabolic quasi-2-D:")
print(f"  K=1 intrinsic maximum: N1 = {N1_K1:.6e} cm^-3")
print(f"  BV_H,max = {BVq_max:.6f} V")
print(f"  K=0 full-depletion endpoint: N1 = {N1_K0:.6e} cm^-3")
print(f"  endpoint BV_H = sigma0 = {sigma0:.6f} V")
print()
print("Direct 2-D finite difference:")
print(f"  sampled maximum: N1 = {N1_fd_max:.6e} cm^-3")
print(f"  sampled BV_H,max = {BV_fd_max:.6f} V")
if np.any(~valid):
    first_bad = Nd[np.where(~valid)[0][0]]
    print(f"  first sampled no-subcritical-state point: N1 = {first_bad:.6e} cm^-3")
print()
print("Saved:", Path(OUT_PNG).resolve())
print("Saved:", Path(OUT_CSV).resolve())
