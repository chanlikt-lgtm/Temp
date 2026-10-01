#!/usr/bin/env python3
"""
pde_efield_2d3d_three_cases.py
================================

Standalone NO-FIT analytical electric-field reconstruction for the
surface-implanted SOI RESURF LDMOS.

It produces 2-D |E(x,y)| maps and 3-D |E(x,y)| surfaces for three
physically selected surface-doping cases:

  1) LOW N1:
     first horizontal/vertical breakdown crossover (K > 1),
     drain-edge-dominated lateral surface field.

  2) MEDIUM N1:
     equal surface-edge fields at the actual device breakdown voltage,
     K = 1 with Vd = BV_vertical, so sigma_c = BV_vertical/2.

  3) HIGH N1:
     second horizontal/vertical breakdown crossover (0 < K < 1),
     gate-edge-dominated lateral surface field, close to the K=0
     full-depletion RESURF charge limit.

No Figure-3 point is fitted.

Analytical chain
----------------
1. Start with depleted 2-D Poisson equations in the two Si layers.
2. Use the parabolic vertical potential representation and the
   top/interface/BOX boundary conditions.
3. Obtain
       psi_s'' - psi_s/l_c^2 = -sigma_c/l_c^2.
4. Solve exactly using the finite-interval Dirichlet Green function.
5. Recover psi_f2 from the sign-audited interface relation.
6. Reconstruct phi_1(x,y1) and phi_2(x,y2) analytically.
7. Differentiate analytically:
       Ex = -dphi/dx,  Ey = -dphi/dy,
       |E| = sqrt(Ex^2 + Ey^2).
8. Evaluate each map at the actual analytical device breakdown voltage
       VBD = min(BV_horizontal, BV_vertical).

The reconstruction corresponds to the quasi-2-D fully depleted model.
It does not include the extra lateral depletion penetration into the
p-well or n+ drain, nor a TCAD field-plate geometry.

Outputs
-------
    efield_2d_three_cases.png
    efield_3d_three_cases.png
    efield_three_cases_summary.csv

Dependencies
------------
    numpy
    matplotlib
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. USER INPUTS
# =============================================================================

q = 1.602176634e-19
eps0 = 8.8541878128e-14       # F/cm
eps_s = 11.8 * eps0
eps_ox = 3.9 * eps0
A_F = 1.8e-35                 # Fulop alpha_eff = A_F E^7, E in V/cm
um = 1e-4                     # cm

L = 40.0 * um
t1 = 1.0 * um
t2 = 4.0 * um
tb = 2.0 * um
tj = 2.0 * um
Vsb = 0.0

# Background drift-layer donor concentration.
N2 = 1.0e14                   # cm^-3

# Spatial resolution.
NX = 401
NY1 = 61
NY2 = 181

OUT_2D = "efield_2d_three_cases.png"
OUT_3D = "efield_3d_three_cases.png"
OUT_CSV = "efield_three_cases_summary.csv"


# =============================================================================
# 2. GEOMETRY AND REDUCED PDE
# =============================================================================

@dataclass(frozen=True)
class Device:
    L: float
    t1: float
    t2: float
    tb: float
    tj: float
    Vsb: float
    eps_s: float
    eps_ox: float

    @property
    def ts(self):
        return self.t1 + self.t2

    @property
    def k(self):
        return self.eps_ox / self.eps_s

    @property
    def tc(self):
        return self.t2 / 2.0 + self.tb / self.k

    @property
    def D(self):
        return self.ts * (self.ts / 2.0 + self.tb / self.k)

    @property
    def alpha(self):
        return 1.0 / self.D

    @property
    def lc(self):
        return math.sqrt(self.D)

    @property
    def lam(self):
        return self.L / self.lc

    @property
    def dv(self):
        return self.ts - self.tj


dev = Device(L, t1, t2, tb, tj, Vsb, eps_s, eps_ox)


def beta_from_pde(N1):
    """
    Sign-audited RHS of
        psi_s'' - alpha psi_s = beta.
    """
    return (
        -q * N1 / dev.eps_s
        - (
            q * (N2 - N1) * dev.tc * dev.t2 / dev.eps_s
            + dev.Vsb
        ) / dev.D
    )


def sigma_c(N1):
    """
    sigma_c = -beta/alpha.

    Simplified:
      sigma_c = q/eps_s [
          N1 t1 (tc + ts/2) + N2 tc t2
      ] + Vsb.
    """
    s = (
        q / dev.eps_s
        * (
            N1 * dev.t1 * (dev.tc + dev.ts / 2.0)
            + N2 * dev.tc * dev.t2
        )
        + dev.Vsb
    )
    check = -beta_from_pde(N1) / dev.alpha
    if not math.isclose(s, check, rel_tol=2e-12, abs_tol=1e-10):
        raise RuntimeError("sigma_c sign audit failed.")
    return s


def N1_from_sigma(sig):
    return (
        (dev.eps_s / q) * (sig - dev.Vsb) - N2 * dev.tc * dev.t2
    ) / (
        dev.t1 * (dev.tc + dev.ts / 2.0)
    )


# =============================================================================
# 3. GREEN-FUNCTION SURFACE SOLUTION
# =============================================================================

def psi1_and_derivative(x, Vd, sig):
    """
    Surface potential psi_f1(x) and dpsi_f1/dx.
    """
    x = np.asarray(x, dtype=float)
    lc = dev.lc
    S = np.sinh(dev.lam)

    psi1 = (
        sig
        + (
            (Vd - sig) * np.sinh(x / lc)
            - sig * np.sinh((dev.L - x) / lc)
        ) / S
    )

    dpsi1 = (
        (Vd - sig) * np.cosh(x / lc)
        + sig * np.cosh((dev.L - x) / lc)
    ) / (lc * S)

    return psi1, dpsi1


def psi2_coefficients(N1):
    """
    Corrected interface relation derived directly from the two Poisson PDEs.

    psi_f2 = Cpsi * psi_f1 + C0

    The charge term is proportional to (N2-N1), not (N1-N2), if the
    Poisson equations and Eq. (17) are to be mutually consistent.
    """
    denom = dev.D
    Cpsi = (dev.tc * dev.ts + dev.t1 * dev.t2 / 2.0) / denom

    C0 = (
        q * (N2 - N1) * dev.t1**2 * dev.t2 * dev.tc
        / (2.0 * dev.eps_s)
        + dev.t1**2 * dev.Vsb / 2.0
    ) / denom

    return Cpsi, C0


def psi2_and_derivative(x, N1, Vd, sig):
    psi1, dpsi1 = psi1_and_derivative(x, Vd, sig)
    Cpsi, C0 = psi2_coefficients(N1)
    psi2 = Cpsi * psi1 + C0
    dpsi2 = Cpsi * dpsi1
    return psi1, dpsi1, psi2, dpsi2


# =============================================================================
# 4. EXACT HORIZONTAL AVALANCHE
# =============================================================================

def J7_closed(K):
    """
    Exact:
      int_0^lambda [K cosh(u)+cosh(lambda-u)]^7 du.
    """
    if K < 0.0:
        raise ValueError("K must be >= 0 on the full-depletion branch.")

    lam = dev.lam
    a = K + math.exp(-lam)
    b = K + math.exp(+lam)

    s = 0.0
    for m in range(8):
        p = 2*m - 7
        s += (
            math.comb(7, m)
            * a**m
            * b**(7-m)
            * math.expm1(p*lam)
            / p
        )
    return s / 2.0**7


def sigma_bd_from_K(K):
    return (
        dev.lc**6 * math.sinh(dev.lam)**7
        / (A_F * J7_closed(K))
    )**(1.0/7.0)


def BVH_from_K(K):
    s = sigma_bd_from_K(K)
    return s * (1.0 + K)


def root_bisect(func, lo, hi, n=120):
    flo = func(lo)
    fhi = func(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo*fhi > 0:
        raise ValueError("Root is not bracketed.")
    for _ in range(n):
        mid = 0.5*(lo+hi)
        fm = func(mid)
        if flo*fm <= 0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return 0.5*(lo+hi)


# =============================================================================
# 5. VERTICAL BREAKDOWN AND THE THREE PHYSICAL CASES
# =============================================================================

def vertical_breakdown():
    Ec = (1.0 / (A_F * dev.dv))**(1.0/7.0)
    BVv = dev.Vsb + Ec * (dev.dv + dev.tb/dev.k)
    return BVv, Ec


def crossover_K_roots(BVv):
    f = lambda K: BVH_from_K(K) - BVv

    roots = []

    # high-dose branch 0 <= K <= 1
    if f(0.0)*f(1.0) <= 0:
        roots.append(root_bisect(f, 0.0, 1.0))

    # low-dose branch K >= 1
    if f(1.0) >= 0:
        hi = 2.0
        while f(hi) > 0 and hi < 1e10:
            hi *= 2.0
        if hi < 1e10:
            roots.append(root_bisect(f, 1.0, hi))

    roots.sort()
    return roots


def make_cases():
    BVv, Ec = vertical_breakdown()
    roots = crossover_K_roots(BVv)
    if len(roots) != 2:
        raise RuntimeError(
            "Expected two horizontal/vertical crossover roots for nominal geometry."
        )

    K_highdose = roots[0]       # K < 1
    K_lowdose = roots[1]        # K > 1

    sig_low = sigma_bd_from_K(K_lowdose)
    sig_high = sigma_bd_from_K(K_highdose)

    # At actual device breakdown Vd=BVv, equal surface-edge fields require K=1:
    # K = Vd/sigma - 1 = 1 -> sigma = Vd/2.
    sig_mid = BVv / 2.0

    cases = [
        {
            "name": "Low N1: drain-edge dominated",
            "short": "low",
            "K": K_lowdose,
            "sigma": sig_low,
            "N1": N1_from_sigma(sig_low),
            "Vd": BVv,
            "mechanism": "horizontal = vertical crossover",
        },
        {
            "name": "Medium N1: equal surface-edge fields",
            "short": "medium",
            "K": 1.0,
            "sigma": sig_mid,
            "N1": N1_from_sigma(sig_mid),
            "Vd": BVv,
            "mechanism": "vertical breakdown; K=1 at VBD",
        },
        {
            "name": "High N1: gate-edge dominated",
            "short": "high",
            "K": K_highdose,
            "sigma": sig_high,
            "N1": N1_from_sigma(sig_high),
            "Vd": BVv,
            "mechanism": "horizontal = vertical crossover",
        },
    ]

    return cases, BVv, Ec


# =============================================================================
# 6. RECONSTRUCT phi_1, phi_2 AND ANALYTIC E-FIELD
# =============================================================================

def field_map(N1, Vd):
    """
    Returns x [cm], y [cm], phi [V], Ex [V/cm], Ey [V/cm], |E| [V/cm].
    """
    sig = sigma_c(N1)

    x = np.linspace(0.0, dev.L, NX)
    y1 = np.linspace(0.0, dev.t1, NY1)
    # Avoid duplicating interface row in layer 2:
    y2 = np.linspace(0.0, dev.t2, NY2)[1:]

    psi1, dpsi1, psi2, dpsi2 = psi2_and_derivative(x, N1, Vd, sig)

    # ---------------- layer 1 ----------------
    X1, Y1 = np.meshgrid(x, y1)
    r = Y1 / dev.t1

    PSI1 = psi1[None, :]
    DPSI1 = dpsi1[None, :]
    PSI2 = psi2[None, :]
    DPSI2 = dpsi2[None, :]

    phi1 = PSI1*(1.0-r**2) + PSI2*r**2
    dphidx1 = DPSI1*(1.0-r**2) + DPSI2*r**2
    dphidy1 = 2.0*Y1/dev.t1**2 * (PSI2-PSI1)

    Ex1 = -dphidx1
    Ey1 = -dphidy1

    # ---------------- layer 2 ----------------
    X2, Y2local = np.meshgrid(x, y2)

    A2 = (dev.ts + 2.0*dev.tc) / (2.0*dev.tc*dev.t1*dev.t2)
    B2 = (2.0*dev.tc + dev.t2) / (2.0*dev.tc*dev.t1*dev.t2)

    F2 = 1.0 + 2.0*Y2local/dev.t1 - A2*Y2local**2
    G2 = 2.0*Y2local/dev.t1 - B2*Y2local**2

    phi2 = (
        PSI2*F2
        - PSI1*G2
        + Y2local**2 * dev.Vsb/(2.0*dev.tc*dev.t2)
    )

    dphidx2 = DPSI2*F2 - DPSI1*G2

    dF2dy = 2.0/dev.t1 - 2.0*A2*Y2local
    dG2dy = 2.0/dev.t1 - 2.0*B2*Y2local

    dphidy2 = (
        PSI2*dF2dy
        - PSI1*dG2dy
        + Y2local*dev.Vsb/(dev.tc*dev.t2)
    )

    Ex2 = -dphidx2
    Ey2 = -dphidy2

    # Global y coordinate measured downward from top Si surface.
    y_global = np.concatenate([y1, dev.t1 + y2])

    phi = np.vstack([phi1, phi2])
    Ex = np.vstack([Ex1, Ex2])
    Ey = np.vstack([Ey1, Ey2])
    Emag = np.sqrt(Ex**2 + Ey**2)

    return x, y_global, phi, Ex, Ey, Emag


# =============================================================================
# 7. PLOTS
# =============================================================================

def plot_2d(cases):
    maps = []
    global_max = 0.0

    for c in cases:
        m = field_map(c["N1"], c["Vd"])
        maps.append(m)
        global_max = max(global_max, float(np.nanmax(m[-1])))

    fig, axes = plt.subplots(3, 1, figsize=(11.4, 10.2), sharex=True)

    last = None
    for ax, c, m in zip(axes, cases, maps):
        x, y, phi, Ex, Ey, E = m
        Xum, Yum = np.meshgrid(x/um, y/um)

        # Common scale across all three cases; field displayed in kV/cm.
        levels = np.linspace(0.0, global_max/1e3, 60)
        last = ax.contourf(Xum, Yum, E/1e3, levels=levels)

        # Sparse analytic field vectors; normalized so arrows show direction.
        sx = slice(0, len(x), 28)
        sy = slice(0, len(y), 18)
        Exs = Ex[sy, sx]
        Eys = Ey[sy, sx]
        norms = np.sqrt(Exs**2 + Eys**2)
        norms[norms == 0] = 1.0

        ax.quiver(
            Xum[sy, sx], Yum[sy, sx],
            Exs/norms, Eys/norms,
            angles="xy", scale_units="xy", scale=1.35,
            width=0.0018,
        )

        ax.axhline(dev.t1/um, linewidth=0.9)
        ax.set_ylim(dev.ts/um, 0.0)
        ax.set_ylabel("Depth y [um]")
        ax.set_title(
            f'{c["name"]}: N1={c["N1"]:.3e} cm^-3, '
            f'VBD={c["Vd"]:.2f} V, K={c["K"]:.3f}'
        )

        # Store map maxima.
        c["Emax_map"] = float(np.nanmax(E))
        idx = np.unravel_index(np.nanargmax(E), E.shape)
        c["Emax_x_um"] = float(x[idx[1]]/um)
        c["Emax_y_um"] = float(y[idx[0]]/um)

    axes[-1].set_xlabel("Drift coordinate x [um]")
    fig.suptitle(
        "No-fit analytical 2-D electric-field magnitude at device breakdown",
        y=0.985,
    )
    # Reserve a dedicated colorbar column so it never overlaps a panel.
    fig.subplots_adjust(left=0.08, right=0.84, top=0.93, bottom=0.07, hspace=0.32)
    cax = fig.add_axes([0.87, 0.13, 0.025, 0.72])
    cbar = fig.colorbar(last, cax=cax)
    cbar.set_label("|E| [kV/cm]")
    fig.savefig(OUT_2D, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_3d(cases):
    maps = [field_map(c["N1"], c["Vd"]) for c in cases]
    global_max = max(float(np.nanmax(m[-1])) for m in maps) / 1e3

    fig = plt.figure(figsize=(15.2, 5.4))

    for i, (c, m) in enumerate(zip(cases, maps), start=1):
        x, y, phi, Ex, Ey, E = m

        # Downsample only for rendering; calculation remains at full resolution.
        Xum, Yum = np.meshgrid(x[::4]/um, y[::3]/um)
        Z = E[::3, ::4]/1e3

        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(Xum, Yum, Z, linewidth=0, antialiased=True)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        ax.set_zlabel("|E| [kV/cm]")
        ax.set_zlim(0.0, global_max)
        ax.set_title(
            f'{c["short"].capitalize()} N1\n'
            f'N1={c["N1"]/1e15:.2f}e15 cm^-3, K={c["K"]:.2f}'
        )
        ax.view_init(elev=28, azim=-128)

    fig.suptitle(
        "No-fit analytical 3-D surfaces of |E(x,y)| at device breakdown",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_3D, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 8. RUN
# =============================================================================

def main():
    cases, BVv, Ec = make_cases()

    print("="*84)
    print("NO-FIT 2-D/3-D ELECTRIC-FIELD RECONSTRUCTION")
    print("="*84)
    print(f"k = eps_ox/eps_s = {dev.k:.8f}")
    print(f"l_c = {dev.lc/um:.6f} um")
    print(f"L/l_c = {dev.lam:.6f}")
    print(f"Vertical BV = {BVv:.6f} V")
    print(f"Vertical Ec = {Ec:.6e} V/cm")
    print()

    for c in cases:
        # Confirm K from Vd/sigma-1 for all three cases.
        Kcheck = c["Vd"]/c["sigma"] - 1.0
        print(c["name"])
        print(f"  N1      = {c['N1']:.8e} cm^-3")
        print(f"  sigma_c = {c['sigma']:.8f} V")
        print(f"  Vd=VBD  = {c['Vd']:.8f} V")
        print(f"  K       = {c['K']:.8f} (direct check {Kcheck:.8f})")
        print(f"  regime  = {c['mechanism']}")
        print()

    plot_2d(cases)
    plot_3d(cases)

    # CSV summary after plotting (contains field-map maxima).
    fields = [
        "case", "N1_cm^-3", "N2_cm^-3", "sigma_c_V", "VBD_V", "K",
        "mechanism", "Emax_map_V_per_cm", "Emax_x_um", "Emax_y_um",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cases:
            w.writerow({
                "case": c["name"],
                "N1_cm^-3": c["N1"],
                "N2_cm^-3": N2,
                "sigma_c_V": c["sigma"],
                "VBD_V": c["Vd"],
                "K": c["K"],
                "mechanism": c["mechanism"],
                "Emax_map_V_per_cm": c["Emax_map"],
                "Emax_x_um": c["Emax_x_um"],
                "Emax_y_um": c["Emax_y_um"],
            })

    print("Field-map maxima:")
    for c in cases:
        print(
            f"  {c['short']:>6s}: "
            f"|E|max={c['Emax_map']/1e3:.3f} kV/cm "
            f"at (x,y)=({c['Emax_x_um']:.3f}, {c['Emax_y_um']:.3f}) um"
        )

    print()
    print("Saved:")
    print(f"  {Path(OUT_2D).resolve()}")
    print(f"  {Path(OUT_3D).resolve()}")
    print(f"  {Path(OUT_CSV).resolve()}")


if __name__ == "__main__":
    main()
