#!/usr/bin/env python3
"""
pde_to_vbd_sweep.py
===================

Standalone, NO-FIT analytical model from the depleted 2-D Poisson equations
to breakdown voltage V_BD versus surface implant concentration N1 for a
surface-implanted SOI RESURF LDMOS.

Run:
    python pde_to_vbd_sweep.py

Outputs:
    vbd_from_pde_sweep.png
    vbd_from_pde_sweep.csv

Dependencies:
    numpy
    matplotlib

NO FITTING:
    - no digitized Fig. 3 points
    - no regression parameters
    - no fitted critical electric field
    - avalanche uses Fulop alpha_eff = A_F E^7
    - geometry/material parameters are explicit inputs

Model chain
-----------
Original depleted 2-D Poisson equations:

    d2(phi_1)/dx2 + d2(phi_1)/dy1^2 = -q N1 / eps_s
    d2(phi_2)/dx2 + d2(phi_2)/dy2^2 = -q N2 / eps_s

Using Chung's parabolic vertical-potential reduction and the interface/BOX
boundary conditions gives the surface equation

    psi_s'' - alpha psi_s = beta

with

    D     = t_s (t_s/2 + t_b/k)
    alpha = 1/D
    l_c   = sqrt(D)

and the sign-consistent characteristic surface potential

    sigma_c = -beta/alpha
            = q/eps_s * [
                  N1*t1*(t_c + t_s/2)
                + N2*t_c*t2
              ] + V_sb,

where

    t_s = t1 + t2,
    t_c = t2/2 + t_b/k,
    k   = eps_ox/eps_s.

Equivalently,

    psi_s'' - psi_s/l_c^2 = -sigma_c/l_c^2.

The Dirichlet Green-function solution for psi_s(0)=0, psi_s(L)=V_d is

    psi_s(x) = sigma_c
             + [(V_d-sigma_c)sinh(x/l_c)
                -sigma_c sinh((L-x)/l_c)]/sinh(L/l_c).

Therefore the lateral field magnitude is

    F(x) = sigma_c/[l_c sinh(lambda)]
           * [K cosh(x/l_c) + cosh((L-x)/l_c)],

    lambda = L/l_c,
    K      = V_d/sigma_c - 1.

Horizontal avalanche uses Fulop's law

    alpha_eff = A_F F^7

and

    integral_0^L alpha_eff dx = 1.

Defining u=x/l_c,

    J7(K,lambda)
      = integral_0^lambda [K cosh(u)+cosh(lambda-u)]^7 du,

the finite-length avalanche equation is

    A_F sigma_c^7 J7 / [l_c^6 sinh(lambda)^7] = 1.

J7 is evaluated here in exact closed form:

    J7 = 2^-7 sum_{m=0}^7 C(7,m)
         a^m b^(7-m) [exp((2m-7)lambda)-1]/(2m-7),

    a = K + exp(-lambda),
    b = K + exp(+lambda).

For each N1, sigma_c is known explicitly.  We solve the above algebraic
avalanche condition for its physical root K >= 0 by bisection
(root finding is NOT parameter fitting), and then

    BV_H = sigma_c (1+K).

Vertical breakdown under the drain follows the paper's low-doping 1-D
approximation.  Let

    d_v = t_s - t_j.

The silicon field is approximately constant at E_c, the BOX field is E_c/k,
and Fulop breakdown gives

    A_F E_c^7 d_v = 1
    E_c = [1/(A_F d_v)]^(1/7),

so

    BV_V = V_sb + E_c (d_v + t_b/k).

Within the fully depleted RESURF domain,

    V_BD = min(BV_H, BV_V).

The horizontal full-depletion branch exists only for K >= 0.  Its upper
charge limit is K=0.  Beyond that point this reduced PDE model is not valid.
An OPTIONAL, separately labeled 1-D post-RESURF asymptote is included below,
but it is OFF by default.

Reference basis:
    S.-K. Chung, IEEE Trans. Electron Devices 47(5), 1006-1009 (2000).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. USER INPUTS
# ============================================================================

# Material constants
Q = 1.602176634e-19              # C
EPS0 = 8.8541878128e-14          # F/cm
EPS_S = 11.8 * EPS0              # silicon permittivity, F/cm
EPS_OX = 3.9 * EPS0              # SiO2 permittivity, F/cm
A_F = 1.8e-35                    # Fulop: alpha_eff = A_F * E^7, E in V/cm

UM = 1e-4                        # 1 um in cm

# Paper-like nominal geometry
L_UM = 40.0                      # drift length
T1_UM = 1.0                      # implanted surface-layer thickness
T2_UM = 4.0                      # lower lightly doped SOI thickness
TB_UM = 2.0                      # buried oxide thickness
TJ_UM = 2.0                      # n+ drain junction depth
VSB = 0.0                        # substrate bias [V]

# Background doping curves to plot.
# Use [1e14] for a single curve, or [1e14, 1e15] to reproduce both Fig.-3 cases.
N2_VALUES = [1e14, 1e15]         # cm^-3

# Surface implant sweep
N1_MIN = 1.0e15                  # cm^-3
N1_MAX = 1.6e16                  # cm^-3
N1_POINTS = 600

# Optional continuation after full-depletion RESURF fails.
# False = mathematically strict reduced-PDE result (recommended).
# True  = additionally show a no-fit 1-D one-sided-junction asymptote.
USE_POST_RESURF_1D = False

# Output names
OUT_PNG = "vbd_from_pde_sweep.png"
OUT_CSV = "vbd_from_pde_sweep.csv"


# ============================================================================
# 2. GEOMETRY / PDE REDUCTION
# ============================================================================

@dataclass(frozen=True)
class Device:
    L: float
    t1: float
    t2: float
    tb: float
    tj: float
    Vsb: float
    eps_s: float = EPS_S
    eps_ox: float = EPS_OX

    @property
    def ts(self) -> float:
        return self.t1 + self.t2

    @property
    def k(self) -> float:
        return self.eps_ox / self.eps_s

    @property
    def tc(self) -> float:
        return self.t2 / 2.0 + self.tb / self.k

    @property
    def D(self) -> float:
        # D = 1/alpha = l_c^2
        return self.ts * (self.ts / 2.0 + self.tb / self.k)

    @property
    def alpha(self) -> float:
        return 1.0 / self.D

    @property
    def lc(self) -> float:
        return math.sqrt(self.D)

    @property
    def lam(self) -> float:
        return self.L / self.lc

    @property
    def dv(self) -> float:
        # vertical depleted Si distance under drain
        return self.ts - self.tj


DEV = Device(
    L=L_UM * UM,
    t1=T1_UM * UM,
    t2=T2_UM * UM,
    tb=TB_UM * UM,
    tj=TJ_UM * UM,
    Vsb=VSB,
)


def beta_from_pde(N1: float, N2: float, dev: Device = DEV) -> float:
    """
    RHS beta in
        psi_s'' - alpha*psi_s = beta

    Sign-consistent form obtained by direct elimination from the two
    depleted Poisson equations and the interface/BOX conditions.
    Units: V/cm^2.
    """
    return (
        -Q * N1 / dev.eps_s
        - (
            Q * (N2 - N1) * dev.tc * dev.t2 / dev.eps_s
            + dev.Vsb
        ) / dev.D
    )


def sigma_c(N1: float, N2: float, dev: Device = DEV) -> float:
    """
    Characteristic surface potential:
        sigma_c = -beta/alpha.

    Algebraically simplified closed form:
        sigma_c = q/eps_s [
                    N1 t1 (tc + ts/2)
                    + N2 tc t2
                  ] + Vsb.
    """
    s_direct = (
        Q / dev.eps_s
        * (
            N1 * dev.t1 * (dev.tc + dev.ts / 2.0)
            + N2 * dev.tc * dev.t2
        )
        + dev.Vsb
    )

    # Internal sign-audit check against -beta/alpha.
    s_from_beta = -beta_from_pde(N1, N2, dev) / dev.alpha
    if not math.isclose(s_direct, s_from_beta, rel_tol=2e-12, abs_tol=1e-10):
        raise RuntimeError(
            "PDE sign audit failed: sigma_c != -beta/alpha. "
            f"{s_direct=} {s_from_beta=}"
        )
    return s_direct


def N1_from_sigma(sig: float, N2: float, dev: Device = DEV) -> float:
    """Invert the linear sigma_c(N1,N2) relation."""
    numerator = (dev.eps_s / Q) * (sig - dev.Vsb) - N2 * dev.tc * dev.t2
    denominator = dev.t1 * (dev.tc + dev.ts / 2.0)
    return numerator / denominator


# ============================================================================
# 3. GREEN-FUNCTION SURFACE SOLUTION
# ============================================================================

def psi_surface(x: np.ndarray | float, Vd: float, sig: float,
                dev: Device = DEV) -> np.ndarray:
    """
    Exact solution of
        psi'' - psi/lc^2 = -sig/lc^2
    with psi(0)=0, psi(L)=Vd.

    This is the same solution obtained from the Dirichlet Green function.
    """
    x = np.asarray(x, dtype=float)
    lc = dev.lc
    lam = dev.lam
    return (
        sig
        + (
            (Vd - sig) * np.sinh(x / lc)
            - sig * np.sinh((dev.L - x) / lc)
        ) / np.sinh(lam)
    )


def field_surface(x: np.ndarray | float, Vd: float, sig: float,
                  dev: Device = DEV) -> np.ndarray:
    """
    Magnitude of the lateral surface field, |d psi_s / dx|.
    """
    x = np.asarray(x, dtype=float)
    K = Vd / sig - 1.0
    return (
        sig
        / (dev.lc * np.sinh(dev.lam))
        * (
            K * np.cosh(x / dev.lc)
            + np.cosh((dev.L - x) / dev.lc)
        )
    )


# ============================================================================
# 4. EXACT CLOSED-FORM HORIZONTAL AVALANCHE
# ============================================================================

def J7_closed(K: float, lam: float) -> float:
    """
    Exact analytic integral

      J7(K,lam) = int_0^lam [K cosh(u)+cosh(lam-u)]^7 du.

    No numerical quadrature is used.
    """
    if K < 0.0:
        raise ValueError("Physical full-depletion branch requires K >= 0.")

    a = K + math.exp(-lam)
    b = K + math.exp(+lam)

    s = 0.0
    for m in range(8):
        p = 2 * m - 7                  # never zero for power 7
        integ_exp = math.expm1(p * lam) / p
        s += math.comb(7, m) * (a ** m) * (b ** (7 - m)) * integ_exp
    return s / (2.0 ** 7)


def sigma_at_breakdown_from_K(K: float, dev: Device = DEV) -> float:
    """
    From Fulop avalanche:
      A_F sig^7 J7 / [lc^6 sinh(lam)^7] = 1

    Therefore
      sig(K) = [lc^6 sinh(lam)^7 / (A_F J7)]^(1/7).
    """
    J = J7_closed(K, dev.lam)
    return (
        dev.lc ** 6
        * math.sinh(dev.lam) ** 7
        / (A_F * J)
    ) ** (1.0 / 7.0)


def bv_horizontal_from_K(K: float, dev: Device = DEV) -> float:
    sig = sigma_at_breakdown_from_K(K, dev)
    return sig * (1.0 + K)


def K_from_sigma(sig: float, dev: Device = DEV) -> float:
    """
    Solve sig(K)=sig for K>=0 by monotonic bisection.

    This is algebraic root finding, NOT fitting.
    """
    sig0 = sigma_at_breakdown_from_K(0.0, dev)

    if sig <= 0.0:
        return math.nan
    if sig > sig0 * (1.0 + 5e-13):
        # Above K=0 charge limit: full-depletion RESURF solution does not exist.
        return math.nan
    if math.isclose(sig, sig0, rel_tol=5e-13, abs_tol=1e-12):
        return 0.0

    lo = 0.0
    hi = 1.0

    # sig(K) decreases monotonically as K increases.
    while sigma_at_breakdown_from_K(hi, dev) > sig:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Could not bracket physical K root.")

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if sigma_at_breakdown_from_K(mid, dev) > sig:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def bv_horizontal_from_sigma(sig: float, dev: Device = DEV) -> tuple[float, float]:
    """
    Returns (BV_H, K).  NaN if outside the full-depletion domain.
    """
    K = K_from_sigma(sig, dev)
    if not math.isfinite(K):
        return math.nan, math.nan
    return sig * (1.0 + K), K


# ============================================================================
# 5. VERTICAL BREAKDOWN (PAPER'S 1-D LOW-DOPING LIMIT)
# ============================================================================

def vertical_breakdown(dev: Device = DEV) -> tuple[float, float]:
    """
    Under the n+ drain, for low enough SOI doping:
        A_F * Ec^7 * (ts-tj) = 1
        BV_V = Vsb + Ec[(ts-tj) + tb/k].

    Returns (BV_V, Ec).
    """
    if dev.dv <= 0.0:
        raise ValueError("Need ts > tj for a positive vertical Si path.")

    Ec = (1.0 / (A_F * dev.dv)) ** (1.0 / 7.0)
    BVv = dev.Vsb + Ec * (dev.dv + dev.tb / dev.k)
    return BVv, Ec


# ============================================================================
# 6. OPTIONAL POST-RESURF 1-D ASYMPTOTE
# ============================================================================

def bv_post_resurf_1d(N1: float, N2: float, dev: Device = DEV) -> float:
    """
    Optional no-fit one-sided abrupt-junction asymptote after RESURF loss.

    Nbar = thickness-averaged donor concentration.

    For E(x)=Emax(1-x/W), W=eps_s Emax/(q Nbar), and
    int A_F E^7 dx = 1,

      BV = (1/(2 A_F))^(1/4) * (eps_s/(q Nbar))^(3/4).

    This is NOT the same full-depletion PDE branch.  It is off by default.
    """
    Nbar = (N1 * dev.t1 + N2 * dev.t2) / dev.ts
    return (1.0 / (2.0 * A_F)) ** 0.25 * (dev.eps_s / (Q * Nbar)) ** 0.75


# ============================================================================
# 7. TURNING POINTS
# ============================================================================

def bisection_root(func, lo: float, hi: float, n: int = 100) -> float:
    flo = func(lo)
    fhi = func(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("Root is not bracketed.")
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = func(mid)
        if flo * fm <= 0.0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)


def horizontal_vertical_crossovers(BVv: float, dev: Device = DEV) -> list[float]:
    """
    Find K roots of BV_H(K)=BV_V.

    BV_H has its intrinsic maximum at K=1.
    If BV_V lies between the endpoint value and the horizontal maximum,
    there are two roots: one K<1 and one K>1.
    """
    f = lambda K: bv_horizontal_from_K(K, dev) - BVv

    K_roots = []

    # low-K branch: 0 <= K <= 1
    if f(0.0) * f(1.0) <= 0.0:
        K_roots.append(bisection_root(f, 0.0, 1.0))

    # high-K branch: 1 <= K < infinity
    if f(1.0) == 0.0:
        if not K_roots or not math.isclose(K_roots[-1], 1.0):
            K_roots.append(1.0)
    elif f(1.0) > 0.0:
        hi = 2.0
        while f(hi) > 0.0 and hi < 1e10:
            hi *= 2.0
        if hi < 1e10:
            K_roots.append(bisection_root(f, 1.0, hi))

    return K_roots


# ============================================================================
# 8. SWEEP: PDE -> sigma_c -> K -> BV_H -> min(BV_H,BV_V)
# ============================================================================

def run_sweep():
    N1_grid = np.linspace(N1_MIN, N1_MAX, N1_POINTS)
    BVv, Ec = vertical_breakdown(DEV)

    sigma0 = sigma_at_breakdown_from_K(0.0, DEV)   # RESURF charge limit
    sigma1 = sigma_at_breakdown_from_K(1.0, DEV)   # equal edge fields
    BVh_max = 2.0 * sigma1

    print("=" * 78)
    print("NO-FIT PDE -> VBD SWEEP")
    print("=" * 78)
    print(f"k = eps_ox/eps_s = {DEV.k:.8f}")
    print(f"t_s = {DEV.ts/UM:.4f} um")
    print(f"t_c = {DEV.tc/UM:.4f} um")
    print(f"D = l_c^2 = {DEV.D:.6e} cm^2")
    print(f"l_c = {DEV.lc/UM:.6f} um")
    print(f"lambda = L/l_c = {DEV.lam:.6f}")
    print(f"vertical Ec = {Ec:.6e} V/cm")
    print(f"vertical BV_V = {BVv:.6f} V")
    print(f"sigma(K=0) = {sigma0:.6f} V  [RESURF full-depletion limit]")
    print(f"sigma(K=1) = {sigma1:.6f} V  [equal edge fields]")
    print(f"max horizontal BV_H(K=1) = {BVh_max:.6f} V")
    print()

    rows = []

    fig, ax = plt.subplots(figsize=(8.8, 6.4))

    for N2 in N2_VALUES:
        bv_total = []
        bv_h_list = []
        bv_v_list = []
        K_list = []
        sigma_list = []
        regime_list = []

        for N1 in N1_grid:
            sig = sigma_c(float(N1), float(N2), DEV)
            BVh, K = bv_horizontal_from_sigma(sig, DEV)

            if math.isfinite(BVh):
                BV = min(BVh, BVv)
                regime = "horizontal" if BVh < BVv else "vertical"
            elif USE_POST_RESURF_1D:
                BV = bv_post_resurf_1d(float(N1), float(N2), DEV)
                regime = "post-RESURF-1D"
            else:
                BV = math.nan
                regime = "outside-full-depletion-model"

            sigma_list.append(sig)
            bv_h_list.append(BVh)
            bv_v_list.append(BVv)
            bv_total.append(BV)
            K_list.append(K)
            regime_list.append(regime)

            rows.append({
                "N1_cm^-3": float(N1),
                "N2_cm^-3": float(N2),
                "sigma_c_V": sig,
                "K": K,
                "BV_horizontal_V": BVh,
                "BV_vertical_V": BVv,
                "VBD_V": BV,
                "regime": regime,
            })

        # Total VBD curve.
        ax.plot(
            N1_grid / 1e15,
            np.asarray(bv_total),
            linewidth=2.0,
            label=f"VBD, N2={N2:.0e} cm^-3",
        )

        # Light horizontal branch reference.
        ax.plot(
            N1_grid / 1e15,
            np.asarray(bv_h_list),
            linestyle="--",
            linewidth=1.0,
            label=f"horizontal BV, N2={N2:.0e}",
        )

        # Key doping locations for this N2.
        N1_k1 = N1_from_sigma(sigma1, N2, DEV)
        N1_k0 = N1_from_sigma(sigma0, N2, DEV)

        print(f"N2 = {N2:.4e} cm^-3")
        print(f"  N1 at K=1  = {N1_k1:.6e} cm^-3")
        print(f"  N1 at K=0  = {N1_k0:.6e} cm^-3")

        # Actual VBD plateau entry/exit if BVv intersects horizontal branch.
        roots = horizontal_vertical_crossovers(BVv, DEV)
        if roots:
            print("  BV_H = BV_V crossovers:")
            for Kr in roots:
                sr = sigma_at_breakdown_from_K(Kr, DEV)
                N1r = N1_from_sigma(sr, N2, DEV)
                print(
                    f"    K={Kr:.8f}, sigma={sr:.6f} V, "
                    f"N1={N1r:.6e} cm^-3"
                )
        else:
            print("  No horizontal/vertical crossover.")
        print()

    # One common vertical cap.
    ax.axhline(BVv, linestyle=":", linewidth=1.5, label=f"vertical cap = {BVv:.1f} V")

    ax.set_xlabel(r"Surface doping $N_1$ [$10^{15}$ cm$^{-3}$]")
    ax.set_ylabel(r"Breakdown voltage $V_{BD}$ [V]")
    ax.set_title("No-fit PDE-to-breakdown sweep")
    ax.set_xlim(N1_MIN / 1e15, N1_MAX / 1e15)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # CSV output.
    fieldnames = [
        "N1_cm^-3",
        "N2_cm^-3",
        "sigma_c_V",
        "K",
        "BV_horizontal_V",
        "BV_vertical_V",
        "VBD_V",
        "regime",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {Path(OUT_PNG).resolve()}")
    print(f"Saved: {Path(OUT_CSV).resolve()}")


if __name__ == "__main__":
    run_sweep()
