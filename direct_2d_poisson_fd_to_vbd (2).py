#!/usr/bin/env python3
"""
direct_2d_poisson_fd_to_vbd.py
==============================

Standalone NO-FIT direct finite-difference solver for the original depleted
2-D Poisson equation in the silicon drift region of a surface-implanted SOI
RESURF LDMOS.

This script does NOT use the parabolic vertical-potential approximation to
solve phi(x,y).  Instead, it discretizes the 2-D Poisson PDE directly on an
(x,y) mesh and solves the sparse linear system.

It automatically produces:

  1) direct 2-D electrostatic potential phi(x,y)
  2) Ex(x,y) = -dphi/dx
  3) Ey(x,y) = -dphi/dy
  4) |E(x,y)| = sqrt(Ex^2 + Ey^2)
  5) horizontal surface-avalanche BV from the direct PDE field
  6) the paper's independent low-doping vertical avalanche cap
  7) device VBD(N1) = min(BV_horizontal_FD, BV_vertical)
  8) low / medium / high N1 field maps
  9) CSV files with the sweep and case summaries

NO FITTING:
  - no digitized data
  - no regression parameters
  - no fitted critical field
  - avalanche uses Fulop: alpha_eff = A_F * |E|^7

Dependencies:
  numpy, scipy, matplotlib

Run:
  python direct_2d_poisson_fd_to_vbd.py

-------------------------------------------------------------------------------
PDE AND BOUNDARY CONDITIONS
-------------------------------------------------------------------------------

Silicon:
    d2(phi)/dx2 + d2(phi)/dy2 = -q*N(y)/eps_s

where
    N(y) = N1,  0 <= y < t1
           N2,  t1 < y <= ts.

The silicon permittivity is the same on both sides of y=t1, so continuity of
phi and eps_s*dphi/dy is automatically enforced by the single finite-difference
mesh when the interface is aligned with a grid row.

Lateral boundaries:
    phi(0,y) = 0
    phi(L,y) = Vd

Top silicon surface:
    dphi/dy = 0

Bottom silicon / BOX effective Robin boundary:
    dphi/dy + (k/tb)*phi = (k/tb)*Vsb,
    k = eps_ox/eps_s.

This Robin condition follows from a charge-free BOX with linear potential and
continuity of normal displacement.

-------------------------------------------------------------------------------
BREAKDOWN CRITERION
-------------------------------------------------------------------------------

The direct 2-D PDE is used for the lateral/surface breakdown path:

    I_H(Vd,N1) = integral_0^L A_F * |Ex(x, y=0)|^7 dx.

The high-voltage surface-breakdown root is defined by
    I_H = 1.

Because I_H(Vd) is generally convex rather than monotonic, the code first
finds its minimum and then finds the HIGH-VOLTAGE root.  Root finding and
scalar minimization are equation solving, not parameter fitting.

The independent vertical cap is the same low-doping 1-D Fulop limit used in
the analytical paper:

    dv = ts - tj
    Ec = [1/(A_F*dv)]^(1/7)
    BV_vertical = Vsb + Ec*(dv + tb/k).

Thus, inside the fully depleted model,

    VBD = min(BV_horizontal_FD, BV_vertical).

IMPORTANT LIMITATION:
The rectangular PDE domain does not contain an explicit n+ drain diffusion
shape or gate-field-plate geometry.  Therefore the vertical avalanche ceiling
is kept as the paper's separate 1-D path rather than extracted from an
arbitrary interior x-column.  This avoids introducing a fitted or arbitrary
"vertical path location".

At sufficiently large N1 the fully depleted PDE can have no subcritical
surface-field state (min I_H > 1).  Such points are flagged as outside the
fully depleted model and are NOT artificially continued.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize_scalar


# =============================================================================
# 1. USER INPUTS
# =============================================================================

# Physical constants
q = 1.602176634e-19
eps0 = 8.8541878128e-14          # F/cm
eps_s = 11.8 * eps0              # Si
eps_ox = 3.9 * eps0              # SiO2
A_F = 1.8e-35                    # Fulop: alpha_eff = A_F E^7
um = 1e-4                        # cm

# Geometry
L = 40.0 * um
t1 = 1.0 * um
t2 = 4.0 * um
ts = t1 + t2
tb = 2.0 * um
tj = 2.0 * um
Vsb = 0.0

# Background doping
N2 = 1.0e14                      # cm^-3

# Surface-doping sweep
N1_MIN = 1.0e14
N1_MAX = 1.2e16
N1_POINTS = 61

# Three field-map cases; edit freely.
CASE_N1 = [
    2.0e15,                      # low
    8.0e15,                      # medium
    1.0e16,                      # high
]

# Direct PDE mesh.
# Choose Ny so t1 lies exactly on a grid row.
NX = 161                         # dx = 0.25 um for L=40 um
NY = 51                          # dy = 0.10 um for ts=5 um

# Avalanche search range
V_SEARCH_MAX = 1200.0            # V

# If True, device VBD is clipped by the independent vertical cap.
USE_VERTICAL_CAP = True

# Optional 3-D surfaces. 2-D maps are always generated.
MAKE_3D = True

# Output directory
OUTDIR = Path("direct_2d_poisson_results")


# =============================================================================
# 2. GRID / DEVICE
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
    def dv(self):
        return self.ts - self.tj


dev = Device(L, t1, t2, tb, tj, Vsb, eps_s, eps_ox)

x = np.linspace(0.0, dev.L, NX)
y = np.linspace(0.0, dev.ts, NY)

dx = x[1] - x[0]
dy = y[1] - y[0]

gamma_box = dev.k / dev.tb

# Confirm the implant interface is grid-aligned.
j_interface = int(np.argmin(np.abs(y - dev.t1)))
if abs(y[j_interface] - dev.t1) > 1e-10 * max(dev.ts, 1.0):
    raise ValueError(
        "t1 is not aligned to a y-grid row. Change NY so y contains t1."
    )


# =============================================================================
# 3. SPARSE FINITE-DIFFERENCE MATRIX
# =============================================================================

def node(j, i):
    return j * NX + i


def build_matrix():
    """
    Build A for A*phi = b.

    Corner precedence:
      x=0 or x=L Dirichlet conditions override top/bottom conditions.
    """
    rows = []
    cols = []
    vals = []

    for j in range(NY):
        for i in range(NX):
            p = node(j, i)

            # Left/right Dirichlet
            if i == 0 or i == NX - 1:
                rows.append(p)
                cols.append(p)
                vals.append(1.0)
                continue

            # Top Neumann: dphi/dy = 0
            # 2nd-order one-sided derivative:
            # (-3 phi_0 + 4 phi_1 - phi_2)/(2dy) = 0
            if j == 0:
                rows.extend([p, p, p])
                cols.extend([
                    node(0, i),
                    node(1, i),
                    node(2, i),
                ])
                vals.extend([
                    -3.0 / (2.0 * dy),
                    +4.0 / (2.0 * dy),
                    -1.0 / (2.0 * dy),
                ])
                continue

            # Bottom Robin:
            # dphi/dy + gamma_box*phi = gamma_box*Vsb
            # derivative = (3 phi_N - 4 phi_N-1 + phi_N-2)/(2dy)
            if j == NY - 1:
                rows.extend([p, p, p])
                cols.extend([
                    node(NY - 1, i),
                    node(NY - 2, i),
                    node(NY - 3, i),
                ])
                vals.extend([
                    3.0 / (2.0 * dy) + gamma_box,
                    -4.0 / (2.0 * dy),
                    +1.0 / (2.0 * dy),
                ])
                continue

            # Interior 5-point Laplacian
            rows.extend([p] * 5)
            cols.extend([
                node(j, i),
                node(j, i - 1),
                node(j, i + 1),
                node(j - 1, i),
                node(j + 1, i),
            ])
            vals.extend([
                -2.0 / dx**2 - 2.0 / dy**2,
                +1.0 / dx**2,
                +1.0 / dx**2,
                +1.0 / dy**2,
                +1.0 / dy**2,
            ])

    A = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(NX * NY, NX * NY),
    )

    return A


A = build_matrix()

# Factor only once. Every N1 and Vd uses the same matrix.
LU = splu(A.tocsc())


# =============================================================================
# 4. RHS AND LINEAR BASIS SOLUTIONS
# =============================================================================

# Interface-node source is averaged to avoid assigning the discontinuity
# entirely to one layer.  This is a standard grid representation of a
# piecewise constant source; the exact point value has zero continuum measure.
w1 = np.zeros(NY)
w2 = np.zeros(NY)

for j, yy in enumerate(y):
    if j < j_interface:
        w1[j] = 1.0
    elif j > j_interface:
        w2[j] = 1.0
    else:
        w1[j] = 0.5
        w2[j] = 0.5


def make_rhs(Vd=0.0, Vsb_value=0.0, N1=0.0, N2_value=0.0):
    b = np.zeros(NX * NY, dtype=float)

    for j in range(NY):
        for i in range(NX):
            p = node(j, i)

            if i == 0:
                b[p] = 0.0

            elif i == NX - 1:
                b[p] = Vd

            elif j == 0:
                b[p] = 0.0

            elif j == NY - 1:
                b[p] = gamma_box * Vsb_value

            else:
                Nd = N1 * w1[j] + N2_value * w2[j]
                b[p] = -q * Nd / dev.eps_s

    return b


def solve_rhs(**kwargs):
    return LU.solve(make_rhs(**kwargs)).reshape(NY, NX)


# Linear superposition makes the N1/Vd sweep fast while remaining a direct
# finite-difference PDE solution. Each basis below is obtained from A*phi=b.
N_SCALE = 1.0e15

phi_Vd = solve_rhs(Vd=1.0)
phi_Vsb = solve_rhs(Vsb_value=1.0)
phi_N1 = solve_rhs(N1=N_SCALE)
phi_N2 = solve_rhs(N2_value=N_SCALE)


def gradient_fields(phi):
    dphi_dy, dphi_dx = np.gradient(
        phi, dy, dx, edge_order=2
    )
    return -dphi_dx, -dphi_dy


Ex_Vd, Ey_Vd = gradient_fields(phi_Vd)
Ex_Vsb, Ey_Vsb = gradient_fields(phi_Vsb)
Ex_N1, Ey_N1 = gradient_fields(phi_N1)
Ex_N2, Ey_N2 = gradient_fields(phi_N2)


# =============================================================================
# 5. COMBINE BASIS -> DIRECT PDE POTENTIAL AND FIELD
# =============================================================================

def direct_solution(N1, Vd):
    """
    Return phi, Ex, Ey, |E| from direct 2-D finite-difference PDE solution.
    """
    r1 = N1 / N_SCALE
    r2 = N2 / N_SCALE

    phi = (
        Vd * phi_Vd
        + dev.Vsb * phi_Vsb
        + r1 * phi_N1
        + r2 * phi_N2
    )

    Ex = (
        Vd * Ex_Vd
        + dev.Vsb * Ex_Vsb
        + r1 * Ex_N1
        + r2 * Ex_N2
    )

    Ey = (
        Vd * Ey_Vd
        + dev.Vsb * Ey_Vsb
        + r1 * Ey_N1
        + r2 * Ey_N2
    )

    Emag = np.sqrt(Ex**2 + Ey**2)

    return phi, Ex, Ey, Emag


# =============================================================================
# 6. DIRECT-PDE SURFACE AVALANCHE
# =============================================================================

def surface_ionization_integral(N1, Vd):
    """
    Fulop surface path:
        I_H = integral A_F |Ex(x, y=0)|^7 dx.
    Breakdown when I_H = 1.
    """
    r1 = N1 / N_SCALE
    r2 = N2 / N_SCALE

    Ex_top = (
        Vd * Ex_Vd[0, :]
        + dev.Vsb * Ex_Vsb[0, :]
        + r1 * Ex_N1[0, :]
        + r2 * Ex_N2[0, :]
    )

    return A_F * np.trapezoid(
        np.abs(Ex_top)**7,
        x,
    )


def horizontal_breakdown_fd(N1):
    """
    Find the HIGH-VOLTAGE root I_H(Vd)=1.

    I_H(Vd) is convex for this linear electrostatic problem and may have
    two roots. The high-voltage root corresponds to the breakdown branch
    after the field-distribution optimum.

    Returns:
      BVH, V_at_I_min, I_min, status
    """
    obj = lambda V: surface_ionization_integral(N1, V)

    res = minimize_scalar(
        obj,
        bounds=(0.0, V_SEARCH_MAX),
        method="bounded",
        options={"xatol": 1e-8},
    )

    Vmin = float(res.x)
    Imin = float(res.fun)

    # No non-avalanche state inside the assumed fully depleted model.
    if Imin > 1.0:
        return math.nan, Vmin, Imin, "no_subcritical_full_depletion_state"

    # Find upper bracket to the right of the minimum.
    lo = Vmin
    flo = obj(lo) - 1.0

    hi = max(Vmin + 25.0, 50.0)

    while obj(hi) < 1.0 and hi < 4.0 * V_SEARCH_MAX:
        hi *= 1.5

    if obj(hi) < 1.0:
        return math.nan, Vmin, Imin, "high_voltage_root_not_bracketed"

    # Bisection on the high-voltage root.
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = obj(mid) - 1.0

        if fm <= 0.0:
            lo = mid
            flo = fm
        else:
            hi = mid

    BVH = 0.5 * (lo + hi)

    return BVH, Vmin, Imin, "ok"


# =============================================================================
# 7. INDEPENDENT VERTICAL CAP
# =============================================================================

def vertical_breakdown_1d():
    """
    Paper's low-doping 1-D vertical path under the drain.
    """
    if dev.dv <= 0.0:
        raise ValueError("Need ts > tj.")

    Ec = (1.0 / (A_F * dev.dv))**(1.0 / 7.0)

    BVv = (
        dev.Vsb
        + Ec * (dev.dv + dev.tb / dev.k)
    )

    return BVv, Ec


BV_VERTICAL, EC_VERTICAL = vertical_breakdown_1d()


# =============================================================================
# 8. VBD VS N1 SWEEP
# =============================================================================

def run_vbd_sweep():
    N1_values = np.linspace(N1_MIN, N1_MAX, N1_POINTS)

    rows = []

    for N1_value in N1_values:
        BVH, Vmin, Imin, status = horizontal_breakdown_fd(N1_value)

        if math.isfinite(BVH):
            if USE_VERTICAL_CAP:
                VBD = min(BVH, BV_VERTICAL)
                mechanism = (
                    "horizontal_surface"
                    if BVH < BV_VERTICAL
                    else "vertical_cap"
                )
            else:
                VBD = BVH
                mechanism = "horizontal_surface"
        else:
            VBD = math.nan
            mechanism = "outside_full_depletion_model"

        rows.append({
            "N1_cm^-3": N1_value,
            "N2_cm^-3": N2,
            "BV_horizontal_FD_V": BVH,
            "BV_vertical_1D_V": BV_VERTICAL,
            "VBD_device_V": VBD,
            "V_at_min_ionization_V": Vmin,
            "min_surface_ionization_integral": Imin,
            "mechanism": mechanism,
            "status": status,
        })

    return rows


# =============================================================================
# 9. PLOTTING HELPERS -- ONE FIGURE PER CHART
# =============================================================================

def save_scalar_map(data, title, label, path, case_N1, case_Vd):
    X, Y = np.meshgrid(x / um, y / um)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    m = ax.contourf(X, Y, data, levels=60)
    ax.contour(X, Y, data, levels=12, linewidths=0.55)

    ax.axhline(dev.t1 / um, linewidth=0.9)
    ax.set_ylim(dev.ts / um, 0.0)
    ax.set_xlabel("Drift coordinate x [um]")
    ax.set_ylabel("Depth y [um]")
    ax.set_title(
        f"{title}\n"
        f"N1={case_N1:.3e} cm^-3, Vd={case_Vd:.2f} V"
    )

    cb = fig.colorbar(m, ax=ax)
    cb.set_label(label)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_field_magnitude_map(
    Emag, Ex, Ey, title, path, case_N1, case_Vd
):
    X, Y = np.meshgrid(x / um, y / um)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    m = ax.contourf(
        X,
        Y,
        Emag / 1e3,
        levels=60,
    )

    sx = slice(0, NX, 12)
    sy = slice(0, NY, 5)

    Exs = Ex[sy, sx]
    Eys = Ey[sy, sx]

    norm = np.sqrt(Exs**2 + Eys**2)
    norm[norm == 0.0] = 1.0

    ax.quiver(
        X[sy, sx],
        Y[sy, sx],
        Exs / norm,
        Eys / norm,
        angles="xy",
        scale_units="xy",
        scale=1.2,
        width=0.002,
    )

    ax.axhline(dev.t1 / um, linewidth=0.9)
    ax.set_ylim(dev.ts / um, 0.0)
    ax.set_xlabel("Drift coordinate x [um]")
    ax.set_ylabel("Depth y [um]")
    ax.set_title(
        f"{title}\n"
        f"N1={case_N1:.3e} cm^-3, Vd={case_Vd:.2f} V"
    )

    cb = fig.colorbar(m, ax=ax)
    cb.set_label("|E| [kV/cm]")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_3d_surface(data, zlabel, title, path, scale=1.0):
    X, Y = np.meshgrid(x[::4] / um, y[::2] / um)
    Z = data[::2, ::4] * scale

    fig = plt.figure(figsize=(8.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        Z,
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=28, azim=-128)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_vbd_sweep(rows):
    N1_arr = np.array([r["N1_cm^-3"] for r in rows])
    BVH_arr = np.array([r["BV_horizontal_FD_V"] for r in rows])
    VBD_arr = np.array([r["VBD_device_V"] for r in rows])

    fig, ax = plt.subplots(figsize=(8.2, 5.5))

    ax.plot(
        N1_arr / 1e15,
        BVH_arr,
        label="Direct 2-D PDE surface BV",
    )

    ax.plot(
        N1_arr / 1e15,
        VBD_arr,
        linewidth=2.0,
        label="Device VBD",
    )

    if USE_VERTICAL_CAP:
        ax.axhline(
            BV_VERTICAL,
            linestyle="--",
            label=f"1-D vertical cap = {BV_VERTICAL:.1f} V",
        )

    ax.set_xlabel(r"Surface doping $N_1$ [$10^{15}$ cm$^{-3}$]")
    ax.set_ylabel(r"Breakdown voltage $V_{BD}$ [V]")
    ax.set_title("No-fit direct 2-D Poisson PDE: breakdown vs surface doping")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTDIR / "vbd_vs_surface_doping.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# 10. FIELD-MAP CASES
# =============================================================================

def nearest_sweep_row(rows, target_N1):
    valid = [
        r for r in rows
        if math.isfinite(r["VBD_device_V"])
    ]

    return min(
        valid,
        key=lambda r: abs(r["N1_cm^-3"] - target_N1),
    )


def generate_case_outputs(rows):
    summaries = []

    for case_id, target_N1 in enumerate(CASE_N1, start=1):
        r = nearest_sweep_row(rows, target_N1)

        N1_case = r["N1_cm^-3"]
        Vd_case = r["VBD_device_V"]

        phi, Ex, Ey, Emag = direct_solution(N1_case, Vd_case)

        # The mixed Dirichlet/Robin corners can create a mathematical
        # corner-field singularity. Report both the raw grid maximum and
        # a guarded physical maximum excluding two lateral boundary cells.
        idx_max_raw = np.unravel_index(
            np.nanargmax(Emag),
            Emag.shape,
        )

        E_guarded = Emag[:, 2:-2]
        idx_guard = np.unravel_index(
            np.nanargmax(E_guarded),
            E_guarded.shape,
        )
        idx_max = (idx_guard[0], idx_guard[1] + 2)

        case_dir = OUTDIR / f"case_{case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)

        save_scalar_map(
            phi,
            "Direct-PDE electrostatic potential",
            "Potential phi [V]",
            case_dir / "potential_2d.png",
            N1_case,
            Vd_case,
        )

        save_scalar_map(
            Ex / 1e3,
            "Direct-PDE lateral electric field Ex",
            "Ex [kV/cm]",
            case_dir / "Ex_2d.png",
            N1_case,
            Vd_case,
        )

        save_scalar_map(
            Ey / 1e3,
            "Direct-PDE vertical electric field Ey",
            "Ey [kV/cm]",
            case_dir / "Ey_2d.png",
            N1_case,
            Vd_case,
        )

        save_field_magnitude_map(
            Emag,
            Ex,
            Ey,
            "Direct-PDE electric-field magnitude",
            case_dir / "E_magnitude_2d.png",
            N1_case,
            Vd_case,
        )

        if MAKE_3D:
            save_3d_surface(
                phi,
                "phi [V]",
                f"Potential surface, N1={N1_case:.3e} cm^-3",
                case_dir / "potential_3d.png",
                scale=1.0,
            )

            save_3d_surface(
                Emag,
                "|E| [kV/cm]",
                f"Field magnitude, N1={N1_case:.3e} cm^-3",
                case_dir / "E_magnitude_3d.png",
                scale=1e-3,
            )

        summaries.append({
            "case": case_id,
            "target_N1_cm^-3": target_N1,
            "actual_N1_cm^-3": N1_case,
            "Vd_V": Vd_case,
            "BV_horizontal_FD_V": r["BV_horizontal_FD_V"],
            "BV_vertical_1D_V": r["BV_vertical_1D_V"],
            "mechanism": r["mechanism"],
            "phi_min_V": float(np.nanmin(phi)),
            "phi_max_V": float(np.nanmax(phi)),
            "Emax_guarded_V_per_cm": float(Emag[idx_max]),
            "Emax_guarded_x_um": float(x[idx_max[1]] / um),
            "Emax_guarded_y_um": float(y[idx_max[0]] / um),
            "Emax_raw_V_per_cm": float(Emag[idx_max_raw]),
            "Emax_raw_x_um": float(x[idx_max_raw[1]] / um),
            "Emax_raw_y_um": float(y[idx_max_raw[0]] / um),
        })

    return summaries


# =============================================================================
# 11. CSV
# =============================================================================

def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("DIRECT 2-D FINITE-DIFFERENCE POISSON PDE -> VBD -> POTENTIAL / E FIELD")
    print("=" * 88)

    print(f"Grid: NX={NX}, NY={NY}, unknowns={NX*NY}")
    print(f"dx = {dx/um:.4f} um")
    print(f"dy = {dy/um:.4f} um")
    print(f"implant interface row y = {y[j_interface]/um:.4f} um")
    print(f"k = eps_ox/eps_s = {dev.k:.8f}")
    print(f"vertical Ec = {EC_VERTICAL:.6e} V/cm")
    print(f"vertical BV cap = {BV_VERTICAL:.6f} V")
    print()

    rows = run_vbd_sweep()

    plot_vbd_sweep(rows)

    write_csv(
        OUTDIR / "vbd_vs_surface_doping.csv",
        rows,
    )

    summaries = generate_case_outputs(rows)

    write_csv(
        OUTDIR / "field_cases_summary.csv",
        summaries,
    )

    valid = [
        r for r in rows
        if math.isfinite(r["BV_horizontal_FD_V"])
    ]

    invalid = [
        r for r in rows
        if not math.isfinite(r["BV_horizontal_FD_V"])
    ]

    print(f"Valid full-depletion sweep points: {len(valid)} / {len(rows)}")

    if valid:
        max_bvh_row = max(
            valid,
            key=lambda r: r["BV_horizontal_FD_V"],
        )

        print(
            "Maximum direct-PDE horizontal BV in valid sweep: "
            f"{max_bvh_row['BV_horizontal_FD_V']:.3f} V "
            f"at N1={max_bvh_row['N1_cm^-3']:.4e} cm^-3"
        )

    if invalid:
        print(
            "First sweep point with no subcritical fully depleted surface state: "
            f"N1={invalid[0]['N1_cm^-3']:.4e} cm^-3"
        )

    print()
    print("Representative cases:")

    for s in summaries:
        print(
            f"  case {s['case']}: "
            f"N1={s['actual_N1_cm^-3']:.4e} cm^-3, "
            f"Vd={s['Vd_V']:.3f} V, "
            f"guarded |E|max={s['Emax_guarded_V_per_cm']/1e3:.3f} kV/cm "
            f"at ({s['Emax_guarded_x_um']:.3f}, {s['Emax_guarded_y_um']:.3f}) um"
        )

    print()
    print("Saved to:")
    print(" ", OUTDIR.resolve())


if __name__ == "__main__":
    main()
