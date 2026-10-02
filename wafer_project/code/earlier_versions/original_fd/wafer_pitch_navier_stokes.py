#!/usr/bin/env python3
"""
2-D incompressible Navier-Stokes wafer-pitch study.

NO FITTING:
The velocity and pressure fields are obtained by directly discretizing

    du/dt + (u . grad)u = -grad(p) + nu*laplacian(u)
    div(u) = 0

using a finite-difference fractional-step / pressure-projection method.

Default normalized geometry:
    Tank: W = 1.0, H = 0.45
    6 wafers
    wafer thickness = 0.015
    wafer vertical span = 0.12 ... 0.25

Pitch cases:
    Small  P/W = 0.040
    Medium P/W = 0.060
    Large  P/W = 0.080

Boundary conditions:
    - five bottom vertical inlet slots centered in the five wafer gaps
    - fixed inlet speed
    - no-slip side/bottom walls and wafer surfaces
    - open top pressure outlet, p = 0
    - top velocity uses zero-normal-gradient plus global flux balance

Outputs:
    - one streamline / speed plot for each pitch
    - vertical Y-cut speed comparison at x = 0.50
    - CSV data and numerical summary

Dependencies:
    numpy, scipy, pandas, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OUTDIR = "wafer_pitch_navier_stokes_output"
os.makedirs(OUTDIR, exist_ok=True)

W = 1.0
H = 0.45
N_WAFERS = 6
WAFER_THICKNESS = 0.015
WAFER_Y0 = 0.12
WAFER_Y1 = 0.25

PITCHES = {
    "Small": 0.040,
    "Medium": 0.060,
    "Large": 0.080,
}

INLET_WIDTH = 0.015
INLET_SPEED = 0.30
NU = 7.5e-4

NX = 201
NY = 91
DT = 0.0012
N_STEPS = 2200

Y_CUT_X = 0.50


def build_geometry(pitch):
    x = np.linspace(0.0, W, NX)
    y = np.linspace(0.0, H, NY)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)

    wafer_centers = 0.5 + (
        np.arange(N_WAFERS) - (N_WAFERS - 1) / 2.0
    ) * pitch

    solid = np.zeros((NY, NX), dtype=bool)
    for xc in wafer_centers:
        solid |= (
            (np.abs(X - xc) <= WAFER_THICKNESS / 2.0 + 1e-12)
            & (Y >= WAFER_Y0 - 1e-12)
            & (Y <= WAFER_Y1 + 1e-12)
        )

    inlet_centers = 0.5 * (wafer_centers[:-1] + wafer_centers[1:])
    inlet = np.zeros(NX, dtype=bool)
    for xc in inlet_centers:
        inlet |= np.abs(x - xc) <= INLET_WIDTH / 2.0 + 1e-12

    return x, y, dx, dy, solid, wafer_centers, inlet_centers, inlet


def build_pressure_solver(fluid, dx, dy):
    idx = -np.ones((NY, NX), dtype=int)

    coords = np.argwhere(fluid[1:-1, 1:-1])
    coords[:, 0] += 1
    coords[:, 1] += 1

    for k, (j, i) in enumerate(coords):
        idx[j, i] = k

    rows, cols, data = [], [], []
    idx2 = 1.0 / dx**2
    idy2 = 1.0 / dy**2

    for k, (j, i) in enumerate(coords):
        diag = 0.0

        for jj, ii, coeff in [
            (j, i + 1, idx2),
            (j, i - 1, idx2),
            (j + 1, i, idy2),
            (j - 1, i, idy2),
        ]:
            if jj == NY - 1:
                if fluid[jj, ii]:
                    diag += coeff
            elif ii == 0 or ii == NX - 1 or jj == 0:
                continue
            elif fluid[jj, ii]:
                kk = idx[jj, ii]
                if kk >= 0:
                    diag += coeff
                    rows.append(k)
                    cols.append(kk)
                    data.append(-coeff)
            else:
                continue

        rows.append(k)
        cols.append(k)
        data.append(diag)

    A = sp.csr_matrix((data, (rows, cols)), shape=(len(coords), len(coords)))
    return coords, spla.factorized(A.tocsc())


def apply_velocity_bc(u, v, solid, inlet, dx, inlet_flux):
    u[:, 0] = 0.0
    v[:, 0] = 0.0
    u[:, -1] = 0.0
    v[:, -1] = 0.0

    u[0, :] = 0.0
    v[0, :] = 0.0
    v[0, inlet] = INLET_SPEED

    u[-1, :] = u[-2, :]
    v[-1, :] = v[-2, :]

    u[solid] = 0.0
    v[solid] = 0.0

    qout = np.sum(v[-1, :]) * dx
    if qout > 1e-14:
        v[-1, :] *= inlet_flux / qout

    v[-1, 0] = 0.0
    v[-1, -1] = 0.0


def solve_case(pitch):
    (
        x, y, dx, dy, solid,
        wafer_centers, inlet_centers, inlet
    ) = build_geometry(pitch)

    fluid = ~solid
    Fi = fluid[1:-1, 1:-1]

    pressure_coords, solve_pressure = build_pressure_solver(fluid, dx, dy)

    u = np.zeros((NY, NX), dtype=float)
    v = np.zeros_like(u)
    p = np.zeros_like(u)

    inlet_flux = np.sum(inlet) * dx * INLET_SPEED
    apply_velocity_bc(u, v, solid, inlet, dx, inlet_flux)

    residual = np.nan

    for n in range(N_STEPS):
        un = u.copy()
        vn = v.copy()

        uc = un[1:-1, 1:-1]
        vc = vn[1:-1, 1:-1]

        dudx = np.where(
            uc >= 0.0,
            (uc - un[1:-1, :-2]) / dx,
            (un[1:-1, 2:] - uc) / dx,
        )
        dudy = np.where(
            vc >= 0.0,
            (uc - un[:-2, 1:-1]) / dy,
            (un[2:, 1:-1] - uc) / dy,
        )
        dvdx = np.where(
            uc >= 0.0,
            (vc - vn[1:-1, :-2]) / dx,
            (vn[1:-1, 2:] - vc) / dx,
        )
        dvdy = np.where(
            vc >= 0.0,
            (vc - vn[:-2, 1:-1]) / dy,
            (vn[2:, 1:-1] - vc) / dy,
        )

        lap_u = (
            (un[1:-1, 2:] - 2.0 * uc + un[1:-1, :-2]) / dx**2
            + (un[2:, 1:-1] - 2.0 * uc + un[:-2, 1:-1]) / dy**2
        )
        lap_v = (
            (vn[1:-1, 2:] - 2.0 * vc + vn[1:-1, :-2]) / dx**2
            + (vn[2:, 1:-1] - 2.0 * vc + vn[:-2, 1:-1]) / dy**2
        )

        us = un.copy()
        vs = vn.copy()

        us_pred = uc + DT * (-uc * dudx - vc * dudy + NU * lap_u)
        vs_pred = vc + DT * (-uc * dvdx - vc * dvdy + NU * lap_v)

        us[1:-1, 1:-1] = np.where(Fi, us_pred, 0.0)
        vs[1:-1, 1:-1] = np.where(Fi, vs_pred, 0.0)

        apply_velocity_bc(us, vs, solid, inlet, dx, inlet_flux)

        div_star = (
            (us[1:-1, 2:] - us[1:-1, :-2]) / (2.0 * dx)
            + (vs[2:, 1:-1] - vs[:-2, 1:-1]) / (2.0 * dy)
        )

        rhs = np.zeros_like(p)
        rhs[1:-1, 1:-1] = np.where(Fi, div_star / DT, 0.0)

        b = -rhs[pressure_coords[:, 0], pressure_coords[:, 1]]
        pvec = solve_pressure(b)

        p.fill(0.0)
        p[pressure_coords[:, 0], pressure_coords[:, 1]] = pvec

        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = 0.0
        p[solid] = 0.0

        pc = p[1:-1, 1:-1]
        pe = np.where(fluid[1:-1, 2:], p[1:-1, 2:], pc)
        pw = np.where(fluid[1:-1, :-2], p[1:-1, :-2], pc)
        pn = np.where(fluid[2:, 1:-1], p[2:, 1:-1], pc)
        ps = np.where(fluid[:-2, 1:-1], p[:-2, 1:-1], pc)

        u = us.copy()
        v = vs.copy()

        u[1:-1, 1:-1] = np.where(
            Fi,
            us[1:-1, 1:-1] - DT * (pe - pw) / (2.0 * dx),
            0.0,
        )
        v[1:-1, 1:-1] = np.where(
            Fi,
            vs[1:-1, 1:-1] - DT * (pn - ps) / (2.0 * dy),
            0.0,
        )

        apply_velocity_bc(u, v, solid, inlet, dx, inlet_flux)

        if n % 100 == 0 and n > 0:
            du = np.sqrt(np.mean((u - un) ** 2 + (v - vn) ** 2))
            umag = np.sqrt(np.mean(u**2 + v**2)) + 1e-14
            residual = du / umag

    speed = np.hypot(u, v)
    speed[solid] = np.nan

    return {
        "pitch": pitch,
        "x": x,
        "y": y,
        "u": u,
        "v": v,
        "p": p,
        "speed": speed,
        "solid": solid,
        "wafer_centers": wafer_centers,
        "inlet_centers": inlet_centers,
        "inlet": inlet,
        "dx": dx,
        "dy": dy,
        "residual": residual,
        "inlet_flux": np.sum(v[0, inlet]) * dx,
        "outlet_flux": np.sum(v[-1, :]) * dx,
    }


def plot_flow(label, result):
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_subplot(111)

    pcm = ax.pcolormesh(
        result["x"],
        result["y"],
        result["speed"],
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=INLET_SPEED,
    )

    u_masked = np.ma.array(result["u"], mask=result["solid"])
    v_masked = np.ma.array(result["v"], mask=result["solid"])

    ax.streamplot(
        result["x"],
        result["y"],
        u_masked,
        v_masked,
        density=1.55,
        linewidth=0.8,
        arrowsize=0.9,
        color="white",
    )

    for xc in result["wafer_centers"]:
        ax.add_patch(
            patches.Rectangle(
                (xc - WAFER_THICKNESS / 2.0, WAFER_Y0),
                WAFER_THICKNESS,
                WAFER_Y1 - WAFER_Y0,
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )
        )

    ax.scatter(
        result["inlet_centers"],
        np.zeros_like(result["inlet_centers"]),
        marker="^",
        s=42,
        color="red",
        zorder=6,
        label="Inlets",
    )

    ax.axvline(
        Y_CUT_X,
        linestyle="--",
        linewidth=1.0,
        color="white",
        alpha=0.75,
        label=f"Y-cut x={Y_CUT_X:.2f}",
    )

    ax.set_xlim(0.0, W)
    ax.set_ylim(0.0, H)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"{label} pitch: P/W={result['pitch']:.3f}  |u| + streamlines + wafer mask"
    )

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("|u|")
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = os.path.join(OUTDIR, f"navier_stokes_{label.lower()}_pitch.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    results = {}
    for label, pitch in PITCHES.items():
        print(f"Solving {label} pitch P/W={pitch:.3f} ...")
        results[label] = solve_case(pitch)
        print(
            f"  residual={results[label]['residual']:.3e}, "
            f"Qin={results[label]['inlet_flux']:.6f}, "
            f"Qout={results[label]['outlet_flux']:.6f}"
        )
        plot_flow(label, results[label])

    x = results["Medium"]["x"]
    y = results["Medium"]["y"]
    ix_cut = int(np.argmin(np.abs(x - Y_CUT_X)))
    x_cut_actual = x[ix_cut]

    ycut = pd.DataFrame({"y": y})
    for label, result in results.items():
        ycut[f"{label.lower()}_pitch_speed"] = result["speed"][:, ix_cut]

    ycut.to_csv(
        os.path.join(OUTDIR, "y_cut_speed_x_0p50.csv"),
        index=False,
    )

    fig = plt.figure(figsize=(7.0, 6.2))
    ax = fig.add_subplot(111)
    for label in ["Small", "Medium", "Large"]:
        ax.plot(
            ycut[f"{label.lower()}_pitch_speed"],
            ycut["y"],
            linewidth=2.0,
            label=f"{label} P/W={PITCHES[label]:.3f}",
        )

    ax.set_xlabel("|u| on vertical cut")
    ax.set_ylabel("y")
    ax.set_title(f"Y-cut flow speed at x={x_cut_actual:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTDIR, "y_cut_speed_x_0p50.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    summary_rows = []
    for label, result in results.items():
        row = {
            "case": label,
            "pitch_P_over_W": result["pitch"],
            "clear_gap": result["pitch"] - WAFER_THICKNESS,
            "max_speed": np.nanmax(result["speed"]),
            "inlet_flux": result["inlet_flux"],
            "outlet_flux": result["outlet_flux"],
            "final_velocity_residual": result["residual"],
        }
        for yy in [0.15, 0.20, 0.30]:
            iy = int(np.argmin(np.abs(result["y"] - yy)))
            row[f"speed_at_y_{yy:.2f}"] = result["speed"][iy, ix_cut]
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTDIR, "simulation_summary.csv"),
        index=False,
    )

    print(f"Done. Files are in: {OUTDIR}")


if __name__ == "__main__":
    main()
