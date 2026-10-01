# Wafer-rack equal-Q Navier–Stokes — corrected bundle, verification & report

2-D incompressible Navier–Stokes solver for flow through a rack of six wafers.
Geometry-aligned MAC staggered grid, fractional-step (projection) method,
first-order donor-cell upwind advection, explicit second-order diffusion. An
equal-rack-flow constraint fixes each pitch's inlet speed by a bracketed scalar
root solve (every evaluation is a converged CFD run).

## Layout

- `code/` — corrected Python bundle
  - `audited_final/` — production entry points (`mac_cfd_adaptive.py`, `equalQ_root_driver.py`)
  - `supporting_scripts/` — report generation, mesh/timestep studies, plotting
  - `earlier_versions/` — retained for provenance only
- `verification/` — independent rerun harness (`verify_case.py`), plot scripts, `*_verify.json`
- `figures/` — rendered PNGs (fields, streamlines, pressure, y=0.20 cuts)
- `report/` — `wafer_report.tex` (+ `Makefile`); first page carries the nomenclature

## Corrections applied (all non-numerical — reproduced physics unchanged)

1. Removed hard-coded `/mnt/data` output paths; output dir is `$WAFER_OUT` (default: CWD), created lazily.
2. Fixed the `mac_cfd_adaptive.py` self-test: `Solver(0.06, 0.0025)` → `Solver(0.06, 20)` (`n_gap` is an integer cell count, not `dt`).
3. De-duplicated the equal-Q drivers (`run_production_equalq.py` is now a thin re-export; single-case runners share one `metrics`).
4. Relabelled the stale legacy-FD inlet speeds (0.5605/0.3000/0.1833) as the earlier finite-difference model.
5. Documented the deliberate near-wall `v` y-diffusion asymmetry in place.

## Reproduce

```bash
pip install numpy scipy pandas numba matplotlib
cd verification
python verify_case.py Small    # also Medium, Large  -> *_verify.json + *_verify.npz
python plot_all.py && python plot_ycut_overlay.py && python plot_extras.py
```

## Independent verification (corrected code; Q_target = 0.0055463)

| Case | P | U_in | Q_rack | (Q_rack−Q0)/Q0 | Δp | R_h | max mass error |
|------|-----|--------|-----------|---------|----------|---------|---------|
| Small | 0.040 | 0.69372 | 5.5440e-3 | −0.041% | 0.081714 | 14.739 | 1.1e-13 % |
| Medium | 0.060 | 0.38844 | 5.5443e-3 | −0.036% | 0.015134 | 2.7297 | 4.4e-14 % |
| Large | 0.080 | 0.23734 | 5.5458e-3 | −0.008% | 0.005473 | 0.9868 | 4.4e-14 % |

- Mass conservation holds to roundoff.
- Q_rack reaches the common target within the 0.075% root tolerance.
- Resistance ordering recovered: R_h(Small) > R_h(Medium) > R_h(Large) = 14.739 > 2.730 > 0.987.

These are **independent reruns** (reported separately from the final production
values); reproduced U_in match the production targets 0.69374 / 0.38871 / 0.23741
within tolerance.

## Build the report

```bash
cd report && make      # needs a TeX toolchain (pdflatex/latexmk)
```
