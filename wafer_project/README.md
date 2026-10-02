# Wafer-rack equal-Q Navier–Stokes — corrected bundle, verification & report

2-D incompressible Navier–Stokes solver for flow through a rack of six wafers.
Geometry-aligned MAC staggered grid, fractional-step (projection) method,
first-order donor-cell upwind advection, explicit second-order diffusion. An
equal-rack-flow constraint fixes each pitch's inlet speed by a bracketed scalar
root solve (each evaluation is advanced to a tightened, enforced steady-state gate
— an accepted steady state, not an exact converged state).

## Layout

- `code/` — corrected Python bundle
  - `audited_final/` — production entry points (`mac_cfd_adaptive.py`, `equalQ_root_driver.py`), portable
  - `supporting_scripts/` — portable runners (`run_production_equalq.py`, `run_one_case_equalq.py`, `run_mesh_small.py`)
  - `deprecated/` — archival scripts that still contain hard-coded `/mnt/data` paths from the original environment (NOT portable; kept for provenance)
  - `earlier_versions/` — retained for provenance only
- `verification/` — reproducibility harness (`verify_case.py`; same `Solver`, not an independent implementation), grid/timestep study (`study_small.py`), plot scripts, `*_verify.json`
- `figures/` — rendered PNGs (fields, streamlines, pressure, y=0.20 cuts)
- `report/` — `wafer_report.tex` (+ `Makefile`); first page carries the nomenclature

## Corrections applied

**Numerical (change the computed results):**
1. **Inlet grid alignment** — inlet slot edges are now exact x-grid faces, so every discrete slot is 0.0150 m (total 0.0750 m) for all pitches. The old centre-in-slot test undersized Medium/Large slots to ~0.0135/0.0130 m (up to 13%), inflating their `U_in`.
2. **Convergence gate tightened + enforced** — residual 3e-5→1e-5, drift 1e-3→2e-4 (now incl. Δp), raises at the step cap instead of returning silently; final polish is convergence-enforced with a secant/bisection correction.

**Non-numerical:**
3. Made the active workflow portable (`audited_final/`, the three `supporting_scripts/` runners, and `verification/`): no hard-coded `/mnt/data` or scratchpad paths; output dir is `$WAFER_OUT` (default: script dir), created lazily. Legacy one-off scripts that still assume `/mnt/data` were moved to `code/deprecated/` as archival.
4. Fixed the self-test: `Solver(0.06, 0.0025)` → `Solver(0.06, 20)`.
5. De-duplicated the equal-Q drivers (`run_production_equalq.py` re-export; shared `metrics`).
6. Relabelled stale legacy-FD speeds (0.5605/0.3000/0.1833); documented the near-wall v y-diffusion asymmetry.
7. Added `code/gen_checksums.sh`; `SHA256SUMS.txt` regenerated for the corrected bundle.

Baseline (pre-correction) state is frozen on branch `audited-baseline-v1`.

## Reproduce

```bash
pip install numpy scipy pandas numba matplotlib
cd verification
python verify_case.py Small    # also Medium, Large  -> *_verify.json + *_verify.npz
python plot_all.py && python plot_ycut_overlay.py && python plot_extras.py
```

## Reproduction results (corrected code, same solver; Q_target = 0.0055463)

Corrected geometry (0.0750 m grid-aligned inlet), enforced convergence:

| Case | P | U_in | Q_rack | (Q_rack−Q0)/Q0 | Δp | R_h | gap/inlet | max mass err |
|------|-----|--------|-----------|---------|----------|---------|------|---------|
| Small | 0.040 | 0.69381 | 5.5469e-3 | +0.011% | 0.081760 | 14.740 | 10.7% | 8.9e-14 % |
| Medium | 0.060 | 0.35199 | 5.5468e-3 | +0.010% | 0.015134 | 2.7283 | 21.0% | 7.8e-14 % |
| Large | 0.080 | 0.20643 | 5.5472e-3 | +0.016% | 0.005473 | 0.9867 | 35.8% | 6.7e-14 % |

- Mass conservation holds to roundoff; Q_rack within the 0.075% root tolerance.
- Resistance ordering: R_h(Small) > R_h(Medium) > R_h(Large) = 14.740 > 2.728 > 0.987.
- Inlet alignment lowered `U_in` for Medium (−9.4%) and Large (−13%); Small was already slot-aligned. `R_h` (rack-only ratio) is essentially unchanged from baseline.
- `gap/inlet` = fraction of inlet flow through the 5 interior gaps: equal rack flow is **not** equal total inlet flow.

**Caveats:** this is a reproducibility check (same `Solver`), not an independent
implementation. R_h is pressure-tap-definition dependent (Large: 0.987 at the
default planes, +8.4% / −4.2% at planes ±2 cells). Grid/timestep convergence is
demonstrated for **Small** only (timestep <0.02%; R_h ~0.6% residual at the
production mesh → ~1%); Medium/Large absolute values carry comparable
unquantified grid uncertainty, though the ordering is robust. See the report
(`report/wafer_report.pdf`) for details.

## Build the report

```bash
cd report && make      # needs a TeX toolchain (pdflatex/latexmk)
```
