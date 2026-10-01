# No-fit Poisson / Green-function / breakdown Python code

This folder collects the Python scripts developed in the conversation.

## Philosophy

No device-curve fitting is used. The workflow is based on:

1. depleted Poisson electrostatics,
2. analytical Green-function / quasi-2-D reduction where applicable,
3. direct sparse finite-difference solution of the original 2-D Poisson PDE,
4. electric field from `E = -grad(phi)`,
5. Fulop impact-ionization law `alpha_eff = A_F * E^7`,
6. avalanche condition `integral alpha_eff ds = 1`.

No digitized Fig. 3 data, regression scaling, fitted critical field, or fitted
turning point is used.

## Files

- `01_fig3_no_fit_analytic.py`
  Earliest no-fit analytical reconstruction of Fig. 3-type BV and drift resistance.

- `02_green_function_closed_form.py`
  Green-function / exact finite-length J7 closed-form breakdown formulation.

- `03_breakdown_thickness_sweeps.py`
  Vertical/horizontal breakdown, turning points, BOX-thickness and SOI-thickness sweeps.

- `04_pde_to_vbd_sweep_quasi2d.py`
  Standalone chain from the Poisson-derived reduced surface PDE to `VBD(N1)`.

- `05_pde_efield_2d3d_three_cases.py`
  Quasi-2-D analytical reconstruction of potential/field for low, medium, high N1 cases.

- `06_all_in_one_quasi2d_potential_efield.py`
  All-in-one quasi-2-D script: PDE reduction -> Green solution -> BV -> 2-D/3-D potential and field.

- `07_direct_2d_poisson_fd_to_vbd.py`
  Direct sparse finite-difference solution of the original 2-D Poisson PDE, followed by
  Ex, Ey, |E|, horizontal avalanche BV, vertical cap, and VBD-vs-N1 sweep.

- `08_compare_surface_breakdown_paths.py`
  No-fit comparison of:
    * parabolic quasi-2-D closed form
    * direct 2-D finite-difference Poisson solution
    * independent 1-D vertical cap

## Install

```bash
pip install -r requirements.txt
```

## Suggested execution order

```bash
python 04_pde_to_vbd_sweep_quasi2d.py
python 05_pde_efield_2d3d_three_cases.py
python 07_direct_2d_poisson_fd_to_vbd.py
python 08_compare_surface_breakdown_paths.py
```

The two most important scripts for the final no-fit comparison are:

- `07_direct_2d_poisson_fd_to_vbd.py`
- `08_compare_surface_breakdown_paths.py`
