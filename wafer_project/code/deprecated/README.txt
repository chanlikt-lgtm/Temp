ARCHIVAL / DEPRECATED scripts.

These scripts retain hard-coded absolute paths (e.g. /mnt/data/...) and other
assumptions from the original authoring environment. They are kept for
provenance of how the earlier figures/reports/validation outputs were produced,
but they are NOT portable and are NOT part of the reproducible workflow.

For reproduction use, instead:
  - audited_final/equalQ_root_driver.py   (production equal-Q driver, portable)
  - supporting_scripts/run_one_case_equalq.py, run_mesh_small.py, run_production_equalq.py
  - ../verification/verify_case.py and study_small.py
