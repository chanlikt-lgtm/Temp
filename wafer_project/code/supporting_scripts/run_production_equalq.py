"""Production equal-Q run.

This file used to be a byte-for-byte copy of
``audited_final/equalQ_root_driver.py``.  To avoid two sources of truth it now
simply re-exports and runs the audited driver.  Run either entry point; they
are identical by construction.

    python run_production_equalq.py        # runs all three cases
    WAFER_OUT=/path/to/out python run_production_equalq.py
"""
import os, sys

_AUDITED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'audited_final')
sys.path.insert(0, os.path.abspath(_AUDITED))

from equalQ_root_driver import *          # noqa: F401,F403  (Solver, metrics, converge, run_case, ...)
from equalQ_root_driver import run_case, CASES, OUT, Q_TARGET, N_GAP, ROOT_TOL  # explicit for clarity

if __name__ == '__main__':
    import runpy
    runpy.run_path(os.path.join(os.path.abspath(_AUDITED), 'equalQ_root_driver.py'),
                   run_name='__main__')
