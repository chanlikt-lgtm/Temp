"""Grid-refinement + timestep-halving study for the Small case (corrected geom).

Grid study: re-root the equal-Q inlet speed at n_gap = 12, 16, 20 and compare
U_in, Q_rack, delta_p, R_h, central_peak_v across meshes.

Timestep study: at the n_gap=20 converged U_in, advance to steady state at the
nominal dt and at dt/2 (fixed U) and compare observables -- isolating temporal
discretization error.

Writes grid_study_small.json and timestep_study_small.json.
"""
import sys, os, time, json
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'code', 'audited_final'))
from equalQ_root_driver import (Solver, NU, Q_TARGET, ROOT_TOL,
                                nominal_dt, converge, metrics, snapshot)

PITCH = 0.040
BRACKET = (0.55, 0.82)

def root_equalQ(pitch, n_gap, dt, min0, max0):
    """Bracketed secant with enforced-convergence final + correction."""
    lo, hi = BRACKET
    s = Solver(pitch, n_gap)
    mlo, slo, _, _, _ = converge(s, lo, dt, None, min_steps=min0, max_steps=max0, require_converged=False)
    mhi, shi, _, _, _ = converge(s, hi, dt, slo, min_steps=min0, max_steps=max0, require_converged=False)
    final = None
    for it in range(8):
        ql, qh = mlo['Q_rack'], mhi['Q_rack']
        usec = lo + (Q_TARGET-ql)*(hi-lo)/(qh-ql); span = hi-lo
        umid = min(max(usec, lo+0.20*span), hi-0.20*span)
        state = slo if abs(umid-lo) <= abs(hi-umid) else shi
        mm, sm, _, rr, _ = converge(s, umid, dt, state, min_steps=min0, max_steps=max0, require_converged=False)
        err = mm['Q_rack']/Q_TARGET-1
        final = (umid, mm, sm)
        if abs(err) <= ROOT_TOL: break
        if mm['Q_rack'] < Q_TARGET: lo, mlo, slo = umid, mm, sm
        else: hi, mhi, shi = umid, mm, sm
    U, mm, sm = final
    mf, sf, _, rf, nf = converge(s, U, dt, sm, min_steps=min0, max_steps=max0, require_converged=True)
    errf = mf['Q_rack']/Q_TARGET-1
    tries = 0
    while abs(errf) > ROOT_TOL and tries < 4:
        tries += 1
        if mf['Q_rack'] < Q_TARGET: lo, mlo, slo = U, mf, sf
        else: hi, mhi, shi = U, mf, sf
        ql, qh = mlo['Q_rack'], mhi['Q_rack']
        U = lo + (Q_TARGET-ql)*(hi-lo)/(qh-ql)
        U = min(max(U, lo+0.10*(hi-lo)), hi-0.10*(hi-lo))
        state = slo if abs(U-lo) <= abs(hi-U) else shi
        mf, sf, _, rf, nf = converge(s, U, dt, state, min_steps=min0, max_steps=max0, require_converged=True)
        errf = mf['Q_rack']/Q_TARGET-1
    return s, U, mf, rf

def grid_study():
    rows = []
    for n_gap in (12, 16, 20):
        t0 = time.time()
        s = Solver(PITCH, n_gap); dt = nominal_dt(s)
        # coarser grids settle in fewer steps; scale caps with n_gap
        cap = int(30000 * (n_gap/20.0)**2) + 4000
        s, U, mf, rf = root_equalQ(PITCH, n_gap, dt, min0=3000, max0=cap)
        rows.append({'n_gap': n_gap, 'Nx': int(s.g['Nx']), 'Ny': int(s.g['Ny']),
                     'dx_min': float(s.g['dx'].min()), 'dt': float(dt),
                     'U_in': float(U), 'Q_rack': float(mf['Q_rack']),
                     'Q_err_pct': float(100*(mf['Q_rack']/Q_TARGET-1)),
                     'delta_p_rack': float(mf['delta_p_rack']), 'R_h': float(mf['R_h']),
                     'central_peak_v': float(mf['central_peak_v']),
                     'final_res': float(rf), 'elapsed_s': time.time()-t0})
        json.dump(rows, open(os.path.join(_HERE, 'grid_study_small.json'), 'w'), indent=2)
        print('grid n_gap=%d U=%.5f R_h=%.4f peakv=%.5f (%.0fs)' %
              (n_gap, U, mf['R_h'], mf['central_peak_v'], rows[-1]['elapsed_s']), flush=True)
    return rows

def timestep_study(U_fixed):
    s = Solver(PITCH, 20); dt0 = nominal_dt(s)
    out = []
    for frac in (1.0, 0.5):
        t0 = time.time()
        dt = dt0*frac
        cap = int(30000/frac) + 4000
        m, _, _, r, n = converge(s, U_fixed, dt, None, min_steps=4000, max_steps=cap, require_converged=True)
        out.append({'dt_frac': frac, 'dt': float(dt), 'steps': int(n),
                    'Q_rack': float(m['Q_rack']), 'delta_p_rack': float(m['delta_p_rack']),
                    'central_peak_v': float(m['central_peak_v']), 'R_h': float(m['R_h']),
                    'final_res': float(r), 'elapsed_s': time.time()-t0})
        json.dump(out, open(os.path.join(_HERE, 'timestep_study_small.json'), 'w'), indent=2)
        print('dt_frac=%.2f steps=%d Q=%.6e peakv=%.5f dp=%.5e (%.0fs)' %
              (frac, n, m['Q_rack'], m['central_peak_v'], m['delta_p_rack'], out[-1]['elapsed_s']), flush=True)
    return out

if __name__ == '__main__':
    t0 = time.time()
    grid = grid_study()
    U20 = [r['U_in'] for r in grid if r['n_gap'] == 20][0]
    timestep_study(U20)
    print('TOTAL %.0fs' % (time.time()-t0), flush=True)
    print('DONE', flush=True)
