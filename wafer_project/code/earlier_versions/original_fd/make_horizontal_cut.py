import os
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE = '/mnt/data/wafer_pitch_navier_stokes'
MODPATH = os.path.join(BASE, 'wafer_pitch_navier_stokes.py')
spec = importlib.util.spec_from_file_location('wps', MODPATH)
wps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wps)

Y_CUT = 0.20
results = {}
for label, pitch in wps.PITCHES.items():
    print(f'Solving {label} pitch {pitch:.3f} ...', flush=True)
    r = wps.solve_case(pitch)
    results[label] = r
    np.savez_compressed(
        os.path.join(BASE, f'field_{label.lower()}_pitch.npz'),
        x=r['x'], y=r['y'], u=r['u'], v=r['v'], p=r['p'], speed=r['speed'], solid=r['solid']
    )

# horizontal cut at nearest grid row
x = results['Medium']['x']
y = results['Medium']['y']
iy = int(np.argmin(np.abs(y - Y_CUT)))
y_actual = float(y[iy])

cut_df = pd.DataFrame({'x': x})
metrics = []

for label, r in results.items():
    s = r['speed'][iy, :].copy()
    cut_df[f'{label.lower()}_pitch_speed'] = s

    arr_min = r['wafer_centers'][0] - r['pitch']/2
    arr_max = r['wafer_centers'][-1] + r['pitch']/2
    in_array = (x >= arr_min) & (x <= arr_max)
    fluid_on_cut = (~r['solid'][iy, :]) & in_array & np.isfinite(s)

    central_gap_half = (r['pitch'] - wps.WAFER_THICKNESS)/2
    central_gap = (np.abs(x - 0.5) < central_gap_half) & (~r['solid'][iy, :]) & np.isfinite(s)

    metrics.append({
        'case': label,
        'pitch_P_over_W': r['pitch'],
        'clear_gap': r['pitch'] - wps.WAFER_THICKNESS,
        'y_cut': y_actual,
        'max_speed_on_full_cut': float(np.nanmax(s)),
        'mean_speed_in_array_fluid': float(np.nanmean(s[fluid_on_cut])),
        'mean_speed_central_gap': float(np.nanmean(s[central_gap])),
        'speed_at_x_0p50': float(s[np.argmin(np.abs(x-0.5))]),
    })

cut_csv = os.path.join(BASE, 'horizontal_cut_speed_y_0p20.csv')
cut_df.to_csv(cut_csv, index=False)
metrics_df = pd.DataFrame(metrics)
metrics_csv = os.path.join(BASE, 'horizontal_cut_summary_y_0p20.csv')
metrics_df.to_csv(metrics_csv, index=False)

# Overlay comparison. NaN in solid wafers creates physical breaks in each profile.
fig, ax = plt.subplots(figsize=(9.3, 4.8))
for label in ['Small', 'Medium', 'Large']:
    ax.plot(x, cut_df[f'{label.lower()}_pitch_speed'], lw=1.8,
            label=f'{label} pitch P/W={wps.PITCHES[label]:.3f}')
ax.set_xlim(0, wps.W)
ax.set_xlabel('x')
ax.set_ylabel(r'$|u|$ at horizontal cut')
ax.set_title(f'Horizontal flow-speed cut at y={y_actual:.3f} (direct PDE solution, no fitting)')
ax.grid(True, alpha=0.25)
ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.14), frameon=True)
fig.tight_layout()
overlay_path = os.path.join(BASE, 'horizontal_cut_speed_y_0p20.png')
fig.savefig(overlay_path, dpi=200, bbox_inches='tight')
plt.close(fig)

# Small-multiple profiles with wafer intersections shaded.
fig, axes = plt.subplots(3, 1, figsize=(9.4, 8.0), sharex=True, sharey=True)
for ax, label in zip(axes, ['Small', 'Medium', 'Large']):
    r = results[label]
    s = r['speed'][iy, :]
    ax.plot(x, s, lw=1.9)
    for xc in r['wafer_centers']:
        ax.axvspan(xc-wps.WAFER_THICKNESS/2, xc+wps.WAFER_THICKNESS/2, alpha=0.22)
    ax.set_ylabel(r'$|u|$')
    ax.grid(True, alpha=0.22)
    ax.set_title(f'{label} pitch P/W={r["pitch"]:.3f}, clear gap={r["pitch"]-wps.WAFER_THICKNESS:.3f}')
axes[-1].set_xlabel('x')
fig.suptitle(f'Horizontal cut through wafer region: y={y_actual:.3f}', y=0.995)
fig.tight_layout(rect=(0,0,1,0.985))
detail_path = os.path.join(BASE, 'horizontal_cut_profiles_y_0p20.png')
fig.savefig(detail_path, dpi=200, bbox_inches='tight')
plt.close(fig)

# Regenerate field plots with the horizontal cut marked.
for label, r in results.items():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    pcm = ax.pcolormesh(r['x'], r['y'], r['speed'], shading='auto', cmap='viridis',
                        vmin=0.0, vmax=wps.INLET_SPEED)
    um = np.ma.array(r['u'], mask=r['solid'])
    vm = np.ma.array(r['v'], mask=r['solid'])
    ax.streamplot(r['x'], r['y'], um, vm, density=1.5, linewidth=0.75,
                  arrowsize=0.85, color='white')
    for xc in r['wafer_centers']:
        ax.add_patch(patches.Rectangle(
            (xc-wps.WAFER_THICKNESS/2, wps.WAFER_Y0),
            wps.WAFER_THICKNESS, wps.WAFER_Y1-wps.WAFER_Y0,
            facecolor='white', edgecolor='black', linewidth=0.9, zorder=5))
    ax.scatter(r['inlet_centers'], np.zeros_like(r['inlet_centers']), marker='^', s=36,
               color='red', zorder=6, label='Inlets')
    ax.axhline(y_actual, ls='--', lw=1.3, color='white', alpha=0.95,
               label=f'Horizontal cut y={y_actual:.2f}')
    ax.set_xlim(0, wps.W); ax.set_ylim(0, wps.H)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'{label} pitch: P/W={r["pitch"]:.3f}  |u| + streamlines + wafer mask')
    cb = fig.colorbar(pcm, ax=ax); cb.set_label(r'$|u|$')
    ax.legend(loc='upper right')
    fig.tight_layout()
    pth = os.path.join(BASE, f'navier_stokes_{label.lower()}_pitch_horizontal_cut.png')
    fig.savefig(pth, dpi=200, bbox_inches='tight')
    plt.close(fig)

print('Horizontal cut y =', y_actual)
print(metrics_df.to_string(index=False))
print('Wrote', cut_csv)
print('Wrote', metrics_csv)
