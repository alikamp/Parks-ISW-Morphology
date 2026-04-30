#!/usr/bin/env python3
"""
ISW Morphology — Aperture Scaling Test
========================================

Diego's suggestion: test at 1.0x, 1.5x, and 2.0x the baseline aperture
to see if the signal extends beyond the virial radius.

Baseline: 15' disk, 15'-45' annulus
1.5x:     22.5' disk, 22.5'-67.5' annulus
2.0x:     30' disk, 30'-90' annulus

Author: Alika M. Parks
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os, json

from astropy.io import fits
import healpy as hp

OUT = "isw_results_aperture"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13, 'axes.labelsize': 15,
    'axes.titlesize': 15, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})

# ── Load data ────────────────────────────────────────────────────────────
print("="*65)
print("APERTURE SCALING TEST")
print("="*65)

hdu = fits.open("isw_data/psz2.fits")
psz2 = hdu[1].data

ra = psz2['RA']; dec = psz2['DEC']; z = psz2['REDSHIFT']
snr = psz2['SNR']; val = psz2['VALIDATION']
M500 = psz2['MSZ']; Y_SZ = psz2['Y5R500']
glat = psz2['GLAT']

good = (val >= 20) & np.isfinite(z) & (z > 0) & \
       np.isfinite(M500) & (M500 > 0) & \
       np.isfinite(Y_SZ) & (Y_SZ > 0) & \
       (np.abs(glat) > 15)

ra = ra[good]; dec = dec[good]; z = z[good]
snr = snr[good]; M500 = M500[good]; Y_SZ = Y_SZ[good]
N = len(ra)
print(f"  {N} clusters")

# Y-M residual morphology
log_Y = np.log10(Y_SZ); log_M = np.log10(M500)
slope, intercept, _, _, _ = stats.linregress(log_M, log_Y)
Y_pred = 10**(intercept + slope * log_M)
Y_resid = (Y_SZ - Y_pred) / Y_pred
morph_dis = Y_resid < 0
morph_rel = Y_resid > 0

# Load CMB
print("  Loading CMB...")
CMB = None
for mf in ["isw_data/smica_nosz.fits", "isw_data/smica.fits"]:
    if os.path.exists(mf) and os.path.getsize(mf) > 1e8:
        CMB = hp.read_map(mf, field=0, verbose=False)
        if np.nanstd(CMB[CMB != hp.UNSEEN]) < 0.01:
            CMB *= 1e6
        break
NSIDE = hp.npix2nside(len(CMB))

# ── Aperture photometry at multiple scales ───────────────────────────────

def ap_phot(ra_d, dec_d, disk_am, ann_am):
    theta = np.radians(90 - dec_d); phi = np.radians(ra_d)
    vec = hp.ang2vec(theta, phi)
    dp = hp.query_disc(NSIDE, vec, np.radians(disk_am / 60))
    op = hp.query_disc(NSIDE, vec, np.radians(ann_am / 60))
    ap = np.setdiff1d(op, dp)
    Td = CMB[dp]; Ta = CMB[ap]
    gd = np.isfinite(Td) & (Td != hp.UNSEEN) & (np.abs(Td) < 1e4)
    ga = np.isfinite(Ta) & (Ta != hp.UNSEEN) & (np.abs(Ta) < 1e4)
    if gd.sum() < 10 or ga.sum() < 10: return np.nan
    return np.mean(Td[gd]) - np.mean(Ta[ga])

# Define aperture scales
scales = [
    {"label": "1.0x (baseline)", "disk": 15.0, "ann": 45.0, "factor": 1.0},
    {"label": "1.5x", "disk": 22.5, "ann": 67.5, "factor": 1.5},
    {"label": "2.0x", "disk": 30.0, "ann": 90.0, "factor": 2.0},
]

# Run all scales
all_DT = {}
for sc in scales:
    print(f"\n  Measuring at {sc['label']}: disk={sc['disk']}', annulus={sc['disk']}'-{sc['ann']}'...")
    DT = np.zeros(N)
    for i in range(N):
        DT[i] = ap_phot(ra[i], dec[i], sc['disk'], sc['ann'])
        if (i+1) % 300 == 0:
            print(f"    {i+1}/{N}...")
    all_DT[sc['label']] = DT
    valid = np.isfinite(DT)
    print(f"    Valid: {valid.sum()}/{N}")


# ── Analysis at each scale ───────────────────────────────────────────────

print("\n" + "="*65)
print("RESULTS BY APERTURE SCALE")
print("="*65)

# Physical scale at z=0.125 (midpoint of key bin)
# At z=0.125, 1 arcmin ≈ 0.147 Mpc (approximate)
MPC_PER_ARCMIN = 0.147

results_all = []

print(f"\n  --- FULL SAMPLE (all z) ---")
print(f"\n  {'Scale':<20} {'Disk':>6} {'Ann':>10} {'Disk Mpc':>9} {'n_r':>5} {'n_d':>5} "
      f"{'DT_r':>7} {'DT_d':>7} {'diff':>7} {'err':>6} {'sigma':>6} {'p':>8}")
print("  " + "-"*95)

for sc in scales:
    DT = all_DT[sc['label']]
    valid = np.isfinite(DT)
    
    m_r = morph_rel & valid
    m_d = morph_dis & valid
    dt_r = DT[m_r]; dt_d = DT[m_d]
    
    diff = dt_d.mean() - dt_r.mean()
    err = np.sqrt(dt_d.var()/len(dt_d) + dt_r.var()/len(dt_r))
    t, p = stats.ttest_ind(dt_d, dt_r, equal_var=False)
    
    disk_mpc = sc['disk'] * MPC_PER_ARCMIN
    ann_str = f"{sc['disk']:.0f}'-{sc['ann']:.0f}'"
    
    star = "**" if p < 0.01 else " *" if p < 0.05 else ""
    print(f"  {sc['label']:<20} {sc['disk']:>5.1f}' {ann_str:>10} {disk_mpc:>8.1f} "
          f"{len(dt_r):>5} {len(dt_d):>5} {dt_r.mean():>7.2f} {dt_d.mean():>7.2f} "
          f"{diff:>7.2f} {err:>6.2f} {abs(t):>5.1f}σ {p:>8.4f} {star}")
    
    results_all.append({
        'scale': sc['label'], 'factor': sc['factor'],
        'disk_arcmin': sc['disk'], 'disk_mpc': round(disk_mpc, 1),
        'diff': round(float(diff), 2), 'err': round(float(err), 2),
        'sigma': round(float(abs(t)), 2), 'p': round(float(p), 4),
        'n_r': int(len(dt_r)), 'n_d': int(len(dt_d)),
        'DT_rel': round(float(dt_r.mean()), 2),
        'DT_dis': round(float(dt_d.mean()), 2),
    })

# Key bin: z = 0.10-0.15
print(f"\n  --- KEY BIN: z = [0.10, 0.15) ---")
print(f"\n  {'Scale':<20} {'Disk':>6} {'Disk Mpc':>9} {'n_r':>5} {'n_d':>5} "
      f"{'DT_r':>7} {'DT_d':>7} {'diff':>7} {'err':>6} {'sigma':>6} {'p':>8}")
print("  " + "-"*85)

results_key = []

for sc in scales:
    DT = all_DT[sc['label']]
    valid = np.isfinite(DT)
    z_mask = (z >= 0.10) & (z < 0.15)
    
    m_r = morph_rel & valid & z_mask
    m_d = morph_dis & valid & z_mask
    
    nr = m_r.sum(); nd = m_d.sum()
    if nr < 5 or nd < 5:
        print(f"  {sc['label']:<20} insufficient data")
        continue
    
    dt_r = DT[m_r]; dt_d = DT[m_d]
    diff = dt_d.mean() - dt_r.mean()
    err = np.sqrt(dt_d.var()/nd + dt_r.var()/nr)
    t, p = stats.ttest_ind(dt_d, dt_r, equal_var=False)
    
    disk_mpc = sc['disk'] * MPC_PER_ARCMIN
    
    star = "**" if p < 0.01 else " *" if p < 0.05 else ""
    print(f"  {sc['label']:<20} {sc['disk']:>5.1f}' {disk_mpc:>8.1f} "
          f"{nr:>5} {nd:>5} {dt_r.mean():>7.2f} {dt_d.mean():>7.2f} "
          f"{diff:>7.2f} {err:>6.2f} {abs(t):>5.1f}σ {p:>8.4f} {star}")
    
    results_key.append({
        'scale': sc['label'], 'factor': sc['factor'],
        'disk_arcmin': sc['disk'], 'disk_mpc': round(disk_mpc, 1),
        'diff': round(float(diff), 2), 'err': round(float(err), 2),
        'sigma': round(float(abs(t)), 2), 'p': round(float(p), 4),
        'n_r': int(nr), 'n_d': int(nd),
    })

# Broader redshift range z = 0.05-0.20
print(f"\n  --- BROADER: z = [0.05, 0.20) ---")
print(f"\n  {'Scale':<20} {'n_r':>5} {'n_d':>5} {'diff':>7} {'sigma':>6} {'p':>8}")
print("  " + "-"*55)

for sc in scales:
    DT = all_DT[sc['label']]
    valid = np.isfinite(DT)
    z_mask = (z >= 0.05) & (z < 0.20)
    
    m_r = morph_rel & valid & z_mask
    m_d = morph_dis & valid & z_mask
    nr = m_r.sum(); nd = m_d.sum()
    
    if nr < 5 or nd < 5: continue
    
    dt_r = DT[m_r]; dt_d = DT[m_d]
    diff = dt_d.mean() - dt_r.mean()
    err = np.sqrt(dt_d.var()/nd + dt_r.var()/nr)
    t, p = stats.ttest_ind(dt_d, dt_r, equal_var=False)
    
    star = "**" if p < 0.01 else " *" if p < 0.05 else ""
    print(f"  {sc['label']:<20} {nr:>5} {nd:>5} {diff:>7.2f} {abs(t):>5.1f}σ {p:>8.4f} {star}")


# ── Figure ───────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("GENERATING FIGURE")
print("="*65)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel a: Full sample
ax = axes[0]
factors = [r['factor'] for r in results_all]
diffs = [r['diff'] for r in results_all]
errs = [r['err'] for r in results_all]
sigs = [r['sigma'] for r in results_all]
colors = ['#e74c3c' if s > 2 else '#f39c12' if s > 1.5 else '#95a5a6' for s in sigs]

ax.errorbar(factors, diffs, yerr=errs, fmt='o-', color='#3498db',
            capsize=8, ms=12, lw=2.5, mfc='white', mew=2, zorder=5)
for i, (f_val, d, s) in enumerate(zip(factors, diffs, sigs)):
    ax.annotate(f'{s:.1f}σ', (f_val, d), textcoords="offset points",
                xytext=(12, 8), fontsize=12, fontweight='bold')
ax.axhline(0, color='gray', ls='-', lw=1)
ax.set_xlabel('Aperture Scale Factor')
ax.set_ylabel('$\\Delta T_{dis} - \\Delta T_{rel}$ ($\\mu$K)')
ax.set_title('(a) Full Sample', fontweight='bold')
ax.set_xticks([1.0, 1.5, 2.0])
ax.set_xticklabels(['1.0×\n(15\')', '1.5×\n(22.5\')', '2.0×\n(30\')'])

# Panel b: Key bin z=0.10-0.15
ax = axes[1]
if results_key:
    factors_k = [r['factor'] for r in results_key]
    diffs_k = [r['diff'] for r in results_key]
    errs_k = [r['err'] for r in results_key]
    sigs_k = [r['sigma'] for r in results_key]
    
    ax.errorbar(factors_k, diffs_k, yerr=errs_k, fmt='s-', color='#e74c3c',
                capsize=8, ms=12, lw=2.5, mfc='white', mew=2, zorder=5)
    for i, (f_val, d, s) in enumerate(zip(factors_k, diffs_k, sigs_k)):
        ax.annotate(f'{s:.1f}σ', (f_val, d), textcoords="offset points",
                    xytext=(12, 8), fontsize=12, fontweight='bold')
ax.axhline(0, color='gray', ls='-', lw=1)
ax.set_xlabel('Aperture Scale Factor')
ax.set_ylabel('$\\Delta T_{dis} - \\Delta T_{rel}$ ($\\mu$K)')
ax.set_title('(b) Key Bin: z = [0.10, 0.15)', fontweight='bold')
ax.set_xticks([1.0, 1.5, 2.0])
ax.set_xticklabels(['1.0×\n(15\')', '1.5×\n(22.5\')', '2.0×\n(30\')'])

# Panel c: Physical scale context
ax = axes[2]
# Show what these apertures correspond to physically
z_ref = 0.125
scales_mpc = [sc['disk'] * MPC_PER_ARCMIN for sc in scales]
r_vir_mpc = 1.5  # typical virial radius for massive cluster

ax.barh([0, 1, 2], scales_mpc, height=0.5, color=['#3498db', '#2ecc71', '#9b59b6'],
        edgecolor='white', alpha=0.8)
ax.axvline(r_vir_mpc, color='#e74c3c', ls='--', lw=2, label=f'$R_{{vir}}$ ≈ {r_vir_mpc} Mpc')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['1.0× (15\')', '1.5× (22.5\')', '2.0× (30\')'])
ax.set_xlabel('Projected Radius (Mpc) at z = 0.125')
ax.set_title('(c) Physical Scale at Key Redshift', fontweight='bold')
ax.legend(fontsize=11)

fig.suptitle('Aperture Scaling Test — Does the Signal Extend Beyond the Virial Radius?',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_aperture_scaling.png')
plt.close()
print("  -> fig_aperture_scaling.png")

# ── Save ─────────────────────────────────────────────────────────────────
results = {
    "full_sample": results_all,
    "key_bin_z010_015": results_key,
    "physical_scale_note": f"At z=0.125: 1 arcmin ≈ {MPC_PER_ARCMIN} Mpc",
    "virial_radius_typical": "~1.5 Mpc for M500 ~ 5e14 Msun",
}
with open(f'{OUT}/aperture_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results: {OUT}/aperture_results.json")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)

print("\n  Does the signal extend beyond the virial radius?")
if results_key:
    for r in results_key:
        beyond = "YES (beyond R_vir)" if r['disk_mpc'] > 1.5 else "at R_vir scale"
        print(f"    {r['scale']}: disk={r['disk_mpc']} Mpc — diff={r['diff']} muK, "
              f"{r['sigma']}σ — {beyond}")

print("\n  Signal stability across scales:")
if len(results_all) >= 3:
    diffs_check = [r['diff'] for r in results_all]
    all_negative = all(d < 0 for d in diffs_check)
    print(f"    All scales show same sign: {'YES' if all_negative else 'NO'}")
    print(f"    Range of differences: [{min(diffs_check):.1f}, {max(diffs_check):.1f}] muK")

print("\nDone.")
