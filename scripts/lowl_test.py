#!/usr/bin/env python3
"""
ISW Morphology — Low-ℓ Cutoff Test
====================================

Diego's suggestion: apply a low-ℓ cutoff to the CMB map
(as in Hansen et al.) and check if significance changes.

Low multipoles (ℓ < ℓ_cut) correspond to large angular scales
(> 180/ℓ degrees) that add noise without carrying ISW signal
from individual clusters.

Tests at ℓ_cut = 5, 10, 20, 30, 50 and compares against
the unfiltered baseline.

Requires: healpy, astropy, numpy, scipy, matplotlib
Run on Google Colab with Planck data in isw_data/

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

OUT = "isw_results_lowl"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13, 'axes.labelsize': 15,
    'axes.titlesize': 15, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})

# ── Load data ────────────────────────────────────────────────────────────
print("="*65)
print("LOW-ℓ CUTOFF TEST")
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

# Load CMB map
print("  Loading CMB map...")
CMB_RAW = None
for mf in ["isw_data/smica_nosz.fits", "isw_data/smica.fits"]:
    if os.path.exists(mf) and os.path.getsize(mf) > 1e8:
        CMB_RAW = hp.read_map(mf, field=0, verbose=False)
        if np.nanstd(CMB_RAW[CMB_RAW != hp.UNSEEN]) < 0.01:
            CMB_RAW *= 1e6
        print(f"  Loaded {mf}")
        break

NSIDE = hp.npix2nside(len(CMB_RAW))
LMAX = 3 * NSIDE - 1  # maximum ℓ for this resolution
print(f"  NSIDE = {NSIDE}, LMAX = {LMAX}")

# ── Compute spherical harmonic coefficients ONCE ─────────────────────────
print("  Computing spherical harmonic decomposition (this takes a minute)...")

# Mask UNSEEN pixels
mask = CMB_RAW != hp.UNSEEN
cmb_masked = CMB_RAW.copy()
cmb_masked[~mask] = 0

# Get alm coefficients
# Use a lower lmax for speed (we only need up to a few hundred for filtering)
LMAX_USE = min(500, LMAX)
alm = hp.map2alm(cmb_masked, lmax=LMAX_USE)
print(f"  alm computed (lmax={LMAX_USE})")


# ── Filter function ──────────────────────────────────────────────────────

def make_filtered_map(alm_in, l_cut, lmax):
    """
    Remove multipoles below l_cut from the CMB map.
    Returns a new map with only ℓ >= l_cut modes.
    """
    alm_filtered = alm_in.copy()
    
    # Zero out all ℓ < l_cut
    for l in range(l_cut):
        for m in range(l + 1):
            idx = hp.Alm.getidx(lmax, l, m)
            alm_filtered[idx] = 0
    
    # Reconstruct map
    filtered_map = hp.alm2map(alm_filtered, NSIDE, verbose=False)
    return filtered_map


# ── Aperture photometry ──────────────────────────────────────────────────

def ap_phot(cmap, ra_d, dec_d, disk_am=15.0, ann_am=45.0):
    theta = np.radians(90 - dec_d); phi = np.radians(ra_d)
    vec = hp.ang2vec(theta, phi)
    dp = hp.query_disc(NSIDE, vec, np.radians(disk_am / 60))
    op = hp.query_disc(NSIDE, vec, np.radians(ann_am / 60))
    ap = np.setdiff1d(op, dp)
    Td = cmap[dp]; Ta = cmap[ap]
    gd = np.isfinite(Td) & (np.abs(Td) < 1e4)
    ga = np.isfinite(Ta) & (np.abs(Ta) < 1e4)
    if gd.sum() < 10 or ga.sum() < 10: return np.nan
    return np.mean(Td[gd]) - np.mean(Ta[ga])

def measure_all(cmap):
    """Measure DeltaT for all clusters on a given map."""
    DT = np.zeros(N)
    for i in range(N):
        DT[i] = ap_phot(cmap, ra[i], dec[i])
    return DT

def analyze(DT, label):
    """Run the standard morphology split analysis."""
    valid = np.isfinite(DT)
    
    results = {'label': label}
    
    # Full sample
    m_r = morph_rel & valid; m_d = morph_dis & valid
    dt_r = DT[m_r]; dt_d = DT[m_d]
    
    if len(dt_r) < 10 or len(dt_d) < 10:
        print(f"  {label}: insufficient data")
        return None
    
    diff = dt_d.mean() - dt_r.mean()
    err = np.sqrt(dt_d.var()/len(dt_d) + dt_r.var()/len(dt_r))
    t, p = stats.ttest_ind(dt_d, dt_r, equal_var=False)
    
    results['full'] = {
        'diff': round(float(diff), 2), 'err': round(float(err), 2),
        'sigma': round(float(abs(t)), 2), 'p': round(float(p), 4),
        'n_r': int(len(dt_r)), 'n_d': int(len(dt_d)),
        'DT_rel': round(float(dt_r.mean()), 2),
        'DT_dis': round(float(dt_d.mean()), 2),
    }
    
    # Key bin z = 0.10-0.15
    z_mask = (z >= 0.10) & (z < 0.15)
    m_r_z = morph_rel & valid & z_mask
    m_d_z = morph_dis & valid & z_mask
    
    if m_r_z.sum() >= 5 and m_d_z.sum() >= 5:
        dt_rz = DT[m_r_z]; dt_dz = DT[m_d_z]
        diff_z = dt_dz.mean() - dt_rz.mean()
        err_z = np.sqrt(dt_dz.var()/len(dt_dz) + dt_rz.var()/len(dt_rz))
        t_z, p_z = stats.ttest_ind(dt_dz, dt_rz, equal_var=False)
        
        results['key_bin'] = {
            'diff': round(float(diff_z), 2), 'err': round(float(err_z), 2),
            'sigma': round(float(abs(t_z)), 2), 'p': round(float(p_z), 4),
            'n_r': int(len(dt_rz)), 'n_d': int(len(dt_dz)),
        }
    else:
        results['key_bin'] = None
    
    # Broader z = 0.05-0.20
    z_broad = (z >= 0.05) & (z < 0.20)
    m_r_b = morph_rel & valid & z_broad
    m_d_b = morph_dis & valid & z_broad
    
    if m_r_b.sum() >= 10 and m_d_b.sum() >= 10:
        dt_rb = DT[m_r_b]; dt_db = DT[m_d_b]
        diff_b = dt_db.mean() - dt_rb.mean()
        err_b = np.sqrt(dt_db.var()/len(dt_db) + dt_rb.var()/len(dt_rb))
        t_b, p_b = stats.ttest_ind(dt_db, dt_rb, equal_var=False)
        
        results['broad_bin'] = {
            'diff': round(float(diff_b), 2), 'err': round(float(err_b), 2),
            'sigma': round(float(abs(t_b)), 2), 'p': round(float(p_b), 4),
        }
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# RUN AT MULTIPLE ℓ CUTOFFS
# ══════════════════════════════════════════════════════════════════════════════

l_cuts = [0, 5, 10, 20, 30, 50]
# ℓ = 0 means no filtering (baseline)

all_results = []

for l_cut in l_cuts:
    label = f"ℓ_cut = {l_cut}" if l_cut > 0 else "No filter (baseline)"
    angular_scale = f">{180/max(l_cut,1):.0f}°" if l_cut > 0 else "all scales"
    
    print(f"\n{'='*65}")
    print(f"  {label} (removes scales {angular_scale})")
    print(f"{'='*65}")
    
    if l_cut == 0:
        # Use raw map
        cmap = CMB_RAW.copy()
        print("  Using unfiltered map")
    else:
        print(f"  Filtering: removing ℓ < {l_cut}...")
        cmap = make_filtered_map(alm, l_cut, LMAX_USE)
    
    print("  Measuring DeltaT...")
    DT = measure_all(cmap)
    valid_count = np.isfinite(DT).sum()
    print(f"  Valid: {valid_count}/{N}")
    
    r = analyze(DT, label)
    if r:
        r['l_cut'] = l_cut
        r['angular_scale'] = angular_scale
        all_results.append(r)
        
        # Print summary
        f = r['full']
        print(f"  Full sample: diff = {f['diff']:.2f} ± {f['err']:.2f} μK, "
              f"{f['sigma']:.1f}σ, p = {f['p']:.4f}")
        
        if r.get('key_bin'):
            k = r['key_bin']
            print(f"  z=[0.10,0.15): diff = {k['diff']:.2f} ± {k['err']:.2f} μK, "
                  f"{k['sigma']:.1f}σ, p = {k['p']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("COMPARISON: SIGNIFICANCE vs ℓ CUTOFF")
print("="*65)

print(f"\n  --- FULL SAMPLE ---")
print(f"\n  {'ℓ_cut':<10} {'Removes':<15} {'diff':>8} {'err':>7} {'sigma':>6} {'p':>8}")
print("  " + "-"*58)
for r in all_results:
    f = r['full']
    star = "**" if f['p'] < 0.01 else " *" if f['p'] < 0.05 else ""
    print(f"  {r['l_cut']:<10} {r['angular_scale']:<15} {f['diff']:>8.2f} {f['err']:>7.2f} "
          f"{f['sigma']:>5.1f}σ {f['p']:>8.4f} {star}")

print(f"\n  --- KEY BIN: z = [0.10, 0.15) ---")
print(f"\n  {'ℓ_cut':<10} {'diff':>8} {'err':>7} {'sigma':>6} {'p':>8}")
print("  " + "-"*42)
for r in all_results:
    if r.get('key_bin'):
        k = r['key_bin']
        star = "**" if k['p'] < 0.01 else " *" if k['p'] < 0.05 else ""
        print(f"  {r['l_cut']:<10} {k['diff']:>8.2f} {k['err']:>7.2f} "
              f"{k['sigma']:>5.1f}σ {k['p']:>8.4f} {star}")

if any(r.get('broad_bin') for r in all_results):
    print(f"\n  --- BROADER: z = [0.05, 0.20) ---")
    print(f"\n  {'ℓ_cut':<10} {'diff':>8} {'sigma':>6} {'p':>8}")
    print("  " + "-"*35)
    for r in all_results:
        if r.get('broad_bin'):
            b = r['broad_bin']
            star = "**" if b['p'] < 0.01 else " *" if b['p'] < 0.05 else ""
            print(f"  {r['l_cut']:<10} {b['diff']:>8.2f} {b['sigma']:>5.1f}σ {b['p']:>8.4f} {star}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("GENERATING FIGURE")
print("="*65)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel a: Full sample significance vs ℓ_cut
ax = axes[0]
l_vals = [r['l_cut'] for r in all_results]
sigs_full = [r['full']['sigma'] for r in all_results]
diffs_full = [r['full']['diff'] for r in all_results]

ax.plot(l_vals, sigs_full, 'o-', color='#3498db', ms=10, lw=2.5,
        mfc='white', mew=2, zorder=5)
ax.axhline(2.0, color='#e74c3c', ls='--', lw=1.5, alpha=0.5, label='2σ threshold')
ax.set_xlabel('$\\ell_{\\rm cut}$')
ax.set_ylabel('Significance ($\\sigma$)')
ax.set_title('(a) Full Sample', fontweight='bold')
ax.legend(fontsize=10)
for i, (l, s) in enumerate(zip(l_vals, sigs_full)):
    ax.annotate(f'{s:.1f}σ', (l, s), textcoords="offset points",
                xytext=(8, 8), fontsize=10)

# Panel b: Key bin significance vs ℓ_cut
ax = axes[1]
l_key = [r['l_cut'] for r in all_results if r.get('key_bin')]
sigs_key = [r['key_bin']['sigma'] for r in all_results if r.get('key_bin')]

ax.plot(l_key, sigs_key, 's-', color='#e74c3c', ms=10, lw=2.5,
        mfc='white', mew=2, zorder=5)
ax.axhline(2.0, color='#e74c3c', ls='--', lw=1.5, alpha=0.5, label='2σ')
ax.axhline(3.0, color='#9b59b6', ls=':', lw=1.5, alpha=0.5, label='3σ')
ax.set_xlabel('$\\ell_{\\rm cut}$')
ax.set_ylabel('Significance ($\\sigma$)')
ax.set_title('(b) Key Bin: z = [0.10, 0.15)', fontweight='bold')
ax.legend(fontsize=10)
for i, (l, s) in enumerate(zip(l_key, sigs_key)):
    ax.annotate(f'{s:.1f}σ', (l, s), textcoords="offset points",
                xytext=(8, 8), fontsize=10)

# Panel c: Effect size (diff) vs ℓ_cut
ax = axes[2]
ax.plot(l_vals, diffs_full, 'o-', color='#3498db', ms=10, lw=2,
        mfc='white', mew=2, label='Full sample')
diffs_key = [r['key_bin']['diff'] for r in all_results if r.get('key_bin')]
ax.plot(l_key, diffs_key, 's-', color='#e74c3c', ms=10, lw=2,
        mfc='white', mew=2, label='z=[0.10,0.15)')
ax.axhline(0, color='gray', ls='-', lw=1)
ax.set_xlabel('$\\ell_{\\rm cut}$')
ax.set_ylabel('$\\Delta T_{dis} - \\Delta T_{rel}$ ($\\mu$K)')
ax.set_title('(c) Effect Size vs Filter', fontweight='bold')
ax.legend(fontsize=10)

fig.suptitle('Low-$\\ell$ Cutoff Test: Does Filtering Improve Significance?',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_lowl_cutoff.png')
plt.close()
print("  -> fig_lowl_cutoff.png")

# ── Save ─────────────────────────────────────────────────────────────────
save_results = []
for r in all_results:
    sr = {k: v for k, v in r.items()}
    save_results.append(sr)

with open(f'{OUT}/lowl_results.json', 'w') as f:
    json.dump(save_results, f, indent=2)

print(f"  -> {OUT}/lowl_results.json")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)

print("\n  Does low-ℓ filtering improve significance?")
if len(sigs_full) >= 2:
    baseline_sig = sigs_full[0]
    best_sig = max(sigs_full)
    best_l = l_vals[sigs_full.index(best_sig)]
    print(f"    Full sample: baseline {baseline_sig:.1f}σ → best {best_sig:.1f}σ at ℓ_cut={best_l}")

if len(sigs_key) >= 2:
    baseline_key = sigs_key[0]
    best_key = max(sigs_key)
    best_l_key = l_key[sigs_key.index(best_key)]
    print(f"    Key bin:     baseline {baseline_key:.1f}σ → best {best_key:.1f}σ at ℓ_cut={best_l_key}")

    if best_key > baseline_key:
        print(f"\n  ✓ Filtering IMPROVES significance (large-scale CMB was adding noise)")
    elif best_key < baseline_key - 0.3:
        print(f"\n  ✗ Filtering REDUCES significance (signal may have large-scale component)")
    else:
        print(f"\n  ~ Filtering has minimal effect (signal is robust to scale selection)")

print("\nDone.")
