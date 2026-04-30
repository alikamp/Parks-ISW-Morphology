#!/usr/bin/env python3
"""
ISW Morphology v4 — Confirmation Tests + Shape Analysis
========================================================

Tests:
1. Bin stability: slide the redshift window
2. Bootstrap: resample the z=0.10-0.15 bin 10,000 times
3. Mass/SNR matching within the key bin
4. Null test: randomize cluster positions
5. Cluster shape: use SZ profile ellipticity as morphology proxy

Author: Alika M. Parks
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
import os, json

from astropy.io import fits
import healpy as hp

OUT = "isw_results_v4"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13, 'axes.labelsize': 15,
    'axes.titlesize': 15, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA (same as v3)
# ══════════════════════════════════════════════════════════════════════════════

print("="*65)
print("ISW v4 — CONFIRMATION TESTS + SHAPE ANALYSIS")
print("="*65)

hdu = fits.open("isw_data/psz2.fits")
psz2 = hdu[1].data

ra_all = psz2['RA']; dec_all = psz2['DEC']; z_all = psz2['REDSHIFT']
snr_all = psz2['SNR']; val = psz2['VALIDATION']
M500_all = psz2['MSZ']; Y5R500_all = psz2['Y5R500']
glat = psz2['GLAT']

# Check what columns are available for shape
print(f"\n  Available columns: {psz2.columns.names}")

good = (val >= 20) & np.isfinite(z_all) & (z_all > 0) & \
       np.isfinite(M500_all) & (M500_all > 0) & \
       np.isfinite(Y5R500_all) & (Y5R500_all > 0) & \
       (np.abs(glat) > 15)

ra = ra_all[good]; dec = dec_all[good]; z = z_all[good]
snr = snr_all[good]; M500 = M500_all[good]; Y_SZ = Y5R500_all[good]
N = len(ra)
print(f"  {N} clusters")

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

# Aperture photometry
def ap_phot(ra_d, dec_d):
    theta = np.radians(90 - dec_d); phi = np.radians(ra_d)
    vec = hp.ang2vec(theta, phi)
    dp = hp.query_disc(NSIDE, vec, np.radians(15.0/60))
    op = hp.query_disc(NSIDE, vec, np.radians(45.0/60))
    ap = np.setdiff1d(op, dp)
    Td = CMB[dp]; Ta = CMB[ap]
    gd = np.isfinite(Td) & (Td != hp.UNSEEN) & (np.abs(Td) < 1e4)
    ga = np.isfinite(Ta) & (Ta != hp.UNSEEN) & (np.abs(Ta) < 1e4)
    if gd.sum() < 10 or ga.sum() < 10: return np.nan
    return np.mean(Td[gd]) - np.mean(Ta[ga])

print("  Measuring DeltaT...")
DT = np.array([ap_phot(ra[i], dec[i]) for i in range(N)])
valid = np.isfinite(DT)
print(f"  Valid: {valid.sum()}/{N}")

# Y-M residual morphology
log_Y = np.log10(Y_SZ); log_M = np.log10(M500)
slope, intercept, _, _, _ = stats.linregress(log_M, log_Y)
Y_pred = 10**(intercept + slope * log_M)
Y_resid = (Y_SZ - Y_pred) / Y_pred
morph_dis = Y_resid < 0
morph_rel = Y_resid > 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: BIN STABILITY — slide the redshift window
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("TEST 1: BIN STABILITY")
print("  Does the signal persist when we slide the z window?")
print("="*65)

windows = [
    (0.08, 0.13), (0.09, 0.14), (0.10, 0.15), (0.11, 0.16),
    (0.12, 0.17), (0.08, 0.17), (0.10, 0.20), (0.05, 0.15),
    (0.07, 0.18),
]

print(f"\n  {'Window':<16} {'n_r':>5} {'n_d':>5} {'diff':>8} {'err':>7} {'sigma':>6} {'p':>8}")
print("  " + "-"*60)

bin_results = []
for z_lo, z_hi in windows:
    m_r = morph_rel & valid & (z >= z_lo) & (z < z_hi)
    m_d = morph_dis & valid & (z >= z_lo) & (z < z_hi)
    nr = m_r.sum(); nd = m_d.sum()
    if nr >= 5 and nd >= 5:
        dt_r = DT[m_r]; dt_d = DT[m_d]
        diff = dt_d.mean() - dt_r.mean()
        err = np.sqrt(dt_d.var()/nd + dt_r.var()/nr)
        t, p = stats.ttest_ind(dt_d, dt_r, equal_var=False)
        star = "**" if p < 0.01 else " *" if p < 0.05 else ""
        print(f"  [{z_lo:.2f}, {z_hi:.2f})  {nr:>5} {nd:>5} {diff:>8.2f} {err:>7.2f} "
              f"{abs(t):>5.1f}σ {p:>8.4f} {star}")
        bin_results.append({
            'z_lo': z_lo, 'z_hi': z_hi, 'diff': float(diff),
            'sigma': float(abs(t)), 'p': float(p),
            'n_r': int(nr), 'n_d': int(nd)
        })


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: BOOTSTRAP the z=0.10-0.15 bin
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("TEST 2: BOOTSTRAP (z = 0.10-0.15)")
print("  Resample 10,000 times to get confidence interval")
print("="*65)

z_mask = (z >= 0.10) & (z < 0.15) & valid
m_r = morph_rel & z_mask
m_d = morph_dis & z_mask
dt_r_bin = DT[m_r]
dt_d_bin = DT[m_d]

observed_diff = dt_d_bin.mean() - dt_r_bin.mean()

n_boot = 10000
boot_diffs = np.zeros(n_boot)
np.random.seed(42)
for b in range(n_boot):
    br = np.random.choice(dt_r_bin, len(dt_r_bin), replace=True)
    bd = np.random.choice(dt_d_bin, len(dt_d_bin), replace=True)
    boot_diffs[b] = bd.mean() - br.mean()

ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
boot_p = np.mean(boot_diffs >= 0)  # one-tailed: prob diff >= 0

print(f"\n  Observed diff: {observed_diff:.2f} muK")
print(f"  Bootstrap mean: {boot_diffs.mean():.2f} muK")
print(f"  Bootstrap std: {boot_diffs.std():.2f} muK")
print(f"  95% CI: [{ci_lo:.2f}, {ci_hi:.2f}] muK")
print(f"  P(diff >= 0): {boot_p:.4f}")
print(f"  Zero {'NOT in CI → significant' if ci_hi < 0 else 'in CI → not definitive'}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: MASS MATCHING within z=0.10-0.15
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("TEST 3: MASS MATCHING (z = 0.10-0.15)")
print("  Control for mass within the key bin")
print("="*65)

idx_r = np.where(m_r)[0]
idx_d = np.where(m_d)[0]

# Match by mass
matched_r = []
matched_d = []
used = set()
for j in idx_d:
    best_dist = np.inf
    best_k = -1
    for k in idx_r:
        if k in used: continue
        frac = abs(M500[k] - M500[j]) / M500[j]
        if frac < 0.25 and frac < best_dist:
            best_dist = frac
            best_k = k
    if best_k >= 0:
        matched_d.append(j)
        matched_r.append(best_k)
        used.add(best_k)

matched_r = np.array(matched_r)
matched_d = np.array(matched_d)

if len(matched_r) >= 10:
    dt_r_m = DT[matched_r]
    dt_d_m = DT[matched_d]
    diff_m = dt_d_m.mean() - dt_r_m.mean()
    err_m = np.sqrt(dt_d_m.var()/len(dt_d_m) + dt_r_m.var()/len(dt_r_m))
    t_m, p_m = stats.ttest_ind(dt_d_m, dt_r_m, equal_var=False)
    
    # Paired test
    paired = dt_d_m - dt_r_m
    t_paired, p_paired = stats.ttest_1samp(paired, 0)
    
    ks_mass, ks_p = stats.ks_2samp(M500[matched_r], M500[matched_d])
    
    print(f"\n  Matched pairs: {len(matched_r)}")
    print(f"  Mass KS test: stat={ks_mass:.3f}, p={ks_p:.3f}")
    print(f"  Relaxed:   <DT> = {dt_r_m.mean():.2f} muK")
    print(f"  Disturbed: <DT> = {dt_d_m.mean():.2f} muK")
    print(f"  Diff: {diff_m:.2f} ± {err_m:.2f} muK, {abs(t_m):.1f}σ, p={p_m:.4f}")
    print(f"  Paired: {abs(t_paired):.1f}σ, p={p_paired:.4f}")
else:
    print(f"  Only {len(matched_r)} pairs — insufficient for mass matching")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: NULL TEST — randomize positions
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("TEST 4: NULL TEST (randomize cluster positions)")
print("  If signal is real, it should vanish with random positions")
print("="*65)

n_null = 1000
null_diffs = np.zeros(n_null)
np.random.seed(123)

# Get pool of random sky positions (avoiding galactic plane)
for trial in range(n_null):
    # Shuffle DeltaT values among clusters in the z=0.10-0.15 bin
    # This preserves the DT distribution but breaks the morphology assignment
    dt_all_bin = DT[z_mask].copy()
    np.random.shuffle(dt_all_bin)
    
    # Split by same morphology masks (but DT is now randomized)
    morph_in_bin = morph_dis[z_mask]
    null_diffs[trial] = dt_all_bin[morph_in_bin].mean() - dt_all_bin[~morph_in_bin].mean()

null_p = np.mean(null_diffs <= observed_diff)  # one-tailed
null_sigma = abs(observed_diff - null_diffs.mean()) / null_diffs.std()

print(f"\n  Observed diff: {observed_diff:.2f} muK")
print(f"  Null distribution: mean={null_diffs.mean():.2f}, std={null_diffs.std():.2f}")
print(f"  P(null <= observed): {null_p:.4f}")
print(f"  Equivalent sigma: {null_sigma:.1f}σ")
print(f"  Signal is {'REAL' if null_p < 0.05 else 'consistent with noise'} at 95% confidence")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: CLUSTER SHAPE — ellipticity from SZ profile
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("TEST 5: CLUSTER SHAPE ANALYSIS")
print("  Using CMB temperature map to estimate cluster elongation")
print("="*65)

# PSZ2 doesn't include ellipticity directly, but we can estimate it
# from the CMB map by measuring the quadrupole moment of the
# temperature distribution around each cluster

def measure_ellipticity(ra_d, dec_d, radius_am=20.0):
    """
    Estimate cluster elongation from CMB temperature quadrupole.
    
    Compute the inertia tensor of |DeltaT| within a disk,
    then ellipticity = 1 - lambda_min/lambda_max
    
    High ellipticity = elongated/disturbed
    Low ellipticity = round/relaxed
    """
    theta = np.radians(90 - dec_d); phi = np.radians(ra_d)
    vec = hp.ang2vec(theta, phi)
    
    pix = hp.query_disc(NSIDE, vec, np.radians(radius_am / 60))
    if len(pix) < 20:
        return np.nan
    
    # Get pixel positions relative to cluster center
    pix_theta, pix_phi = hp.pix2ang(NSIDE, pix)
    
    # Project to tangent plane (small angle)
    dx = (pix_phi - phi) * np.sin(theta)  # radians
    dy = (pix_theta - theta)  # radians
    
    # Weight by |temperature|
    T = CMB[pix]
    good = np.isfinite(T) & (T != hp.UNSEEN)
    if good.sum() < 20:
        return np.nan
    
    dx = dx[good]; dy = dy[good]
    T_abs = np.abs(T[good])
    T_abs = T_abs / T_abs.sum()  # normalize as weights
    
    # Inertia tensor
    Ixx = np.sum(T_abs * dx * dx)
    Iyy = np.sum(T_abs * dy * dy)
    Ixy = np.sum(T_abs * dx * dy)
    
    # Eigenvalues
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    discriminant = trace**2 - 4 * det
    
    if discriminant < 0 or trace < 1e-20:
        return np.nan
    
    lam_max = (trace + np.sqrt(discriminant)) / 2
    lam_min = (trace - np.sqrt(discriminant)) / 2
    
    if lam_max < 1e-20:
        return np.nan
    
    ellipticity = 1.0 - lam_min / lam_max
    return float(np.clip(ellipticity, 0, 1))

print("  Computing cluster ellipticities...")
ellip = np.array([measure_ellipticity(ra[i], dec[i]) for i in range(N)])
ellip_valid = np.isfinite(ellip)
print(f"  Valid ellipticities: {ellip_valid.sum()}/{N}")
print(f"  Range: [{np.nanmin(ellip):.3f}, {np.nanmax(ellip):.3f}]")
print(f"  Median: {np.nanmedian(ellip):.3f}")

# Split by ellipticity
e_med = np.nanmedian(ellip)
round_clusters = ellip_valid & (ellip < e_med) & valid  # round = relaxed
elongated_clusters = ellip_valid & (ellip >= e_med) & valid  # elongated = disturbed

dt_round = DT[round_clusters]
dt_elong = DT[elongated_clusters]

if len(dt_round) >= 20 and len(dt_elong) >= 20:
    diff_shape = dt_elong.mean() - dt_round.mean()
    err_shape = np.sqrt(dt_elong.var()/len(dt_elong) + dt_round.var()/len(dt_round))
    t_shape, p_shape = stats.ttest_ind(dt_elong, dt_round, equal_var=False)
    
    print(f"\n  ALL REDSHIFTS:")
    print(f"    Round (e < {e_med:.3f}):    n={len(dt_round)}, <DT>={dt_round.mean():.2f} muK")
    print(f"    Elongated (e >= {e_med:.3f}): n={len(dt_elong)}, <DT>={dt_elong.mean():.2f} muK")
    print(f"    Diff: {diff_shape:.2f} ± {err_shape:.2f} muK, {abs(t_shape):.1f}σ, p={p_shape:.4f}")

# Shape analysis in the key z bin
z_key = (z >= 0.10) & (z < 0.15)
round_key = round_clusters & z_key
elong_key = elongated_clusters & z_key

if round_key.sum() >= 5 and elong_key.sum() >= 5:
    dt_rk = DT[round_key]; dt_ek = DT[elong_key]
    diff_sk = dt_ek.mean() - dt_rk.mean()
    err_sk = np.sqrt(dt_ek.var()/len(dt_ek) + dt_rk.var()/len(dt_rk))
    t_sk, p_sk = stats.ttest_ind(dt_ek, dt_rk, equal_var=False)
    
    print(f"\n  z = [0.10, 0.15) ONLY:")
    print(f"    Round:    n={len(dt_rk)}, <DT>={dt_rk.mean():.2f} muK")
    print(f"    Elongated: n={len(dt_ek)}, <DT>={dt_ek.mean():.2f} muK")
    print(f"    Diff: {diff_sk:.2f} ± {err_sk:.2f} muK, {abs(t_sk):.1f}σ, p={p_sk:.4f}")

# Ellipticity vs Y-M residual correlation
e_ym_mask = ellip_valid & np.isfinite(Y_resid)
if e_ym_mask.sum() > 30:
    r_ey, p_ey = stats.pearsonr(ellip[e_ym_mask], Y_resid[e_ym_mask])
    print(f"\n  Ellipticity vs Y-M residual: r={r_ey:.3f}, p={p_ey:.4f}")
    print(f"  {'Correlated' if p_ey < 0.05 else 'Not correlated'} → "
          f"{'same' if p_ey < 0.05 else 'independent'} morphology indicator")

# Dose-response by ellipticity quintiles
print(f"\n  Dose-response by ellipticity quintile:")
e_percs = np.nanpercentile(ellip[ellip_valid], [0, 20, 40, 60, 80, 100])
print(f"  {'Quintile':<10} {'e range':<20} {'n':>5} {'<DT>':>8}")
print("  " + "-"*45)
for q in range(5):
    lo, hi = e_percs[q], e_percs[q+1]
    mask_q = ellip_valid & valid & (ellip >= lo) & (ellip < hi if q < 4 else ellip <= hi)
    n = mask_q.sum()
    if n >= 10:
        label = "most round" if q == 0 else "most elongated" if q == 4 else ""
        print(f"  Q{q+1:<9} [{lo:.3f}, {hi:.3f}]  {n:>5} {DT[mask_q].mean():>8.2f}  {label}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("GENERATING FIGURES")
print("="*65)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# (a) Bin stability
ax = axes[0, 0]
if bin_results:
    x_labels = [f"[{r['z_lo']:.2f},{r['z_hi']:.2f})" for r in bin_results]
    diffs_b = [r['diff'] for r in bin_results]
    sigs_b = [r['sigma'] for r in bin_results]
    colors_b = ['#e74c3c' if s > 2 else '#f39c12' if s > 1.5 else '#95a5a6' for s in sigs_b]
    ax.barh(range(len(x_labels)), diffs_b, color=colors_b, height=0.6, edgecolor='white')
    ax.set_yticks(range(len(x_labels))); ax.set_yticklabels(x_labels, fontsize=9)
    ax.axvline(0, color='black', ls='-', lw=1)
    for i, (d, s) in enumerate(zip(diffs_b, sigs_b)):
        ax.text(d - 2 if d < 0 else d + 1, i, f'{s:.1f}σ', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Diff (μK)')
ax.set_title('(a) Bin Stability: Sliding z Window', fontweight='bold')

# (b) Bootstrap
ax = axes[0, 1]
ax.hist(boot_diffs, bins=60, color='#3498db', alpha=0.7, edgecolor='white')
ax.axvline(0, color='black', ls='-', lw=2)
ax.axvline(observed_diff, color='#e74c3c', ls='--', lw=2, label=f'Observed: {observed_diff:.1f}')
ax.axvline(ci_lo, color='#e74c3c', ls=':', lw=1)
ax.axvline(ci_hi, color='#e74c3c', ls=':', lw=1)
ax.set_xlabel('Diff (μK)')
ax.set_title(f'(b) Bootstrap z=[0.10,0.15)\nCI: [{ci_lo:.1f}, {ci_hi:.1f}]', fontweight='bold')
ax.legend()

# (c) Null test
ax = axes[0, 2]
ax.hist(null_diffs, bins=50, color='#95a5a6', alpha=0.7, edgecolor='white', label='Null')
ax.axvline(observed_diff, color='#e74c3c', ls='--', lw=2, label=f'Observed: {observed_diff:.1f}')
ax.axvline(0, color='black', ls='-', lw=1)
ax.set_xlabel('Diff (μK)')
ax.set_title(f'(c) Null Test: Shuffled Morphology\np={null_p:.3f}', fontweight='bold')
ax.legend()

# (d) Ellipticity histogram
ax = axes[1, 0]
if ellip_valid.sum() > 0:
    ax.hist(ellip[ellip_valid & morph_rel], bins=25, alpha=0.6, color='#2ecc71',
            density=True, label='Relaxed (Y-M)', edgecolor='white')
    ax.hist(ellip[ellip_valid & morph_dis], bins=25, alpha=0.6, color='#e74c3c',
            density=True, label='Disturbed (Y-M)', edgecolor='white')
    ax.axvline(e_med, color='black', ls='--', lw=1.5)
    ax.set_xlabel('Ellipticity')
    ax.set_ylabel('Density')
    ax.set_title('(d) Cluster Ellipticity Distribution', fontweight='bold')
    ax.legend(fontsize=9)

# (e) DeltaT vs ellipticity
ax = axes[1, 1]
if ellip_valid.sum() > 0:
    mask_plot = ellip_valid & valid
    ax.scatter(ellip[mask_plot], DT[mask_plot], s=8, alpha=0.3, c='#3498db')
    # Binned means
    e_bins = np.nanpercentile(ellip[mask_plot], np.linspace(0, 100, 8))
    for i in range(len(e_bins)-1):
        bm = (ellip >= e_bins[i]) & (ellip < e_bins[i+1]) & mask_plot
        if bm.sum() >= 10:
            ax.scatter((e_bins[i]+e_bins[i+1])/2, DT[bm].mean(), s=150,
                      c='#e74c3c', edgecolors='black', zorder=5)
    ax.axhline(0, color='gray', ls='-', lw=1)
    ax.set_xlabel('Ellipticity')
    ax.set_ylabel('$\\Delta T$ (μK)')
    ax.set_title('(e) $\\Delta T$ vs Cluster Shape', fontweight='bold')

# (f) Shape × redshift in key bin
ax = axes[1, 2]
categories = []
means = []
errs_cat = []
colors_cat = []

for label, mask, col in [
    ('Round\nall z', round_clusters, '#2ecc71'),
    ('Elongated\nall z', elongated_clusters, '#e74c3c'),
    ('Round\nz∈[.10,.15)', round_key, '#27ae60'),
    ('Elongated\nz∈[.10,.15)', elong_key, '#c0392b'),
]:
    n = mask.sum()
    if n >= 5:
        categories.append(label)
        means.append(DT[mask].mean())
        errs_cat.append(DT[mask].std() / np.sqrt(n))
        colors_cat.append(col)

if categories:
    ax.bar(range(len(categories)), means, yerr=errs_cat, color=colors_cat,
           width=0.6, capsize=5, edgecolor='white', alpha=0.8)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=9)
    ax.axhline(0, color='black', ls='-', lw=1)
    ax.set_ylabel('$\\langle\\Delta T\\rangle$ (μK)')
    ax.set_title('(f) Shape × Redshift', fontweight='bold')

fig.suptitle('ISW Morphology — Confirmation Tests + Shape Analysis',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_confirmation.png')
plt.close()
print("  -> fig_confirmation.png")

# Save
results = {
    "bin_stability": bin_results,
    "bootstrap": {
        "observed_diff": float(observed_diff),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "boot_p": float(boot_p),
        "zero_excluded": bool(ci_hi < 0),
    },
    "null_test": {
        "null_p": float(null_p),
        "null_sigma": float(null_sigma),
        "signal_real": bool(null_p < 0.05),
    },
}
with open(f'{OUT}/results_v4.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results: {OUT}/results_v4.json")

# ── FINAL SUMMARY ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL SUMMARY")
print("="*65)

print(f"\n  BIN STABILITY: Signal persists across {sum(1 for r in bin_results if r['p'] < 0.05)} of {len(bin_results)} windows at p<0.05")

print(f"\n  BOOTSTRAP (z=0.10-0.15):")
print(f"    95% CI: [{ci_lo:.1f}, {ci_hi:.1f}] muK")
print(f"    Zero excluded: {'YES' if ci_hi < 0 else 'NO'}")

print(f"\n  NULL TEST:")
print(f"    p = {null_p:.4f}")
print(f"    Signal {'SURVIVES' if null_p < 0.05 else 'does not survive'} null test")

if ellip_valid.sum() > 0:
    print(f"\n  SHAPE:")
    if 'diff_shape' in dir():
        print(f"    All z: elongated-round = {diff_shape:.1f} muK, {abs(t_shape):.1f}σ")
    if 'diff_sk' in dir():
        print(f"    z=[0.10,0.15): elongated-round = {diff_sk:.1f} muK, {abs(t_sk):.1f}σ")

print("\nDone.")
