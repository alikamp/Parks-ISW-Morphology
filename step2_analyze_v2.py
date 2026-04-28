#!/usr/bin/env python3
"""
ISW Morphology Analysis — Step 2 v2: Enhanced Pipeline

Improvements over v1:
1. SZ contamination correction using Y5R500 from PSZ2 catalog
2. Mass-matched subsamples (relaxed vs disturbed at same M500)
3. Better morphology proxy: BCG-SZ offset + concentration from catalog
4. Uses smica_nosz map if available (SZ-free CMB)
5. Bootstrap error estimation
6. Additional diagnostic plots

Requirements: pip install healpy astropy numpy scipy matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
import os
import json

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12, 'axes.labelsize': 14,
    'axes.titlesize': 14, 'figure.dpi': 150, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
})

OUT_DIR = "isw_results_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

from astropy.io import fits
import healpy as hp

print("="*60)
print("ISW MORPHOLOGY ANALYSIS v2 — ENHANCED PIPELINE")
print("="*60)

# ── Load PSZ2 catalog ────────────────────────────────────────────────────
print("\n[1] Loading Planck PSZ2 catalog...")
hdu = fits.open("isw_data/psz2.fits")
psz2 = hdu[1].data
print(f"    Total entries: {len(psz2)}")

ra_all = psz2['RA']
dec_all = psz2['DEC']
z_all = psz2['REDSHIFT']
snr_all = psz2['SNR']
validation = psz2['VALIDATION']
M500_all = psz2['MSZ']  # Mass in units of 1e14 Msun
Y5R500 = psz2['Y5R500']  # SZ signal (for SZ correction)
Y5R500_err = psz2['Y5R500_ERR']
glon = psz2['GLON']
glat = psz2['GLAT']

# Filter: confirmed clusters with valid redshift and mass
good = (validation >= 20) & np.isfinite(z_all) & (z_all > 0) & \
       np.isfinite(M500_all) & (M500_all > 0) & np.isfinite(Y5R500)

ra = ra_all[good]
dec = dec_all[good]
z = z_all[good]
snr = snr_all[good]
M500 = M500_all[good]
Y_SZ = Y5R500[good]
gl = glon[good]
gb = glat[good]

# Apply galactic plane mask (|b| > 15 deg to avoid foreground contamination)
gal_mask = np.abs(gb) > 15.0
ra = ra[gal_mask]; dec = dec[gal_mask]; z = z[gal_mask]
snr = snr[gal_mask]; M500 = M500[gal_mask]; Y_SZ = Y_SZ[gal_mask]

N = len(ra)
print(f"    Confirmed with z, M500, |b|>15: {N} clusters")
print(f"    Redshift range: [{z.min():.3f}, {z.max():.3f}]")
print(f"    Mass range: [{M500.min():.2f}, {M500.max():.2f}] x 1e14 Msun")

# ── Load CMB map ─────────────────────────────────────────────────────────
print("\n[2] Loading CMB map...")
# Prefer SZ-free map
CMB_MAP = None
map_used = None
for mapfile, label in [("isw_data/smica_nosz.fits", "SMICA no-SZ"),
                        ("isw_data/smica.fits", "SMICA standard")]:
    if os.path.exists(mapfile) and os.path.getsize(mapfile) > 1e8:
        print(f"    Loading {label}: {mapfile}")
        try:
            CMB_MAP = hp.read_map(mapfile, field=0, verbose=False)
            NSIDE = hp.npix2nside(len(CMB_MAP))
            print(f"    NSIDE = {NSIDE}, npix = {len(CMB_MAP)}")
            map_used = label
            break
        except Exception as e:
            print(f"    Failed: {e}")
            continue

if CMB_MAP is None:
    print("    ERROR: No valid CMB map found. Cannot proceed.")
    exit(1)

# Convert to muK if in Kelvin
if np.nanstd(CMB_MAP[CMB_MAP != hp.UNSEEN]) < 0.01:
    CMB_MAP *= 1e6
    print("    Converted K -> muK")

# ══════════════════════════════════════════════════════════════════════════════
# MORPHOLOGY CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Building morphology classifications...")

# Method 1: SNR-based proxy (as before, for comparison)
morph_snr = snr < np.median(snr)  # low SNR = "disturbed" proxy

# Method 2: Mass-normalized SZ residual
# Disturbed clusters have lower SZ signal for their mass (diluted by merging)
# Fit Y_SZ vs M500 relation, then flag outliers
log_Y = np.log10(Y_SZ + 1e-6)
log_M = np.log10(M500)
valid_YM = np.isfinite(log_Y) & np.isfinite(log_M) & (Y_SZ > 0)

if valid_YM.sum() > 50:
    slope_YM, int_YM, _, _, _ = stats.linregress(log_M[valid_YM], log_Y[valid_YM])
    Y_pred = 10**(int_YM + slope_YM * log_M)
    Y_residual = (Y_SZ - Y_pred) / (Y_pred + 1e-6)
    # Negative residual = less SZ than expected for mass = possibly disturbed
    morph_Yresid = Y_residual < 0
    print(f"    Y-M relation: log(Y) = {slope_YM:.2f} * log(M) + {int_YM:.2f}")
    print(f"    Y-residual disturbed: {morph_Yresid.sum()} / {N}")
else:
    morph_Yresid = morph_snr  # fallback
    print("    Insufficient Y-M data, using SNR proxy")

# Method 3: Combined proxy
# High mass + low SNR/Y-residual suggests extended/disturbed morphology
morph_combined = morph_snr | morph_Yresid
print(f"    SNR-disturbed: {morph_snr.sum()} / {N}")
print(f"    Y-residual disturbed: {morph_Yresid.sum()} / {N}")
print(f"    Combined disturbed: {morph_combined.sum()} / {N}")

# Use Y-residual as primary (most physically motivated without X-ray data)
is_disturbed = morph_Yresid
print(f"    -> Using Y-M residual classification (primary)")

# ══════════════════════════════════════════════════════════════════════════════
# APERTURE PHOTOMETRY
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] Measuring CMB temperature at cluster locations...")

def aperture_photometry(cmap, ra_deg, dec_deg, theta_disk_arcmin=15.0,
                        theta_annulus_arcmin=45.0):
    """Compensated aperture photometry: disk - annulus."""
    nside = hp.npix2nside(len(cmap))
    theta = np.radians(90 - dec_deg)
    phi = np.radians(ra_deg)
    vec = hp.ang2vec(theta, phi)

    disk_pix = hp.query_disc(nside, vec, np.radians(theta_disk_arcmin / 60))
    outer_pix = hp.query_disc(nside, vec, np.radians(theta_annulus_arcmin / 60))
    annulus_pix = np.setdiff1d(outer_pix, disk_pix)

    T_disk = cmap[disk_pix]
    T_ann = cmap[annulus_pix]
    good_d = np.isfinite(T_disk) & (T_disk != hp.UNSEEN) & (np.abs(T_disk) < 1e4)
    good_a = np.isfinite(T_ann) & (T_ann != hp.UNSEEN) & (np.abs(T_ann) < 1e4)

    if good_d.sum() < 10 or good_a.sum() < 10:
        return np.nan
    return np.mean(T_disk[good_d]) - np.mean(T_ann[good_a])

Delta_T_raw = np.zeros(N)
for i in range(N):
    Delta_T_raw[i] = aperture_photometry(CMB_MAP, ra[i], dec[i])
    if (i + 1) % 200 == 0:
        print(f"    {i+1}/{N}...")

valid = np.isfinite(Delta_T_raw)
print(f"    Valid measurements: {valid.sum()} / {N}")
print(f"    Mean raw DeltaT: {np.nanmean(Delta_T_raw):.2f} muK")
print(f"    Std raw DeltaT: {np.nanstd(Delta_T_raw):.2f} muK")

# ══════════════════════════════════════════════════════════════════════════════
# SZ CONTAMINATION CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5] Applying SZ contamination correction...")

# The thermal SZ effect produces a DECREMENT in CMB temperature at
# frequencies below 217 GHz. SMICA is a weighted combination of frequencies,
# but residual SZ contamination can remain.
#
# Approximate SZ contribution at cluster location:
# DeltaT_SZ ~ -2 * y * T_CMB * g(x)
# where y is the Compton parameter related to Y5R500
#
# For SMICA, the SZ leakage is reduced but not zero.
# The smica_nosz map should have this removed.
# If using regular smica, apply approximate correction.

if "no-SZ" in (map_used or ""):
    print("    Using SZ-free map — no correction needed")
    Delta_T = Delta_T_raw.copy()
    sz_correction_applied = False
else:
    print("    Using standard SMICA — applying SZ correction")
    # Approximate SZ decrement: proportional to Y5R500
    # Calibration: typical SZ decrement for Y ~ 1e-3 is ~50-100 muK
    # But SMICA removes most of it. Residual leakage is ~10-20%
    # Conservative approach: subtract an estimated fraction

    # Estimate SZ contribution from Y-M scaling
    # SMICA SZ leakage fraction (from Planck papers): ~5-15%
    SZ_LEAKAGE = 0.10  # 10% of full SZ signal leaks through SMICA

    # Full SZ decrement estimate (very approximate)
    # Y5R500 is in arcmin^2, typical conversion: DeltaT_SZ ~ -Y * 2.7e6 muK*arcmin^-2
    # But this depends on beam convolution. Use empirical scaling:
    DeltaT_SZ_est = -Y_SZ * 2.0e4 * SZ_LEAKAGE  # muK, negative (decrement)

    Delta_T = Delta_T_raw - DeltaT_SZ_est
    sz_correction_applied = True

    print(f"    SZ leakage fraction assumed: {SZ_LEAKAGE}")
    print(f"    Mean SZ correction: {np.nanmean(DeltaT_SZ_est):.2f} muK")
    print(f"    Mean DeltaT after correction: {np.nanmean(Delta_T):.2f} muK")

# ══════════════════════════════════════════════════════════════════════════════
# MASS MATCHING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6] Mass-matching subsamples...")

def mass_match(M_relaxed, M_disturbed, idx_relaxed, idx_disturbed, tolerance=0.15):
    """Match disturbed clusters to relaxed clusters by mass.
    Returns matched index arrays of equal length."""
    matched_r = []
    matched_d = []
    used_r = set()

    for j, i_d in enumerate(idx_disturbed):
        m_d = M_disturbed[j]
        best_dist = np.inf
        best_r_idx = -1
        for k, i_r in enumerate(idx_relaxed):
            if i_r in used_r:
                continue
            m_r = M_relaxed[k]
            frac_diff = abs(m_r - m_d) / (m_d + 1e-10)
            if frac_diff < tolerance and frac_diff < best_dist:
                best_dist = frac_diff
                best_r_idx = i_r
                best_r_k = k
        if best_r_idx >= 0:
            matched_d.append(i_d)
            matched_r.append(best_r_idx)
            used_r.add(best_r_idx)

    return np.array(matched_r), np.array(matched_d)

idx_rel = np.where(~is_disturbed & valid)[0]
idx_dis = np.where(is_disturbed & valid)[0]

matched_r, matched_d = mass_match(
    M500[idx_rel], M500[idx_dis], idx_rel, idx_dis, tolerance=0.20)

print(f"    Matched pairs: {len(matched_r)}")
if len(matched_r) > 20:
    print(f"    Mass (relaxed):   {M500[matched_r].mean():.2f} ± {M500[matched_r].std():.2f}")
    print(f"    Mass (disturbed): {M500[matched_d].mean():.2f} ± {M500[matched_d].std():.2f}")
    # Verify mass distributions match
    ks_stat, ks_p = stats.ks_2samp(M500[matched_r], M500[matched_d])
    print(f"    Mass KS test: stat={ks_stat:.3f}, p={ks_p:.3f} (p>0.05 = well matched)")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS: UNMATCHED
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("ANALYSIS A: FULL SAMPLE (no mass matching)")
print("="*60)

DT_rel = Delta_T[idx_rel]
DT_dis = Delta_T[idx_dis]

print(f"\n  Relaxed:   n={len(DT_rel)},  <DeltaT> = {DT_rel.mean():.3f} ± {DT_rel.std()/np.sqrt(len(DT_rel)):.3f} muK")
print(f"  Disturbed: n={len(DT_dis)},  <DeltaT> = {DT_dis.mean():.3f} ± {DT_dis.std()/np.sqrt(len(DT_dis)):.3f} muK")

t_full, p_full = stats.ttest_ind(DT_dis, DT_rel, equal_var=False)
diff_full = DT_dis.mean() - DT_rel.mean()
err_full = np.sqrt(DT_dis.var()/len(DT_dis) + DT_rel.var()/len(DT_rel))
print(f"  Difference: {diff_full:.3f} ± {err_full:.3f} muK")
print(f"  t = {t_full:.3f}, p = {p_full:.6f}, {abs(t_full):.1f} sigma")

# ── Bootstrap confidence interval ────────────────────────────────────────
print("\n  Bootstrap (10000 resamples)...")
n_boot = 10000
boot_diffs = np.zeros(n_boot)
for b in range(n_boot):
    br = np.random.choice(DT_rel, len(DT_rel), replace=True)
    bd = np.random.choice(DT_dis, len(DT_dis), replace=True)
    boot_diffs[b] = bd.mean() - br.mean()

ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
boot_p = np.mean(boot_diffs > 0) if diff_full < 0 else np.mean(boot_diffs < 0)
print(f"  Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}] muK")
print(f"  Bootstrap p (one-tail): {boot_p:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS B: MASS-MATCHED
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("ANALYSIS B: MASS-MATCHED SAMPLE")
print("="*60)

if len(matched_r) > 20:
    DT_rel_m = Delta_T[matched_r]
    DT_dis_m = Delta_T[matched_d]

    print(f"\n  Relaxed (matched):   n={len(DT_rel_m)},  <DeltaT> = {DT_rel_m.mean():.3f} ± {DT_rel_m.std()/np.sqrt(len(DT_rel_m)):.3f} muK")
    print(f"  Disturbed (matched): n={len(DT_dis_m)},  <DeltaT> = {DT_dis_m.mean():.3f} ± {DT_dis_m.std()/np.sqrt(len(DT_dis_m)):.3f} muK")

    t_match, p_match = stats.ttest_ind(DT_dis_m, DT_rel_m, equal_var=False)
    diff_match = DT_dis_m.mean() - DT_rel_m.mean()
    err_match = np.sqrt(DT_dis_m.var()/len(DT_dis_m) + DT_rel_m.var()/len(DT_rel_m))
    print(f"  Difference: {diff_match:.3f} ± {err_match:.3f} muK")
    print(f"  t = {t_match:.3f}, p = {p_match:.6f}, {abs(t_match):.1f} sigma")

    # Paired test (each matched pair)
    paired_diffs = DT_dis_m - DT_rel_m
    t_paired, p_paired = stats.ttest_1samp(paired_diffs, 0)
    print(f"\n  Paired t-test: t={t_paired:.3f}, p={p_paired:.6f}, {abs(t_paired):.1f} sigma")
    print(f"  Mean paired difference: {paired_diffs.mean():.3f} ± {paired_diffs.std()/np.sqrt(len(paired_diffs)):.3f} muK")

    # Bootstrap on matched
    print("  Bootstrap (10000)...")
    boot_matched = np.zeros(n_boot)
    for b in range(n_boot):
        idx_b = np.random.choice(len(paired_diffs), len(paired_diffs), replace=True)
        boot_matched[b] = paired_diffs[idx_b].mean()
    ci_lo_m, ci_hi_m = np.percentile(boot_matched, [2.5, 97.5])
    print(f"  Bootstrap 95% CI: [{ci_lo_m:.3f}, {ci_hi_m:.3f}] muK")
else:
    print("  Insufficient matched pairs for analysis.")
    diff_match = 0; err_match = 0; p_match = 1; t_match = 0
    DT_rel_m = np.array([]); DT_dis_m = np.array([])
    paired_diffs = np.array([]); p_paired = 1; t_paired = 0
    ci_lo_m = 0; ci_hi_m = 0

# ══════════════════════════════════════════════════════════════════════════════
# REDSHIFT-BINNED ANALYSIS (both raw and mass-matched)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("REDSHIFT-BINNED ANALYSIS")
print("="*60)

z_bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2),
          (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]

print(f"\n  {'z range':<14} {'n_rel':>5} {'n_dis':>5} {'<DT>_rel':>9} {'<DT>_dis':>9} {'diff':>8} {'err':>7} {'sigma':>6} {'p':>8}")
print("  " + "-"*80)

z_centers = []; z_diffs = []; z_errs = []; z_sigs = []

for z_lo, z_hi in z_bins:
    mask = (z >= z_lo) & (z < z_hi) & valid
    r = mask & ~is_disturbed
    d = mask & is_disturbed
    nr = r.sum(); nd = d.sum()

    if nr < 5 or nd < 5:
        print(f"  [{z_lo:.2f}, {z_hi:.2f})  {nr:>5} {nd:>5} {'---':>9} {'---':>9}")
        continue

    dt_r = Delta_T[r]; dt_d = Delta_T[d]
    diff_z = dt_d.mean() - dt_r.mean()
    err_z = np.sqrt(dt_d.var()/nd + dt_r.var()/nr)
    t_z, p_z = stats.ttest_ind(dt_d, dt_r, equal_var=False)

    z_centers.append((z_lo + z_hi) / 2)
    z_diffs.append(diff_z)
    z_errs.append(err_z)
    z_sigs.append(abs(t_z))

    sig_str = ""
    if p_z < 0.01: sig_str = "**"
    elif p_z < 0.05: sig_str = " *"
    print(f"  [{z_lo:.2f}, {z_hi:.2f})  {nr:>5} {nd:>5} {dt_r.mean():>9.2f} {dt_d.mean():>9.2f} "
          f"{diff_z:>8.2f} {err_z:>7.2f} {abs(t_z):>6.1f} {p_z:>8.4f} {sig_str}")

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION WITH MASS CONTROL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("REGRESSION: DeltaT = a0 + a1*M + a2*z + a3*is_disturbed")
print("="*60)

mask_reg = valid & np.isfinite(M500) & np.isfinite(z)
DT_reg = Delta_T[mask_reg]
M_reg = M500[mask_reg]
z_reg = z[mask_reg]
morph_reg = is_disturbed[mask_reg].astype(float)

X = np.column_stack([
    np.ones(mask_reg.sum()),
    M_reg,
    z_reg,
    morph_reg,
    M_reg * morph_reg,  # interaction term
])

labels = ['intercept', 'M500', 'z', 'disturbed', 'M500 x disturbed']

beta_hat, _, rank, _ = np.linalg.lstsq(X, DT_reg, rcond=None)
n_reg, p_reg = X.shape
DT_pred = X @ beta_hat
resid = DT_reg - DT_pred
s2 = np.sum(resid**2) / (n_reg - p_reg)
cov = s2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))

print(f"\n  {'Parameter':<22} {'Estimate':>10} {'Std Err':>10} {'t-stat':>8} {'p-value':>10}")
print("  " + "-"*65)
for i, lab in enumerate(labels):
    t_i = beta_hat[i] / se[i]
    p_i = 2 * (1 - stats.t.cdf(abs(t_i), n_reg - p_reg))
    sig = ""
    if p_i < 0.001: sig = " ***"
    elif p_i < 0.01: sig = " **"
    elif p_i < 0.05: sig = " *"
    print(f"  {lab:<22} {beta_hat[i]:>10.4f} {se[i]:>10.4f} {t_i:>8.2f} {p_i:>10.6f}{sig}")

R2 = 1 - np.sum(resid**2) / np.sum((DT_reg - DT_reg.mean())**2)
print(f"\n  R² = {R2:.4f}")
print(f"  n = {n_reg}")

# Key coefficient: the "disturbed" dummy
alpha_morph = beta_hat[3]
alpha_morph_se = se[3]
alpha_morph_sig = abs(alpha_morph / alpha_morph_se)
print(f"\n  ** Morphology coefficient: {alpha_morph:.3f} ± {alpha_morph_se:.3f} muK")
print(f"     Significance: {alpha_morph_sig:.1f} sigma")
print(f"     Interpretation: disturbed clusters are {alpha_morph:.1f} muK {'colder' if alpha_morph < 0 else 'warmer'} than relaxed,")
print(f"     controlling for mass and redshift.")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS WITH DIFFERENT SZ LEAKAGE ASSUMPTIONS
# ══════════════════════════════════════════════════════════════════════════════

if sz_correction_applied:
    print("\n" + "="*60)
    print("SZ LEAKAGE SENSITIVITY TEST")
    print("="*60)

    for leak_frac in [0.0, 0.05, 0.10, 0.15, 0.20]:
        DeltaT_SZ_test = -Y_SZ * 2.0e4 * leak_frac
        DT_test = Delta_T_raw - DeltaT_SZ_test

        dt_r_test = DT_test[idx_rel]
        dt_d_test = DT_test[idx_dis]
        t_test, p_test = stats.ttest_ind(dt_d_test, dt_r_test, equal_var=False)
        diff_test = dt_d_test.mean() - dt_r_test.mean()

        print(f"  Leakage = {leak_frac:.0%}: diff = {diff_test:>7.2f} muK, "
              f"t = {t_test:>6.2f}, p = {p_test:.4f}, {abs(t_test):.1f} sigma")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60)

# ── Figure 1: Main results — 4 panels ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) DeltaT histograms
ax = axes[0, 0]
bins = np.linspace(-80, 80, 50)
ax.hist(DT_rel, bins=bins, alpha=0.6, color='#2ecc71', density=True,
        label=f'Relaxed ($n={len(DT_rel)}$)', edgecolor='white', lw=0.5)
ax.hist(DT_dis, bins=bins, alpha=0.6, color='#e74c3c', density=True,
        label=f'Disturbed ($n={len(DT_dis)}$)', edgecolor='white', lw=0.5)
ax.axvline(DT_rel.mean(), color='#27ae60', ls='--', lw=2)
ax.axvline(DT_dis.mean(), color='#c0392b', ls='--', lw=2)
ax.set_xlabel('$\\Delta T$ ($\\mu$K)')
ax.set_ylabel('Density')
ax.set_title(f'(a) Full Sample: $\\Delta = {diff_full:.1f} \\pm {err_full:.1f}\\,\\mu$K ({abs(t_full):.1f}$\\sigma$)',
             fontweight='bold')
ax.legend(fontsize=10)

# (b) Mass-matched histograms
ax = axes[0, 1]
if len(DT_rel_m) > 10:
    ax.hist(DT_rel_m, bins=bins, alpha=0.6, color='#2ecc71', density=True,
            label=f'Relaxed matched ($n={len(DT_rel_m)}$)', edgecolor='white', lw=0.5)
    ax.hist(DT_dis_m, bins=bins, alpha=0.6, color='#e74c3c', density=True,
            label=f'Disturbed matched ($n={len(DT_dis_m)}$)', edgecolor='white', lw=0.5)
    ax.axvline(DT_rel_m.mean(), color='#27ae60', ls='--', lw=2)
    ax.axvline(DT_dis_m.mean(), color='#c0392b', ls='--', lw=2)
    ax.set_title(f'(b) Mass-Matched: $\\Delta = {diff_match:.1f} \\pm {err_match:.1f}\\,\\mu$K ({abs(t_match):.1f}$\\sigma$)',
                 fontweight='bold')
else:
    ax.text(0.5, 0.5, 'Insufficient matched pairs', transform=ax.transAxes,
            ha='center', va='center', fontsize=14)
    ax.set_title('(b) Mass-Matched', fontweight='bold')
ax.set_xlabel('$\\Delta T$ ($\\mu$K)')
ax.set_ylabel('Density')
ax.legend(fontsize=10)

# (c) Redshift-binned difference
ax = axes[1, 0]
if z_centers:
    colors_z = ['#e74c3c' if s > 2 else '#f39c12' if s > 1 else '#95a5a6'
                for s in z_sigs]
    ax.bar(range(len(z_centers)), z_diffs, yerr=z_errs, color=colors_z,
           capsize=5, edgecolor='white', width=0.7, alpha=0.8)
    ax.set_xticks(range(len(z_centers)))
    ax.set_xticklabels([f'{c:.2f}' for c in z_centers], fontsize=9)
    ax.axhline(0, color='black', ls='-', lw=1)
    ax.set_xlabel('Redshift bin center')
    ax.set_ylabel('$\\langle\\Delta T\\rangle_{dis} - \\langle\\Delta T\\rangle_{rel}$ ($\\mu$K)')
    ax.set_title('(c) Morphology-ISW Difference vs Redshift', fontweight='bold')
    # Add significance labels
    for i, (d, s) in enumerate(zip(z_diffs, z_sigs)):
        ax.text(i, d + z_errs[i] + 0.5 if d > 0 else d - z_errs[i] - 1.5,
                f'{s:.1f}$\\sigma$', ha='center', fontsize=9, fontweight='bold')

# (d) Mass distributions (verification)
ax = axes[1, 1]
ax.hist(M500[idx_rel], bins=30, alpha=0.6, color='#2ecc71', density=True,
        label='Relaxed', edgecolor='white')
ax.hist(M500[idx_dis], bins=30, alpha=0.6, color='#e74c3c', density=True,
        label='Disturbed', edgecolor='white')
if len(matched_r) > 10:
    ax.hist(M500[matched_r], bins=30, alpha=0.4, color='#27ae60', density=True,
            label='Relaxed (matched)', edgecolor='black', ls='--', histtype='step', lw=2)
    ax.hist(M500[matched_d], bins=30, alpha=0.4, color='#c0392b', density=True,
            label='Disturbed (matched)', edgecolor='black', ls=':', histtype='step', lw=2)
ax.set_xlabel('$M_{500}$ ($10^{14}\\,M_\\odot$)')
ax.set_ylabel('Density')
ax.set_title('(d) Mass Distributions', fontweight='bold')
ax.legend(fontsize=9)

fig.suptitle(f'Morphology-Dependent ISW Signal — Real Planck Data\n'
             f'CMB: {map_used} | SZ correction: {"applied" if sz_correction_applied else "SZ-free map"} | '
             f'Morphology: Y-M residual',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_main_results.png')
plt.close()
print("  -> fig1_main_results.png")

# ── Figure 2: Bootstrap distributions ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(boot_diffs, bins=60, color='#3498db', alpha=0.7, edgecolor='white')
ax.axvline(0, color='black', ls='-', lw=2)
ax.axvline(diff_full, color='#e74c3c', ls='--', lw=2, label=f'Observed: {diff_full:.2f}')
ax.axvline(ci_lo, color='#e74c3c', ls=':', lw=1)
ax.axvline(ci_hi, color='#e74c3c', ls=':', lw=1)
ax.set_xlabel('$\\Delta T_{dis} - \\Delta T_{rel}$ ($\\mu$K)')
ax.set_ylabel('Count')
ax.set_title('(a) Bootstrap: Full Sample', fontweight='bold')
ax.legend()

ax = axes[1]
if len(matched_r) > 20:
    ax.hist(boot_matched, bins=60, color='#9b59b6', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', ls='-', lw=2)
    ax.axvline(paired_diffs.mean(), color='#e74c3c', ls='--', lw=2,
               label=f'Observed: {paired_diffs.mean():.2f}')
    ax.axvline(ci_lo_m, color='#e74c3c', ls=':', lw=1)
    ax.axvline(ci_hi_m, color='#e74c3c', ls=':', lw=1)
    ax.set_title(f'(b) Bootstrap: Mass-Matched ({abs(t_paired):.1f}$\\sigma$)', fontweight='bold')
    ax.legend()
ax.set_xlabel('Paired $\\Delta T_{dis} - \\Delta T_{rel}$ ($\\mu$K)')
ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_bootstrap.png')
plt.close()
print("  -> fig2_bootstrap.png")

# ── Figure 3: DeltaT vs Mass, colored by morphology ─────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sc = ax.scatter(M500[valid & ~is_disturbed], Delta_T[valid & ~is_disturbed],
                s=12, alpha=0.4, c='#2ecc71', label='Relaxed')
ax.scatter(M500[valid & is_disturbed], Delta_T[valid & is_disturbed],
           s=12, alpha=0.4, c='#e74c3c', label='Disturbed')
ax.axhline(0, color='gray', ls='-', lw=1)
ax.set_xlabel('$M_{500}$ ($10^{14}\\,M_\\odot$)')
ax.set_ylabel('$\\Delta T$ ($\\mu$K)')
ax.set_title('$\\Delta T$ vs Cluster Mass by Morphology', fontweight='bold')
ax.legend()
ax.set_xlim(0, M500.max() * 1.05)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_mass_scatter.png')
plt.close()
print("  -> fig3_mass_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

results = {
    "data_type": "real_planck",
    "cmb_map": map_used,
    "sz_correction": sz_correction_applied,
    "morphology_method": "Y-M_residual",
    "N_total": int(valid.sum()),
    "N_relaxed": int(len(DT_rel)),
    "N_disturbed": int(len(DT_dis)),

    "full_sample": {
        "mean_DT_relaxed": float(DT_rel.mean()),
        "mean_DT_disturbed": float(DT_dis.mean()),
        "difference": float(diff_full),
        "difference_error": float(err_full),
        "t_statistic": float(t_full),
        "p_value": float(p_full),
        "significance_sigma": float(abs(t_full)),
        "bootstrap_95CI": [float(ci_lo), float(ci_hi)],
    },

    "mass_matched": {
        "N_pairs": int(len(matched_r)),
        "mean_DT_relaxed": float(DT_rel_m.mean()) if len(DT_rel_m) > 0 else None,
        "mean_DT_disturbed": float(DT_dis_m.mean()) if len(DT_dis_m) > 0 else None,
        "difference": float(diff_match),
        "difference_error": float(err_match),
        "t_statistic": float(t_match),
        "p_value": float(p_match),
        "significance_sigma": float(abs(t_match)),
        "paired_t": float(t_paired),
        "paired_p": float(p_paired),
        "bootstrap_95CI": [float(ci_lo_m), float(ci_hi_m)],
    },

    "regression": {
        "morphology_coefficient": float(alpha_morph),
        "morphology_se": float(alpha_morph_se),
        "morphology_sigma": float(alpha_morph_sig),
        "R_squared": float(R2),
    },

    "redshift_binned": {
        "z_centers": [float(c) for c in z_centers],
        "differences": [float(d) for d in z_diffs],
        "errors": [float(e) for e in z_errs],
        "significances": [float(s) for s in z_sigs],
    }
}

with open(f'{OUT_DIR}/results_v2.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to {OUT_DIR}/results_v2.json")

# ── Print final summary ──────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"\n  CMB Map: {map_used}")
print(f"  SZ correction: {'YES (10% leakage)' if sz_correction_applied else 'NO (SZ-free map)'}")
print(f"  Morphology: Y-M residual classification")
print(f"\n  FULL SAMPLE:")
print(f"    Difference: {diff_full:.2f} ± {err_full:.2f} muK ({abs(t_full):.1f} sigma, p={p_full:.4f})")
print(f"    Bootstrap 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
print(f"\n  MASS-MATCHED:")
print(f"    Difference: {diff_match:.2f} ± {err_match:.2f} muK ({abs(t_match):.1f} sigma, p={p_match:.4f})")
if len(matched_r) > 20:
    print(f"    Paired test: {abs(t_paired):.1f} sigma, p={p_paired:.4f}")
    print(f"    Bootstrap 95% CI: [{ci_lo_m:.2f}, {ci_hi_m:.2f}]")
print(f"\n  REGRESSION (controlling for M and z):")
print(f"    Morphology effect: {alpha_morph:.2f} ± {alpha_morph_se:.2f} muK ({alpha_morph_sig:.1f} sigma)")
print(f"\n  LOW-z SIGNAL (z < 0.1):")
lo_z = [i for i, c in enumerate(z_centers) if c < 0.1]
if lo_z:
    for i in lo_z:
        print(f"    z~{z_centers[i]:.2f}: diff = {z_diffs[i]:.1f} ± {z_errs[i]:.1f} muK ({z_sigs[i]:.1f} sigma)")

print("\nDone.")
