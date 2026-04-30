#!/usr/bin/env python3
"""
ISW Morphology Analysis v3 — Diego's Suggestions
==================================================

New analyses:
1. EXTREME morphology cuts (top/bottom 20%, 30%)
2. MULTIPLE independent dynamical state indicators
3. CLUSTER ENVIRONMENT: separation from nearest neighbor
4. Dose-response: signal strength vs morphology extremity

Run on Google Colab with Planck data already downloaded.

Author: Alika M. Parks
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from scipy.fft import fft
import os, json

from astropy.io import fits
import healpy as hp

OUT = "isw_results_v3"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13, 'axes.labelsize': 15,
    'axes.titlesize': 15, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("="*65)
print("ISW MORPHOLOGY v3 — EXTREME CUTS + ENVIRONMENT")
print("="*65)

# Load PSZ2
print("\n[1] Loading Planck PSZ2...")
hdu = fits.open("isw_data/psz2.fits")
psz2 = hdu[1].data

ra_all = psz2['RA']
dec_all = psz2['DEC']
z_all = psz2['REDSHIFT']
snr_all = psz2['SNR']
val = psz2['VALIDATION']
M500_all = psz2['MSZ']
Y5R500_all = psz2['Y5R500']
Y5R500_err_all = psz2['Y5R500_ERR']
glat = psz2['GLAT']
glon = psz2['GLON']

# Filter
good = (val >= 20) & np.isfinite(z_all) & (z_all > 0) & \
       np.isfinite(M500_all) & (M500_all > 0) & \
       np.isfinite(Y5R500_all) & (Y5R500_all > 0) & \
       np.isfinite(Y5R500_err_all) & \
       (np.abs(glat) > 15)

ra = ra_all[good]; dec = dec_all[good]; z = z_all[good]
snr = snr_all[good]; M500 = M500_all[good]
Y_SZ = Y5R500_all[good]; Y_err = Y5R500_err_all[good]

N = len(ra)
print(f"  {N} clusters after cuts")

# Load CMB
print("\n[2] Loading CMB map...")
CMB = None
for mf, label in [("isw_data/smica_nosz.fits", "SMICA no-SZ"),
                   ("isw_data/smica.fits", "SMICA")]:
    if os.path.exists(mf) and os.path.getsize(mf) > 1e8:
        print(f"  Loading {label}...")
        CMB = hp.read_map(mf, field=0, verbose=False)
        if np.nanstd(CMB[CMB != hp.UNSEEN]) < 0.01:
            CMB *= 1e6
            print("  Converted K -> muK")
        MAP_LABEL = label
        break

if CMB is None:
    print("  ERROR: No CMB map found"); exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# APERTURE PHOTOMETRY
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Measuring DeltaT at cluster locations...")

NSIDE = hp.npix2nside(len(CMB))

def ap_phot(ra_deg, dec_deg, disk_am=15.0, ann_am=45.0):
    theta = np.radians(90 - dec_deg)
    phi = np.radians(ra_deg)
    vec = hp.ang2vec(theta, phi)
    dp = hp.query_disc(NSIDE, vec, np.radians(disk_am/60))
    op = hp.query_disc(NSIDE, vec, np.radians(ann_am/60))
    ap = np.setdiff1d(op, dp)
    Td = CMB[dp]; Ta = CMB[ap]
    gd = np.isfinite(Td) & (Td != hp.UNSEEN) & (np.abs(Td) < 1e4)
    ga = np.isfinite(Ta) & (Ta != hp.UNSEEN) & (np.abs(Ta) < 1e4)
    if gd.sum() < 10 or ga.sum() < 10:
        return np.nan
    return np.mean(Td[gd]) - np.mean(Ta[ga])

DT = np.zeros(N)
for i in range(N):
    DT[i] = ap_phot(ra[i], dec[i])
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{N}...")

valid = np.isfinite(DT)
print(f"  Valid: {valid.sum()}/{N}")


# ══════════════════════════════════════════════════════════════════════════════
# MULTIPLE INDEPENDENT MORPHOLOGY INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] Computing morphology indicators...")

# Indicator 1: Y-M residual (primary, from v2)
log_Y = np.log10(Y_SZ)
log_M = np.log10(M500)
slope, intercept, _, _, _ = stats.linregress(log_M, log_Y)
Y_pred = 10**(intercept + slope * log_M)
Y_resid = (Y_SZ - Y_pred) / Y_pred  # fractional residual
print(f"  Y-M relation: log(Y) = {slope:.3f} * log(M) + {intercept:.3f}")

# Indicator 2: SNR-mass residual
# Disturbed clusters may have different SNR for their mass
log_snr = np.log10(snr)
s_slope, s_int, _, _, _ = stats.linregress(log_M, log_snr)
snr_pred = 10**(s_int + s_slope * log_M)
snr_resid = (snr - snr_pred) / snr_pred
print(f"  SNR-M relation: log(SNR) = {s_slope:.3f} * log(M) + {s_int:.3f}")

# Indicator 3: Y/Y_err ratio (signal significance as morphology proxy)
# Low Y/Y_err could indicate extended/disturbed pressure profile
Y_significance = Y_SZ / (Y_err + 1e-10)

# Indicator 4: Combined morphology score
# Normalize each indicator to [0,1] and average
def percentile_rank(x):
    return stats.rankdata(x) / len(x)

morph_score = (
    (1 - percentile_rank(Y_resid)) +     # low Y_resid = disturbed
    (1 - percentile_rank(snr_resid)) +    # low SNR_resid = disturbed
    (1 - percentile_rank(Y_significance)) # low significance = disturbed
) / 3.0

print(f"  Morphology score range: [{morph_score.min():.3f}, {morph_score.max():.3f}]")


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER ENVIRONMENT: NEAREST NEIGHBOR SEPARATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5] Computing cluster environment (nearest neighbor)...")

# Convert RA/Dec to 3D positions on unit sphere
ra_rad = np.radians(ra)
dec_rad = np.radians(dec)
x3d = np.cos(dec_rad) * np.cos(ra_rad)
y3d = np.cos(dec_rad) * np.sin(ra_rad)
z3d = np.sin(dec_rad)
coords = np.column_stack([x3d, y3d, z3d])

# Build KD-tree for fast neighbor search
tree = cKDTree(coords)

# Find nearest neighbor distance (angular separation in degrees)
dd, ii = tree.query(coords, k=2)  # k=2 because nearest is self
nn_dist_rad = dd[:, 1]  # second nearest = actual nearest neighbor
nn_ang_sep = np.degrees(2 * np.arcsin(nn_dist_rad / 2))  # chord to angle

# Also compute physical separation using redshift
# Approximate comoving distance (flat LCDM, Omega_M=0.3)
def comoving_dist(z_val):
    """Simple numerical integration for comoving distance (Mpc)."""
    from scipy.integrate import quad
    H0 = 70.0  # km/s/Mpc
    c = 3e5  # km/s
    OmM = 0.3; OmL = 0.7
    def integrand(zp):
        return 1.0 / np.sqrt(OmM*(1+zp)**3 + OmL)
    result, _ = quad(integrand, 0, z_val)
    return c / H0 * result

# Compute for each cluster (approximate: use angular sep * distance)
nn_phys_sep = np.zeros(N)
for i in range(N):
    d_c = comoving_dist(z[i])
    nn_phys_sep[i] = d_c * np.radians(nn_ang_sep[i])  # Mpc

med_sep = np.median(nn_phys_sep)
print(f"  Median nearest-neighbor separation: {med_sep:.1f} Mpc")
print(f"  Range: [{nn_phys_sep.min():.1f}, {nn_phys_sep.max():.1f}] Mpc")

# Define isolated vs clustered
isolated = nn_phys_sep > np.percentile(nn_phys_sep, 70)  # top 30% most isolated
clustered = nn_phys_sep < np.percentile(nn_phys_sep, 30)  # bottom 30% most clustered
print(f"  Isolated (top 30%): {isolated.sum()}")
print(f"  Clustered (bottom 30%): {clustered.sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: EXTREME MORPHOLOGY CUTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("ANALYSIS 1: EXTREME MORPHOLOGY CUTS")
print("  Does signal strengthen with more extreme definitions?")
print("="*65)

def compare_subsamples(mask_rel, mask_dis, label, mass_match=False):
    """Compare DeltaT between two subsamples."""
    m_r = mask_rel & valid
    m_d = mask_dis & valid
    
    dt_r = DT[m_r]
    dt_d = DT[m_d]
    
    if len(dt_r) < 10 or len(dt_d) < 10:
        print(f"  {label}: insufficient data (n_r={len(dt_r)}, n_d={len(dt_d)})")
        return None
    
    diff = dt_d.mean() - dt_r.mean()
    err = np.sqrt(dt_d.var()/len(dt_d) + dt_r.var()/len(dt_r))
    t_stat, p_val = stats.ttest_ind(dt_d, dt_r, equal_var=False)
    
    print(f"  {label}:")
    print(f"    Relaxed:   n={len(dt_r):>4},  <DT> = {dt_r.mean():>7.2f} muK")
    print(f"    Disturbed: n={len(dt_d):>4},  <DT> = {dt_d.mean():>7.2f} muK")
    print(f"    Diff = {diff:>7.2f} ± {err:.2f} muK,  {abs(t_stat):.1f}σ,  p = {p_val:.4f}")
    
    return {
        'label': label,
        'n_rel': int(len(dt_r)), 'n_dis': int(len(dt_d)),
        'DT_rel': float(dt_r.mean()), 'DT_dis': float(dt_d.mean()),
        'diff': float(diff), 'err': float(err),
        'sigma': float(abs(t_stat)), 'p': float(p_val),
    }

# Using Y-M residual (primary indicator)
cuts_results = []

# Median split (50/50)
r = compare_subsamples(Y_resid > 0, Y_resid <= 0, "Y-M residual: median split")
if r: cuts_results.append(r)

# 40/40 (drop middle 20%)
p30 = np.percentile(Y_resid, 30)
p70 = np.percentile(Y_resid, 70)
r = compare_subsamples(Y_resid > p70, Y_resid < p30, "Y-M residual: 30/70 split (drop middle)")
if r: cuts_results.append(r)

# 25/25 (extreme quartiles)
p25 = np.percentile(Y_resid, 25)
p75 = np.percentile(Y_resid, 75)
r = compare_subsamples(Y_resid > p75, Y_resid < p25, "Y-M residual: extreme quartiles")
if r: cuts_results.append(r)

# 20/20 (extreme quintiles)
p20 = np.percentile(Y_resid, 20)
p80 = np.percentile(Y_resid, 80)
r = compare_subsamples(Y_resid > p80, Y_resid < p20, "Y-M residual: extreme 20%")
if r: cuts_results.append(r)

# 10/10 (extreme deciles)
p10 = np.percentile(Y_resid, 10)
p90 = np.percentile(Y_resid, 90)
r = compare_subsamples(Y_resid > p90, Y_resid < p10, "Y-M residual: extreme 10%")
if r: cuts_results.append(r)

# Using combined morphology score
print("\n  --- Combined morphology score ---")
r = compare_subsamples(morph_score < 0.4, morph_score > 0.6, "Combined score: 40/60 split")
if r: cuts_results.append(r)

r = compare_subsamples(morph_score < 0.25, morph_score > 0.75, "Combined score: extreme quartiles")
if r: cuts_results.append(r)

r = compare_subsamples(morph_score < 0.2, morph_score > 0.8, "Combined score: extreme 20%")
if r: cuts_results.append(r)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: INDEPENDENT DYNAMICAL STATE INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("ANALYSIS 2: INDEPENDENT DYNAMICAL STATE INDICATORS")
print("  Does the signal appear regardless of which indicator we use?")
print("="*65)

indicators = {
    "Y-M residual": Y_resid,
    "SNR-M residual": snr_resid,
    "Y significance": Y_significance,
    "Combined score": morph_score,
}

indicator_results = []
for name, ind in indicators.items():
    med = np.median(ind)
    
    # For combined score, high = disturbed
    if name == "Combined score":
        mask_r = ind < med
        mask_d = ind > med
    else:
        # For residuals, low = disturbed (less signal than expected)
        mask_r = ind > med
        mask_d = ind <= med
    
    r = compare_subsamples(mask_r, mask_d, f"Indicator: {name}")
    if r: indicator_results.append(r)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: CLUSTER ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("ANALYSIS 3: CLUSTER ENVIRONMENT (nearest neighbor separation)")
print("  Does isolation vs clustering affect the ISW signal?")
print("="*65)

env_results = []

# Isolated vs clustered (all morphologies)
r = compare_subsamples(clustered, isolated, "Clustered vs Isolated (all)")
if r: env_results.append(r)

# Disturbed + isolated vs disturbed + clustered
morph_dis = Y_resid < 0  # disturbed by Y-M residual
morph_rel = Y_resid > 0

r = compare_subsamples(morph_dis & clustered, morph_dis & isolated, 
                       "Disturbed: clustered vs isolated")
if r: env_results.append(r)

r = compare_subsamples(morph_rel & clustered, morph_rel & isolated,
                       "Relaxed: clustered vs isolated")
if r: env_results.append(r)

# Cross-cut: morphology x environment (2x2)
print("\n  --- 2×2 Cross-cut: Morphology × Environment ---")
for m_label, m_mask in [("Relaxed", morph_rel), ("Disturbed", morph_dis)]:
    for e_label, e_mask in [("Clustered", clustered), ("Isolated", isolated)]:
        subset = m_mask & e_mask & valid
        n = subset.sum()
        if n >= 10:
            dt_sub = DT[subset]
            print(f"  {m_label:>10} + {e_label:<10}: n={n:>4}, "
                  f"<DT> = {dt_sub.mean():>7.2f} ± {dt_sub.std()/np.sqrt(n):.2f} muK")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: DOSE-RESPONSE (signal vs morphology extremity)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("ANALYSIS 4: DOSE-RESPONSE")
print("  Does signal strength scale with morphology extremity?")
print("="*65)

# Divide into quintiles of Y-M residual
n_bins = 5
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(Y_resid, percentiles)

dose_results = []
print(f"\n  {'Quintile':<12} {'Y_resid range':<20} {'n':>5} {'<DT>':>8} {'err':>7}")
print("  " + "-"*55)

for b in range(n_bins):
    lo = bin_edges[b]
    hi = bin_edges[b + 1]
    mask = (Y_resid >= lo) & (Y_resid < hi) & valid
    if b == n_bins - 1:  # include upper edge for last bin
        mask = (Y_resid >= lo) & (Y_resid <= hi) & valid
    
    n = mask.sum()
    if n >= 5:
        dt_bin = DT[mask]
        mean_dt = dt_bin.mean()
        err_dt = dt_bin.std() / np.sqrt(n)
        mid = (lo + hi) / 2
        
        dose_results.append({
            'quintile': b + 1,
            'Y_resid_mid': float(mid),
            'n': int(n),
            'DT_mean': float(mean_dt),
            'DT_err': float(err_dt),
        })
        
        label = "most disturbed" if b == 0 else "most relaxed" if b == n_bins-1 else ""
        print(f"  Q{b+1:<10} [{lo:>7.3f}, {hi:>7.3f}]  {n:>5} {mean_dt:>8.2f} {err_dt:>7.2f}  {label}")

# Linear trend
if len(dose_results) >= 3:
    x_dose = [d['Y_resid_mid'] for d in dose_results]
    y_dose = [d['DT_mean'] for d in dose_results]
    slope_dose, int_dose, r_val, p_dose, _ = stats.linregress(x_dose, y_dose)
    print(f"\n  Linear trend: slope = {slope_dose:.2f} muK per unit Y_resid")
    print(f"  r = {r_val:.3f}, p = {p_dose:.4f}")
    print(f"  {'SIGNIFICANT' if p_dose < 0.05 else 'NOT significant'} dose-response")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("GENERATING FIGURES")
print("="*65)

# ── Figure 1: Dose-response curve ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
if dose_results:
    x = [d['Y_resid_mid'] for d in dose_results]
    y = [d['DT_mean'] for d in dose_results]
    yerr = [d['DT_err'] for d in dose_results]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(x)))
    
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=6, ms=10, lw=2,
                color='#3498db', mfc='white', mew=2, zorder=5)
    for i, d in enumerate(dose_results):
        ax.scatter(x[i], y[i], s=150, c=[colors[i]], edgecolors='black',
                   linewidths=1, zorder=6)
    
    if len(x) >= 3:
        x_fit = np.linspace(min(x), max(x), 100)
        ax.plot(x_fit, int_dose + slope_dose * x_fit, 'r--', lw=2,
                label=f'Trend: slope={slope_dose:.1f}, p={p_dose:.3f}')
    
    ax.axhline(0, color='gray', ls='-', lw=1)
    ax.set_xlabel('Y-M Residual (← disturbed | relaxed →)')
    ax.set_ylabel('$\\langle \\Delta T \\rangle$ ($\\mu$K)')
    ax.set_title('(a) Dose-Response: $\\Delta T$ vs Morphology', fontweight='bold')
    ax.legend(fontsize=10)

# ── Panel b: Extreme cuts comparison ─────────────────────────────────────
ax = axes[1]
if cuts_results:
    labels_c = [r['label'].split(': ')[1] if ':' in r['label'] else r['label'] 
                for r in cuts_results[:5]]
    diffs = [r['diff'] for r in cuts_results[:5]]
    errs = [r['err'] for r in cuts_results[:5]]
    sigs = [r['sigma'] for r in cuts_results[:5]]
    
    colors_c = ['#e74c3c' if s > 2 else '#f39c12' if s > 1 else '#95a5a6' for s in sigs]
    
    y_pos = range(len(labels_c))
    ax.barh(y_pos, diffs, xerr=errs, color=colors_c, capsize=5,
            height=0.6, edgecolor='white', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_c, fontsize=10)
    ax.axvline(0, color='black', ls='-', lw=1)
    
    for i, (d, s) in enumerate(zip(diffs, sigs)):
        ax.text(d - errs[i] - 1 if d < 0 else d + errs[i] + 0.5, i,
                f'{s:.1f}σ', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('$\\Delta T_{dis} - \\Delta T_{rel}$ ($\\mu$K)')
    ax.set_title('(b) Signal vs Cut Extremity', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/fig1_dose_response.png')
plt.close()
print("  -> fig1_dose_response.png")

# ── Figure 2: Environment cross-cut ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2x2 heatmap
ax = axes[0]
grid_labels = [['Relaxed\nClustered', 'Relaxed\nIsolated'],
               ['Disturbed\nClustered', 'Disturbed\nIsolated']]
grid_vals = np.zeros((2, 2))
grid_n = np.zeros((2, 2), dtype=int)

for mi, (m_mask, m_label) in enumerate([(morph_rel, "R"), (morph_dis, "D")]):
    for ei, (e_mask, e_label) in enumerate([(clustered, "C"), (isolated, "I")]):
        subset = m_mask & e_mask & valid
        n = subset.sum()
        grid_n[mi, ei] = n
        if n >= 5:
            grid_vals[mi, ei] = DT[subset].mean()

im = ax.imshow(grid_vals, cmap='RdBu_r', vmin=-15, vmax=15, aspect='auto')
for mi in range(2):
    for ei in range(2):
        ax.text(ei, mi, f'{grid_vals[mi,ei]:.1f} μK\nn={grid_n[mi,ei]}',
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white' if abs(grid_vals[mi,ei]) > 8 else 'black')

ax.set_xticks([0, 1]); ax.set_xticklabels(['Clustered', 'Isolated'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Relaxed', 'Disturbed'])
ax.set_title('(a) Morphology × Environment: $\\langle\\Delta T\\rangle$', fontweight='bold')
plt.colorbar(im, ax=ax, label='$\\mu$K', shrink=0.8)

# NN separation histogram
ax = axes[1]
ax.hist(nn_phys_sep[valid & morph_rel], bins=30, alpha=0.6, color='#2ecc71',
        density=True, label='Relaxed', edgecolor='white')
ax.hist(nn_phys_sep[valid & morph_dis], bins=30, alpha=0.6, color='#e74c3c',
        density=True, label='Disturbed', edgecolor='white')
ax.axvline(np.percentile(nn_phys_sep, 30), color='gray', ls='--', lw=1.5,
           label='Clustered cut')
ax.axvline(np.percentile(nn_phys_sep, 70), color='gray', ls=':', lw=1.5,
           label='Isolated cut')
ax.set_xlabel('Nearest Neighbor Separation (Mpc)')
ax.set_ylabel('Density')
ax.set_title('(b) Cluster Environment Distribution', fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT}/fig2_environment.png')
plt.close()
print("  -> fig2_environment.png")

# ── Figure 3: All indicators comparison ──────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
if indicator_results:
    names = [r['label'].replace('Indicator: ', '') for r in indicator_results]
    diffs_i = [r['diff'] for r in indicator_results]
    errs_i = [r['err'] for r in indicator_results]
    sigs_i = [r['sigma'] for r in indicator_results]
    
    colors_i = ['#e74c3c' if s > 2 else '#f39c12' if s > 1.5 else '#95a5a6' for s in sigs_i]
    
    y_pos = range(len(names))
    ax.barh(y_pos, diffs_i, xerr=errs_i, color=colors_i, capsize=5,
            height=0.5, edgecolor='white', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.axvline(0, color='black', ls='-', lw=1)
    
    for i, (d, s) in enumerate(zip(diffs_i, sigs_i)):
        side = -1 if d < 0 else 1
        ax.text(d + side * (errs_i[i] + 1), i,
                f'{s:.1f}σ', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('$\\Delta T_{dis} - \\Delta T_{rel}$ ($\\mu$K)')
    ax.set_title('Independent Dynamical State Indicators\n'
                 'All showing same sign?', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/fig3_indicators.png')
plt.close()
print("  -> fig3_indicators.png")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

results = {
    "analysis": "ISW Morphology v3 — Diego suggestions",
    "N_clusters": int(N),
    "CMB_map": MAP_LABEL,
    
    "extreme_cuts": cuts_results,
    "independent_indicators": indicator_results,
    "environment": env_results,
    "dose_response": {
        "quintiles": dose_results,
        "linear_slope": float(slope_dose) if 'slope_dose' in dir() else None,
        "linear_p": float(p_dose) if 'p_dose' in dir() else None,
    },
}

with open(f'{OUT}/results_v3.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SUMMARY")
print("="*65)

print("\n  DOSE-RESPONSE:")
if 'slope_dose' in dir():
    print(f"    Trend slope: {slope_dose:.2f} muK per unit Y_resid, p={p_dose:.4f}")
    print(f"    {'SIGNIFICANT' if p_dose < 0.05 else 'Not significant'}")

print("\n  EXTREME CUTS (does signal strengthen?):")
if cuts_results:
    for r in cuts_results[:5]:
        print(f"    {r['label'].split(': ')[1] if ':' in r['label'] else r['label']}: "
              f"{r['diff']:.1f} muK, {r['sigma']:.1f}σ")

print("\n  INDEPENDENT INDICATORS (all same sign?):")
if indicator_results:
    signs = [np.sign(r['diff']) for r in indicator_results]
    all_same = len(set(signs)) == 1
    for r in indicator_results:
        print(f"    {r['label'].replace('Indicator: ','')}: "
              f"{r['diff']:.1f} muK, {r['sigma']:.1f}σ")
    print(f"    All same sign: {'YES' if all_same else 'NO'}")

print("\n  ENVIRONMENT:")
if env_results:
    for r in env_results:
        print(f"    {r['label']}: {r['diff']:.1f} muK, {r['sigma']:.1f}σ")

print(f"\n  Results: {OUT}/results_v3.json")
print("Done.")
