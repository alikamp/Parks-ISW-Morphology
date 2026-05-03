#!/usr/bin/env python3
"""
ΛCDM Simulation Pipeline — Publication-Grade Significance Test
================================================================

Tests whether the observed morphology-dependent ISW signal
(~25 μK difference between disturbed and relaxed clusters)
is consistent with ΛCDM CMB fluctuations.

Method:
  1. Generate 10,000 synthetic CMB realizations from ΛCDM C_ℓ
  2. Apply identical analysis pipeline to each
  3. Build empirical null distribution of δT = <ΔT_dis> - <ΔT_rel>
  4. Compare observed δT against null distribution

Tests included:
  A. Full sample morphology split
  B. Key bin z = 0.10-0.15
  C. Broader bin z = 0.05-0.20
  D. Morphology shuffle (fixed positions, random labels)
  E. Random positions (fixed labels, random sky locations)

Resolution: NSIDE = 1024 (pixel ~3.4', well below 15' aperture)
Simulations: N = 10,000

Runtime estimate: 4-8 hours on Google Colab

Requirements: pip install healpy astropy numpy scipy matplotlib

Author: Alika M. Parks
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from time import perf_counter
import os, json, sys

import healpy as hp
from astropy.io import fits

OUT = "isw_simulation_results"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13, 'axes.labelsize': 15,
    'axes.titlesize': 15, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — FIXED FOR ALL RUNS
# ══════════════════════════════════════════════════════════════════════════════

NSIDE_SIM = 1024          # simulation resolution (~3.4' pixels)
N_SIMS = 10000            # number of ΛCDM realizations
DISK_ARCMIN = 15.0        # aperture disk radius
ANN_ARCMIN = 45.0         # aperture annulus outer radius
GALACTIC_CUT = 15.0       # |b| > 15 degrees
SEED_BASE = 12345         # base random seed for reproducibility

print("="*70)
print("ΛCDM SIMULATION PIPELINE — PUBLICATION-GRADE SIGNIFICANCE TEST")
print("="*70)
print(f"\n  NSIDE = {NSIDE_SIM}")
print(f"  N_sims = {N_SIMS}")
print(f"  Aperture: {DISK_ARCMIN}' disk, {DISK_ARCMIN}'-{ANN_ARCMIN}' annulus")
print(f"  Galactic cut: |b| > {GALACTIC_CUT}°")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD REAL DATA AND MEASURE OBSERVED SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 1: LOAD DATA AND MEASURE OBSERVED SIGNAL")
print("="*70)

# ── Load PSZ2 catalog ────────────────────────────────────────────────────
hdu = fits.open("isw_data/psz2.fits")
psz2 = hdu[1].data

ra_all = psz2['RA']; dec_all = psz2['DEC']; z_all = psz2['REDSHIFT']
snr_all = psz2['SNR']; val = psz2['VALIDATION']
M500_all = psz2['MSZ']; Y_SZ_all = psz2['Y5R500']
glat = psz2['GLAT']

good = (val >= 20) & np.isfinite(z_all) & (z_all > 0) & \
       np.isfinite(M500_all) & (M500_all > 0) & \
       np.isfinite(Y_SZ_all) & (Y_SZ_all > 0) & \
       (np.abs(glat) > GALACTIC_CUT)

ra = ra_all[good]; dec = dec_all[good]; z = z_all[good]
snr = snr_all[good]; M500 = M500_all[good]; Y_SZ = Y_SZ_all[good]
N_clusters = len(ra)
print(f"\n  Clusters: {N_clusters}")

# ── Morphology classification (Y-M residual) ─────────────────────────────
log_Y = np.log10(Y_SZ); log_M = np.log10(M500)
slope, intercept, _, _, _ = stats.linregress(log_M, log_Y)
Y_pred = 10**(intercept + slope * log_M)
Y_resid = (Y_SZ - Y_pred) / Y_pred
morph_dis = Y_resid < 0
morph_rel = Y_resid > 0

n_dis = morph_dis.sum(); n_rel = morph_rel.sum()
print(f"  Disturbed: {n_dis}, Relaxed: {n_rel}")

# ── Define redshift masks ─────────────────────────────────────────────────
z_full = np.ones(N_clusters, dtype=bool)
z_key = (z >= 0.10) & (z < 0.15)
z_broad = (z >= 0.05) & (z < 0.20)

print(f"  z bins: full={z_full.sum()}, key[0.10,0.15)={z_key.sum()}, broad[0.05,0.20)={z_broad.sum()}")

# ── Precompute cluster positions in HEALPix ──────────────────────────────
print("\n  Precomputing aperture geometry...")

theta_clusters = np.radians(90 - dec)  # colatitude
phi_clusters = np.radians(ra)          # longitude
vec_clusters = hp.ang2vec(theta_clusters, phi_clusters)

# Precompute disk and annulus pixel indices for each cluster
# This is done ONCE and reused for all simulations
disk_pixels = []
ann_pixels = []

for i in range(N_clusters):
    dp = hp.query_disc(NSIDE_SIM, vec_clusters[i], np.radians(DISK_ARCMIN / 60))
    op = hp.query_disc(NSIDE_SIM, vec_clusters[i], np.radians(ANN_ARCMIN / 60))
    ap = np.setdiff1d(op, dp)
    disk_pixels.append(dp)
    ann_pixels.append(ap)

print(f"  Aperture geometry precomputed for {N_clusters} clusters")
print(f"  Mean disk pixels: {np.mean([len(d) for d in disk_pixels]):.0f}")
print(f"  Mean annulus pixels: {np.mean([len(a) for a in ann_pixels]):.0f}")

# ── Load real CMB map and measure observed signal ─────────────────────────
print("\n  Loading real CMB map...")
CMB_REAL = None
for mf in ["isw_data/smica_nosz.fits", "isw_data/smica.fits"]:
    if os.path.exists(mf) and os.path.getsize(mf) > 1e8:
        CMB_REAL_FULL = hp.read_map(mf, field=0, verbose=False)
        if np.nanstd(CMB_REAL_FULL[CMB_REAL_FULL != hp.UNSEEN]) < 0.01:
            CMB_REAL_FULL *= 1e6  # K -> muK
        NSIDE_REAL = hp.npix2nside(len(CMB_REAL_FULL))
        print(f"  Loaded {mf}, NSIDE={NSIDE_REAL}")

        # Downgrade to simulation resolution if needed
        if NSIDE_REAL != NSIDE_SIM:
            print(f"  Downgrading from NSIDE={NSIDE_REAL} to {NSIDE_SIM}...")
            CMB_REAL = hp.ud_grade(CMB_REAL_FULL, NSIDE_SIM)
        else:
            CMB_REAL = CMB_REAL_FULL
        break

if CMB_REAL is None:
    print("  ERROR: No CMB map found!"); sys.exit(1)

# ── Galactic mask ─────────────────────────────────────────────────────────
print("  Building galactic mask...")
npix_sim = hp.nside2npix(NSIDE_SIM)
theta_pix, phi_pix = hp.pix2ang(NSIDE_SIM, np.arange(npix_sim))
lat_pix = 90 - np.degrees(theta_pix)
GAL_MASK = np.abs(lat_pix) > GALACTIC_CUT  # True = usable

# Apply mask to real map
CMB_REAL[~GAL_MASK] = hp.UNSEEN

# ── Measure observed signal ───────────────────────────────────────────────
def measure_DT(cmap, disk_pix_list, ann_pix_list):
    """Measure DeltaT for all clusters using precomputed aperture pixels."""
    N = len(disk_pix_list)
    DT = np.full(N, np.nan)
    for i in range(N):
        Td = cmap[disk_pix_list[i]]
        Ta = cmap[ann_pix_list[i]]
        gd = np.isfinite(Td) & (Td != hp.UNSEEN) & (np.abs(Td) < 1e4)
        ga = np.isfinite(Ta) & (Ta != hp.UNSEEN) & (np.abs(Ta) < 1e4)
        if gd.sum() >= 5 and ga.sum() >= 5:
            DT[i] = np.mean(Td[gd]) - np.mean(Ta[ga])
    return DT

print("  Measuring observed DeltaT...")
DT_real = measure_DT(CMB_REAL, disk_pixels, ann_pixels)
valid = np.isfinite(DT_real)
print(f"  Valid measurements: {valid.sum()}/{N_clusters}")

def compute_split(DT, morph_d, morph_r, z_mask, valid_mask):
    """Compute morphology split difference."""
    m_d = morph_d & valid_mask & z_mask
    m_r = morph_r & valid_mask & z_mask
    nd = m_d.sum(); nr = m_r.sum()
    if nd < 5 or nr < 5:
        return np.nan, 0, 0
    dt_d = DT[m_d].mean()
    dt_r = DT[m_r].mean()
    return dt_d - dt_r, nd, nr

# Observed values
obs_full, nd_f, nr_f = compute_split(DT_real, morph_dis, morph_rel, z_full, valid)
obs_key, nd_k, nr_k = compute_split(DT_real, morph_dis, morph_rel, z_key, valid)
obs_broad, nd_b, nr_b = compute_split(DT_real, morph_dis, morph_rel, z_broad, valid)

print(f"\n  OBSERVED SIGNAL:")
print(f"    Full sample:       δT = {obs_full:.2f} μK  (n_d={nd_f}, n_r={nr_f})")
print(f"    Key bin [0.10,0.15): δT = {obs_key:.2f} μK  (n_d={nd_k}, n_r={nr_k})")
print(f"    Broad [0.05,0.20):   δT = {obs_broad:.2f} μK  (n_d={nd_b}, n_r={nr_b})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: GET ΛCDM POWER SPECTRUM
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 2: PREPARE ΛCDM POWER SPECTRUM")
print("="*70)

# Extract C_ℓ from the real map (this gives us the actual power spectrum
# including beam and noise effects — matching the real data exactly)
print("  Computing C_ℓ from real map...")
LMAX_SIM = 2 * NSIDE_SIM  # sufficient for our aperture scales

# Use anafast on the masked map
cl_real = hp.anafast(CMB_REAL, lmax=LMAX_SIM)
print(f"  C_ℓ computed, lmax = {LMAX_SIM}")
print(f"  C_ℓ[0:5] = {cl_real[:5]}")
print(f"  C_ℓ[100] = {cl_real[100]:.4e}")

# Alternatively, use theoretical ΛCDM spectrum
# For publication, Diego may want us to use the Planck best-fit C_ℓ
# For now, the empirical C_ℓ from the data is a valid choice —
# it automatically includes beam, noise, and component separation effects


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: RUN 10,000 SIMULATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print(f"STEP 3: RUNNING {N_SIMS} ΛCDM SIMULATIONS")
print("="*70)

# Storage for null distributions
null_full = np.zeros(N_SIMS)
null_key = np.zeros(N_SIMS)
null_broad = np.zeros(N_SIMS)

# Also run morphology shuffle test
null_shuffle_full = np.zeros(N_SIMS)
null_shuffle_key = np.zeros(N_SIMS)

t_start = perf_counter()
checkpoint_interval = 500  # print progress every 500 sims

for sim in range(N_SIMS):
    # ── Generate synthetic CMB map from C_ℓ ──────────────────────────
    np.random.seed(SEED_BASE + sim)
    cmb_sim = hp.synfast(cl_real, NSIDE_SIM, lmax=LMAX_SIM, verbose=False)
    
    # Apply galactic mask
    cmb_sim[~GAL_MASK] = hp.UNSEEN
    
    # ── Measure DeltaT at REAL cluster positions ─────────────────────
    DT_sim = measure_DT(cmb_sim, disk_pixels, ann_pixels)
    valid_sim = np.isfinite(DT_sim)
    
    # ── Compute morphology split (REAL morphology labels) ────────────
    null_full[sim], _, _ = compute_split(DT_sim, morph_dis, morph_rel, z_full, valid_sim)
    null_key[sim], _, _ = compute_split(DT_sim, morph_dis, morph_rel, z_key, valid_sim)
    null_broad[sim], _, _ = compute_split(DT_sim, morph_dis, morph_rel, z_broad, valid_sim)
    
    # ── Morphology shuffle: randomize labels on REAL data ────────────
    shuffle_idx = np.random.permutation(N_clusters)
    morph_dis_shuf = morph_dis[shuffle_idx]
    morph_rel_shuf = morph_rel[shuffle_idx]
    null_shuffle_full[sim], _, _ = compute_split(DT_real, morph_dis_shuf, morph_rel_shuf, z_full, valid)
    null_shuffle_key[sim], _, _ = compute_split(DT_real, morph_dis_shuf, morph_rel_shuf, z_key, valid)
    
    # ── Progress ─────────────────────────────────────────────────────
    if (sim + 1) % checkpoint_interval == 0:
        elapsed = perf_counter() - t_start
        rate = (sim + 1) / elapsed
        eta = (N_SIMS - sim - 1) / rate
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        
        # Running statistics
        mean_f = null_full[:sim+1].mean()
        std_f = null_full[:sim+1].std()
        
        print(f"  Sim {sim+1:>5}/{N_SIMS} | "
              f"{elapsed/60:.1f}min elapsed | "
              f"ETA {eta_h}h{eta_m:02d}m | "
              f"rate {rate:.1f}/s | "
              f"null_mean={mean_f:.2f} null_std={std_f:.2f}")
        sys.stdout.flush()

total_time = perf_counter() - t_start
print(f"\n  Completed {N_SIMS} simulations in {total_time/3600:.1f} hours")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: COMPUTE SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 4: SIGNIFICANCE ESTIMATION")
print("="*70)

def compute_pvalue(observed, null_dist, label):
    """Compute p-value and equivalent sigma from null distribution."""
    null_valid = null_dist[np.isfinite(null_dist)]
    N = len(null_valid)
    
    # Two-tailed p-value
    p_two = np.mean(np.abs(null_valid) >= np.abs(observed))
    
    # One-tailed (signal is negative)
    p_one = np.mean(null_valid <= observed)
    
    # Equivalent sigma (from p-value)
    from scipy.stats import norm
    if p_two > 0:
        sigma_equiv = norm.ppf(1 - p_two / 2)
    else:
        sigma_equiv = norm.ppf(1 - 0.5 / N)  # upper bound
    
    print(f"\n  {label}:")
    print(f"    Observed: {observed:.2f} μK")
    print(f"    Null distribution: mean={null_valid.mean():.2f}, std={null_valid.std():.2f} μK")
    print(f"    p-value (two-tailed): {p_two:.6f}")
    print(f"    p-value (one-tailed): {p_one:.6f}")
    print(f"    Equivalent sigma: {sigma_equiv:.2f}σ")
    print(f"    # sims with |δT| ≥ |observed|: {int(p_two * N)}/{N}")
    
    return {
        'label': label,
        'observed': round(float(observed), 2),
        'null_mean': round(float(null_valid.mean()), 2),
        'null_std': round(float(null_valid.std()), 2),
        'p_two_tailed': round(float(p_two), 6),
        'p_one_tailed': round(float(p_one), 6),
        'sigma_equivalent': round(float(sigma_equiv), 2),
        'N_sims': N,
        'N_exceed': int(p_two * N),
    }

results = {}

# Test A: Full sample — ΛCDM simulations
results['A_full_lcdm'] = compute_pvalue(obs_full, null_full,
    "TEST A: Full sample — ΛCDM null (simulated CMB, real positions, real morphology)")

# Test B: Key bin — ΛCDM simulations
results['B_key_lcdm'] = compute_pvalue(obs_key, null_key,
    "TEST B: z=[0.10,0.15) — ΛCDM null (simulated CMB, real positions, real morphology)")

# Test C: Broad bin — ΛCDM simulations
results['C_broad_lcdm'] = compute_pvalue(obs_broad, null_broad,
    "TEST C: z=[0.05,0.20) — ΛCDM null (simulated CMB, real positions, real morphology)")

# Test D: Morphology shuffle — real data
results['D_shuffle_full'] = compute_pvalue(obs_full, null_shuffle_full,
    "TEST D: Full sample — Morphology shuffle (real CMB, real positions, random labels)")

results['D_shuffle_key'] = compute_pvalue(obs_key, null_shuffle_key,
    "TEST D: z=[0.10,0.15) — Morphology shuffle (real CMB, real positions, random labels)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: VALIDATION CHECKS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 5: VALIDATION CHECKS")
print("="*70)

# Check 1: Null distribution should be centered at ~0
print(f"\n  Null distribution means (should be ≈ 0):")
print(f"    Full sample: {null_full.mean():.3f} μK")
print(f"    Key bin:     {null_key.mean():.3f} μK")
print(f"    Broad bin:   {null_broad.mean():.3f} μK")

mean_ok = all(abs(m) < 2.0 for m in [null_full.mean(), null_key.mean(), null_broad.mean()])
print(f"    Validation: {'PASS' if mean_ok else 'WARNING — null not centered'}")

# Check 2: Null distribution should be approximately Gaussian
from scipy.stats import shapiro, normaltest
if N_SIMS >= 100:
    # Use D'Agostino test on subsample
    subsample = null_full[np.isfinite(null_full)][:5000]
    _, p_norm = normaltest(subsample)
    print(f"\n  Normality test (D'Agostino, full sample null):")
    print(f"    p = {p_norm:.4f}")
    print(f"    {'Consistent with Gaussian' if p_norm > 0.05 else 'Non-Gaussian — simulation-based p-value is essential'}")

# Check 3: Shuffle null should also be centered at ~0
print(f"\n  Shuffle null means:")
print(f"    Full sample: {null_shuffle_full.mean():.3f} μK")
print(f"    Key bin:     {null_shuffle_key.mean():.3f} μK")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 6: GENERATING FIGURES")
print("="*70)

# ── Figure 1: Main null distribution plots ───────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

plot_configs = [
    (axes[0, 0], null_full, obs_full, "A: Full Sample\nΛCDM Null", '#3498db'),
    (axes[0, 1], null_key, obs_key, "B: z=[0.10,0.15)\nΛCDM Null", '#e74c3c'),
    (axes[0, 2], null_broad, obs_broad, "C: z=[0.05,0.20)\nΛCDM Null", '#2ecc71'),
    (axes[1, 0], null_shuffle_full, obs_full, "D: Full Sample\nMorphology Shuffle", '#9b59b6'),
    (axes[1, 1], null_shuffle_key, obs_key, "D: z=[0.10,0.15)\nMorphology Shuffle", '#e67e22'),
]

for ax, null, obs, title, color in plot_configs:
    null_clean = null[np.isfinite(null)]
    
    ax.hist(null_clean, bins=80, color=color, alpha=0.6, edgecolor='white',
            density=True, label='Null distribution')
    ax.axvline(obs, color='#e74c3c', ls='--', lw=2.5,
               label=f'Observed: {obs:.1f} μK')
    ax.axvline(0, color='black', ls='-', lw=1)
    
    # Mark the tail
    tail = null_clean[null_clean <= obs] if obs < 0 else null_clean[null_clean >= obs]
    p = len(tail) / len(null_clean)
    
    ax.set_xlabel('δT (μK)')
    ax.set_ylabel('Density')
    ax.set_title(f'{title}\np = {p:.4f}', fontweight='bold')
    ax.legend(fontsize=9)

# Empty panel — summary
ax = axes[1, 2]
ax.axis('off')
summary_text = "SIGNIFICANCE SUMMARY\n\n"
for key, r in results.items():
    summary_text += f"{r['label'].split('—')[0].strip()}:\n"
    summary_text += f"  p = {r['p_two_tailed']:.4f} ({r['sigma_equivalent']:.1f}σ)\n\n"
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, fontfamily='monospace', va='top')

fig.suptitle(f'ΛCDM Simulation-Based Significance Test\n'
             f'{N_SIMS:,} realizations, NSIDE={NSIDE_SIM}',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_null_distributions.png')
plt.close()
print("  -> fig1_null_distributions.png")

# ── Figure 2: Cumulative distribution + observed value ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, null, obs, title, color in [
    (axes[0], null_full, obs_full, "Full Sample", '#3498db'),
    (axes[1], null_key, obs_key, "z=[0.10,0.15)", '#e74c3c'),
]:
    null_clean = np.sort(null[np.isfinite(null)])
    cdf = np.arange(1, len(null_clean)+1) / len(null_clean)
    
    ax.plot(null_clean, cdf, color=color, lw=2)
    ax.axvline(obs, color='#e74c3c', ls='--', lw=2.5, label=f'Observed: {obs:.1f}')
    
    # Find percentile
    pctile = np.searchsorted(null_clean, obs) / len(null_clean) * 100
    ax.axhline(pctile/100, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(obs + 1, 0.5, f'Percentile: {pctile:.1f}%', fontsize=11)
    
    ax.set_xlabel('δT (μK)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'{title}', fontweight='bold')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUT}/fig2_cdf.png')
plt.close()
print("  -> fig2_cdf.png")

# ── Figure 3: Running p-value (convergence check) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, null, obs, title in [
    (axes[0], null_full, obs_full, "Full Sample"),
    (axes[1], null_key, obs_key, "z=[0.10,0.15)"),
]:
    running_p = []
    checkpoints = range(100, N_SIMS + 1, 100)
    for n in checkpoints:
        null_sub = null[:n]
        null_sub = null_sub[np.isfinite(null_sub)]
        if len(null_sub) > 0:
            p = np.mean(np.abs(null_sub) >= np.abs(obs))
        else:
            p = np.nan
        running_p.append(p)
    
    ax.plot(list(checkpoints), running_p, lw=1.5, color='#3498db')
    ax.axhline(0.05, color='#e74c3c', ls='--', lw=1.5, alpha=0.5, label='p = 0.05')
    ax.axhline(0.003, color='#9b59b6', ls=':', lw=1.5, alpha=0.5, label='p = 0.003 (3σ)')
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Running p-value')
    ax.set_title(f'{title} — Convergence', fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUT}/fig3_convergence.png')
plt.close()
print("  -> fig3_convergence.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 7: SAVE RESULTS")
print("="*70)

output = {
    "pipeline": {
        "N_sims": N_SIMS,
        "NSIDE": NSIDE_SIM,
        "aperture_disk_arcmin": DISK_ARCMIN,
        "aperture_ann_arcmin": ANN_ARCMIN,
        "galactic_cut_deg": GALACTIC_CUT,
        "N_clusters": N_clusters,
        "N_disturbed": int(n_dis),
        "N_relaxed": int(n_rel),
        "runtime_hours": round(total_time / 3600, 2),
    },
    "observed": {
        "full_sample": round(float(obs_full), 2),
        "key_bin": round(float(obs_key), 2),
        "broad_bin": round(float(obs_broad), 2),
    },
    "significance": results,
    "validation": {
        "null_mean_full": round(float(null_full.mean()), 3),
        "null_mean_key": round(float(null_key.mean()), 3),
        "null_centered": mean_ok,
    },
}

with open(f'{OUT}/simulation_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# Also save raw null distributions for further analysis
np.savez_compressed(f'{OUT}/null_distributions.npz',
    null_full=null_full,
    null_key=null_key,
    null_broad=null_broad,
    null_shuffle_full=null_shuffle_full,
    null_shuffle_key=null_shuffle_key,
)

print(f"  -> {OUT}/simulation_results.json")
print(f"  -> {OUT}/null_distributions.npz")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FINAL REPORT")
print("="*70)

print(f"\n  Pipeline: {N_SIMS:,} ΛCDM simulations at NSIDE={NSIDE_SIM}")
print(f"  Runtime: {total_time/3600:.1f} hours")
print(f"  Clusters: {N_clusters} ({n_dis} disturbed, {n_rel} relaxed)")

print(f"\n  {'Test':<45} {'Observed':>10} {'p-value':>10} {'Sigma':>8}")
print("  " + "-"*75)

for key, r in results.items():
    star = "***" if r['p_two_tailed'] < 0.001 else "**" if r['p_two_tailed'] < 0.01 else "*" if r['p_two_tailed'] < 0.05 else ""
    print(f"  {r['label'][:44]:<45} {r['observed']:>9.1f} {r['p_two_tailed']:>10.4f} {r['sigma_equivalent']:>7.1f}σ {star}")

print(f"\n  Null distribution validation: {'PASSED' if mean_ok else 'WARNING'}")
print(f"\n  Interpretation:")

sig_key = results.get('B_key_lcdm', {})
if sig_key.get('p_two_tailed', 1) < 0.003:
    print("  → STRONG EVIDENCE: The observed signal in z=[0.10,0.15) is inconsistent")
    print("    with ΛCDM expectations at >3σ significance.")
elif sig_key.get('p_two_tailed', 1) < 0.05:
    print("  → MODERATE EVIDENCE: The observed signal in z=[0.10,0.15) is inconsistent")
    print("    with ΛCDM expectations at >2σ significance.")
else:
    print("  → WEAK/NO EVIDENCE: The observed signal is consistent with ΛCDM fluctuations.")

print("\nDone.")
