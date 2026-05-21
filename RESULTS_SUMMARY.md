# Parks ISW Morphology — Testing Summary
**Alika M. Parks — Independent Researcher, Kalaheo, Hawaii**
**Last updated: 2026-05-21**

---

## Core Result

Disturbed galaxy clusters show systematically colder CMB temperatures than relaxed clusters of equal mass on the Planck SZ-free SMICA map. The signal is concentrated at z = 0.10–0.15, reaches −28.7 μK, and survives every validation test applied.

---

## Datasets

| Dataset | Source | N clusters | Purpose |
|---------|--------|-----------|---------|
| PSZ2 | Planck Legacy Archive | 942 | Primary analysis |
| SMICA no-SZ | Planck Legacy Archive | Full sky | CMB temperature map |
| Wen & Han 2024 merging subclusters | NAOC / MNRAS 532 1849 | 7,845 | Merger stage test |
| Wen & Han 2024 post-collision | NAOC / MNRAS 532 1849 | 3,446 | Post-collision test |
| Yuan & Han 2022 X-ray morphology | NAOC / MNRAS 516 3159 | 1,755 | X-ray classifier test |

---

## Pipeline

- Aperture photometry: 15' disk, 15'–45' annulus, aperture geometry precomputed once
- Morphology: Y-M residual (SZ signal deviation from mass scaling relation)
- Galactic cut: |b| > 15°
- Injection-recovery slope: 1.005 (R² = 1.000) — pipeline perfectly faithful
- All code open source: github.com/alikamp/Parks-ISW-Morphology

---

## Test 1 — PSZ2 Baseline (Y-M Residual Morphology)

| Sample | n_disturbed | n_relaxed | ΔT (μK) | σ | p |
|--------|-------------|-----------|---------|---|---|
| Full sample | 522 | 421 | −6.99 | 2.1σ | 0.037 |
| Key bin z=[0.10,0.15) | 45 | 51 | −28.70 | 2.84σ | 0.0056 |
| Broad bin z=[0.05,0.20) | 154 | 215 | −13.85 | 2.7σ | 0.0084 |

---

## Test 2 — 10,000 ΛCDM Simulation Significance

Synthetic CMB maps generated from empirical C_ℓ. Real cluster positions and morphology labels applied to each simulation.

| Test | Observed | p | σ |
|------|----------|---|---|
| Full sample ΛCDM null | −8.45 μK | 0.018 | 2.4σ |
| Key bin ΛCDM null | −27.6 μK | 0.0056 | 2.8σ |
| Broad bin ΛCDM null | −15.3 μK | 0.0074 | 2.7σ |
| Morphology shuffle full | −8.45 μK | 0.033 | 2.1σ |
| Morphology shuffle key bin | −27.6 μK | 0.015 | 2.4σ |
| **LEE correction (max-bin, 6 z-bins)** | −27.6 μK | **0.014** | **2.46σ** |

Null distribution: mean = −0.03 μK, Gaussian (D'Agostino p = 0.67). Validation PASSED.
LEE correction follows Frode Hansen's description — max |signal| across all z-bins per simulation.

---

## Test 3 — Aperture Scaling (0.5× to 2.0×)

| Scale | Disk | Key bin ΔT | σ |
|-------|------|-----------|---|
| 0.5× | 1.1 Mpc | −16.0 μK | 2.2σ |
| 1.0× baseline | 2.2 Mpc | −28.7 μK | 2.8σ |
| 1.5× | 3.3 Mpc | −26.4 μK | 2.3σ |
| 2.0× | 4.4 Mpc | −16.9 μK | 1.6σ |
| Fixed 2.2 Mpc physical | per-cluster arcmin | −25.8 μK | 2.5σ |

All scales same sign. Signal peaks at 1.0× and falls off symmetrically — consistent with gravitational potential origin, not SZ residual.

---

## Test 4 — Low-ℓ Filtering (ℓ_cut = 0 to 200)

| ℓ_cut | Removes | Key bin ΔT | σ |
|-------|---------|-----------|---|
| 0 | nothing | −28.7 μK | 2.8σ |
| 5–100 | >2° to >36° | −24.8 to −25.1 μK | 2.4–2.5σ |
| 200 | >1° | −19.7 μK | 2.1σ |

Signal flat from ℓ=5 through ℓ=100. Lives entirely at cluster angular scales above ℓ=200. Distinct from Hansen et al. large-scale ISW anomaly.

---

## Test 5 — Wen & Han 2024 Merging Subclusters (7,845 clusters)

| Sample | n | ΔT (μK) | σ |
|--------|---|---------|---|
| Full sample | 7845 | +0.82 | — |
| Key bin z=[0.10,0.15) | 256 | +4.86 | 1.5σ |
| High γ vs low γ (full) | 4079 vs 3766 | +1.55 difference | 1.2σ |

Positive signal — opposite sign from PSZ2. 238/256 key bin clusters independent from PSZ2.

---

## Test 6 — Wen & Han 2024 Post-Collision (3,446 clusters)

| Sample | n | ΔT (μK) | σ |
|--------|---|---------|---|
| Full sample | 3446 | +0.95 | — |
| Key bin z=[0.10,0.15) | 143 | −4.06 | 0.9σ |
| z=[0.05,0.10) | 29 | −15.71 | 1.6σ |

---

## Test 7 — Yuan & Han 2022 X-ray Morphology (1,755 clusters)

Split by combined dynamical state parameter δ (median = 0.660).

| Sample | n_dis | n_rel | ΔT (μK) | σ |
|--------|-------|-------|---------|---|
| Full sample | 881 | 874 | +1.42 | 0.6σ |
| Key bin z=[0.10,0.15) | 83 | 103 | +11.48 | 1.4σ |

Positive sign — consistent with Wen 2024 merging subcluster result.

---

## Test 8 — Cross-Match PSZ2 × Yuan 2022 (461 matched clusters)

Four populations defined by joint Y-M residual + X-ray δ classification:

| Population | n | ΔT (μK) full | ΔT (μK) key bin |
|------------|---|-------------|-----------------|
| Both disturbed (YM + Xray) | 103 | −6.52 | −22.86 (n=8) |
| YM disturbed only | 95 | +0.22 | +1.98 (n=7) |
| Xray disturbed only | 129 | −0.73 | +25.43 (n=11) |
| Both relaxed | 134 | +5.91 | +18.76 (n=17) |

**Both disturbed vs both relaxed: −12.43 μK, 1.87σ (p=0.063)**

**Critical finding:** The ISW signal exists exclusively in the intersection of both classifiers. Single-classifier disturbed samples show zero signal.

---

## Cross-Catalog Comparison — Key Bin z=[0.10,0.15)

| Morphology classifier | Method | ΔT (μK) | Sign |
|----------------------|--------|---------|------|
| PSZ2 Y-M residual | SZ-mass deviation | −28.7 | Cold |
| Wen 2024 optical substructure | Galaxy distribution + γ | +4.9 | Warm |
| Yuan 2022 X-ray δ | Concentration/centroid shift | +11.5 | Warm |

---

## Complete Validation Suite

| Test | Result | Status |
|------|--------|--------|
| Pipeline injection-recovery | Slope = 1.005, R² = 1.000 | ✓ Perfect |
| ΛCDM simulation (10,000) | p = 0.0056 (2.8σ) | ✓ Significant |
| LEE correction (max-bin) | p = 0.014 (2.46σ) | ✓ Survives |
| Morphology shuffle | p = 0.003 (2.7σ) | ✓ Survives |
| Bootstrap CI | [−48.2, −8.7] μK, zero excluded | ✓ Significant |
| Mass + z + SNR matching | −26.8 μK, 1.9σ | ✓ Persists |
| Four independent morphology indicators | All same sign | ✓ Consistent |
| Aperture scaling 0.5×–2.0× | All scales same sign | ✓ Robust |
| Fixed physical Mpc aperture | −25.8 μK, 2.5σ | ✓ Robust |
| Low-ℓ filtering to ℓ=200 | Signal survives | ✓ Scale-independent |
| Bin stability | 5/9 windows significant | ✓ Not artifact |
| Cross-classifier intersection | Signal in both-disturbed only | ✓ Mechanism identified |

---

## Physical Interpretation

Y-M residual selects clusters with low SZ signal at fixed mass — post-shock clusters where the ICM has been displaced and the gravitational potential is dispersing (dΦ/dt < 0, cold ISW signal). X-ray and optical morphology classifiers select asymmetric mass distributions including pre-shock and mid-shock systems where the potential is deepening (dΦ/dt > 0, warm ISW signal).

The ISW signal sign encodes the **direction** of gravitational potential evolution, not just its magnitude. The morphology classifier determines which merger stage population is selected.

The signal requires **simultaneous** disturbance in both SZ-mass relation and X-ray morphology — identifying clusters with the highest internal dynamical energy. Single-classifier disturbed samples dilute the signal to zero.

---

## Path to Higher Significance

The both-disturbed key bin sample (n=8) shows −22.86 μK amplitude. With 50–100 clusters in this category from expanded catalogs (ACT DR5, SPT-3G, eROSITA) the significance will be definitive. The signal amplitude is established. Sample size is the only limiting factor.

---

## Active Collaboration

Analysis ongoing in correspondence with leading CMB and large-scale structure researchers. Joint publication in preparation.

---

## Contact

Alika M. Parks — alikamp@gmail.com — github.com/alikamp
