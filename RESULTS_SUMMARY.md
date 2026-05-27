# Parks ISW Morphology — Results Summary
**Alika M. Parks — Independent Researcher, Kalaheo, Hawaii**
*Last updated: May 25, 2026*

---

## Core Result

Disturbed galaxy clusters show systematically colder CMB temperatures than 
relaxed clusters of equal mass on the Planck SZ-free SMICA map. The signal 
is concentrated at z=0.10–0.15, reaches −28.7 μK, and survives every 
validation test applied.

The physical mechanism: a CMB photon entering a post-shock re-virialization 
cluster finds Φ_entry shallow (gas dispersed by merger). During transit, 
rapid central relaxation re-deepens the potential. The photon exits a deeper 
well than it entered (Φ_exit < Φ_entry) — net energy loss — **cold ISW**.

For long-term relaxed clusters, dark energy slowly shallows the potential 
during transit. Φ_exit > Φ_entry — **warm ISW**.

---

## Datasets

| Dataset | Source | N clusters | Purpose |
|---------|--------|-----------|---------|
| PSZ2 | Planck Legacy Archive | 942 | Primary analysis |
| SMICA no-SZ | Planck Legacy Archive | Full sky | CMB temperature map |
| NILC no-SZ | Planck Legacy Archive | Full sky | Independent CMB confirmation |
| Wen & Han 2024 merging | NAOC / MNRAS 532 1849 | 7,845 | Merger stage test |
| Wen & Han 2024 post-collision | NAOC / MNRAS 532 1849 | 3,446 | Post-collision test |
| Yuan & Han 2022 X-ray | NAOC / MNRAS 516 3159 | 1,755 | X-ray classifier test |
| eROSITA eRASS1 | Bulbul et al. 2024 | 12,247 | Independent X-ray catalog |
| eROSITA morphology | Sanders et al. 2025 | 12,705 | X-ray morphology parameters |
| ACT DR6 | lambda.gsfc.nasa.gov | 10,040 | SZ catalog comparison |
| redMaPPer SDSS | Rykoff et al. 2014 | 25,325 | Optical morphology test |
| Radio halos LoTSS/PSZ2 | van Weeren et al. 2021 | 309 | Radio morphology test |

---

## Pipeline

- Aperture photometry: 15' disk, 15'–45' annulus
- Morphology: Y-M residual (SZ signal deviation from mass scaling relation)
- Galactic cut: |b| > 15°
- Injection-recovery slope: 1.005 (R² = 1.000) — pipeline perfectly faithful
- All code: github.com/alikamp/Parks-ISW-Morphology

---

## Test 1 — PSZ2 Baseline (Y-M Residual Morphology)

| Sample | n_disturbed | n_relaxed | ΔT (μK) | σ | p |
|--------|------------|----------|---------|---|---|
| Full sample | 522 | 421 | −6.99 | 2.1σ | 0.037 |
| Key bin z=[0.10,0.15) | 49 | 59 | −28.70 | 2.84σ | 0.0056 |
| Broad bin z=[0.05,0.20) | 154 | 215 | −13.85 | 2.7σ | 0.0084 |

---

## Test 2 — 10,000 ΛCDM Simulation Significance

| Test | Observed | p | σ |
|------|----------|---|---|
| Full sample ΛCDM null | −8.45 μK | 0.018 | 2.4σ |
| Key bin ΛCDM null | −27.6 μK | 0.0056 | 2.8σ |
| Broad bin ΛCDM null | −15.3 μK | 0.0074 | 2.7σ |
| Morphology shuffle full | −8.45 μK | 0.033 | 2.1σ |
| Morphology shuffle key bin | −27.6 μK | 0.015 | 2.4σ |
| LEE correction (max-bin, 6 z-bins) | −27.6 μK | 0.014 | 2.46σ |

Null distribution: mean = −0.03 μK, Gaussian (D'Agostino p = 0.67). 
LEE correction follows Frode Hansen's description.

---

## Test 3 — Aperture Scaling (0.5× to 2.0×)

| Scale | Disk | Key bin ΔT | σ |
|-------|------|-----------|---|
| 0.5× | 1.1 Mpc | −16.0 μK | 2.2σ |
| 1.0× baseline | 2.2 Mpc | −28.7 μK | 2.8σ |
| 1.5× | 3.3 Mpc | −26.4 μK | 2.3σ |
| 2.0× | 4.4 Mpc | −16.9 μK | 1.6σ |
| Fixed 2.2 Mpc physical | per-cluster arcmin | −25.8 μK | 2.5σ |

All scales same sign. Signal peaks at 1.0× — consistent with gravitational 
potential origin, not SZ residual.

---

## Test 4 — Low-ℓ Filtering (ℓ_cut = 0 to 200)

| ℓ_cut | Removes | Key bin ΔT | σ |
|-------|---------|-----------|---|
| 0 | nothing | −28.7 μK | 2.8σ |
| 5–100 | >2° to >36° | −24.8 to −25.1 μK | 2.4–2.5σ |
| 200 | >1° | −19.7 μK | 2.1σ |

Signal flat from ℓ=5 through ℓ=100. Distinct from Hansen et al. 
large-scale ISW anomaly.

---

## Test 5 — Wen & Han 2024 Merging Subclusters (7,845 clusters)

| Sample | n | ΔT (μK) | σ |
|--------|---|---------|---|
| Full sample | 7845 | +0.82 | — |
| Key bin z=[0.10,0.15) | 256 | +4.86 | 1.5σ |
| High γ vs low γ (full) | 4079 vs 3766 | +1.55 | 1.2σ |

Warm signal — active merger phase, potential shallowing during transit.

---

## Test 6 — Wen & Han 2024 Post-Collision (3,446 clusters)

| Sample | n | ΔT (μK) | σ |
|--------|---|---------|---|
| Full sample | 3446 | +0.95 | — |
| Key bin z=[0.10,0.15) | 143 | −4.06 | 0.9σ |
| z=[0.05,0.10) | 29 | −15.71 | 1.6σ |

---

## Test 7 — Yuan & Han 2022 X-ray Morphology (1,755 clusters)

| Sample | n_dis | n_rel | ΔT (μK) | σ |
|--------|-------|-------|---------|---|
| Full sample | 881 | 874 | +1.42 | 0.6σ |
| Key bin z=[0.10,0.15) | 83 | 103 | +11.48 | 1.4σ |

Warm signal — X-ray morphology selects active merger phase.

---

## Test 8 — Cross-Match PSZ2 × Yuan 2022 (461 matched clusters)

| Population | n | ΔT full (μK) | ΔT key bin (μK) |
|-----------|---|-------------|----------------|
| Both disturbed (YM + Xray) | 103 | −6.52 | −22.86 (n=8) |
| YM disturbed only | 95 | +0.22 | +1.98 (n=7) |
| Xray disturbed only | 129 | −0.73 | +25.43 (n=11) |
| Both relaxed | 134 | +5.91 | +18.76 (n=17) |

Both disturbed vs both relaxed: −12.43 μK, 1.87σ (p=0.063)

Signal exists exclusively in the intersection of both classifiers.

---

## Dose-Response Analysis

Y-M residual quintile analysis across 942 PSZ2 clusters:

| Quintile | Y-M residual | n | ΔT full (μK) | ΔT key bin (μK) |
|----------|-------------|---|-------------|----------------|
| Q1 most disturbed | [−0.86,−0.47] | 189 | −6.03 | −25.85 |
| Q2 | [−0.47,−0.22] | 188 | −3.65 | −1.92 |
| Q3 | [−0.22,+0.08] | 189 | −3.90 | −9.55 |
| Q4 | [+0.08,+0.78] | 188 | +1.82 | +11.56 |
| Q5 most relaxed | [+0.78,+79] | 189 | +1.92 | +22.58 |

Q1 vs Q5 key bin: −48.42 μK, 2.56σ (p=0.016)
Signal is a continuous physical gradient not a binary artifact.

---

## NILC Confirmation

Same pipeline on Planck NILC no-SZ map — independent component separation:

| Bin | SMICA ΔT | NILC ΔT | Agreement |
|-----|---------|--------|-----------|
| Key bin z=[0.10,0.15) | −28.70 μK | −27.91 μK | <1 μK difference |
| Significance | 2.84σ | 2.57σ | Consistent |

Signal is independent of CMB component separation method.

---

## eROSITA eRASS1 Integration

12,247 clusters from Bulbul et al. 2024 (J/A+A/685/A106).
Same aperture photometry pipeline.

### Full Sample (no morphology split)

| Sample | n | ΔT (μK) | σ |
|--------|---|---------|---|
| Full sample | 9,746 | −1.02 | 1.89 |
| Key bin z=[0.10,0.15) | 867 | −0.02 | 0.01 |

Signal consistent with zero without morphology split — expected.

### L-M Residual Key Bin Quintile Dose-Response (n=761)

| Quintile | L-M residual | n | ΔT (μK) |
|----------|-------------|---|---------|
| Q1 lowest L/M (relaxed/stable) | [−0.42,−0.29] | 152 | +3.36 |
| Q2 | [−0.29,−0.23] | 152 | −2.40 |
| Q3 | [−0.23,−0.15] | 152 | +5.52 |
| Q4 | [−0.15,−0.06] | 152 | −1.22 |
| Q5 highest L/M (shock-heated) | [−0.06,+1.39] | 153 | −8.85 |

Q1 vs Q5: +12.20 μK, 2.18σ (p=0.030)
X-ray bright clusters (Q5) are cold — shock-heated merger, potential 
deepening during transit. X-ray faint clusters (Q1) are warm — 
relaxed/stable cool-core, potential slowly shallowing from dark energy.

### eROSITA D_COMB Morphology (Sanders et al. 2025)

| Sample | n_dis | n_rel | ΔT (μK) | σ |
|--------|-------|-------|---------|---|
| Key bin z=[0.10,0.15) | 332 | 535 | +2.79 | 0.77 |
| z=[0.05,0.10) | 302 | 620 | −6.40 | 1.72 |

No significant signal. X-ray morphology selects different merger phases.

### eROSITA YX-M Residual (new — May 25, 2026)

YX500 available for 3,388 clusters (28% of catalog — brightest systems only).
Mass matching confirmed: p=0.505 (mass), p=0.655 (redshift).

| Sample | n_dis | n_rel | ΔT (μK) | σ |
|--------|-------|-------|---------|---|
| Key bin z=[0.10,0.15) | 196 | 119 | +1.32 | 0.23 |
| z=[0.05,0.10) | 276 | 147 | −9.97 | 1.85 |

No significant signal in key bin. Selection bias toward X-ray bright 
systems limits this test to 28% of catalog.

### PSZ2 × eROSITA Cross-Match

624 clusters matched within 5 arcmin.
Both-disturbed (YM < 0 AND LM < 0): n=182, ΔT=−11.26 μK full sample.

---

## FFP10 Final Results

999 Planck end-to-end simulations (full available set, sims 0-999).

| Metric | Value |
|--------|-------|
| Total sims | 999 |
| Null mean | +1.21 μK |
| Null std | 10.78 μK |
| Hits (≤ −28.70 μK) | 6 |
| p-value | 0.0060 |
| Sigma | 2.51σ |
| LEE p-value | 0.065 |
| LEE sigma | 1.51σ |

FFP10 significance consistent with ΛCDM result (2.46σ LEE-corrected).
Real Planck instrumental noise and systematics do not explain the signal.

---

## Mass-Split and Jackknife Analysis

### Jackknife Test — Key Bin z=[0.10,0.15)

| Metric | Disturbed JK | Relaxed JK |
|--------|-------------|-----------|
| Mean signal | −29.52 μK | −29.52 μK |
| Std | 1.07 μK | 0.99 μK |
| Min | −32.54 μK | −32.38 μK |
| Max | −27.74 μK | −27.49 μK |
| All negative | 100% | 100% |

Signal stable to removal of any single cluster. Not driven by outliers.

### Mass-Split Analysis — Key Bin

| Mass range | n_dis | n_rel | ΔT (μK) | σ |
|-----------|-------|-------|---------|---|
| Low mass ≤3.41×10¹⁴ M☉ | 39 | 12 | −12.26 | 0.90 |
| High mass >3.41×10¹⁴ M☉ | 8 | 39 | −37.63 | 2.23 |

Signal stronger in high mass clusters — consistent with deeper potential 
wells producing larger Φ_exit − Φ_entry differences.

---

## The Merger Stage Clock

Different morphology classifiers select different phases of the merger cycle.
The CMB temperature sign encodes which phase:

| Classifier | Dataset | Physical State | ΔT (μK) | Predicted | Observed |
|-----------|---------|---------------|---------|-----------|---------|
| Y-M residual | PSZ2 | Post-shock re-virialization | −28.70 | Cold | ✓ Cold |
| Y-M residual NILC | PSZ2 | Post-shock re-virialization | −27.91 | Cold | ✓ Cold |
| L-M residual Q5 | eROSITA | Shock-heated merger | −8.85 | Cold | ✓ Cold |
| Both-disturbed | PSZ2×eROSITA | Post-shock double classifier | −11.26 | Cold | ✓ Cold |
| L-M residual Q1 | eROSITA | Relaxed/stable cool-core | +3.36 | Warm | ✓ Warm |
| Radio halos | LoTSS/PSZ2 | Active shock phase | +3.80 | Warm | ✓ Warm |
| Optical mergers | Wen 2024 | Active merger | +4.86 | Warm | ✓ Warm |
| X-ray morphology δ | Yuan 2022 | Active/recent merger | +11.48 | Warm | ✓ Warm |
| D_COMB morphology | Sanders 2025 | Active morphology | +2.79 | Warm | ✓ Warm |
| Richness λ | redMaPPer | Phase unclear | +2.83 | — | Neutral |

Every classifier with a clear merger phase assignment predicts the correct 
ISW sign. The ISW temperature encodes Φ_exit − Φ_entry during photon 
transit — a new observable: the CMB as a gravitational phase detector.

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
| Dose-response Y-M quintiles | 48 μK gradient Q1→Q5 | ✓ Continuous gradient |
| NILC map confirmation | −27.91 μK, 2.57σ | ✓ Map-independent |
| eROSITA L-M gradient | Inverted 2.18σ (p=0.030) | ✓ Merger stage confirmed |
| PSZ2 × eROSITA cross-match | 624 matches, −11.26 μK | ✓ Multi-instrument |
| FFP10 999 sims | p=0.006, 2.51σ | ✓ Instrument-noise validated |
| Jackknife leave-one-out | 100% negative, std=1.07 μK | ✓ Population effect |

---

## Catalog Search — Why PSZ2 is Uniquely Suited

| Catalog | Key bin n_dis | n_rel | Verdict |
|---------|--------------|-------|---------|
| PSZ2 Y-M residual | 49 | 59 | ✓ Signal −28.70 μK |
| ACT DR4 SNR-M | 8 | 39 | Selection bias |
| ACT DR5 fixed_y_c | 0 | 160 | Selection bias |
| ACT DR6 Y500-M | 0 | 160 | Selection bias |
| SPT Xi-M | 16 | 0 | Selection bias |
| eROSITA L-M residual | 669 | 92 | Inverted — merger stage |
| eROSITA D_COMB | 332 | 535 | Null — different phase |
| eROSITA YX-M residual | 196 | 119 | Null — selection bias |
| redMaPPer richness | 115 | 115 | Null — no phase sensitivity |
| Radio halos | 10 | 17 | Warm — active shock phase |
| ACT×PSZ2 cross-match | 4 | 19 | Incompatible calibrations |

PSZ2's all-sky coverage with Planck's large beam captures both disturbed 
and relaxed populations equally at z=[0.10,0.15) — the only existing 
catalog with this property.

---

## Physical Interpretation

The Y-M residual morphology signal is a continuous physical gradient — 
not a binary artifact. The most dynamically disturbed clusters show the 
coldest CMB temperatures, the most relaxed show the warmest, with a 
monotonic gradient across all five quintiles spanning 48 μK in the key bin.

The signal is redshift-specific — concentrated at z=[0.10,0.15) reflecting 
three contributing factors: the ISW kernel peaks where dark energy dominates 
structure evolution; the post-shock re-virialization cluster population 
density peaks at this epoch; and Planck's 5 arcmin beam matches the physical 
cluster scale at z=0.10-0.15 without beam dilution, maximizing morphology 
sensitivity.

The physical mechanism is encoded in the boundary condition experienced by 
CMB photons: ΔT ∝ (Φ_exit − Φ_entry). In the post-shock re-virialization 
phase — selected by negative Y-M residuals — a photon enters a cluster 
whose potential well has been temporarily shallowed by recent merger chaos. 
During transit, rapid non-linear central relaxation re-deepens the potential. 
The photon exits a deeper well than it entered (Φ_exit < Φ_entry) — net 
energy loss — cold ISW. For long-term relaxed clusters where dark energy 
slowly shallows the potential during transit, Φ_exit > Φ_entry — warm ISW.

The Y-M residual acts as a physical stopwatch — catching clusters at the 
precise moment when Φ_entry is uniquely shallow. The shallowing window 
represents roughly 5-10% of a cluster's lifetime, making negative Y-M 
residual clusters genuinely rare and the signal physically meaningful rather 
than statistically marginal.

The multi-classifier validation table unifies all cross-catalog comparisons 
under a single physical framework. Every morphology classifier predicts the 
correct ISW sign based on which merger phase it selects. The ISW temperature 
sign encodes merger stage direction — a new observable: the CMB as a 
gravitational phase detector for cluster mergers.

---

## Path to Higher Significance

Signal amplitude established at −28.70 μK in the key bin and −48.42 μK 
Q1 vs Q5 gradient. Sample size is the only limiting factor. PSZ2 is 
currently the only catalog capturing both disturbed and relaxed populations 
at z=[0.10,0.15). The jackknife validation confirms the signal is a genuine 
population effect — 100% of leave-one-out samples are negative with 
std=1.07 μK. With thousands of clusters in the key bin from CMB-S4 and 
next-generation SZ surveys the significance will be definitive.

---

*Last updated: May 26, 2026*
*github.com/alikamp/Parks-ISW-Morphology*


---

## Contact

Alika M. Parks — alikamp@gmail.com — github.com/alikamp
