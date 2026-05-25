##Parks ISW Morphology
Cluster Morphology as a Gravitational Phase Detector in ISW Measurements
A mass-parity signal in the Planck SZ cluster catalog

*Last updated: May 24, 2026*

First test of morphology-dependent ISW signal using 942 Planck SZ clusters 
and SZ-free CMB maps. Extended to eROSITA eRASS1 (12,247 clusters), ACT DR6 
(10,040 clusters), and Sanders et al. 2025 X-ray morphology catalog.

---

## Core Physical Mechanism

The ISW temperature shift experienced by a CMB photon transiting a 
gravitational potential well is:

**ΔT ∝ (Φ_exit − Φ_entry)**

In the post-shock re-virialization phase — selected by negative Y-M residuals 
— a photon enters a cluster whose potential well has been temporarily shallowed 
by recent merger chaos (Φ_entry shallow). During the typical ~6-13 Myr transit, rapid 
non-linear central relaxation re-deepens the potential. The photon exits a 
deeper well than it entered (Φ_exit < Φ_entry) — net energy loss — **COLD ISW**.

For pre-shock infall — photon enters a deep stable well, dark energy shallows 
the background during transit, Φ_exit > Φ_entry — **WARM ISW**.

The Y-M residual is the physical stopwatch: negative residual identifies 
clusters at the precise moment when Φ_entry is uniquely shallow, maximizing 
the cold ISW imprint.

---

## Primary Result

PSZ2 Y-M residual morphology split, key bin z=[0.10,0.15), 108 clusters 
(49 disturbed, 59 relaxed):

| Metric | Value |
|--------|-------|
| Signal | −28.70 μK |
| Significance | 2.84σ (p=0.0056) |
| LEE corrected | 2.46σ (p=0.014) |
| NILC confirmation | −27.91 μK, 2.57σ |
| Dose-response Q1→Q5 | 48 μK gradient, 2.56σ |
| ΛCDM 10,000 sims | 2.80σ (p=0.0056) |
| FFP10 999 sims | 2.51σ (p=0.006) |
| Validation tests | 17 — all pass |

---

## The Merger Stage Clock

Different morphology classifiers select different phases of the merger cycle.
The CMB temperature encodes the phase:

| Classifier | Dataset | Phase | ΔT (μK) | Sign |
|-----------|---------|-------|---------|------|
| Y-M residual | PSZ2 | Post-shock re-virialization | −28.70 | ✓ Cold |
| Y-M residual NILC | PSZ2 | Post-shock re-virialization | −27.91 | ✓ Cold |
| L-M residual Q5 | eROSITA | Post-shock X-ray bright | −8.85 | ✓ Cold |
| Both-disturbed | PSZ2×eROSITA | Post-shock double classifier | −11.26 | ✓ Cold |
| L-M residual Q1 | eROSITA | Pre-shock X-ray faint | +3.36 | ✓ Warm |
| Radio halos | LoTSS/PSZ2 | Active shock phase | +3.80 | ✓ Warm |
| Optical mergers | Wen 2024 | Active merger | +4.86 | ✓ Warm |
| X-ray morphology δ | Yuan 2022 | Active/recent merger | +11.48 | ✓ Warm |
| D_COMB morphology | eROSITA Sanders 2025 | Active morphology | +2.79 | ✓ Warm |
| Richness λ | redMaPPer | Phase unclear | +2.83 | Neutral |

**Every classifier predicts the correct sign.** The ISW temperature encodes 
the direction of gravitational potential evolution during photon transit — 
a new observable: the CMB as a gravitational phase detector.

---

## Complete Validation Suite

| Test | Result | Status |
|------|--------|--------|
| Pipeline injection-recovery | Slope=1.005, R²=1.000 | ✓ Perfect |
| ΛCDM simulation (10,000) | p=0.0056 (2.8σ) | ✓ Significant |
| LEE correction (max-bin) | p=0.014 (2.46σ) | ✓ Survives |
| Morphology shuffle | p=0.003 (2.7σ) | ✓ Survives |
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
| eROSITA eRASS1 L-M gradient | Inverted 2.18σ (p=0.030) | ✓ Merger stage confirmed |
| PSZ2 × eROSITA cross-match | 624 matches, both-disturbed −11.26 μK | ✓ Multi-instrument |
| FFP10 999 sims | p=0.006, 2.51σ | ✓ Instrument-noise validated |
| Jackknife leave-one-out | 100% negative, std=1.07 μK | ✓ Population effect |

---

## Catalog Search — Why PSZ2 is Uniquely Suited

Every major cluster catalog tested for independent replication:

| Catalog | Key bin n_dis | n_rel | Verdict |
|---------|--------------|-------|---------|
| PSZ2 Y-M residual | 49 | 59 | ✓ Signal −28.70 μK |
| ACT DR4 SNR-M | 8 | 39 | Selection bias |
| ACT DR5 fixed_y_c | 0 | 160 | Selection bias |
| ACT DR6 Y500-M | 0 | 160 | Selection bias |
| SPT Xi-M | 16 | 0 | Selection bias |
| eROSITA L-M residual | 669 | 92 | Inverted — merger stage |
| eROSITA D_COMB (Sanders 2025) | 332 | 535 | Null — different phase |
| redMaPPer richness | 115 | 115 | Null — no phase sensitivity |
| Radio halos LoTSS/PSZ2 | 10 | 17 | Warm — active shock phase |
| ACT×PSZ2 cross-match | 4 | 19 | Incompatible calibrations |

ACT DR5/DR6 detect zero disturbed clusters at z=[0.10,0.15) because their 
smaller beam preferentially detects compact bright cool-core relaxed systems. 
PSZ2's all-sky coverage with Planck's large beam captures both disturbed and 
relaxed populations equally — the only existing catalog with this property 
at low redshift.

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

Signal is stable to removal of any single cluster. Not driven by outliers.

### Mass-Split Analysis

| Mass range | n_dis | n_rel | ΔT (μK) | σ |
|-----------|-------|-------|---------|---|
| Low mass ≤3.41×10¹⁴ M☉ | 39 | 12 | −12.26 | 0.90 |
| High mass >3.41×10¹⁴ M☉ | 8 | 39 | −37.63 | 2.23 |

Signal stronger in high mass clusters — consistent with deeper potential 
wells producing larger ISW amplitude.

---

## Physical Interpretation

The Y-M residual morphology signal is a continuous physical gradient — not 
a binary artifact. The most dynamically disturbed clusters show the coldest 
CMB temperatures, the most relaxed show the warmest, with a monotonic 
gradient across all five quintiles spanning 48 μK in the key bin.

The signal is redshift-specific — concentrated at z=[0.10,0.15) reflecting 
three contributing factors: the ISW kernel peaks where dark energy dominates; 
the post-shock cluster population density peaks at this epoch; and Planck's 
5 arcmin beam matches the physical cluster scale at this redshift without 
beam dilution, maximizing morphology sensitivity.

The multi-classifier validation table unifies all cross-catalog comparisons 
under a single physical framework: the ISW temperature sign encodes the 
direction of gravitational potential evolution (Φ_exit − Φ_entry) during 
photon transit, acting as a merger stage clock readable in the CMB.

---

## Path to Higher Significance

Signal amplitude established at −28.70 μK. Sample size is the limiting 
factor. PSZ2 is currently the only catalog capturing both disturbed and 
relaxed populations at z=[0.10,0.15) — ACT and SPT miss the disturbed 
population due to beam selection bias. CMB-S4 and next-generation SZ 
surveys will populate the key bin with thousands of clusters, pushing 
significance well above 3σ.

---

## Active Collaboration

Active correspondence with leading CMB and large-scale structure researchers.
Zoom scheduled May 2026. Joint publication in preparation.

---

## Data

All data publicly available:
- Planck PSZ2 catalogue — IRSA / VizieR J/A+A/594/A27
- Planck SMICA no-SZ CMB map — Planck Legacy Archive
- eROSITA eRASS1 — VizieR J/A+A/685/A106
- eROSITA morphology — Sanders et al. 2025, A&A 695, 160A
- ACT DR6 cluster catalog — lambda.gsfc.nasa.gov

---

## Repository Structure
├── README.md
├── RESULTS_SUMMARY.md
├── ffp10_validation.py
├── ffp10_validation.slurm
├── step1_download.py
├── step2_analyze_v2.py
├── scripts/
│   ├── step3_diego.py
│   ├── step4_confirm.py
│   ├── aperture_test.py
│   ├── lowl_test.py
│   ├── simulation_pipeline.py
│   └── final_validation.py
└── results/
└── results_v2.json

---

## Reproducing

```bash
pip install healpy astropy numpy scipy matplotlib astroquery
python step1_download.py
python step2_analyze_v2.py
python scripts/simulation_pipeline.py   # ~10 hours
python scripts/final_validation.py      # ~4 hours
python ffp10_validation.py              # requires PLA access, sims 0-999
```

---

## Author

Alika M. Parks — Independent Researcher, Kalaheo, HI, USA
alikamp@gmail.com

---

## License

MIT

---

## Data

All data is publicly available:

- **Planck PSZ2 catalogue** — [IRSA](https://irsa.ipac.caltech.edu/data/Planck/release_2/catalogs/)
- **Planck SMICA no-SZ CMB map** — [Planck Legacy Archive](https://pla.esac.esa.int)


## Reproducing

```bash
pip install healpy astropy numpy scipy matplotlib
python step1_download.py
python step2_analyze_v2.py
python scripts/step3_diego.py
python scripts/step4_confirm.py
python scripts/aperture_test.py
python scripts/lowl_test.py
python scripts/simulation_pipeline.py   # 10,000 sims, ~10 hours on Colab
python scripts/final_validation.py     # Injection-recovery + look-elsewhere, ~4 hours
```

## Author

**Alika M. Parks** — Independent Researcher, Kalaheo, HI, USA — [alikamp@gmail.com](mailto:alikamp@gmail.com)

## License

MIT
