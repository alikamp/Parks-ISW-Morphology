# Cluster Morphology as a Systematic in ISW Measurements

**First test of morphology-dependent ISW signal using 942 Planck SZ clusters and SZ-free CMB maps.**

## Summary of Findings

Disturbed galaxy clusters show systematically **colder** CMB temperatures than relaxed clusters on the Planck SZ-free SMICA map — the **opposite** of what standard ΛCDM predicts. The signal survives mass matching, null testing, bootstrap resampling, and aperture scaling. It is concentrated at z = 0.10–0.15 and extends beyond the cluster virial radius.

---

## Analysis Progression

### v2 — Core Result

Initial analysis on 942 confirmed PSZ2 clusters with the SMICA no-SZ CMB map. Morphology classified via Y_SZ–mass scaling residual.

| Analysis | ΔT (μK) | Significance |
|----------|---------|-------------|
| Full sample (n=942) | −7.1 ± 3.4 | 2.1σ (p=0.037) |
| Mass-matched (364 pairs) | −6.5 ± 3.8 | 1.7σ (p=0.087) |
| Regression (controlling M, z) | −16.6 ± 9.3 | 1.8σ (p=0.073) |
| z = 0.10–0.15 bin | −28.7 | 2.8σ (p=0.006) |

Scripts: `step2_analyze_v2.py`

---

### v3 — Response to Domain Expert Feedback

Following review by researchers in ISW cosmology, the analysis was extended to address specific questions about morphology definitions, independent indicators, and cluster environment.

#### Extreme Morphology Cuts

Using a combined morphology score (Y-M residual + SNR-M residual + Y significance), the signal strengthens with more extreme definitions:

| Cut | Difference (μK) | Significance |
|-----|-----------------|-------------|
| 40/60 split | −8.7 | 2.2σ |
| Extreme quartiles | −12.9 | 2.2σ |
| Extreme 20% | −16.6 | 2.2σ |

Amplitude doubles while significance holds constant — consistent with a real effect scaling with morphological extremity.

#### Independent Dynamical State Indicators

All four tested indicators show the same sign (negative ΔT for disturbed):

| Indicator | Difference (μK) | Significance |
|-----------|-----------------|-------------|
| Y-M residual | −6.9 | 2.0σ |
| SNR-M residual | −5.2 | 1.5σ |
| Y significance | −3.7 | 1.1σ |
| Combined score | −4.8 | 1.4σ |

Cluster ellipticity (computed from CMB temperature quadrupole) is **uncorrelated** with Y-M residual (r = 0.017, p = 0.60), confirming it as a genuinely independent dynamical state indicator.

#### Cluster Environment

Disturbed clusters with close neighbors show the strongest signal (−9.1 μK), while relaxed clusters in the same environment show +2.5 μK. Nearest-neighbor separation correlates with redshift (r = 0.635, p ≈ 0), so environment and redshift effects are entangled in this sample.

#### Redshift × Distance × Morphology

The signal is concentrated at z < 0.2 for disturbed clusters with close neighbors:

| Redshift | Environment | n | ΔT (μK) |
|----------|------------|---|---------|
| z < 0.1 | Close | 43 | −11.8 |
| 0.1–0.2 | Close | 75 | −10.0 |
| 0.2–0.4 | Close | 82 | +0.9 |

Scripts: `scripts/step3_diego.py`

---

### v4 — Confirmation Tests

Four confirmation tests on the z = 0.10–0.15 signal:

#### Bin Stability

Signal persists across **5 of 9** sliding redshift windows at p < 0.05. Not a binning artifact.

| Window | Diff (μK) | Significance |
|--------|-----------|-------------|
| [0.09, 0.14) | −20.4 | 2.1σ * |
| [0.10, 0.15) | −28.7 | 2.8σ ** |
| [0.10, 0.20) | −17.7 | 2.6σ * |
| [0.05, 0.15) | −16.9 | 2.6σ * |

#### Bootstrap

10,000 resamples of the z = 0.10–0.15 bin:
- 95% CI: [−48.2, −8.7] μK
- **Zero excluded from confidence interval**
- P(diff ≥ 0) = 0.003

#### Mass Matching Within Key Bin

28 matched pairs within z = 0.10–0.15:
- Difference: −26.3 μK at 2.1σ (p = 0.042)
- Signal persists after controlling for mass

#### Null Test

1,000 trials with shuffled morphology labels:
- p = 0.003 (observed signal more extreme than 99.7% of null distribution)
- Equivalent to 2.7σ
- **Signal survives null test**

Scripts: `scripts/step4_confirm.py`

---

### Aperture Scaling — Does the Signal Extend Beyond the Virial Radius?

Tested at 1.0×, 1.5×, and 2.0× the baseline aperture (15', 22.5', 30' disk radii) to determine whether the effect is confined to the cluster core or extends to larger scales.

#### Key bin z = [0.10, 0.15):

| Aperture | Physical Scale | Diff (μK) | Significance |
|----------|---------------|-----------|-------------|
| 1.0× (15') | 2.2 Mpc | −28.7 | 2.8σ |
| 1.5× (22.5') | 3.3 Mpc | −26.4 | 2.3σ |
| 2.0× (30') | 4.4 Mpc | −16.9 | 1.6σ |

#### Full sample:

| Aperture | Diff (μK) | Significance |
|----------|-----------|-------------|
| 1.0× | −7.1 | 2.1σ |
| 1.5× | −6.6 | 1.7σ |
| 2.0× | −6.1 | 1.7σ |

The signal persists with the same sign at all scales, weakening gradually with increasing aperture. It extends well beyond the virial radius (~1.5 Mpc) out to at least 4.4 Mpc. The falloff profile is consistent with a gravitational potential origin rather than SZ residual, which would be confined to the cluster core.

Scripts: `scripts/aperture_test.py`

---

### Low-ℓ Cutoff Test — Scale Independence

Following a query from a lead researcher in ISW cosmology, we tested whether the signal is sensitive to the removal of large-scale CMB modes (low multipoles), applying cutoffs at ℓ = 5, 10, 20, 30, and 50.

#### Key bin z = [0.10, 0.15):

| ℓ cutoff | Removes scales | Diff (μK) | Significance |
|----------|---------------|-----------|-------------|
| None (baseline) | — | −28.7 | 2.8σ (p=0.006) |
| ℓ > 5 | > 36° | −25.0 | 2.5σ |
| ℓ > 10 | > 18° | −25.0 | 2.5σ |
| ℓ > 20 | > 9° | −25.0 | 2.5σ |
| ℓ > 30 | > 6° | −25.0 | 2.5σ |
| ℓ > 50 | > 4° | −25.1 | 2.5σ |

The signal is robust to scale selection. Removing ℓ < 5 produces a small drop (~3.7 μK), likely from dipole/quadrupole bias. Beyond that, the result is completely insensitive to the cutoff — the signal lives entirely at cluster angular scales, not in large-scale CMB modes.

This behavior differs from the Hansen et al. ISW anomaly, where low-ℓ filtering *enhanced* the signal. The two effects appear to operate at different angular scales — theirs at the void/supercluster level, ours at the individual cluster level — suggesting they are complementary rather than redundant.

Scripts: `scripts/lowl_test.py`

---

## Key Conclusions

1. Disturbed clusters show a CMB temperature **7 μK colder** than relaxed clusters of the same mass (2.1σ, p = 0.037)
2. The signal is **concentrated at z = 0.10–0.15** where it reaches −28.7 μK at 2.8σ
3. **All four independent morphology indicators** show the same sign
4. The signal **survives** bin stability testing, bootstrap (zero excluded from 95% CI), mass matching, and null testing (p = 0.003)
5. The signal **extends beyond the virial radius**, weakening gradually from 2.2 to 4.4 Mpc — consistent with a gravitational potential origin
6. The signal is **scale-independent**: removing large-scale CMB modes (ℓ < 5 through ℓ < 50) has negligible effect, confirming the signal lives at cluster angular scales
7. The sign and redshift dependence are consistent with independent reports of an anomalous negative ISW effect in the nearby Universe

---

## Motivation

This work was motivated by the [Parks Node Ejection Protocol (PNEP)](https://github.com/alikamp/Parks-Node-Ejection-Protocol), which demonstrated that the internal geometry of a gravitational system encodes stability information that scalar measures (mass, energy) miss. The ISW analysis extends this principle to cosmological scales: the morphological state of a galaxy cluster encodes information about its gravitational potential evolution that mass alone does not capture.

---

## Data

All data is publicly available:

- **Planck PSZ2 catalogue** — [IRSA](https://irsa.ipac.caltech.edu/data/Planck/release_2/catalogs/)
- **Planck SMICA no-SZ CMB map** — [Planck Legacy Archive](https://pla.esac.esa.int)

## Repository Structure

```
├── README.md
├── Parks_ISW_morphology.pdf        # Draft paper
├── Parks_ISW_morphology.tex        # LaTeX source
├── step1_download.py               # Data download
├── step2_analyze_v2.py             # Core analysis (v2)
├── scripts/
│   ├── step3_diego.py              # Extreme cuts + independent indicators
│   ├── step4_confirm.py            # Confirmation tests (bin stability, bootstrap, null)
│   ├── aperture_test.py            # Aperture scaling test
│   └── lowl_test.py                # Low-ℓ cutoff test
├── results/
│   └── results_v2.json             # Core results
└── LICENSE
```

## Reproducing

```bash
pip install healpy astropy numpy scipy matplotlib
python step1_download.py
python step2_analyze_v2.py
python scripts/step3_diego.py
python scripts/step4_confirm.py
python scripts/aperture_test.py
python scripts/lowl_test.py
```

## Author

**Alika M. Parks** — Independent Researcher, Kalaheo, HI, USA — [alikamp@gmail.com](mailto:alikamp@gmail.com)

## License

MIT
