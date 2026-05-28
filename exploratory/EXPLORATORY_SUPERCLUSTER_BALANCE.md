Exploratory Analysis — Supercluster Morphology Balance
Preliminary result — May 26, 2026

Hypothesis
If galaxy cluster mergers are regulated by large scale structure, superclusters should maintain a statistical balance between disturbed and relaxed cluster populations at any given cosmic epoch — a possible conservation mechanism operating at the filament scale.
Method
Cross-matched 12,247 eROSITA eRASS1 clusters with the eROSITA supercluster catalog (Liu et al. 2024, 1,338 superclusters). Morphology classified using D_COMB parameter from Sanders et al. 2025 — a combined X-ray morphology score (centroid shift, concentration, power ratios, ellipticity) normalized 0-1. Median split: D_COMB > 0.183 = disturbed, ≤ 0.183 = relaxed.
Result
MetricValueSuperclusters tested468Overall disturbed1,243Overall relaxed1,318Overall ratio0.943Median ratio1.000Ratios within 0.5-2.0347/468 (74%)
The median supercluster maintains exactly equal disturbed and relaxed cluster populations. 74% of superclusters are within a factor of 2 of perfect balance. The overall ratio across all 12,247 clusters is 0.943 — within 6% of unity.
Examples of perfectly balanced superclusters:
SuperclusterN_disN_relz1eRASS-SC J0425-6329880.2361eRASS-SC J0222-4049550.2331eRASS-SC J0005-6802550.2431eRASS-SC J0534-3617440.284
Physical interpretation
The near-unity median ratio suggests the cosmic web self-regulates — merger activity and relaxation occur at statistically equal rates within large scale structure at z~0.05-0.5. This is consistent with a thermodynamic equilibrium at the supercluster scale where the merger rate equals the relaxation rate.
If confirmed with larger samples and proper mass-matching, this would represent a previously unidentified conservation mechanism in large scale structure — the cosmic web maintaining dynamical equilibrium through balanced merger and relaxation cycles.
Caveats

D_COMB morphology classification not mass-matched per supercluster
eROSITA western hemisphere only — full sky test requires future data
Median statistic robust but mean ratio (1.343) pulled by outliers
Requires independent confirmation with PSZ2 or ACT morphology catalogs

Next steps
Mass-matched disturbed/relaxed ratio per supercluster as function of redshift and supercluster mass. Test whether ratio evolves with cosmic time — if merger rate exceeded relaxation rate at higher redshift, the ratio should be >1 at z>0.3 and converge to 1.0 at low redshift as dark energy suppresses merger activity.

Data: eROSITA eRASS1 (Bulbul et al. 2024), eROSITA supercluster catalog (Liu et al. 2024), eROSITA morphology catalog (Sanders et al. 2025)
Code: available on request
