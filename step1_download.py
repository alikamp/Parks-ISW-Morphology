#!/usr/bin/env python3
"""
ISW Morphology Analysis — Step 1: Data Download
Run this on your local machine to download all required datasets.

Requirements: pip install healpy astropy requests
"""

import os
import subprocess

DATA_DIR = "isw_data"
os.makedirs(DATA_DIR, exist_ok=True)

downloads = [
    # Planck PSZ2 cluster catalog
    {
        "url": "https://irsa.ipac.caltech.edu/data/Planck/release_2/catalogs/HFI_PCCS_SZ-union_R2.08.fits",
        "file": "psz2.fits",
        "desc": "Planck PSZ2 cluster catalog (1653 SZ-selected clusters)"
    },
    # Planck SMICA CMB temperature map (component-separated, no SZ)
    {
        "url": "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits",
        "file": "smica_nosz.fits",
        "desc": "Planck SMICA CMB map (SZ-free version, ~1.6 GB)"
    },
    # Planck SMICA regular (backup, smaller)
    {
        "url": "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "file": "smica.fits",
        "desc": "Planck SMICA CMB map (regular version, ~1.6 GB)"
    },
]

print("ISW Morphology Analysis — Data Download")
print("=" * 60)
print(f"Download directory: {DATA_DIR}/")
print()

for dl in downloads:
    fpath = os.path.join(DATA_DIR, dl["file"])
    if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
        print(f"  SKIP (exists): {dl['file']}")
        continue
    print(f"  Downloading: {dl['desc']}")
    print(f"    URL: {dl['url']}")
    print(f"    -> {fpath}")
    try:
        subprocess.run(["wget", "-q", "--show-progress", "-O", fpath, dl["url"]], check=True)
        print(f"    OK ({os.path.getsize(fpath)/1e6:.1f} MB)")
    except Exception as e:
        print(f"    FAILED: {e}")
        print(f"    Download manually from: {dl['url']}")
    print()

print()
print("Additional data (download manually if needed):")
print("  - Lovisari+ 2017 morphology data:")
print("    https://ui.adsabs.harvard.edu/abs/2017ApJ...846...51L")
print("    (Table 2: concentration and centroid shift for ESZ clusters)")
print()
print("  - UPCluster-SZ updated catalog:")
print("    https://arxiv.org/abs/2403.03818")
print()
print("Done. Run step2_analyze.py next.")
