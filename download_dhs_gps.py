"""
download_dhs_gps.py
===================
Downloads DHS cluster GPS coordinates via the IPUMS DHS API (ipumspy).

PREREQUISITES:
  1. pip install ipumspy
  2. An IPUMS account (register at https://www.ipums.org/)
  3. An IPUMS API key (create at https://account.ipums.org/api_keys)
  4. DHS GPS data authorization from DHS Program
     (apply at https://dhsprogram.com/data/Access-Instructions.cfm)

     Without GPS authorization, the extract will fail with an error
     about unauthorized samples. You MUST have GPS access approved
     by the DHS Program before running this script.

  Set your API key as an environment variable:
      export IPUMS_API_KEY="your_key_here"

  Or pass it interactively when prompted.

Usage:
    python download_dhs_gps.py

After download, run fix_gps_merge.py to merge GPS into the analysis data.
"""

import os
import sys
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

BASE = Path(__file__).resolve().parent
GPS_DIR = BASE / "DHS" / "GPS"
COLLECTION = "dhs"  # IPUMS DHS collection code

# Bangladesh DHS household-member-level samples
# Format: "{country_code}{year}{record_type}" — pr = Members
# Discovered via ipums.get_all_sample_info("dhs")
BANGLADESH_SAMPLES = [
    "bd2000pr",  # Bangladesh 1999-00 DHS, members
    "bd2004pr",  # Bangladesh 2004 DHS
    "bd2007pr",  # Bangladesh 2007 DHS
    "bd2011pr",  # Bangladesh 2011 DHS
    "bd2014pr",  # Bangladesh 2014 DHS
]

# Variables to request
# IDHSPSU = primary sampling unit (cluster) ID
# YEAR = survey year
# GPSLATLONG = GPS lat/lon (becomes GPSLAT + GPSLONG in output)
GPS_VARIABLES = ["IDHSPSU", "YEAR", "GPSLATLONG"]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("DOWNLOAD DHS CLUSTER GPS VIA IPUMS API")
    print("=" * 70)

    # --- Get API key ---
    api_key = os.environ.get("IPUMS_API_KEY")
    if not api_key:
        print("\nIPUMS_API_KEY environment variable not set.")
        api_key = input("Enter your IPUMS API key: ").strip()
        if not api_key:
            print("ERROR: No API key provided. Exiting.")
            sys.exit(1)

    import requests
    import json
    import time

    API_BASE = "https://api.ipums.org/extracts"
    HEADERS = {"Authorization": api_key, "Content-Type": "application/json"}

    # --- Build extract payload ---
    # Using raw API to avoid ipumspy bug with attachedCharacteristics on DHS
    payload = {
        "description": "Bangladesh DHS cluster GPS for child labor analysis",
        "dataFormat": "csv",
        "dataStructure": {"rectangular": {"on": "P"}},
        "samples": {s: {} for s in BANGLADESH_SAMPLES},
        "variables": {v: {} for v in GPS_VARIABLES},
    }

    print(f"\nSubmitting IPUMS DHS extract...")
    print(f"  Collection: {COLLECTION}")
    print(f"  Samples: {BANGLADESH_SAMPLES}")
    print(f"  Variables: {GPS_VARIABLES}")

    # --- Submit ---
    r = requests.post(
        f"{API_BASE}?collection={COLLECTION}&version=2",
        headers=HEADERS,
        json=payload,
    )

    if r.status_code == 403:
        data = r.json()
        detail = data.get("detail", str(data))
        print(f"\n  ERROR 403 Forbidden: {detail}")
        if "Gps" in data.get("type", "") or "GPS" in str(detail):
            print("\n  You do NOT have DHS GPS authorization for these samples.")
            print("  To fix this:")
            print("    1. Go to https://dhsprogram.com/data/Access-Instructions.cfm")
            print("    2. Log in with your DHS account")
            print("    3. Request access to Geographic Datasets for Bangladesh")
            print("    4. Wait for approval (typically 24-48 hours)")
            print("    5. Re-run this script")
        sys.exit(1)
    elif r.status_code != 200:
        print(f"\n  ERROR {r.status_code}: {r.text[:500]}")
        sys.exit(1)

    extract_data = r.json()
    extract_id = extract_data.get("number")
    print(f"  Extract submitted! ID: {extract_id}")

    # --- Wait for completion ---
    print("\nWaiting for extract to complete...")
    for attempt in range(120):  # up to 10 minutes
        time.sleep(5)
        status_r = requests.get(
            f"{API_BASE}/{extract_id}?collection={COLLECTION}&version=2",
            headers=HEADERS,
        )
        if status_r.status_code != 200:
            print(f"  Status check error: {status_r.status_code}")
            continue

        status_data = status_r.json()
        status = status_data.get("status", "unknown")

        if status == "completed":
            print(f"  Extract ready! (took ~{(attempt+1)*5}s)")
            break
        elif status in ("failed", "canceled"):
            print(f"  Extract {status}!")
            print(f"  Details: {json.dumps(status_data, indent=2)[:500]}")
            sys.exit(1)
        else:
            if attempt % 6 == 0:  # print every 30s
                print(f"  Status: {status}...")
    else:
        print("  Timed out waiting for extract (10 minutes).")
        print(f"  Check manually: extract ID {extract_id}")
        sys.exit(1)

    # --- Download ---
    GPS_DIR.mkdir(parents=True, exist_ok=True)
    download_links = status_data.get("downloadLinks", {})
    print(f"\nDownloading to {GPS_DIR}...")

    for link_name, url in download_links.items():
        fname = url.split("/")[-1].split("?")[0]
        if not fname:
            fname = f"dhs_extract_{extract_id}_{link_name}"
        fpath = GPS_DIR / fname
        print(f"  Downloading {fname}...")
        dr = requests.get(url, headers={"Authorization": api_key}, stream=True)
        dr.raise_for_status()
        with open(fpath, "wb") as f:
            for chunk in dr.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"    Saved: {fpath.name} ({fpath.stat().st_size:,} bytes)")

    # --- List all downloaded files ---
    downloaded = list(GPS_DIR.glob("*"))
    print(f"\n  Total files in {GPS_DIR.name}/: {len(downloaded)}")
    for f in sorted(downloaded):
        print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    # --- Try to read and verify ---
    _verify_download(GPS_DIR)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"\nNext step: python fix_gps_merge.py")


def _verify_download(download_dir):
    """Try to read downloaded GPS data and verify it has real coordinates."""
    import pandas as pd

    # Look for CSV files
    csv_files = list(download_dir.glob("*.csv")) + list(download_dir.glob("*.csv.gz"))
    if not csv_files:
        # ipumspy may download as .dat with a .xml codebook
        dat_files = list(download_dir.glob("*.dat.gz")) + list(download_dir.glob("*.dat"))
        if dat_files:
            print(f"\n  Downloaded as fixed-width format ({dat_files[0].name}).")
            print("  To convert: use ipumspy readers or re-request as CSV format.")
            # Try to read with ipumspy
            try:
                from ipumspy import readers
                ddi_files = list(download_dir.glob("*.xml"))
                if ddi_files:
                    ddi = readers.read_ipums_ddi(ddi_files[0])
                    df = readers.read_microdata(ddi, dat_files[0])
                    _check_gps_dataframe(df, "ipumspy reader")
            except Exception as e:
                print(f"  Could not auto-read: {e}")
        return

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            _check_gps_dataframe(df, csv_file.name)
            # Save as standardized name for fix_gps_merge.py
            out_path = download_dir / "ipums_gps_extract.csv"
            if csv_file.name != "ipums_gps_extract.csv":
                df.to_csv(out_path, index=False)
                print(f"  Saved as: {out_path.name}")
            break
        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")


def _check_gps_dataframe(df, source_name):
    """Verify a GPS dataframe has real coordinates."""
    print(f"\n  Verifying GPS data from {source_name}...")
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    # Find lat/lon columns
    lat_col = lon_col = None
    for c in df.columns:
        cu = c.upper()
        if cu in ('GPSLAT', 'LATNUM', 'LATITUDE'):
            lat_col = c
        elif cu in ('GPSLONG', 'LONGNUM', 'LONGITUDE'):
            lon_col = c

    if lat_col and lon_col:
        valid = df[(df[lat_col] != 0) & (df[lon_col] != 0)].dropna(
            subset=[lat_col, lon_col])
        n_unique = valid.groupby([lat_col, lon_col]).ngroups
        print(f"    Valid GPS rows: {len(valid):,}")
        print(f"    Unique coordinates: {n_unique}")
        if n_unique > 100:
            print(f"    PASS: Real cluster-level GPS data!")
        elif n_unique <= 10:
            print(f"    WARNING: Only {n_unique} unique coords "
                  f"(may be division centroids)")
    else:
        print(f"    WARNING: Could not find lat/lon columns")


if __name__ == "__main__":
    main()
