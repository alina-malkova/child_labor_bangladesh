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
COLLECTION = "idhs"  # IPUMS DHS collection code

# Bangladesh DHS household-level samples
# Format: "{country_code}{year}{record_type}" — H = household member
# These IDs can be discovered via ipums.get_all_sample_info("idhs")
BANGLADESH_SAMPLES = [
    "bd2000h",   # Bangladesh 1999-00 DHS, household members
    "bd2004h",   # Bangladesh 2004 DHS
    "bd2007h",   # Bangladesh 2007 DHS
    "bd2011h",   # Bangladesh 2011 DHS
    "bd2014h",   # Bangladesh 2014 DHS
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

    # --- Import ipumspy ---
    try:
        from ipumspy import IpumsApiClient, MicrodataExtract
    except ImportError:
        print("ERROR: ipumspy not installed. Run: pip install ipumspy")
        sys.exit(1)

    ipums = IpumsApiClient(api_key)

    # --- Discover available samples (optional, for debugging) ---
    print(f"\nQuerying available samples for collection '{COLLECTION}'...")
    try:
        all_samples = ipums.get_all_sample_info(COLLECTION)
        bd_samples = {k: v for k, v in all_samples.items()
                      if k.startswith("bd")}
        print(f"  Found {len(bd_samples)} Bangladesh samples:")
        for sid, info in sorted(bd_samples.items()):
            desc = info.get("description", str(info)) if isinstance(info, dict) else str(info)
            print(f"    {sid}: {desc}")
        print()

        # Validate our requested samples exist
        for s in BANGLADESH_SAMPLES:
            if s not in all_samples:
                print(f"  WARNING: Sample '{s}' not found in API. "
                      f"Will try anyway.")
    except Exception as e:
        print(f"  Could not query sample metadata: {e}")
        print("  Proceeding with predefined sample IDs...\n")

    # --- Create extract ---
    print("Creating IPUMS DHS extract request...")
    print(f"  Collection: {COLLECTION}")
    print(f"  Samples: {BANGLADESH_SAMPLES}")
    print(f"  Variables: {GPS_VARIABLES}")

    extract = MicrodataExtract(
        collection=COLLECTION,
        description="Bangladesh DHS cluster GPS coordinates for child labor analysis",
        samples=BANGLADESH_SAMPLES,
        variables=GPS_VARIABLES,
        data_format="csv",
    )

    # --- Submit ---
    print("\nSubmitting extract request...")
    try:
        ipums.submit_extract(extract)
    except Exception as e:
        error_msg = str(e)
        if "unauthorized" in error_msg.lower() or "permission" in error_msg.lower():
            print(f"\nERROR: {e}")
            print("\nThis likely means you don't have DHS GPS authorization.")
            print("Apply at: https://dhsprogram.com/data/Access-Instructions.cfm")
            print("Once approved, re-run this script.")
        elif "sample" in error_msg.lower():
            print(f"\nERROR: {e}")
            print("\nSample IDs may be incorrect. Trying to list valid IDs...")
            _try_list_samples(ipums)
        else:
            print(f"\nERROR submitting extract: {e}")
        sys.exit(1)

    print(f"  Extract submitted! ID: {extract.extract_id}")

    # --- Wait ---
    print("\nWaiting for extract to complete (this may take a few minutes)...")
    try:
        ipums.wait_for_extract(extract, timeout=600)
    except Exception as e:
        print(f"\nERROR waiting for extract: {e}")
        print(f"  You can check status later with extract ID: {extract.extract_id}")
        sys.exit(1)

    print("  Extract complete!")

    # --- Download ---
    GPS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading to {GPS_DIR}...")
    try:
        ipums.download_extract(extract, download_dir=GPS_DIR)
    except Exception as e:
        print(f"\nERROR downloading: {e}")
        sys.exit(1)

    # --- Find and report downloaded files ---
    downloaded = list(GPS_DIR.glob("*"))
    print(f"\n  Downloaded {len(downloaded)} files:")
    for f in downloaded:
        size = f.stat().st_size
        print(f"    {f.name}  ({size:,} bytes)")

    # --- Try to read and verify ---
    _verify_download(GPS_DIR)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"\nNext step: python fix_gps_merge.py")


def _try_list_samples(ipums):
    """Try to list valid Bangladesh sample IDs."""
    try:
        all_samples = ipums.get_all_sample_info(COLLECTION)
        bd_samples = {k: v for k, v in all_samples.items()
                      if k.startswith("bd")}
        if bd_samples:
            print("\nValid Bangladesh sample IDs:")
            for sid in sorted(bd_samples.keys()):
                print(f"  {sid}")
            print(f"\nUpdate BANGLADESH_SAMPLES in this script with "
                  f"the correct IDs and re-run.")
    except Exception:
        print("Could not retrieve sample list.")


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
