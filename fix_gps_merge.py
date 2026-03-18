"""
fix_gps_merge.py
================
Fixes the GPS coordinate bug in the child labor analysis.

PROBLEM: The original do-file (DHS/do-file.do) assigned division-capital
coordinates to all observations based on geo_bd1994_2018 (5 divisions),
producing only 5 unique lat/lon pairs for 173,065 observations. This made
all distance thresholds (2km-30km) identical and collapsed the treatment
variable to a simple Dhaka-vs-rest dummy.

FIX: Merge actual DHS cluster-level GPS coordinates from the DHS Geographic
Datasets, which provide displaced GPS points for each of ~2,263 survey
clusters (PSUs). Then recalculate factory distances and treatment indicators.

GPS DATA: Two options to obtain cluster-level coordinates:

  OPTION A — IPUMS DHS (recommended if you already have an IPUMS account):
    1. Go to https://www.idhsdata.org/idhs/
    2. You must have DHS GPS authorization (apply at dhsprogram.com if not)
    3. Create an extract with variables: GPSLATLONG (or GPSLAT + GPSLONG),
       IDHSPSU, CLUSTERNOALL, YEAR
    4. Download as CSV and place in DHS/GPS/ipums_gps_extract.csv

  OPTION B — DHS Geographic Datasets (direct from DHS Program):
    1. Go to https://dhsprogram.com/data/
    2. Log in and request GPS data access for Bangladesh
    3. Download Geographic Datasets for each round:
         Bangladesh DHS 1999-00  ->  BDGE42FL.zip
         Bangladesh DHS 2004     ->  BDGE4JFL.zip
         Bangladesh DHS 2007     ->  BDGE52FL.zip
         Bangladesh DHS 2011     ->  BDGE61FL.zip
         Bangladesh DHS 2014     ->  BDGE71FL.zip
         Bangladesh DHS 2017-18  ->  BDGE7SFL.zip  (if using 2018)
    4. Extract .shp files into DHS/GPS/

  NOTE: DHS GPS data requires separate authorization beyond standard data
  access. Apply at https://dhsprogram.com/data/Access-Instructions.cfm

Place files in: DHS/GPS/

Usage:
    python fix_gps_merge.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(__file__).resolve().parent
DHS_DIR = BASE / "DHS"
GPS_DIR = DHS_DIR / "GPS"
FACTORY_FILE = BASE / "Accord" / "bangladesh_factories_complete_20251023_161252.csv"
OUTPUT_FILE = DHS_DIR / "final_analysis_data.csv"
OUTPUT_CORRECTED = BASE / "Results" / "final_analysis_data_corrected.csv"

# DHS Geographic Dataset filenames by survey round
# Keys: IPUMS sample code prefix from idhspsu (e.g. 5003 = 1999-00 round)
# Values: DHS GE file prefix (confirmed via DHS API)
GPS_FILES = {
    5003: "BDGE42FL",  # 1999-00
    5004: "BDGE4JFL",  # 2004
    5005: "BDGE52FL",  # 2007
    5006: "BDGE61FL",  # 2011
    5007: "BDGE71FL",  # 2014
    5008: "BDGE7SFL",  # 2017-18
}

YEAR_FROM_SAMPLE = {
    5003: 2000,
    5004: 2004,
    5005: 2007,
    5006: 2011,
    5007: 2014,
    5008: 2018,
}

# Bangladesh bounding box for filtering bad factory coordinates
BD_LAT_MIN, BD_LAT_MAX = 20.5, 26.6
BD_LON_MIN, BD_LON_MAX = 88.0, 92.7

# Distance thresholds for treatment indicators (km)
DISTANCE_THRESHOLDS = [2, 5, 7, 10, 15, 20, 25, 30]


# =============================================================================
# STEP 1: LOAD DHS CLUSTER GPS COORDINATES
# =============================================================================

def load_gps_data():
    """
    Load cluster-level GPS coordinates from DHS Geographic Datasets.

    Tries sources in order:
      1. IPUMS DHS GPS extract (single CSV with all rounds)
      2. Individual DHS GE shapefiles/dbf/csv per round

    DHS GE files contain one row per cluster with fields:
        DHSCLUST  - cluster number (matches clusternoall in IPUMS)
        LATNUM    - latitude (displaced for privacy)
        LONGNUM   - longitude (displaced for privacy)

    Returns DataFrame with columns: [sample_code, clusternoall, latitude, longitude]
    """
    print("=" * 70)
    print("STEP 1: LOADING DHS CLUSTER GPS COORDINATES")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Option A: IPUMS DHS GPS extract (single file, all rounds)
    # ------------------------------------------------------------------
    ipums_candidates = [
        GPS_DIR / "ipums_gps_extract.csv",
        GPS_DIR / "ipums_dhs_gps.csv",
        DHS_DIR / "ipums_gps_extract.csv",
    ]
    # Also check for any CSV in GPS_DIR that might be an IPUMS extract
    if GPS_DIR.exists():
        ipums_candidates += sorted(GPS_DIR.glob("*.csv"))

    for ipums_path in ipums_candidates:
        if not ipums_path.exists():
            continue
        try:
            ipums = pd.read_csv(ipums_path)
            # Standardize columns (IPUMS uses GPSLAT/GPSLONG or LATNUM/LONGNUM)
            col_map = {}
            for c in ipums.columns:
                cu = c.upper().replace(' ', '')
                if cu in ('GPSLAT', 'LATNUM', 'LAT', 'LATITUDE'):
                    col_map[c] = 'latitude'
                elif cu in ('GPSLONG', 'LONGNUM', 'LON', 'LONG', 'LONGITUDE'):
                    col_map[c] = 'longitude'
                elif cu in ('IDHSPSU',):
                    col_map[c] = 'idhspsu'
                elif cu in ('DHSCLUST', 'CLUSTERNOALL', 'CLUSTER'):
                    col_map[c] = 'clusternoall'
                elif cu in ('YEAR', 'SURVEYYEAR'):
                    col_map[c] = 'year'
                elif cu in ('SAMPLE', 'SAMPLECODE'):
                    col_map[c] = 'sample_code'
            ipums = ipums.rename(columns=col_map)

            if 'latitude' not in ipums.columns or 'longitude' not in ipums.columns:
                continue

            # If we have idhspsu, extract sample_code and clusternoall
            if 'idhspsu' in ipums.columns and 'clusternoall' not in ipums.columns:
                ipums['idhspsu_str'] = ipums['idhspsu'].astype(str)
                ipums['sample_code'] = ipums['idhspsu_str'].str[:4].astype(int)
                ipums['clusternoall'] = ipums['idhspsu_str'].str[4:].astype(int)
                ipums = ipums.drop(columns=['idhspsu_str'])

            if 'sample_code' not in ipums.columns and 'year' in ipums.columns:
                year_to_sample = {v: k for k, v in YEAR_FROM_SAMPLE.items()}
                ipums['sample_code'] = ipums['year'].map(year_to_sample)

            # Drop missing/zero coordinates
            ipums = ipums.dropna(subset=['latitude', 'longitude'])
            ipums = ipums[(ipums['latitude'] != 0) & (ipums['longitude'] != 0)]

            # Deduplicate to cluster level
            gps_all = (
                ipums
                .groupby(['sample_code', 'clusternoall'])
                .agg({'latitude': 'first', 'longitude': 'first'})
                .reset_index()
            )
            gps_all['year'] = gps_all['sample_code'].map(YEAR_FROM_SAMPLE)

            n_unique = gps_all.groupby(['latitude', 'longitude']).ngroups
            if n_unique > 10:  # sanity check: more than division centroids
                print(f"  Loaded IPUMS GPS extract: {ipums_path.name}")
                print(f"  {len(gps_all)} cluster-year coordinates, {n_unique} unique locations")
                return gps_all
            else:
                print(f"  {ipums_path.name}: only {n_unique} unique coords (likely division centroids), skipping")
                continue

        except Exception as e:
            print(f"  Error reading {ipums_path.name}: {e}")
            continue

    # ------------------------------------------------------------------
    # Option B: Individual DHS GE shapefiles / dbf / flat files per round
    # ------------------------------------------------------------------
    if not GPS_DIR.exists():
        GPS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n  Created GPS directory at {GPS_DIR}")
        print("  Please download DHS Geographic Datasets and place them here.")
        _print_download_instructions()
        return None

    all_gps = []

    for sample_code, file_prefix in GPS_FILES.items():
        year = YEAR_FROM_SAMPLE[sample_code]

        # Try multiple file formats
        loaded = False
        for ext in ['.shp', '.dbf', '.csv', '.dat']:
            fpath = GPS_DIR / f"{file_prefix}{ext}"
            if not fpath.exists():
                fpath = GPS_DIR / f"{file_prefix.lower()}{ext}"
            if not fpath.exists():
                fpath = GPS_DIR / f"{file_prefix.upper()}{ext}"
            if not fpath.exists():
                # Try inside a subdirectory (common when unzipping)
                for subdir in GPS_DIR.iterdir():
                    if subdir.is_dir():
                        candidate = subdir / f"{file_prefix}{ext}"
                        if candidate.exists():
                            fpath = candidate
                            break
            if not fpath.exists():
                continue

            try:
                if ext == '.shp':
                    import geopandas as gpd
                    gdf = gpd.read_file(fpath)
                    gps = pd.DataFrame({
                        'clusternoall': gdf['DHSCLUST'].astype(int),
                        'latitude': gdf['LATNUM'].astype(float),
                        'longitude': gdf['LONGNUM'].astype(float),
                    })
                elif ext == '.dbf':
                    try:
                        from dbfread import DBF
                        table = DBF(str(fpath))
                        gps = pd.DataFrame(iter(table))
                    except ImportError:
                        import geopandas as gpd
                        gps = gpd.read_file(fpath)
                    gps = gps.rename(columns={
                        'DHSCLUST': 'clusternoall',
                        'LATNUM': 'latitude',
                        'LONGNUM': 'longitude',
                    })
                    gps = gps[['clusternoall', 'latitude', 'longitude']]
                    gps['clusternoall'] = gps['clusternoall'].astype(int)
                elif ext in ('.csv', '.dat'):
                    gps = pd.read_csv(fpath)
                    col_map = {}
                    for c in gps.columns:
                        cl = c.upper()
                        if cl in ('DHSCLUST', 'CLUSTER', 'CLUST'):
                            col_map[c] = 'clusternoall'
                        elif cl in ('LATNUM', 'LAT', 'LATITUDE'):
                            col_map[c] = 'latitude'
                        elif cl in ('LONGNUM', 'LON', 'LONG', 'LONGITUDE'):
                            col_map[c] = 'longitude'
                    gps = gps.rename(columns=col_map)
                    gps = gps[['clusternoall', 'latitude', 'longitude']]
                    gps['clusternoall'] = gps['clusternoall'].astype(int)

                # Drop clusters with missing/zero coordinates
                gps = gps[(gps['latitude'] != 0) & (gps['longitude'] != 0)]
                gps = gps.dropna(subset=['latitude', 'longitude'])

                gps['sample_code'] = sample_code
                gps['year'] = year
                all_gps.append(gps)

                print(f"  {year} ({fpath.name}): {len(gps)} clusters loaded")
                loaded = True
                break

            except ImportError as e:
                print(f"  {year}: Need to install package for {ext} files: {e}")
                print(f"         Try: pip install geopandas  or  pip install dbfread")
                continue
            except Exception as e:
                print(f"  {year}: Error reading {fpath.name}: {e}")
                continue

        if not loaded:
            print(f"  {year}: WARNING - GPS file not found ({file_prefix}.*)")

    if not all_gps:
        print("\n  ERROR: No GPS files could be loaded.")
        _print_download_instructions()
        return None

    gps_all = pd.concat(all_gps, ignore_index=True)
    print(f"\n  Total: {len(gps_all)} cluster-year GPS coordinates loaded")
    print(f"  Unique clusters: {gps_all['clusternoall'].nunique()}")
    print(f"  Lat range: {gps_all['latitude'].min():.4f} - {gps_all['latitude'].max():.4f}")
    print(f"  Lon range: {gps_all['longitude'].min():.4f} - {gps_all['longitude'].max():.4f}")

    return gps_all


def _print_download_instructions():
    """Print instructions for downloading DHS GPS data."""
    print(f"""
  ===================================================================
  HOW TO GET DHS CLUSTER GPS COORDINATES
  ===================================================================

  OPTION A — IPUMS DHS (if you have an IPUMS account + DHS GPS access):
    1. Go to https://www.idhsdata.org/idhs/
    2. Select Bangladesh samples: 2000, 2004, 2007, 2011, 2014
    3. Add variables: IDHSPSU, YEAR, GPSLAT, GPSLONG
    4. Submit extract, download CSV
    5. Place at: {GPS_DIR}/ipums_gps_extract.csv

  OPTION B — DHS Program website:
    1. Go to https://dhsprogram.com/data/
    2. Log in and request GPS access for Bangladesh
    3. Download Geographic Datasets (GE files):
         BD 1999-00: BDGE42FL.zip
         BD 2004:    BDGE4JFL.zip
         BD 2007:    BDGE52FL.zip
         BD 2011:    BDGE61FL.zip
         BD 2014:    BDGE71FL.zip
    4. Extract into: {GPS_DIR}/

  NOTE: GPS data requires separate authorization. Apply at:
    https://dhsprogram.com/data/Access-Instructions.cfm
  ===================================================================
""")


# =============================================================================
# STEP 2: MERGE GPS INTO DHS DATA
# =============================================================================

def merge_gps_with_dhs(gps_df):
    """
    Merge cluster GPS coordinates into the main DHS analysis dataset.

    The IPUMS DHS idhspsu encodes sample + cluster:
        idhspsu = <sample_code> + <zero-padded cluster number>
    E.g., 5003000003 = sample 5003, cluster 3

    We extract sample_code and clusternoall from idhspsu to merge.
    """
    print("\n" + "=" * 70)
    print("STEP 2: MERGING GPS COORDINATES INTO DHS DATA")
    print("=" * 70)

    # Load the main DHS data
    print(f"\n  Loading DHS data from {OUTPUT_FILE}...")
    df = pd.read_csv(OUTPUT_FILE)
    print(f"  Loaded: {len(df):,} observations, {df.shape[1]} columns")
    print(f"  Unique (lat,lon) pairs BEFORE fix: {df.groupby(['latitude', 'longitude']).ngroups}")

    # Extract sample_code from idhspsu
    # idhspsu format: SSSSCCCCCC where SSSS = sample code, CCCCCC = cluster
    df['idhspsu_str'] = df['idhspsu'].astype(str)
    df['sample_code'] = df['idhspsu_str'].str[:4].astype(int)

    # clusternoall should already exist; verify it matches idhspsu encoding
    if 'clusternoall' in df.columns:
        print(f"  Using existing clusternoall column")
    else:
        # Extract cluster number from idhspsu (last 6 digits)
        df['clusternoall'] = df['idhspsu_str'].str[4:].astype(int)
        print(f"  Extracted clusternoall from idhspsu")

    # Drop old (wrong) coordinates
    df = df.drop(columns=['latitude', 'longitude'], errors='ignore')

    # Merge on sample_code + clusternoall
    merge_key = ['sample_code', 'clusternoall']
    gps_merge = gps_df[['sample_code', 'clusternoall', 'latitude', 'longitude']].drop_duplicates()

    df = df.merge(gps_merge, on=merge_key, how='left')

    n_matched = df['latitude'].notna().sum()
    n_missing = df['latitude'].isna().sum()
    print(f"\n  GPS merge results:")
    print(f"    Matched: {n_matched:,} ({n_matched/len(df)*100:.1f}%)")
    print(f"    Missing: {n_missing:,} ({n_missing/len(df)*100:.1f}%)")
    print(f"    Unique (lat,lon) pairs AFTER fix: {df.dropna(subset=['latitude']).groupby(['latitude', 'longitude']).ngroups}")

    if n_missing > 0:
        # Show which survey rounds have missing GPS
        missing_by_year = df[df['latitude'].isna()].groupby('year').size()
        print(f"\n    Missing by year:")
        for yr, n in missing_by_year.items():
            total_yr = (df['year'] == yr).sum()
            print(f"      {yr}: {n:,} / {total_yr:,} ({n/total_yr*100:.1f}%)")

    # Clean up temp column
    df = df.drop(columns=['idhspsu_str', 'sample_code'], errors='ignore')

    return df


# =============================================================================
# STEP 3: LOAD AND CLEAN FACTORY COORDINATES
# =============================================================================

def load_factories():
    """Load factory data, filtering to valid Bangladesh coordinates."""
    print("\n" + "=" * 70)
    print("STEP 3: LOADING FACTORY COORDINATES")
    print("=" * 70)

    fac = pd.read_csv(FACTORY_FILE)
    print(f"  Total factories: {len(fac):,}")

    # Filter to valid coordinates within Bangladesh
    fac = fac.dropna(subset=['latitude', 'longitude'])
    fac = fac[(fac['latitude'] != 0) & (fac['longitude'] != 0)]
    fac = fac[
        (fac['latitude'] >= BD_LAT_MIN) & (fac['latitude'] <= BD_LAT_MAX) &
        (fac['longitude'] >= BD_LON_MIN) & (fac['longitude'] <= BD_LON_MAX)
    ]

    print(f"  Factories with valid Bangladesh coordinates: {len(fac):,}")
    print(f"  Unique locations: {fac.groupby(['latitude', 'longitude']).ngroups}")

    return fac


# =============================================================================
# STEP 4: CALCULATE DISTANCES AND TREATMENT INDICATORS
# =============================================================================

def calculate_distances(df, factories):
    """
    Calculate distance from each DHS cluster to nearest factory using
    BallTree with haversine metric. Then create binary treatment indicators
    at multiple distance thresholds.
    """
    print("\n" + "=" * 70)
    print("STEP 4: CALCULATING DISTANCES TO NEAREST FACTORIES")
    print("=" * 70)

    # Work only with observations that have valid GPS
    has_gps = df['latitude'].notna() & df['longitude'].notna()
    print(f"  Observations with GPS: {has_gps.sum():,} / {len(df):,}")

    # Get unique cluster coordinates (no need to compute per-individual)
    cluster_coords = (
        df[has_gps]
        .groupby(['latitude', 'longitude'])
        .size()
        .reset_index(name='n_obs')
    )
    print(f"  Unique cluster locations: {len(cluster_coords):,}")

    # Build BallTree from factory coordinates
    fact_radians = np.radians(factories[['latitude', 'longitude']].values)
    tree = BallTree(fact_radians, metric='haversine')

    # Query distances for each unique cluster location
    cluster_radians = np.radians(cluster_coords[['latitude', 'longitude']].values)
    distances, indices = tree.query(cluster_radians, k=3)
    distances_km = distances * 6371  # Earth radius in km

    cluster_coords['nearest_factory_km'] = distances_km[:, 0]
    cluster_coords['second_nearest_km'] = distances_km[:, 1]
    cluster_coords['third_nearest_km'] = distances_km[:, 2]

    # Create binary treatment indicators at each threshold
    for threshold in DISTANCE_THRESHOLDS:
        cluster_coords[f'within_{threshold}km'] = (
            cluster_coords['nearest_factory_km'] <= threshold
        ).astype(int)

    # Continuous treatment measures
    cluster_coords['log_distance'] = np.log1p(cluster_coords['nearest_factory_km'])
    cluster_coords['proximity_score'] = 1 / (1 + cluster_coords['nearest_factory_km'])

    # Print distance distribution
    print(f"\n  Distance distribution (nearest factory):")
    print(f"    Min:    {cluster_coords['nearest_factory_km'].min():.2f} km")
    print(f"    25th:   {cluster_coords['nearest_factory_km'].quantile(0.25):.2f} km")
    print(f"    Median: {cluster_coords['nearest_factory_km'].median():.2f} km")
    print(f"    75th:   {cluster_coords['nearest_factory_km'].quantile(0.75):.2f} km")
    print(f"    Max:    {cluster_coords['nearest_factory_km'].max():.2f} km")

    print(f"\n  Treatment shares by threshold:")
    for threshold in DISTANCE_THRESHOLDS:
        col = f'within_{threshold}km'
        # Weighted by n_obs to get observation-level share
        share = (
            (cluster_coords[col] * cluster_coords['n_obs']).sum()
            / cluster_coords['n_obs'].sum()
        )
        n_clusters = cluster_coords[col].sum()
        print(f"    {threshold:>2d} km: {share*100:5.1f}% of observations "
              f"({n_clusters} clusters)")

    # Merge back to individual-level data
    distance_cols = (
        ['nearest_factory_km', 'second_nearest_km', 'third_nearest_km',
         'log_distance', 'proximity_score']
        + [f'within_{t}km' for t in DISTANCE_THRESHOLDS]
    )

    # Drop old distance columns if they exist
    for col in distance_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(
        cluster_coords[['latitude', 'longitude'] + distance_cols],
        on=['latitude', 'longitude'],
        how='left'
    )

    # Recreate standard treatment variables for main spec (10km)
    df['near_factory'] = df['within_10km']
    df['post'] = (df['year'] >= 2014).astype(int) if 'post' not in df.columns else df['post']
    df['near_x_post'] = df['near_factory'] * df['post']

    # Create interaction terms for each threshold
    for threshold in DISTANCE_THRESHOLDS:
        df[f'within_{threshold}km_x_post'] = df[f'within_{threshold}km'] * df['post']

    # Continuous interactions
    df['proximity_score_X_post'] = df['proximity_score'] * df['post']
    df['log_distance_X_post'] = df['log_distance'] * df['post']

    return df


# =============================================================================
# STEP 5: VALIDATE THE FIX
# =============================================================================

def validate(df):
    """Run validation checks on the corrected data."""
    print("\n" + "=" * 70)
    print("STEP 5: VALIDATION")
    print("=" * 70)

    # Check coordinate diversity
    n_coords = df.dropna(subset=['latitude']).groupby(['latitude', 'longitude']).ngroups
    print(f"\n  Unique (lat,lon) pairs: {n_coords}")
    assert n_coords > 100, f"FAIL: Only {n_coords} unique coordinates (expected hundreds)"
    print(f"    PASS (expected ~2000+)")

    # Check distance thresholds are distinct
    print(f"\n  Treatment shares by threshold (should be DIFFERENT):")
    shares = {}
    for threshold in DISTANCE_THRESHOLDS:
        col = f'within_{threshold}km'
        if col in df.columns:
            share = df[col].mean()
            shares[threshold] = share
            print(f"    {threshold:>2d} km: {share*100:.1f}%")

    # Verify monotonicity
    prev_share = 0
    for t in sorted(shares.keys()):
        assert shares[t] >= prev_share, (
            f"FAIL: Share at {t}km ({shares[t]:.3f}) < share at previous threshold ({prev_share:.3f})"
        )
        prev_share = shares[t]
    print(f"    PASS: Shares are monotonically increasing")

    # Verify thresholds are not all identical
    unique_shares = len(set(round(s, 4) for s in shares.values()))
    assert unique_shares > 1, "FAIL: All distance thresholds still identical!"
    print(f"    PASS: {unique_shares} distinct treatment shares across {len(shares)} thresholds")

    # Quick DID sanity check at 10km
    has_data = df.dropna(subset=['child_labor', 'near_factory', 'post', 'near_x_post'])
    print(f"\n  Quick DID check (10km threshold):")
    print(f"    N = {len(has_data):,}")
    print(f"    Treated (within 10km): {has_data['near_factory'].mean()*100:.1f}%")

    for label, mask in [('Pre-Treat', (has_data['near_factory']==1) & (has_data['post']==0)),
                        ('Pre-Control', (has_data['near_factory']==0) & (has_data['post']==0)),
                        ('Post-Treat', (has_data['near_factory']==1) & (has_data['post']==1)),
                        ('Post-Control', (has_data['near_factory']==0) & (has_data['post']==1))]:
        sub = has_data[mask]
        if len(sub) > 0:
            print(f"    {label:>15s}: CL rate = {sub['child_labor'].mean()*100:.2f}%  (N={len(sub):,})")

    print(f"\n  VALIDATION COMPLETE")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FIX GPS COORDINATES: CHILD LABOR ANALYSIS")
    print("=" * 70)
    print()
    print("This script replaces division-centroid coordinates (5 unique points)")
    print("with actual DHS cluster-level GPS coordinates (~2000+ unique points).")
    print()

    # Step 1: Load GPS data
    gps_df = load_gps_data()
    if gps_df is None:
        print("\n" + "=" * 70)
        print("CANNOT PROCEED WITHOUT DHS GPS DATA")
        print("=" * 70)
        print("After downloading the GPS data, re-run this script.")
        return

    # Step 2: Merge GPS into DHS data
    df = merge_gps_with_dhs(gps_df)

    # Step 3: Load factories
    factories = load_factories()

    # Step 4: Calculate distances
    df = calculate_distances(df, factories)

    # Step 5: Validate
    validate(df)

    # Save corrected data
    print(f"\n  Saving corrected data to {OUTPUT_CORRECTED}...")
    OUTPUT_CORRECTED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CORRECTED, index=False)
    print(f"  Saved: {len(df):,} observations, {df.shape[1]} columns")

    # Also overwrite the original analysis data so downstream scripts use it
    print(f"  Overwriting {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Done.")

    print("\n" + "=" * 70)
    print("GPS FIX COMPLETE")
    print("=" * 70)
    print(f"""
Next steps:
  1. Re-run child_labor_replication.py to regenerate all results
  2. Re-run robustness_analysis.py for updated robustness checks
  3. The distance robustness table should now show DIFFERENT coefficients
     at each threshold (monotonically decreasing with distance)
""")


if __name__ == "__main__":
    main()
