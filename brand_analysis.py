"""
Brand-Specific Analysis: Do brands with more factories near children
show different child labor effects post-Rana Plaza?

For each major brand (H&M, Walmart, Primark, C&A, PVH, Mango, M&S),
create a treatment indicator: child lives within 10km of that brand's factory.
Then run brand-specific DiD regressions.

Uses corrected GPS cluster coordinates from DHS shapefiles.
"""
import os
import ast
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import BallTree
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

BASE = "/Users/amalkova/Library/CloudStorage/OneDrive-FloridaInstituteofTechnology/_Research/Labor_Economics/Child Labor"
OUT = f"{BASE}/results/new_results"
EARTH_R = 6371.0

BLUE = '#1f77b4'
DARK_BLUE = '#0d4f8b'
RED = '#d62728'
GRAY = '#888888'

# ============================================================================
# 1. Build brand-factory mapping from all sources
# ============================================================================
print("=" * 70)
print("1. BUILDING BRAND-FACTORY MAPPING")
print("=" * 70)

# Source A: Main factory census (has GPS + primary_brand)
factories = pd.read_csv(f"{BASE}/Factories/Bangladesh Factories Complete Sept 2025.csv")
factories = factories.dropna(subset=['latitude', 'longitude'])
factories = factories[
    (factories['latitude'] >= 20) & (factories['latitude'] <= 27) &
    (factories['longitude'] >= 88) & (factories['longitude'] <= 93)
].copy()
print(f"Factory census: {len(factories)} with valid GPS")

# Parse identified_brands
def parse_brands(val):
    if pd.isna(val) or val == '[]':
        return []
    try:
        return ast.literal_eval(val)
    except:
        return [b.strip() for b in str(val).split(',') if b.strip()]

factories['brand_list'] = factories['identified_brands'].apply(parse_brands)

# Source B: Accord factories (has matched_brands + some GPS)
accord = pd.read_csv(f"{BASE}/Accord/accord_factories_20251029_190336.csv")
alliance = pd.read_csv(f"{BASE}/Accord/alliance_factories_20251029_190336.csv")

# Build brand -> factory locations mapping
# Use the complete factory census as primary (best GPS quality)
brand_factory_coords = {}

# From factory census
for _, row in factories.iterrows():
    brands = row['brand_list']
    if row['primary_brand'] and pd.notna(row['primary_brand']):
        if row['primary_brand'] not in brands:
            brands.append(row['primary_brand'])
    for brand in brands:
        brand = brand.strip()
        if brand not in brand_factory_coords:
            brand_factory_coords[brand] = []
        brand_factory_coords[brand].append((row['latitude'], row['longitude']))

# From Accord/Alliance (only use factories with GPS)
for src_df, src_name in [(accord, 'Accord'), (alliance, 'Alliance')]:
    has_gps = src_df.dropna(subset=['latitude', 'longitude'])
    has_gps = has_gps[(has_gps['latitude'] != 0) & (has_gps['longitude'] != 0)]

    for _, row in has_gps.iterrows():
        if pd.notna(row.get('matched_brands')):
            for brand in str(row['matched_brands']).split(','):
                brand = brand.strip()
                if brand and brand not in ['nan']:
                    if brand not in brand_factory_coords:
                        brand_factory_coords[brand] = []
                    brand_factory_coords[brand].append(
                        (row['latitude'], row['longitude'])
                    )

# Deduplicate coordinates per brand
for brand in brand_factory_coords:
    coords = list(set(brand_factory_coords[brand]))
    brand_factory_coords[brand] = coords

# Standardize brand names
name_map = {
    'H&M': 'H&M', 'hm': 'H&M',
    'Walmart': 'Walmart',
    'Primark': 'Primark',
    'C&A': 'C&A', 'Canda': 'C&A', 'ca': 'C&A',
    'PVH': 'PVH', 'Pvh': 'PVH',
    'Mango': 'Mango',
    'Marks & Spencer': 'M&S', 'M&S': 'M&S',
    'Gap': 'Gap',
    'Uniqlo': 'Uniqlo',
    'Next': 'Next',
    'Esprit': 'Esprit',
    'Tesco': 'Tesco',
    'Target': 'Target',
}

std_brand_coords = {}
for raw_name, coords in brand_factory_coords.items():
    std_name = name_map.get(raw_name, raw_name)
    if std_name not in std_brand_coords:
        std_brand_coords[std_name] = []
    std_brand_coords[std_name].extend(coords)

# Deduplicate again
for brand in std_brand_coords:
    std_brand_coords[brand] = list(set(std_brand_coords[brand]))

print("\nBrand factory counts (unique locations):")
for brand, coords in sorted(std_brand_coords.items(), key=lambda x: -len(x[1])):
    if len(coords) >= 5:
        print(f"  {brand:20s}: {len(coords):4d} factory locations")


# ============================================================================
# 2. Load DHS data with corrected GPS
# ============================================================================
print("\n" + "=" * 70)
print("2. LOADING DHS DATA (CORRECTED GPS)")
print("=" * 70)

df = pd.read_csv(f"{BASE}/DHS/final_analysis_data.csv", low_memory=False)
df['child_labor'] = pd.to_numeric(df['child_labor'], errors='coerce')
df['near_factory'] = df['within_10km']
df['post'] = df['post_rana']
df['near_x_post'] = df['within_10km'] * df['post_rana']

df = df[(df['age'] >= 5) & (df['age'] <= 17)]
df = df.dropna(subset=['child_labor', 'near_factory', 'year', 'age',
                        'latitude', 'longitude'])
df = df[df['year'] != 2018]

# Age group for main analysis
df['age_10_13'] = ((df['age'] >= 10) & (df['age'] <= 13)).astype(int)

if 'sex' in df.columns:
    if df['sex'].dtype == object:
        df['sex'] = (df['sex'] == 'Male').astype(float)
    else:
        df['sex'] = pd.to_numeric(df['sex'], errors='coerce')

print(f"Sample: {len(df)} observations")
print(f"Ages 10-13: {df['age_10_13'].sum()} observations")


# ============================================================================
# 3. Compute brand-specific treatment indicators
# ============================================================================
print("\n" + "=" * 70)
print("3. COMPUTING BRAND-SPECIFIC TREATMENT (WITHIN 10KM)")
print("=" * 70)

# Get unique DHS cluster locations
cluster_coords = df[['latitude', 'longitude']].drop_duplicates().values
cluster_rad = np.radians(cluster_coords)

THRESHOLD_KM = 10

brands_to_analyze = ['Mango', 'C&A', 'PVH', 'Primark', 'Walmart',
                     'H&M', 'M&S', 'Gap', 'Uniqlo']

brand_treatment = {}

for brand in brands_to_analyze:
    if brand not in std_brand_coords or len(std_brand_coords[brand]) < 3:
        print(f"  {brand}: skipped (< 3 factory locations)")
        continue

    factory_locs = np.array(std_brand_coords[brand])
    factory_rad = np.radians(factory_locs)

    tree = BallTree(factory_rad, metric='haversine')
    dist_rad, _ = tree.query(cluster_rad, k=1)
    dist_km = dist_rad.ravel() * EARTH_R

    # Map back to all observations
    near_brand = (dist_km <= THRESHOLD_KM).astype(int)
    coord_to_near = {}
    for i, (lat, lon) in enumerate(cluster_coords):
        coord_to_near[(lat, lon)] = near_brand[i]

    col_name = f'near_{brand.replace("&", "").replace(" ", "_").lower()}'
    df[col_name] = df.apply(
        lambda r: coord_to_near.get((r['latitude'], r['longitude']), 0), axis=1
    )
    df[f'{col_name}_x_post'] = df[col_name] * df['post']

    n_treated = df[col_name].sum()
    pct = 100 * df[col_name].mean()
    brand_treatment[brand] = {
        'col': col_name,
        'n_factories': len(std_brand_coords[brand]),
        'n_treated': n_treated,
        'pct_treated': pct,
    }
    print(f"  {brand:12s}: {len(std_brand_coords[brand]):4d} factories, "
          f"{n_treated:6d} treated ({pct:.1f}%)")


# ============================================================================
# 4. Brand-specific DiD regressions (Ages 10-13)
# ============================================================================
print("\n" + "=" * 70)
print("4. BRAND-SPECIFIC DiD REGRESSIONS (AGES 10-13)")
print("=" * 70)

sub = df[df['age_10_13'] == 1].copy()

# Controls
age_dummies = pd.get_dummies(sub['age'], prefix='age', drop_first=True).astype(float)
sub = pd.concat([sub, age_dummies], axis=1)
age_cols = [c for c in sub.columns if c.startswith('age_') and c[4:].isdigit()]

base_controls = age_cols.copy()
if 'sex' in sub.columns:
    base_controls.append('sex')

results = []

print(f"\n{'Brand':<12} {'Factories':>9} {'Treated%':>9} {'Coef':>9} "
      f"{'SE':>8} {'p-val':>8} {'N':>7}")
print("-" * 75)

for brand in brands_to_analyze:
    if brand not in brand_treatment:
        continue

    info = brand_treatment[brand]
    col = info['col']
    interact = f'{col}_x_post'

    # DiD regression: child_labor = a + b*near_brand + c*post + d*near_brand*post + controls
    X_vars = [col, 'post', interact] + base_controls
    X = sm.add_constant(sub[X_vars].astype(float))
    y = sub['child_labor'].astype(float)

    if 'geolev2' in sub.columns:
        model = sm.OLS(y, X).fit(cov_type='cluster',
                                  cov_kwds={'groups': sub['geolev2']})
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1')

    coef = model.params[interact]
    se = model.bse[interact]
    pval = model.pvalues[interact]
    n = int(model.nobs)

    stars = ''
    if pval < 0.01: stars = '***'
    elif pval < 0.05: stars = '**'
    elif pval < 0.10: stars = '*'

    print(f"{brand:<12} {info['n_factories']:>9} {info['pct_treated']:>8.1f}% "
          f"{coef:>+9.4f} ({se:>7.4f}) {pval:>7.3f}{stars:>3} {n:>6}")

    results.append({
        'brand': brand,
        'n_factories': info['n_factories'],
        'pct_treated': info['pct_treated'],
        'coef': coef,
        'se': se,
        'p_value': pval,
        'ci_lo': coef - 1.96 * se,
        'ci_hi': coef + 1.96 * se,
        'n_obs': n,
    })


# ============================================================================
# 5. Brand-tier analysis (aggregate by tier)
# ============================================================================
print("\n" + "=" * 70)
print("5. BRAND-TIER ANALYSIS")
print("=" * 70)

# Premium fast fashion (H&M, Uniqlo, Mango)
# Value fast fashion (Primark, C&A)
# Mass market (Walmart, Gap, Target)
# European premium (M&S, PVH)

tier_map = {
    'premium_fast_fashion': ['H&M', 'Uniqlo', 'Mango'],
    'value_fast_fashion': ['Primark', 'C&A'],
    'mass_market': ['Walmart', 'Gap'],
    'european_premium': ['M&S', 'PVH'],
}

tier_results = []
for tier_name, tier_brands in tier_map.items():
    # Combine factory locations from all brands in tier
    all_coords = []
    for b in tier_brands:
        if b in std_brand_coords:
            all_coords.extend(std_brand_coords[b])
    all_coords = list(set(all_coords))

    if len(all_coords) < 3:
        continue

    factory_rad = np.radians(np.array(all_coords))
    tree = BallTree(factory_rad, metric='haversine')
    dist_rad, _ = tree.query(cluster_rad, k=1)
    dist_km = dist_rad.ravel() * EARTH_R

    near_tier = (dist_km <= THRESHOLD_KM).astype(int)
    coord_to_near = {}
    for i, (lat, lon) in enumerate(cluster_coords):
        coord_to_near[(lat, lon)] = near_tier[i]

    col = f'near_tier_{tier_name}'
    sub[col] = sub.apply(
        lambda r: coord_to_near.get((r['latitude'], r['longitude']), 0), axis=1
    )
    sub[f'{col}_x_post'] = sub[col] * sub['post']

    X_vars = [col, 'post', f'{col}_x_post'] + base_controls
    X = sm.add_constant(sub[X_vars].astype(float))
    y = sub['child_labor'].astype(float)

    if 'geolev2' in sub.columns:
        model = sm.OLS(y, X).fit(cov_type='cluster',
                                  cov_kwds={'groups': sub['geolev2']})
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1')

    interact = f'{col}_x_post'
    coef = model.params[interact]
    se = model.bse[interact]
    pval = model.pvalues[interact]

    stars = ''
    if pval < 0.01: stars = '***'
    elif pval < 0.05: stars = '**'
    elif pval < 0.10: stars = '*'

    brands_str = ', '.join(tier_brands)
    print(f"  {tier_name:<25} ({brands_str})")
    print(f"    Factories: {len(all_coords)}, "
          f"Treated: {sub[col].sum()} ({100*sub[col].mean():.1f}%)")
    print(f"    DiD coef: {coef:+.4f} (SE={se:.4f}, p={pval:.3f}){stars}")
    print()

    tier_results.append({
        'tier': tier_name,
        'brands': brands_str,
        'n_factories': len(all_coords),
        'coef': coef, 'se': se, 'p_value': pval,
        'ci_lo': coef - 1.96 * se, 'ci_hi': coef + 1.96 * se,
    })


# ============================================================================
# 6. Accord vs Alliance comparison
# ============================================================================
print("=" * 70)
print("6. ACCORD vs ALLIANCE COMPARISON")
print("=" * 70)

# Accord factories with GPS
accord_gps = accord.dropna(subset=['latitude', 'longitude'])
accord_gps = accord_gps[(accord_gps['latitude'] != 0) & (accord_gps['longitude'] != 0)]
accord_coords = accord_gps[['latitude', 'longitude']].drop_duplicates().values

# Alliance factories with GPS
alliance_gps = alliance.dropna(subset=['latitude', 'longitude'])
alliance_gps = alliance_gps[(alliance_gps['latitude'] != 0) & (alliance_gps['longitude'] != 0)]
alliance_coords = alliance_gps[['latitude', 'longitude']].drop_duplicates().values

org_results = []
for org_name, org_coords in [('Accord', accord_coords), ('Alliance', alliance_coords)]:
    if len(org_coords) < 3:
        print(f"  {org_name}: skipped (< 3 GPS locations)")
        continue

    factory_rad = np.radians(org_coords)
    tree = BallTree(factory_rad, metric='haversine')
    dist_rad, _ = tree.query(cluster_rad, k=1)
    dist_km = dist_rad.ravel() * EARTH_R

    near_org = (dist_km <= THRESHOLD_KM).astype(int)
    coord_to_near = {}
    for i, (lat, lon) in enumerate(cluster_coords):
        coord_to_near[(lat, lon)] = near_org[i]

    col = f'near_{org_name.lower()}'
    sub[col] = sub.apply(
        lambda r: coord_to_near.get((r['latitude'], r['longitude']), 0), axis=1
    )
    sub[f'{col}_x_post'] = sub[col] * sub['post']

    X_vars = [col, 'post', f'{col}_x_post'] + base_controls
    X = sm.add_constant(sub[X_vars].astype(float))
    y = sub['child_labor'].astype(float)

    if 'geolev2' in sub.columns:
        model = sm.OLS(y, X).fit(cov_type='cluster',
                                  cov_kwds={'groups': sub['geolev2']})
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1')

    interact = f'{col}_x_post'
    coef = model.params[interact]
    se = model.bse[interact]
    pval = model.pvalues[interact]
    stars = ''
    if pval < 0.01: stars = '***'
    elif pval < 0.05: stars = '**'
    elif pval < 0.10: stars = '*'

    print(f"  {org_name}: {len(org_coords)} GPS locations, "
          f"Treated: {sub[col].sum()} ({100*sub[col].mean():.1f}%)")
    print(f"    DiD coef: {coef:+.4f} (SE={se:.4f}, p={pval:.3f}){stars}")
    print()

    org_results.append({
        'organization': org_name,
        'n_factories': len(org_coords),
        'coef': coef, 'se': se, 'p_value': pval,
    })


# ============================================================================
# 7. Figure: Brand-specific coefficients
# ============================================================================
print("=" * 70)
print("7. GENERATING FIGURE")
print("=" * 70)

if results:
    res_df = pd.DataFrame(results).sort_values('n_factories', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = range(len(res_df))
    ax.errorbar(res_df['coef'], y_pos,
                xerr=1.96 * res_df['se'],
                fmt='o', color=DARK_BLUE, markersize=7, capsize=4,
                linewidth=1.3, markerfacecolor=BLUE,
                markeredgecolor=DARK_BLUE, ecolor=BLUE, elinewidth=1.3)

    ax.axvline(x=0, color=GRAY, linestyle='-', alpha=0.5, linewidth=0.8)

    labels = [f"{r['brand']} ({r['n_factories']})" for _, r in res_df.iterrows()]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Treatment Effect (Near Brand Factory \u00d7 Post)', fontsize=11)
    ax.set_title('Brand-Specific DiD Estimates: Ages 10\u201313',
                 fontsize=14, fontweight='bold')

    # Add baseline (all factories) reference line
    baseline = results[0]  # We'll compute this separately below
    # Just use the overall within_10km result
    X_all = sm.add_constant(sub[['near_factory', 'post', 'near_x_post'] +
                                 base_controls].astype(float))
    y_all = sub['child_labor'].astype(float)
    if 'geolev2' in sub.columns:
        m_all = sm.OLS(y_all, X_all).fit(cov_type='cluster',
                                          cov_kwds={'groups': sub['geolev2']})
    else:
        m_all = sm.OLS(y_all, X_all).fit(cov_type='HC1')
    all_coef = m_all.params['near_x_post']
    ax.axvline(x=all_coef, color=RED, linestyle='--', alpha=0.5, linewidth=1,
               label=f'All factories ({all_coef:+.4f})')
    ax.legend(fontsize=9, framealpha=0.8, loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{OUT}/brand_specific_did.png", dpi=300)
    plt.close()
    print(f"  Saved brand_specific_did.png")


# ============================================================================
# 8. Save results
# ============================================================================
print("\n" + "=" * 70)
print("8. SAVING RESULTS")
print("=" * 70)

res_df = pd.DataFrame(results)
res_df.to_csv(f"{OUT}/brand_did_results.csv", index=False)
print(f"  brand_did_results.csv")

if tier_results:
    tier_df = pd.DataFrame(tier_results)
    tier_df.to_csv(f"{OUT}/brand_tier_results.csv", index=False)
    print(f"  brand_tier_results.csv")

if org_results:
    org_df = pd.DataFrame(org_results)
    org_df.to_csv(f"{OUT}/accord_vs_alliance_results.csv", index=False)
    print(f"  accord_vs_alliance_results.csv")

print("\n=== Brand analysis complete ===")
