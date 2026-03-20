"""
Generate all corrected coefficients for paper tables using GPS-corrected data.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/amalkova/Library/CloudStorage/OneDrive-FloridaInstituteofTechnology/_Research/Labor_Economics/Child Labor"

def load():
    df = pd.read_csv(f"{BASE}/DHS/final_analysis_data.csv")
    df['near_factory'] = df['within_10km']
    df['post'] = df['post_rana']
    # IMPORTANT: treat_X_post in CSV uses OLD (broken) treatment assignment
    # Must recompute from corrected within_10km
    df['near_x_post'] = df['near_factory'] * df['post']
    # Create female dummy from sex variable
    df['female'] = (df['sex'] == 'Female').astype(int)
    # Wealth: convert categorical to numeric
    wealth_map = {'Poorest': 1, 'Poorer': 2, 'Middle': 3, 'Richer': 4, 'Richest': 5}
    df['wealth_num'] = df['wealthqhh'].map(wealth_map)
    # District for clustering
    df['district'] = df['geolev2']
    return df

def cluster_ols(formula, data, cluster_var='district'):
    """OLS with district-clustered SEs."""
    d = data.dropna(subset=[cluster_var])
    mod = smf.ols(formula, data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d[cluster_var].astype(int)})
    return mod

def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

def print_coef(mod, var, label=None):
    if label is None: label = var
    b = mod.params[var]
    se = mod.bse[var]
    p = mod.pvalues[var]
    print(f"  {label}: {b:.4f}{stars(p)} (SE={se:.4f}, p={p:.3f})")
    return b, se, p

# ============================================================================
print("=" * 70)
print("TABLE 1: MAIN DiD RESULTS (ages 5-17)")
print("=" * 70)
df = load()
ages517 = df[(df['age'] >= 5) & (df['age'] <= 17)].copy()
ages517 = ages517.dropna(subset=['child_labor', 'near_factory', 'post', 'near_x_post'])

print(f"\nN = {len(ages517):,}")
print(f"Mean child labor = {ages517['child_labor'].mean():.4f}")

# M1: Baseline
m1 = cluster_ols('child_labor ~ near_x_post + near_factory + post', ages517)
print("\n(1) Baseline:")
print_coef(m1, 'near_x_post', 'Near×Post')
print_coef(m1, 'near_factory', 'Near Factory')
print_coef(m1, 'post', 'Post')
print(f"  N={m1.nobs:.0f}, R²={m1.rsquared:.4f}")

# M2: + Individual controls (age FE + female)
m2 = cluster_ols('child_labor ~ near_x_post + near_factory + post + C(age) + female', ages517)
print("\n(2) + Individual controls:")
print_coef(m2, 'near_x_post', 'Near×Post')
print_coef(m2, 'near_factory', 'Near Factory')
print_coef(m2, 'post', 'Post')
print(f"  N={m2.nobs:.0f}, R²={m2.rsquared:.4f}")

# M3: + Household controls
m3 = cluster_ols('child_labor ~ near_x_post + near_factory + post + C(age) + female + hhsize + wealth_num', ages517)
print("\n(3) + Household controls:")
print_coef(m3, 'near_x_post', 'Near×Post')
print_coef(m3, 'near_factory', 'Near Factory')
print_coef(m3, 'post', 'Post')
print(f"  N={m3.nobs:.0f}, R²={m3.rsquared:.4f}")

# M4: + Year FE (absorbs post)
m4 = cluster_ols('child_labor ~ near_x_post + near_factory + C(year) + C(age) + female + hhsize + wealth_num', ages517)
print("\n(4) + Year FE:")
print_coef(m4, 'near_x_post', 'Near×Post')
print_coef(m4, 'near_factory', 'Near Factory')
print(f"  N={m4.nobs:.0f}, R²={m4.rsquared:.4f}")

# ============================================================================
print("\n" + "=" * 70)
print("TABLE 2: AGE HETEROGENEITY (baseline spec)")
print("=" * 70)

for label, lo, hi in [("All 5-17", 5, 17), ("Ages 5-9", 5, 9), ("Ages 10-13", 10, 13), ("Ages 14-17", 14, 17)]:
    sub = df[(df['age'] >= lo) & (df['age'] <= hi)].copy()
    sub = sub.dropna(subset=['child_labor', 'near_factory', 'post', 'near_x_post'])
    mod = cluster_ols('child_labor ~ near_x_post + near_factory + post + C(age) + female + hhsize + wealth_num', sub)
    b, se, p = mod.params['near_x_post'], mod.bse['near_x_post'], mod.pvalues['near_x_post']
    mean_cl = sub['child_labor'].mean()
    mean_treat = sub[sub['near_factory']==1]['child_labor'].mean()
    print(f"\n{label}: Near×Post = {b:.4f}{stars(p)} (SE={se:.4f})")
    print(f"  N={mod.nobs:.0f}, Mean CL={mean_cl:.4f}, Treatment mean={mean_treat:.4f}")
    if mean_treat > 0:
        print(f"  Percent effect: {b/mean_treat*100:.1f}%")

# ============================================================================
print("\n" + "=" * 70)
print("TABLE 3: EVENT STUDY (ages 10-13, ref=2007)")
print("=" * 70)

ages1013 = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
ages1013 = ages1013.dropna(subset=['child_labor', 'near_factory', 'district'])

# Create year dummies and interactions
for y in [2000, 2004, 2011, 2014, 2018]:
    ages1013[f'yr_{y}'] = (ages1013['year'] == y).astype(int)
    ages1013[f'near_yr_{y}'] = ages1013['near_factory'] * ages1013[f'yr_{y}']

formula_es = ('child_labor ~ near_factory + '
              'yr_2000 + yr_2004 + yr_2011 + yr_2014 + yr_2018 + '
              'near_yr_2000 + near_yr_2004 + near_yr_2011 + near_yr_2014 + near_yr_2018 + '
              'C(age) + female + hhsize + wealth_num')
mod_es = cluster_ols(formula_es, ages1013)
print(f"\nEvent study (N={mod_es.nobs:.0f}):")
for y in [2000, 2004, 2011, 2014, 2018]:
    v = f'near_yr_{y}'
    if v in mod_es.params:
        b, se, p = mod_es.params[v], mod_es.bse[v], mod_es.pvalues[v]
        print(f"  {y}: {b:.4f}{stars(p)} (SE={se:.4f})")

# ============================================================================
print("\n" + "=" * 70)
print("TABLE 4: TREND-ADJUSTED DiD (ages 10-13)")
print("=" * 70)

ages1013['time'] = ages1013['year'] - 2013
ages1013['time_sq'] = ages1013['time'] ** 2
ages1013['time_cu'] = ages1013['time'] ** 3
ages1013['near_x_time'] = ages1013['near_factory'] * ages1013['time']
ages1013['near_x_time_sq'] = ages1013['near_factory'] * ages1013['time_sq']
ages1013['near_x_time_cu'] = ages1013['near_factory'] * ages1013['time_cu']
ages1013['near_x_post'] = ages1013['near_factory'] * ages1013['post']

base_controls = 'C(age) + female + hhsize + wealth_num'

# (1) Standard DiD
m_std = cluster_ols(f'child_labor ~ near_x_post + near_factory + post + {base_controls}', ages1013)
print(f"\n(1) Standard DiD:")
print_coef(m_std, 'near_x_post', 'Near×Post')
print(f"  N={m_std.nobs:.0f}")

# (2) + Common linear trend
m_ct = cluster_ols(f'child_labor ~ near_x_post + near_factory + post + time + {base_controls}', ages1013)
print(f"\n(2) + Common trend:")
print_coef(m_ct, 'near_x_post', 'Near×Post')

# (3) + Group-specific linear trend
m_gt = cluster_ols(f'child_labor ~ near_x_post + near_factory + post + time + near_x_time + {base_controls}', ages1013)
print(f"\n(3) + Group-specific linear trend:")
print_coef(m_gt, 'near_x_post', 'Near×Post')
print_coef(m_gt, 'near_x_time', 'Near×Time (pre-trend)')

# (4) + Group-specific quadratic
m_gq = cluster_ols(f'child_labor ~ near_x_post + near_factory + post + time + time_sq + near_x_time + near_x_time_sq + {base_controls}', ages1013)
print(f"\n(4) + Group-specific quadratic:")
print_coef(m_gq, 'near_x_post', 'Near×Post')

# (5) + Group-specific cubic
m_gc = cluster_ols(f'child_labor ~ near_x_post + near_factory + post + time + time_sq + time_cu + near_x_time + near_x_time_sq + near_x_time_cu + {base_controls}', ages1013)
print(f"\n(5) + Group-specific cubic:")
print_coef(m_gc, 'near_x_post', 'Near×Post')

# ============================================================================
print("\n" + "=" * 70)
print("TABLE 5: HONEST DiD - Reference year 2011 (ages 10-13)")
print("=" * 70)

# Event study with ref=2011
for y in [2000, 2004, 2007, 2014, 2018]:
    ages1013[f'yr11_{y}'] = (ages1013['year'] == y).astype(int)
    ages1013[f'near_yr11_{y}'] = ages1013['near_factory'] * ages1013[f'yr11_{y}']

formula_es11 = ('child_labor ~ near_factory + '
                'yr11_2000 + yr11_2004 + yr11_2007 + yr11_2014 + yr11_2018 + '
                'near_yr11_2000 + near_yr11_2004 + near_yr11_2007 + near_yr11_2014 + near_yr11_2018 + '
                'C(age) + female + hhsize + wealth_num')
mod_es11 = cluster_ols(formula_es11, ages1013)
print(f"\nEvent study ref=2011 (N={mod_es11.nobs:.0f}):")
for y in [2000, 2004, 2007, 2014, 2018]:
    v = f'near_yr11_{y}'
    if v in mod_es11.params:
        b, se, p = mod_es11.params[v], mod_es11.bse[v], mod_es11.pvalues[v]
        print(f"  {y}: {b:.4f}{stars(p)} (SE={se:.4f})")

# Key: 2014 coeff (the treatment effect relative to 2011)
b14 = mod_es11.params.get('near_yr11_2014', np.nan)
se14 = mod_es11.bse.get('near_yr11_2014', np.nan)
print(f"\n  Point estimate (2014, ref=2011): {b14:.4f}")
print(f"  SE: {se14:.4f}")
print(f"  95% CI: [{b14-1.96*se14:.4f}, {b14+1.96*se14:.4f}]")

# Max pre-trend coeff
pre_coefs = []
for y in [2000, 2004, 2007]:
    v = f'near_yr11_{y}'
    if v in mod_es11.params:
        pre_coefs.append(abs(mod_es11.params[v]))
max_pre = max(pre_coefs) if pre_coefs else 0
print(f"  Max pre-trend coefficient: {max_pre:.4f}")

# Breakdown M*
if abs(b14) > 1.96 * se14:
    print(f"  Breakdown M*: >0 (effect is significant at M=0)")
else:
    print(f"  Breakdown M*: 0.00 (already insignificant at M=0)")

# ============================================================================
print("\n" + "=" * 70)
print("TABLE 6: MATCHING & CALLAWAY-SANT'ANNA (ages 10-13)")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

match_df = ages1013.dropna(subset=['wealth_num', 'hhsize', 'female', 'child_labor', 'near_factory', 'post', 'district']).copy()

# IPW
X_match = match_df[['wealth_num', 'hhsize', 'female']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_match)
y_treat = match_df['near_factory'].values

lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, y_treat)
ps = lr.predict_proba(X_scaled)[:, 1]
match_df['pscore'] = ps

# Trim
trimmed = match_df[(match_df['pscore'] > 0.05) & (match_df['pscore'] < 0.95)].copy()
treated = trimmed['near_factory']
trimmed['ipw'] = np.where(treated == 1, 1/trimmed['pscore'], 1/(1-trimmed['pscore']))
trimmed['ipw'] = trimmed['ipw'] / trimmed['ipw'].mean()

mod_ipw = cluster_ols('child_labor ~ near_x_post + near_factory + post + C(age) + female + hhsize + wealth_num', trimmed)
print(f"\nIPW-Weighted DiD:")
print_coef(mod_ipw, 'near_x_post', 'Near×Post')
print(f"  N={mod_ipw.nobs:.0f}")

# NN matching
treat_idx = match_df[match_df['near_factory']==1].index
ctrl_idx = match_df[match_df['near_factory']==0].index
X_treat = scaler.transform(match_df.loc[treat_idx, ['wealth_num', 'hhsize', 'female']].values)
X_ctrl = scaler.transform(match_df.loc[ctrl_idx, ['wealth_num', 'hhsize', 'female']].values)

nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
nn.fit(X_ctrl)
distances, indices = nn.kneighbors(X_treat)
matched_ctrl_idx = ctrl_idx[indices.flatten()]
matched_df = pd.concat([match_df.loc[treat_idx], match_df.loc[matched_ctrl_idx]]).copy()
matched_df['near_x_post'] = matched_df['near_factory'] * matched_df['post']

mod_nn = cluster_ols('child_labor ~ near_x_post + near_factory + post + C(age) + female + hhsize + wealth_num', matched_df)
print(f"\nNN-Matched DiD:")
print_coef(mod_nn, 'near_x_post', 'Near×Post')
print(f"  N={mod_nn.nobs:.0f}")

# Callaway-Sant'Anna style: ATT by base period
print(f"\nCallaway-Sant'Anna style ATT by base period:")
for base_yr in [2000, 2004, 2007, 2011]:
    cs_df = ages1013[ages1013['year'].isin([base_yr, 2014])].copy()
    cs_df = cs_df.dropna(subset=['child_labor', 'near_factory', 'district'])
    cs_df['post_cs'] = (cs_df['year'] == 2014).astype(int)
    cs_df['near_x_post_cs'] = cs_df['near_factory'] * cs_df['post_cs']
    try:
        mod_cs = cluster_ols('child_labor ~ near_x_post_cs + near_factory + post_cs + C(age) + female + hhsize + wealth_num', cs_df)
        b, se, p = mod_cs.params['near_x_post_cs'], mod_cs.bse['near_x_post_cs'], mod_cs.pvalues['near_x_post_cs']
        print(f"  Base={base_yr}: {b:.4f}{stars(p)} (SE={se:.4f}, N={mod_cs.nobs:.0f})")
    except Exception as e:
        print(f"  Base={base_yr}: ERROR - {e}")

# Aggregate ATT (simple average of base period ATTs)
att_list = []
for base_yr in [2000, 2004, 2007, 2011]:
    cs_df = ages1013[ages1013['year'].isin([base_yr, 2014])].copy()
    cs_df = cs_df.dropna(subset=['child_labor', 'near_factory', 'district'])
    cs_df['post_cs'] = (cs_df['year'] == 2014).astype(int)
    cs_df['near_x_post_cs'] = cs_df['near_factory'] * cs_df['post_cs']
    try:
        mod_cs = cluster_ols('child_labor ~ near_x_post_cs + near_factory + post_cs + C(age) + female + hhsize + wealth_num', cs_df)
        att_list.append(mod_cs.params['near_x_post_cs'])
    except:
        pass
if att_list:
    agg_att = np.mean(att_list)
    print(f"  Aggregate ATT (mean): {agg_att:.4f}")

# ============================================================================
print("\n" + "=" * 70)
print("TABLE 7: TRIPLE DIFFERENCES (ages 10-17)")
print("=" * 70)

ages1017 = df[(df['age'] >= 10) & (df['age'] <= 17)].copy()
ages1017 = ages1017.dropna(subset=['child_labor', 'near_factory', 'post', 'district'])
ages1017['factory_age'] = (ages1017['age'] <= 13).astype(int)
ages1017['near_x_post'] = ages1017['near_factory'] * ages1017['post']
ages1017['near_x_fa'] = ages1017['near_factory'] * ages1017['factory_age']
ages1017['post_x_fa'] = ages1017['post'] * ages1017['factory_age']
ages1017['triple'] = ages1017['near_factory'] * ages1017['post'] * ages1017['factory_age']

# Time variables for DDD
ages1017['time'] = ages1017['year'] - 2013
ages1017['near_x_time'] = ages1017['near_factory'] * ages1017['time']
ages1017['fa_x_time'] = ages1017['factory_age'] * ages1017['time']
ages1017['near_fa_time'] = ages1017['near_factory'] * ages1017['factory_age'] * ages1017['time']

# DDD baseline
mod_ddd = cluster_ols('child_labor ~ near_x_post + near_factory + post + factory_age + near_x_fa + post_x_fa + triple + C(age) + female + hhsize + wealth_num', ages1017)
print(f"\n(1) Triple diff:")
print_coef(mod_ddd, 'near_x_post', 'Near×Post (14-17 effect)')
print_coef(mod_ddd, 'triple', 'Near×Post×FactoryAge (DDD)')
b_ddd_14_17 = mod_ddd.params['near_x_post']
b_ddd_triple = mod_ddd.params['triple']
print(f"  Effect on ages 10-13: {b_ddd_14_17 + b_ddd_triple:.4f}")
print(f"  N={mod_ddd.nobs:.0f}")

# DDD + trends
mod_ddd_t = cluster_ols('child_labor ~ near_x_post + near_factory + post + factory_age + near_x_fa + post_x_fa + triple + time + near_x_time + fa_x_time + near_fa_time + C(age) + female + hhsize + wealth_num', ages1017)
print(f"\n(2) Triple diff + trends:")
print_coef(mod_ddd_t, 'near_x_post', 'Near×Post (14-17 effect)')
print_coef(mod_ddd_t, 'triple', 'Near×Post×FactoryAge (DDD)')
print_coef(mod_ddd_t, 'near_fa_time', 'Near×FactoryAge×Time')
b2_14_17 = mod_ddd_t.params['near_x_post']
b2_triple = mod_ddd_t.params['triple']
print(f"  Effect on ages 10-13: {b2_14_17 + b2_triple:.4f}")
print(f"  N={mod_ddd_t.nobs:.0f}")

# ============================================================================
print("\n" + "=" * 70)
print("DISTANCE ROBUSTNESS (ages 10-13)")
print("=" * 70)

ages1013_full = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
ages1013_full = ages1013_full.dropna(subset=['child_labor', 'district'])

for km in [2, 5, 7, 10, 15, 20, 25, 30]:
    col = f'within_{km}km'
    xcol = f'{col}_x_post'
    if col in ages1013_full.columns and xcol in ages1013_full.columns:
        sub = ages1013_full.dropna(subset=[col, xcol])
        treat_pct = sub[col].mean() * 100
        try:
            mod = cluster_ols(f'child_labor ~ {xcol} + {col} + post + C(age) + female + hhsize + wealth_num', sub)
            b, se, p = mod.params[xcol], mod.bse[xcol], mod.pvalues[xcol]
            print(f"  {km}km: {b:.4f}{stars(p)} (SE={se:.4f}) [treat={treat_pct:.1f}%]")
        except Exception as e:
            print(f"  {km}km: ERROR - {e}")

# ============================================================================
print("\n" + "=" * 70)
print("SCHOOL ENROLLMENT (ages 10-13)")
print("=" * 70)

if 'in_school' in ages1013.columns:
    school_df = ages1013.dropna(subset=['in_school', 'near_factory', 'district']).copy()
    # Convert in_school to numeric
    if school_df['in_school'].dtype == object:
        school_df['enrolled'] = school_df['in_school'].map({'Yes': 1, 'No': 0})
    else:
        school_df['enrolled'] = school_df['in_school']
    school_df = school_df.dropna(subset=['enrolled'])

    mod_school = cluster_ols('enrolled ~ near_x_post + near_factory + post + C(age) + female + hhsize + wealth_num', school_df)
    print(f"\nSchool enrollment:")
    print_coef(mod_school, 'near_x_post', 'Near×Post')
    print(f"  N={mod_school.nobs:.0f}, Mean enrollment={school_df['enrolled'].mean():.4f}")

print("\n" + "=" * 70)
print("DONE — All coefficients generated")
print("=" * 70)
