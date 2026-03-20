"""
Alternative analyses to make the null-result paper publishable.
Produces tables, figures, and CSVs for:
  1. Multiple outcomes (school enrollment, education, work formality)
  2. Heterogeneous treatment effects (wealth, gender, urban/rural)
  3. Dose-response / continuous treatment intensity
  4. Minimum detectable effect / power analysis
  5. Equivalence testing (TOST procedure)
  6. Multi-outcome event study figure
  7. Decomposition of where child labor occurs
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os

warnings.filterwarnings('ignore')

BASE = "/Users/amalkova/Library/CloudStorage/OneDrive-FloridaInstituteofTechnology/_Research/Labor_Economics/Child Labor"
OUT = os.path.join(BASE, "results", "new_results")
os.makedirs(OUT, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare the analysis dataset."""
    df = pd.read_csv(os.path.join(BASE, "DHS", "final_analysis_data.csv"),
                     low_memory=False)

    # Treatment variables (recompute interaction — CSV may have stale version)
    df['within_10km'] = df['within_10km'].astype(float)
    df['near_factory'] = df['within_10km']
    df['post'] = df['post_rana']
    df['near_x_post'] = df['near_factory'] * df['post']

    # Female dummy
    df['female'] = (df['sex'] == 'Female').astype(int)

    # Wealth numeric
    wealth_map = {'Poorest': 1, 'Poorer': 2, 'Middle': 3, 'Richer': 4, 'Richest': 5}
    df['wealth_num'] = df['wealthqhh'].map(wealth_map)

    # District for clustering
    df['district'] = df['geolev2']

    # School enrollment (binary)
    df['enrolled'] = np.where(df['in_school'] == 'Yes', 1,
                     np.where(df['in_school'] == 'No', 0, np.nan))

    # Education years (numeric)
    df['edyears_num'] = pd.to_numeric(df['edyears'], errors='coerce')

    # Currently working (binary from hhcurrwork)
    df['working'] = np.where(df['hhcurrwork'] == 'Yes', 1,
                    np.where(df['hhcurrwork'] == 'No', 0, np.nan))

    # Paid work (binary from hhcurrworkpay — Cash or Both)
    df['paid_work'] = np.where(df['hhcurrworkpay'].isin(['Cash', 'Both cash and in-kind']), 1,
                      np.where(df['hhcurrworkpay'].isin(['In-kind', 'NIU (not in universe)']), 0, np.nan))
    # For hhcurrworkpay, NIU means not working, so paid_work=0
    # But only define paid_work for those with valid hhcurrwork data
    df.loc[df['hhcurrwork'] == 'NIU (not in universe)', 'paid_work'] = np.nan

    # Urban dummy
    df['urban'] = (df['urban_combined'] == 'Urban').astype(int)

    # Exclude 2018 (geolev2 all NaN for that wave)
    df = df[df['year'] != 2018].copy()

    return df


def cluster_ols(formula, data, cluster_var='district'):
    """OLS with district-clustered standard errors."""
    d = data.dropna(subset=[cluster_var])
    mod = smf.ols(formula, data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d[cluster_var].astype(int)})
    return mod


def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


# =============================================================================
# ANALYSIS 1: MULTIPLE OUTCOMES
# =============================================================================

def analysis1_multiple_outcomes(df):
    """DiD on 4 outcomes: enrolled, edyears, working, paid_work."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: MULTIPLE OUTCOMES FRAMEWORK (ages 10-13)")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
    sub['near_x_post'] = sub['near_factory'] * sub['post']

    controls = 'C(age) + female + hhsize + wealth_num'
    outcomes = {
        'child_labor': ('Child Labor', 'child_labor'),
        'enrolled': ('School Enrollment', 'enrolled'),
        'edyears_num': ('Education Years', 'edyears_num'),
        'paid_work': ('Paid Work', 'paid_work'),
    }

    results = []
    for outcome_var, (label, col) in outcomes.items():
        d = sub.dropna(subset=[col, 'near_factory', 'post', 'district']).copy()
        d['near_x_post'] = d['near_factory'] * d['post']
        try:
            mod = cluster_ols(f'{col} ~ near_x_post + near_factory + post + {controls}', d)
            b = mod.params['near_x_post']
            se = mod.bse['near_x_post']
            p = mod.pvalues['near_x_post']
            mean_y = d[col].mean()
            n = int(mod.nobs)
            results.append({
                'Outcome': label, 'Coefficient': b, 'SE': se,
                'p_value': p, 'Mean_DV': mean_y, 'N': n
            })
            print(f"\n  {label}: {b:.4f}{stars(p)} (SE={se:.4f}, p={p:.3f})")
            print(f"    N={n:,}, Mean={mean_y:.4f}")
        except Exception as e:
            print(f"\n  {label}: ERROR - {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT, 'multiple_outcomes.csv'), index=False)

    # Generate LaTeX table
    latex = generate_multiple_outcomes_latex(results)
    with open(os.path.join(OUT, 'table_multiple_outcomes.tex'), 'w') as f:
        f.write(latex)
    print(f"\n  Saved: multiple_outcomes.csv, table_multiple_outcomes.tex")
    return results_df


def generate_multiple_outcomes_latex(results):
    """Generate LaTeX table for multiple outcomes."""
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Effect of Factory Monitoring on Multiple Outcomes: Ages 10-13}',
        r'\label{tab:multiple_outcomes}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{l' + 'c' * len(results) + '}',
        r'\toprule',
    ]
    # Header
    header = ' & '.join([''] + [f'({i+1})' for i in range(len(results))]) + r' \\'
    labels = ' & '.join([''] + [r['Outcome'] for r in results]) + r' \\'
    lines.extend([header, labels, r'\midrule'])

    # Coefficients
    coefs = ' & '.join(['Near Factory $\\times$ Post'] +
        [f"${r['Coefficient']:.4f}{stars(r['p_value'])}$" for r in results]) + r' \\'
    ses = ' & '.join([''] + [f"({r['SE']:.4f})" for r in results]) + r' \\'
    lines.extend([coefs, ses, r'\\'])

    # Footer stats
    lines.append(r'\midrule')
    obs = ' & '.join(['Observations'] + [f"{r['N']:,}" for r in results]) + r' \\'
    means = ' & '.join(['Mean of Dep.\\ Var.'] + [f"{r['Mean_DV']:.3f}" for r in results]) + r' \\'
    lines.extend([obs, means])

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: Each column presents DiD estimates from separate regressions. All specifications include age fixed effects, female indicator, household size, and wealth controls. Standard errors clustered at district level. Sample: ages 10--13. Significance: *** p$<$0.01, ** p$<$0.05, * p$<$0.10.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# =============================================================================
# ANALYSIS 2: HETEROGENEOUS TREATMENT EFFECTS
# =============================================================================

def analysis2_heterogeneity(df):
    """Subgroup analyses by wealth, gender, urban/rural."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: HETEROGENEOUS TREATMENT EFFECTS (ages 10-13)")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
    sub['near_x_post'] = sub['near_factory'] * sub['post']
    controls = 'C(age) + female + hhsize + wealth_num'

    all_results = []

    # --- By Wealth Quintile ---
    print("\n  --- By Wealth Quintile ---")
    wealth_results = []
    for wq in ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']:
        d = sub[sub['wealthqhh'] == wq].copy()
        d = d.dropna(subset=['child_labor', 'near_factory', 'post', 'district'])
        d['near_x_post'] = d['near_factory'] * d['post']
        try:
            mod = cluster_ols('child_labor ~ near_x_post + near_factory + post + C(age) + female + hhsize', d)
            b, se, p = mod.params['near_x_post'], mod.bse['near_x_post'], mod.pvalues['near_x_post']
            mean_cl = d['child_labor'].mean()
            n = int(mod.nobs)
            wealth_results.append({
                'Subgroup': wq, 'Outcome': 'Child Labor',
                'Coefficient': b, 'SE': se, 'p_value': p,
                'Mean_DV': mean_cl, 'N': n, 'Category': 'Wealth'
            })
            print(f"    {wq}: {b:.4f}{stars(p)} (SE={se:.4f}, N={n:,}, Mean CL={mean_cl:.4f})")
        except Exception as e:
            print(f"    {wq}: ERROR - {e}")

    # Also run school enrollment by wealth
    print("\n  --- School Enrollment by Wealth ---")
    for wq in ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']:
        d = sub[sub['wealthqhh'] == wq].copy()
        d = d.dropna(subset=['enrolled', 'near_factory', 'post', 'district'])
        d['near_x_post'] = d['near_factory'] * d['post']
        try:
            mod = cluster_ols('enrolled ~ near_x_post + near_factory + post + C(age) + female + hhsize', d)
            b, se, p = mod.params['near_x_post'], mod.bse['near_x_post'], mod.pvalues['near_x_post']
            mean_e = d['enrolled'].mean()
            n = int(mod.nobs)
            wealth_results.append({
                'Subgroup': wq, 'Outcome': 'Enrollment',
                'Coefficient': b, 'SE': se, 'p_value': p,
                'Mean_DV': mean_e, 'N': n, 'Category': 'Wealth'
            })
            print(f"    {wq}: {b:.4f}{stars(p)} (SE={se:.4f}, N={n:,}, Mean Enroll={mean_e:.4f})")
        except Exception as e:
            print(f"    {wq}: ERROR - {e}")

    all_results.extend(wealth_results)

    # --- By Gender ---
    print("\n  --- By Gender ---")
    gender_results = []
    for sex_val, sex_label in [('Male', 'Boys'), ('Female', 'Girls')]:
        d = sub[sub['sex'] == sex_val].copy()
        d = d.dropna(subset=['child_labor', 'near_factory', 'post', 'district'])
        d['near_x_post'] = d['near_factory'] * d['post']
        try:
            mod = cluster_ols('child_labor ~ near_x_post + near_factory + post + C(age) + hhsize + wealth_num', d)
            b, se, p = mod.params['near_x_post'], mod.bse['near_x_post'], mod.pvalues['near_x_post']
            mean_cl = d['child_labor'].mean()
            n = int(mod.nobs)
            gender_results.append({
                'Subgroup': sex_label, 'Outcome': 'Child Labor',
                'Coefficient': b, 'SE': se, 'p_value': p,
                'Mean_DV': mean_cl, 'N': n, 'Category': 'Gender'
            })
            print(f"    {sex_label}: {b:.4f}{stars(p)} (SE={se:.4f}, N={n:,}, Mean CL={mean_cl:.4f})")
        except Exception as e:
            print(f"    {sex_label}: ERROR - {e}")
    all_results.extend(gender_results)

    # --- By Urban/Rural ---
    print("\n  --- By Urban/Rural ---")
    urban_results = []
    for ur_val, ur_label in [('Urban', 'Urban'), ('Rural', 'Rural')]:
        d = sub[sub['urban_combined'] == ur_val].copy()
        d = d.dropna(subset=['child_labor', 'near_factory', 'post', 'district'])
        d['near_x_post'] = d['near_factory'] * d['post']
        try:
            mod = cluster_ols('child_labor ~ near_x_post + near_factory + post + C(age) + female + hhsize + wealth_num', d)
            b, se, p = mod.params['near_x_post'], mod.bse['near_x_post'], mod.pvalues['near_x_post']
            mean_cl = d['child_labor'].mean()
            n = int(mod.nobs)
            urban_results.append({
                'Subgroup': ur_label, 'Outcome': 'Child Labor',
                'Coefficient': b, 'SE': se, 'p_value': p,
                'Mean_DV': mean_cl, 'N': n, 'Category': 'Urban/Rural'
            })
            print(f"    {ur_label}: {b:.4f}{stars(p)} (SE={se:.4f}, N={n:,}, Mean CL={mean_cl:.4f})")
        except Exception as e:
            print(f"    {ur_label}: ERROR - {e}")
    all_results.extend(urban_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUT, 'heterogeneity_results.csv'), index=False)

    # Generate LaTeX tables
    latex_wealth = generate_wealth_hetero_latex(wealth_results)
    with open(os.path.join(OUT, 'table_wealth_heterogeneity.tex'), 'w') as f:
        f.write(latex_wealth)

    latex_demo = generate_demo_hetero_latex(gender_results, urban_results)
    with open(os.path.join(OUT, 'table_demographic_heterogeneity.tex'), 'w') as f:
        f.write(latex_demo)

    print(f"\n  Saved: heterogeneity_results.csv, table_wealth_heterogeneity.tex, table_demographic_heterogeneity.tex")
    return results_df


def generate_wealth_hetero_latex(results):
    """LaTeX table: wealth quintile heterogeneity."""
    cl_results = [r for r in results if r['Outcome'] == 'Child Labor']
    en_results = [r for r in results if r['Outcome'] == 'Enrollment']

    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Heterogeneous Effects by Wealth Quintile: Ages 10-13}',
        r'\label{tab:wealth_hetero}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'& \multicolumn{3}{c}{Child Labor} & \multicolumn{3}{c}{School Enrollment} \\',
        r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}',
        r'Wealth Quintile & Coeff. & SE & Mean & Coeff. & SE & Mean \\',
        r'\midrule',
    ]
    for cl, en in zip(cl_results, en_results):
        line = (f"{cl['Subgroup']} & ${cl['Coefficient']:.4f}{stars(cl['p_value'])}$ & "
                f"({cl['SE']:.4f}) & {cl['Mean_DV']:.3f} & "
                f"${en['Coefficient']:.4f}{stars(en['p_value'])}$ & "
                f"({en['SE']:.4f}) & {en['Mean_DV']:.3f} \\\\")
        lines.append(line)

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: Each cell presents DiD estimates from separate regressions within each wealth quintile. Controls: age FE, female, household size. Standard errors clustered at district level. Significance: *** p$<$0.01, ** p$<$0.05, * p$<$0.10.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


def generate_demo_hetero_latex(gender_results, urban_results):
    """LaTeX table: gender and urban/rural heterogeneity."""
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Heterogeneous Effects by Gender and Location: Ages 10-13}',
        r'\label{tab:demo_hetero}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Subgroup & Coefficient & SE & Mean CL & N \\',
        r'\midrule',
        r'\multicolumn{5}{l}{\textit{Panel A: By Gender}} \\',
    ]
    for r in gender_results:
        lines.append(f"\\quad {r['Subgroup']} & ${r['Coefficient']:.4f}{stars(r['p_value'])}$ & "
                     f"({r['SE']:.4f}) & {r['Mean_DV']:.3f} & {r['N']:,} \\\\")
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{5}{l}{\textit{Panel B: By Urban/Rural}} \\')
    for r in urban_results:
        lines.append(f"\\quad {r['Subgroup']} & ${r['Coefficient']:.4f}{stars(r['p_value'])}$ & "
                     f"({r['SE']:.4f}) & {r['Mean_DV']:.3f} & {r['N']:,} \\\\")
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: Each row presents DiD estimates from separate regressions by subgroup. Controls: age FE, household size, wealth. Standard errors clustered at district level. Significance: *** p$<$0.01, ** p$<$0.05, * p$<$0.10.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# =============================================================================
# ANALYSIS 3: DOSE-RESPONSE / CONTINUOUS TREATMENT
# =============================================================================

def analysis3_dose_response(df):
    """Continuous treatment: proximity score, log distance, factory count."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: DOSE-RESPONSE / CONTINUOUS TREATMENT (ages 10-13)")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
    controls = 'C(age) + female + hhsize + wealth_num'

    results = []

    # (1) Proximity score × Post
    sub['prox_x_post'] = sub['proximity_score'] * sub['post']
    d = sub.dropna(subset=['child_labor', 'proximity_score', 'post', 'district']).copy()
    mod1 = cluster_ols(f'child_labor ~ prox_x_post + proximity_score + post + {controls}', d)
    b, se, p = mod1.params['prox_x_post'], mod1.bse['prox_x_post'], mod1.pvalues['prox_x_post']
    results.append({'Treatment': 'Proximity Score', 'Coefficient': b, 'SE': se,
                    'p_value': p, 'N': int(mod1.nobs)})
    print(f"\n  Proximity Score × Post: {b:.4f}{stars(p)} (SE={se:.4f})")

    # (2) Log distance × Post
    sub['logdist_x_post'] = sub['log_distance'] * sub['post']
    d = sub.dropna(subset=['child_labor', 'log_distance', 'post', 'district']).copy()
    mod2 = cluster_ols(f'child_labor ~ logdist_x_post + log_distance + post + {controls}', d)
    b, se, p = mod2.params['logdist_x_post'], mod2.bse['logdist_x_post'], mod2.pvalues['logdist_x_post']
    results.append({'Treatment': 'Log Distance', 'Coefficient': b, 'SE': se,
                    'p_value': p, 'N': int(mod2.nobs)})
    print(f"  Log Distance × Post: {b:.4f}{stars(p)} (SE={se:.4f})")

    # (3) Factory count within 10km
    # Compute from nearest, second, third factory distances
    sub['n_factories_10km'] = (
        (sub['nearest_factory_km'] <= 10).astype(int) +
        (sub['second_nearest_km'] <= 10).astype(int) +
        (sub['third_nearest_km'] <= 10).astype(int)
    )
    sub['nfact_x_post'] = sub['n_factories_10km'] * sub['post']
    d = sub.dropna(subset=['child_labor', 'n_factories_10km', 'post', 'district']).copy()
    mod3 = cluster_ols(f'child_labor ~ nfact_x_post + n_factories_10km + post + {controls}', d)
    b, se, p = mod3.params['nfact_x_post'], mod3.bse['nfact_x_post'], mod3.pvalues['nfact_x_post']
    results.append({'Treatment': 'Factory Count (10km)', 'Coefficient': b, 'SE': se,
                    'p_value': p, 'N': int(mod3.nobs)})
    print(f"  Factory Count × Post: {b:.4f}{stars(p)} (SE={se:.4f})")
    print(f"    Factory count distribution: {sub['n_factories_10km'].value_counts().sort_index().to_dict()}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT, 'dose_response_results.csv'), index=False)

    # LaTeX table
    latex = generate_dose_response_latex(results)
    with open(os.path.join(OUT, 'table_dose_response.tex'), 'w') as f:
        f.write(latex)
    print(f"\n  Saved: dose_response_results.csv, table_dose_response.tex")
    return results_df


def generate_dose_response_latex(results):
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Dose-Response Analysis: Continuous Treatment Intensity}',
        r'\label{tab:dose_response}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{l' + 'c' * len(results) + '}',
        r'\toprule',
    ]
    header = ' & '.join([''] + [f'({i+1})' for i in range(len(results))]) + r' \\'
    labels = ' & '.join([''] + [r['Treatment'] for r in results]) + r' \\'
    lines.extend([header, labels, r'\midrule'])

    coefs = ' & '.join(['Treatment $\\times$ Post'] +
        [f"${r['Coefficient']:.4f}{stars(r['p_value'])}$" for r in results]) + r' \\'
    ses = ' & '.join([''] + [f"({r['SE']:.4f})" for r in results]) + r' \\'
    lines.extend([coefs, ses, r'\\', r'\midrule'])
    obs = ' & '.join(['Observations'] + [f"{r['N']:,}" for r in results]) + r' \\'
    lines.append(obs)

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: Column 1 uses proximity score ($1/(1+\text{distance})$). Column 2 uses log distance to nearest factory. Column 3 uses count of factories within 10km (based on distances to nearest, second-nearest, and third-nearest factories). All specifications include age FE, female, household size, and wealth controls. Standard errors clustered at district level. Ages 10--13. Significance: *** p$<$0.01, ** p$<$0.05, * p$<$0.10.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# =============================================================================
# ANALYSIS 4: POWER ANALYSIS / MINIMUM DETECTABLE EFFECT
# =============================================================================

def analysis4_power_mde(df):
    """Compute MDE and generate power curve."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: POWER ANALYSIS / MINIMUM DETECTABLE EFFECT")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
    sub['near_x_post'] = sub['near_factory'] * sub['post']
    controls = 'C(age) + female + hhsize + wealth_num'
    d = sub.dropna(subset=['child_labor', 'near_factory', 'post', 'district']).copy()

    mod = cluster_ols(f'child_labor ~ near_x_post + near_factory + post + {controls}', d)
    se_hat = mod.bse['near_x_post']
    beta_hat = mod.params['near_x_post']
    mean_cl = d['child_labor'].mean()
    n_obs = int(mod.nobs)

    # MDE at 80% power, two-sided alpha=0.05
    # MDE = (z_alpha/2 + z_beta) * SE = (1.96 + 0.84) * SE = 2.80 * SE
    z_alpha = 1.96
    z_beta = 0.842  # 80% power
    mde_80 = (z_alpha + z_beta) * se_hat

    # MDE at 90% power
    z_beta_90 = 1.282
    mde_90 = (z_alpha + z_beta_90) * se_hat

    print(f"\n  SE of treatment effect: {se_hat:.4f}")
    print(f"  Point estimate: {beta_hat:.4f}")
    print(f"  Baseline child labor rate: {mean_cl:.4f} ({mean_cl*100:.1f}%)")
    print(f"  N = {n_obs:,}")
    print(f"\n  MDE at 80% power: {mde_80:.4f} ({mde_80*100:.2f} pp)")
    print(f"  MDE at 90% power: {mde_90:.4f} ({mde_90*100:.2f} pp)")
    print(f"  MDE as % of baseline: {mde_80/mean_cl*100:.1f}%")

    # Benchmark effects from literature
    benchmarks = [
        ('CCT programs (Dehoop \\& Rosati 2014)', 0.05, 0.10),
        ('Vietnam rice price (Edmonds 2005)', 0.06, 0.06),
        ('India trade (Edmonds 2010)', 0.03, 0.03),
        ('School construction (Ravallion 2000)', 0.03, 0.05),
    ]
    print("\n  Benchmark effects from literature:")
    for name, lo, hi in benchmarks:
        if lo == hi:
            detectable = "Yes" if lo > mde_80 else "No"
            print(f"    {name}: {lo*100:.0f} pp — Detectable: {detectable}")
        else:
            print(f"    {name}: {lo*100:.0f}-{hi*100:.0f} pp")

    # Power curve
    effect_sizes = np.linspace(0, 0.05, 200)
    power_vals = []
    for effect in effect_sizes:
        if se_hat > 0:
            ncp = effect / se_hat  # noncentrality parameter
            power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
        else:
            power = 0
        power_vals.append(power)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(effect_sizes * 100, power_vals, 'b-', linewidth=2)
    ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='80% power')
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% power')
    ax.axvline(x=mde_80 * 100, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=mde_90 * 100, color='orange', linestyle=':', alpha=0.5)

    # Mark MDE
    ax.annotate(f'MDE (80%) = {mde_80*100:.1f} pp',
                xy=(mde_80*100, 0.80), xytext=(mde_80*100 + 0.3, 0.65),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    # Mark baseline
    ax.axvline(x=mean_cl * 100, color='green', linestyle='-.', alpha=0.5,
               label=f'Baseline CL rate ({mean_cl*100:.1f}%)')

    ax.set_xlabel('True Effect Size (percentage points)', fontsize=12)
    ax.set_ylabel('Statistical Power', fontsize=12)
    ax.set_title('Power Curve: Detectable Effects on Child Labor', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'power_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    power_results = {
        'se': se_hat, 'beta': beta_hat, 'mean_cl': mean_cl, 'n': n_obs,
        'mde_80': mde_80, 'mde_90': mde_90, 'mde_pct_baseline': mde_80 / mean_cl
    }

    # LaTeX table comparing MDE to benchmarks
    latex = generate_power_latex(power_results, benchmarks, mde_80)
    with open(os.path.join(OUT, 'table_power_analysis.tex'), 'w') as f:
        f.write(latex)

    print(f"\n  Saved: power_curve.png, table_power_analysis.tex")
    return power_results


def generate_power_latex(results, benchmarks, mde):
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Minimum Detectable Effect and Comparison to Literature}',
        r'\label{tab:power_analysis}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'& Effect Size (pp) & Detectable? \\',
        r'\midrule',
        r'\multicolumn{3}{l}{\textit{Panel A: Study Parameters}} \\',
        f"\\quad Standard error of $\\hat{{\\beta}}_{{DiD}}$ & {results['se']*100:.2f} & \\\\",
        f"\\quad MDE at 80\\% power & {results['mde_80']*100:.2f} & \\\\",
        f"\\quad MDE at 90\\% power & {results['mde_90']*100:.2f} & \\\\",
        f"\\quad Baseline child labor rate & {results['mean_cl']*100:.1f} & \\\\",
        f"\\quad MDE as \\% of baseline & {results['mde_pct_baseline']*100:.0f}\\% & \\\\",
        r'\\',
        r'\midrule',
        r'\multicolumn{3}{l}{\textit{Panel B: Benchmark Effects from Literature}} \\',
    ]
    for name, lo, hi in benchmarks:
        effect_str = f"{lo*100:.0f}--{hi*100:.0f}" if lo != hi else f"{lo*100:.0f}"
        detectable = "Yes" if hi > mde else "No"
        lines.append(f"\\quad {name} & {effect_str} & {detectable} \\\\")

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: MDE computed as $(z_{\alpha/2} + z_\beta) \times SE$. Two-sided test at $\alpha = 0.05$. ``Detectable'' indicates whether the upper bound of the benchmark effect exceeds the MDE at 80\% power.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# =============================================================================
# ANALYSIS 5: EQUIVALENCE TESTING (TOST)
# =============================================================================

def analysis5_equivalence_test(df):
    """Two One-Sided Tests (TOST) for equivalence."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: EQUIVALENCE TESTING (TOST PROCEDURE)")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
    sub['near_x_post'] = sub['near_factory'] * sub['post']
    controls = 'C(age) + female + hhsize + wealth_num'
    d = sub.dropna(subset=['child_labor', 'near_factory', 'post', 'district']).copy()

    mod = cluster_ols(f'child_labor ~ near_x_post + near_factory + post + {controls}', d)
    beta = mod.params['near_x_post']
    se = mod.bse['near_x_post']
    n_clusters = d['district'].nunique()

    # Use t-distribution with n_clusters - 1 df for cluster-robust
    dof = n_clusters - 1

    deltas = [0.01, 0.02, 0.03, 0.05]  # equivalence bounds in pp (as proportions)
    results = []

    print(f"\n  Point estimate: {beta:.4f}")
    print(f"  SE: {se:.4f}")
    print(f"  Clusters: {n_clusters}, df={dof}")
    print(f"\n  {'Delta (pp)':>12} {'t_upper':>10} {'t_lower':>10} {'p_upper':>10} {'p_lower':>10} {'Reject?':>10}")
    print(f"  {'-'*62}")

    for delta in deltas:
        # Test H0: beta >= delta  (upper test)
        t_upper = (beta - delta) / se
        p_upper = stats.t.cdf(t_upper, dof)  # want this small (beta < delta)

        # Test H0: beta <= -delta (lower test)
        t_lower = (beta + delta) / se
        p_lower = 1 - stats.t.cdf(t_lower, dof)  # want this small (beta > -delta)

        # Equivalence: reject both one-sided tests
        p_tost = max(p_upper, p_lower)
        reject = p_tost < 0.05

        results.append({
            'Delta_pp': delta * 100, 'Delta': delta,
            't_upper': t_upper, 'p_upper': p_upper,
            't_lower': t_lower, 'p_lower': p_lower,
            'p_TOST': p_tost, 'Reject_H0': reject
        })
        print(f"  {delta*100:>10.0f} pp {t_upper:>10.3f} {t_lower:>10.3f} "
              f"{p_upper:>10.4f} {p_lower:>10.4f} {'YES' if reject else 'no':>10}")

    # Find smallest delta at which we reject
    smallest_reject = None
    for delta_test in np.arange(0.005, 0.10, 0.001):
        t_u = (beta - delta_test) / se
        p_u = stats.t.cdf(t_u, dof)
        t_l = (beta + delta_test) / se
        p_l = 1 - stats.t.cdf(t_l, dof)
        if max(p_u, p_l) < 0.05:
            smallest_reject = delta_test
            break

    if smallest_reject:
        print(f"\n  Smallest delta for equivalence rejection: {smallest_reject*100:.1f} pp")
    else:
        print(f"\n  Cannot reject equivalence at any tested delta <= 10 pp")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT, 'tost_results.csv'), index=False)

    # LaTeX table
    latex = generate_tost_latex(results, beta, se, smallest_reject)
    with open(os.path.join(OUT, 'table_tost.tex'), 'w') as f:
        f.write(latex)
    print(f"\n  Saved: tost_results.csv, table_tost.tex")
    return results_df


def generate_tost_latex(results, beta, se, smallest_reject):
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Equivalence Testing: Two One-Sided Tests (TOST) Procedure}',
        r'\label{tab:tost}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Equivalence Bound ($\pm\delta$) & $t$-stat (upper) & $t$-stat (lower) & $p$-value (TOST) & Reject $H_0$? \\',
        r'\midrule',
    ]
    for r in results:
        reject_str = "Yes" if r['Reject_H0'] else "No"
        lines.append(f"$\\pm${r['Delta_pp']:.0f} pp & {r['t_upper']:.3f} & "
                     f"{r['t_lower']:.3f} & {r['p_TOST']:.4f} & {reject_str} \\\\")

    lines.append(r'\midrule')
    lines.append(f"Point estimate: $\\hat{{\\beta}} = {beta:.4f}$, SE $= {se:.4f}$ & & & & \\\\")
    if smallest_reject:
        lines.append(f"Smallest equivalence bound rejected: $\\pm${smallest_reject*100:.1f} pp & & & & \\\\")

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: TOST procedure tests $H_0: |\beta| \geq \delta$ against $H_1: |\beta| < \delta$. ``Reject'' means we can conclude the true effect is smaller than $\pm\delta$ at 5\% significance. $p$-value (TOST) is the maximum of the two one-sided $p$-values. Degrees of freedom based on number of clusters minus one. Sample: ages 10--13.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# =============================================================================
# ANALYSIS 6: MULTI-OUTCOME EVENT STUDY FIGURE
# =============================================================================

def analysis6_event_study_multi(df):
    """Event study with multiple outcomes in a 3-panel figure."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: MULTI-OUTCOME EVENT STUDY (ages 10-13)")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()

    # Create year × treatment interactions (ref = 2011)
    years = [2000, 2004, 2007, 2014]  # omit 2011 as reference
    for y in years:
        sub[f'yr_{y}'] = (sub['year'] == y).astype(int)
        sub[f'near_yr_{y}'] = sub['near_factory'] * sub[f'yr_{y}']

    controls = 'C(age) + female + hhsize + wealth_num'
    es_vars = ' + '.join([f'yr_{y}' for y in years] + [f'near_yr_{y}' for y in years])

    outcomes = {
        'child_labor': 'Child Labor',
        'enrolled': 'School Enrollment',
        'working': 'Currently Working',
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    all_years = [2000, 2004, 2007, 2011, 2014]

    for idx, (outcome_var, label) in enumerate(outcomes.items()):
        ax = axes[idx]
        d = sub.dropna(subset=[outcome_var, 'near_factory', 'district']).copy()
        for y in years:
            d[f'near_yr_{y}'] = d['near_factory'] * d[f'yr_{y}']

        try:
            formula = f'{outcome_var} ~ near_factory + {es_vars} + {controls}'
            mod = cluster_ols(formula, d)

            coefs = [0.0]  # placeholder for years before ref
            ses_list = [0.0]
            plot_years = []
            for y in all_years:
                if y == 2011:
                    coefs.append(0.0)
                    ses_list.append(0.0)
                else:
                    v = f'near_yr_{y}'
                    if v in mod.params:
                        coefs.append(mod.params[v])
                        ses_list.append(mod.bse[v])
                    else:
                        coefs.append(np.nan)
                        ses_list.append(np.nan)
                plot_years.append(y)

            # Remove the initial placeholder
            coefs = coefs[1:]
            ses_list = ses_list[1:]

            coefs = np.array(coefs)
            ses_arr = np.array(ses_list)

            ax.errorbar(plot_years, coefs, yerr=1.96 * ses_arr,
                       fmt='o-', color='navy', capsize=4, markersize=6, linewidth=1.5)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(x=2013, color='red', linestyle='--', alpha=0.5, label='Rana Plaza')
            ax.fill_between([2012, 2015], ax.get_ylim()[0] - 0.1, ax.get_ylim()[1] + 0.1,
                           alpha=0.05, color='red')

            mean_y = d[outcome_var].mean()
            ax.set_title(f'{label}\n(Mean = {mean_y:.3f})', fontsize=11)
            ax.set_xlabel('Year')
            if idx == 0:
                ax.set_ylabel('Coefficient (Near × Year)')

            # Mark reference year
            ax.annotate('Ref', xy=(2011, 0), fontsize=8, ha='center', va='bottom',
                       color='gray')

            print(f"  {label}: event study computed")
            for i, y in enumerate(plot_years):
                if y != 2011:
                    print(f"    {y}: {coefs[i]:.4f} (SE={ses_arr[i]:.4f})")

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes, ha='center')
            print(f"  {label}: ERROR - {e}")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'multi_outcome_event_study.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: multi_outcome_event_study.png")


# =============================================================================
# ANALYSIS 7: DECOMPOSITION — WHERE CHILD LABOR OCCURS
# =============================================================================

def analysis7_decomposition(df):
    """Cross-tabulate work types by treatment status and period."""
    print("\n" + "=" * 70)
    print("ANALYSIS 7: DECOMPOSITION OF CHILD LABOR")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()

    # Focus on those with valid work data
    work_sub = sub[sub['hhcurrwork'].isin(['Yes', 'No'])].copy()
    work_sub = work_sub.dropna(subset=['near_factory'])
    work_sub['near'] = work_sub['near_factory'].astype(int)
    work_sub['period'] = np.where(work_sub['post'] == 1, 'Post (2014)', 'Pre (2000-2011)')

    print("\n  --- Work Status Distribution ---")
    for period in ['Pre (2000-2011)', 'Post (2014)']:
        for near in [0, 1]:
            d = work_sub[(work_sub['period'] == period) & (work_sub['near'] == near)]
            n_total = len(d)
            n_working = d['hhcurrwork'].eq('Yes').sum()
            pct = n_working / n_total * 100 if n_total > 0 else 0
            near_label = 'Near' if near == 1 else 'Far'
            print(f"    {period}, {near_label}: {n_working}/{n_total} = {pct:.1f}%")

    # Pay status among workers
    workers = work_sub[work_sub['hhcurrwork'] == 'Yes'].copy()
    print("\n  --- Pay Status Among Working Children ---")

    decomp_results = []
    for period in ['Pre (2000-2011)', 'Post (2014)']:
        for near in [0, 1]:
            d = workers[(workers['period'] == period) & (workers['near'] == near)]
            n_total = len(d)
            near_label = 'Near' if near == 1 else 'Far'

            if n_total > 0:
                pay_dist = d['hhcurrworkpay'].value_counts()
                for pay_type in ['Cash', 'In-kind', 'Both cash and in-kind', 'NIU (not in universe)']:
                    count = pay_dist.get(pay_type, 0)
                    pct = count / n_total * 100
                    decomp_results.append({
                        'Period': period, 'Treatment': near_label,
                        'Pay_Type': pay_type, 'Count': count,
                        'Total_Workers': n_total, 'Percent': pct
                    })
                print(f"    {period}, {near_label} (N={n_total}):")
                for pay_type, count in pay_dist.items():
                    print(f"      {pay_type}: {count} ({count/n_total*100:.1f}%)")

    # Overall summary: share of child labor in paid vs unpaid work
    print("\n  --- Summary: Paid vs Unpaid Child Labor ---")
    for period in ['Pre (2000-2011)', 'Post (2014)']:
        d = workers[workers['period'] == period]
        if len(d) > 0:
            paid = d['hhcurrworkpay'].isin(['Cash', 'Both cash and in-kind']).sum()
            unpaid = len(d) - paid
            print(f"    {period}: Paid={paid} ({paid/len(d)*100:.1f}%), Unpaid={unpaid} ({unpaid/len(d)*100:.1f}%)")

    decomp_df = pd.DataFrame(decomp_results)
    decomp_df.to_csv(os.path.join(OUT, 'work_decomposition.csv'), index=False)

    # LaTeX table
    latex = generate_decomposition_latex(work_sub, workers)
    with open(os.path.join(OUT, 'table_work_decomposition.tex'), 'w') as f:
        f.write(latex)
    print(f"\n  Saved: work_decomposition.csv, table_work_decomposition.tex")
    return decomp_df


def generate_decomposition_latex(work_sub, workers):
    """Generate decomposition table."""
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Decomposition of Child Labor by Work Type and Treatment Status}',
        r'\label{tab:decomposition}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'& \multicolumn{2}{c}{Pre-Rana Plaza} & \multicolumn{2}{c}{Post-Rana Plaza} \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}',
        r'& Near & Far & Near & Far \\',
        r'\midrule',
        r'\multicolumn{5}{l}{\textit{Panel A: Work Participation Rate}} \\',
    ]

    # Panel A: work rates
    for near in [1, 0]:
        for post in [0, 1]:
            period = 'Post (2014)' if post == 1 else 'Pre (2000-2011)'
            d = work_sub[(work_sub['period'] == period) & (work_sub['near'] == near)]
            n_total = len(d)
            n_working = d['hhcurrwork'].eq('Yes').sum()
            pct = n_working / n_total * 100 if n_total > 0 else 0

    # Build row
    vals = []
    for post in [0, 1]:
        for near in [1, 0]:
            period = 'Post (2014)' if post == 1 else 'Pre (2000-2011)'
            d = work_sub[(work_sub['period'] == period) & (work_sub['near'] == near)]
            n_total = len(d)
            n_working = d['hhcurrwork'].eq('Yes').sum()
            pct = n_working / n_total * 100 if n_total > 0 else 0
            vals.append(f"{pct:.1f}\\%")
    lines.append(f"\\quad Working (\\%) & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\")

    # Panel B: pay type among workers
    lines.extend([
        r'\\',
        r'\multicolumn{5}{l}{\textit{Panel B: Pay Type Among Working Children}} \\',
    ])
    for pay_label, pay_vals in [('Cash', ['Cash']),
                                 ('In-kind', ['In-kind']),
                                 ('Both', ['Both cash and in-kind'])]:
        vals = []
        for post in [0, 1]:
            for near in [1, 0]:
                period = 'Post (2014)' if post == 1 else 'Pre (2000-2011)'
                d = workers[(workers['period'] == period) & (workers['near'] == near)]
                if len(d) > 0:
                    count = d['hhcurrworkpay'].isin(pay_vals).sum()
                    pct = count / len(d) * 100
                    vals.append(f"{pct:.1f}\\%")
                else:
                    vals.append("---")
        lines.append(f"\\quad {pay_label} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\")

    # N row
    vals = []
    for post in [0, 1]:
        for near in [1, 0]:
            period = 'Post (2014)' if post == 1 else 'Pre (2000-2011)'
            d = work_sub[(work_sub['period'] == period) & (work_sub['near'] == near)]
            vals.append(f"{len(d):,}")
    lines.extend([
        r'\midrule',
        f"N (total children) & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\",
    ])

    vals_w = []
    for post in [0, 1]:
        for near in [1, 0]:
            period = 'Post (2014)' if post == 1 else 'Pre (2000-2011)'
            d = workers[(workers['period'] == period) & (workers['near'] == near)]
            vals_w.append(f"{len(d):,}")
    lines.append(f"N (working children) & {vals_w[0]} & {vals_w[1]} & {vals_w[2]} & {vals_w[3]} \\\\")

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: Sample restricted to ages 10--13 with valid work status data. ``Near'' indicates residence within 10km of a textile factory. Pre-Rana Plaza includes survey waves 2000--2011; post includes 2014 wave. Panel B shows the distribution of pay type among children reported as currently working.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# =============================================================================
# ANALYSIS 8: EDUCATION YEARS WAVE DECOMPOSITION
# =============================================================================

def analysis8_edyears_wave(df):
    """Test whether education years effect is driven by specific waves (composition artifact)."""
    print("\n" + "=" * 70)
    print("ANALYSIS 8: EDUCATION YEARS BY REFERENCE WAVE (ages 10-13)")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
    sub = sub.dropna(subset=['edyears_num', 'near_factory', 'district']).copy()
    controls = 'C(age) + female + hhsize + wealth_num'

    pre_years = [2000, 2004, 2007, 2011]
    results = []

    for ref_year in pre_years:
        d = sub[sub['year'].isin([ref_year, 2014])].copy()
        d['post'] = (d['year'] == 2014).astype(int)
        d['near_x_post'] = d['near_factory'] * d['post']
        try:
            mod = cluster_ols(f'edyears_num ~ near_x_post + near_factory + post + {controls}', d)
            b = mod.params['near_x_post']
            se = mod.bse['near_x_post']
            p = mod.pvalues['near_x_post']
            n = int(mod.nobs)
            results.append({
                'Reference_Year': ref_year, 'Coefficient': b, 'SE': se,
                'p_value': p, 'N': n
            })
            print(f"  Base={ref_year}: {b:.4f}{stars(p)} (SE={se:.4f}, p={p:.3f}, N={n:,})")
        except Exception as e:
            print(f"  Base={ref_year}: ERROR - {e}")

    # Also run pooled pre-period (standard DiD)
    d = sub.copy()
    d['post'] = (d['year'] == 2014).astype(int)
    d['near_x_post'] = d['near_factory'] * d['post']
    mod = cluster_ols(f'edyears_num ~ near_x_post + near_factory + post + {controls}', d)
    b = mod.params['near_x_post']
    se = mod.bse['near_x_post']
    p = mod.pvalues['near_x_post']
    results.append({
        'Reference_Year': 'Pooled', 'Coefficient': b, 'SE': se,
        'p_value': p, 'N': int(mod.nobs)
    })
    print(f"  Pooled pre: {b:.4f}{stars(p)} (SE={se:.4f}, p={p:.3f})")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT, 'edyears_by_wave.csv'), index=False)

    # Generate LaTeX table
    latex = generate_edyears_wave_latex(results)
    with open(os.path.join(OUT, 'table_edyears_wave.tex'), 'w') as f:
        f.write(latex)
    print(f"  Saved: edyears_by_wave.csv, table_edyears_wave.tex")
    return results_df


def generate_edyears_wave_latex(results):
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Education Years: Sensitivity to Reference Wave (Ages 10--13)}',
        r'\label{tab:edyears_wave}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Reference Wave & Coefficient & SE & $p$-value & $N$ \\',
        r'\midrule',
    ]
    for r in results:
        ref = str(r['Reference_Year'])
        b = r['Coefficient']
        se = r['SE']
        p = r['p_value']
        n = r['N']
        s = stars(p)
        lines.append(f'{ref} & {b:.4f}{s} & ({se:.4f}) & {p:.3f} & {int(n):,} \\\\')
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: Each row estimates the DiD coefficient for education years using only the indicated pre-period wave and the 2014 post-period. ``Pooled'' uses all pre-period waves. Controls: age dummies, sex, household size, wealth quintile. Standard errors clustered at district level. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# =============================================================================
# ANALYSIS 9: GENDER DIVERGENCE
# =============================================================================

def analysis9_gender(df):
    """Gender divergence: triple interaction + gender-specific event studies."""
    print("\n" + "=" * 70)
    print("ANALYSIS 9: GENDER DIVERGENCE (ages 10-13)")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
    sub = sub.dropna(subset=['child_labor', 'near_factory', 'district']).copy()
    sub['near_x_post'] = sub['near_factory'] * sub['post']
    controls = 'C(age) + hhsize + wealth_num'

    # --- Triple interaction: Near × Post × Female ---
    print("\n  Triple interaction test: Near × Post × Female")
    sub['near_x_post_x_female'] = sub['near_x_post'] * sub['female']
    sub['near_x_female'] = sub['near_factory'] * sub['female']
    sub['post_x_female'] = sub['post'] * sub['female']

    for outcome, label in [('child_labor', 'Child Labor'), ('enrolled', 'Enrollment')]:
        d = sub.dropna(subset=[outcome]).copy()
        try:
            mod = cluster_ols(
                f'{outcome} ~ near_x_post + near_factory + post + female + '
                f'near_x_female + post_x_female + near_x_post_x_female + {controls}', d)
            b3 = mod.params['near_x_post_x_female']
            se3 = mod.bse['near_x_post_x_female']
            p3 = mod.pvalues['near_x_post_x_female']
            print(f"    {label}: Near×Post×Female = {b3:.4f}{stars(p3)} (SE={se3:.4f}, p={p3:.3f})")
        except Exception as e:
            print(f"    {label}: ERROR - {e}")

    # --- Gender-specific DiD for child labor + enrollment ---
    print("\n  Gender-specific DiD estimates:")
    gender_results = []
    for gender, glabel in [('Male', 'Boys'), ('Female', 'Girls')]:
        d = sub[sub['sex'] == gender].copy()
        d['near_x_post'] = d['near_factory'] * d['post']
        for outcome, olabel in [('child_labor', 'Child Labor'), ('enrolled', 'Enrollment')]:
            dd = d.dropna(subset=[outcome]).copy()
            try:
                mod = cluster_ols(
                    f'{outcome} ~ near_x_post + near_factory + post + C(age) + hhsize + wealth_num', dd)
                b = mod.params['near_x_post']
                se = mod.bse['near_x_post']
                p = mod.pvalues['near_x_post']
                mean_y = dd[outcome].mean()
                gender_results.append({
                    'Gender': glabel, 'Outcome': olabel,
                    'Coefficient': b, 'SE': se, 'p_value': p,
                    'Mean_DV': mean_y, 'N': int(mod.nobs)
                })
                print(f"    {glabel}, {olabel}: {b:.4f}{stars(p)} (SE={se:.4f}, p={p:.3f}, N={int(mod.nobs):,})")
            except Exception as e:
                print(f"    {glabel}, {olabel}: ERROR - {e}")

    gender_df = pd.DataFrame(gender_results)
    gender_df.to_csv(os.path.join(OUT, 'gender_divergence.csv'), index=False)

    # --- Gender-specific event studies ---
    print("\n  Generating gender-specific event studies...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ref_year = 2011
    years = sorted(sub['year'].unique())

    for col_idx, (outcome, olabel) in enumerate([('child_labor', 'Child Labor'), ('enrolled', 'School Enrollment')]):
        for row_idx, (gender, glabel) in enumerate([('Male', 'Boys'), ('Female', 'Girls')]):
            ax = axes[row_idx, col_idx]
            d = sub[(sub['sex'] == gender)].dropna(subset=[outcome]).copy()
            coefs, ses, yr_labels = [], [], []
            for yr in years:
                if yr == ref_year:
                    coefs.append(0)
                    ses.append(0)
                    yr_labels.append(yr)
                    continue
                d[f'near_x_{yr}'] = (d['near_factory'] * (d['year'] == yr)).astype(float)

            # Build event study formula
            yr_vars = [f'near_x_{yr}' for yr in years if yr != ref_year]
            yr_dummies = ' + '.join([f'C(year == {yr})' for yr in years if yr != ref_year])
            formula = f'{outcome} ~ near_factory + {" + ".join(yr_vars)} + {yr_dummies} + C(age) + hhsize + wealth_num'
            try:
                mod = cluster_ols(formula, d)
                coefs, ses, yr_labels = [], [], []
                for yr in years:
                    if yr == ref_year:
                        coefs.append(0)
                        ses.append(0)
                    else:
                        coefs.append(mod.params[f'near_x_{yr}'])
                        ses.append(mod.bse[f'near_x_{yr}'])
                    yr_labels.append(yr)

                coefs = np.array(coefs)
                ses = np.array(ses)
                x = np.arange(len(yr_labels))
                ax.errorbar(x, coefs, yerr=1.96*ses, fmt='o-', capsize=4, color='steelblue')
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.axvline(x[yr_labels.index(2014)] - 0.5, color='red', linestyle='--',
                           linewidth=1.0, alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(yr_labels, fontsize=9)
                ax.set_title(f'{glabel}: {olabel}', fontsize=11, fontweight='bold')
                ax.set_ylabel('Coefficient')
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes, ha='center')

    fig.suptitle('Gender-Specific Event Studies (Ages 10-13, ref=2011)', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, 'gender_event_study.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gender_divergence.csv, gender_event_study.png")

    # Generate LaTeX table
    latex = generate_gender_latex(gender_results)
    with open(os.path.join(OUT, 'table_gender_divergence.tex'), 'w') as f:
        f.write(latex)
    print(f"  Saved: table_gender_divergence.tex")
    return gender_df


def generate_gender_latex(results):
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Gender-Specific Treatment Effects (Ages 10--13)}',
        r'\label{tab:gender_divergence}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{llccccc}',
        r'\toprule',
        r'Gender & Outcome & Coefficient & SE & $p$-value & Mean DV & $N$ \\',
        r'\midrule',
    ]
    for r in results:
        s = stars(r['p_value'])
        lines.append(f"{r['Gender']} & {r['Outcome']} & {r['Coefficient']:.4f}{s} & "
                     f"({r['SE']:.4f}) & {r['p_value']:.3f} & {r['Mean_DV']:.4f} & {r['N']:,} \\\\")
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: Separate DiD regressions for boys and girls, ages 10--13. Controls: age dummies, household size, wealth quintile. Standard errors clustered at district level. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# =============================================================================
# ANALYSIS 10: ENROLLMENT BY WEALTH (STRENGTHENED)
# =============================================================================

def analysis10_enrollment_wealth(df):
    """Strengthen enrollment-by-wealth finding: pooled middle+richer, event studies."""
    print("\n" + "=" * 70)
    print("ANALYSIS 10: ENROLLMENT BY WEALTH (ages 10-13)")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
    sub = sub.dropna(subset=['enrolled', 'near_factory', 'district', 'wealthqhh']).copy()
    sub['near_x_post'] = sub['near_factory'] * sub['post']
    controls = 'C(age) + female + hhsize'

    # --- Quintile-specific enrollment effects ---
    print("\n  Quintile-specific enrollment effects:")
    quintile_results = []
    for q in ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']:
        d = sub[sub['wealthqhh'] == q].copy()
        d['near_x_post'] = d['near_factory'] * d['post']
        try:
            mod = cluster_ols(f'enrolled ~ near_x_post + near_factory + post + {controls}', d)
            b = mod.params['near_x_post']
            se = mod.bse['near_x_post']
            p = mod.pvalues['near_x_post']
            mean_y = d['enrolled'].mean()
            quintile_results.append({
                'Quintile': q, 'Coefficient': b, 'SE': se,
                'p_value': p, 'Mean_DV': mean_y, 'N': int(mod.nobs)
            })
            print(f"    {q}: {b:.4f}{stars(p)} (SE={se:.4f}, p={p:.3f}, N={int(mod.nobs):,})")
        except Exception as e:
            print(f"    {q}: ERROR - {e}")

    # --- Pooled middle + richer ---
    print("\n  Pooled middle + richer:")
    d = sub[sub['wealthqhh'].isin(['Middle', 'Richer'])].copy()
    d['near_x_post'] = d['near_factory'] * d['post']
    mod = cluster_ols(f'enrolled ~ near_x_post + near_factory + post + {controls}', d)
    b = mod.params['near_x_post']
    se = mod.bse['near_x_post']
    p = mod.pvalues['near_x_post']
    pooled_result = {
        'Quintile': 'Middle+Richer', 'Coefficient': b, 'SE': se,
        'p_value': p, 'Mean_DV': d['enrolled'].mean(), 'N': int(mod.nobs)
    }
    quintile_results.append(pooled_result)
    print(f"    Middle+Richer: {b:.4f}{stars(p)} (SE={se:.4f}, p={p:.3f}, N={int(mod.nobs):,})")

    qdf = pd.DataFrame(quintile_results)
    qdf.to_csv(os.path.join(OUT, 'enrollment_by_wealth.csv'), index=False)

    # --- Event study by wealth group (3 groups: poor, middle, rich) ---
    print("\n  Generating enrollment event studies by wealth group...")
    wealth_groups = {
        'Poorest+Poorer': ['Poorest', 'Poorer'],
        'Middle': ['Middle'],
        'Richer+Richest': ['Richer', 'Richest'],
    }
    ref_year = 2011
    years = sorted(sub['year'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (glabel, quintiles) in enumerate(wealth_groups.items()):
        ax = axes[idx]
        d = sub[sub['wealthqhh'].isin(quintiles)].copy()

        # Create year-treatment interactions
        for yr in years:
            if yr != ref_year:
                d[f'near_x_{yr}'] = (d['near_factory'] * (d['year'] == yr)).astype(float)

        yr_vars = [f'near_x_{yr}' for yr in years if yr != ref_year]
        yr_dummies = ' + '.join([f'C(year == {yr})' for yr in years if yr != ref_year])
        formula = f'enrolled ~ near_factory + {" + ".join(yr_vars)} + {yr_dummies} + C(age) + female + hhsize'

        try:
            mod = cluster_ols(formula, d)
            coefs, ses = [], []
            for yr in years:
                if yr == ref_year:
                    coefs.append(0)
                    ses.append(0)
                else:
                    coefs.append(mod.params[f'near_x_{yr}'])
                    ses.append(mod.bse[f'near_x_{yr}'])

            coefs = np.array(coefs)
            ses = np.array(ses)
            x = np.arange(len(years))
            ax.errorbar(x, coefs, yerr=1.96*ses, fmt='o-', capsize=4, color='steelblue')
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(x[years.index(2014)] - 0.5, color='red', linestyle='--',
                       linewidth=1.0, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(years, fontsize=9)
            ax.set_title(f'{glabel}\n(N={len(d):,})', fontsize=11, fontweight='bold')
            ax.set_ylabel('Coefficient (School Enrollment)')

            # Annotate post-treatment coefficient
            post_idx = years.index(2014)
            ax.annotate(f'{coefs[post_idx]:.3f}', xy=(x[post_idx], coefs[post_idx]),
                       xytext=(5, 10), textcoords='offset points', fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes, ha='center')

    fig.suptitle('School Enrollment Event Study by Wealth Group (Ages 10-13, ref=2011)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, 'enrollment_wealth_event_study.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Generate LaTeX table
    latex = generate_enrollment_wealth_latex(quintile_results)
    with open(os.path.join(OUT, 'table_enrollment_wealth.tex'), 'w') as f:
        f.write(latex)
    print(f"  Saved: enrollment_by_wealth.csv, enrollment_wealth_event_study.png, table_enrollment_wealth.tex")
    return qdf


def generate_enrollment_wealth_latex(results):
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{School Enrollment Effects by Wealth Quintile (Ages 10--13)}',
        r'\label{tab:enrollment_wealth}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'Wealth Group & Coefficient & SE & $p$-value & Mean DV & $N$ \\',
        r'\midrule',
    ]
    for r in results:
        s = stars(r['p_value'])
        q = r['Quintile']
        if q == 'Middle+Richer':
            lines.append(r'\midrule')
        lines.append(f"{q} & {r['Coefficient']:.4f}{s} & ({r['SE']:.4f}) & "
                     f"{r['p_value']:.3f} & {r['Mean_DV']:.4f} & {r['N']:,} \\\\")
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: Separate DiD regressions for each wealth quintile, ages 10--13. Outcome: school enrollment (binary). Controls: age dummies, sex, household size. Standard errors clustered at district level. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# =============================================================================
# ANALYSIS 11: LEE (2009) BOUNDS
# =============================================================================

def analysis11_lee_bounds(df):
    """Lee (2009) trimming bounds for subgroups with significant effects."""
    print("\n" + "=" * 70)
    print("ANALYSIS 11: LEE (2009) BOUNDS FOR SIGNIFICANT SUBGROUPS")
    print("=" * 70)

    sub = df[(df['age'] >= 10) & (df['age'] <= 13)].copy()
    sub = sub.dropna(subset=['near_factory', 'district']).copy()
    controls = 'C(age) + female + hhsize + wealth_num'

    def lee_bounds(data, outcome_var, treatment_var='near_x_post', controls_str=controls):
        """
        Compute Lee (2009) bounds by trimming the outcome distribution.

        Under selective attrition, treatment may change who we observe.
        Lee bounds trim the "excess" observations from the group with higher
        response rate, from the top and bottom of the outcome distribution,
        giving upper and lower bounds on the treatment effect.
        """
        d = data.dropna(subset=[outcome_var, 'near_factory', 'post', 'district']).copy()
        d['near_x_post'] = d['near_factory'] * d['post']

        # Step 1: Check differential non-response
        # Create indicator for having non-missing outcome
        full = data.copy()
        full['observed'] = full[outcome_var].notna().astype(int)
        full['near_x_post'] = full['near_factory'] * full['post']
        try:
            sel_mod = cluster_ols(f'observed ~ near_x_post + near_factory + post', full)
            sel_coef = sel_mod.params['near_x_post']
            sel_p = sel_mod.pvalues['near_x_post']
        except Exception:
            sel_coef, sel_p = 0, 1.0

        # Step 2: Compute response rates by treatment × period cells
        cells = d.groupby(['near_factory', 'post']).size().reset_index(name='n_obs')
        full_cells = full.groupby(['near_factory', 'post']).size().reset_index(name='n_total')
        rates = cells.merge(full_cells, on=['near_factory', 'post'])
        rates['response_rate'] = rates['n_obs'] / rates['n_total']

        # Treatment group post-period vs control group post-period
        try:
            rate_t1 = rates[(rates['near_factory'] == 1) & (rates['post'] == 1)]['response_rate'].values[0]
            rate_c1 = rates[(rates['near_factory'] == 0) & (rates['post'] == 1)]['response_rate'].values[0]
        except (IndexError, KeyError):
            rate_t1, rate_c1 = 1.0, 1.0

        trim_frac = abs(rate_t1 - rate_c1) / max(rate_t1, rate_c1)

        # Step 3: Trim from group with higher response rate
        if trim_frac < 0.001:
            # No meaningful differential attrition — bounds equal point estimate
            mod = cluster_ols(f'{outcome_var} ~ near_x_post + near_factory + post + {controls_str}', d)
            b = mod.params['near_x_post']
            se = mod.bse['near_x_post']
            return {
                'lower_bound': b, 'upper_bound': b, 'point_estimate': b,
                'se': se, 'trim_fraction': trim_frac,
                'selection_coef': sel_coef, 'selection_p': sel_p,
                'N': int(mod.nobs)
            }

        # Determine which group to trim
        if rate_t1 > rate_c1:
            trim_group = d[(d['near_factory'] == 1) & (d['post'] == 1)]
            keep_group = d[~((d['near_factory'] == 1) & (d['post'] == 1))]
        else:
            trim_group = d[(d['near_factory'] == 0) & (d['post'] == 1)]
            keep_group = d[~((d['near_factory'] == 0) & (d['post'] == 1))]

        n_trim = int(np.ceil(len(trim_group) * trim_frac))

        # Lower bound: trim top of outcome distribution
        trimmed_lower = trim_group.nsmallest(len(trim_group) - n_trim, outcome_var)
        d_lower = pd.concat([keep_group, trimmed_lower])
        d_lower['near_x_post'] = d_lower['near_factory'] * d_lower['post']

        # Upper bound: trim bottom of outcome distribution
        trimmed_upper = trim_group.nlargest(len(trim_group) - n_trim, outcome_var)
        d_upper = pd.concat([keep_group, trimmed_upper])
        d_upper['near_x_post'] = d_upper['near_factory'] * d_upper['post']

        try:
            mod_lower = cluster_ols(f'{outcome_var} ~ near_x_post + near_factory + post + {controls_str}', d_lower)
            mod_upper = cluster_ols(f'{outcome_var} ~ near_x_post + near_factory + post + {controls_str}', d_upper)
            mod_point = cluster_ols(f'{outcome_var} ~ near_x_post + near_factory + post + {controls_str}', d)

            return {
                'lower_bound': mod_lower.params['near_x_post'],
                'upper_bound': mod_upper.params['near_x_post'],
                'point_estimate': mod_point.params['near_x_post'],
                'se': mod_point.bse['near_x_post'],
                'trim_fraction': trim_frac,
                'selection_coef': sel_coef, 'selection_p': sel_p,
                'N': int(mod_point.nobs)
            }
        except Exception as e:
            print(f"      Lee bounds regression error: {e}")
            return None

    # Subgroups to test
    subgroups = [
        ('Full sample (ages 10-13)', sub, 'child_labor', controls),
        ('Full sample: enrollment', sub, 'enrolled', controls),
        ('Girls: child labor', sub[sub['sex'] == 'Female'], 'child_labor',
         'C(age) + hhsize + wealth_num'),
        ('Girls: enrollment', sub[sub['sex'] == 'Female'], 'enrolled',
         'C(age) + hhsize + wealth_num'),
        ('Middle+Richer: enrollment',
         sub[sub['wealthqhh'].isin(['Middle', 'Richer'])], 'enrolled',
         'C(age) + female + hhsize'),
    ]

    results = []
    for label, data, outcome, ctrls in subgroups:
        print(f"\n  {label}:")
        bounds = lee_bounds(data, outcome, controls_str=ctrls)
        if bounds:
            results.append({'Subgroup': label, **bounds})
            print(f"    Point estimate: {bounds['point_estimate']:.4f} (SE={bounds['se']:.4f})")
            print(f"    Lee bounds: [{bounds['lower_bound']:.4f}, {bounds['upper_bound']:.4f}]")
            print(f"    Trim fraction: {bounds['trim_fraction']:.4f}")
            print(f"    Selection test: {bounds['selection_coef']:.4f} (p={bounds['selection_p']:.3f})")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT, 'lee_bounds.csv'), index=False)

    # Generate LaTeX table
    latex = generate_lee_bounds_latex(results)
    with open(os.path.join(OUT, 'table_lee_bounds.tex'), 'w') as f:
        f.write(latex)
    print(f"\n  Saved: lee_bounds.csv, table_lee_bounds.tex")
    return results_df


def generate_lee_bounds_latex(results):
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Lee (2009) Trimming Bounds (Ages 10--13)}',
        r'\label{tab:lee_bounds}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'Subgroup & Point Est. & Lower Bound & Upper Bound & Trim \% & Selection $p$ \\',
        r'\midrule',
    ]
    for r in results:
        s = stars(r.get('selection_p', 1))
        lines.append(
            f"{r['Subgroup']} & {r['point_estimate']:.4f} & {r['lower_bound']:.4f} & "
            f"{r['upper_bound']:.4f} & {r['trim_fraction']*100:.1f} & {r.get('selection_p', 999):.3f} \\\\"
        )
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item Notes: Lee (2009) trimming bounds account for potential selective attrition. Point estimates from standard DiD with controls. Lower and upper bounds trim the ``excess'' observations from the treatment-period cell with higher response rate, from the top and bottom of the outcome distribution respectively. Trim \% indicates the fraction of observations trimmed. Selection $p$ tests for differential non-response ($H_0$: treatment does not affect observation probability). All specifications include age dummies, household size, and wealth quintile (or sex where applicable). Standard errors clustered at district level.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df):,} observations, {df['year'].nunique()} waves")
    print(f"  Years: {sorted(df['year'].unique())}")

    ages1013 = df[(df['age'] >= 10) & (df['age'] <= 13)]
    print(f"  Ages 10-13 sample: {len(ages1013):,}")

    # Run original analyses (1-7)
    r1 = analysis1_multiple_outcomes(df)
    r2 = analysis2_heterogeneity(df)
    r3 = analysis3_dose_response(df)
    r4 = analysis4_power_mde(df)
    r5 = analysis5_equivalence_test(df)
    analysis6_event_study_multi(df)
    r7 = analysis7_decomposition(df)

    # Run new analyses (8-11)
    r8 = analysis8_edyears_wave(df)
    r9 = analysis9_gender(df)
    r10 = analysis10_enrollment_wealth(df)
    r11 = analysis11_lee_bounds(df)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print(f"\nOutput files in: {OUT}")
    for f in sorted(os.listdir(OUT)):
        if any(f.startswith(p) for p in ('table_', 'multiple_', 'heterogeneity_',
               'dose_', 'tost_', 'work_', 'power_curve', 'multi_outcome',
               'edyears_', 'gender_', 'enrollment_', 'lee_')):
            fpath = os.path.join(OUT, f)
            size = os.path.getsize(fpath)
            print(f"  {f} ({size:,} bytes)")


if __name__ == '__main__':
    main()
