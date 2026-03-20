"""
Regenerate all figures with clean, publication-quality styling.
Uses error bars (not fill_between) for event studies — standard in economics.
Output to: results/new_results/
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
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

BASE_PATH = "/Users/amalkova/Library/CloudStorage/OneDrive-FloridaInstituteofTechnology/_Research/Labor_Economics/Child Labor"
OUT_DIR = f"{BASE_PATH}/results/new_results"

BLUE = '#1f77b4'
DARK_BLUE = '#0d4f8b'
RED = '#d62728'
GRAY = '#888888'

# Load data
df = pd.read_csv(f"{BASE_PATH}/DHS/final_analysis_data.csv", low_memory=False)
df['child_labor'] = pd.to_numeric(df['child_labor'], errors='coerce')
df['near_factory'] = df['within_10km']
df['post'] = df['post_rana']
df['near_x_post'] = df['within_10km'] * df['post_rana']

df = df[(df['age'] >= 5) & (df['age'] <= 17)]
df = df.dropna(subset=['child_labor', 'near_factory', 'year', 'age'])
df = df[df['year'] != 2018]

df['age_5_9'] = ((df['age'] >= 5) & (df['age'] <= 9)).astype(int)
df['age_10_13'] = ((df['age'] >= 10) & (df['age'] <= 13)).astype(int)
df['age_14_17'] = ((df['age'] >= 14) & (df['age'] <= 17)).astype(int)

if 'sex' in df.columns:
    if df['sex'].dtype == object:
        df['sex'] = (df['sex'] == 'Male').astype(float)
    else:
        df['sex'] = pd.to_numeric(df['sex'], errors='coerce')

print(f"Sample: {len(df)} observations")


def run_event_study(data, ref_year=2007):
    data = data.copy()
    years = sorted(data['year'].unique())
    years_no_ref = [y for y in years if y != ref_year]

    for y in years_no_ref:
        data[f'near_x_{y}'] = (data['near_factory'] * (data['year'] == y)).astype(float)

    controls = ['near_factory']
    for y in years:
        if y != years[0]:
            data[f'year_{y}'] = (data['year'] == y).astype(float)
            controls.append(f'year_{y}')
    for a in sorted(data['age'].unique()):
        if a != data['age'].min():
            data[f'age_{a}'] = (data['age'] == a).astype(float)
            controls.append(f'age_{a}')
    if 'sex' in data.columns:
        controls.append('sex')

    interaction_vars = [f'near_x_{y}' for y in years_no_ref]
    X = data[interaction_vars + controls].astype(float)
    X = sm.add_constant(X)
    y = data['child_labor'].astype(float)

    if 'geolev2' in data.columns:
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data['geolev2']})
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1')

    results = {}
    for y_name in years_no_ref:
        var = f'near_x_{y_name}'
        results[y_name] = {
            'coef': model.params[var],
            'se': model.bse[var],
        }
    results[ref_year] = {'coef': 0, 'se': 0}
    return results


def plot_event_study(ax, results, ref_year, show_ylabel=True, title=None):
    """Plot event study with error bars (no fill_between)."""
    years = sorted(results.keys())
    coefs = [results[y]['coef'] for y in years]
    ses = [results[y]['se'] for y in years]

    # Error bars for all non-reference years
    for y_val, c, s in zip(years, coefs, ses):
        if y_val == ref_year:
            # Open circle for reference year
            ax.plot(y_val, 0, 'o', color=DARK_BLUE, markersize=7,
                    markerfacecolor='white', markeredgewidth=1.5, zorder=5)
        else:
            ax.errorbar(y_val, c, yerr=1.96 * s, fmt='o', color=DARK_BLUE,
                        markersize=6, capsize=4, linewidth=1.3, elinewidth=1.3,
                        markerfacecolor=BLUE, markeredgecolor=DARK_BLUE,
                        markeredgewidth=0.8, zorder=4)

    # Connect points with line
    ax.plot(years, coefs, '-', color=DARK_BLUE, linewidth=1, alpha=0.5, zorder=2)

    ax.axhline(y=0, color=GRAY, linestyle='-', alpha=0.4, linewidth=0.6)
    ax.axvline(x=2013, color=RED, linestyle='--', alpha=0.5, linewidth=1,
               label='Rana Plaza (2013)')

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Survey Year', fontsize=10)
    if show_ylabel:
        ax.set_ylabel('Coefficient (Near Factory \u00d7 Year)', fontsize=10)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=9)
    ax.tick_params(axis='y', labelsize=9)


# ============================================================================
# Figure 1: Event Study by Age Group
# ============================================================================
print("--- Figure 1: event_study_by_age.png ---")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
age_groups = [
    ('Ages 5\u20139', df[df['age_5_9'] == 1].copy()),
    ('Ages 10\u201313', df[df['age_10_13'] == 1].copy()),
    ('Ages 14\u201317', df[df['age_14_17'] == 1].copy())
]

for idx, (title, sub_df) in enumerate(age_groups):
    results = run_event_study(sub_df, ref_year=2007)
    plot_event_study(axes[idx], results, ref_year=2007,
                     show_ylabel=(idx == 0), title=title)
    if idx == 2:
        axes[idx].legend(fontsize=8, framealpha=0.8, loc='upper right')

plt.tight_layout(w_pad=2.5)
plt.savefig(f"{OUT_DIR}/event_study_by_age.png", dpi=300)
plt.close()
print("  Saved")


# ============================================================================
# Figure 2: Reference Year Comparison
# ============================================================================
print("--- Figure 2: event_study_age10_13_ref_comparison.png ---")

sub_1013 = df[df['age_10_13'] == 1].copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, ref_year in enumerate([2007, 2011]):
    results = run_event_study(sub_1013.copy(), ref_year=ref_year)
    plot_event_study(axes[ax_idx], results, ref_year=ref_year,
                     show_ylabel=(ax_idx == 0),
                     title=f'Reference Year: {ref_year}')

fig.suptitle('Event Study: Ages 10\u201313, Alternative Reference Years',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(w_pad=3)
plt.savefig(f"{OUT_DIR}/event_study_age10_13_ref_comparison.png", dpi=300)
plt.close()
print("  Saved")


# ============================================================================
# Figure 3: Trend-Adjusted Analysis
# ============================================================================
print("--- Figure 3: trend_adjusted_analysis.png ---")

sub_1013 = df[df['age_10_13'] == 1].copy()
sub_1013['time'] = sub_1013['year'] - 2013
sub_1013['near_x_time'] = sub_1013['near_factory'] * sub_1013['time']
sub_1013['time2'] = sub_1013['time'] ** 2
sub_1013['near_x_time2'] = sub_1013['near_factory'] * sub_1013['time2']
sub_1013['time3'] = sub_1013['time'] ** 3
sub_1013['near_x_time3'] = sub_1013['near_factory'] * sub_1013['time3']

age_dummies = pd.get_dummies(sub_1013['age'], prefix='age', drop_first=True).astype(float)
sub_1013 = pd.concat([sub_1013, age_dummies], axis=1)
age_cols = [c for c in sub_1013.columns if c.startswith('age_') and c[4:].isdigit()]

base_controls = ['near_factory', 'post', 'near_x_post'] + age_cols
if 'sex' in sub_1013.columns:
    base_controls.append('sex')

specs = {
    'Standard DiD': base_controls,
    '+ Common Linear': base_controls + ['time'],
    '+ Group Linear': base_controls + ['time', 'near_x_time'],
    '+ Group Quadratic': base_controls + ['time', 'near_x_time', 'time2', 'near_x_time2'],
    '+ Group Cubic': base_controls + ['time', 'near_x_time', 'time2', 'near_x_time2', 'time3', 'near_x_time3'],
}

trend_results = {}
for name, controls in specs.items():
    X = sm.add_constant(sub_1013[controls].astype(float))
    y = sub_1013['child_labor'].astype(float)
    if 'geolev2' in sub_1013.columns:
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': sub_1013['geolev2']})
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1')
    trend_results[name] = {
        'coef': model.params['near_x_post'],
        'se': model.bse['near_x_post'],
    }

fig, ax = plt.subplots(figsize=(8, 5))
names = list(trend_results.keys())
coefs = [trend_results[n]['coef'] for n in names]
ses = [trend_results[n]['se'] for n in names]

x_pos = range(len(names))
ax.errorbar(x_pos, coefs, yerr=[1.96 * s for s in ses],
            fmt='o', markersize=8, color=DARK_BLUE, capsize=4, linewidth=1.5,
            markerfacecolor=BLUE, markeredgecolor=DARK_BLUE, ecolor=BLUE,
            elinewidth=1.3)
ax.axhline(y=0, color=RED, linestyle='--', alpha=0.6, linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
ax.set_ylabel('Treatment Effect (Near Factory \u00d7 Post)', fontsize=11)
ax.set_title('Trend-Adjusted DiD Estimates: Ages 10\u201313',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/trend_adjusted_analysis.png", dpi=300)
plt.close()
print("  Saved")


# ============================================================================
# Figure 4: Honest DiD Analysis (2 panels)
# ============================================================================
print("--- Figure 4: honest_did_analysis.png ---")

sub_1013 = df[df['age_10_13'] == 1].copy()
results_2011 = run_event_study(sub_1013.copy(), ref_year=2011)

coef_2014 = results_2011[2014]['coef']
se_2014 = results_2011[2014]['se']
pre_years = [y for y in results_2011.keys() if y < 2013 and y != 2011]
max_pre = max(abs(results_2011[y]['coef']) for y in pre_years)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Event study with error bars
plot_event_study(axes[0], results_2011, ref_year=2011,
                 show_ylabel=True, title='(a) Event Study (ref = 2011)')
axes[0].set_ylabel('Coefficient', fontsize=10)

# Panel B: Sensitivity to M-bar (fill_between is fine here — continuous x)
ax = axes[1]
m_values = np.linspace(0, 2.5, 200)
ci_los = [coef_2014 - 1.96 * se_2014 - m * max_pre for m in m_values]
ci_his = [coef_2014 + 1.96 * se_2014 + m * max_pre for m in m_values]

ax.fill_between(m_values, ci_los, ci_his, alpha=0.15, color=BLUE, linewidth=0)
ax.plot(m_values, ci_los, color=BLUE, linewidth=0.8, alpha=0.5)
ax.plot(m_values, ci_his, color=BLUE, linewidth=0.8, alpha=0.5)
ax.plot(m_values, [coef_2014] * len(m_values), color=DARK_BLUE, linewidth=1.5,
        label=f'Point estimate = {coef_2014:.4f}')
ax.axhline(y=0, color=RED, linestyle='--', alpha=0.6, linewidth=1, label='Zero effect')

ax.set_xlabel(r'$\bar{M}$ (trend violation bound)', fontsize=11)
ax.set_ylabel('Treatment Effect', fontsize=10)
ax.set_title(r'(b) Robust CI by $\bar{M}$', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.8, loc='lower left')

fig.suptitle('Honest DiD Sensitivity Analysis (Ages 10\u201313)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(w_pad=3)
plt.savefig(f"{OUT_DIR}/honest_did_analysis.png", dpi=300)
plt.close()
print("  Saved")


# ============================================================================
# Figure 5: Event Study Replication (main figure)
# ============================================================================
print("--- Figure 5: event_study_replication.png ---")

sub_1013 = df[df['age_10_13'] == 1].copy()
results_main = run_event_study(sub_1013.copy(), ref_year=2007)

fig, ax = plt.subplots(figsize=(8, 5))
plot_event_study(ax, results_main, ref_year=2007, show_ylabel=True)
ax.set_ylabel('Near Factory \u00d7 Year Coefficient', fontsize=11)
ax.set_title('Event Study: Effect of Factory Proximity on Child Labor\n'
             '(Ages 10\u201313, Reference Year = 2007)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/event_study_replication.png", dpi=300)
plt.close()
print("  Saved")


# ============================================================================
# Figure 6: Robustness Analysis (3 panels)
# ============================================================================
print("--- Figure 6: robustness_analysis.png ---")

sub_1013 = df[df['age_10_13'] == 1].copy()

# Panel A data: trend specs
sub_t = sub_1013.copy()
sub_t['time'] = sub_t['year'] - 2013
sub_t['near_x_time'] = sub_t['near_factory'] * sub_t['time']
sub_t['time2'] = sub_t['time'] ** 2
sub_t['near_x_time2'] = sub_t['near_factory'] * sub_t['time2']
sub_t['time3'] = sub_t['time'] ** 3
sub_t['near_x_time3'] = sub_t['near_factory'] * sub_t['time3']

age_dum = pd.get_dummies(sub_t['age'], prefix='age', drop_first=True).astype(float)
sub_t = pd.concat([sub_t, age_dum], axis=1)
a_cols = [c for c in sub_t.columns if c.startswith('age_') and c[4:].isdigit()]
b_ctrl = ['near_factory', 'post', 'near_x_post'] + a_cols
if 'sex' in sub_t.columns:
    b_ctrl.append('sex')

t_specs = {
    'Standard DiD': b_ctrl,
    '+ Common Trend': b_ctrl + ['time'],
    '+ Group Linear': b_ctrl + ['time', 'near_x_time'],
    '+ Group Quadratic': b_ctrl + ['time', 'near_x_time', 'time2', 'near_x_time2'],
    '+ Group Cubic': b_ctrl + ['time', 'near_x_time', 'time2', 'near_x_time2', 'time3', 'near_x_time3'],
}

t_res = {}
for name, ctrls in t_specs.items():
    X = sm.add_constant(sub_t[ctrls].astype(float))
    y = sub_t['child_labor'].astype(float)
    if 'geolev2' in sub_t.columns:
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': sub_t['geolev2']})
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1')
    t_res[name] = {'coef': model.params['near_x_post'], 'se': model.bse['near_x_post']}

# Panel B data: Rambachan-Roth
res_2011 = run_event_study(sub_1013.copy(), ref_year=2011)
c14 = res_2011[2014]['coef']
s14 = res_2011[2014]['se']
pre_yrs = [y for y in res_2011.keys() if y < 2013 and y != 2011]
max_p = max(abs(res_2011[y]['coef']) for y in pre_yrs)

fig = plt.figure(figsize=(15, 5))

# Panel A: horizontal coefficient plot
ax1 = fig.add_subplot(131)
names_a = list(t_res.keys())
coefs_a = [t_res[n]['coef'] for n in names_a]
ses_a = [t_res[n]['se'] for n in names_a]
y_pos = list(range(len(names_a) - 1, -1, -1))

ax1.errorbar(coefs_a, y_pos, xerr=[1.96 * s for s in ses_a],
             fmt='o', color=DARK_BLUE, markersize=7, capsize=4, linewidth=1.3,
             markerfacecolor=BLUE, markeredgecolor=DARK_BLUE, ecolor=BLUE,
             elinewidth=1.3)
ax1.axvline(x=0, color=GRAY, linestyle='-', alpha=0.5, linewidth=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names_a, fontsize=9)
ax1.set_xlabel('Treatment Effect (pp)', fontsize=10)
ax1.set_title('(a) Trend Adjustment', fontsize=12, fontweight='bold')

# Panel B: Rambachan-Roth
ax2 = fig.add_subplot(132)
m_vals = np.linspace(0, 2.0, 200)
ci_lo_b = [c14 - 1.96 * s14 - m * max_p for m in m_vals]
ci_hi_b = [c14 + 1.96 * s14 + m * max_p for m in m_vals]

ax2.fill_between(m_vals, ci_lo_b, ci_hi_b, alpha=0.15, color=BLUE, linewidth=0)
ax2.plot(m_vals, ci_lo_b, color=BLUE, linewidth=0.8, alpha=0.5)
ax2.plot(m_vals, ci_hi_b, color=BLUE, linewidth=0.8, alpha=0.5)
ax2.plot(m_vals, [c14] * len(m_vals), color=DARK_BLUE, linewidth=1.5)
ax2.axhline(y=0, color=RED, linestyle='--', alpha=0.6, linewidth=1)
ax2.set_xlabel(r'$\bar{M}$ (Relative Magnitudes Bound)', fontsize=10)
ax2.set_ylabel('Treatment Effect', fontsize=10)
ax2.set_title('(b) Rambachan\u2013Roth', fontsize=12, fontweight='bold')

# Panel C: Event study (ref=2011) with error bars
ax3 = fig.add_subplot(133)
plot_event_study(ax3, res_2011, ref_year=2011, show_ylabel=True,
                 title='(c) Event Study (ref = 2011)')
ax3.set_ylabel('Coefficient', fontsize=10)

plt.tight_layout(w_pad=2.5)
plt.savefig(f"{OUT_DIR}/robustness_analysis.png", dpi=300)
plt.close()
print("  Saved")

print("\n=== All figures regenerated ===")
