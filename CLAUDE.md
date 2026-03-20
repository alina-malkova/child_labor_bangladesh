# Supply Chain Monitoring and Child Labor: Evidence from Bangladesh Following Rana Plaza

**Authors:** Alina Malkova & Clara Canon (Florida Institute of Technology)

## Project Overview

Difference-in-Differences study examining whether post-Rana Plaza factory monitoring (Accord/Alliance) reduced child labor in Bangladesh. Uses DHS survey data (2000-2014) merged with geocoded factory census data. Treatment: children within 10km of monitored factories.

**Main finding (March 2026):** No statistically significant effect. All treatment effects are null across specifications, age groups, and robustness checks.

## Key Files

### Paper
- `Child_Labor_Paper_Updated.tex` — Current main paper with corrected GPS results (March 2026)
- `Child Labor Oct 28.tex` — Earlier full version (condensed intro/background/conclusion)
- `title page.docx`, `The authors have no relevant...docx` — Front matter

### Code
- `child_labor_replication.py` — Main analysis: DID estimates, event studies, age heterogeneity, distance robustness
- `robustness_analysis.py` — Trend adjustments, matching/IPW, Callaway-Sant'Anna, Honest DiD, triple differences
- `generate_paper_tables.py` — Generates all coefficients for paper tables from corrected data
- `fix_gps_merge.py` — Merges real DHS cluster GPS coordinates into analysis data
- `download_dhs_gps.py` — IPUMS DHS API download script (alternative GPS source)

### Data (not in repo — too large)
- `DHS/final_analysis_data.csv` — Main analysis dataset (173,065 obs, corrected GPS)
- `DHS/GPS/` — DHS cluster GPS shapefiles (BDGE42FL through BDGE81FL)
- `Factories/` — Factory geocoded data
- `Accord/` — Accord compliance inspection data

### Results
- `Results/` — All output CSVs and figures

## GPS Data Fix (Critical)

The original analysis used 5 division-capital coordinates for all 173,065 observations (assigned in `DHS/do-file.do`). This caused identical treatment shares at all distance thresholds and artificial pre-trends.

**Fix:** `fix_gps_merge.py` merges real DHS cluster-level GPS from shapefiles:
- 2,932 cluster-year coordinates, 99.9% match rate
- Treatment shares properly monotonic: 9% (2km) to 59% (30km)
- **IMPORTANT:** `treat_X_post` in CSV was stale — must use `within_10km * post_rana`
  (Fixed in CSV as of March 2026; also fixed in `generate_paper_tables.py`)

## Key Results (Corrected GPS)

- Ages 10-13 standard DiD: -0.0056 (SE=0.0075, p=0.46) — **null**
- Pre-trend: +0.0002 (SE=0.0017, p=0.93) — **essentially zero**
- Treatment effect stable across trend specs (no sign reversal)
- All distance thresholds: null
- Triple-diff: null
- Honest DiD M*=0.00 (already insignificant at exact parallel trends)
- School enrollment: +0.016 (p=0.16) — null

## Technical Notes

- Python analysis uses `statsmodels` (OLS with cluster-robust SEs)
- Clustering: geolev2 (65 districts) — 2018 wave excluded (all geolev2 = NaN)
- Treatment definition: `within_10km = 1` if cluster within 10km of any factory
- Reference year: 2007 (main) or 2011 (alternative)
- DHS waves used: 2000, 2004, 2007, 2011, 2014
- Variable mapping: sex (not female), wealthqhh (not wealthqhh_num)
