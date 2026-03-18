# Supply Chain Monitoring and Child Labor: Evidence from Bangladesh Following Rana Plaza

**Authors:** Alina Malkova & Clara Canon (Florida Institute of Technology)

## Project Overview

Difference-in-Differences study examining whether post-Rana Plaza factory monitoring (Accord/Alliance) reduced child labor in Bangladesh. Uses DHS survey data (2000-2018) merged with geocoded factory census data. Treatment: children within 10km of monitored factories.

## Key Files

### Paper
- `Child_Labor_Paper_Updated.tex` — Current main paper (includes updated robustness results)
- `Child Labor Oct 28.tex` — Earlier full version with detailed institutional background (Sections 2-2.5)
- `Child Labor Updated Results.tex` — Updated results section standalone
- `Updated_Abstract.tex` — Current abstract
- `title page.docx`, `The authors have no relevant...docx` — Front matter

### Code
- `child_labor_replication.py` — Main analysis: DID estimates, event studies, age heterogeneity, distance robustness, brand analysis
- `robustness_analysis.py` — Trend adjustments, matching/IPW, Callaway-Sant'Anna, Honest DiD (Rambachan-Roth), triple differences

### Data (not in repo — too large)
- `DHS/` — Bangladesh DHS rounds (2000, 2004, 2007, 2011, 2014, 2018) + IPUMS extracts
- `Results/DHS_imputed.csv` — Main analysis dataset (173,065 children)
- `Results/processed_data_for_child_analysis.csv` — Processed child labor dataset
- `Factories/` — Factory geocoded data
- `Accord/` — Accord compliance inspection data
- MICS datasets (zip archives)

### Results
- `Results/` — All output CSVs and figures
- `Results/FINAL/` — Consolidated final outputs
- Key figures: `event_study_replication.png`, `robustness_analysis.png`, `honest_did_analysis.png`, `trend_adjusted_analysis.png`

### Exploratory
- `Mess code/` — Jupyter notebooks (data processing, brand matching, individual FE specs)

## Known Issues (Pre-Revision)

1. **Identification fragility**: Pre-trend significant (p=0.011); effect disappears with group-specific trends or triple-diff; Honest DiD breakdown M*=0.00
2. **Distance robustness bug**: Coefficients identical across all thresholds (2km-30km) — likely a coding error in treatment variable construction
3. **Brand analysis broken**: Coefficients of ~-10^10, NaN p-values, 83.7% missing brand data; some brands show identical coefficients/SEs suggesting data construction error
4. **Framing disconnect**: Robustness shows suggestive/descriptive evidence at best, but conclusion claims "causal evidence"
5. **Presentation**: Placeholder captions, duplicate figure labels, "Your Name" author placeholder; Rana Plaza background (~3000 words) too long for econ journal

## Revision Plan

- Fix distance robustness code (treatment variable construction)
- Cut or appendix the brand analysis with heavy caveats
- Condense Rana Plaza background to ~1 page of institutional context
- Reframe as suggestive/descriptive evidence of accelerated trends
- Polish all figures, tables, captions, and metadata

## Technical Notes

- Python analysis uses `statsmodels` (OLS with cluster-robust SEs at district level)
- LaTeX compilation: standard `pdflatex` pipeline
- Treatment definition: `near_factory = 1` if household within 10km of any Accord/Alliance factory
- Reference year for event study: 2007 (main) or 2011 (alternative)
- DHS survey waves: 2000, 2004, 2007, 2011, 2014 (main), 2018 (supplementary)
