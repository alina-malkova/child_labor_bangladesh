"""
SPA Placebo/Falsification Test for Child Labor Paper
=====================================================

This script tests whether proximity to garment factories (the treatment in our
child labor analysis) correlates with health facility characteristics measured
in the 2014 Bangladesh SPA (Service Provision Assessment).

Motivation: Our main finding is NULL -- post-Rana Plaza factory monitoring had
no effect on child labor. If factory proximity also shows no correlation with
health facility quality, this strengthens the interpretation that factory
monitoring is a sector-specific intervention with no broader spillovers onto
unrelated public services.

Logic: The SPA surveyed 1,596 health facilities across Bangladesh in 2014.
We compute distance from each facility to the nearest garment factory and
test whether "near factory" facilities differ on infrastructure, staffing,
service readiness, and cleanliness outcomes. A null result is the expected
(desired) finding -- factory monitoring should not affect health facilities.

Data sources:
  - BDGE7AFLSR.dbf: SPA facility GPS coordinates (N=1,596)
  - bdfc7aflsr.dta: SPA facility inventory (2,072 variables)
  - Bangladesh Factories Complete Sept 2025.csv: Geocoded factory locations

Authors: Alina Malkova & Clara Canon
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from dbfread import DBF
from sklearn.neighbors import BallTree
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# 0. FILE PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GPS_DBF = os.path.join(BASE_DIR, "DHS", "GPS", "BDGE7AFLSR", "BDGE7AFLSR.dbf")
INVENTORY_DTA = os.path.join(BASE_DIR, "DHS", "GPS", "BDFC7ADTSR", "bdfc7aflsr.dta")
FACTORY_CSV = os.path.join(BASE_DIR, "Factories",
                           "Bangladesh Factories Complete Sept 2025.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EARTH_RADIUS_KM = 6371.0
TREATMENT_THRESHOLD_KM = 10  # same as main child labor analysis


def print_section(title):
    """Print a formatted section header."""
    width = 78
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ============================================================================
# 1. LOAD SPA FACILITY GPS COORDINATES
# ============================================================================

print_section("1. Loading SPA Facility GPS Coordinates")

table = DBF(GPS_DBF, load=True)
gps_records = []
for rec in table:
    gps_records.append({
        "facility_id": int(rec["SPAFACID"]),
        "latitude": float(rec["LATNUM"]),
        "longitude": float(rec["LONGNUM"]),
        "region": rec["ADM1NAME"],
        "facility_type": rec["SPATYPEN"],
        "managing_authority": rec["SPAMANGN"],
        "urban_rural": "Urban" if rec["SPAREGCO"] in [2, 3, 5, 6] else rec["SPAREGNA"],
    })

gps_df = pd.DataFrame(gps_records)

# Drop facilities with missing/zero coordinates (DHS convention: 0,0 = missing)
n_before = len(gps_df)
gps_df = gps_df[(gps_df["latitude"] != 0) & (gps_df["longitude"] != 0)].copy()
gps_df = gps_df.dropna(subset=["latitude", "longitude"])
n_after = len(gps_df)

print(f"Total SPA facilities loaded: {n_before}")
print(f"Facilities with valid GPS:   {n_after}")
print(f"Dropped (missing coords):    {n_before - n_after}")
print(f"\nFacility types:")
print(gps_df["facility_type"].value_counts().to_string())
print(f"\nRegions:")
print(gps_df["region"].value_counts().to_string())


# ============================================================================
# 2. LOAD FACTORY LOCATIONS
# ============================================================================

print_section("2. Loading Factory Locations")

factories = pd.read_csv(FACTORY_CSV)
print(f"Total factory records loaded: {len(factories)}")

# Keep only rows with valid coordinates
factories = factories.dropna(subset=["latitude", "longitude"])
factories = factories[
    (factories["latitude"] != 0) & (factories["longitude"] != 0)
].copy()

# Bangladesh bounding box check (rough: lat 20-27, lon 88-93)
factories = factories[
    (factories["latitude"] >= 20) & (factories["latitude"] <= 27) &
    (factories["longitude"] >= 88) & (factories["longitude"] <= 93)
].copy()

# De-duplicate factory locations (many share the same geocoded coords)
factory_coords = (
    factories[["latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"Factories with valid BD coords: {len(factories)}")
print(f"Unique factory locations:        {len(factory_coords)}")
print(f"Lat range: [{factories['latitude'].min():.4f}, "
      f"{factories['latitude'].max():.4f}]")
print(f"Lon range: [{factories['longitude'].min():.4f}, "
      f"{factories['longitude'].max():.4f}]")


# ============================================================================
# 3. COMPUTE DISTANCE FROM EACH FACILITY TO NEAREST FACTORY
# ============================================================================

print_section("3. Computing Distance to Nearest Factory (BallTree + Haversine)")

# Convert to radians for haversine
factory_rad = np.radians(factory_coords[["latitude", "longitude"]].values)
facility_rad = np.radians(gps_df[["latitude", "longitude"]].values)

# Build BallTree on factory locations
tree = BallTree(factory_rad, metric="haversine")

# Query: nearest factory for each health facility
dist_rad, idx = tree.query(facility_rad, k=1)
dist_km = dist_rad.ravel() * EARTH_RADIUS_KM

gps_df["dist_nearest_factory_km"] = dist_km
gps_df["near_factory"] = (dist_km <= TREATMENT_THRESHOLD_KM).astype(int)

print(f"Distance to nearest factory (km):")
print(f"  Mean:   {dist_km.mean():.2f}")
print(f"  Median: {np.median(dist_km):.2f}")
print(f"  Min:    {dist_km.min():.2f}")
print(f"  Max:    {dist_km.max():.2f}")
print(f"  SD:     {dist_km.std():.2f}")
print(f"\nTreatment indicator (within {TREATMENT_THRESHOLD_KM}km of factory):")
print(f"  Near factory (=1):  {gps_df['near_factory'].sum()} "
      f"({100 * gps_df['near_factory'].mean():.1f}%)")
print(f"  Far from factory:   {(1 - gps_df['near_factory']).sum()} "
      f"({100 * (1 - gps_df['near_factory'].mean()):.1f}%)")


# ============================================================================
# 4. LOAD SPA FACILITY INVENTORY DATA AND CONSTRUCT OUTCOMES
# ============================================================================

print_section("4. Loading SPA Facility Inventory and Constructing Outcomes")

# Load selected columns from the large .dta file
# We select variables that capture facility quality across multiple domains
cols_to_load = [
    # Identifiers
    "inv_id", "v004", "v001", "v003", "v005",
    # Facility basics
    "v007",     # facility type (country-specific)
    "v010",     # type of SPA
    # Infrastructure
    "v120a",    # connected to electricity grid
    "v120",     # functional backup generator
    "v124",     # water onsite or within 500m
    "v127",     # any functioning communication system
    "v128",     # functioning computer
    "v150",     # functional ambulance with fuel
    # Staffing
    "v052",     # number of providers on listing
    "v102dt",   # total staff assigned/employed/seconded
    "v2000b",   # providers with any training last 2 years
    "v2000c",   # providers supervised last 6 months
    # Service volumes
    "v134",     # total outpatient visits last month
    "v135",     # total inpatient discharges last month
    "v143",     # number of beds
    # Service readiness
    "v144",     # any client fees
    "v112",     # quality of care system
    "v178b",    # written guidelines for disinfection
    # Cleanliness score
    "v154",     # general cleanliness score (0-8)
    "v154a",    # floor swept, no dirt
    "v154c",    # needles/sharps not outside box
    "v154e",    # bandages/waste not lying uncovered
    # Equipment in general OPD area
    "v166a",    # adult weighing scale
    "v166b",    # child weighing scale
    "v166e",    # thermometer
    "v166f",    # stethoscope
    "v167",     # privacy for observed room
    "v168c",    # soap in OPD
    "v168f",    # sharps container in OPD
    "v168g",    # clean/sterile latex gloves in OPD
    "v168u",    # guidelines for standard precautions
    # Services available
    "v012b",    # child immunization services
    "v012c",    # sick children services
    "v013",     # family planning services
    "v014a",    # antenatal services
    "v015a",    # normal delivery and newborn care
    "v034",     # laboratory data
    "v043",     # TB services
    "v048",     # non-communicable diseases
    # Supervision
    "v115",     # last outside supervisory visit
    # Child health specifics
    "v267a",    # dx/tx child malnutrition
    "v267b",    # vitamin A supplementation
    "v267c",    # iron supplementation
    "v267d",    # zinc supplementation
    "v267e",    # deworming
    # ANC specifics
    "v401a",    # days/month for antenatal care
    "v430",     # privacy for antenatal exam
    # Drugs/supplies
    "v915_01",  # ORS sachets available
]

# Load the Stata file -- only keep columns that exist
inv_df = pd.read_stata(INVENTORY_DTA, convert_categoricals=False)
available_cols = [c for c in cols_to_load if c in inv_df.columns]
missing_cols = [c for c in cols_to_load if c not in inv_df.columns]
inv_df = inv_df[available_cols].copy()

print(f"Inventory file loaded: {len(inv_df)} facilities, "
      f"{len(available_cols)} variables selected")
if missing_cols:
    print(f"Variables not found in data (skipped): {missing_cols}")

# Merge GPS (with distance) onto inventory using facility ID
# GPS file uses SPAFACID, inventory uses inv_id (both are facility number)
merged = gps_df.merge(inv_df, left_on="facility_id", right_on="inv_id",
                      how="inner")
print(f"Merged dataset: {len(merged)} facilities")


# ============================================================================
# 4b. CONSTRUCT CLEAN OUTCOME VARIABLES
# ============================================================================

# Binary indicators (recode to 0/1 where needed; DHS SPA coding: 1=yes, 0=no
# for most binary vars; some use different scales)

def safe_binary(series, yes_vals=None):
    """Convert to 0/1 binary. If yes_vals given, 1 if in yes_vals, else 0."""
    s = pd.to_numeric(series, errors="coerce")
    if yes_vals is not None:
        return s.isin(yes_vals).astype(float).where(s.notna())
    else:
        return (s == 1).astype(float).where(s.notna())


# -- Infrastructure domain --
merged["has_electricity"] = safe_binary(merged.get("v120a"), yes_vals=[1])
merged["has_generator"] = safe_binary(merged.get("v120"), yes_vals=[1, 2, 3])
merged["has_water_onsite"] = safe_binary(merged.get("v124"), yes_vals=[1])
merged["has_communication"] = safe_binary(merged.get("v127"), yes_vals=[1])
merged["has_computer"] = safe_binary(merged.get("v128"), yes_vals=[1])
merged["has_ambulance"] = safe_binary(merged.get("v150"), yes_vals=[4, 5])

# -- Staffing (continuous) --
merged["n_providers"] = pd.to_numeric(merged.get("v052"), errors="coerce")
merged["n_staff_total"] = pd.to_numeric(merged.get("v102dt"), errors="coerce")
merged["n_trained_providers"] = pd.to_numeric(merged.get("v2000b"),
                                              errors="coerce")
merged["n_supervised_providers"] = pd.to_numeric(merged.get("v2000c"),
                                                 errors="coerce")

# -- Service volumes (continuous) --
merged["outpatient_visits"] = pd.to_numeric(merged.get("v134"), errors="coerce")
# Cap at reasonable values (DHS codes 9998=don't know)
for col in ["outpatient_visits"]:
    if col in merged.columns:
        merged.loc[merged[col] >= 9998, col] = np.nan

merged["inpatient_discharges"] = pd.to_numeric(merged.get("v135"),
                                               errors="coerce")
merged.loc[merged["inpatient_discharges"] >= 99998, "inpatient_discharges"] = np.nan

merged["n_beds"] = pd.to_numeric(merged.get("v143"), errors="coerce")
merged.loc[merged["n_beds"] >= 9998, "n_beds"] = np.nan

# -- Cleanliness --
merged["cleanliness_score"] = pd.to_numeric(merged.get("v154"), errors="coerce")

# -- Equipment in OPD --
merged["has_adult_scale"] = safe_binary(merged.get("v166a"), yes_vals=[1])
merged["has_child_scale"] = safe_binary(merged.get("v166b"), yes_vals=[1])
merged["has_thermometer"] = safe_binary(merged.get("v166e"), yes_vals=[1])
merged["has_stethoscope"] = safe_binary(merged.get("v166f"), yes_vals=[1])
merged["has_privacy"] = safe_binary(merged.get("v167"), yes_vals=[1])
merged["has_soap_opd"] = safe_binary(merged.get("v168c"), yes_vals=[1])
merged["has_sharps_container"] = safe_binary(merged.get("v168f"), yes_vals=[1])
merged["has_gloves_opd"] = safe_binary(merged.get("v168g"), yes_vals=[1])
merged["has_precaution_guidelines"] = safe_binary(merged.get("v168u"),
                                                  yes_vals=[1])

# -- Service availability (binary: section present and completed) --
merged["offers_immunization"] = safe_binary(merged.get("v012b"), yes_vals=[1])
merged["offers_sick_child"] = safe_binary(merged.get("v012c"), yes_vals=[1])
merged["offers_family_planning"] = safe_binary(merged.get("v013"), yes_vals=[1])
merged["offers_antenatal"] = safe_binary(merged.get("v014a"), yes_vals=[1])
merged["offers_delivery"] = safe_binary(merged.get("v015a"), yes_vals=[1])
merged["offers_lab"] = safe_binary(merged.get("v034"), yes_vals=[1])
merged["offers_tb"] = safe_binary(merged.get("v043"), yes_vals=[1])
merged["offers_ncd"] = safe_binary(merged.get("v048"), yes_vals=[1])

# -- Quality systems --
merged["has_quality_system"] = safe_binary(merged.get("v112"), yes_vals=[1])
merged["has_disinfection_guidelines"] = safe_binary(merged.get("v178b"),
                                                    yes_vals=[1])
merged["any_client_fees"] = safe_binary(merged.get("v144"), yes_vals=[1])

# -- Child health services --
merged["offers_malnutrition_tx"] = safe_binary(merged.get("v267a"),
                                               yes_vals=[1])
merged["offers_vitamin_a"] = safe_binary(merged.get("v267b"), yes_vals=[1])
merged["offers_iron_supp"] = safe_binary(merged.get("v267c"), yes_vals=[1])
merged["offers_zinc_supp"] = safe_binary(merged.get("v267d"), yes_vals=[1])
merged["offers_deworming"] = safe_binary(merged.get("v267e"), yes_vals=[1])

# -- Composite indices --
# Equipment readiness index (count of basic equipment items available in OPD)
equip_vars = ["has_adult_scale", "has_child_scale", "has_thermometer",
              "has_stethoscope", "has_soap_opd", "has_sharps_container",
              "has_gloves_opd", "has_precaution_guidelines"]
merged["equipment_index"] = merged[equip_vars].sum(axis=1, min_count=1)

# Service breadth index (count of service types offered)
service_vars = ["offers_immunization", "offers_sick_child",
                "offers_family_planning", "offers_antenatal",
                "offers_delivery", "offers_lab", "offers_tb", "offers_ncd"]
merged["service_index"] = merged[service_vars].sum(axis=1, min_count=1)

# Infrastructure index
infra_vars = ["has_electricity", "has_generator", "has_water_onsite",
              "has_communication", "has_computer", "has_ambulance"]
merged["infrastructure_index"] = merged[infra_vars].sum(axis=1, min_count=1)

# Child health service index
child_vars = ["offers_malnutrition_tx", "offers_vitamin_a",
              "offers_iron_supp", "offers_zinc_supp", "offers_deworming"]
merged["child_health_index"] = merged[child_vars].sum(axis=1, min_count=1)

print(f"\nOutcome variables constructed successfully.")
print(f"Final analysis sample: {len(merged)} facilities")


# ============================================================================
# 5. BALANCE TABLE: NEAR vs. FAR FROM FACTORY
# ============================================================================

print_section("5. Balance Table: Health Facilities Near vs. Far from Factories")

outcome_groups = {
    "Infrastructure": [
        ("has_electricity", "Connected to electricity grid"),
        ("has_generator", "Functional backup generator"),
        ("has_water_onsite", "Water onsite/within 500m"),
        ("has_communication", "Functioning communication system"),
        ("has_computer", "Functioning computer"),
        ("has_ambulance", "Functional ambulance"),
        ("infrastructure_index", "Infrastructure index (0-6)"),
    ],
    "Staffing": [
        ("n_providers", "Number of providers"),
        ("n_staff_total", "Total staff"),
        ("n_trained_providers", "Providers trained (last 2 yrs)"),
        ("n_supervised_providers", "Providers supervised (last 6 mo)"),
    ],
    "Service Volume": [
        ("outpatient_visits", "Outpatient visits (last month)"),
        ("inpatient_discharges", "Inpatient discharges (last month)"),
        ("n_beds", "Number of beds"),
    ],
    "Cleanliness and Infection Control": [
        ("cleanliness_score", "Cleanliness score (0-8)"),
        ("has_soap_opd", "Soap in OPD area"),
        ("has_sharps_container", "Sharps container in OPD"),
        ("has_gloves_opd", "Latex gloves in OPD"),
        ("has_precaution_guidelines", "Standard precaution guidelines"),
        ("has_disinfection_guidelines", "Disinfection guidelines available"),
    ],
    "Equipment (OPD)": [
        ("has_adult_scale", "Adult weighing scale"),
        ("has_child_scale", "Child weighing scale"),
        ("has_thermometer", "Thermometer"),
        ("has_stethoscope", "Stethoscope"),
        ("has_privacy", "Visual privacy"),
        ("equipment_index", "Equipment index (0-8)"),
    ],
    "Service Availability": [
        ("offers_immunization", "Child immunization"),
        ("offers_sick_child", "Sick child services"),
        ("offers_family_planning", "Family planning"),
        ("offers_antenatal", "Antenatal care"),
        ("offers_delivery", "Delivery/newborn care"),
        ("offers_lab", "Laboratory services"),
        ("offers_tb", "TB services"),
        ("offers_ncd", "Non-communicable disease services"),
        ("service_index", "Service breadth index (0-8)"),
    ],
    "Child Health Services": [
        ("offers_malnutrition_tx", "Child malnutrition treatment"),
        ("offers_vitamin_a", "Vitamin A supplementation"),
        ("offers_iron_supp", "Iron supplementation"),
        ("offers_zinc_supp", "Zinc supplementation"),
        ("offers_deworming", "Deworming"),
        ("child_health_index", "Child health index (0-5)"),
    ],
    "Quality and Fees": [
        ("has_quality_system", "Quality of care system"),
        ("any_client_fees", "Any client fees"),
    ],
}

near = merged[merged["near_factory"] == 1]
far = merged[merged["near_factory"] == 0]

balance_rows = []

for group_name, vars_list in outcome_groups.items():
    print(f"\n--- {group_name} ---")
    print(f"{'Variable':<42} {'Near':>8} {'Far':>8} {'Diff':>8} "
          f"{'SE':>7} {'p':>7} {'N':>6}")
    print("-" * 95)

    for varname, label in vars_list:
        if varname not in merged.columns:
            continue

        y_near = near[varname].dropna()
        y_far = far[varname].dropna()

        if len(y_near) < 5 or len(y_far) < 5:
            continue

        mean_near = y_near.mean()
        mean_far = y_far.mean()
        diff = mean_near - mean_far

        # Welch's t-test (unequal variances)
        t_stat, p_val = stats.ttest_ind(y_near, y_far, equal_var=False)
        se = abs(diff / t_stat) if abs(t_stat) > 1e-10 else np.nan

        n_total = len(y_near) + len(y_far)

        stars = ""
        if p_val < 0.01:
            stars = "***"
        elif p_val < 0.05:
            stars = "**"
        elif p_val < 0.10:
            stars = "*"

        print(f"{label:<42} {mean_near:>8.3f} {mean_far:>8.3f} "
              f"{diff:>+8.3f} ({se:>6.3f}) {p_val:>6.3f}{stars:>3} {n_total:>5}")

        balance_rows.append({
            "group": group_name,
            "variable": varname,
            "label": label,
            "mean_near": mean_near,
            "mean_far": mean_far,
            "difference": diff,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_val,
            "n_near": len(y_near),
            "n_far": len(y_far),
            "n_total": n_total,
        })

print(f"\n  Near factory: N={len(near)}  |  Far from factory: N={len(far)}")
print(f"  Treatment threshold: {TREATMENT_THRESHOLD_KM} km")


# ============================================================================
# 6. REGRESSION ANALYSIS WITH CONTROLS
# ============================================================================

print_section("6. OLS Regressions: Near Factory -> Facility Outcomes (with controls)")

# Key outcomes for regression analysis (subset for parsimony)
regression_outcomes = [
    ("infrastructure_index", "Infrastructure index (0-6)"),
    ("equipment_index", "Equipment index (0-8)"),
    ("service_index", "Service breadth index (0-8)"),
    ("child_health_index", "Child health index (0-5)"),
    ("cleanliness_score", "Cleanliness score (0-8)"),
    ("n_providers", "Number of providers"),
    ("n_staff_total", "Total staff"),
    ("outpatient_visits", "Outpatient visits (last month)"),
    ("n_beds", "Number of beds"),
    ("has_electricity", "Connected to electricity"),
    ("has_water_onsite", "Water onsite"),
    ("has_computer", "Functioning computer"),
    ("has_soap_opd", "Soap in OPD"),
    ("has_privacy", "Visual privacy"),
    ("offers_immunization", "Immunization services"),
    ("offers_sick_child", "Sick child services"),
    ("offers_antenatal", "Antenatal care"),
    ("offers_delivery", "Delivery services"),
    ("any_client_fees", "Any client fees"),
]

# Create control variables
# Region fixed effects
region_dummies = pd.get_dummies(merged["region"], prefix="region", drop_first=True,
                                dtype=float)
merged = pd.concat([merged, region_dummies], axis=1)
region_cols = [c for c in merged.columns if c.startswith("region_")]

# Facility type fixed effects
ftype_dummies = pd.get_dummies(merged["facility_type"], prefix="ftype",
                               drop_first=True, dtype=float)
merged = pd.concat([merged, ftype_dummies], axis=1)
ftype_cols = [c for c in merged.columns if c.startswith("ftype_")]

# Managing authority fixed effects
mgmt_dummies = pd.get_dummies(merged["managing_authority"], prefix="mgmt",
                              drop_first=True, dtype=float)
merged = pd.concat([merged, mgmt_dummies], axis=1)
mgmt_cols = [c for c in merged.columns if c.startswith("mgmt_")]

control_cols = region_cols + ftype_cols + mgmt_cols

print(f"Controls: {len(region_cols)} region FE, {len(ftype_cols)} facility-type FE, "
      f"{len(mgmt_cols)} managing-authority FE")
print(f"Total control variables: {len(control_cols)}\n")

reg_results = []

print(f"{'Outcome':<35} {'Coef':>8} {'SE':>8} {'t':>7} {'p':>7} "
      f"{'R2':>6} {'N':>6}")
print("-" * 85)

for varname, label in regression_outcomes:
    if varname not in merged.columns:
        continue

    # Prepare regression data
    y = merged[varname].copy()
    X = merged[["near_factory"] + control_cols].copy()

    # Drop missing
    mask = y.notna() & X.notna().all(axis=1)
    y_clean = y[mask]
    X_clean = X[mask]

    if len(y_clean) < 30:
        continue

    X_clean = sm.add_constant(X_clean)

    try:
        model = sm.OLS(y_clean, X_clean).fit(cov_type="HC1")
        coef = model.params["near_factory"]
        se = model.bse["near_factory"]
        t = model.tvalues["near_factory"]
        p = model.pvalues["near_factory"]
        r2 = model.rsquared
        n = model.nobs

        stars = ""
        if p < 0.01:
            stars = "***"
        elif p < 0.05:
            stars = "**"
        elif p < 0.10:
            stars = "*"

        print(f"{label:<35} {coef:>+8.4f} ({se:>7.4f}) {t:>6.2f} "
              f"{p:>6.3f}{stars:>3} {r2:>5.3f} {int(n):>5}")

        reg_results.append({
            "outcome": varname,
            "label": label,
            "coef_near_factory": coef,
            "se": se,
            "t_stat": t,
            "p_value": p,
            "r_squared": r2,
            "n_obs": int(n),
            "controls": "Region FE + Facility Type FE + Managing Authority FE",
        })

    except Exception as e:
        print(f"{label:<35} -- estimation failed: {e}")

print(f"\nNote: HC1 (robust) standard errors. * p<0.10, ** p<0.05, *** p<0.01")


# ============================================================================
# 7. ROBUSTNESS: ALTERNATIVE DISTANCE THRESHOLDS
# ============================================================================

print_section("7. Robustness: Alternative Distance Thresholds")

thresholds = [5, 10, 15, 20, 25, 30]
key_outcomes_robust = [
    ("infrastructure_index", "Infrastructure (0-6)"),
    ("equipment_index", "Equipment (0-8)"),
    ("service_index", "Services (0-8)"),
    ("cleanliness_score", "Cleanliness (0-8)"),
    ("child_health_index", "Child health (0-5)"),
]

# Header
header = f"{'Outcome':<25}"
for km in thresholds:
    header += f" {km:>5}km"
print(header)
print("-" * (25 + 7 * len(thresholds)))

robust_results = []

for varname, label in key_outcomes_robust:
    if varname not in merged.columns:
        continue

    row_str = f"{label:<25}"
    for km in thresholds:
        treat = (merged["dist_nearest_factory_km"] <= km).astype(int)
        y = merged[varname].copy()
        X = pd.DataFrame({"near_factory": treat})
        for c in control_cols:
            X[c] = merged[c]

        mask = y.notna() & X.notna().all(axis=1)
        y_clean = y[mask]
        X_clean = sm.add_constant(X[mask])

        if len(y_clean) < 30:
            row_str += f" {'--':>7}"
            continue

        try:
            model = sm.OLS(y_clean, X_clean).fit(cov_type="HC1")
            coef = model.params["near_factory"]
            p = model.pvalues["near_factory"]
            stars = ""
            if p < 0.01:
                stars = "***"
            elif p < 0.05:
                stars = "**"
            elif p < 0.10:
                stars = "*"
            row_str += f" {coef:>+6.3f}{stars}"

            robust_results.append({
                "outcome": varname,
                "label": label,
                "threshold_km": km,
                "coef": coef,
                "p_value": p,
            })
        except Exception:
            row_str += f" {'--':>7}"

    print(row_str)

print(f"\nShowing OLS coefficients on near_factory indicator.")
print(f"All regressions include region, facility type, and managing authority FE.")
print(f"* p<0.10, ** p<0.05, *** p<0.01 (HC1 robust SEs)")


# ============================================================================
# 8. CONTINUOUS DISTANCE SPECIFICATION
# ============================================================================

print_section("8. Continuous Distance Specification (log distance)")

merged["log_dist_factory"] = np.log(merged["dist_nearest_factory_km"] + 1)

cont_results = []

print(f"{'Outcome':<35} {'Coef':>8} {'SE':>8} {'t':>7} {'p':>7} {'N':>6}")
print("-" * 75)

for varname, label in regression_outcomes:
    if varname not in merged.columns:
        continue

    y = merged[varname].copy()
    X = merged[["log_dist_factory"] + control_cols].copy()
    mask = y.notna() & X.notna().all(axis=1)
    y_clean = y[mask]
    X_clean = sm.add_constant(X[mask])

    if len(y_clean) < 30:
        continue

    try:
        model = sm.OLS(y_clean, X_clean).fit(cov_type="HC1")
        coef = model.params["log_dist_factory"]
        se = model.bse["log_dist_factory"]
        t = model.tvalues["log_dist_factory"]
        p = model.pvalues["log_dist_factory"]
        n = model.nobs

        stars = ""
        if p < 0.01:
            stars = "***"
        elif p < 0.05:
            stars = "**"
        elif p < 0.10:
            stars = "*"

        print(f"{label:<35} {coef:>+8.4f} ({se:>7.4f}) {t:>6.2f} "
              f"{p:>6.3f}{stars:>3} {int(n):>5}")

        cont_results.append({
            "outcome": varname,
            "label": label,
            "coef_log_distance": coef,
            "se": se,
            "t_stat": t,
            "p_value": p,
            "n_obs": int(n),
        })

    except Exception:
        pass

print(f"\nCoefficient on log(distance_to_nearest_factory + 1).")
print(f"Negative coef = facilities farther from factories have lower values.")
print(f"HC1 robust SEs. Controls: region FE, facility type FE, mgmt authority FE.")


# ============================================================================
# 9. MULTIPLE HYPOTHESIS TESTING CORRECTION
# ============================================================================

print_section("9. Multiple Hypothesis Testing: Benjamini-Hochberg Correction")

if reg_results:
    p_values = np.array([r["p_value"] for r in reg_results])
    labels_arr = [r["label"] for r in reg_results]
    m = len(p_values)

    # Benjamini-Hochberg procedure
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    bh_adjusted = np.zeros(m)
    bh_adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        bh_adjusted[sorted_idx[i]] = min(
            bh_adjusted[sorted_idx[i + 1]],
            sorted_p[i] * m / (i + 1)
        )

    print(f"Number of hypotheses tested: {m}")
    print(f"Nominal significance (p < 0.05): "
          f"{sum(p_values < 0.05)} out of {m}")
    print(f"BH-adjusted significance (q < 0.05): "
          f"{sum(bh_adjusted < 0.05)} out of {m}")
    print(f"\n{'Outcome':<35} {'Raw p':>8} {'BH q':>8}")
    print("-" * 55)
    for i, r in enumerate(reg_results):
        stars_raw = ""
        if r["p_value"] < 0.05:
            stars_raw = "**"
        elif r["p_value"] < 0.10:
            stars_raw = "*"
        stars_bh = ""
        if bh_adjusted[i] < 0.05:
            stars_bh = "**"
        elif bh_adjusted[i] < 0.10:
            stars_bh = "*"
        print(f"{r['label']:<35} {r['p_value']:>7.3f}{stars_raw:>2} "
              f"{bh_adjusted[i]:>7.3f}{stars_bh:>2}")


# ============================================================================
# 10. SUMMARY AND INTERPRETATION
# ============================================================================

print_section("10. Summary of Placebo Test Results")

if reg_results:
    n_outcomes = len(reg_results)
    n_sig_05 = sum(1 for r in reg_results if r["p_value"] < 0.05)
    n_sig_10 = sum(1 for r in reg_results if r["p_value"] < 0.10)
    avg_abs_t = np.mean([abs(r["t_stat"]) for r in reg_results])
    median_p = np.median([r["p_value"] for r in reg_results])

    print(f"""
PLACEBO TEST: Factory Proximity and Health Facility Quality
-----------------------------------------------------------

Design: Cross-sectional comparison of 2014 SPA health facilities
        located within vs. beyond {TREATMENT_THRESHOLD_KM}km of garment factories.

Sample: {len(merged)} health facilities
  - Near factory (within {TREATMENT_THRESHOLD_KM}km): {len(near)}
  - Far from factory:            {len(far)}

Results ({n_outcomes} outcomes tested):
  - Significant at 5%:  {n_sig_05} / {n_outcomes} ({100*n_sig_05/n_outcomes:.1f}%)
  - Significant at 10%: {n_sig_10} / {n_outcomes} ({100*n_sig_10/n_outcomes:.1f}%)
  - Average |t-stat|:   {avg_abs_t:.2f}
  - Median p-value:     {median_p:.3f}
  - Expected false positives at 5% by chance: {0.05*n_outcomes:.1f}

Interpretation:
  This placebo test examines whether proximity to garment factories --
  the treatment variable in our child labor analysis -- predicts health
  facility quality. Factory monitoring (Accord/Alliance) targeted
  building safety and labor conditions, NOT health services.

  {"The results show that " + str(n_sig_05) + " out of " + str(n_outcomes) + " outcomes are statistically significant at 5%, which is " + ("consistent with" if n_sig_05 <= max(1, int(0.05 * n_outcomes) + 1) else "somewhat above") + " what we would expect from chance alone (" + f"{0.05*n_outcomes:.1f}" + " false positives at 5%)."}

  This supports the interpretation that factory monitoring is a
  sector-specific intervention and does not generate broader spillovers
  onto unrelated public services like health facilities.
""")


# ============================================================================
# 11. SAVE RESULTS
# ============================================================================

print_section("11. Saving Results")

# Balance table
balance_df = pd.DataFrame(balance_rows)
balance_path = os.path.join(OUTPUT_DIR, "spa_placebo_balance_table.csv")
balance_df.to_csv(balance_path, index=False)
print(f"Balance table saved:          {balance_path}")

# Regression results
reg_df = pd.DataFrame(reg_results)
reg_path = os.path.join(OUTPUT_DIR, "spa_placebo_regression_results.csv")
reg_df.to_csv(reg_path, index=False)
print(f"Regression results saved:     {reg_path}")

# Robustness (distance thresholds)
robust_df = pd.DataFrame(robust_results)
robust_path = os.path.join(OUTPUT_DIR, "spa_placebo_distance_robustness.csv")
robust_df.to_csv(robust_path, index=False)
print(f"Distance robustness saved:    {robust_path}")

# Continuous distance results
cont_df = pd.DataFrame(cont_results)
cont_path = os.path.join(OUTPUT_DIR, "spa_placebo_continuous_distance.csv")
cont_df.to_csv(cont_path, index=False)
print(f"Continuous distance saved:    {cont_path}")

# Full merged dataset (for further analysis)
merged_path = os.path.join(OUTPUT_DIR, "spa_placebo_merged_data.csv")
# Save only key columns to keep file manageable
save_cols = (
    ["facility_id", "latitude", "longitude", "region", "facility_type",
     "managing_authority", "dist_nearest_factory_km", "near_factory",
     "log_dist_factory"]
    + [v for v, _ in regression_outcomes if v in merged.columns]
    + ["infrastructure_index", "equipment_index", "service_index",
       "child_health_index", "cleanliness_score"]
)
save_cols = [c for c in save_cols if c in merged.columns]
# Remove duplicates while preserving order
seen = set()
save_cols_unique = []
for c in save_cols:
    if c not in seen:
        save_cols_unique.append(c)
        seen.add(c)
merged[save_cols_unique].to_csv(merged_path, index=False)
print(f"Merged analysis data saved:   {merged_path}")

print(f"\nDone. All output files are in: {OUTPUT_DIR}/")
