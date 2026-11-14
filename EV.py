"""
EV Charging Infrastructure Analysis - India
=============================================
Data-driven analysis of EV charging infrastructure patterns, gaps and
utilisation, aligned to the three research questions of the thesis:

  RQ1: Temporal usage patterns (peak hours, seasonal/festival demand)
  RQ2: Geographic gaps (EV demand vs. charging station supply, Gini coefficient)
  RQ3: Dashboard-ready summary tables and visualisations for infra planners

Inputs (place these in a ./data/ folder next to this script):
  1_VAHAN_EV_Registrations_India_2021_2024.csv
  2_OpenChargeMap_India_Stations.csv
  3_PlugShare_Tier2_India_Stations.csv
  4_BEE_FAMEII_Station_Utilisation.csv
  5_Census_India_District_Demographics.csv

Outputs are written to ./outputs/ (created automatically):
  PNG charts + summary CSVs used for the dashboard / thesis figures.

  Extensions added after supervisor feedback (original analysis unchanged):
  - extra charts for RQ1-RQ3
  - verification_report.csv : checks figures quoted in the thesis against the data
  - RQ4: EDGPI method (priority index + station allocation) validated against
    baselines/benchmarks -> rq4_*.csv / rq4_*.png / rq4_results_summary.txt
  - RQ4b: EDGPI compared with three existing published methods (CRITIC deficit
    index, entropy-TOPSIS, equity-first) -> rq4b_*.csv / rq4b_*.png

Usage:
  pip install pandas matplotlib seaborn numpy
  python ev_infra_analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

FILES = {
    "vahan": "1_VAHAN_EV_Registrations_India_2021_2024.csv",
    "ocm": "2_OpenChargeMap_India_Stations.csv",
    "plugshare": "3_PlugShare_Tier2_India_Stations.csv",
    "bee": "4_BEE_FAMEII_Station_Utilisation.csv",
    "census": "5_Census_India_District_Demographics.csv",
}


def load_data():
    """Load all five source CSVs into a dict of DataFrames."""
    dfs = {}
    for key, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        dfs[key] = pd.read_csv(path)
        print(f"Loaded {fname}: {dfs[key].shape[0]} rows, {dfs[key].shape[1]} cols")
    return dfs


def clean_data(dfs):
    """Light cleaning / type fixes shared across the analysis."""
    vahan, ocm, plugshare, bee, census = (
        dfs["vahan"], dfs["ocm"], dfs["plugshare"], dfs["bee"], dfs["census"]
    )

    # Standardise text columns
    for df in (vahan, ocm, plugshare, bee, census):
        for col in df.select_dtypes(include=["object", "str"]).columns:
            df[col] = df[col].astype(str).str.strip()

    # Parse dates
    bee["Month"] = pd.to_datetime(bee["Month"], format="%Y-%m")
    bee["Sanctioned_Date"] = pd.to_datetime(bee["Sanctioned_Date"], errors="coerce")
    bee["Commissioning_Date"] = pd.to_datetime(bee["Commissioning_Date"], errors="coerce")
    bee["Peak_Hour_IST"] = pd.to_datetime(bee["Peak_Hour_IST"], format="%H:%M").dt.hour

    ocm["Date_Last_Verified"] = pd.to_datetime(ocm["Date_Last_Verified"], errors="coerce")

    return vahan, ocm, plugshare, bee, census


# ---------------------------------------------------------------------------
# RQ1 - TEMPORAL DEMAND PATTERNS
# ---------------------------------------------------------------------------
def analyse_temporal_patterns(bee):
    print("\n=== RQ1: Temporal usage patterns ===")

    # --- Peak charging hour distribution ---
    peak_hour_counts = bee["Peak_Hour_IST"].value_counts().sort_index()
    peak_hour_counts.to_csv(os.path.join(OUT_DIR, "rq1_peak_hour_distribution.csv"))

    plt.figure(figsize=(9, 5))
    sns.barplot(x=peak_hour_counts.index, y=peak_hour_counts.values, color="#2E7D32")
    plt.title("Distribution of Reported Peak Charging Hour (BEE FAME-II Stations)")
    plt.xlabel("Hour of Day (IST)")
    plt.ylabel("Number of station-months")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_peak_hour_distribution.png"))
    plt.close()

    # --- Monthly / seasonal demand: total sessions & energy over time ---
    monthly = bee.groupby("Month").agg(
        Total_Sessions=("Total_Sessions", "sum"),
        Total_Energy_KWH=("Total_Energy_KWH", "sum"),
        Avg_Utilisation_Pct=("Utilisation_Rate_Pct", "mean"),
    ).reset_index()
    monthly.to_csv(os.path.join(OUT_DIR, "rq1_monthly_demand.csv"), index=False)

    plt.figure(figsize=(11, 5))
    plt.plot(monthly["Month"], monthly["Total_Sessions"], marker="o", color="#1565C0")
    plt.title("Total Monthly Charging Sessions (2023-2024)")
    plt.xlabel("Month")
    plt.ylabel("Total Sessions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_monthly_sessions_trend.png"))
    plt.close()

    # --- Festival season uplift (Oct-Nov vs. annual mean) ---
    monthly["Month_Num"] = monthly["Month"].dt.month
    annual_mean = monthly["Total_Sessions"].mean()
    festival = monthly[monthly["Month_Num"].isin([10, 11])]
    uplift_pct = (festival["Total_Sessions"].mean() / annual_mean - 1) * 100
    print(f"Festival-season (Oct-Nov) session uplift vs. annual mean: {uplift_pct:.1f}%")

    # --- Utilisation rate by state (metro vs tier-2 proxy) ---
    util_by_state = bee.groupby("State")["Utilisation_Rate_Pct"].mean().sort_values(ascending=False)
    util_by_state.to_csv(os.path.join(OUT_DIR, "rq1_utilisation_by_state.csv"))

    plt.figure(figsize=(9, 6))
    sns.barplot(x=util_by_state.values, y=util_by_state.index, color="#EF6C00")
    plt.title("Average Station Utilisation Rate (%) by State")
    plt.xlabel("Avg Utilisation Rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_utilisation_by_state.png"))
    plt.close()

    return monthly, util_by_state


# ---------------------------------------------------------------------------
# RQ2 - GEOGRAPHIC GAP ANALYSIS
# ---------------------------------------------------------------------------
def gini_coefficient(values):
    """Compute the Gini coefficient for a 1D array of non-negative values."""
    values = np.sort(np.array(values, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cum_values = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cum_values) / cum_values[-1]) / n


def analyse_geographic_gaps(vahan, ocm, plugshare, census):
    print("\n=== RQ2: Geographic gap analysis ===")

    # --- EV registrations by state (2021-2024) ---
    ev_by_state = vahan.groupby("State")["Registration_Count"].sum().sort_values(ascending=False)

    # --- Charging stations by state (combine OCM + PlugShare) ---
    stations_ocm = ocm.groupby("State_Province").size()
    stations_plugshare = plugshare.groupby("State").size()
    stations_by_state = stations_ocm.add(stations_plugshare, fill_value=0)
    stations_by_state.index.name = "State"

    gap = pd.DataFrame({
        "EV_Registrations": ev_by_state,
        "Charging_Stations": stations_by_state,
    }).fillna(0)
    gap["EVs_Per_Station"] = gap["EV_Registrations"] / gap["Charging_Stations"].replace(0, np.nan)
    gap = gap.sort_values("EVs_Per_Station", ascending=False)
    gap.to_csv(os.path.join(OUT_DIR, "rq2_state_ev_station_gap.csv"))

    print("States with the highest EV-to-station ratio (most under-served):")
    print(gap.head(5))

    # --- Gini coefficient of station distribution across states ---
    gini_stations = gini_coefficient(stations_by_state.values)
    gini_ev = gini_coefficient(ev_by_state.reindex(stations_by_state.index).fillna(0).values)
    print(f"Gini coefficient - charging station distribution across states: {gini_stations:.3f}")
    print(f"Gini coefficient - EV registration distribution across states: {gini_ev:.3f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=gap.reset_index(), x="Charging_Stations", y="EV_Registrations",
        hue="EVs_Per_Station", size="EVs_Per_Station", palette="rocket_r", legend=False
    )
    for _, row in gap.reset_index().iterrows():
        plt.text(row["Charging_Stations"], row["EV_Registrations"], row["State"], fontsize=7)
    plt.title("EV Registrations vs. Charging Stations by State")
    plt.xlabel("Number of Charging Stations")
    plt.ylabel("Total EV Registrations (2021-2024)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq2_ev_vs_stations_scatter.png"))
    plt.close()

    # --- District-level view (merged with Census, inner join on overlapping districts) ---
    ev_by_district = vahan.groupby(["State", "District"])["Registration_Count"].sum().reset_index()
    district_merge = ev_by_district.merge(census, on=["State", "District"], how="inner")
    district_merge["EV_Per_10k_Pop"] = (
        district_merge["Registration_Count"] / district_merge["Projected_Population_2024"] * 10000
    )
    district_merge.to_csv(os.path.join(OUT_DIR, "rq2_district_ev_demographics.csv"), index=False)
    print(f"Note: Census file covers {census['District'].nunique()} major districts; "
          f"{len(district_merge)} of these matched directly to VAHAN district names "
          f"(some districts use different spellings, e.g. 'Kanpur' vs 'Kanpur Nagar', "
          f"and would need manual name mapping for a full match).")

    return gap, district_merge


# ---------------------------------------------------------------------------
# RQ3 - DASHBOARD-READY SUMMARY VIEWS
# ---------------------------------------------------------------------------
def build_dashboard_summary(vahan, bee, gap):
    print("\n=== RQ3: Dashboard summary views ===")

    # Vehicle category share (2W/3W/4W/Bus/LCV) - highlights the 2W/3W infra mismatch
    veh_share = vahan.groupby("Vehicle_Category")["Registration_Count"].sum()
    veh_share_pct = (veh_share / veh_share.sum() * 100).sort_values(ascending=False)
    veh_share_pct.to_csv(os.path.join(OUT_DIR, "rq3_vehicle_category_share.csv"))

    plt.figure(figsize=(7, 7))
    plt.pie(veh_share_pct.values, labels=veh_share_pct.index, autopct="%1.1f%%",
            colors=sns.color_palette("Set2"))
    plt.title("Share of EV Registrations by Vehicle Category (2021-2024)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq3_vehicle_category_share.png"))
    plt.close()

    # Operator market share among BEE-tracked stations (operational efficiency lens)
    operator_util = bee.groupby("Operator")["Utilisation_Rate_Pct"].mean().sort_values(ascending=False)
    operator_util.to_csv(os.path.join(OUT_DIR, "rq3_operator_utilisation.csv"))

    # Combined top-line KPI table for a planner-facing dashboard
    kpis = pd.DataFrame({
        "Metric": [
            "Total EV Registrations (2021-2024)",
            "Total Charging Stations (OCM + PlugShare)",
            "Avg Station Utilisation Rate (%)",
            "Most Under-served State (EVs per station)",
            "Gini Coefficient (station distribution)",
        ],
        "Value": [
            int(vahan["Registration_Count"].sum()),
            int(gap["Charging_Stations"].sum()),
            round(bee["Utilisation_Rate_Pct"].mean(), 1),
            gap["EVs_Per_Station"].idxmax(),
            round(gini_coefficient(gap["Charging_Stations"].values), 3),
        ],
    })
    kpis.to_csv(os.path.join(OUT_DIR, "rq3_dashboard_kpis.csv"), index=False)
    print(kpis.to_string(index=False))

    return veh_share_pct, operator_util, kpis


# ===========================================================================
# ===========================================================================
#  EXTENSIONS (added for supervisor feedback)
#  Everything ABOVE this line is the original analysis, unchanged.
#  Everything below adds: (a) more charts for RQ1-RQ3, (b) a verification
#  report that checks the figures quoted in the thesis against the data, and
#  (c) RQ4 - an implemented and validated method for prioritising and
#  allocating new charging stations (the EDGPI method).
# ===========================================================================
# ===========================================================================
import heapq

# --- Method parameters (change here; all are reported in the thesis) --------
GUIDELINE_THRESHOLD = 23          # EVs per station; global median used in Section 3.6
SENSITIVITY_THRESHOLDS = [23, 30, 50]   # 23 (median), 30 (policy target in Sec 1.1), 50 (Sec 6.2)
BUDGET_FRACTION = 0.20            # new stations = 20% of existing stations (primary scenario)
BUDGET_SWEEP = [0.05, 0.10, 0.20, 0.50]
# EDGPI weights: demand pressure, utilisation, low-power (2W/3W) need, EV growth
EDGPI_WEIGHTS = {"Demand_Pressure": 0.50, "Utilisation": 0.20,
                 "LowPower_Need": 0.15, "Growth": 0.15}
N_RANDOM_TRIALS = 1000
N_WEIGHT_TRIALS = 1000
N_NOISE_TRIALS = 500
RNG_SEED = 42

STATE_ALIASES = {
    "nct of delhi": "delhi", "delhi nct": "delhi", "nct delhi": "delhi",
    "new delhi": "delhi", "orissa": "odisha", "pondicherry": "puducherry",
    "uttaranchal": "uttarakhand",
}


def norm_state(series):
    """Normalise state spellings so VAHAN / OCM / PlugShare / BEE can be joined."""
    s = (series.astype(str).str.strip().str.replace("&", "and", regex=False)
         .str.replace(r"\s+", " ", regex=True).str.lower())
    return s.map(lambda x: STATE_ALIASES.get(x, x)).str.title()


def _find_col(df, candidates):
    """Return the first column name in `candidates` that exists in df (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _minmax(s):
    s = s.astype(float)
    rng = s.max() - s.min()
    if not np.isfinite(rng) or rng == 0:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / rng


# ---------------------------------------------------------------------------
# A. MORE CHARTS - RQ1
# ---------------------------------------------------------------------------
def extend_temporal_charts(bee, monthly):
    print("\n=== Extra RQ1 charts ===")
    m = monthly.copy()
    m["Month_Num"] = m["Month"].dt.month

    # Rows per month = number of reporting station-months, so this removes the
    # effect of the number of reporting stations changing over time.
    per_row = bee.groupby("Month")["Total_Sessions"].mean().rename("Sessions_per_StationMonth")
    m = m.merge(per_row.reset_index(), on="Month", how="left")

    si_raw = m.groupby("Month_Num")["Total_Sessions"].mean()
    si_raw = si_raw / m["Total_Sessions"].mean()
    si_norm = m.groupby("Month_Num")["Sessions_per_StationMonth"].mean()
    si_norm = si_norm / m["Sessions_per_StationMonth"].mean()
    si = pd.DataFrame({"Seasonal_Index_Raw": si_raw, "Seasonal_Index_Per_StationMonth": si_norm})
    si.to_csv(os.path.join(OUT_DIR, "rq1_seasonal_index.csv"))

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.4
    x = np.arange(len(si))
    ax.bar(x - width / 2, si["Seasonal_Index_Raw"], width, label="Total sessions", color="#1565C0")
    ax.bar(x + width / 2, si["Seasonal_Index_Per_StationMonth"], width,
           label="Sessions per station-month", color="#EF6C00")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(2000, int(i), 1).strftime("%b") for i in si.index])
    ax.set_title("Seasonal Index of Charging Sessions by Calendar Month (1.0 = average month)")
    ax.set_ylabel("Seasonal index")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_seasonal_index.png"))
    plt.close()

    # Peak hour x month heatmap
    hm = (bee.assign(Month_Num=bee["Month"].dt.month)
             .pivot_table(index="Peak_Hour_IST", columns="Month_Num",
                          values="Total_Sessions", aggfunc="count", fill_value=0))
    plt.figure(figsize=(10, 6))
    sns.heatmap(hm, cmap="YlGnBu", annot=False, cbar_kws={"label": "Station-months"})
    plt.title("Reported Peak Charging Hour by Calendar Month")
    plt.xlabel("Month")
    plt.ylabel("Peak hour (IST)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_peak_hour_by_month_heatmap.png"))
    plt.close()

    # Utilisation spread by state
    order = bee.groupby("State")["Utilisation_Rate_Pct"].median().sort_values(ascending=False).index
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=bee, y="State", x="Utilisation_Rate_Pct", order=order, color="#90CAF9")
    plt.title("Distribution of Station Utilisation Rate (%) by State")
    plt.xlabel("Utilisation rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_utilisation_boxplot_by_state.png"))
    plt.close()

    # Energy per session over time
    e = bee.groupby("Month").agg(E=("Total_Energy_KWH", "sum"), S=("Total_Sessions", "sum")).reset_index()
    e["kWh_per_session"] = e["E"] / e["S"].replace(0, np.nan)
    plt.figure(figsize=(11, 4.5))
    plt.plot(e["Month"], e["kWh_per_session"], marker="o", color="#2E7D32")
    plt.title("Average Energy Delivered per Session (kWh)")
    plt.xlabel("Month")
    plt.ylabel("kWh per session")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_energy_per_session.png"))
    plt.close()
    return si


# ---------------------------------------------------------------------------
# B. MORE CHARTS - RQ2
# ---------------------------------------------------------------------------
def lorenz_points(values):
    v = np.sort(np.array(values, dtype=float))
    cum = np.insert(np.cumsum(v), 0, 0)
    return np.linspace(0, 1, len(cum)), cum / cum[-1]


def extend_geographic_charts(gap, district_merge):
    print("\n=== Extra RQ2 charts ===")
    # Lorenz curves
    plt.figure(figsize=(6.5, 6.5))
    for col, label, colour in [("Charging_Stations", "Charging stations", "#C62828"),
                               ("EV_Registrations", "EV registrations", "#1565C0")]:
        if gap[col].sum() > 0:
            x, y = lorenz_points(gap[col].values)
            plt.plot(x, y, label=f"{label} (Gini = {gini_coefficient(gap[col].values):.3f})", color=colour)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect equality")
    plt.title("Lorenz Curves Across States")
    plt.xlabel("Cumulative share of states (smallest to largest)")
    plt.ylabel("Cumulative share")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq2_lorenz_curves.png"))
    plt.close()

    # Top 10 most under-served states (EVs per station)
    top = gap["EVs_Per_Station"].dropna().sort_values(ascending=False).head(10)
    plt.figure(figsize=(9, 5.5))
    sns.barplot(x=top.values, y=top.index, color="#C62828")
    plt.axvline(GUIDELINE_THRESHOLD, color="black", linestyle="--", label=f"{GUIDELINE_THRESHOLD}:1 threshold")
    plt.title("Top 10 States by EVs per Charging Station")
    plt.xlabel("EVs per station")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq2_top_underserved_states.png"))
    plt.close()

    # States with no stations at all (invisible in a ratio, so listed explicitly)
    zero = gap[gap["Charging_Stations"] == 0]
    zero.to_csv(os.path.join(OUT_DIR, "rq2_states_with_zero_stations.csv"))
    print(f"States with EV registrations but zero listed stations: {len(zero)}")

    # District chart (only if the district join matched anything)
    if len(district_merge) > 0:
        d = district_merge.sort_values("EV_Per_10k_Pop", ascending=False).head(15)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=d, x="EV_Per_10k_Pop", y="District", color="#6A1B9A")
        plt.title("Top 15 Matched Districts by EV Registrations per 10,000 Population")
        plt.xlabel("EVs per 10,000 population")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "rq2_district_ev_per_10k.png"))
        plt.close()
    else:
        print("No districts matched between VAHAN and Census; district chart skipped.")


# ---------------------------------------------------------------------------
# C. MORE CHARTS - RQ3
# ---------------------------------------------------------------------------
def extend_dashboard_charts(vahan, operator_util):
    print("\n=== Extra RQ3 charts ===")
    plt.figure(figsize=(9, 5.5))
    sns.barplot(x=operator_util.values, y=operator_util.index, color="#00838F")
    plt.title("Average Utilisation Rate (%) by Operator")
    plt.xlabel("Average utilisation rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq3_operator_utilisation.png"))
    plt.close()

    v = vahan.assign(State=norm_state(vahan["State"]))
    top_states = v.groupby("State")["Registration_Count"].sum().nlargest(10).index
    mix = (v[v["State"].isin(top_states)]
           .pivot_table(index="State", columns="Vehicle_Category",
                        values="Registration_Count", aggfunc="sum", fill_value=0))
    mix = mix.div(mix.sum(axis=1), axis=0) * 100
    mix = mix.loc[top_states]
    mix.plot(kind="barh", stacked=True, figsize=(10, 6), colormap="Set2")
    plt.title("Vehicle Category Mix in the Top 10 EV States (%)")
    plt.xlabel("Share of registrations (%)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq3_vehicle_mix_top_states.png"))
    plt.close()


# ---------------------------------------------------------------------------
# D. VERIFICATION OF FIGURES QUOTED IN THE THESIS
# ---------------------------------------------------------------------------
def _status(claim, computed, tol, absolute=False):
    if computed is None or (isinstance(computed, float) and not np.isfinite(computed)):
        return "NOT COMPUTABLE"
    diff = abs(computed - claim) if absolute else abs(computed - claim) / abs(claim)
    return "CONSISTENT" if diff <= tol else "MISMATCH"


def verify_thesis_claims(vahan, ocm, plugshare, bee, monthly, gap, si):
    """Compare numbers quoted in the thesis with what the data and code produce."""
    print("\n=== Verification of thesis figures ===")
    rows = []

    def add(claim_text, where, claimed, computed, tol, note="", absolute=False):
        rows.append({"Claim": claim_text, "Thesis location": where, "Claimed": claimed,
                     "Computed from data": None if computed is None else round(float(computed), 3),
                     "Status": _status(claimed, computed, tol, absolute), "Note": note})

    util_state = bee.assign(State=norm_state(bee["State"])).groupby("State")["Utilisation_Rate_Pct"].mean()
    delhi_u = util_state.get("Delhi", np.nan)
    add("Delhi NCR average utilisation (%)", "Abstract, Table 4.1, 6.1", 72, delhi_u, 0.10,
        f"Highest state in data: {util_state.idxmax()} = {util_state.max():.1f}%")
    add("Tier-2 cluster average utilisation (%)", "Abstract, Table 4.1", 28, None, 0.10,
        "BEE file has no city-tier field; lowest state shown instead: "
        f"{util_state.idxmin()} = {util_state.min():.1f}%")

    add("October seasonal index", "Section 4.2, 6.1", 1.73,
        si["Seasonal_Index_Raw"].get(10, np.nan), 0.10,
        f"Per-station-month version = {si['Seasonal_Index_Per_StationMonth'].get(10, np.nan):.2f}")
    add("May seasonal index", "Section 6.1", 0.78,
        si["Seasonal_Index_Raw"].get(5, np.nan), 0.10,
        f"Per-station-month version = {si['Seasonal_Index_Per_StationMonth'].get(5, np.nan):.2f}. "
        "Note: an index of 0.78 is 22% BELOW the mean, not 78% below as written in 6.1.")

    m = monthly.copy()
    m["Month_Num"] = m["Month"].dt.month
    fest = m[m["Month_Num"].isin([10, 11])]["Total_Sessions"].mean() / m["Total_Sessions"].mean() - 1
    add("Oct-Nov uplift vs annual mean (%)", "Abstract, 5.2", 73, fest * 100, 0.10)

    add("Gini, station distribution", "Abstract, 4.3, 5.3", 0.64,
        gini_coefficient(gap["Charging_Stations"].values), 0.05,
        "Code computes Gini across STATES, unweighted. Thesis text says 'EV-demand-weighted district cells'.",
        absolute=True)

    top5 = gap["Charging_Stations"].nlargest(5).index
    add("Share of stations in top-5 states (%)", "Section 4.3", 58,
        gap.loc[top5, "Charging_Stations"].sum() / gap["Charging_Stations"].sum() * 100, 0.10,
        "Top-5 by stations in data: " + ", ".join(map(str, top5)))
    add("Share of registrations in those states (%)", "Section 4.3", 40,
        gap.loc[top5, "EV_Registrations"].sum() / gap["EV_Registrations"].sum() * 100, 0.10)

    cat = vahan["Vehicle_Category"].astype(str).str.upper()
    low = cat.str.contains(r"2W|3W|TWO|THREE|RICK", regex=True)
    add("2W+3W share of registrations (%)", "Section 1.1, 4.3",
        75, vahan.loc[low, "Registration_Count"].sum() / vahan["Registration_Count"].sum() * 100, 0.10)

    evening = bee["Peak_Hour_IST"].between(18, 21).mean() * 100
    add("Station-months with peak hour 18:00-21:00 (%)", "Abstract, 4.2", 50, evening, 0.30,
        "Thesis gives no percentage; 50% is a placeholder. Modal peak hour = "
        f"{int(bee['Peak_Hour_IST'].mode().iloc[0])}:00. Data holds ONE peak hour per station-month, "
        "so a weekday/weekend or bimodal profile (Fig 4.1) cannot be derived from it.")

    add("Total EV registrations", "Abstract ('1.97 lakh') vs 4.2 ('1.97 million')", 1_970_000,
        vahan["Registration_Count"].sum(), 0.10,
        "The thesis states 1.97 lakh in the abstract and 1.97 million in 4.2 - these differ 10x.")
    add("Total public stations", "Abstract, 1.1 ('about 25,000')", 25000,
        gap["Charging_Stations"].sum(), 0.20)

    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(OUT_DIR, "verification_report.csv"), index=False)
    print(rep[["Claim", "Claimed", "Computed from data", "Status"]].to_string(index=False))
    n_bad = (rep["Status"] != "CONSISTENT").sum()
    print(f"\n{n_bad} of {len(rep)} quoted figures are NOT confirmed by the data. "
          "See outputs/verification_report.csv before presenting them as findings.")
    return rep


# ---------------------------------------------------------------------------
# E. RQ4 - EDGPI METHOD (implementation + validation)
#     EDGPI = Equity-weighted Demand-Gap Priority Index
# ---------------------------------------------------------------------------
def build_state_table(vahan, ocm, plugshare, bee, station_noise=None):
    """One row per state: EVs, stations, utilisation, 2W/3W share, growth."""
    v = vahan.assign(S=norm_state(vahan["State"]))
    ev = v.groupby("S")["Registration_Count"].sum().rename("EV")

    stn = (ocm.assign(S=norm_state(ocm["State_Province"])).groupby("S").size()
           .add(plugshare.assign(S=norm_state(plugshare["State"])).groupby("S").size(), fill_value=0)
           ).rename("Stations")
    util = bee.assign(S=norm_state(bee["State"])).groupby("S")["Utilisation_Rate_Pct"].mean().rename("Util")

    cat = v["Vehicle_Category"].astype(str).str.upper()
    low_mask = cat.str.contains(r"2W|3W|TWO|THREE|RICK", regex=True)
    share = (v[low_mask].groupby("S")["Registration_Count"].sum() / ev).rename("Share2W3W")

    # Growth: latest year vs previous year, if VAHAN has a year column
    growth = pd.Series(dtype=float, name="Growth")
    ycol = _find_col(vahan, ["Year", "Registration_Year", "Reg_Year"])
    if ycol is not None:
        yr = pd.to_numeric(v[ycol], errors="coerce")
        years = sorted(yr.dropna().unique())
        if len(years) >= 2:
            last, prev = years[-1], years[-2]
            a = v[yr == last].groupby("S")["Registration_Count"].sum()
            b = v[yr == prev].groupby("S")["Registration_Count"].sum()
            growth = ((a / b.replace(0, np.nan)) - 1).rename("Growth")
            growth = growth.clip(upper=growth.quantile(0.95))

    tbl = pd.concat([ev, stn, util, share, growth], axis=1)
    tbl["Stations"] = tbl["Stations"].fillna(0)
    tbl = tbl[tbl["EV"].notna() & (tbl["EV"] > 0)].copy()
    if station_noise is not None:
        tbl["Stations"] = np.maximum(0, np.round(tbl["Stations"] * station_noise(len(tbl))))

    # Diagnostic: names that exist in only some sources (data-quality check)
    only_ev = set(ev.index) - set(stn.index)
    only_stn = set(stn.index) - set(ev.index)
    if station_noise is None:
        if only_ev:
            print(f"States in VAHAN but with no OCM/PlugShare stations: {sorted(only_ev)}")
        if only_stn:
            print(f"States with stations but no VAHAN registrations (check spelling): {sorted(only_stn)}")
    return tbl


def edgpi_scores(tbl, weights=None, bounds_hi=None):
    """Return components (0-1), EDGPI score and need multiplier m in [1, 2]."""
    w = dict(weights or EDGPI_WEIGHTS)
    t = tbl.copy()
    t["EV_per_station"] = t["EV"] / (t["Stations"] + 1)      # +1 keeps zero-station states in the ranking
    dp = np.log1p(t["EV_per_station"])
    hi = bounds_hi if bounds_hi is not None else dp.max()
    comp = pd.DataFrame(index=t.index)
    comp["Demand_Pressure"] = dp / hi
    comp["Utilisation"] = _minmax(t["Util"]).fillna(0.5) if "Util" in t else 0.5
    comp["LowPower_Need"] = _minmax(t["Share2W3W"]).fillna(0.5) if "Share2W3W" in t else 0.5
    if "Growth" in t and t["Growth"].notna().any():
        comp["Growth"] = _minmax(t["Growth"]).fillna(0.5)
    else:
        w.pop("Growth", None)                                   # drop and renormalise if not available
    tot = sum(w[k] for k in w)
    w = {k: v / tot for k, v in w.items()}
    comp = comp[list(w.keys())]
    score = sum(comp[k] * w[k] for k in w)
    need_cols = [k for k in w if k != "Demand_Pressure"]
    need_w = sum(w[k] for k in need_cols)
    need = sum(comp[k] * w[k] for k in need_cols) / need_w if need_w > 0 else pd.Series(0.0, index=t.index)
    return comp, score, 1 + need, w


# ---- allocation methods ----------------------------------------------------
def _largest_remainder(shares, k):
    raw = shares / shares.sum() * k
    base = np.floor(raw).astype(int)
    rem = k - base.sum()
    if rem > 0:
        order = np.argsort(-(raw - base))
        base[order[:rem]] += 1
    return base


def alloc_random(n, k, rng):
    return np.bincount(rng.integers(0, n, size=k), minlength=n)


def alloc_proportional(EV, k):
    return _largest_remainder(EV.astype(float), k)


def alloc_guideline(EV, S, k, threshold):
    """Established-guideline benchmark: repeatedly serve the state with the highest
    EVs-per-station while it exceeds the threshold; leftover budget goes pro rata to EVs."""
    add = np.zeros(len(EV), dtype=int)
    for _ in range(k):
        r = EV / (S + add + 1)
        i = int(np.argmax(r))
        if r[i] <= threshold:
            break
        add[i] += 1
    left = k - add.sum()
    if left > 0:
        add += _largest_remainder(EV.astype(float), left)
    return add


def alloc_greedy(EV, S, k, m):
    """Marginal-gain greedy. Objective: minimise sum_i m_i * EV_i * (EV_i / (S_i + 1)),
    i.e. need-weighted exposure of EVs to crowding. Each term is convex in S_i, so greedy is optimal.
    m = 1 gives the demand-only baseline; m = EDGPI multiplier gives the proposed method."""
    add = np.zeros(len(EV), dtype=int)

    def gain(i):
        s = S[i] + add[i]
        return m[i] * EV[i] ** 2 * (1.0 / (s + 1) - 1.0 / (s + 2))

    heap = [(-gain(i), i) for i in range(len(EV))]
    heapq.heapify(heap)
    for _ in range(k):
        _, i = heapq.heappop(heap)
        add[i] += 1
        heapq.heappush(heap, (-gain(i), i))
    return add


def evaluate_allocation(tbl, add, threshold, m=None):
    S = tbl["Stations"].values + add
    EV = tbl["EV"].values.astype(float)
    r = EV / (S + 1)
    high_util = (tbl["Util"] >= tbl["Util"].median()).values if "Util" in tbl else np.zeros(len(tbl), bool)
    out = {
        "EV_weighted_mean_EVs_per_station": float((EV * r).sum() / EV.sum()),
        "Gini_stations_per_EV": float(gini_coefficient((S + 1) / EV)),
        "Pct_EVs_in_compliant_states": float(EV[r <= threshold].sum() / EV.sum() * 100),
        "States_above_threshold": int((r > threshold).sum()),
        "Max_EVs_per_station": float(r.max()),
        "Pct_new_to_high_util_states": float(add[high_util].sum() / add.sum() * 100) if add.sum() > 0 else np.nan,
    }
    if m is not None:
        out["Need_weighted_exposure"] = float((m * EV * r).sum() / EV.sum())
    return out


def run_all_methods(tbl, k, m, threshold, rng, n_random=N_RANDOM_TRIALS):
    EV = tbl["EV"].values.astype(float)
    S = tbl["Stations"].values.astype(float)
    n = len(tbl)
    res, allocs = {}, {}
    allocs["No new stations"] = np.zeros(n, dtype=int)
    allocs["Proportional to EVs"] = alloc_proportional(EV, k)
    allocs["Guideline (worst-first to threshold)"] = alloc_guideline(EV, S, k, threshold)
    allocs["Demand-only greedy"] = alloc_greedy(EV, S, k, np.ones(n))
    allocs["EDGPI greedy (proposed)"] = alloc_greedy(EV, S, k, m.values)
    for name, a in allocs.items():
        res[name] = evaluate_allocation(tbl, a, threshold, m.values)
    # random baseline: mean and SD over many trials
    rand_rows = [evaluate_allocation(tbl, alloc_random(n, k, rng), threshold, m.values)
                 for _ in range(n_random)]
    rdf = pd.DataFrame(rand_rows)
    res["Random (mean of trials)"] = rdf.mean().to_dict()
    return pd.DataFrame(res).T, allocs, rdf


def run_edgpi_method(vahan, ocm, plugshare, bee):
    print("\n=== RQ4: EDGPI method - implementation and validation ===")
    rng = np.random.default_rng(RNG_SEED)
    tbl = build_state_table(vahan, ocm, plugshare, bee)
    comp, score, m, w_used = edgpi_scores(tbl)
    print("Weights used:", {k: round(v, 3) for k, v in w_used.items()})

    priority = pd.concat([tbl, comp.add_prefix("c_"), score.rename("EDGPI"), m.rename("Need_multiplier")], axis=1)
    priority["EV_per_station"] = tbl["EV"] / (tbl["Stations"] + 1)
    priority = priority.sort_values("EDGPI", ascending=False)
    priority.to_csv(os.path.join(OUT_DIR, "rq4_priority_index_by_state.csv"))
    print("Top 8 states by EDGPI:")
    print(priority[["EV", "Stations", "EV_per_station", "EDGPI"]].head(8).round(2))

    # ---- chart: stacked components --------------------------------------
    top = priority.head(15)
    contrib = pd.DataFrame({c: top["c_" + c] * w_used[c] for c in comp.columns})
    contrib.plot(kind="barh", stacked=True, figsize=(10, 7), colormap="viridis")
    plt.gca().invert_yaxis()
    plt.title("EDGPI Priority Score by State (weighted components, top 15)")
    plt.xlabel("EDGPI score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq4_priority_index_by_state.png"))
    plt.close()

    # ---- primary scenario ----------------------------------------------------
    k = max(1, int(round(BUDGET_FRACTION * tbl["Stations"].sum())))
    print(f"\nPrimary budget: {k} new stations ({BUDGET_FRACTION:.0%} of {int(tbl['Stations'].sum())} existing)")
    results, allocs, rdf = run_all_methods(tbl, k, m, GUIDELINE_THRESHOLD, rng)
    results.to_csv(os.path.join(OUT_DIR, "rq4_method_comparison.csv"))
    print(results.round(3).to_string())

    # significance vs random: share of random trials at least as good as the proposed method
    prop = results.loc["EDGPI greedy (proposed)"]
    p_vals = {
        "EV_weighted_mean_EVs_per_station": (rdf["EV_weighted_mean_EVs_per_station"] <= prop["EV_weighted_mean_EVs_per_station"]).mean(),
        "Gini_stations_per_EV": (rdf["Gini_stations_per_EV"] <= prop["Gini_stations_per_EV"]).mean(),
        "Pct_EVs_in_compliant_states": (rdf["Pct_EVs_in_compliant_states"] >= prop["Pct_EVs_in_compliant_states"]).mean(),
    }
    pd.Series(p_vals, name="share_of_random_trials_as_good_or_better").to_csv(
        os.path.join(OUT_DIR, "rq4_vs_random_pvalues.csv"))

    # chart: metric comparison with random SD error bars
    metrics = ["EV_weighted_mean_EVs_per_station", "Gini_stations_per_EV",
               "Pct_EVs_in_compliant_states", "Pct_new_to_high_util_states"]
    titles = ["EV-weighted mean EVs per station (lower is better)",
              "Gini of stations per EV (lower is better)",
              "% of EVs in states meeting threshold (higher is better)",
              "% of new stations to high-utilisation states"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    order = list(results.index)
    palette = ["#9E9E9E", "#90CAF9", "#FFB74D", "#A1887F", "#2E7D32", "#CE93D8"]
    for ax, met, ttl in zip(axes.ravel(), metrics, titles):
        vals = results.loc[order, met].values
        errs = [rdf[met].std() if o.startswith("Random") else 0 for o in order]
        ax.barh(order, vals, xerr=errs, color=palette[:len(order)])
        ax.set_title(ttl, fontsize=10)
        ax.invert_yaxis()
    plt.suptitle(f"Method Comparison - {k} New Stations, threshold {GUIDELINE_THRESHOLD}:1", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq4_method_comparison_metrics.png"))
    plt.close()

    # chart: where do the stations go (top 10 states by EVs)
    alloc_df = pd.DataFrame({k_: v for k_, v in allocs.items() if k_ != "No new stations"}, index=tbl.index)
    show = alloc_df.loc[tbl["EV"].nlargest(10).index]
    show.plot(kind="bar", figsize=(12, 6), colormap="tab10")
    plt.title("Allocation of New Stations to the 10 Largest EV States, by Method")
    plt.ylabel("New stations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq4_allocation_by_method.png"))
    plt.close()
    alloc_df.to_csv(os.path.join(OUT_DIR, "rq4_allocation_by_method.csv"))

    # chart: before vs after EVs per station
    after = tbl["EV"] / (tbl["Stations"] + allocs["EDGPI greedy (proposed)"] + 1)
    cmp_df = pd.DataFrame({"Before": tbl["EV"] / (tbl["Stations"] + 1), "After EDGPI": after}).loc[priority.index[:12]]
    cmp_df.plot(kind="bar", figsize=(11, 5.5), color=["#C62828", "#2E7D32"])
    plt.axhline(GUIDELINE_THRESHOLD, color="black", linestyle="--", label=f"{GUIDELINE_THRESHOLD}:1")
    plt.title("EVs per Station Before and After EDGPI Allocation (top 12 priority states)")
    plt.ylabel("EVs per station")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq4_before_after_evs_per_station.png"))
    plt.close()

    # ---- budget sweep ----------------------------------------------------------
    sweep = []
    for frac in BUDGET_SWEEP:
        kk = max(1, int(round(frac * tbl["Stations"].sum())))
        r, _, rd = run_all_methods(tbl, kk, m, GUIDELINE_THRESHOLD, rng, n_random=200)
        for meth in r.index:
            sweep.append({"Budget_fraction": frac, "K": kk, "Method": meth, **r.loc[meth].to_dict()})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(os.path.join(OUT_DIR, "rq4_budget_sweep.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for meth, g in sweep.groupby("Method"):
        axes[0].plot(g["Budget_fraction"] * 100, g["Gini_stations_per_EV"], marker="o", label=meth)
        axes[1].plot(g["Budget_fraction"] * 100, g["EV_weighted_mean_EVs_per_station"], marker="o", label=meth)
    axes[0].set_title("Gini of stations per EV vs budget")
    axes[1].set_title("EV-weighted mean EVs per station vs budget")
    for ax in axes:
        ax.set_xlabel("New stations as % of existing network")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq4_budget_sweep.png"))
    plt.close()

    # ---- threshold sensitivity -----------------------------------------------
    th_rows = []
    for th in SENSITIVITY_THRESHOLDS:
        r, _, _ = run_all_methods(tbl, k, m, th, rng, n_random=200)
        th_rows.append(r[["Pct_EVs_in_compliant_states", "States_above_threshold"]].assign(Threshold=th))
    pd.concat(th_rows).to_csv(os.path.join(OUT_DIR, "rq4_threshold_sensitivity.csv"))

    # ---- validation 1: weight sensitivity ----------------------------------
    base_top5 = set(priority.index[:5])
    ranks, jacc = [], []
    keys = list(w_used.keys())
    for _ in range(N_WEIGHT_TRIALS):
        wt = dict(zip(keys, rng.dirichlet(np.ones(len(keys)))))
        _, sc, _, _ = edgpi_scores(tbl, weights=wt)
        ranks.append(sc.rank(ascending=False))
        top5 = set(sc.nlargest(5).index)
        jacc.append(len(base_top5 & top5) / len(base_top5 | top5))
    rank_df = pd.DataFrame(ranks)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=rank_df[list(priority.index[:10])], orient="h", color="#80CBC4")
    plt.gca().invert_yaxis()
    plt.title("Rank Stability of Top 10 States Under 1,000 Random Weight Sets")
    plt.xlabel("Rank (1 = highest priority)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq4_weight_sensitivity.png"))
    plt.close()
    weight_stab = float(np.mean(jacc))

    # ---- validation 2: station-count uncertainty (coverage bias in OCM/PlugShare) --
    jn = []
    for _ in range(N_NOISE_TRIALS):
        noisy = build_state_table(vahan, ocm, plugshare, bee,
                                  station_noise=lambda n: rng.uniform(0.8, 1.2, size=n))
        _, sc, _, _ = edgpi_scores(noisy)
        top5 = set(sc.nlargest(5).index)
        jn.append(len(base_top5 & top5) / len(base_top5 | top5))
    noise_stab = float(np.mean(jn))

    # ---- validation 3: temporal hold-out against later utilisation --------------
    holdout = {}
    months = np.sort(bee["Month"].unique())
    if len(months) >= 6:
        cut = months[len(months) // 2]
        b_train = bee[bee["Month"] < cut]
        b_test = bee[bee["Month"] >= cut]
        tbl_tr = build_state_table(vahan, ocm, plugshare, b_train)
        _, sc_tr, _, _ = edgpi_scores(tbl_tr)
        ratio_tr = tbl_tr["EV"] / (tbl_tr["Stations"] + 1)
        u_test = b_test.assign(S=norm_state(b_test["State"])).groupby("S")["Utilisation_Rate_Pct"].mean()
        d = pd.DataFrame({"EDGPI_train": sc_tr, "Ratio_only": ratio_tr, "Util_test": u_test}).dropna()
        if len(d) >= 5:
            holdout = {
                "n_states": len(d),
                "spearman_EDGPI_vs_future_util": d["EDGPI_train"].corr(d["Util_test"], method="spearman"),
                "spearman_ratio_only_vs_future_util": d["Ratio_only"].corr(d["Util_test"], method="spearman"),
                "top5_overlap_EDGPI": len(set(d["EDGPI_train"].nlargest(5).index) & set(d["Util_test"].nlargest(5).index)),
                "top5_overlap_ratio_only": len(set(d["Ratio_only"].nlargest(5).index) & set(d["Util_test"].nlargest(5).index)),
            }
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, col, ttl in zip(axes, ["EDGPI_train", "Ratio_only"],
                                    ["EDGPI (trained on earlier months)", "EVs per station only"]):
                sns.regplot(data=d, x=col, y="Util_test", ax=ax, color="#1565C0")
                rho = d[col].corr(d["Util_test"], method="spearman")
                ax.set_title(f"{ttl}\nSpearman rho = {rho:.2f}")
                ax.set_ylabel("Utilisation in held-out later months (%)")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "rq4_holdout_validation.png"))
            plt.close()
            d.to_csv(os.path.join(OUT_DIR, "rq4_holdout_validation.csv"))
    else:
        print("Fewer than 6 months in BEE data: temporal hold-out validation skipped.")

    # ---- validation summary table (feeds Table 4.6 in the thesis) --------------
    vs = [("Share of random allocations as good or better: EV-weighted mean EVs/station", p_vals["EV_weighted_mean_EVs_per_station"]),
          ("Share of random allocations as good or better: Gini of stations per EV", p_vals["Gini_stations_per_EV"]),
          ("Weight sensitivity: mean top-5 Jaccard overlap (1,000 random weight sets)", weight_stab),
          ("Station-count uncertainty +/-20%: mean top-5 Jaccard overlap", noise_stab)]
    for key, label in [("n_states", "Hold-out: states compared"),
                       ("spearman_EDGPI_vs_future_util", "Hold-out: Spearman, EDGPI vs later utilisation"),
                       ("spearman_ratio_only_vs_future_util", "Hold-out: Spearman, EVs-per-station only vs later utilisation"),
                       ("top5_overlap_EDGPI", "Hold-out: top-5 overlap with later highest-utilisation states, EDGPI"),
                       ("top5_overlap_ratio_only", "Hold-out: top-5 overlap, EVs-per-station only")]:
        if key in holdout:
            vs.append((label, float(holdout[key])))
    pd.DataFrame(vs, columns=["Measure", "Value"]).round(3).to_csv(
        os.path.join(OUT_DIR, "rq4_validation_summary.csv"), index=False)

    # ---- summary text for the thesis --------------------------------------------
    lines = [
        "RQ4 RESULTS SUMMARY (auto-generated - paste these numbers into Section 4.5)",
        f"States analysed: {len(tbl)}; existing stations: {int(tbl['Stations'].sum())}; "
        f"budget K = {k} ({BUDGET_FRACTION:.0%}); threshold = {GUIDELINE_THRESHOLD}:1",
        "Weights: " + str({kk: round(v, 3) for kk, v in w_used.items()}),
        "Top-5 priority states: " + ", ".join(priority.index[:5]),
        "",
        results.round(3).to_string(),
        "",
        "Share of 1000 random allocations at least as good as EDGPI:",
        str(pd.Series(p_vals).round(3).to_dict()),
        f"Weight sensitivity: mean top-5 Jaccard overlap = {weight_stab:.2f}",
        f"Station-count uncertainty (+/-20%): mean top-5 Jaccard overlap = {noise_stab:.2f}",
        "Hold-out validation: " + (str({a: (round(b, 3) if isinstance(b, float) else b)
                                        for a, b in holdout.items()}) if holdout else "not run"),
    ]
    with open(os.path.join(OUT_DIR, "rq4_results_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines[-5:]))
    return priority, results, holdout


# ---------------------------------------------------------------------------
# F. RQ4b - COMPARISON WITH EXISTING PUBLISHED METHODS (supervisor request)
#    EDGPI is kept as the proposed method. Three existing, published approaches
#    are re-implemented from their described steps and run on the SAME state
#    table, budget and evaluation metrics:
#      M1  CRITIC-weighted deficit index  (Maric & Ravnjak, 2026, Trans. GIS,
#          doi:10.1111/tgis.70261) - objective CRITIC weights, score-based allocation
#      M2  Entropy-weight TOPSIS          (standard MCDM used in charging-station
#          planning, e.g. Selva Lopez et al., 2026, Energies 19(4):1041)
#      M3  Equity-first allocation        (lowest supply-per-EV first; the equity-
#          prioritising logic of Cai et al., 2024 and Huang et al., 2026)
#    No public source code was identified for these papers, so they are
#    re-implemented. Adaptations to state level are documented in the thesis
#    (Section 3.12). Indicators used by M1/M2 deliberately EXCLUDE the 2W/3W
#    share, which is EDGPI's own equity proxy, so the comparison is not circular.
# ---------------------------------------------------------------------------
M_EDGPI = "EDGPI greedy (proposed)"
M_CRITIC = "M1 CRITIC deficit index"
M_TOPSIS = "M2 Entropy-TOPSIS"
M_EQUITY = "M3 Equity-first (lowest supply per EV)"
COMPARE_METHODS = [M_EDGPI, M_CRITIC, M_TOPSIS, M_EQUITY]
N_NOISE_TRIALS_CMP = 200


def _indicator_matrix(tbl):
    """Benefit-type indicators shared by M1 and M2 (all: higher = greater deficit/need)."""
    X = pd.DataFrame(index=tbl.index)
    X["EVs_per_station"] = tbl["EV"] / (tbl["Stations"] + 1)
    X["Demand_volume_logEV"] = np.log1p(tbl["EV"])
    if "Util" in tbl:
        X["Utilisation"] = tbl["Util"]
    if "Growth" in tbl and tbl["Growth"].notna().any():
        X["Growth"] = tbl["Growth"]
    X = X.apply(lambda c: c.fillna(c.median()) if c.notna().any() else c.fillna(0.0))
    return X


def critic_weights(X):
    """CRITIC: weight_j proportional to sd_j * sum_k (1 - corr_jk) on min-max scaled data."""
    Z = X.apply(_minmax)
    sd = Z.std(ddof=1).values
    corr = np.corrcoef(Z.values, rowvar=False)
    corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)
    info = sd * (1 - corr).sum(axis=1)
    w = info / info.sum() if info.sum() > 0 else np.full(len(sd), 1 / len(sd))
    return pd.Series(w, index=X.columns)


def critic_scores(tbl):
    X = _indicator_matrix(tbl)
    w = critic_weights(X)
    return (X.apply(_minmax) * w).sum(axis=1), w


def entropy_weights(X):
    Z = X.apply(_minmax) + 1e-9
    P = Z / Z.sum()
    n = len(Z)
    e = -(P * np.log(P)).sum() / np.log(n)
    d = 1 - e
    return d / d.sum() if d.sum() > 0 else pd.Series(1 / len(d), index=d.index)


def topsis_scores(tbl):
    """Entropy-weighted TOPSIS closeness coefficient (1 = closest to ideal 'highest deficit')."""
    X = _indicator_matrix(tbl)
    w = entropy_weights(X)
    R = X / np.sqrt((X ** 2).sum())
    V = R * w
    best, worst = V.max(), V.min()
    d_plus = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - worst) ** 2).sum(axis=1))
    return d_minus / (d_plus + d_minus).replace(0, np.nan), w


def equity_scores(tbl):
    """Ranking score for the equity-first rule: current EVs per station (lowest supply per EV first)."""
    return tbl["EV"] / (tbl["Stations"] + 1), None


def alloc_by_score(score, k):
    """Share K stations in proportion to a non-negative deficit score (largest remainder)."""
    s = score.fillna(0).clip(lower=0).values.astype(float) + 1e-9
    return _largest_remainder(s, k)


def alloc_equity_first(EV, S, k):
    """Each new station goes to the state with the fewest stations per EV, (S+a+1)/EV."""
    add = np.zeros(len(EV), dtype=int)
    heap = [((S[i] + 1) / EV[i], i) for i in range(len(EV))]
    heapq.heapify(heap)
    for _ in range(k):
        _, i = heapq.heappop(heap)
        add[i] += 1
        heapq.heappush(heap, ((S[i] + add[i] + 1) / EV[i], i))
    return add


def allocations_for_methods(tbl, k, m):
    """Allocations for the proposed method and the three existing methods."""
    EV = tbl["EV"].values.astype(float)
    S = tbl["Stations"].values.astype(float)
    sc1, _ = critic_scores(tbl)
    sc2, _ = topsis_scores(tbl)
    return {
        M_EDGPI: alloc_greedy(EV, S, k, m.values),
        M_CRITIC: alloc_by_score(sc1, k),
        M_TOPSIS: alloc_by_score(sc2, k),
        M_EQUITY: alloc_equity_first(EV, S, k),
    }


def method_scores(tbl):
    """Priority scores used for RANKING comparisons (higher = higher priority)."""
    _, e_score, _, _ = edgpi_scores(tbl)
    return {M_EDGPI: e_score, M_CRITIC: critic_scores(tbl)[0],
            M_TOPSIS: topsis_scores(tbl)[0], M_EQUITY: equity_scores(tbl)[0]}


def _top5_jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


def run_published_method_comparison(vahan, ocm, plugshare, bee):
    print("\n=== RQ4b: EDGPI vs existing published methods ===")
    rng = np.random.default_rng(RNG_SEED + 1)
    tbl = build_state_table(vahan, ocm, plugshare, bee)
    _, _, m, _ = edgpi_scores(tbl)
    k = max(1, int(round(BUDGET_FRACTION * tbl["Stations"].sum())))

    # ---- 1. allocation outcomes (same metrics, budget, data) ----------------
    # Need_weighted_exposure is EDGPI's own objective, so m is NOT passed here.
    allocs = allocations_for_methods(tbl, k, m)
    rows = {name: evaluate_allocation(tbl, a, GUIDELINE_THRESHOLD) for name, a in allocs.items()}
    rows["No new stations (reference)"] = evaluate_allocation(tbl, np.zeros(len(tbl), int), GUIDELINE_THRESHOLD)
    res = pd.DataFrame(rows).T.loc[COMPARE_METHODS + ["No new stations (reference)"]]
    res.to_csv(os.path.join(OUT_DIR, "rq4b_method_comparison.csv"))
    pd.DataFrame(allocs, index=tbl.index).to_csv(os.path.join(OUT_DIR, "rq4b_allocation_by_method.csv"))
    print(f"Budget K = {k} new stations; threshold {GUIDELINE_THRESHOLD}:1")
    print(res.round(3).to_string())

    # ---- 2. indicator weights actually used by M1 / M2 ----------------------
    X = _indicator_matrix(tbl)
    wts = pd.DataFrame({"CRITIC (M1)": critic_weights(X), "Entropy (M2)": entropy_weights(X)})
    wts.to_csv(os.path.join(OUT_DIR, "rq4b_objective_weights.csv"))

    # ---- 3. ranking agreement with EDGPI --------------------------------------
    sc = method_scores(tbl)
    agree = []
    for name in COMPARE_METHODS[1:]:
        agree.append({"Method": name,
                      "Spearman_vs_EDGPI": sc[M_EDGPI].corr(sc[name], method="spearman"),
                      "Top5_overlap_with_EDGPI": len(set(sc[M_EDGPI].nlargest(5).index) & set(sc[name].nlargest(5).index)),
                      "Top10_overlap_with_EDGPI": len(set(sc[M_EDGPI].nlargest(10).index) & set(sc[name].nlargest(10).index))})
    pd.DataFrame(agree).to_csv(os.path.join(OUT_DIR, "rq4b_rank_agreement.csv"), index=False)
    pd.DataFrame(sc).sort_values(M_EDGPI, ascending=False).to_csv(os.path.join(OUT_DIR, "rq4b_scores_by_state.csv"))

    # ---- 4. budget sweep -------------------------------------------------------
    sweep = []
    for frac in BUDGET_SWEEP:
        kk = max(1, int(round(frac * tbl["Stations"].sum())))
        for name, a in allocations_for_methods(tbl, kk, m).items():
            sweep.append({"Budget_fraction": frac, "K": kk, "Method": name,
                          **evaluate_allocation(tbl, a, GUIDELINE_THRESHOLD)})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(os.path.join(OUT_DIR, "rq4b_budget_sweep.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for meth, g in sweep.groupby("Method"):
        axes[0].plot(g["Budget_fraction"] * 100, g["Gini_stations_per_EV"], marker="o", label=meth)
        axes[1].plot(g["Budget_fraction"] * 100, g["EV_weighted_mean_EVs_per_station"], marker="o", label=meth)
    axes[0].set_title("Gini of stations per EV vs budget (lower = more equal)")
    axes[1].set_title("EV-weighted mean EVs per station vs budget (lower = better)")
    for ax in axes:
        ax.set_xlabel("New stations as % of existing network")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq4b_budget_sweep.png"))
    plt.close()

    # ---- 5. temporal hold-out: does each method's ranking anticipate later utilisation? ----
    ho_rows = []
    months = np.sort(bee["Month"].unique())
    if len(months) >= 6:
        cut = months[len(months) // 2]
        b_train, b_test = bee[bee["Month"] < cut], bee[bee["Month"] >= cut]
        tbl_tr = build_state_table(vahan, ocm, plugshare, b_train)
        u_test = b_test.assign(S=norm_state(b_test["State"])).groupby("S")["Utilisation_Rate_Pct"].mean()
        sc_tr = method_scores(tbl_tr)
        for name in COMPARE_METHODS:
            d = pd.DataFrame({"s": sc_tr[name], "u": u_test}).dropna()
            if len(d) >= 5:
                ho_rows.append({"Method": name, "n_states": len(d),
                                "Spearman_vs_later_util": d["s"].corr(d["u"], method="spearman"),
                                "Top5_overlap_with_later_top5_util":
                                    len(set(d["s"].nlargest(5).index) & set(d["u"].nlargest(5).index))})
        pd.DataFrame(ho_rows).to_csv(os.path.join(OUT_DIR, "rq4b_holdout.csv"), index=False)
    else:
        print("Fewer than 6 BEE months: hold-out comparison skipped.")

    # ---- 6. robustness: top-5 stability under +/-20% station-count noise ---------
    base_top5 = {n: set(sc[n].nlargest(5).index) for n in COMPARE_METHODS}
    jac = {n: [] for n in COMPARE_METHODS}
    for _ in range(N_NOISE_TRIALS_CMP):
        noisy = build_state_table(vahan, ocm, plugshare, bee,
                                  station_noise=lambda n: rng.uniform(0.8, 1.2, size=n))
        scn = method_scores(noisy)
        for n in COMPARE_METHODS:
            jac[n].append(_top5_jaccard(base_top5[n], scn[n].nlargest(5).index))
    rob = pd.DataFrame({"Method": COMPARE_METHODS,
                        "Mean_top5_Jaccard_under_noise": [float(np.nanmean(jac[n])) for n in COMPARE_METHODS]})
    rob.to_csv(os.path.join(OUT_DIR, "rq4b_robustness.csv"), index=False)

    # ---- 7. chart: metrics side by side --------------------------------------------
    metrics = ["EV_weighted_mean_EVs_per_station", "Gini_stations_per_EV",
               "Pct_EVs_in_compliant_states", "Pct_new_to_high_util_states"]
    titles = ["EV-weighted mean EVs per station (lower better)", "Gini of stations per EV (lower better)",
              "% EVs in states meeting threshold (higher better)", "% new stations to high-utilisation states"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, met, ttl in zip(axes.ravel(), metrics, titles):
        ax.barh(COMPARE_METHODS, res.loc[COMPARE_METHODS, met].values,
                color=["#2E7D32", "#1565C0", "#EF6C00", "#8E24AA"])
        ax.set_title(ttl, fontsize=10)
        ax.invert_yaxis()
    plt.suptitle(f"EDGPI vs Existing Methods - {k} New Stations, threshold {GUIDELINE_THRESHOLD}:1", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq4b_method_comparison_metrics.png"))
    plt.close()

    lines = ["RQ4b RESULTS SUMMARY (auto-generated - paste into Section 4.6)",
             f"States: {len(tbl)}; existing stations: {int(tbl['Stations'].sum())}; K = {k}",
             "", res.round(3).to_string(), "",
             "Indicator weights (M1 CRITIC, M2 Entropy):", wts.round(3).to_string(), "",
             "Ranking agreement with EDGPI:", pd.DataFrame(agree).round(3).to_string(index=False), "",
             "Hold-out (earlier months -> later utilisation):",
             pd.DataFrame(ho_rows).round(3).to_string(index=False) if ho_rows else "not run", "",
             "Robustness (+/-20% station counts):", rob.round(3).to_string(index=False)]
    with open(os.path.join(OUT_DIR, "rq4b_results_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-12:]))
    return res, agree, ho_rows, rob


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    dfs = load_data()
    vahan, ocm, plugshare, bee, census = clean_data(dfs)

    # --- original analysis (unchanged) ---
    monthly, util_by_state = analyse_temporal_patterns(bee)
    gap, district_merge = analyse_geographic_gaps(vahan, ocm, plugshare, census)
    veh_share_pct, operator_util, kpis = build_dashboard_summary(vahan, bee, gap)

    # --- extensions: more charts, verification, RQ4 method ---
    si = extend_temporal_charts(bee, monthly)
    extend_geographic_charts(gap, district_merge)
    extend_dashboard_charts(vahan, operator_util)
    verify_thesis_claims(vahan, ocm, plugshare, bee, monthly, gap, si)
    run_edgpi_method(vahan, ocm, plugshare, bee)
    run_published_method_comparison(vahan, ocm, plugshare, bee)   # RQ4b: EDGPI vs existing methods

    print(f"\nAll outputs (charts + CSV summaries) saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
