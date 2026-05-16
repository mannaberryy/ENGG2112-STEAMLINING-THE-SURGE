"""
ICU Triage Simulation — Streamlining the Surge (v3)
ENGG2112 Group 3 | Aayan Shukla

Rebuilt to exactly match the deployed binary triage matrix:

  Model 1: severe / non_severe
  Model 2: high_risk / low_risk

  Priority 4 — Severe    + High Risk  → ICU escalation candidate (HIGHEST)
  Priority 3 — Severe    + Low Risk   → HDU review
  Priority 2 — Non-severe + High Risk  → Increased monitoring
  Priority 1 — Non-severe + Low Risk   → Routine ward (LOWEST)

NHS England hospital activity data (Alpha surge, Jan 2021) provides
real-world capacity constraints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# 1. NHS CAPACITY CONSTANTS
#    Source: NHS England Weekly Admissions and Beds, up to 6 April 2021
# =============================================================================

NHS_CAPACITY = {
    "baseline_icu_beds":          1420,   # Nov 2020 pre-surge CC beds
    "peak_icu_beds_occupied":     4096,   # 22-Jan-2021 crisis peak
    "peak_mv_beds_occupied":      3736,   # 24-Jan-2021 MV peak
    "peak_total_beds":           34336,   # 18-Jan-2021 total beds peak
    "peak_daily_admissions":      4134,   # 12-Jan-2021 admissions peak
    "baseline_daily_admissions":    53,   # Aug 2020 low prevalence baseline
    "total_icu_capacity":         4500,   # NHS England pre-pandemic adult CC ceiling
    "total_mv_capacity":          4000,
}

# =============================================================================
# 2. SURGE SCENARIO DEFINITIONS
#    Anchored to NHS Alpha surge data (Dec 2020 – Feb 2021)
# =============================================================================

SCENARIOS = {
    "Moderate Surge": {
        "daily_admissions":       1300,   # 01-Dec-2020 actual
        "icu_occupied_baseline":  1343,   # 01-Dec-2020 CC beds occupied
        "duration_days":            30,
        "real_world_anchor":  "01-Dec-2020",
        "description": "Early Alpha wave — system under pressure, triage decisions begin to matter",
    },
    "Severe Surge": {
        "daily_admissions":       2900,   # 29-Dec-2020 actual
        "icu_occupied_baseline":  2007,   # 29-Dec-2020 CC beds occupied
        "duration_days":            30,
        "real_world_anchor":  "29-Dec-2020",
        "description": "Alpha wave peak build-up — ICU near capacity, staffing strained",
    },
    "Crisis": {
        "daily_admissions":       4134,   # 12-Jan-2021 actual peak
        "icu_occupied_baseline":  3724,   # 12-Jan-2021 CC beds occupied
        "duration_days":            30,
        "real_world_anchor":  "12-Jan-2021",
        "description": "Crisis peak — ICU overwhelmed, triage determines who receives critical care",
    },
}

# =============================================================================
# 3. PATIENT DATA CLASS
# =============================================================================

@dataclass
class Patient:
    patient_id:      int
    severity:        str   # "severe" / "non_severe"  — Model 1 output
    deterioration:   str   # "high_risk" / "low_risk" — Model 2 output
    priority_score:  int   # 1–4 from triage matrix (4 = highest escalation)
    icu_allocated:   bool  = False
    outcome:         str   = "Pending"

# =============================================================================
# 4. BINARY TRIAGE MATRIX
#    Matches app matrix exactly (index.html, triage-matrix-table)
#
#    Score 4 — Severe    + High Risk  → ICU escalation candidate (highest)
#    Score 3 — Severe    + Low Risk   → HDU review
#    Score 2 — Non-severe + High Risk  → Increased monitoring
#    Score 1 — Non-severe + Low Risk   → Routine ward (lowest)
# =============================================================================

TRIAGE_MATRIX = {
    ("severe",     "high_risk"):  (4, "ICU escalation candidate / urgent clinical review"),
    ("severe",     "low_risk"):   (3, "HDU review and frequent reassessment"),
    ("non_severe", "high_risk"):  (2, "Increased monitoring and repeat biomarker assessment"),
    ("non_severe", "low_risk"):   (1, "Routine ward monitoring"),
}

def triage_decision(severity: str, deterioration: str) -> Tuple[int, str]:
    return TRIAGE_MATRIX.get((severity, deterioration), (1, "Routine ward monitoring"))

# =============================================================================
# 5. SYNTHETIC PATIENT GENERATION
#    Distributions derived from ST001849 dataset structure
# =============================================================================

SEVERITY_DIST     = {"severe": 0.45, "non_severe": 0.55}
DETERIORATION_DIST = {"high_risk": 0.40, "low_risk": 0.60}

def generate_synthetic_patients(n: int) -> List[Patient]:
    severities     = np.random.choice(list(SEVERITY_DIST.keys()),     size=n, p=list(SEVERITY_DIST.values()))
    deteriorations = np.random.choice(list(DETERIORATION_DIST.keys()), size=n, p=list(DETERIORATION_DIST.values()))
    patients = []
    for i in range(n):
        score, _ = triage_decision(severities[i], deteriorations[i])
        patients.append(Patient(
            patient_id=i,
            severity=severities[i],
            deterioration=deteriorations[i],
            priority_score=score,
        ))
    return patients

# =============================================================================
# 6. ICU LENGTH OF STAY
#    Source: published COVID-19 critical care LOS data
# =============================================================================

LOS_DAYS = {
    "severe":     {"mean": 8, "std": 3},
    "non_severe": {"mean": 3, "std": 1},
}

def sample_los(severity: str) -> int:
    p = LOS_DAYS[severity]
    return max(1, int(np.random.normal(p["mean"], p["std"])))

# Mortality probability if ICU is denied when needed
UNTREATED_MORTALITY = {
    ("severe",     "high_risk"):  0.75,
    ("severe",     "low_risk"):   0.40,
    ("non_severe", "high_risk"):  0.25,
    ("non_severe", "low_risk"):   0.05,
}

# ICU needed if priority score >= 3 (severe patients)
def needs_icu(patient: Patient) -> bool:
    return patient.priority_score >= 3

# =============================================================================
# 7. TRIAGE STRATEGIES
# =============================================================================

def strategy_severity_only(patients: List[Patient]) -> List[Patient]:
    """Baseline — severity class alone, approximating NEWS2/SOFA behaviour."""
    order = {"severe": 0, "non_severe": 1}
    return sorted(patients, key=lambda p: order[p.severity])

def strategy_deterioration_only(patients: List[Patient]) -> List[Patient]:
    """Deterioration risk alone."""
    order = {"high_risk": 0, "low_risk": 1}
    return sorted(patients, key=lambda p: order[p.deterioration])

def strategy_combined_matrix(patients: List[Patient]) -> List[Patient]:
    """Combined matrix — Priority 4 first (highest escalation first)."""
    return sorted(patients, key=lambda p: -p.priority_score)

STRATEGIES = {
    "Severity Only (Baseline)": strategy_severity_only,
    "Deterioration Only":       strategy_deterioration_only,
    "Combined Matrix":          strategy_combined_matrix,
}

# =============================================================================
# 8. CORE SIMULATION LOOP
# =============================================================================

def run_simulation(scenario_name: str, strategy_name: str) -> dict:
    scenario    = SCENARIOS[scenario_name]
    strategy_fn = STRATEGIES[strategy_name]
    days        = scenario["duration_days"]

    total_icu_capacity = NHS_CAPACITY["total_icu_capacity"]
    icu_occupied       = scenario["icu_occupied_baseline"]

    total_patients        = 0
    preventable_deaths    = 0
    survivors             = 0
    under_triage_count    = 0
    icu_utilisation_daily = []
    icu_patients          = []   # (severity, days_remaining)

    for day in range(days):
        # Discharge patients whose LOS has ended
        icu_patients = [(sev, d - 1) for sev, d in icu_patients if d > 1]
        icu_occupied = len(icu_patients)

        # Daily admissions with ±8% Gaussian noise
        n_today = max(1, int(np.random.normal(
            scenario["daily_admissions"],
            scenario["daily_admissions"] * 0.08
        )))
        patients_today = generate_synthetic_patients(n_today)
        total_patients += n_today

        ordered = strategy_fn(patients_today)

        for patient in ordered:
            if needs_icu(patient):
                if icu_occupied < total_icu_capacity:
                    los = sample_los(patient.severity)
                    icu_patients.append((patient.severity, los))
                    icu_occupied += 1
                    patient.icu_allocated = True
                    patient.outcome = "Survived"
                    survivors += 1
                else:
                    mort = UNTREATED_MORTALITY.get((patient.severity, patient.deterioration), 0.10)
                    if np.random.random() < mort:
                        patient.outcome = "Preventable Death"
                        preventable_deaths += 1
                    else:
                        patient.outcome = "Survived"
                        survivors += 1
                    under_triage_count += 1
            else:
                patient.outcome = "Survived"
                survivors += 1

        utilisation = min(icu_occupied / total_icu_capacity, 1.0)
        icu_utilisation_daily.append(utilisation)

    return {
        "scenario":              scenario_name,
        "strategy":              strategy_name,
        "total_patients":        total_patients,
        "preventable_deaths":    preventable_deaths,
        "survivors":             survivors,
        "mortality_rate":        preventable_deaths / total_patients if total_patients > 0 else 0,
        "mean_icu_utilisation":  np.mean(icu_utilisation_daily),
        "peak_icu_utilisation":  np.max(icu_utilisation_daily),
        "under_triage_count":    under_triage_count,
        "icu_utilisation_daily": icu_utilisation_daily,
    }

# =============================================================================
# 9. RUN ALL 9 COMBINATIONS (3 scenarios × 3 strategies)
# =============================================================================

def run_all_simulations() -> pd.DataFrame:
    results = []
    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            print(f"  Running: {scenario} | {strategy}...")
            results.append(run_simulation(scenario, strategy))
    return pd.DataFrame(results)

# =============================================================================
# 10. VISUALISATIONS
# =============================================================================

def plot_results(df: pd.DataFrame, save_dir: str = "/home/claude/simulation"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "ICU Triage Simulation — Streamlining the Surge\n"
        "Alpha Surge Scenarios (NHS England Data, Dec 2020 – Jan 2021)",
        fontsize=14, fontweight="bold", y=1.02
    )

    scenarios  = list(SCENARIOS.keys())
    strategies = list(STRATEGIES.keys())
    colours    = ["#2196F3", "#FF9800", "#4CAF50"]
    x          = np.arange(len(scenarios))
    width      = 0.25

    # Plot 1: Preventable Deaths
    ax1 = axes[0]
    for i, (strat, col) in enumerate(zip(strategies, colours)):
        vals = [df[(df.scenario == s) & (df.strategy == strat)]["preventable_deaths"].values[0] for s in scenarios]
        bars = ax1.bar(x + i * width, vals, width, label=strat, color=col, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom", fontsize=8)
    ax1.set_xlabel("Surge Scenario")
    ax1.set_ylabel("Preventable Deaths (30-day window)")
    ax1.set_title("Preventable Deaths by Triage Strategy")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(scenarios, fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Mean ICU Utilisation
    ax2 = axes[1]
    for i, (strat, col) in enumerate(zip(strategies, colours)):
        vals = [df[(df.scenario == s) & (df.strategy == strat)]["mean_icu_utilisation"].values[0] * 100 for s in scenarios]
        bars = ax2.bar(x + i * width, vals, width, label=strat, color=col, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    ax2.axhline(y=85, color="orange", linestyle="--", linewidth=1.5, label="Severe threshold (85%)")
    ax2.axhline(y=95, color="red",    linestyle="--", linewidth=1.5, label="Crisis threshold (95%)")
    ax2.set_xlabel("Surge Scenario")
    ax2.set_ylabel("Mean ICU Utilisation (%)")
    ax2.set_title("ICU Utilisation Rate by Strategy")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(scenarios, fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 115)

    # Plot 3: Mortality Rate Heatmap
    ax3 = axes[2]
    heatmap_data = df.pivot(index="strategy", columns="scenario", values="mortality_rate") * 100
    heatmap_data = heatmap_data[scenarios]
    sns.heatmap(heatmap_data, ax=ax3, annot=True, fmt=".2f", cmap="RdYlGn_r",
                cbar_kws={"label": "Mortality Rate (%)"}, linewidths=0.5)
    ax3.set_title("Mortality Rate Heatmap (%)")
    ax3.set_xlabel("Surge Scenario")
    ax3.set_ylabel("Triage Strategy")
    ax3.tick_params(axis="x", rotation=15)
    ax3.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = f"{save_dir}/simulation_results_v3.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_utilisation_over_time(df: pd.DataFrame, save_dir: str = "/home/claude/simulation"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Daily ICU Utilisation — 30-Day Simulation Window", fontsize=13, fontweight="bold")
    colours = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax, scenario in zip(axes, SCENARIOS.keys()):
        for strat, col in zip(STRATEGIES.keys(), colours):
            row   = df[(df.scenario == scenario) & (df.strategy == strat)].iloc[0]
            daily = row["icu_utilisation_daily"]
            ax.plot(range(1, len(daily)+1), [v*100 for v in daily],
                    label=strat, color=col, linewidth=2)
        ax.axhline(y=85, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="85% threshold")
        ax.axhline(y=95, color="red",    linestyle="--", linewidth=1, alpha=0.7, label="95% threshold")
        ax.fill_between(range(1, 31), 85, 95,  alpha=0.05, color="orange")
        ax.fill_between(range(1, 31), 95, 105, alpha=0.05, color="red")
        ax.set_xlabel("Simulation Day")
        ax.set_ylabel("ICU Utilisation (%)")
        ax.set_title(scenario)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    path = f"{save_dir}/icu_utilisation_over_time_v3.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# =============================================================================
# 11. SUMMARY
# =============================================================================

def print_summary(df: pd.DataFrame):
    print("\n" + "="*80)
    print("SIMULATION RESULTS SUMMARY — 30-Day Window per Scenario")
    print("="*80)
    summary = df[["scenario", "strategy", "total_patients", "preventable_deaths",
                  "mortality_rate", "mean_icu_utilisation", "under_triage_count"]].copy()
    summary["mortality_rate"]       = (summary["mortality_rate"] * 100).round(2).astype(str) + "%"
    summary["mean_icu_utilisation"] = (summary["mean_icu_utilisation"] * 100).round(1).astype(str) + "%"
    print(summary.to_string(index=False))

    print("\nKEY FINDINGS:")
    for scenario in SCENARIOS.keys():
        subset  = df[df.scenario == scenario]
        best    = subset.loc[subset["preventable_deaths"].idxmin()]
        worst   = subset.loc[subset["preventable_deaths"].idxmax()]
        diff    = worst["preventable_deaths"] - best["preventable_deaths"]
        print(f"  {scenario}: '{best['strategy']}' saves {diff} lives vs '{worst['strategy']}'")

    print("\nCOMBINED MATRIX vs SEVERITY-ONLY BASELINE:")
    for scenario in SCENARIOS.keys():
        combined = df[(df.scenario == scenario) & (df.strategy == "Combined Matrix")]["preventable_deaths"].values[0]
        baseline = df[(df.scenario == scenario) & (df.strategy == "Severity Only (Baseline)")]["preventable_deaths"].values[0]
        saved    = baseline - combined
        direction = "saves" if saved >= 0 else "costs"
        print(f"  {scenario}: Combined Matrix {direction} {abs(saved)} lives vs Severity Only baseline")

# =============================================================================
# 12. MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/simulation", exist_ok=True)

    print("="*60)
    print("ICU TRIAGE SIMULATION v3 — Exact Binary Matrix Match")
    print("ENGG2112 | Aayan Shukla")
    print("NHS Data: Weekly Admissions & Beds, Alpha Surge")
    print("Triage matrix: matches deployed app (index.html)")
    print("="*60)
    print("\nRunning 9 combinations (3 scenarios × 3 strategies)...")

    df = run_all_simulations()

    print("\nGenerating figures...")
    plot_results(df, "/home/claude/simulation")
    plot_utilisation_over_time(df, "/home/claude/simulation")

    print_summary(df)

    csv_path = "/home/claude/simulation/simulation_results_v3.csv"
    df.drop(columns=["icu_utilisation_daily"]).to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    print("\nDone.")
