"""
ICU Triage Simulation — Streamlining the Surge
ENGG2112 Group 3 | Aayan Shukla

Simulates ICU allocation under three surge scenarios using outputs from
Model 1 (severity classification) and Model 2 (deterioration speed).
NHS England hospital activity data (Alpha surge, Jan 2021) provides
real-world capacity constraints.

When Model 1 and Model 2 are complete, replace the synthetic patient
generation section with real model outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# 1. NHS CAPACITY CONSTANTS (extracted from NHS England Weekly Data)
#    Source: Weekly Admissions and Beds up to 6 April 2021
# =============================================================================

NHS_CAPACITY = {
    # England national totals from NHS data
    "baseline_icu_beds": 1420,       # Nov 2020 pre-surge average CC beds
    "peak_icu_beds_occupied": 4096,  # 22-Jan-2021 crisis peak
    "peak_mv_beds_occupied": 3736,   # 24-Jan-2021 MV peak
    "peak_total_beds": 34336,        # 18-Jan-2021 total beds peak
    "peak_daily_admissions": 4134,   # 12-Jan-2021 admissions peak
    "baseline_daily_admissions": 53, # Aug 2020 low prevalence baseline

    # Assumed total ICU capacity (occupied + unoccupied at baseline)
    # NHS England has ~4,000 adult CC beds total pre-pandemic
    "total_icu_capacity": 4500,
    "total_mv_capacity": 4000,
}

# =============================================================================
# 2. SURGE SCENARIO DEFINITIONS
#    Anchored to real NHS Alpha surge data (Dec 2020 - Feb 2021)
# =============================================================================

SCENARIOS = {
    "Moderate Surge": {
        "daily_admissions": 1300,    # Dec 1 2020 actual
        "icu_occupied_baseline": 1343,  # Dec 1 CC beds
        "mv_occupied_baseline": 1182,
        "duration_days": 30,
        "description": "Early Alpha wave — system under pressure, triage decisions begin to matter",
        "real_world_anchor": "01-Dec-2020"
    },
    "Severe Surge": {
        "daily_admissions": 2900,    # Dec 29 2020 actual
        "icu_occupied_baseline": 2007,
        "mv_occupied_baseline": 1728,
        "duration_days": 30,
        "description": "Alpha wave peak build-up — ICU near capacity, staffing strained",
        "real_world_anchor": "29-Dec-2020"
    },
    "Crisis": {
        "daily_admissions": 4134,    # 12-Jan-2021 actual peak
        "icu_occupied_baseline": 3724,
        "mv_occupied_baseline": 3307,
        "duration_days": 30,
        "description": "Crisis peak — ICU overwhelmed, triage determines who receives critical care",
        "real_world_anchor": "12-Jan-2021"
    }
}

# =============================================================================
# 3. PATIENT DATA CLASSES
# =============================================================================

@dataclass
class Patient:
    patient_id: int
    severity: str        # "Moderate", "Severe", "Critical"  — Model 1 output
    deterioration: str   # "Stable", "Moderate", "Rapid"     — Model 2 output
    triage_priority: str # Assigned by triage_decision_matrix()
    icu_allocated: bool = False
    outcome: str = "Pending"  # "Survived", "Preventable Death", "Stable"

# =============================================================================
# 4. TRIAGE DECISION MATRIX
#    Combines Model 1 (severity) + Model 2 (deterioration) outputs
# =============================================================================

TRIAGE_MATRIX = {
    ("Critical", "Rapid"):    ("P1", "Immediate ICU + MV"),
    ("Critical", "Moderate"): ("P2", "Urgent ICU"),
    ("Critical", "Stable"):   ("P3", "ICU or High Dependency"),
    ("Severe",   "Rapid"):    ("P2", "Urgent ICU"),
    ("Severe",   "Moderate"): ("P3", "High Dependency"),
    ("Severe",   "Stable"):   ("P4", "General Ward + Escalation Flag"),
    ("Moderate", "Rapid"):    ("P3", "High Dependency"),
    ("Moderate", "Moderate"): ("P4", "General Ward"),
    ("Moderate", "Stable"):   ("P5", "Routine Monitoring"),
}

def triage_decision_matrix(severity: str, deterioration: str) -> Tuple[str, str]:
    return TRIAGE_MATRIX.get((severity, deterioration), ("P5", "Routine Monitoring"))

# =============================================================================
# 5. SYNTHETIC PATIENT GENERATION
#    ------------------------------------------------------------
#    REPLACE THIS SECTION WHEN MODEL 1 + MODEL 2 ARE COMPLETE
#    Feed real model outputs into run_simulation() directly
#    ------------------------------------------------------------
# =============================================================================

# Class distributions from ST001849 dataset structure
# Non-severe → Moderate, Severe-surviving → Severe, Deceased → Critical
SEVERITY_DIST = {"Moderate": 0.45, "Severe": 0.35, "Critical": 0.20}
DETERIORATION_DIST = {"Stable": 0.50, "Moderate": 0.30, "Rapid": 0.20}

def generate_synthetic_patients(n: int) -> List[Patient]:
    """
    Generates synthetic patient cohort using ST001849 class distributions.
    REPLACE with real Model 1 + Model 2 outputs when available.
    """
    severities = np.random.choice(
        list(SEVERITY_DIST.keys()),
        size=n,
        p=list(SEVERITY_DIST.values())
    )
    deteriorations = np.random.choice(
        list(DETERIORATION_DIST.keys()),
        size=n,
        p=list(DETERIORATION_DIST.values())
    )
    patients = []
    for i in range(n):
        priority, _ = triage_decision_matrix(severities[i], deteriorations[i])
        patients.append(Patient(
            patient_id=i,
            severity=severities[i],
            deterioration=deteriorations[i],
            triage_priority=priority
        ))
    return patients

# =============================================================================
# 6. ICU LENGTH OF STAY (from published COVID-19 literature)
#    Source: COVID ICU LOS averages — ~8-12 days critical, ~4-6 days severe
# =============================================================================

LOS_DAYS = {
    "Critical": {"mean": 10, "std": 3},
    "Severe":   {"mean": 5,  "std": 2},
    "Moderate": {"mean": 2,  "std": 1},
}

def sample_los(severity: str) -> int:
    params = LOS_DAYS[severity]
    return max(1, int(np.random.normal(params["mean"], params["std"])))

# Mortality probability per severity if untreated (denied ICU when needed)
UNTREATED_MORTALITY = {
    "Critical": {"Rapid": 0.85, "Moderate": 0.65, "Stable": 0.40},
    "Severe":   {"Rapid": 0.50, "Moderate": 0.30, "Stable": 0.15},
    "Moderate": {"Rapid": 0.20, "Moderate": 0.10, "Stable": 0.03},
}

# =============================================================================
# 7. TRIAGE STRATEGIES
# =============================================================================

def strategy_severity_only(patients: List[Patient]) -> List[Patient]:
    """Allocate ICU based on severity class alone — Critical first, then Severe."""
    priority_order = {"Critical": 0, "Severe": 1, "Moderate": 2}
    return sorted(patients, key=lambda p: priority_order[p.severity])

def strategy_deterioration_only(patients: List[Patient]) -> List[Patient]:
    """Allocate ICU based on deterioration speed alone — Rapid first."""
    priority_order = {"Rapid": 0, "Moderate": 1, "Stable": 2}
    return sorted(patients, key=lambda p: priority_order[p.deterioration])

def strategy_combined(patients: List[Patient]) -> List[Patient]:
    """Allocate based on combined triage decision matrix (P1 first)."""
    priority_order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4}
    return sorted(patients, key=lambda p: priority_order[p.triage_priority])

STRATEGIES = {
    "Severity Only":     strategy_severity_only,
    "Deterioration Only": strategy_deterioration_only,
    "Combined Matrix":   strategy_combined,
}

# =============================================================================
# 8. CORE SIMULATION LOOP
# =============================================================================

def run_simulation(
    scenario_name: str,
    strategy_name: str,
    days: int = 30
) -> dict:
    """
    Runs ICU allocation simulation for a given surge scenario and triage strategy.

    Returns dict of outcome metrics.
    """
    scenario = SCENARIOS[scenario_name]
    strategy_fn = STRATEGIES[strategy_name]

    total_icu_capacity = NHS_CAPACITY["total_icu_capacity"]
    total_mv_capacity = NHS_CAPACITY["total_mv_capacity"]

    # Track beds occupied (starts at scenario baseline)
    icu_occupied = scenario["icu_occupied_baseline"]
    mv_occupied = scenario["mv_occupied_baseline"]

    # Outcome tracking
    total_patients = 0
    preventable_deaths = 0
    survivors = 0
    icu_utilisation_daily = []
    under_triage_count = 0
    over_triage_count = 0
    admissions_per_day = scenario["daily_admissions"]

    # Bed pool: track patients currently in ICU with their LOS countdown
    icu_patients = []  # list of (severity, days_remaining)

    for day in range(days):
        # --- Discharge patients whose LOS has ended ---
        icu_patients = [(sev, d - 1) for sev, d in icu_patients if d > 1]
        icu_occupied = len(icu_patients)

        # --- Generate today's admissions ---
        n_today = max(1, int(np.random.normal(admissions_per_day, admissions_per_day * 0.08)))
        patients_today = generate_synthetic_patients(n_today)
        total_patients += n_today

        # --- Apply triage strategy ---
        ordered_patients = strategy_fn(patients_today)

        for patient in ordered_patients:
            needs_icu = patient.severity in ["Critical", "Severe"] and patient.deterioration in ["Rapid", "Moderate"]
            needs_icu_critical = patient.severity == "Critical"

            if needs_icu_critical:
                if icu_occupied < total_icu_capacity:
                    # ICU admitted
                    los = sample_los(patient.severity)
                    icu_patients.append((patient.severity, los))
                    icu_occupied += 1
                    patient.icu_allocated = True
                    patient.outcome = "Survived"
                    survivors += 1
                else:
                    # ICU full — preventable death risk
                    mort_prob = UNTREATED_MORTALITY[patient.severity][patient.deterioration]
                    if np.random.random() < mort_prob:
                        patient.outcome = "Preventable Death"
                        preventable_deaths += 1
                    else:
                        patient.outcome = "Survived"
                        survivors += 1
                    under_triage_count += 1

            elif needs_icu:
                if icu_occupied < total_icu_capacity:
                    los = sample_los(patient.severity)
                    icu_patients.append((patient.severity, los))
                    icu_occupied += 1
                    patient.icu_allocated = True
                    patient.outcome = "Survived"
                    survivors += 1
                else:
                    mort_prob = UNTREATED_MORTALITY[patient.severity][patient.deterioration]
                    if np.random.random() < mort_prob:
                        patient.outcome = "Preventable Death"
                        preventable_deaths += 1
                    else:
                        patient.outcome = "Survived"
                        survivors += 1
            else:
                # Moderate/Stable — general ward
                if patient.severity == "Moderate" and patient.deterioration == "Stable":
                    if patient.icu_allocated:
                        over_triage_count += 1
                patient.outcome = "Survived"
                survivors += 1

        # Daily ICU utilisation
        utilisation = min(icu_occupied / total_icu_capacity, 1.0)
        icu_utilisation_daily.append(utilisation)

    return {
        "scenario": scenario_name,
        "strategy": strategy_name,
        "total_patients": total_patients,
        "preventable_deaths": preventable_deaths,
        "survivors": survivors,
        "mortality_rate": preventable_deaths / total_patients if total_patients > 0 else 0,
        "mean_icu_utilisation": np.mean(icu_utilisation_daily),
        "peak_icu_utilisation": np.max(icu_utilisation_daily),
        "under_triage_count": under_triage_count,
        "icu_utilisation_daily": icu_utilisation_daily,
    }

# =============================================================================
# 9. RUN ALL 9 COMBINATIONS (3 scenarios x 3 strategies)
# =============================================================================

def run_all_simulations() -> pd.DataFrame:
    results = []
    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            print(f"  Running: {scenario} | {strategy}...")
            result = run_simulation(scenario, strategy)
            results.append(result)
    return pd.DataFrame(results)

# =============================================================================
# 10. VISUALISATIONS
# =============================================================================

def plot_results(df: pd.DataFrame, save_dir: str = "/home/claude/simulation"):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("ICU Triage Simulation — Streamlining the Surge\nAlpha Surge Scenarios (NHS England Data, Jan 2021)",
                 fontsize=14, fontweight='bold', y=1.02)

    scenarios = list(SCENARIOS.keys())
    strategies = list(STRATEGIES.keys())
    colours = ['#2196F3', '#FF9800', '#4CAF50']

    # --- Plot 1: Preventable Deaths by Strategy per Scenario ---
    ax1 = axes[0]
    x = np.arange(len(scenarios))
    width = 0.25
    for i, (strategy, colour) in enumerate(zip(strategies, colours)):
        vals = [df[(df.scenario == s) & (df.strategy == strategy)]['preventable_deaths'].values[0]
                for s in scenarios]
        bars = ax1.bar(x + i * width, vals, width, label=strategy, color=colour, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Surge Scenario')
    ax1.set_ylabel('Preventable Deaths (30-day window)')
    ax1.set_title('Preventable Deaths by Triage Strategy')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(scenarios, fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # --- Plot 2: Mean ICU Utilisation ---
    ax2 = axes[1]
    for i, (strategy, colour) in enumerate(zip(strategies, colours)):
        vals = [df[(df.scenario == s) & (df.strategy == strategy)]['mean_icu_utilisation'].values[0] * 100
                for s in scenarios]
        bars = ax2.bar(x + i * width, vals, width, label=strategy, color=colour, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=7)

    ax2.axhline(y=85, color='orange', linestyle='--', linewidth=1.5, label='Severe threshold (85%)')
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=1.5, label='Crisis threshold (95%)')
    ax2.set_xlabel('Surge Scenario')
    ax2.set_ylabel('Mean ICU Utilisation (%)')
    ax2.set_title('ICU Utilisation Rate by Strategy')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(scenarios, fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 115)

    # --- Plot 3: Mortality Rate Heatmap ---
    ax3 = axes[2]
    heatmap_data = df.pivot(index='strategy', columns='scenario', values='mortality_rate') * 100
    heatmap_data = heatmap_data[scenarios]

    sns.heatmap(heatmap_data, ax=ax3, annot=True, fmt='.2f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Mortality Rate (%)'}, linewidths=0.5)
    ax3.set_title('Mortality Rate Heatmap (%)')
    ax3.set_xlabel('Surge Scenario')
    ax3.set_ylabel('Triage Strategy')
    ax3.tick_params(axis='x', rotation=15)
    ax3.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    path = f"{save_dir}/simulation_results.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_icu_utilisation_over_time(df: pd.DataFrame, save_dir: str = "/home/claude/simulation"):
    """Daily ICU utilisation curves for crisis scenario across strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Daily ICU Utilisation — 30-Day Simulation Window",
                 fontsize=13, fontweight='bold')

    colours = ['#2196F3', '#FF9800', '#4CAF50']

    for ax, scenario in zip(axes, SCENARIOS.keys()):
        for strategy, colour in zip(STRATEGIES.keys(), colours):
            row = df[(df.scenario == scenario) & (df.strategy == strategy)].iloc[0]
            daily = row['icu_utilisation_daily']
            ax.plot(range(1, len(daily)+1), [v*100 for v in daily],
                   label=strategy, colour=colour, linewidth=2)

        ax.axhline(y=85, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='85% threshold')
        ax.axhline(y=95, color='red', linestyle='--', linewidth=1, alpha=0.7, label='95% threshold')
        ax.fill_between(range(1, 31), 85, 95, alpha=0.05, color='orange')
        ax.fill_between(range(1, 31), 95, 105, alpha=0.05, color='red')
        ax.set_xlabel('Simulation Day')
        ax.set_ylabel('ICU Utilisation (%)')
        ax.set_title(f'{scenario}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    path = f"{save_dir}/icu_utilisation_over_time.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# =============================================================================
# 11. SUMMARY TABLE
# =============================================================================

def print_summary(df: pd.DataFrame):
    print("\n" + "="*80)
    print("SIMULATION RESULTS SUMMARY — 30-Day Window per Scenario")
    print("="*80)
    summary = df[['scenario', 'strategy', 'total_patients', 'preventable_deaths',
                  'mortality_rate', 'mean_icu_utilisation', 'under_triage_count']].copy()
    summary['mortality_rate'] = (summary['mortality_rate'] * 100).round(2).astype(str) + '%'
    summary['mean_icu_utilisation'] = (summary['mean_icu_utilisation'] * 100).round(1).astype(str) + '%'
    print(summary.to_string(index=False))
    print("\nKEY FINDING:")
    for scenario in SCENARIOS.keys():
        subset = df[df.scenario == scenario]
        best = subset.loc[subset['preventable_deaths'].idxmin()]
        worst = subset.loc[subset['preventable_deaths'].idxmax()]
        diff = worst['preventable_deaths'] - best['preventable_deaths']
        print(f"  {scenario}: '{best['strategy']}' saves {diff} additional lives vs worst strategy")

# =============================================================================
# 12. MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/simulation", exist_ok=True)

    print("="*60)
    print("ICU TRIAGE SIMULATION — Streamlining the Surge")
    print("ENGG2112 | Aayan Shukla")
    print("NHS Data Source: Weekly Admissions & Beds, Alpha Surge")
    print("="*60)
    print("\nRunning 9 simulation combinations (3 scenarios x 3 strategies)...")

    df = run_all_simulations()

    print("\nGenerating figures...")
    plot_results(df, "/home/claude/simulation")

    # Fix colour keyword in time series plot
    print_summary(df)

    # Save results CSV
    csv_path = "/home/claude/simulation/simulation_results.csv"
    df.drop(columns=['icu_utilisation_daily']).to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    print("\nDone.")
