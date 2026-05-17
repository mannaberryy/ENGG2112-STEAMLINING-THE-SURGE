import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# ICU TRIAGE DECISION-SUPPORT SIMULATION
# Streamlining the Surge
#
# Hospitals enter:
#   - number of patients in each priority category
#   - ICU/resource availability
#   - surge condition
#
# The simulation compares:
#   1. Severity Only (Baseline)
#   2. Deterioration Only
#   3. Combined Matrix
#
# It recommends the triage method that minimises expected preventable deaths
# and under-triage under the entered resource constraints.
# =============================================================================


# =============================================================================
# 1. NHS CAPACITY CONSTANTS
# =============================================================================

NHS_CAPACITY = {
    "baseline_icu_beds": 1420,
    "peak_icu_beds_occupied": 4096,
    "peak_mv_beds_occupied": 3736,
    "peak_total_beds": 34336,
    "peak_daily_admissions": 4134,
    "baseline_daily_admissions": 53,
    "total_icu_capacity": 4500,
    "total_mv_capacity": 4000,
}


# =============================================================================
# 2. SURGE SCENARIO DEFINITIONS
# =============================================================================

SCENARIOS = {
    "Moderate Surge": {
        "description": "System under pressure; triage decisions begin to affect resource allocation.",
    },
    "Severe Surge": {
        "description": "ICU near capacity; escalation decisions become resource constrained.",
    },
    "Crisis": {
        "description": "ICU overwhelmed; triage strategy strongly affects patient prioritisation.",
    },
    "Custom": {
        "description": "Hospital-defined resource and patient-load condition.",
    },
}


# =============================================================================
# 3. PATIENT DATA CLASS
# =============================================================================

@dataclass
class Patient:
    patient_id: int
    severity: str          # "severe" / "non_severe"
    deterioration: str     # "high_risk" / "low_risk"
    priority_score: int    # 1–4, where 4 = highest priority
    icu_allocated: bool = False
    outcome: str = "Pending"


# =============================================================================
# 4. BINARY TRIAGE MATRIX
# =============================================================================

TRIAGE_MATRIX = {
    ("severe", "high_risk"): (4, "ICU escalation candidate / urgent clinical review"),
    ("severe", "low_risk"): (3, "HDU review and frequent reassessment"),
    ("non_severe", "high_risk"): (2, "Increased monitoring and repeat biomarker assessment"),
    ("non_severe", "low_risk"): (1, "Routine ward monitoring"),
}


def triage_decision(severity: str, deterioration: str) -> Tuple[int, str]:
    return TRIAGE_MATRIX.get((severity, deterioration), (1, "Routine ward monitoring"))


# =============================================================================
# 5. ICU NEED AND MORTALITY ASSUMPTIONS
# =============================================================================

# Assumption: ICU-level escalation is required for priority 4 and priority 3.
def needs_icu(patient: Patient) -> bool:
    return patient.priority_score >= 3


# Expected mortality risk if ICU escalation is denied.
UNTREATED_MORTALITY = {
    ("severe", "high_risk"): 0.75,
    ("severe", "low_risk"): 0.40,
    ("non_severe", "high_risk"): 0.25,
    ("non_severe", "low_risk"): 0.05,
}


# =============================================================================
# 6. TRIAGE STRATEGIES
# =============================================================================

def strategy_severity_only(patients: List[Patient]) -> List[Patient]:
    """
    Baseline strategy.
    Prioritises severe patients first, approximating severity-only escalation logic.
    """
    order = {"severe": 0, "non_severe": 1}
    return sorted(patients, key=lambda p: order[p.severity])


def strategy_deterioration_only(patients: List[Patient]) -> List[Patient]:
    """
    Deterioration-only strategy.
    Prioritises high-risk deterioration patients first.
    """
    order = {"high_risk": 0, "low_risk": 1}
    return sorted(patients, key=lambda p: order[p.deterioration])


def strategy_combined_matrix(patients: List[Patient]) -> List[Patient]:
    """
    Proposed strategy.
    Prioritises patients using the combined severity + deterioration triage matrix.
    """
    return sorted(patients, key=lambda p: -p.priority_score)


STRATEGIES = {
    "Severity Only (Baseline)": strategy_severity_only,
    "Deterioration Only": strategy_deterioration_only,
    "Combined Matrix": strategy_combined_matrix,
}


# =============================================================================
# 7. GENERATE PATIENTS FROM HOSPITAL-ENTERED PRIORITY COUNTS
# =============================================================================

def generate_patients_from_priority_counts(priority_counts: dict) -> List[Patient]:
    """
    Creates patient objects from hospital-entered patient counts.

    Priority 4 = Severe + High Risk
    Priority 3 = Severe + Low Risk
    Priority 2 = Non-Severe + High Risk
    Priority 1 = Non-Severe + Low Risk
    """
    priority_map = {
        4: ("severe", "high_risk"),
        3: ("severe", "low_risk"),
        2: ("non_severe", "high_risk"),
        1: ("non_severe", "low_risk"),
    }

    patients = []
    patient_id = 0

    for priority in [4, 3, 2, 1]:
        count = int(priority_counts.get(priority, 0))
        severity, deterioration = priority_map[priority]

        for _ in range(count):
            patients.append(
                Patient(
                    patient_id=patient_id,
                    severity=severity,
                    deterioration=deterioration,
                    priority_score=priority,
                )
            )
            patient_id += 1

    return patients


# =============================================================================
# 8. EVALUATE ONE TRIAGE STRATEGY
# =============================================================================

def evaluate_triage_strategy(
    patients: List[Patient],
    strategy_name: str,
    available_icu_beds: int,
) -> dict:
    """
    Evaluates a triage strategy under the hospital's current available ICU capacity.
    """
    ordered_patients = STRATEGIES[strategy_name](patients)

    icu_used = 0
    under_triage_count = 0
    expected_preventable_deaths = 0.0

    allocated_priority_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    denied_priority_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for patient in ordered_patients:
        if needs_icu(patient):
            if icu_used < available_icu_beds:
                icu_used += 1
                patient.icu_allocated = True
                patient.outcome = "ICU allocated"
                allocated_priority_counts[patient.priority_score] += 1
            else:
                patient.icu_allocated = False
                patient.outcome = "ICU denied"
                under_triage_count += 1
                denied_priority_counts[patient.priority_score] += 1
                expected_preventable_deaths += UNTREATED_MORTALITY[
                    (patient.severity, patient.deterioration)
                ]
        else:
            patient.outcome = "No ICU required"

    return {
        "strategy": strategy_name,
        "available_icu_beds": available_icu_beds,
        "icu_used": icu_used,
        "icu_remaining": max(available_icu_beds - icu_used, 0),
        "under_triage_count": under_triage_count,
        "expected_preventable_deaths": round(expected_preventable_deaths, 2),
        "priority_4_allocated": allocated_priority_counts[4],
        "priority_3_allocated": allocated_priority_counts[3],
        "priority_4_denied": denied_priority_counts[4],
        "priority_3_denied": denied_priority_counts[3],
    }


# =============================================================================
# 9. MAIN HOSPITAL DECISION-SUPPORT FUNCTION
# =============================================================================

def run_hospital_decision_support(
    priority_4_patients: int,
    priority_3_patients: int,
    priority_2_patients: int,
    priority_1_patients: int,
    total_icu_beds: int,
    occupied_icu_beds: int,
    surge_condition: str = "Custom",
) -> pd.DataFrame:
    """
    Hospital-facing simulation.

    Inputs:
    - patient counts in each priority category
    - total ICU beds
    - currently occupied ICU beds
    - surge condition label

    Output:
    - comparison of triage methods
    - recommended triage strategy
    """
    available_icu_beds = max(total_icu_beds - occupied_icu_beds, 0)

    priority_counts = {
        4: priority_4_patients,
        3: priority_3_patients,
        2: priority_2_patients,
        1: priority_1_patients,
    }

    patients = generate_patients_from_priority_counts(priority_counts)

    results = []
    for strategy_name in STRATEGIES:
        result = evaluate_triage_strategy(
            patients=patients.copy(),
            strategy_name=strategy_name,
            available_icu_beds=available_icu_beds,
        )
        result["surge_condition"] = surge_condition
        result["total_patients"] = len(patients)
        result["priority_4_patients"] = priority_4_patients
        result["priority_3_patients"] = priority_3_patients
        result["priority_2_patients"] = priority_2_patients
        result["priority_1_patients"] = priority_1_patients
        results.append(result)

    df = pd.DataFrame(results)

    # Best strategy = lowest expected preventable deaths, then lowest under-triage.
    df = df.sort_values(
        by=["expected_preventable_deaths", "under_triage_count"],
        ascending=[True, True],
    ).reset_index(drop=True)

    recommended_strategy = df.loc[0, "strategy"]

    print("\n" + "=" * 80)
    print("HOSPITAL TRIAGE DECISION-SUPPORT RESULTS")
    print("=" * 80)
    print(f"Surge condition: {surge_condition}")
    print(f"Scenario description: {SCENARIOS.get(surge_condition, SCENARIOS['Custom'])['description']}")
    print(f"Total patients entered: {len(patients)}")
    print(f"Total ICU beds: {total_icu_beds}")
    print(f"Occupied ICU beds: {occupied_icu_beds}")
    print(f"Available ICU beds: {available_icu_beds}")

    print("\nPatient priority distribution:")
    print(f"  Priority 4 — Severe + High Risk:      {priority_4_patients}")
    print(f"  Priority 3 — Severe + Low Risk:       {priority_3_patients}")
    print(f"  Priority 2 — Non-Severe + High Risk:  {priority_2_patients}")
    print(f"  Priority 1 — Non-Severe + Low Risk:   {priority_1_patients}")

    print("\nStrategy comparison:")
    print(df.to_string(index=False))

    print("\nRECOMMENDED TRIAGE METHOD:")
    print(f"  {recommended_strategy}")

    return df


# =============================================================================
# 10. OPTIONAL: RUN STANDARD EXAMPLE SCENARIOS
# =============================================================================

def run_example_scenarios() -> pd.DataFrame:
    """
    Runs three example hospital-input scenarios.
    These are illustrative examples for the report.
    """
    example_scenarios = [
        {
            "surge_condition": "Moderate Surge",
            "priority_4_patients": 20,
            "priority_3_patients": 40,
            "priority_2_patients": 80,
            "priority_1_patients": 160,
            "total_icu_beds": 120,
            "occupied_icu_beds": 70,
        },
        {
            "surge_condition": "Severe Surge",
            "priority_4_patients": 80,
            "priority_3_patients": 120,
            "priority_2_patients": 160,
            "priority_1_patients": 240,
            "total_icu_beds": 450,
            "occupied_icu_beds": 380,
        },
        {
            "surge_condition": "Crisis",
            "priority_4_patients": 160,
            "priority_3_patients": 240,
            "priority_2_patients": 300,
            "priority_1_patients": 400,
            "total_icu_beds": 700,
            "occupied_icu_beds": 650,
        },
    ]

    all_results = []

    for scenario in example_scenarios:
        df = run_hospital_decision_support(**scenario)
        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df


# =============================================================================
# 11. MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Option 1: Single hospital input example
    # Edit these numbers to represent the current hospital condition.
    # -------------------------------------------------------------------------

    results_df = run_hospital_decision_support(
        priority_4_patients=80,       # Severe + High Risk
        priority_3_patients=120,      # Severe + Low Risk
        priority_2_patients=160,      # Non-Severe + High Risk
        priority_1_patients=240,      # Non-Severe + Low Risk
        total_icu_beds=450,
        occupied_icu_beds=380,
        surge_condition="Severe Surge",
    )

    results_df.to_csv("hospital_triage_decision_support_results.csv", index=False)

    # -------------------------------------------------------------------------
    # Option 2: Uncomment to run all example scenarios for report comparison.
    # -------------------------------------------------------------------------

    # all_results_df = run_example_scenarios()
    # all_results_df.to_csv("hospital_triage_example_scenarios.csv", index=False)

    print("\nResults saved to: hospital_triage_decision_support_results.csv")
