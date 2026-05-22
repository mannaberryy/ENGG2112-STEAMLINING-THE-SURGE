import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# ICU TRIAGE DECISION-SUPPORT SIMULATION
# Streamlining the Surge — ENGG2112
# =============================================================================

SCENARIOS = {
    "Moderate Surge": {
        "description": "System under pressure; triage decisions begin to affect resource allocation.",
        "mortality_multiplier": 1.0,
        "deterioration_bonus": 1.0,
        "bed_buffer_fraction": 0.0,
        "exclude_priority_1": False,
    },
    "Severe Surge": {
        "description": "ICU near capacity; escalation decisions become resource constrained.",
        "mortality_multiplier": 1.25,
        "deterioration_bonus": 1.0,
        "bed_buffer_fraction": 0.15,
        "exclude_priority_1": False,
    },
    "Crisis": {
        "description": "ICU overwhelmed; deterioration risk becomes the dominant mortality driver.",
        "mortality_multiplier": 1.50,
        "deterioration_bonus": 2.0,
        "bed_buffer_fraction": 0.30,
        "exclude_priority_1": True,
    },
    "Custom": {
        "description": "Hospital-defined resource and patient-load condition.",
        "mortality_multiplier": 1.0,
        "deterioration_bonus": 1.0,
        "bed_buffer_fraction": 0.0,
        "exclude_priority_1": False,
    },
}

@dataclass
class Patient:
    patient_id: int
    severity: str
    deterioration: str
    priority_score: int
    icu_allocated: bool = False
    outcome: str = "Pending"

TRIAGE_MATRIX = {
    ("severe", "high_risk"):     (4, "ICU escalation candidate / urgent clinical review"),
    ("severe", "low_risk"):      (3, "HDU review and frequent reassessment"),
    ("non_severe", "high_risk"): (2, "Increased monitoring and repeat biomarker assessment"),
    ("non_severe", "low_risk"):  (1, "Routine ward monitoring"),
}

def triage_decision(severity: str, deterioration: str) -> Tuple[int, str]:
    return TRIAGE_MATRIX.get((severity, deterioration), (1, "Routine ward monitoring"))


# =============================================================================
# MORTALITY WEIGHTS
#
# Untreated mortality = expected mortality if ICU escalation is denied.
# Treated mortality = residual expected mortality even if ICU escalation is provided.
#
# Treated mortality is lower than untreated mortality, but not zero.
# This avoids assuming ICU treatment makes all patients completely safe.
# =============================================================================

BASE_UNTREATED_MORTALITY = {
    ("severe", "high_risk"):     0.75,
    ("severe", "low_risk"):      0.40,
    ("non_severe", "high_risk"): 0.25,
    ("non_severe", "low_risk"):  0.05,
}

BASE_TREATED_MORTALITY = {
    ("severe", "high_risk"):     0.30,
    ("severe", "low_risk"):      0.14,
    ("non_severe", "high_risk"): 0.10,
    ("non_severe", "low_risk"):  0.02,
}

def get_adjusted_mortality_weights(surge_condition: str) -> Tuple[dict, dict]:
    scenario = SCENARIOS.get(surge_condition, SCENARIOS["Custom"])
    mult = scenario["mortality_multiplier"]
    det_bonus = scenario["deterioration_bonus"]

    untreated_weights = {}
    treated_weights = {}

    for key, base in BASE_UNTREATED_MORTALITY.items():
        severity, deterioration = key

        multiplier = mult
        if deterioration == "high_risk":
            multiplier *= det_bonus

        untreated_weights[key] = min(base * multiplier, 1.0)

    for key, base in BASE_TREATED_MORTALITY.items():
        severity, deterioration = key

        multiplier = mult

        # ICU treatment reduces, but does not remove, deterioration-related risk.
        if deterioration == "high_risk":
            multiplier *= (1 + 0.35 * (det_bonus - 1))

        treated_weights[key] = min(base * multiplier, 1.0)

    return untreated_weights, treated_weights


def needs_icu(patient: Patient, exclude_priority_1: bool) -> bool:
    if exclude_priority_1 and patient.priority_score == 1:
        return False
    return patient.priority_score >= 2


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

def strategy_severity_only(patients: List[Patient]) -> List[Patient]:
    # P4 -> P3 -> P2 -> P1
    priority_order = {4: 0, 3: 1, 2: 2, 1: 3}
    return sorted(patients, key=lambda p: priority_order[p.priority_score])


def strategy_deterioration_only(patients: List[Patient]) -> List[Patient]:
    # P4 -> P2 -> P3 -> P1
    priority_order = {4: 0, 2: 1, 3: 2, 1: 3}
    return sorted(patients, key=lambda p: priority_order[p.priority_score])


STRATEGIES = {
    "Severity Only": strategy_severity_only,
    "Deterioration Only": strategy_deterioration_only,
}


def generate_patients_from_priority_counts(priority_counts: dict) -> List[Patient]:
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


def evaluate_triage_strategy(
    patients: List[Patient],
    strategy_name: str,
    available_icu_beds: int,
    surge_condition: str = "Custom",
) -> dict:

    scenario = SCENARIOS.get(surge_condition, SCENARIOS["Custom"])
    exclude_priority_1 = scenario["exclude_priority_1"]

    untreated_weights, treated_weights = get_adjusted_mortality_weights(surge_condition)

    bed_buffer = int(available_icu_beds * scenario["bed_buffer_fraction"])
    effective_beds = max(available_icu_beds - bed_buffer, 0)

    ordered_patients = STRATEGIES[strategy_name](patients)

    icu_used = 0
    under_triage_count = 0

    expected_deaths_treated = 0.0
    expected_deaths_untreated = 0.0

    allocated_priority_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    denied_priority_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for patient in ordered_patients:
        key = (patient.severity, patient.deterioration)

        if needs_icu(patient, exclude_priority_1):
            if icu_used < effective_beds:
                icu_used += 1
                patient.icu_allocated = True
                patient.outcome = "ICU allocated"
                allocated_priority_counts[patient.priority_score] += 1

                # Treated patients still carry some residual mortality risk.
                expected_deaths_treated += treated_weights[key]

            else:
                patient.icu_allocated = False
                patient.outcome = "ICU denied"
                under_triage_count += 1
                denied_priority_counts[patient.priority_score] += 1

                # Denied patients carry higher untreated mortality risk.
                expected_deaths_untreated += untreated_weights[key]

        else:
            patient.outcome = "No ICU required"

            # Routine ward patients still carry low baseline mortality.
            expected_deaths_treated += treated_weights[key]

    total_expected_deaths = expected_deaths_treated + expected_deaths_untreated

    return {
        "strategy": strategy_name,
        "surge_condition": surge_condition,
        "available_icu_beds": available_icu_beds,
        "effective_icu_beds": effective_beds,
        "bed_buffer_held": bed_buffer,
        "icu_used": icu_used,
        "icu_remaining": max(effective_beds - icu_used, 0),
        "under_triage_count": under_triage_count,

        "expected_deaths_treated": round(expected_deaths_treated, 2),
        "expected_deaths_untreated": round(expected_deaths_untreated, 2),
        "total_expected_deaths": round(total_expected_deaths, 2),

        "priority_4_allocated": allocated_priority_counts[4],
        "priority_3_allocated": allocated_priority_counts[3],
        "priority_2_allocated": allocated_priority_counts[2],

        "priority_4_denied": denied_priority_counts[4],
        "priority_3_denied": denied_priority_counts[3],
        "priority_2_denied": denied_priority_counts[2],
    }


def run_hospital_decision_support(
    priority_4_patients: int,
    priority_3_patients: int,
    priority_2_patients: int,
    priority_1_patients: int,
    total_icu_beds: int,
    occupied_icu_beds: int,
    surge_condition: str = "Custom",
) -> pd.DataFrame:

    available_icu_beds = max(total_icu_beds - occupied_icu_beds, 0)

    priority_counts = {
        4: priority_4_patients,
        3: priority_3_patients,
        2: priority_2_patients,
        1: priority_1_patients,
    }

    patients = generate_patients_from_priority_counts(priority_counts)
    scenario = SCENARIOS.get(surge_condition, SCENARIOS["Custom"])

    results = []

    for strategy_name in STRATEGIES:
        result = evaluate_triage_strategy(
            patients=generate_patients_from_priority_counts(priority_counts),
            strategy_name=strategy_name,
            available_icu_beds=available_icu_beds,
            surge_condition=surge_condition,
        )

        result["total_patients"] = len(patients)
        result["priority_4_patients"] = priority_4_patients
        result["priority_3_patients"] = priority_3_patients
        result["priority_2_patients"] = priority_2_patients
        result["priority_1_patients"] = priority_1_patients

        results.append(result)

    df = pd.DataFrame(results)

    df = df.sort_values(
        by=["total_expected_deaths", "under_triage_count"],
        ascending=[True, True],
    ).reset_index(drop=True)

    recommended_strategy = df.loc[0, "strategy"]

    print("\n" + "=" * 80)
    print("HOSPITAL TRIAGE DECISION-SUPPORT RESULTS")
    print("=" * 80)

    print(f"Surge condition:      {surge_condition}")
    print(f"Description:          {scenario['description']}")
    print(f"Mortality multiplier: {scenario['mortality_multiplier']}x")
    print(f"Deterioration bonus:  {scenario['deterioration_bonus']}x")
    print(f"Bed buffer:           {int(scenario['bed_buffer_fraction'] * 100)}% held in reserve")
    print(f"Priority 1 excluded:  {scenario['exclude_priority_1']}")
    print(f"Total patients:       {len(patients)}")
    print(f"Total ICU beds:       {total_icu_beds}")
    print(f"Occupied ICU beds:    {occupied_icu_beds}")
    print(f"Available ICU beds:   {available_icu_beds}")

    print("\nStrategy comparison:")
    print(
        df[
            [
                "strategy",
                "effective_icu_beds",
                "icu_used",
                "under_triage_count",
                "expected_deaths_treated",
                "expected_deaths_untreated",
                "total_expected_deaths",
            ]
        ].to_string(index=False)
    )

    print(f"\nRECOMMENDED TRIAGE METHOD: {recommended_strategy}")

    return df


if __name__ == "__main__":

    print("\n--- TEST 1: Moderate Surge ---")
    run_hospital_decision_support(
        priority_4_patients=80,
        priority_3_patients=120,
        priority_2_patients=160,
        priority_1_patients=240,
        total_icu_beds=450,
        occupied_icu_beds=250,
        surge_condition="Moderate Surge",
    )

    print("\n--- TEST 2: Severe Surge ---")
    run_hospital_decision_support(
        priority_4_patients=80,
        priority_3_patients=120,
        priority_2_patients=160,
        priority_1_patients=240,
        total_icu_beds=450,
        occupied_icu_beds=250,
        surge_condition="Severe Surge",
    )

    print("\n--- TEST 3: Crisis ---")
    run_hospital_decision_support(
        priority_4_patients=80,
        priority_3_patients=120,
        priority_2_patients=160,
        priority_1_patients=240,
        total_icu_beds=450,
        occupied_icu_beds=250,
        surge_condition="Crisis",
    )
