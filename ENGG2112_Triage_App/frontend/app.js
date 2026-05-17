const navItems = document.querySelectorAll(".nav-item");
const tabPanels = document.querySelectorAll(".tab-panel");

let inputMode = "full";
let surgeMode = "test";

const demoPatients = {
    "demo-1": { severity: "Non-severe", deterioration: "Low Risk", confidence: 78 },
    "demo-2": { severity: "Severe", deterioration: "Low Risk", confidence: 79 },
    "demo-3": { severity: "Severe", deterioration: "High Risk", confidence: 88 }
};

// Helper function to safely update textContent if an element exists
function safeSetText(id, value) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = value;
    }
}

navItems.forEach((item) => {
    item.addEventListener("click", () => {
        const target = item.dataset.tab;
        if (!target) return;

        navItems.forEach((nav) => nav.classList.remove("active"));
        tabPanels.forEach((panel) => panel.classList.remove("active"));

        item.classList.add("active");

        const targetPanel = document.getElementById(target);
        if (targetPanel) {
            targetPanel.classList.add("active");
        }

        updatePageTitle(target);
    });
});

function updatePageTitle(tab) {
    const titles = {
        "patient-risk": "Patient Risk Assessment",
        "triage-matrix": "Triage Matrix",
        "surge-simulation": "Surge Triage Simulation"
    };

    const subtitles = {
        "patient-risk": "Upload patient biomarker data and generate patient-specific severity, deterioration risk, confidence and escalation recommendation.",
        "triage-matrix": "View the complete binary triage matrix used to integrate Model 1 and Model 2 outputs.",
        "surge-simulation": "Test allocation frameworks or recommend a triage strategy under ICU surge pressure."
    };

    safeSetText("page-title", titles[tab] || "");
    safeSetText("page-subtitle", subtitles[tab] || "");
}

function connectSlider(sliderId, outputId, suffix = "") {
    const slider = document.getElementById(sliderId);
    const output = document.getElementById(outputId);

    if (!slider || !output) return;

    output.textContent = slider.value + suffix;
    slider.addEventListener("input", () => {
        output.textContent = slider.value + suffix;
    });
}

connectSlider("icu-capacity", "icu-capacity-value", "%");
connectSlider("ward-pressure", "ward-pressure-value", "%");
connectSlider("incoming-patients", "incoming-patients-value");
connectSlider("high-risk-proportion", "high-risk-proportion-value", "%");

document.querySelectorAll("[data-input-mode]").forEach((button) => {
    button.addEventListener("click", () => {
        inputMode = button.dataset.inputMode;
        document.querySelectorAll("[data-input-mode]").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        updateCsvFormat();
    });
});

function updateCsvFormat() {
    if (inputMode === "full") {
        safeSetText("csv-label", "Full biomarker CSV");
        safeSetText("csv-format-title", "Full model input");
        safeSetText("csv-format-text", "One patient row containing the complete full-biomarker feature set. The backend aligns the CSV to the training features, then the saved pipelines perform internal feature selection.");
    } else {
        safeSetText("csv-label", "Quick biomarker CSV");
        safeSetText("csv-format-title", "Quick assessment input");
        safeSetText("csv-format-text", "Future extension: one patient row containing a reduced quick biomarker panel. This pathway is not connected to the backend yet.");
    }
}

// Store the file reference before the button is clicked so it is never lost
let storedFile = null;

const upload = document.getElementById("biomarker-upload");
const uploadName = document.getElementById("upload-name");

if (upload && uploadName) {
    upload.addEventListener("change", () => {
        if (upload.files && upload.files[0]) {
            storedFile = upload.files[0];
            uploadName.textContent = storedFile.name;
        } else {
            storedFile = null;
            uploadName.textContent = "No file selected";
        }
    });
}

function getRecommendation(severity, deterioration) {
    const matrix = {
        "Non-severe-Low Risk": {
            score: 1,
            action: "Routine ward monitoring",
            band: "Low escalation priority",
            explanation: "The patient is classified as non-severe and has low predicted deterioration risk."
        },
        "Non-severe-High Risk": {
            score: 2,
            action: "Increased monitoring and repeat biomarker assessment",
            band: "Elevated monitoring priority",
            explanation: "The patient is currently non-severe but has high predicted deterioration risk."
        },
        "Severe-Low Risk": {
            score: 3,
            action: "HDU review and frequent reassessment",
            band: "Moderate escalation priority",
            explanation: "The patient is classified as severe but has low predicted deterioration risk."
        },
        "Severe-High Risk": {
            score: 4,
            action: "ICU escalation candidate / urgent clinical review",
            band: "High escalation priority",
            explanation: "The patient is severe and has high predicted deterioration risk, giving the highest escalation priority in the final binary matrix."
        }
    };
    return matrix[`${severity}-${deterioration}`];
}

function riskColour(score) {
    if (score <= 1) return "var(--risk-low)";
    if (score <= 3) return "var(--risk-moderate)";
    return "var(--risk-high)";
}

function updateRingColour(ringId, score) {
    const ring = document.getElementById(ringId);

    if (!ring) return;

    ring.style.setProperty("--ring-colour", riskColour(score));

    const percentages = { 1: "25%", 2: "50%", 3: "75%", 4: "100%" };
    ring.style.setProperty("--risk", percentages[score] || "0%");
}

function cleanProbability(val) {
    if (Array.isArray(val)) return cleanProbability(val[0]);
    const num = Number(val);
    return isNaN(num) ? 0 : num;
}

async function runPrediction(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    const file = storedFile;

    if (!file) {
        alert("Please upload a CSV file first.");
        return;
    }

    const runButton = document.getElementById("run-prediction");
    if (runButton) {
        runButton.disabled = true;
        runButton.textContent = "Generating...";
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData,
            mode: "cors"
        });

        if (!response.ok) {
            throw new Error(`Backend returned status ${response.status}`);
        }

        const result = await response.json();
        console.log("RAW BACKEND RESULT:", result);

        const patient = Array.isArray(result) ? result[0] : result;
        if (!patient) {
            throw new Error("No patient data found in backend response.");
        }

        const rawSeverity = patient.model1_severity_prediction || patient.severity_prediction || patient.severity || "";
        const severity = String(rawSeverity).toLowerCase().includes("severe") && !String(rawSeverity).toLowerCase().includes("non")
            ? "Severe"
            : "Non-severe";

        const rawDeterioration = patient.model2_deterioration_prediction || patient.deterioration_prediction || patient.deterioration || "";
        const deterioration = String(rawDeterioration).toLowerCase().includes("high")
            ? "High Risk"
            : "Low Risk";

        const resultMatrix = getRecommendation(severity, deterioration);
        if (!resultMatrix) {
            throw new Error(`No triage matrix match for ${severity}-${deterioration}`);
        }

        const model1Prob = cleanProbability(patient.model1_severe_probability || patient.model1_prob || patient.severe_probability || 0.5);
        const model2Prob = cleanProbability(patient.model2_deterioration_probability || patient.model2_prob || patient.deterioration_probability || 0.5);

        const confidence = (
            Math.abs(model1Prob - 0.5) * 2 +
            Math.abs(model2Prob - 0.5) * 2
        ) / 2;

        safeSetText("risk-score", resultMatrix.score);
        safeSetText("risk-band", resultMatrix.band);
        safeSetText("calibration-label", resultMatrix.action);
        safeSetText("severity-card", severity);
        safeSetText("deterioration-card", deterioration);
        safeSetText("confidence-card", `${(confidence * 100).toFixed(1)}%`);

        safeSetText(
            "decision-explanation",
            `Model 1 classified the patient as ${severity}. Model 2 classified deterioration risk as ${deterioration}. The patient maps to priority score ${resultMatrix.score}. ${resultMatrix.explanation}`
        );

        const predictionOutput = document.getElementById("prediction-output");
        if (predictionOutput) {
            predictionOutput.textContent = JSON.stringify(result, null, 2);
        }

        updateRingColour("patient-risk-ring", resultMatrix.score);
        safeSetText("backend-status", "Backend: connected");

        if (uploadName) {
            uploadName.textContent = file.name;
        }

    } catch (error) {
        console.error("FRONTEND ERROR:", error);
        alert("Frontend layout update failed: " + error.message);
    } finally {
        if (runButton) {
            runButton.disabled = false;
            runButton.textContent = "Generate Assessment";
        }
    }
}

const runPredictionButton = document.getElementById("run-prediction");
if (runPredictionButton) {
    runPredictionButton.addEventListener("click", function(e) {
        e.preventDefault();
        e.stopPropagation();
        runPrediction(e);
    });
}

// ─── SURGE SIMULATION ────────────────────────────────────────────────────────

const simulateSurgeButton = document.getElementById("simulate-surge");

if (simulateSurgeButton) {
    simulateSurgeButton.addEventListener("click", async (e) => {
        e.preventDefault();

        simulateSurgeButton.disabled = true;
        simulateSurgeButton.textContent = "Running...";

        const p4 = Number(document.getElementById("priority4")?.value) || 0;
        const p3 = Number(document.getElementById("priority3")?.value) || 0;
        const p2 = Number(document.getElementById("priority2")?.value) || 0;
        const p1 = Number(document.getElementById("priority1")?.value) || 0;
        const totalBeds = Number(document.getElementById("totalIcuBeds")?.value) || 0;
        const occupiedBeds = Number(document.getElementById("occupiedIcuBeds")?.value) || 0;
        const condition = document.getElementById("surgeCondition")?.value || "Severe Surge";

        const payload = {
            priority_4_patients: p4,
            priority_3_patients: p3,
            priority_2_patients: p2,
            priority_1_patients: p1,
            total_icu_beds: totalBeds,
            occupied_icu_beds: occupiedBeds,
            surge_condition: condition
        };

        try {
            const response = await fetch("http://127.0.0.1:5000/surge", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Backend returned ${response.status}`);
            }

            const data = await response.json();

            const recommended = data.recommended_strategy || "Unknown";

            safeSetText("surge-band", "Simulation Complete");
            safeSetText("surge-action", recommended);
            safeSetText("surge-mode-label", "Decision-support output");
            safeSetText("surge-explanation", "Lowest expected mortality strategy under current surge conditions.");

            const ringScore = recommended.includes("Combined") ? 4 : recommended.includes("Severity") ? 3 : 2;
            safeSetText("surge-score", ringScore);
            updateRingColour("surge-risk-ring", ringScore);

            // Build new content first, then swap in one atomic operation — no blank frame shown
            const tableDiv = document.getElementById("surge-results-table");
            if (tableDiv && data.results && data.results.length > 0) {
                const grid = document.createElement("div");
                grid.className = "surge-results-grid";

                data.results.forEach((row, index) => {
                    const card = document.createElement("div");
                    card.className = "surge-result-card" + (index === 0 ? " recommended-card" : "");
                    card.innerHTML = `
                        <h4>${row.strategy}</h4>
                        <div class="surge-metric"><span>ICU Used</span><strong>${row.icu_used}</strong></div>
                        <div class="surge-metric"><span>Under-triage</span><strong>${row.under_triage_count}</strong></div>
                        <div class="surge-metric"><span>Expected Deaths</span><strong>${row.expected_preventable_deaths}</strong></div>
                    `;
                    grid.appendChild(card);
                });

                // replaceChildren swaps old for new in one operation — no flicker
                tableDiv.replaceChildren(grid);
            }

        } catch (error) {
            console.error("SURGE ERROR:", error);
            alert("Surge simulation failed: " + error.message);
        } finally {
            simulateSurgeButton.disabled = false;
            simulateSurgeButton.textContent = "Run Surge Simulation";
        }
    });
}

// ─── SIDEBAR TOGGLE ──────────────────────────────────────────────────────────

const sidebarToggle = document.getElementById("sidebar-toggle");
const appShell = document.querySelector(".app-shell");

if (sidebarToggle && appShell) {
    sidebarToggle.addEventListener("click", () => {
        appShell.classList.toggle("sidebar-collapsed");
        const collapsed = appShell.classList.contains("sidebar-collapsed");
        sidebarToggle.setAttribute("aria-expanded", String(!collapsed));

        const span = sidebarToggle.querySelector("span");
        if (span) {
            span.textContent = collapsed ? "→" : "←";
        }
    });
}

// Initialize status indicator
safeSetText("backend-status", "Backend: Ready to connect");