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

// FIX: store the file reference before the button is clicked so it is never lost
let storedFile = null;

const upload = document.getElementById("biomarker-upload");
const uploadName = document.getElementById("upload-name");

if (upload && uploadName) {
    upload.addEventListener("change", () => {
        if (upload.files && upload.files[0]) {
            storedFile = upload.files[0];           // save reference immediately
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

    ring.style.setProperty(
        "--ring-colour",
        riskColour(score)
    );

    const percentages = {
        1: "25%",
        2: "50%",
        3: "75%",
        4: "100%"
    };

    ring.style.setProperty(
        "--risk",
        percentages[score] || "0%"
    );
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

    // FIX: use the stored file reference instead of re-reading from the input
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

        // Restore the filename display after successful prediction
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

// FIX: use "click" only — NOT "mousedown" which fires before the browser
// finishes processing the file input, causing the displayed filename to reset.
const runPredictionButton = document.getElementById("run-prediction");
if (runPredictionButton) {
    runPredictionButton.addEventListener("click", function(e) {
        e.preventDefault();
        e.stopPropagation();
        runPrediction(e);
    });
}

document.querySelectorAll("[data-surge-mode]").forEach((button) => {
    button.addEventListener("click", () => {
        surgeMode = button.dataset.surgeMode;
        document.querySelectorAll("[data-surge-mode]").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
    });
});

const simulateSurgeButton = document.getElementById("simulate-surge");
if (simulateSurgeButton) {
    simulateSurgeButton.addEventListener("click", (e) => {
        e.preventDefault();
        const icuCapacity = Number(document.getElementById("icu-capacity")?.value || 0);
        const wardPressure = Number(document.getElementById("ward-pressure")?.value || 0);
        const incoming = Number(document.getElementById("incoming-patients")?.value || 0);
        const highRisk = Number(document.getElementById("high-risk-proportion")?.value || 0);
        const framework = document.getElementById("triage-framework")?.value || "";

        const pressureScore = Math.round(
            (100 - icuCapacity) * 0.35 +
            wardPressure * 0.30 +
            (incoming / 100) * 0.20 * 100 +
            highRisk * 0.15
        );

        let band;
        let action;
        let explanation;

        if (surgeMode === "recommend") {
            band = "Recommended strategy";
            action = "Use combined matrix allocation";
            explanation = "The combined matrix allocation strategy is recommended because it uses both current severity and predicted deterioration risk. This supports patient-level prioritisation during surge conditions instead of relying only on current severity or only on predicted decline.";
        } else {
            if (framework === "severity-first") {
                band = "Severity-first framework tested";
                action = "Useful for immediate severe cases";
                explanation = "Severity-first allocation prioritises patients who are already severe. It is simple and clinically intuitive, but may miss non-severe patients likely to deteriorate.";
            } else if (framework === "deterioration-first") {
                band = "Deterioration-risk-first framework tested";
                action = "Useful for early escalation";
                explanation = "Deterioration-risk-first allocation prioritises patients predicted to decline. It can support early intervention, but may under-prioritise patients already classified as severe.";
            } else {
                band = "Combined matrix framework tested";
                action = "Best aligned with this project";
                explanation = "Combined matrix allocation integrates Model 1 binary severity and Model 2 deterioration risk, making it the most aligned with the project aim of linking patient-level prediction to ICU escalation decisions.";
            }

            if (pressureScore >= 70) {
                explanation += " Under severe surge pressure, this framework should be applied with strict escalation thresholds.";
            } else if (pressureScore >= 40) {
                explanation += " Under moderate surge pressure, high-risk patients should be reviewed early.";
            } else {
                explanation += " Under manageable surge pressure, standard escalation workflow remains feasible.";
            }
        }

        safeSetText("surge-score", `${pressureScore}%`);
        safeSetText("surge-band", band);
        safeSetText("surge-action", action);
        safeSetText("surge-explanation", explanation);
        safeSetText("surge-mode-label", surgeMode === "recommend" ? "Strategy recommendation" : "Framework test");

        const colourScore = pressureScore < 40 ? 1 : pressureScore < 70 ? 3 : 4;
        updateRingColour("surge-risk-ring", colourScore);
    });
}

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

safeSetText("backend-status", "Backend: connecting...");