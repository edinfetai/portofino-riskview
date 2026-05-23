let weightsChart = null;
let riskChart = null;

const textTab = document.getElementById("textTab");
const imageTab = document.getElementById("imageTab");
const textPane = document.getElementById("textPane");
const imagePane = document.getElementById("imagePane");
const loadingPanel = document.getElementById("loadingPanel");
const errorPanel = document.getElementById("errorPanel");
const errorText = document.getElementById("errorText");

function switchTab(mode) {
    const textActive = mode === "text";
    textTab.classList.toggle("active", textActive);
    imageTab.classList.toggle("active", !textActive);
    textPane.classList.toggle("active", textActive);
    imagePane.classList.toggle("active", !textActive);
}

textTab.addEventListener("click", () => switchTab("text"));
imageTab.addEventListener("click", () => switchTab("image"));

document.querySelectorAll(".example-card").forEach((button) => {
    button.addEventListener("click", () => {
        document.getElementById("portfolioText").value = button.dataset.example || "";
    });
});


let selectedExampleImage = null;

document.querySelectorAll(".image-example-card").forEach((button) => {
    button.addEventListener("click", () => {
        const examplePreview = button.querySelector("img");
        const exampleLabel = button.querySelector("span")?.textContent?.trim() || "Beispielbild";
        const exampleKey = button.dataset.exampleImage || null;

        selectedExampleImage = exampleKey;

        // Ein zuvor selbst hochgeladenes Bild wird abgewählt,
        // damit eindeutig ist, welches Bild analysiert werden soll.
        portfolioImageInput.value = "";

        document.querySelectorAll(".image-example-card").forEach((card) => {
            card.classList.remove("selected");
        });
        button.classList.add("selected");

        if (examplePreview?.src) {
            showPortfolioPreview(
                examplePreview.src,
                `Beispielbild ausgewählt: ${exampleLabel}`,
                false
            );
        }

        clearError();
    });
});

const portfolioImageInput = document.getElementById("portfolioImage");
const portfolioImageDrop = document.getElementById("portfolioImageDrop");
const portfolioPreview = document.getElementById("portfolioPreview");
const portfolioPreviewCaption = document.getElementById("portfolioPreviewCaption");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const fileName = document.getElementById("fileName");

let currentPortfolioPreviewUrl = null;

function releasePortfolioPreviewUrl() {
    if (currentPortfolioPreviewUrl) {
        URL.revokeObjectURL(currentPortfolioPreviewUrl);
        currentPortfolioPreviewUrl = null;
    }
}

function showPortfolioPreview(src, caption, objectUrl = false) {
    releasePortfolioPreviewUrl();

    if (objectUrl) {
        currentPortfolioPreviewUrl = src;
    }

    portfolioPreview.src = src;
    portfolioPreview.classList.remove("hidden");
    portfolioPreviewCaption.textContent = caption || "Portfolio-Bild ausgewählt";
    portfolioPreviewCaption.classList.remove("hidden");
    uploadPlaceholder.classList.add("hidden");
    portfolioImageDrop.classList.add("has-preview");
}

function clearPortfolioPreview() {
    releasePortfolioPreviewUrl();
    portfolioPreview.src = "";
    portfolioPreview.classList.add("hidden");
    portfolioPreviewCaption.textContent = "";
    portfolioPreviewCaption.classList.add("hidden");
    uploadPlaceholder.classList.remove("hidden");
    portfolioImageDrop.classList.remove("has-preview");
    fileName.textContent = "PNG, JPG oder WEBP auswählen";
}

portfolioImageInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];

    if (!file) {
        clearPortfolioPreview();
        return;
    }

    selectedExampleImage = null;
    document.querySelectorAll(".image-example-card").forEach((card) => {
        card.classList.remove("selected");
    });

    fileName.textContent = file.name;
    const previewUrl = URL.createObjectURL(file);
    showPortfolioPreview(previewUrl, file.name, true);
});

function setLoading(isLoading) {
    loadingPanel.classList.toggle("hidden", !isLoading);
    document.getElementById("analyzeTextButton").disabled = isLoading;
    document.getElementById("analyzeImageButton").disabled = isLoading;
}

function showError(message) {
    errorText.textContent = message;
    errorPanel.classList.remove("hidden");
}

function clearError() {
    errorPanel.classList.add("hidden");
    errorText.textContent = "";
}

function formatPercent(value, decimals = 1) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return `${(Number(value) * 100).toFixed(decimals)}%`;
}

function formatDecimal(value, decimals = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return Number(value).toFixed(decimals);
}

function shortSummary(text) {
    if (!text) return "Die Analyse wurde erfolgreich ausgeführt.";
    const sentences = text.split(/(?<=[.!?])\s+/);
    return sentences.slice(0, 1).join(" ");
}

function localTimeLabel() {
    const now = new Date();
    return now.toLocaleTimeString("de-CH", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    }) + " CET";
}

async function requestJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json();

    if (!response.ok) {
        const message = payload?.detail || "Die Analyse konnte nicht abgeschlossen werden.";
        throw new Error(message);
    }

    return payload.data;
}

document.getElementById("analyzeTextButton").addEventListener("click", async () => {
    clearError();
    setLoading(true);

    try {
        const text = document.getElementById("portfolioText").value.trim();
        if (!text) {
            throw new Error("Bitte gib zuerst eine Portfolio-Beschreibung ein.");
        }

        const data = await requestJson("/api/analyze/text", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({text}),
        });

        renderDashboard(data);
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
});

document.getElementById("analyzeImageButton").addEventListener("click", async () => {
    clearError();
    setLoading(true);

    try {
        const file = document.getElementById("portfolioImage").files?.[0];
        const question = document.getElementById("imageQuestion").value.trim();

        let data = null;

        if (file) {
            const formData = new FormData();
            formData.append("image", file);
            formData.append("question", question);

            data = await requestJson("/api/analyze/image", {
                method: "POST",
                body: formData,
            });
        } else if (selectedExampleImage) {
            const params = new URLSearchParams({
                example: selectedExampleImage,
                question: question || "Analysiere mein Portfolio-Risiko.",
            });

            data = await requestJson(`/api/analyze/example-image?${params.toString()}`, {
                method: "POST",
            });
        } else {
            throw new Error(
                "Bitte lade zuerst einen Portfolio-Screenshot hoch oder wähle eines der Beispielbilder aus."
            );
        }

        renderDashboard(data);
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
});

function renderDashboard(data) {
    document.getElementById("dashboard").classList.remove("hidden");

    const prediction = data.prediction;
    const features = prediction.model_features;
    const probabilities = prediction.risk_probabilities;
    const riskClass = prediction.risk_class;
    const riskClassDe = prediction.risk_class_de;
    const confidence = probabilities[riskClass] || 0;

    document.getElementById("lastUpdate").textContent = localTimeLabel();
    document.getElementById("riskLabel").textContent = riskClassDe;
    document.getElementById("riskConfidence").textContent = `${(confidence * 100).toFixed(1)}%`;
    document.getElementById("riskTrackFill").style.width = `${Math.max(confidence * 100, 4)}%`;
    document.getElementById("riskSummary").textContent = shortSummary(data.explanation);
    document.getElementById("riskId").textContent = `ID: RISK-${Math.floor(confidence * 1000).toString().padStart(3, "0")}`;

    const riskCard = document.getElementById("riskCard");
    riskCard.classList.remove("risk-low", "risk-medium", "risk-high");
    riskCard.classList.add(`risk-${riskClass.toLowerCase()}`);

    setMetricDisplay("metricVolatility", formatPercent(features.volatility_63d), classifyVolatility(features.volatility_63d));
    setMetricDisplay("metricDrawdown", formatPercent(features.max_drawdown_63d), classifyDrawdown(features.max_drawdown_63d));
    setMetricDisplay("metricBeta", formatDecimal(features.market_beta), classifyBeta(features.market_beta));
    setMetricDisplay("metricLargest", formatPercent(features.largest_position_weight), classifyLargestPosition(features.largest_position_weight));
    setMetricDisplay("metricTop3", formatPercent(features.top_3_concentration), classifyTop3(features.top_3_concentration));
    setMetricDisplay("metricCorrelation", formatDecimal(features.average_correlation), classifyCorrelation(features.average_correlation));

    document.getElementById("aiExplanation").textContent = data.explanation;

    document.getElementById("extractedJson").textContent = JSON.stringify(
        data.extracted,
        null,
        2
    );

    document.getElementById("technicalJson").textContent = JSON.stringify(
        prediction,
        null,
        2
    );

    const freshness = prediction.data_freshness || {};
    document.getElementById("freshMarket").textContent = freshness.market_source || "—";
    document.getElementById("freshInterval").textContent = freshness.price_interval || "—";
    document.getElementById("freshTimestamp").textContent = String(freshness.latest_price_timestamp || "—").slice(0, 10);
    document.getElementById("freshRows").textContent = freshness.number_of_price_rows ?? "—";

    renderStressTest(data.dashboard.stress_scenario);
    renderWeightsChart(data.dashboard.portfolio_weights);
    renderRiskChart(data.dashboard.risk_probability_chart);
}


function setMetricDisplay(elementId, value, state) {
    const element = document.getElementById(elementId);
    element.textContent = value;
    element.classList.remove("metric-good", "metric-neutral", "metric-bad");
    element.classList.add(state);
}

function classifyVolatility(value) {
    const v = Number(value);
    if (v <= 0.15) return "metric-good";
    if (v >= 0.30) return "metric-bad";
    return "metric-neutral";
}

function classifyDrawdown(value) {
    const absoluteDrawdown = Math.abs(Number(value));
    if (absoluteDrawdown <= 0.08) return "metric-good";
    if (absoluteDrawdown >= 0.15) return "metric-bad";
    return "metric-neutral";
}

function classifyBeta(value) {
    const v = Number(value);
    if (v <= 0.80) return "metric-good";
    if (v >= 1.25) return "metric-bad";
    return "metric-neutral";
}

function classifyLargestPosition(value) {
    const v = Number(value);
    if (v <= 0.35) return "metric-good";
    if (v >= 0.60) return "metric-bad";
    return "metric-neutral";
}

function classifyTop3(value) {
    const v = Number(value);
    if (v <= 0.65) return "metric-good";
    if (v >= 0.85) return "metric-bad";
    return "metric-neutral";
}

function classifyCorrelation(value) {
    const v = Number(value);
    if (v <= 0.35) return "metric-good";
    if (v >= 0.65) return "metric-bad";
    return "metric-neutral";
}

function renderStressTest(stress) {
    const container = document.getElementById("stressContent");

    if (!stress || stress.portfolio_reaction === null || stress.portfolio_reaction === undefined) {
        container.innerHTML = "<p>Kein explizites Stress-Szenario erkannt.</p>";
        return;
    }

    const scenario = stress.description || "Individuelles Schockszenario";
    const reaction = formatPercent(stress.portfolio_reaction);
    const shocks = stress.shocks || {};

    const items = Object.entries(shocks).map(([ticker, value]) => {
        const cls = Number(value) < 0 ? "stress-negative" : "stress-positive";
        return `<li><span>${ticker}</span><strong class="${cls}">${formatPercent(value)}</strong></li>`;
    }).join("");

    container.innerHTML = `
        <p><strong>Szenario:</strong> ${scenario}</p>
        <ul class="stress-list">${items}</ul>
        <p><strong>Portfolio-Reaktion:</strong> <span class="stress-negative">${reaction}</span></p>
    `;
}

function renderWeightsChart(weights) {
    const labels = weights.map((item) => item.ticker);
    const values = weights.map((item) => item.weight_pct);

    if (weightsChart) weightsChart.destroy();

    weightsChart = new Chart(document.getElementById("weightsChart"), {
        type: "bar",
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: ["#49c49e", "#35bd92", "#22b382", "#11a674", "#0d9568", "#0a865d"],
                borderRadius: 4,
                barPercentage: 0.78,
                categoryPercentage: 0.74,
            }],
        },
        options: baseChartOptions({
            yMax: Math.max(...values, 40) * 1.15,
            valueSuffix: "%",
        }),
    });
}

function renderRiskChart(items) {
    const labels = items.map((item) => item.label);
    const values = items.map((item) => item.value_pct);

    if (riskChart) riskChart.destroy();

    riskChart = new Chart(document.getElementById("riskChart"), {
        type: "bar",
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: ["#37ad69", "#d89d1e", "#e7474f"],
                borderRadius: 4,
                barPercentage: 0.72,
                categoryPercentage: 0.72,
            }],
        },
        options: baseChartOptions({
            yMax: Math.max(...values, 80) * 1.05,
            valueSuffix: "%",
        }),
    });
}

function baseChartOptions({yMax, valueSuffix}) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: {duration: 350},
        plugins: {
            legend: {display: false},
            tooltip: {
                callbacks: {
                    label(context) {
                        return `${context.raw.toFixed(1)}${valueSuffix}`;
                    },
                },
            },
        },
        scales: {
            x: {
                grid: {display: false},
                ticks: {
                    color: "#737987",
                    font: {family: "IBM Plex Mono", size: 9},
                },
                border: {color: "#ececf0"},
            },
            y: {
                beginAtZero: true,
                suggestedMax: yMax,
                grid: {
                    color: "#ececf0",
                    borderDash: [4, 4],
                },
                ticks: {
                    color: "#737987",
                    callback(value) {
                        return `${value}%`;
                    },
                    font: {family: "IBM Plex Mono", size: 9},
                },
                border: {display: false},
            },
        },
    };
}
