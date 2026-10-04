/**
 * Frontend Application Controller for Tata Power Load Analysis Platform.
 * Coordinates Chart.js visualizations, DTW overlays, Causal/HTE cards, Safe Parameter-to-SQL,
 * and Grounded Local LLM explanations (LM Studio / gemma-4-e2b).
 */

const API_BASE = "";

// Global Chart References
let loadCurveChart = null;
let hteChart = null;
let shapChart = null;

let currentDayData = null;
let currentDtwMatches = [];
let showDtwOverlay = true;
let currentDayHteData = [];
let hteViewMode = "all"; // 'all' (all 6 treatment drivers) or 'temp' (temperature only)

// DOM Elements
const seasonSelect = document.getElementById("seasonSelect");
const dateSelect = document.getElementById("dateSelect");
const dayMeanLoadEl = document.getElementById("dayMeanLoad");
const dayPeakLoadEl = document.getElementById("dayPeakLoad");
const dayMeanTempEl = document.getElementById("dayMeanTemp");

const hteDateBadge = document.getElementById("hteDateBadge");
const hteDateDisplay = document.getElementById("hteDateDisplay");
const hteViewAllBtn = document.getElementById("hteViewAllBtn");
const hteViewTempBtn = document.getElementById("hteViewTempBtn");

const betaTempValueEl = document.getElementById("betaTempValue");
const tempCiEl = document.getElementById("tempCi");
const betaRhValueEl = document.getElementById("betaRhValue");
const rhCiEl = document.getElementById("rhCi");
const betaFestValueEl = document.getElementById("betaFestValue");
const festCiEl = document.getElementById("festCi");

const dtwMatchesList = document.getElementById("dtwMatchesList");
const toggleDtwMatchesBtn = document.getElementById("toggleDtwMatchesBtn");

const nlQueryInput = document.getElementById("nlQueryInput");
const runQueryBtn = document.getElementById("runQueryBtn");
const sqlPreviewCode = document.getElementById("sqlPreviewCode");
const queryAnswerVal = document.getElementById("queryAnswerVal");
const querySampleVal = document.getElementById("querySampleVal");

const generateExplanationBtn = document.getElementById("generateExplanationBtn");
const evidenceJsonViewer = document.getElementById("evidenceJsonViewer");
const explanationText = document.getElementById("explanationText");
const llmPromptInput = document.getElementById("llmPromptInput");
const askLlmBtn = document.getElementById("askLlmBtn");
const modelStatusTag = document.getElementById("modelStatusTag");
const explanationTitle = document.getElementById("explanationTitle");

// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  setupEventListeners();
  await handleSeasonChange();
});

function initCharts() {
  // 1. 96-Interval Diurnal Load Curve Chart
  const ctxLoad = document.getElementById("loadCurveChart").getContext("2d");
  const timeLabels = [];
  for (let h = 0; h < 24; h++) {
    for (let m = 0; m < 60; m += 15) {
      const hh = h.toString().padStart(2, "0");
      const mm = m.toString().padStart(2, "0");
      timeLabels.push(`${hh}:${mm}`);
    }
  }

  loadCurveChart = new Chart(ctxLoad, {
    type: "line",
    data: {
      labels: timeLabels,
      datasets: []
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          labels: { color: "#94a3b8", font: { family: "Inter", size: 12 } }
        },
        tooltip: {
          backgroundColor: "#1e293b",
          borderColor: "#334155",
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b", maxTicksLimit: 12 }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b" },
          title: { display: true, text: "Grid Load (MW)", color: "#94a3b8" }
        }
      }
    }
  });

  // 2. Diurnal 24-Hour HTE Chart (Dynamic single-day multi-driver diurnal effects)
  const ctxHte = document.getElementById("hteChart").getContext("2d");
  const hoursLabels = Array.from({ length: 24 }, (_, i) => `${i}:00`);

  hteChart = new Chart(ctxHte, {
    type: "line",
    data: {
      labels: hoursLabels,
      datasets: []
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: {
            color: "#94a3b8",
            font: { family: "Inter", size: 11 },
            usePointStyle: true,
            boxWidth: 8,
            padding: 12
          }
        },
        tooltip: {
          backgroundColor: "#1e293b",
          borderColor: "#334155",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: (ctx) => {
              const val = ctx.raw !== null && ctx.raw !== undefined ? Number(ctx.raw).toFixed(2) : "0.00";
              const prefix = Number(ctx.raw) > 0 ? "+" : "";
              return ` ${ctx.dataset.label}: ${prefix}${val} MW`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8", maxRotation: 0, autoSkip: false },
          title: { display: true, text: "Hour of the Day (0-23)", color: "#94a3b8" }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.07)" },
          ticks: { color: "#94a3b8" },
          title: { display: true, text: "Treatment Effect (MW)", color: "#94a3b8" }
        }
      }
    }
  });

  // 3. SHAP Feature Importance Chart
  const ctxShap = document.getElementById("shapChart").getContext("2d");
  shapChart = new Chart(ctxShap, {
    type: "bar",
    data: {
      labels: [],
      datasets: [
        {
          label: "Global SHAP Importance",
          data: [],
          backgroundColor: "rgba(59, 130, 246, 0.75)",
          borderColor: "#3b82f6",
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b" }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#cbd5e1", font: { family: "JetBrains Mono", size: 11 } }
        }
      }
    }
  });
}

function setupEventListeners() {
  seasonSelect.addEventListener("change", handleSeasonChange);
  dateSelect.addEventListener("change", handleDateChange);

  toggleDtwMatchesBtn.addEventListener("click", () => {
    showDtwOverlay = !showDtwOverlay;
    toggleDtwMatchesBtn.classList.toggle("active", showDtwOverlay);
    renderLoadCurve();
  });

  if (hteViewAllBtn) {
    hteViewAllBtn.addEventListener("click", () => {
      hteViewMode = "all";
      hteViewAllBtn.classList.add("active");
      if (hteViewTempBtn) hteViewTempBtn.classList.remove("active");
      renderHteChart(dateSelect.value);
    });
  }

  if (hteViewTempBtn) {
    hteViewTempBtn.addEventListener("click", () => {
      hteViewMode = "temp";
      hteViewTempBtn.classList.add("active");
      if (hteViewAllBtn) hteViewAllBtn.classList.remove("active");
      renderHteChart(dateSelect.value);
    });
  }

  runQueryBtn.addEventListener("click", executeNlQuery);
  nlQueryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") executeNlQuery();
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      nlQueryInput.value = chip.getAttribute("data-q");
      executeNlQuery();
    });
  });

  generateExplanationBtn.addEventListener("click", handleGenerateExplanation);

  askLlmBtn.addEventListener("click", handleUserPrompt);
  llmPromptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleUserPrompt();
  });

  document.querySelectorAll(".prompt-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      llmPromptInput.value = chip.getAttribute("data-p");
      handleUserPrompt();
    });
  });
}

async function handleSeasonChange() {
  const season = seasonSelect.value;

  // 1. Fetch available dates
  const resDates = await fetch(`${API_BASE}/api/dates?season=${season}`);
  const dataDates = await resDates.json();

  dateSelect.innerHTML = "";
  dataDates.dates.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    dateSelect.appendChild(opt);
  });

  // Select a relevant date (e.g. middle of season or Diwali for Autumn)
  if (season === "Autumn") {
    dateSelect.value = "2024-11-01"; // Diwali date
  } else if (dataDates.dates.length > 0) {
    dateSelect.value = dataDates.dates[Math.floor(dataDates.dates.length / 2)];
  }

  // 2. Fetch Causal Effects and Diurnal HTE
  const resCausal = await fetch(`${API_BASE}/api/causal/${season}`);
  const dataCausal = await resCausal.json();

  if (dataCausal.ate) {
    const t = dataCausal.ate.temp;
    if (t) {
      betaTempValueEl.innerHTML = `+${t.effect_value} <span class="unit">MW / °C</span>`;
      tempCiEl.textContent = `[+${t.ci_lower}, +${t.ci_upper}]`;
    }
    const rh = dataCausal.ate.rh;
    if (rh) {
      betaRhValueEl.innerHTML = `${rh.effect_value > 0 ? "+" : ""}${rh.effect_value} <span class="unit">MW / %</span>`;
      rhCiEl.textContent = `[${rh.ci_lower}, ${rh.ci_upper}]`;
    }
    const fest = dataCausal.ate.festival;
    if (fest) {
      betaFestValueEl.innerHTML = `+${fest.effect_value} <span class="unit">MW</span>`;
      festCiEl.textContent = `[+${fest.ci_lower}, +${fest.ci_upper}]`;
    }
  }

  // 3. Fetch SHAP Feature Importances
  const resXai = await fetch(`${API_BASE}/api/xai/${season}`);
  const dataXai = await resXai.json();
  if (dataXai.shap_features) {
    shapChart.data.labels = dataXai.shap_features.map((f) => f.feature);
    shapChart.data.datasets[0].data = dataXai.shap_features.map((f) => f.importance);
    shapChart.update();
  }

  await handleDateChange();
}

async function handleDateChange() {
  const dateStr = dateSelect.value;
  if (!dateStr) return;

  // 1. Fetch Day Curve
  const resDay = await fetch(`${API_BASE}/api/day/${dateStr}`);
  currentDayData = await resDay.json();

  dayMeanLoadEl.textContent = `${currentDayData.mean_load.toFixed(1)} MW`;
  dayPeakLoadEl.textContent = `${currentDayData.peak_load.toFixed(1)} MW`;
  dayMeanTempEl.textContent = `${currentDayData.mean_temp.toFixed(1)} °C`;

  // 2. Fetch Day-Specific 24-Hour Diurnal HTE (dynamic directly from SQLite load_data)
  try {
    const resHte = await fetch(`${API_BASE}/api/hte/day/${dateStr}`);
    const dataHte = await resHte.json();
    currentDayHteData = dataHte.hourly_hte || [];
    renderHteChart(dateStr);
  } catch (err) {
    console.error("Failed to load day diurnal HTE:", err);
  }

  // 3. Fetch DTW Top Matches
  dtwMatchesList.innerHTML = `<div class="loading-spinner">Searching nearest historical days...</div>`;
  const resDtw = await fetch(`${API_BASE}/api/dtw/${dateStr}`);
  const dataDtw = await resDtw.json();
  currentDtwMatches = dataDtw.matches || [];

  renderDtwList();
  renderLoadCurve();
}

function renderHteChart(dateStr) {
  if (!hteChart) return;
  const targetDate = dateStr || (dateSelect ? dateSelect.value : "");
  if (hteDateBadge && targetDate) hteDateBadge.textContent = targetDate;
  if (hteDateDisplay && targetDate) hteDateDisplay.textContent = targetDate;

  if (!currentDayHteData || currentDayHteData.length === 0) {
    hteChart.data.datasets = [];
    hteChart.update();
    return;
  }

  const hours = currentDayHteData.map((d) => `${d.hour}:00`);
  hteChart.data.labels = hours;

  if (hteViewMode === "temp") {
    hteChart.data.datasets = [
      {
        label: "Temperature Effect (MW/°C)",
        data: currentDayHteData.map((d) => d.temp_hte),
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245, 158, 11, 0.15)",
        fill: true,
        tension: 0.35,
        borderWidth: 2.5,
        pointStyle: "circle",
        pointRadius: 4,
        pointHoverRadius: 6
      }
    ];
  } else {
    // All 6 treatment drivers mirroring sociowinter_v2.ipynb Cell 13
    hteChart.data.datasets = [
      {
        label: "Temperature",
        data: currentDayHteData.map((d) => d.temp_hte),
        borderColor: "#f59e0b",
        backgroundColor: "#f59e0b",
        fill: false,
        tension: 0.25,
        borderWidth: 2,
        pointStyle: "circle",
        pointRadius: 3.5
      },
      {
        label: "Relative Humidity",
        data: currentDayHteData.map((d) => d.rh_hte),
        borderColor: "#06b6d4",
        backgroundColor: "#06b6d4",
        fill: false,
        tension: 0.25,
        borderWidth: 2,
        pointStyle: "rect",
        pointRadius: 3.5
      },
      {
        label: "Weekend",
        data: currentDayHteData.map((d) => d.weekend_hte),
        borderColor: "#8b5cf6",
        backgroundColor: "#8b5cf6",
        fill: false,
        tension: 0.25,
        borderWidth: 2,
        pointStyle: "triangle",
        pointRadius: 4
      },
      {
        label: "Holiday",
        data: currentDayHteData.map((d) => d.holiday_hte),
        borderColor: "#ec4899",
        backgroundColor: "#ec4899",
        fill: false,
        tension: 0.25,
        borderWidth: 2,
        pointStyle: "rectRot",
        pointRadius: 4
      },
      {
        label: "Festival",
        data: currentDayHteData.map((d) => d.festival_hte),
        borderColor: "#10b981",
        backgroundColor: "#10b981",
        fill: false,
        tension: 0.25,
        borderWidth: 2,
        pointStyle: "crossRot",
        pointRadius: 4
      },
      {
        label: "Office Hours",
        data: currentDayHteData.map((d) => d.office_hte),
        borderColor: "#3b82f6",
        backgroundColor: "#3b82f6",
        fill: false,
        tension: 0.25,
        borderWidth: 2,
        pointStyle: "star",
        pointRadius: 4
      }
    ];
  }

  hteChart.update();
}

function renderDtwList() {
  dtwMatchesList.innerHTML = "";
  if (!currentDtwMatches || currentDtwMatches.length === 0) {
    dtwMatchesList.innerHTML = `<p class="disclaimer-text">No prior dates match temporal mask filter.</p>`;
    return;
  }

  currentDtwMatches.forEach((m) => {
    const item = document.createElement("div");
    item.className = "match-item";
    item.innerHTML = `
      <div class="match-item-top">
        <span class="match-rank">#${m.rank} ${m.historical_date}</span>
        <span class="match-dist">dist: ${m.distance}</span>
      </div>
      <div class="match-metrics">
        <span>Mean: ${m.historical_mean_load.toFixed(1)} MW</span>
        <span>Peak: ${m.historical_peak_load.toFixed(1)} MW</span>
      </div>
    `;
    dtwMatchesList.appendChild(item);
  });
}

function renderLoadCurve() {
  if (!currentDayData) return;

  const datasets = [
    {
      label: `Observed Load (${currentDayData.date})`,
      data: currentDayData.raw_load,
      borderColor: "#38bdf8",
      backgroundColor: "rgba(56, 189, 248, 0.12)",
      fill: true,
      borderWidth: 2.5,
      tension: 0.35,
      pointRadius: 0
    }
  ];

  if (showDtwOverlay && currentDtwMatches) {
    const colors = ["#a855f7", "#ec4899", "#10b981"];
    currentDtwMatches.forEach((m, idx) => {
      datasets.push({
        label: `DTW #${m.rank}: ${m.historical_date}`,
        data: m.historical_curve,
        borderColor: colors[idx % colors.length],
        borderWidth: 1.5,
        borderDash: [5, 5],
        fill: false,
        tension: 0.35,
        pointRadius: 0
      });
    });
  }

  loadCurveChart.data.datasets = datasets;
  loadCurveChart.update();
}

async function executeNlQuery() {
  const q = nlQueryInput.value.trim();
  if (!q) return;

  runQueryBtn.disabled = true;
  runQueryBtn.textContent = "Running...";

  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: jsonString({ question: q })
    });
    const data = await res.json();

    sqlPreviewCode.textContent = data.sql || "N/A";
    queryAnswerVal.textContent = data.value !== null ? `${data.value} ${data.unit}` : "No match";
    querySampleVal.textContent = `${data.sample_count} intervals`;
  } catch (err) {
    sqlPreviewCode.textContent = `Error: ${err.message}`;
  } finally {
    runQueryBtn.disabled = false;
    runQueryBtn.textContent = "Execute";
  }
}

async function handleGenerateExplanation() {
  const dateStr = dateSelect.value;
  if (!dateStr) return;

  generateExplanationBtn.disabled = true;
  generateExplanationBtn.innerHTML = `<span>⏳ Synthesizing (gemma-4-e2b)...</span>`;
  explanationText.innerHTML = `<p class="placeholder-text">Prompting LM Studio local model (gemma-4-e2b)...</p>`;

  try {
    const res = await fetch(`${API_BASE}/api/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: jsonString({ date: dateStr })
    });
    const data = await res.json();

    // Render Evidence JSON
    evidenceJsonViewer.textContent = JSON.stringify(data.evidence, null, 2);

    // Render Markdown as formatted HTML
    explanationText.innerHTML = renderMarkdown(data.answer);
  } catch (err) {
    explanationText.innerHTML = `<p style="color: #f43f5e;">Error: ${err.message}</p>`;
  } finally {
    generateExplanationBtn.disabled = false;
    generateExplanationBtn.innerHTML = `<span>📅 Explain Selected Day</span>`;
  }
}

async function handleUserPrompt() {
  const prompt = llmPromptInput.value.trim();
  if (!prompt) return;

  askLlmBtn.disabled = true;
  askLlmBtn.innerHTML = `<span>⏳ Reasoning (gemma-4-e2b)...</span>`;
  explanationTitle.textContent = `Reasoning: "${prompt.slice(0, 38)}..."`;
  explanationText.innerHTML = `<p class="placeholder-text">Querying SQLite database, gathering EconML causal estimates, checking HTE & DTW, and prompting <strong>gemma-4-e2b</strong>...</p>`;

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: jsonString({
        prompt: prompt,
        date: dateSelect.value,
        season: seasonSelect.value
      })
    });
    const data = await res.json();

    // 1. Display full structured evidence packet
    evidenceJsonViewer.textContent = JSON.stringify(data.evidence, null, 2);

    // 2. Render reasoned answer
    explanationText.innerHTML = renderMarkdown(data.answer);

    // 3. Update model badge
    if (modelStatusTag) {
      modelStatusTag.textContent = data.model;
    }

    // 4. Also synchronize SQL Console preview if SQL was executed
    if (data.sql_executed) {
      sqlPreviewCode.textContent = data.sql_executed;
      queryAnswerVal.textContent = data.calculated_value !== null ? `${data.calculated_value} ${data.unit}` : "Multiple records";
      querySampleVal.textContent = `${data.sample_count} intervals`;
    }
  } catch (err) {
    explanationText.innerHTML = `<p style="color: #f43f5e;">Error processing prompt: ${err.message}</p>`;
  } finally {
    askLlmBtn.disabled = false;
    askLlmBtn.innerHTML = `<span>⚡ Ask gemma-4-e2b</span>`;
  }
}

function jsonString(obj) {
  return JSON.stringify(obj);
}

function renderMarkdown(md) {
  if (!md) return "";
  const paragraphs = md.split(/\n\s*\n/);
  return paragraphs.map(p => {
    let formatted = p.trim()
      .replace(/^### (.*$)/gim, "<h3>$1</h3>")
      .replace(/^## (.*$)/gim, "<h2>$1</h2>")
      .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/gim, "<em>$1</em>")
      .replace(/^[-*] (.*$)/gim, "<li>$1</li>")
      .replace(/\n/g, "<br>");
    return `<p style="margin-bottom: 0.85rem; line-height: 1.65;">${formatted}</p>`;
  }).join("");
}

