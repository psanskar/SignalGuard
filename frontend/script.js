const API_BASE_URL = "http://127.0.0.1:5000";

const analyzeView = document.getElementById("analyzeView");
const resultsView = document.getElementById("resultsView");
const errorView = document.getElementById("errorView");

const brandButton = document.getElementById("brandButton");
const backButton = document.getElementById("backButton");
const retryButton = document.getElementById("retryButton");

const urlTab = document.getElementById("urlTab");
const manualTab = document.getElementById("manualTab");
const urlPanel = document.getElementById("urlPanel");
const manualPanel = document.getElementById("manualPanel");

const jobUrl = document.getElementById("jobUrl");
const analyzeUrlButton = document.getElementById("analyzeUrlButton");
const jobForm = document.getElementById("jobForm");
const analyzeButton = document.getElementById("analyzeButton");

const urlError = document.getElementById("urlError");
const manualError = document.getElementById("manualError");
const loadingInline = document.getElementById("loadingInline");

const assessmentCard = document.getElementById("assessmentCard");
const riskBadge = document.getElementById("riskBadge");
const warningCount = document.getElementById("warningCount");
const assessmentHeadline = document.getElementById("assessmentHeadline");
const assessmentText = document.getElementById("assessmentText");
const fraudProbability = document.getElementById("fraudProbability");
const thresholdEl = document.getElementById("threshold");
const mlClassification = document.getElementById("mlClassification");
const probabilityFill = document.getElementById("probabilityFill");
const summary = document.getElementById("summary");
const warningCountBadge = document.getElementById("warningCountBadge");

const redFlags = document.getElementById("redFlags");
const shapSignals = document.getElementById("shapSignals");
const ragResults = document.getElementById("ragResults");
const riskExplanation = document.getElementById("riskExplanation");
const recommendations = document.getElementById("recommendations");
const confidenceNote = document.getElementById("confidenceNote");

const sourceMeta = document.getElementById("sourceMeta");
const postingSource = document.getElementById("postingSource");
const postingPreview = document.getElementById("postingPreview");

const errorMessage = document.getElementById("errorMessage");

let lastAnalysisMode = "url";
let lastAnalysisInput = null;
let lastResult = null;

function showView(view, shouldScroll = false) {
  [analyzeView, resultsView, errorView].forEach((node) => {
    node.classList.add("hidden");
  });

  view.classList.remove("hidden");

  if (shouldScroll) {
    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  }
}

function setBusy(busy) {
  analyzeButton.disabled = busy;
  analyzeUrlButton.disabled = busy;
  urlTab.disabled = busy;
  manualTab.disabled = busy;

  if (busy) {
    loadingInline.classList.remove("hidden");

    analyzeButton.querySelector("span").textContent = "Analyzing...";
    analyzeUrlButton.querySelector("span").textContent = "Analyzing...";
  } else {
    loadingInline.classList.add("hidden");

    analyzeButton.querySelector("span").textContent = "Analyze Job";
    analyzeUrlButton.querySelector("span").textContent = "Analyze Job";

    urlTab.disabled = false;
    manualTab.disabled = false;
  }
}

function setTab(tab) {
  const urlActive = tab === "url";

  urlTab.classList.toggle("active", urlActive);
  manualTab.classList.toggle("active", !urlActive);

  urlTab.setAttribute("aria-selected", String(urlActive));
  manualTab.setAttribute("aria-selected", String(!urlActive));

  urlPanel.classList.toggle("hidden", !urlActive);
  manualPanel.classList.toggle("hidden", urlActive);

  clearInlineErrors();
}

function clearInlineErrors() {
  urlError.textContent = "";
  manualError.textContent = "";

  urlError.classList.add("hidden");
  manualError.classList.add("hidden");
}

function showFieldError(node, message) {
  node.textContent = message;
  node.classList.remove("hidden");
}

function formatCategory(value) {
  return String(value || "unknown")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function normalizeRiskCode(data) {
  const raw = String(
    data?.risk_code || data?.risk_level || ""
  )
    .trim()
    .toLowerCase();

  if (raw.includes("high")) return "high";

  if (
    raw.includes("medium") ||
    raw.includes("moderate")
  ) {
    return "medium";
  }

  if (raw.includes("low")) return "low";

  return "unknown";
}

function riskHeadline(code, count) {
  if (code === "high") {
    return count === 1
      ? "Potential fraud indicator detected"
      : "Potential fraud indicators detected";
  }

  if (code === "medium") {
    return "Proceed with caution";
  }

  if (code === "low") {
    return "No significant warning signs detected";
  }

  return "Assessment unavailable";
}

function riskSummary(code, count) {
  if (code === "high") {
    return count === 1
      ? "This posting contains a high-severity warning sign associated with fraudulent job offers. Review the evidence before proceeding."
      : "This posting contains multiple signals commonly associated with fraudulent job offers. Review the warning signs and supporting evidence before proceeding.";
  }

  if (code === "medium") {
    return "This posting contains one or more signals that warrant caution. Review the warning signs and verify the opportunity before proceeding.";
  }

  if (code === "low") {
    return "No significant rule-based fraud indicators were detected in the available posting information. Standard verification practices are still recommended.";
  }

  return count
    ? `${count} warning sign(s) detected.`
    : "Review the available evidence before proceeding.";
}

function mlClassificationText(data) {
  if (typeof data?.ml_prediction === "boolean") {
    return data.ml_prediction
      ? "Fraud signal detected"
      : "Not classified as fraudulent";
  }

  return "Unavailable";
}

function getPostingData(data) {
  // ---------------------------------------------------------
  // 1. Manual analysis
  // We already have the exact job data entered by the user.
  // Use that directly for the preview.
  // ---------------------------------------------------------
  if (
    lastAnalysisMode === "manual" &&
    lastAnalysisInput &&
    typeof lastAnalysisInput === "object"
  ) {
    return {
      title: lastAnalysisInput.title || "",
      company: lastAnalysisInput.company_profile || "",
      description: lastAnalysisInput.description || "",
      requirements: lastAnalysisInput.requirements || "",
      benefits: lastAnalysisInput.benefits || ""
    };
  }

  // ---------------------------------------------------------
  // 2. URL analysis
  // Prefer actual extracted posting content returned
  // by the backend.
  // ---------------------------------------------------------
  const candidates = [
    data?.extracted_preview,
    data?.job_data,
    data?.job,
    data?.posting
  ];

  const source = candidates.find(
    (value) =>
      value &&
      typeof value === "object" &&
      (
        value.title ||
        value.job_title ||
        value.company ||
        value.company_profile ||
        value.description ||
        value.requirements ||
        value.benefits
      )
  );

  if (!source) {
    return {};
  }

  return {
    title: source.title || source.job_title || "",
    company:
      source.company_profile ||
      source.company ||
      "",
    description: source.description || "",
    requirements: source.requirements || "",
    benefits: source.benefits || ""
  };
}

async function analyzeJob(endpoint, body) {
  setBusy(true);
  clearInlineErrors();
  showView(analyzeView, false);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    let data = {};

    try {
      data = await response.json();
    } catch {
      throw new Error(
        "The server returned an invalid response."
      );
    }

    if (!response.ok) {
      throw new Error(
        data.error || "Unable to analyze the job."
      );
    }

    lastResult = data;

    displayResults(data);

    showView(resultsView, true);
  } catch (error) {
    console.error(
      "SignalGuard API Error:",
      error
    );

    errorMessage.textContent =
      error.message ||
      "The analysis could not be completed. Please try again.";

    showView(errorView, true);
  } finally {
    setBusy(false);
  }
}

function renderWarningFlags(flags) {
  redFlags.innerHTML = "";

  if (
    !Array.isArray(flags) ||
    flags.length === 0
  ) {
    redFlags.innerHTML = `
      <div class="warning-card">
        <div class="warning-top">
          <span class="warning-title">
            No significant warning signs detected
          </span>

          <span class="severity-badge low">
            NONE
          </span>
        </div>

        <p class="warning-description">
          The available posting information did not trigger
          the configured SignalGuard warning checks.
        </p>
      </div>
    `;

    return;
  }

  flags.forEach((flag) => {
    const severity = String(
      flag.severity || "unknown"
    ).toLowerCase();

    const tone = severity.includes("high")
      ? "high"
      : severity.includes("medium") ||
        severity.includes("moderate")
        ? "medium"
        : severity.includes("low")
          ? "low"
          : "medium";

    const title = formatCategory(flag.category);

    const message =
      flag.message ||
      flag.description ||
      "A warning sign was detected.";

    const rawMatches = Array.isArray(
      flag.matches
    )
      ? flag.matches
      : Array.isArray(flag.matched_text)
        ? flag.matched_text
        : flag.matched_text
          ? [flag.matched_text]
          : [];

    const evidenceItems =
      rawMatches.length
        ? rawMatches
        : ["Specific evidence not provided"];

    const article =
      document.createElement("article");

    article.className =
      `warning-card ${tone}`;

    article.innerHTML = `
      <div class="warning-top">
        <span class="warning-title"></span>

        <span class="severity-badge ${tone}">
          ${tone.toUpperCase()}
        </span>
      </div>

      <p class="warning-description"></p>

      <div class="evidence-box">
        <div class="evidence-label">
          Evidence
        </div>

        <ul class="evidence-list-inline"></ul>
      </div>
    `;

    article.querySelector(
      ".warning-title"
    ).textContent = title;

    article.querySelector(
      ".warning-description"
    ).textContent = message;

    const list =
      article.querySelector(
        ".evidence-list-inline"
      );

    evidenceItems.forEach((item) => {
      const li =
        document.createElement("li");

      li.className = "evidence-chip";

      li.textContent = String(item);

      list.appendChild(li);
    });

    redFlags.appendChild(article);
  });
}

function renderShapSignals(signals) {
  shapSignals.innerHTML = "";

  if (
    !Array.isArray(signals) ||
    signals.length === 0
  ) {
    shapSignals.innerHTML = `
      <div class="evidence-empty">
        No model signals available.
      </div>
    `;

    return;
  }

  const sorted = [...signals].sort(
    (a, b) =>
      Math.abs(
        Number(b.contribution || 0)
      ) -
      Math.abs(
        Number(a.contribution || 0)
      )
  );

  const visible = sorted.slice(0, 8);

  const maxAbs = Math.max(
    ...visible.map((s) =>
      Math.abs(
        Number(s.contribution || 0)
      )
    ),
    0.0001
  );

  visible.forEach((signal) => {
    const value = Number(
      signal.contribution || 0
    );

    const positive = value > 0;

    const width = Math.max(
      5,
      Math.min(
        50,
        (Math.abs(value) / maxAbs) * 50
      )
    );

    const feature = formatCategory(
      signal.feature || "Unknown"
    );

    const direction = positive
      ? "Increasing fraud risk"
      : "Supporting legitimacy";

    const row =
      document.createElement("div");

    row.className = "signal-row";

    row.innerHTML = `
      <div class="signal-top">
        <span class="signal-feature"></span>

        <span class="signal-value ${
          positive ? "fraud" : "legit"
        }"></span>
      </div>

      <div class="signal-track">
        <div class="signal-fill ${
          positive ? "fraud" : "legit"
        }"></div>
      </div>

      <div class="signal-direction"></div>
    `;

    row.querySelector(
      ".signal-feature"
    ).textContent = feature;

    row.querySelector(
      ".signal-value"
    ).textContent =
      `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;

    row.querySelector(
      ".signal-direction"
    ).textContent = direction;

    row.querySelector(
      ".signal-fill"
    ).style.width = `${width}%`;

    shapSignals.appendChild(row);
  });
}

function cleanRagText(text) {
  return String(text || "")
    .replace(/^#+\s*/gm, "")
    .replace(
      /\*\*(.*?)\*\*/g,
      "$1"
    )
    .replace(
      /\*(.*?)\*/g,
      "$1"
    )
    .replace(
      /`(.*?)`/g,
      "$1"
    )
    .replace(
      /^\s*[-*]\s+/gm,
      "• "
    )
    .replace(
      /\s+/g,
      " "
    )
    .trim();
}

function renderRag(data) {
  ragResults.innerHTML = "";

  const items =
    Array.isArray(
      data?.rag?.results
    )
      ? data.rag.results
      : [];

  if (items.length === 0) {
    ragResults.innerHTML = `
      <div class="evidence-empty">
        No specific scam-related safety guidance
        was retrieved for this assessment.
        General job-safety checks may still be useful.
      </div>
    `;

    return;
  }

  items.slice(0, 4).forEach((item) => {
    const source = formatCategory(
      String(
        item.source ||
        "Safety guidance"
      ).replace(".md", "")
    );

    const score = (
      Number(item.score || 0) * 100
    ).toFixed(1);

    const content = cleanRagText(
      item.content || ""
    );

    const card =
      document.createElement("article");

    card.className =
      "evidence-card";

    const top =
      document.createElement("div");

    top.className =
      "evidence-card-top";

    const title =
      document.createElement("div");

    title.className =
      "evidence-source";

    title.textContent = source;

    const match =
      document.createElement("span");

    match.className =
      "evidence-score";

    match.textContent = "Relevant guidance";

    top.append(
      title,
      match
    );

    const p =
      document.createElement("p");

    p.className =
      "evidence-text";

    p.textContent =
      content.length > 320
        ? `${content
            .slice(0, 320)
            .trim()}…`
        : content;

    card.append(
      top,
      p
    );

    ragResults.appendChild(card);
  });
}

function renderRecommendations(data) {
  recommendations.innerHTML = "";

  const items =
    Array.isArray(
      data?.llm_report
        ?.safety_recommendations
    )
      ? data.llm_report
          .safety_recommendations
      : Array.isArray(
          data?.recommendations
        )
        ? data.recommendations
        : [];

  if (items.length === 0) {
    recommendations.innerHTML = `
      <div class="evidence-empty">
        No specific recommendations were returned.
        Review the detected evidence carefully before proceeding.
      </div>
    `;

    return;
  }

  items.forEach((recommendation) => {
    const row =
      document.createElement("div");

    row.className =
      "recommendation-item";

    row.innerHTML = `
      <span
        class="recommendation-icon"
        aria-hidden="true"
      >✓</span>

      <span></span>
    `;

    row.querySelector(
      "span:last-child"
    ).textContent =
      recommendation;

    recommendations.appendChild(row);
  });
}

function renderPostingPreview(data) {
  const posting = getPostingData(data);
  const sourceUrl = data?.source_url || "";

  postingSource.textContent = sourceUrl
    ? "Source URL available below"
    : "Job details supplied for this analysis";

  // Nothing available
  if (
    !posting.title &&
    !posting.company &&
    !posting.description &&
    !posting.requirements &&
    !posting.benefits
  ) {
    postingPreview.innerHTML = `
      <div class="posting-empty">
        The original job posting content was not available
        in the analysis response.
      </div>
    `;

    return;
  }

  postingPreview.innerHTML = "";

  // ---------------------------------------------------------
  // Header
  // ---------------------------------------------------------
  const head = document.createElement("div");
  head.className = "posting-head";

  const title = document.createElement("h3");
  title.className = "posting-title";
  title.textContent = posting.title || "Job posting";

  const company = document.createElement("p");
  company.className = "posting-company";
  company.textContent =
    posting.company || "Company information unavailable";

  head.append(title, company);
  postingPreview.appendChild(head);

  // ---------------------------------------------------------
  // Content sections
  // ---------------------------------------------------------
  const sections = [
    ["DESCRIPTION", posting.description],
    ["REQUIREMENTS", posting.requirements],
    ["BENEFITS", posting.benefits]
  ];

  sections.forEach(([label, value]) => {
    if (!value) return;

    const block = document.createElement("div");
    block.className = "posting-block";

    const heading = document.createElement("h3");
    heading.textContent = label;

    const p = document.createElement("p");
    p.textContent = value;

    block.append(heading, p);
    postingPreview.appendChild(block);
  });

  // ---------------------------------------------------------
  // URL source
  // ---------------------------------------------------------
  if (sourceUrl) {
    const sourceBlock = document.createElement("div");
    sourceBlock.className = "posting-source-block";

    const label = document.createElement("span");
    label.textContent = "SOURCE";

    const link = document.createElement("a");
    link.href = sourceUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = sourceUrl;

    sourceBlock.append(label, link);
    postingPreview.appendChild(sourceBlock);
  }
}

function displayResults(data) {
  const riskCode =
    normalizeRiskCode(data);

  const flags =
    Array.isArray(data?.red_flags)
      ? data.red_flags
      : [];

  const count = flags.length;

  assessmentCard.classList.remove(
    "high",
    "medium",
    "low"
  );

  if (
    ["high", "medium", "low"]
      .includes(riskCode)
  ) {
    assessmentCard.classList.add(
      riskCode
    );
  }

  riskBadge.textContent =
    riskCode.toUpperCase();

  warningCount.textContent =
    `${count} warning sign${
      count === 1 ? "" : "s"
    } detected`;

  warningCountBadge.textContent =
    `${count} detected`;

  assessmentHeadline.textContent =
    riskHeadline(riskCode, count);

  assessmentText.textContent =
    riskSummary(
      riskCode,
      count
    );

  const probability =
    Number(
      data?.fraud_probability || 0
    );

  const percentage =
    Math.max(
      0,
      Math.min(
        100,
        probability * 100
      )
    );

  fraudProbability.textContent =
    `${percentage.toFixed(2)}%`;

  const threshold =
    Number(data?.threshold);

  thresholdEl.textContent =
    Number.isFinite(threshold)
      ? `Threshold: ${
          (threshold * 100).toFixed(0)
        }%`
      : "Threshold: —";

  mlClassification.textContent =
    mlClassificationText(data);

  probabilityFill.style.width =
    `${percentage}%`;

  summary.textContent =
    data?.llm_report?.summary ||
    "No summary available.";

  riskExplanation.textContent =
    data?.llm_report
      ?.risk_explanation ||
    "No explanation available.";

  confidenceNote.textContent =
    data?.llm_report
      ?.confidence_note ||
    "SignalGuard provides a risk assessment based on available posting information. Results are not proof that a job is fraudulent or legitimate.";

  const sourceUrl =
    data?.source_url || "";

  sourceMeta.classList.toggle(
    "hidden",
    !sourceUrl
  );

  sourceMeta.textContent =
    sourceUrl
      ? `Source: ${getHost(sourceUrl)}`
      : "";

  renderWarningFlags(flags);

  renderShapSignals(
    data?.shap_signals
  );

  renderRag(data);

  renderRecommendations(data);

  renderPostingPreview(data);
}

function getHost(url) {
  try {
    return new URL(url)
      .hostname
      .replace(/^www\./, "");
  } catch {
    return "source";
  }
}

function getManualPayload() {
  return {
    title:
      document
        .getElementById("title")
        .value
        .trim(),

    company_profile:
      document
        .getElementById(
          "company_profile"
        )
        .value
        .trim(),

    description:
      document
        .getElementById(
          "description"
        )
        .value
        .trim(),

    requirements:
      document
        .getElementById(
          "requirements"
        )
        .value
        .trim(),

    benefits:
      document
        .getElementById(
          "benefits"
        )
        .value
        .trim(),
  };
}

jobForm.addEventListener(
  "submit",
  async (event) => {
    event.preventDefault();

    const payload =
      getManualPayload();

    if (!payload.title) {
      showFieldError(
        manualError,
        "Please enter a job title."
      );

      return;
    }

    if (!payload.description) {
      showFieldError(
        manualError,
        "Please enter a job description."
      );

      return;
    }

    lastAnalysisMode =
      "manual";

    lastAnalysisInput =
      payload;

    await analyzeJob(
      `${API_BASE_URL}/predict`,
      payload
    );
  }
);

analyzeUrlButton.addEventListener(
  "click",
  async () => {
    const url =
      jobUrl.value.trim();

    urlError.classList.add(
      "hidden"
    );

    urlError.textContent = "";

    if (!url) {
      showFieldError(
        urlError,
        "Please enter a job posting URL."
      );

      return;
    }

    try {
      const parsed =
        new URL(url);

      if (
        !["http:", "https:"]
          .includes(
            parsed.protocol
          )
      ) {
        throw new Error(
          "URL must start with http:// or https://"
        );
      }
    } catch (error) {
      showFieldError(
        urlError,
        error.message ||
          "Please enter a valid URL."
      );

      return;
    }

    lastAnalysisMode =
      "url";

    lastAnalysisInput = {
      url,
    };

    await analyzeJob(
      `${API_BASE_URL}/analyze-url`,
      { url }
    );
  }
);

urlTab.addEventListener(
  "click",
  () => setTab("url")
);

manualTab.addEventListener(
  "click",
  () => setTab("manual")
);

brandButton.addEventListener(
  "click",
  () => {
    clearInlineErrors();
    showView(analyzeView);
  }
);

backButton.addEventListener(
  "click",
  () => {
    showView(analyzeView);
  }
);

retryButton.addEventListener(
  "click",
  async () => {
    if (!lastAnalysisInput) {
      showView(analyzeView);
      return;
    }

    if (lastAnalysisMode === "url") {
      await analyzeJob(
        `${API_BASE_URL}/analyze-url`,
        lastAnalysisInput
      );
    } else {
      await analyzeJob(
        `${API_BASE_URL}/predict`,
        lastAnalysisInput
      );
    }
  }
);

setTab("url");