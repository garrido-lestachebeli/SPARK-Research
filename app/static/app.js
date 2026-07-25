const state = {
  config: null,
  environmentNames: {},
  agentSnapshots: [],
  leagueMembers: [],
  allLeagueMembers: [],
  leagueEnvironment: null,
  selectedLeagueMember: null,
  runs: [],
  selectedRun: null,
  selectedEpisode: null,
  episodes: [],
  episode: null,
  step: 0,
  playing: null,
  activeTab: "overview",
  draggedMember: null,
};

const $ = (selector) => document.querySelector(selector);
const json = (value) => JSON.stringify(value ?? {}, null, 2);

// Open OnDemand serves the app under /rnode/<host>/<port>/ or /node/<host>/<port>/.
// Absolute paths like /api/config would miss that prefix, so derive a base path.
function appBasePath() {
  let path = window.location.pathname || "/";
  if (!path.endsWith("/")) {
    const last = path.split("/").pop() || "";
    path = last.includes(".") ? path.replace(/\/[^/]+$/, "/") : `${path}/`;
  }
  return path;
}

const APP_BASE = appBasePath();
const ACCESS_TOKEN = new URLSearchParams(window.location.search).get("token") || "";

function apiUrl(path) {
  const normalized = path.startsWith("/") ? path.slice(1) : path;
  return new URL(normalized, window.location.origin + APP_BASE).toString();
}

async function api(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(ACCESS_TOKEN ? { "X-SPARK-Token": ACCESS_TOKEN } : {}),
    ...(options.headers || {}),
  };
  const response = await fetch(apiUrl(path), {
    credentials: "same-origin",
    ...options,
    headers,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg || JSON.stringify(item)).join("; ")
      : detail || `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return response.json();
}

function setTab(name) {
  state.activeTab = name;
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === name);
  });
  // Refresh league data on entry so it is current without periodic clobbering.
  if (name === "league") {
    loadLeagueMembers();
  }
}

async function initialize() {
  try {
    const config = await api("/api/config");
    state.config = config;
    state.environmentNames = Object.fromEntries(
      config.environments.map((environment) => [environment.id, environment.name]),
    );
    $("#connection").textContent = config.slurm_available
      ? "ROSIE / Slurm available"
      : config.local_jobs_enabled ? "Local job mode" : "Read-only mode";
    $("#connection").classList.add("online");
    renderConfig(config);
    setupLeagueEnvironmentSelect(config);
    await Promise.all([
      loadRuns(),
      loadCampaigns(),
      loadAgentSnapshots(),
      loadLeagueMembers(),
      loadAllLeagueMembers(),
    ]);
  } catch (error) {
    $("#connection").textContent = `Failed: ${error.message}`;
  }
}

async function loadAllLeagueMembers() {
  state.allLeagueMembers = await api("/api/league/members");
  renderStartingAgentOptions();
}

function renderConfig(config) {
  $("#environment-options").innerHTML = config.environments.map((environment) => `
    <label title="${environment.reason_disabled || ""}">
      <input type="checkbox" name="environment" value="${environment.id}"
        ${environment.enabled ? "" : "disabled"}>
      ${environment.name}${environment.enabled ? "" : " (coming later)"}
      <small>${config.agents.filter((agent) => agent.environment_id === environment.id).length} agent implementation(s)</small>
    </label>
  `).join("");
  document.querySelectorAll('input[name="environment"]').forEach((input) => {
    input.addEventListener("change", renderStartingAgentOptions);
  });
  renderStartingAgentOptions();
  if (config.agent_discovery_errors?.length) {
    $("#campaign-message").textContent =
      `${config.agent_discovery_errors.length} agent script(s) could not be loaded.`;
  }
}

async function loadRuns() {
  state.runs = await api("/api/runs");
  renderRunSelect("#run-select");
  renderRunSelect("#replay-run-select");
  if (state.runs.length) {
    const selectionExists = state.runs.some((run) => run.id === state.selectedRun);
    if (!selectionExists) {
      state.selectedRun = state.runs[0].id;
      state.selectedEpisode = null;
    }
    $("#run-select").value = state.selectedRun;
    $("#replay-run-select").value = state.selectedRun;
    updateRunEnvironmentLabels();
    await Promise.all([loadStatistics(), loadEpisodes()]);
  } else {
    state.selectedRun = null;
    updateRunEnvironmentLabels();
    renderEmptyStats();
  }
  renderAgentSourceRuns();
}

function renderAgentSourceRuns() {
  const select = $("#agent-source-run");
  if (!select) return;
  const previous = select.value;
  const options = state.runs.length
    ? state.runs.map((run) => {
      const campaignName = run.campaign_name || run.id;
      const environment = state.environmentNames[run.environment] || run.environment;
      return new Option(`${campaignName} · ${environment}`, run.id);
    })
    : [new Option("No runs with checkpoints", "")];
  select.replaceChildren(...options);
  if (state.runs.some((run) => run.id === previous)) {
    select.value = previous;
  }
}

function renderRunSelect(selector) {
  const select = $(selector);
  const options = state.runs.length
    ? state.runs.map((run) => {
      const campaignName = run.campaign_name || run.id;
      const environment = state.environmentNames[run.environment] || run.environment;
      return new Option(`${campaignName} · ${environment}`, run.id);
    })
    : [new Option("No runs yet", "")];
  select.replaceChildren(...options);
}

function updateRunEnvironmentLabels() {
  const run = state.runs.find((item) => item.id === state.selectedRun);
  const environment = run
    ? state.environmentNames[run.environment] || run.environment
    : "";
  $("#run-environment").textContent = environment
    ? `Environment: ${environment}`
    : "";
  $("#replay-run-environment").textContent = environment
    ? `Environment: ${environment}`
    : "";
}

async function loadStatistics() {
  if (!state.selectedRun) return;
  const stats = await api(`/api/runs/${encodeURIComponent(state.selectedRun)}/statistics`);
  const seatZero = stats.seat_payoffs?.["0"] || {};
  const recent = stats.recent_seat_payoffs?.["0"] || {};
  const winRate = stats.seat_outcomes?.["0"]?.win_rate || {};
  $("#stat-cards").innerHTML = [
    [
      `Recent payoff`,
      formatSigned(recent.mean),
      formatCi(recent.ci_low, recent.ci_high),
      "primary",
    ],
    [
      "Overall payoff",
      formatSigned(seatZero.mean),
      formatCi(seatZero.ci_low, seatZero.ci_high),
    ],
    [
      "Win rate",
      formatPercent(winRate.rate),
      formatCi(winRate.ci_low, winRate.ci_high, true),
    ],
    ["Episodes", formatNumber(stats.episodes), `agent version ${stats.latest_agent_version}`],
    ["Training pace", `${formatNumber(stats.episodes_per_second, 1)} ep/s`, `${formatNumber(stats.total_duration, 1)}s observed`],
  ].map(([label, value, detail, className = ""]) => `
    <article class="stat ${className}">
      <span>${label}</span>
      <strong>${value}</strong>
      <small>${detail || ""}</small>
    </article>
  `).join("");
  $("#trend-window").textContent =
    `Rolling window: up to ${stats.rolling_window || 100} episodes`;
  drawPayoffChart(stats.rolling || []);
  renderPayoffSummary(stats);
  renderActions(stats.actions || []);
  renderOutcomes(stats.seat_outcomes || stats.outcomes || {});
  renderEpisodeDiagnostics(stats);
}

function formatNumber(value, digits = 0) {
  const number = Number(value ?? 0);
  return Number.isFinite(number) ? number.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }) : "—";
}

function formatSigned(value, digits = 3) {
  const number = Number(value ?? 0);
  if (!Number.isFinite(number)) return "—";
  const text = number.toFixed(digits);
  return number > 0 ? `+${text}` : text;
}

function formatPercent(value) {
  const number = Number(value ?? 0);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
}

function formatCi(low, high, percent = false) {
  if (low == null || high == null || Number.isNaN(low) || Number.isNaN(high)) {
    return "CI unavailable";
  }
  if (percent) {
    return `95% CI ${(low * 100).toFixed(1)}% … ${(high * 100).toFixed(1)}%`;
  }
  return `95% CI ${formatSigned(low)} … ${formatSigned(high)}`;
}

function renderEmptyStats() {
  $("#stat-cards").innerHTML = '<article class="stat"><span>No training runs found</span><strong>—</strong></article>';
  $("#payoff-summary").innerHTML = "";
  $("#episode-diagnostics").innerHTML = "";
  $("#action-bars").innerHTML = "";
  $("#outcome-grid").innerHTML = "";
}

function drawPayoffChart(points) {
  const canvas = $("#payoff-chart");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#203954";
  context.lineWidth = 1;
  for (let index = 1; index < 5; index += 1) {
    const y = (height / 5) * index;
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
  }
  if (!points.length) return;
  const values = points.flatMap((point) => [
    point.seat_0_average_payoff,
    point.seat_0_cumulative_average ?? point.seat_0_average_payoff,
  ]);
  const minimum = Math.min(...values, -1);
  const maximum = Math.max(...values, 1);
  const yAt = (value) => height - ((value - minimum) / (maximum - minimum)) * height;
  const xAt = (index) => (points.length === 1 ? width / 2 : index / (points.length - 1) * width);

  context.strokeStyle = "#67a8ff";
  context.lineWidth = 2;
  context.beginPath();
  points.forEach((point, index) => {
    const y = yAt(point.seat_0_cumulative_average ?? point.seat_0_average_payoff);
    if (index === 0) context.moveTo(xAt(index), y); else context.lineTo(xAt(index), y);
  });
  context.stroke();

  context.strokeStyle = "#55d6be";
  context.lineWidth = 3;
  context.beginPath();
  points.forEach((point, index) => {
    const y = yAt(point.seat_0_average_payoff);
    if (index === 0) context.moveTo(xAt(index), y); else context.lineTo(xAt(index), y);
  });
  context.stroke();
}

function renderPayoffSummary(stats) {
  const seats = Object.keys(stats.seat_payoffs || stats.seat_average_payoffs || {});
  $("#payoff-summary").innerHTML = seats.map((seat) => {
    const overall = stats.seat_payoffs?.[seat] || {};
    const recent = stats.recent_seat_payoffs?.[seat] || {};
    return `
      <div class="metric-row">
        <div>
          <strong>Seat ${seat}</strong>
          <span>Overall mean ${formatSigned(overall.mean)}</span>
          <span>${formatCi(overall.ci_low, overall.ci_high)}</span>
        </div>
        <div>
          <strong>Recent mean ${formatSigned(recent.mean)}</strong>
          <span>${formatCi(recent.ci_low, recent.ci_high)}</span>
          <span>Based on ${formatNumber(recent.n)} episodes</span>
        </div>
        <details class="metric-details">
          <summary>Distribution details</summary>
          <span title="Standard deviation measures episode-to-episode spread.">Std dev ${formatNumber(overall.std, 3)}</span>
          <span title="Standard error measures uncertainty in the estimated mean.">Std error ${formatNumber(overall.stderr, 3)}</span>
          <span title="IQR is the middle 50% of observed payoffs.">Median ${formatSigned(overall.median)} · IQR ${formatSigned(overall.p25)} … ${formatSigned(overall.p75)}</span>
          <span>Range ${formatSigned(overall.min)} … ${formatSigned(overall.max)} · total ${formatSigned(overall.sum)}</span>
        </details>
      </div>
    `;
  }).join("") || "<p>No payoffs recorded.</p>";
}

function renderActions(actions) {
  const maximum = Math.max(...actions.map((item) => item.count), 1);
  $("#action-bars").innerHTML = actions.map((item) => `
    <div class="bar-row">
      <span>P${item.player_id} ${item.action}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${item.count / maximum * 100}%"></div>
      </div>
      <strong>${formatPercent(item.frequency ?? 0)}</strong>
    </div>
  `).join("") || "<p>No actions recorded.</p>";
}

function renderOutcomes(outcomes) {
  $("#outcome-grid").innerHTML = Object.entries(outcomes).map(([seat, data]) => {
    const counts = data.win != null && data.win_rate
      ? data
      : { win: data.win || 0, loss: data.loss || 0, draw: data.draw || 0, win_rate: {}, draw_rate: {} };
    const winRate = counts.win_rate || {};
    return `
      <div class="outcome-seat">
        <strong><span>Player ${seat}</span><span>${counts.win || 0} W · ${counts.loss || 0} L · ${counts.draw || 0} D</span></strong>
        <span>Win rate ${formatPercent(winRate.rate)} · ${formatCi(winRate.ci_low, winRate.ci_high, true)}</span>
        <details class="metric-details">
          <summary>More outcome rates</summary>
          <span>Draw rate ${formatPercent(counts.draw_rate?.rate)}</span>
          <span>Win rate excluding draws ${formatPercent(counts.win_rate_excluding_draws?.rate)}</span>
        </details>
      </div>
    `;
  }).join("") || "<p>No outcomes recorded.</p>";
}

function renderEpisodeDiagnostics(stats) {
  const length = stats.episode_length || {};
  const duration = stats.episode_duration || {};
  $("#episode-diagnostics").innerHTML = `
    <div class="metric-row">
      <div>
        <strong>Actions per episode</strong>
        <span>mean ${formatNumber(length.mean, 2)} · median ${formatNumber(length.median, 2)}</span>
        <span>${formatCi(length.ci_low, length.ci_high)}</span>
      </div>
      <div>
        <strong>Duration (seconds)</strong>
        <span>mean ${formatNumber(duration.mean, 3)} · std ${formatNumber(duration.std, 3)}</span>
        <span>${formatCi(duration.ci_low, duration.ci_high)}</span>
      </div>
    </div>
    <div class="metric-row">
      <div>
        <strong>Payoff variability</strong>
        <span>std dev ${formatNumber(stats.seat_payoffs?.["0"]?.std, 3)}</span>
        <span>std error ${formatNumber(stats.seat_payoffs?.["0"]?.stderr, 3)}</span>
      </div>
      <div>
        <strong>Throughput</strong>
        <span>${formatNumber(stats.episodes_per_second, 1)} episodes/second</span>
        <span>${formatNumber(stats.total_duration, 1)} seconds observed</span>
      </div>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[character]));
}

async function loadAgentSnapshots() {
  state.agentSnapshots = await api("/api/agent-snapshots");
  renderAgentLibrary();
  renderStartingAgentOptions();
  renderLeagueImportOptions();
}

function setupLeagueEnvironmentSelect(config) {
  const select = $("#league-environment");
  if (!select) return;
  const enabled = config.environments.filter((environment) => environment.enabled);
  select.innerHTML = enabled.map((environment) =>
    `<option value="${escapeHtml(environment.id)}">${escapeHtml(environment.name)}</option>`
  ).join("");
  if (!state.leagueEnvironment || !enabled.some((item) => item.id === state.leagueEnvironment)) {
    state.leagueEnvironment = enabled[0]?.id || null;
  }
  if (state.leagueEnvironment) {
    select.value = state.leagueEnvironment;
  }
  renderLeagueCreateOptions();
  renderLeagueImportOptions();
}

async function loadLeagueMembers() {
  if (!state.leagueEnvironment) {
    state.leagueMembers = [];
    renderLeagueLists();
    return;
  }
  const query = new URLSearchParams({ environment: state.leagueEnvironment });
  state.leagueMembers = await api(`/api/league/members?${query}`);
  renderLeagueCreateOptions();
  renderLeagueImportOptions();
  renderLeagueLists();
  if (state.selectedLeagueMember) {
    const stillPresent = state.leagueMembers.some(
      (member) => member.id === state.selectedLeagueMember,
    );
    if (stillPresent) {
      await showLeagueMemberDetails(state.selectedLeagueMember, { quiet: true });
    } else {
      clearLeagueDetails();
    }
  }
}

function renderLeagueCreateOptions() {
  const select = $("#league-create-agent");
  if (!select || !state.config) return;
  const agents = state.config.agents.filter(
    (agent) => agent.environment_id === state.leagueEnvironment,
  );
  select.innerHTML = agents.map((agent) =>
    `<option value="${escapeHtml(agent.id)}">${escapeHtml(agent.name)}${agent.supports_training ? "" : " (eval only)"}</option>`
  ).join("") || '<option value="">No implementations</option>';
}

function renderLeagueImportOptions() {
  const select = $("#league-import-snapshot");
  if (!select) return;
  const snapshots = state.agentSnapshots.filter(
    (snapshot) => snapshot.environment === state.leagueEnvironment,
  );
  select.innerHTML = snapshots.map((snapshot) => {
    const implementation = state.config?.agents.find(
      (agent) => agent.id === snapshot.agent_id,
    )?.name || snapshot.agent_id;
    return `<option value="${escapeHtml(snapshot.id)}">${escapeHtml(snapshot.name)} · ${escapeHtml(implementation)}</option>`;
  }).join("") || '<option value="">No snapshots for this environment</option>';
}

function renderLeagueLists() {
  const active = state.leagueMembers.filter((member) => member.status === "league");
  const bench = state.leagueMembers.filter((member) => member.status === "bench");
  const activeList = $("#league-active-list");
  const benchList = $("#league-bench-list");
  if ($("#league-count")) {
    $("#league-count").textContent = `${active.length} agent${active.length === 1 ? "" : "s"}`;
  }
  if ($("#bench-count")) {
    $("#bench-count").textContent = `${bench.length} agent${bench.length === 1 ? "" : "s"}`;
  }
  if (activeList) {
    activeList.innerHTML = active.map((member) => leagueMemberCard(member)).join("")
      || '<p class="help">No active league members yet. Drag agents here.</p>';
  }
  if (benchList) {
    benchList.innerHTML = bench.map((member) => leagueMemberCard(member)).join("")
      || '<p class="help">Bench is empty. Drag agents here to sideline them.</p>';
  }
  bindLeagueMemberActions();
  bindLeagueDropZones();
}

async function moveLeagueMember(memberId, status) {
  const member = state.leagueMembers.find((item) => item.id === memberId);
  if (!member || member.status === status) return;
  try {
    await api(`/api/league/members/${encodeURIComponent(memberId)}`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    });
    $("#league-message").textContent = `Moved ${member.name} to ${status}.`;
    await loadLeagueMembers();
  } catch (error) {
    $("#league-message").textContent = error.message;
  }
}

function bindLeagueDropZones() {
  const zones = [
    { el: $("#league-active-list"), status: "league" },
    { el: $("#league-bench-list"), status: "bench" },
  ];
  zones.forEach(({ el, status }) => {
    if (!el) return;
    el.addEventListener("dragover", (event) => {
      event.preventDefault();
      el.classList.add("drop-target");
    });
    el.addEventListener("dragleave", () => el.classList.remove("drop-target"));
    el.addEventListener("drop", (event) => {
      event.preventDefault();
      el.classList.remove("drop-target");
      const memberId = event.dataTransfer?.getData("text/plain") || state.draggedMember;
      if (memberId) moveLeagueMember(memberId, status);
    });
  });
}

function leagueMemberCard(member) {
  const stats = member.stats || {};
  const isTraining = member.mode === "train";
  const canTrain = member.supports_training !== false;
  const modeLabel = isTraining ? "Training" : "Snapshot";
  const toggleDisabled = canTrain ? "" : "disabled";
  const toggleTitle = canTrain
    ? "Toggle training (learns while playing) vs snapshot (frozen)"
    : "This implementation cannot train; it stays a frozen snapshot";
  return `
    <article class="league-member" draggable="true"
      data-member="${escapeHtml(member.id)}" data-status="${escapeHtml(member.status)}">
      <div class="league-member-header">
        <div>
          <strong><span class="drag-handle" aria-hidden="true">⠿</span> ${escapeHtml(member.name)}</strong>
          <span>${escapeHtml(member.agent_name || member.agent_id)} · ${escapeHtml(member.source_kind || "blank")}</span>
          <small>
            ${formatNumber(stats.matches)} matches ·
            ${formatNumber(stats.wins)}W /
            ${formatNumber(stats.losses)}L /
            ${formatNumber(stats.draws)}D
          </small>
        </div>
        <div class="league-elo">
          ${formatNumber(member.elo)}
          <small>ELO (placeholder)</small>
        </div>
      </div>
      <div class="league-member-controls">
        <label class="mode-switch ${isTraining ? "is-training" : ""}" title="${toggleTitle}">
          <input type="checkbox" class="league-mode-toggle" data-member="${escapeHtml(member.id)}"
            ${isTraining ? "checked" : ""} ${toggleDisabled}>
          <span class="mode-switch-track"><span class="mode-switch-thumb"></span></span>
          <span class="mode-switch-text">${modeLabel}</span>
        </label>
        <div class="league-member-actions">
          <button type="button" class="league-view" data-member="${escapeHtml(member.id)}">Details</button>
          <button type="button" class="league-copy" data-member="${escapeHtml(member.id)}" data-name="${escapeHtml(member.name)}">Copy</button>
          <button type="button" class="league-delete danger" data-member="${escapeHtml(member.id)}">Delete</button>
        </div>
      </div>
    </article>
  `;
}

function bindLeagueMemberActions() {
  document.querySelectorAll(".league-member").forEach((card) => {
    card.addEventListener("dragstart", (event) => {
      state.draggedMember = card.dataset.member;
      card.classList.add("dragging");
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", card.dataset.member);
      }
    });
    card.addEventListener("dragend", () => {
      state.draggedMember = null;
      card.classList.remove("dragging");
    });
  });
  document.querySelectorAll(".league-mode-toggle").forEach((toggle) => {
    toggle.addEventListener("change", async () => {
      const mode = toggle.checked ? "train" : "snapshot";
      try {
        await api(`/api/league/members/${encodeURIComponent(toggle.dataset.member)}`, {
          method: "PATCH",
          body: JSON.stringify({ mode }),
        });
        $("#league-message").textContent = `Set ${mode === "train" ? "training" : "snapshot"} mode.`;
        await loadLeagueMembers();
      } catch (error) {
        $("#league-message").textContent = error.message;
        await loadLeagueMembers();
      }
    });
  });
  document.querySelectorAll(".league-view").forEach((button) => {
    button.addEventListener("click", () => showLeagueMemberDetails(button.dataset.member));
  });
  document.querySelectorAll(".league-copy").forEach((button) => {
    button.addEventListener("click", async () => {
      const name = window.prompt("Name for the copied member", `${button.dataset.name}-copy`);
      if (!name) return;
      try {
        const result = await api(
          `/api/league/members/${encodeURIComponent(button.dataset.member)}/copy`,
          { method: "POST", body: JSON.stringify({ name }) },
        );
        $("#league-message").textContent = `Copied as ${result.name}.`;
        await loadLeagueMembers();
      } catch (error) {
        $("#league-message").textContent = error.message;
      }
    });
  });
  document.querySelectorAll(".league-delete").forEach((button) => {
    button.addEventListener("click", async () => {
      if (!window.confirm("Delete this league member? This does not delete Agent library snapshots.")) {
        return;
      }
      try {
        await api(`/api/league/members/${encodeURIComponent(button.dataset.member)}`, {
          method: "DELETE",
        });
        if (state.selectedLeagueMember === button.dataset.member) {
          clearLeagueDetails();
        }
        $("#league-message").textContent = "League member deleted.";
        await loadLeagueMembers();
      } catch (error) {
        $("#league-message").textContent = error.message;
      }
    });
  });
}

function clearLeagueDetails() {
  state.selectedLeagueMember = null;
  const detail = $("#league-detail");
  const empty = $("#league-detail-empty");
  if (detail) {
    detail.classList.add("hidden");
    detail.innerHTML = "";
  }
  if (empty) empty.classList.remove("hidden");
}

async function showLeagueMemberDetails(memberId, { quiet = false } = {}) {
  try {
    const detail = await api(`/api/league/members/${encodeURIComponent(memberId)}`);
    state.selectedLeagueMember = memberId;
    const panel = $("#league-detail");
    const empty = $("#league-detail-empty");
    if (empty) empty.classList.add("hidden");
    if (!panel) return;
    panel.classList.remove("hidden");
    const stats = detail.stats || {};
    panel.innerHTML = `
      <div class="league-member-header">
        <div>
          <strong>${escapeHtml(detail.name)}</strong>
          <span>${escapeHtml(detail.agent_name || detail.agent_id)} · ${escapeHtml(detail.environment)}</span>
          <small>Source: ${escapeHtml(detail.source_kind)} · Status: ${escapeHtml(detail.status)} · Mode: ${escapeHtml(detail.mode || "snapshot")}</small>
        </div>
        <div class="league-elo">
          ${formatNumber(detail.elo)}
          <small>ELO (placeholder)</small>
        </div>
      </div>
      <p class="help">${escapeHtml(detail.elo_note || "ELO is a placeholder.")}</p>
      <div class="league-stat-grid">
        <div class="league-stat"><span>Matches</span><strong>${formatNumber(stats.matches)}</strong></div>
        <div class="league-stat"><span>Wins</span><strong>${formatNumber(stats.wins)}</strong></div>
        <div class="league-stat"><span>Losses</span><strong>${formatNumber(stats.losses)}</strong></div>
        <div class="league-stat"><span>Draws</span><strong>${formatNumber(stats.draws)}</strong></div>
      </div>
      <div>
        <h4>Source</h4>
        <pre>${escapeHtml(json(detail.source || {}))}</pre>
      </div>
      <div>
        <h4>State preview</h4>
        <pre>${escapeHtml(json(detail.state_preview || {}))}</pre>
      </div>
    `;
    if (!quiet) {
      $("#league-message").textContent = `Showing details for ${detail.name}.`;
    }
  } catch (error) {
    if (!quiet) {
      $("#league-message").textContent = error.message;
    }
  }
}

async function createLeagueMember(event) {
  event.preventDefault();
  // currentTarget is cleared once the handler yields, so keep the element.
  const element = event.currentTarget;
  const form = new FormData(element);
  try {
    const result = await api("/api/league/members", {
      method: "POST",
      body: JSON.stringify({
        name: form.get("name"),
        environment: state.leagueEnvironment,
        agent_id: form.get("agent_id"),
        status: form.get("status"),
      }),
    });
    $("#league-message").textContent = `Created ${result.name}.`;
    element.reset();
    await loadLeagueMembers();
  } catch (error) {
    $("#league-message").textContent = error.message;
  }
}

async function importLeagueMember(event) {
  event.preventDefault();
  // currentTarget is cleared once the handler yields, so keep the element.
  const element = event.currentTarget;
  const form = new FormData(element);
  const snapshotId = form.get("snapshot_id");
  if (!snapshotId) {
    $("#league-message").textContent = "Choose an Agent library snapshot to import.";
    return;
  }
  try {
    const result = await api("/api/league/members/import-snapshot", {
      method: "POST",
      body: JSON.stringify({
        name: form.get("name"),
        snapshot_id: snapshotId,
        status: form.get("status"),
      }),
    });
    $("#league-message").textContent = `Imported ${result.name}.`;
    element.reset();
    await loadLeagueMembers();
  } catch (error) {
    $("#league-message").textContent = error.message;
  }
}

async function seedLeagueBuiltins() {
  if (!state.leagueEnvironment) return;
  try {
    const result = await api("/api/league/seed-builtins", {
      method: "POST",
      body: JSON.stringify({ environment: state.leagueEnvironment }),
    });
    $("#league-message").textContent = result.count
      ? `Added ${result.count} built-in opponent${result.count === 1 ? "" : "s"}.`
      : "Built-in opponents already present for this environment.";
    await Promise.all([loadLeagueMembers(), loadAllLeagueMembers()]);
  } catch (error) {
    $("#league-message").textContent = error.message;
  }
}

function renderAgentLibrary() {
  const list = $("#agent-library-list");
  if (!list) return;
  list.innerHTML = state.agentSnapshots.map((snapshot) => {
    const environment =
      state.environmentNames[snapshot.environment] || snapshot.environment;
    const implementation = state.config?.agents.find(
      (agent) => agent.id === snapshot.agent_id,
    )?.name || snapshot.agent_id;
    return `
      <article class="agent-snapshot">
        <div>
          <strong>${escapeHtml(snapshot.name)}</strong>
          <span>${escapeHtml(environment)} · ${escapeHtml(implementation)}</span>
          <small>Source ${escapeHtml(snapshot.source?.run_id)} seat ${formatNumber(snapshot.source?.seat)} · agent version ${formatNumber(snapshot.source?.agent_version)}</small>
        </div>
        <button class="delete-agent-snapshot danger" data-snapshot="${snapshot.id}">Delete</button>
      </article>
    `;
  }).join("") || '<p class="help">No saved agents yet.</p>';

  document.querySelectorAll(".delete-agent-snapshot").forEach((button) => {
    button.addEventListener("click", async () => {
      if (!window.confirm("Delete this saved agent snapshot? Source run data will be kept.")) {
        return;
      }
      try {
        await api(`/api/agent-snapshots/${encodeURIComponent(button.dataset.snapshot)}`, {
          method: "DELETE",
        });
        $("#agent-library-message").textContent = "Saved agent deleted.";
        await loadAgentSnapshots();
      } catch (error) {
        $("#agent-library-message").textContent = error.message;
      }
    });
  });
}

function activeLeagueMembers(environmentId) {
  return (state.allLeagueMembers || []).filter(
    (member) => member.environment === environmentId && member.status === "league",
  );
}

function defaultSeatSource(environmentId, seatIndex) {
  const league = activeLeagueMembers(environmentId);
  if (seatIndex === 0) return "fresh";
  return league.length ? "league_random" : "fresh";
}

function renderStartingAgentOptions() {
  const container = $("#starting-agent-options");
  if (!container || !state.config) return;
  const previous = {};
  container.querySelectorAll(".table-configuration").forEach((table) => {
    const environment = table.dataset.environment;
    previous[environment] = {
      tableSize: Number(table.querySelector(".table-size")?.value || 2),
      seats: [...table.querySelectorAll(".seat-configuration")].map((seat) => ({
        source: seat.querySelector(".seat-source")?.value || "fresh",
        agentId: seat.querySelector(".seat-agent")?.value,
        mode: seat.querySelector(".seat-mode")?.value,
        snapshotId: seat.querySelector(".seat-snapshot")?.value,
        leagueMemberId: seat.querySelector(".seat-league-member")?.value,
      })),
    };
  });
  const selectedEnvironments = [
    ...document.querySelectorAll('input[name="environment"]:checked'),
  ].map((input) => input.value);
  container.innerHTML = selectedEnvironments.map((environmentId) => {
    const environment = state.config.environments.find(
      (item) => item.id === environmentId,
    );
    const agents = state.config.agents.filter(
      (agent) => agent.environment_id === environmentId,
    );
    const defaultAgent = agents.find((agent) => agent.default) || agents[0];
    const trainable = agents.find((agent) => agent.supports_training) || defaultAgent;
    const league = activeLeagueMembers(environmentId);
    const old = previous[environmentId] || {};
    const tableSize = Math.max(
      environment.min_players,
      Math.min(old.tableSize || environment.num_players, environment.max_players),
    );
    const seats = Array.from({ length: tableSize }, (_, seatIndex) => {
      const oldSeat = old.seats?.[seatIndex] || {};
      const source = oldSeat.source || defaultSeatSource(environmentId, seatIndex);
      const selectedAgent = agents.some((agent) => agent.id === oldSeat.agentId)
        ? oldSeat.agentId
        : (seatIndex === 0 ? trainable?.id : defaultAgent?.id);
      const agentOptions = agents.map((agent) =>
        `<option value="${escapeHtml(agent.id)}" ${agent.id === selectedAgent ? "selected" : ""}>${escapeHtml(agent.name)}</option>`
      ).join("");
      const snapshots = state.agentSnapshots.filter(
        (snapshot) =>
          snapshot.environment === environmentId
          && snapshot.agent_id === selectedAgent,
      );
      const snapshotOptions = [
        '<option value="">Fresh weights</option>',
        ...snapshots.map((snapshot) =>
          `<option value="${snapshot.id}" ${snapshot.id === oldSeat.snapshotId ? "selected" : ""}>${escapeHtml(snapshot.name)}</option>`),
      ].join("");
      const leagueOptions = league.length
        ? league.map((member) =>
          `<option value="${escapeHtml(member.id)}" ${member.id === oldSeat.leagueMemberId ? "selected" : ""}>${escapeHtml(member.name)} · ELO ${formatNumber(member.elo)} · ${escapeHtml(member.mode)}</option>`
        ).join("")
        : '<option value="">No active League members</option>';
      return `
        <div class="seat-configuration" data-seat="${seatIndex}" data-source="${source}">
          <strong>Seat ${seatIndex}</strong>
          <label>Source
            <select class="seat-source">
              <option value="fresh" ${source === "fresh" ? "selected" : ""}>Fresh implementation</option>
              <option value="league_member" ${source === "league_member" ? "selected" : ""}>League member</option>
              <option value="league_random" ${source === "league_random" ? "selected" : ""}>League (random)</option>
            </select>
          </label>
          <div class="seat-fresh-fields ${source === "fresh" ? "" : "hidden"}">
            <label>Implementation<select class="seat-agent">${agentOptions}</select></label>
            <label>Mode<select class="seat-mode">
              <option value="train" ${oldSeat.mode !== "evaluate" ? "selected" : ""}>Train</option>
              <option value="evaluate" ${oldSeat.mode === "evaluate" ? "selected" : ""}>Evaluate (frozen)</option>
            </select></label>
            <details class="seat-advanced">
              <summary>Advanced warm-start</summary>
              <label>Library snapshot<select class="seat-snapshot">${snapshotOptions}</select></label>
            </details>
          </div>
          <div class="seat-league-member-fields ${source === "league_member" ? "" : "hidden"}">
            <label>Member<select class="seat-league-member">${leagueOptions}</select></label>
            <p class="help">Uses that member's train/snapshot mode.</p>
          </div>
          <div class="seat-league-random-fields ${source === "league_random" ? "" : "hidden"}">
            <p class="help">Drawn from ${league.length} active League member(s). Re-rolls every K episodes (K=0 keeps one).</p>
          </div>
        </div>
      `;
    }).join("");
    return `
      <fieldset class="table-configuration" data-environment="${environmentId}">
        <legend>${escapeHtml(environment.name)} table</legend>
        <label>Table size
          <input class="table-size" type="number"
            min="${environment.min_players}" max="${environment.max_players}"
            value="${tableSize}" ${environment.min_players === environment.max_players ? "disabled" : ""}>
        </label>
        <div class="seat-configurations">${seats}</div>
      </fieldset>
    `;
  }).join("");
  container.querySelectorAll(".table-size, .seat-agent, .seat-source").forEach((control) => {
    control.addEventListener("change", (event) => {
      if (event.currentTarget.classList.contains("seat-source")) {
        const seat = event.currentTarget.closest(".seat-configuration");
        const source = event.currentTarget.value;
        seat.dataset.source = source;
        seat.querySelector(".seat-fresh-fields")?.classList.toggle("hidden", source !== "fresh");
        seat.querySelector(".seat-league-member-fields")?.classList.toggle("hidden", source !== "league_member");
        seat.querySelector(".seat-league-random-fields")?.classList.toggle("hidden", source !== "league_random");
        return;
      }
      renderStartingAgentOptions();
    });
  });
}

function applyCampaignPreset(preset) {
  document.querySelectorAll(".table-configuration").forEach((table) => {
    const environmentId = table.dataset.environment;
    const agents = state.config.agents.filter(
      (agent) => agent.environment_id === environmentId,
    );
    const trainable = agents.find((agent) => agent.supports_training) || agents[0];
    const baseline = agents.find((agent) => agent.default) || agents[0];
    const league = activeLeagueMembers(environmentId);
    table.querySelectorAll(".seat-configuration").forEach((seat, seatIndex) => {
      const sourceSelect = seat.querySelector(".seat-source");
      const agentSelect = seat.querySelector(".seat-agent");
      const modeSelect = seat.querySelector(".seat-mode");
      let source = "fresh";
      if (preset === "learner-vs-league") {
        source = seatIndex === 0 ? "fresh" : (league.length ? "league_random" : "fresh");
      } else if (preset === "all-league") {
        source = league.length ? "league_random" : "fresh";
      } else {
        source = "fresh";
      }
      if (sourceSelect) sourceSelect.value = source;
      if (agentSelect) {
        agentSelect.value = seatIndex === 0 ? (trainable?.id || "") : (baseline?.id || "");
      }
      if (modeSelect) {
        modeSelect.value = seatIndex === 0 && preset !== "all-league" ? "train" : "evaluate";
      }
      seat.dataset.source = source;
      seat.querySelector(".seat-fresh-fields")?.classList.toggle("hidden", source !== "fresh");
      seat.querySelector(".seat-league-member-fields")?.classList.toggle("hidden", source !== "league_member");
      seat.querySelector(".seat-league-random-fields")?.classList.toggle("hidden", source !== "league_random");
    });
  });
  $("#campaign-message").textContent = `Applied preset: ${preset}.`;
}

async function saveAgentSnapshot(event) {
  event.preventDefault();
  // currentTarget is cleared once the handler yields, so keep the element.
  const element = event.currentTarget;
  const form = new FormData(element);
  try {
    const result = await api("/api/agent-snapshots", {
      method: "POST",
      body: JSON.stringify({
        name: form.get("name"),
        run_id: form.get("run_id"),
        seat: Number(form.get("seat")),
      }),
    });
    $("#agent-library-message").textContent = `Saved ${result.name}.`;
    element.reset();
    await loadAgentSnapshots();
  } catch (error) {
    $("#agent-library-message").textContent = error.message;
  }
}

async function loadCampaigns() {
  const values = await api("/api/campaigns");
  $("#campaign-list").innerHTML = values.map((campaign) => `
    <article class="campaign">
      <p class="eyebrow">${campaign.id}</p><h3>${campaign.name}</h3>
      <button class="stop-campaign danger" data-campaign="${campaign.id}">Stop all</button>
      <button class="resume-campaign" data-campaign="${campaign.id}">Resume all</button>
      ${campaign.jobs.map((job) => `
        <div class="job">
          <div><strong>${job.environment}</strong>
            <small>${job.run_id} · ${job.scheduler_id || "not submitted"}</small>
            <small>${(campaign.config?.tables?.[job.environment]?.seats || [])
              .map((seat, index) => {
                const source = seat.source || "fresh";
                if (source === "league_random") {
                  return `Seat ${index}: League random → ${seat.agent_id}`;
                }
                if (source === "league_member") {
                  return `Seat ${index}: League member (${seat.agent_id})`;
                }
                return `Seat ${index}: ${seat.agent_id} (${seat.mode})`;
              })
              .join(" · ")}</small></div>
          <span>${job.run?.status || job.status}</span>
          <div><button class="danger stop-job" data-run="${job.run_id}">Stop</button>
          <button class="resume-job" data-run="${job.run_id}">Resume</button>
          <button class="remove-job" data-run="${job.run_id}">Remove</button></div>
        </div>
        <details><summary>Worker logs</summary>
          <button class="load-logs" data-run="${job.run_id}">Load latest</button>
          <pre id="logs-${job.run_id}">Not loaded.</pre>
        </details>`).join("")}
    </article>
  `).join("") || '<p class="help">No campaigns submitted yet.</p>';
  document.querySelectorAll(".stop-job").forEach((button) => button.addEventListener("click", async () => {
    await api(`/api/jobs/${encodeURIComponent(button.dataset.run)}/stop`, { method: "POST" });
    await loadCampaigns();
  }));
  document.querySelectorAll(".resume-job").forEach((button) => button.addEventListener("click", async () => {
    await api(`/api/jobs/${encodeURIComponent(button.dataset.run)}/resume`, { method: "POST" });
    await loadCampaigns();
  }));
  document.querySelectorAll(".remove-job").forEach((button) => button.addEventListener("click", async () => {
    const confirmed = window.confirm(
      "Remove this job from campaign controls? Its run data and replays will be kept.",
    );
    if (!confirmed) return;
    await api(`/api/jobs/${encodeURIComponent(button.dataset.run)}`, { method: "DELETE" });
    await Promise.all([loadCampaigns(), loadRuns()]);
  }));
  document.querySelectorAll(".stop-campaign").forEach((button) => button.addEventListener("click", async () => {
    await api(`/api/campaigns/${encodeURIComponent(button.dataset.campaign)}/stop`, { method: "POST" });
    await loadCampaigns();
  }));
  document.querySelectorAll(".resume-campaign").forEach((button) => button.addEventListener("click", async () => {
    await api(`/api/campaigns/${encodeURIComponent(button.dataset.campaign)}/resume`, { method: "POST" });
    await loadCampaigns();
  }));
  document.querySelectorAll(".load-logs").forEach((button) => button.addEventListener("click", async () => {
    const result = await api(`/api/jobs/${encodeURIComponent(button.dataset.run)}/logs`);
    document.getElementById(`logs-${button.dataset.run}`).textContent = result.logs;
  }));
}

async function submitCampaign(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const environments = [...document.querySelectorAll('input[name="environment"]:checked')].map((item) => item.value);
  const tables = Object.fromEntries(
    [...document.querySelectorAll(".table-configuration")].map((table) => {
      const seats = [...table.querySelectorAll(".seat-configuration")].map((seat) => {
        const source = seat.querySelector(".seat-source")?.value || "fresh";
        if (source === "league_random") {
          return { source: "league_random" };
        }
        if (source === "league_member") {
          return {
            source: "league_member",
            league_member_id: seat.querySelector(".seat-league-member")?.value || null,
          };
        }
        return {
          source: "fresh",
          agent_id: seat.querySelector(".seat-agent")?.value,
          mode: seat.querySelector(".seat-mode")?.value || "train",
          snapshot_id: seat.querySelector(".seat-snapshot")?.value || null,
        };
      });
      return [
        table.dataset.environment,
        {
          table_size: Number(table.querySelector(".table-size").value),
          seats,
        },
      ];
    }),
  );
  const optionalNumber = (name) => form.get(name) ? Number(form.get(name)) : null;
  const payload = {
    name: form.get("name"),
    environments,
    tables,
    seed: Number(form.get("seed")),
    episodes: optionalNumber("episodes"),
    checkpoint_interval: Number(form.get("checkpoint_interval")),
    update_interval: Number(form.get("update_interval") || 1),
    league_reroll_interval: Number(form.get("league_reroll_interval") || 0),
    agent_memory_size: Number(form.get("agent_memory_size")),
    cpus: Number(form.get("cpus")),
    memory: form.get("memory"),
    wall_time: form.get("wall_time"),
    partition: form.get("partition") || null,
  };
  try {
    const result = await api("/api/campaigns", { method: "POST", body: JSON.stringify(payload) });
    $("#campaign-message").textContent = `Submitted ${result.jobs.length} job(s) in ${result.campaign_id}.`;
    await loadCampaigns();
  } catch (error) {
    $("#campaign-message").textContent = error.message;
  }
}

async function loadEpisodes() {
  if (!state.selectedRun) return;
  state.episodes = await api(`/api/runs/${encodeURIComponent(state.selectedRun)}/episodes?limit=200`);
  $("#episode-select").innerHTML = state.episodes.map((episode) =>
    `<option value="${episode.id}">#${episode.episode_number} · ${episode.outcomes.join(" / ")} · ${episode.step_count} actions</option>`
  ).join("") || '<option value="">No episodes</option>';
  if (!state.episodes.length) {
    state.selectedEpisode = null;
    state.episode = null;
    return;
  }
  const stillAvailable = state.episodes.some(
    (episode) => String(episode.id) === String(state.selectedEpisode),
  );
  const episodeId = stillAvailable ? state.selectedEpisode : state.episodes[0].id;
  $("#episode-select").value = String(episodeId);
  await loadEpisode(episodeId, { resetStep: !stillAvailable });
}

async function loadEpisode(episodeId, { resetStep = true } = {}) {
  if (!episodeId) return;
  const sameEpisode = String(state.selectedEpisode) === String(episodeId) && state.episode;
  state.selectedEpisode = episodeId;
  state.episode = await api(`/api/runs/${encodeURIComponent(state.selectedRun)}/episodes/${episodeId}`);
  if (resetStep || !sameEpisode) {
    state.step = 0;
  } else {
    state.step = Math.max(0, Math.min(state.step, state.episode.steps.length));
  }
  $("#timeline").max = state.episode.steps.length;
  $("#timeline").value = state.step;
  renderReplay();
}

function replayFrame() {
  if (!state.episode?.steps.length) return null;
  if (state.step === 0) {
    const first = state.episode.steps[0];
    return { visible: first.visible_before, hidden: first.hidden_before, action: null, raw: first };
  }
  const step = state.episode.steps[state.step - 1];
  return { visible: step.visible_after, hidden: step.hidden_after, action: step, raw: step };
}

function renderReplay() {
  const episode = state.episode;
  const frame = replayFrame();
  if (!episode || !frame) return;
  $("#timeline").value = state.step;
  $("#timeline-label").textContent = `${state.step} / ${episode.steps.length}`;
  $("#visible-json").textContent = json(frame.visible);
  $("#hidden-json").textContent = json(frame.hidden);
  $("#raw-json").textContent = json(frame.raw);
  $("#action-banner").textContent = frame.action
    ? `Player ${frame.action.player_id} → ${frame.action.action_name.toUpperCase()} · legal: ${frame.action.legal_actions.join(", ")}`
    : `Initial deal · seed ${episode.seed}`;

  const hidden = frame.hidden || {};
  const hands = hidden.hand_cards || hidden.hands || [];
  const playerCount = Math.max(
    hands.length,
    episode.payoffs?.length || 0,
    frame.raw?.player_id + 1 || 0,
  );
  $("#replay-players").innerHTML = Array.from(
    { length: playerCount },
    (_, playerId) => `
      <div class="player-card ${playerId === frame.raw?.player_id ? "active" : ""}">
        <span>Player ${playerId} hidden hand</span>
        <strong>${escapeHtml(formatCards(hands[playerId]))}</strong>
      </div>
    `,
  ).join("");
  $("#public-cards").textContent = formatCards(hidden.public_cards || hidden.public_card);
  const chips = hidden.chips || frame.visible?.raw?.all_chips || [];
  $("#pot-state").textContent = chips.length ? `Committed chips: ${chips.join(" / ")}` : "";
  if (state.step === episode.steps.length) {
    $("#action-banner").textContent += ` · Final payoffs: ${episode.payoffs.join(" / ")}`;
  }
}

function formatCards(value) {
  if (!value || (Array.isArray(value) && !value.length)) return "—";
  return Array.isArray(value) ? value.join("  ") : String(value);
}

function moveStep(value) {
  if (!state.episode) return;
  state.step = Math.max(0, Math.min(value, state.episode.steps.length));
  renderReplay();
}

document.querySelectorAll(".tabs button").forEach((button) =>
  button.addEventListener("click", () => setTab(button.dataset.tab)));
$("#run-select").addEventListener("change", async (event) => {
  state.selectedRun = event.target.value;
  state.selectedEpisode = null;
  $("#replay-run-select").value = state.selectedRun;
  updateRunEnvironmentLabels();
  await Promise.all([loadStatistics(), loadEpisodes()]);
});
$("#replay-run-select").addEventListener("change", async (event) => {
  state.selectedRun = event.target.value;
  state.selectedEpisode = null;
  $("#run-select").value = state.selectedRun;
  updateRunEnvironmentLabels();
  await Promise.all([loadStatistics(), loadEpisodes()]);
});
$("#episode-select").addEventListener("change", (event) => {
  loadEpisode(event.target.value, { resetStep: true });
});
$("#campaign-form").addEventListener("submit", submitCampaign);
$("#save-agent-form").addEventListener("submit", saveAgentSnapshot);
$("#refresh-jobs").addEventListener("click", loadCampaigns);
document.querySelectorAll(".campaign-preset").forEach((button) => {
  button.addEventListener("click", () => applyCampaignPreset(button.dataset.preset));
});
$("#league-environment")?.addEventListener("change", async (event) => {
  state.leagueEnvironment = event.target.value;
  clearLeagueDetails();
  await loadLeagueMembers();
});
$("#league-refresh")?.addEventListener("click", loadLeagueMembers);
$("#league-seed-builtins")?.addEventListener("click", seedLeagueBuiltins);
$("#league-create-form")?.addEventListener("submit", createLeagueMember);
$("#league-import-form")?.addEventListener("submit", importLeagueMember);
$("#timeline").addEventListener("input", (event) => moveStep(Number(event.target.value)));
$("#step-first").addEventListener("click", () => moveStep(0));
$("#step-prev").addEventListener("click", () => moveStep(state.step - 1));
$("#step-next").addEventListener("click", () => moveStep(state.step + 1));
$("#step-last").addEventListener("click", () => moveStep(state.episode?.steps.length || 0));
$("#step-play").addEventListener("click", () => {
  if (state.playing) {
    clearInterval(state.playing); state.playing = null; $("#step-play").textContent = "Play"; return;
  }
  $("#step-play").textContent = "Pause";
  state.playing = setInterval(() => {
    if (!state.episode || state.step >= state.episode.steps.length) {
      clearInterval(state.playing); state.playing = null; $("#step-play").textContent = "Play"; return;
    }
    moveStep(state.step + 1);
  }, 800);
});

initialize();
setInterval(() => {
  loadRuns();
  loadCampaigns();
  // Skip league refresh while its tab is open so selections and drags are not lost.
  if (state.activeTab !== "league") {
    loadAgentSnapshots();
    loadLeagueMembers();
    loadAllLeagueMembers();
  }
}, 15000);
