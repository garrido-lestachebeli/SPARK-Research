const state = {
  config: null,
  environmentNames: {},
  agentSnapshots: [],
  runs: [],
  selectedRun: null,
  selectedEpisode: null,
  episodes: [],
  episode: null,
  step: 0,
  playing: null,
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
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === name);
  });
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
    await Promise.all([loadRuns(), loadCampaigns(), loadAgentSnapshots()]);
  } catch (error) {
    $("#connection").textContent = `Failed: ${error.message}`;
  }
}

function renderConfig(config) {
  $("#environment-options").innerHTML = config.environments.map((environment) => `
    <label title="${environment.reason_disabled || ""}">
      <input type="checkbox" name="environment" value="${environment.id}"
        ${environment.enabled ? "" : "disabled"}>
      ${environment.name}${environment.enabled ? "" : " (coming later)"}
      <small>${config.agents?.[environment.id] || "No agent"}</small>
    </label>
  `).join("");
  document.querySelectorAll('input[name="environment"]').forEach((input) => {
    input.addEventListener("change", renderStartingAgentOptions);
  });
  renderStartingAgentOptions();
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
}

function renderAgentLibrary() {
  const list = $("#agent-library-list");
  if (!list) return;
  list.innerHTML = state.agentSnapshots.map((snapshot) => {
    const environment =
      state.environmentNames[snapshot.environment] || snapshot.environment;
    const topology = snapshot.shared_agents ? "shared seats" : "separate seats";
    return `
      <article class="agent-snapshot">
        <div>
          <strong>${escapeHtml(snapshot.name)}</strong>
          <span>${escapeHtml(environment)} · ${topology}</span>
          <small>Source ${escapeHtml(snapshot.source?.run_id)} · agent version ${formatNumber(snapshot.source?.agent_version)}</small>
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

function renderStartingAgentOptions() {
  const container = $("#starting-agent-options");
  if (!container) return;
  const previous = Object.fromEntries(
    [...container.querySelectorAll(".agent-snapshot-select")]
      .map((select) => [select.dataset.environment, select.value]),
  );
  const selectedEnvironments = [
    ...document.querySelectorAll('input[name="environment"]:checked'),
  ].map((input) => input.value);
  container.innerHTML = selectedEnvironments.map((environmentId) => {
    const environment =
      state.environmentNames[environmentId] || environmentId;
    const compatible = state.agentSnapshots.filter(
      (snapshot) => snapshot.environment === environmentId,
    );
    const options = [
      '<option value="">Fresh random agent</option>',
      ...compatible.map((snapshot) =>
        `<option value="${snapshot.id}">${escapeHtml(snapshot.name)}</option>`),
    ].join("");
    return `
      <label>
        Starting agent — ${escapeHtml(environment)}
        <select class="agent-snapshot-select" data-environment="${environmentId}">
          ${options}
        </select>
      </label>
    `;
  }).join("");
  container.querySelectorAll(".agent-snapshot-select").forEach((select) => {
    const oldValue = previous[select.dataset.environment];
    if ([...select.options].some((option) => option.value === oldValue)) {
      select.value = oldValue;
    }
  });
}

async function saveAgentSnapshot(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  try {
    const result = await api("/api/agent-snapshots", {
      method: "POST",
      body: JSON.stringify({
        name: form.get("name"),
        run_id: form.get("run_id"),
      }),
    });
    $("#agent-library-message").textContent = `Saved ${result.name}.`;
    event.currentTarget.reset();
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
            <small>${job.run_id} · ${job.scheduler_id || "not submitted"}</small></div>
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
  const agentSnapshots = Object.fromEntries(
    [...document.querySelectorAll(".agent-snapshot-select")]
      .filter((select) => select.value)
      .map((select) => [select.dataset.environment, select.value]),
  );
  const optionalNumber = (name) => form.get(name) ? Number(form.get(name)) : null;
  const payload = {
    name: form.get("name"),
    environments,
    agent_snapshots: agentSnapshots,
    seed: Number(form.get("seed")),
    episodes: optionalNumber("episodes"),
    checkpoint_interval: Number(form.get("checkpoint_interval")),
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
  $("#player-zero-hand").textContent = formatCards(hands[0]);
  $("#player-one-hand").textContent = formatCards(hands[1]);
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
setInterval(() => { loadRuns(); loadCampaigns(); loadAgentSnapshots(); }, 15000);
