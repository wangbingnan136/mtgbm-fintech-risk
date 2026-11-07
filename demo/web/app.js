const MODEL_URL = "model.json";

const state = {
  model: null,
};

const $ = (selector) => document.querySelector(selector);

function sigmoid(value) {
  return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));
}

function scoreNode(node, values) {
  while (Object.prototype.hasOwnProperty.call(node, "f")) {
    node = values[node.f] <= node.t ? node.l : node.r;
  }
  return node.v;
}

function scoreTask(task, values) {
  return task.trees.reduce((total, tree) => total + scoreNode(tree, values), 0);
}

function formatValue(feature, value) {
  if (feature.step < 1) {
    return Number(value).toFixed(2);
  }
  const unit = feature.unit ? ` ${feature.unit}` : "";
  return `${Number(value).toLocaleString("en-US")}${unit}`;
}

function renderFields() {
  const grid = $("#field-grid");
  grid.innerHTML = "";
  for (const feature of state.model.features) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";
    wrapper.innerHTML = `
      <div class="field-top">
        <label for="${feature.id}">${feature.label}</label>
        <output for="${feature.id}" id="${feature.id}-out"></output>
      </div>
      <input
        id="${feature.id}"
        name="${feature.id}"
        type="range"
        min="${feature.min}"
        max="${feature.max}"
        step="${feature.step}"
        value="${feature.value}"
      >
      <div class="field-hint">${feature.hint}</div>
    `;
    grid.appendChild(wrapper);
  }
  grid.addEventListener("input", update);
}

function readValues() {
  return state.model.features.map((feature) =>
    Number(document.getElementById(feature.id).value)
  );
}

function setValues(values) {
  for (let index = 0; index < state.model.features.length; index += 1) {
    const feature = state.model.features[index];
    document.getElementById(feature.id).value = values[index];
  }
  update();
}

function describeTask(taskName) {
  if (taskName === "card_fraud") {
    return "Card-not-present payment fraud";
  }
  if (taskName === "account_takeover") {
    return "Account-takeover risk";
  }
  return taskName.replaceAll("_", " ");
}

function renderScores(tasks) {
  const container = $("#task-scores");
  container.innerHTML = "";
  for (const task of tasks) {
    const card = document.createElement("div");
    card.className = "task-score";
    card.innerHTML = `
      <div class="task-score-top">
        <div>
          <h3>${describeTask(task.name)}</h3>
          <small>${task.tree_count} exported trees</small>
        </div>
        <div class="score-number">${(task.probability * 100).toFixed(1)}%</div>
      </div>
      <div class="score-track">
        <div class="score-fill" style="width:${(task.probability * 100).toFixed(1)}%"></div>
      </div>
    `;
    container.appendChild(card);
  }
}

function renderFlags(values) {
  const features = state.model.features;
  const flags = [];
  const at = (id) => values[features.findIndex((feature) => feature.id === id)];

  if (at("amount_usd") >= 1500) flags.push(["Large amount", true]);
  if (at("tx_velocity_1h") >= 7) flags.push(["High velocity", true]);
  if (at("failed_logins_24h") >= 3) flags.push(["Failed-login burst", true]);
  if (at("geo_mismatch") === 1) flags.push(["Geo mismatch", true]);
  if (at("merchant_risk") >= 0.7) flags.push(["High-risk merchant", true]);
  if (at("device_trust") <= 0.35) flags.push(["Low device trust", true]);
  if (at("card_present") === 0) flags.push(["Card not present", false]);
  if (at("device_trust") >= 0.75) flags.push(["Trusted device", false]);

  const container = $("#flag-list");
  container.innerHTML = flags.length
    ? flags
        .slice(0, 6)
        .map(
          ([label, hot]) =>
            `<span class="flag ${hot ? "hot" : ""}">${label}</span>`
        )
        .join("")
    : '<span class="flag">No strong context flags</span>';
}

function renderMetrics() {
  const rows = state.model.tasks
    .map(
      (task) => `
        <tr>
          <td>${describeTask(task.name)}</td>
          <td>${task.auc.toFixed(4)}</td>
          <td>${task.single_task.auc.toFixed(4)}</td>
        </tr>
      `
    )
    .join("");

  $("#metrics").innerHTML = `
    <table class="metrics-table">
      <thead>
        <tr>
          <th>Task</th>
          <th>MT-GBM AUC</th>
          <th>Single-task AUC</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
  $("#sample-note").textContent =
    `${state.model.training.test_rows.toLocaleString("en-US")} test rows`;
}

function update() {
  if (!state.model) return;
  const values = readValues();

  for (const feature of state.model.features) {
    document.getElementById(`${feature.id}-out`).textContent = formatValue(
      feature,
      document.getElementById(feature.id).value
    );
  }

  const scored = state.model.tasks.map((task) => ({
    ...task,
    raw: scoreTask(task, values),
  }));
  for (const task of scored) {
    task.probability = sigmoid(task.raw);
  }

  renderScores(scored);
  renderFlags(values);

  const mainRisk = scored.find((task) => task.name === "card_fraud").probability;
  const chip = $("#decision-chip");
  chip.className = "decision-chip";
  if (mainRisk >= 0.72) {
    chip.textContent = "Block / step-up";
    chip.classList.add("block");
  } else if (mainRisk >= 0.45) {
    chip.textContent = "Manual review";
    chip.classList.add("review");
  } else {
    chip.textContent = "Low risk";
    chip.classList.add("pass");
  }
}

function loadExample(kind) {
  const values = state.model.features.map((feature) => feature.value);
  const set = (id, value) => {
    const index = state.model.features.findIndex((feature) => feature.id === id);
    values[index] = value;
  };

  if (kind === "normal") {
    set("amount_usd", 68);
    set("hour_of_day", 14);
    set("tx_velocity_1h", 1);
    set("distance_from_home_km", 12);
    set("device_trust", 0.93);
    set("email_age_days", 1800);
    set("failed_logins_24h", 0);
    set("merchant_risk", 0.12);
    set("card_present", 1);
    set("cards_per_device_7d", 1);
    set("geo_mismatch", 0);
    set("account_age_days", 1400);
  } else {
    set("amount_usd", 2450);
    set("hour_of_day", 2);
    set("tx_velocity_1h", 9);
    set("distance_from_home_km", 820);
    set("device_trust", 0.16);
    set("email_age_days", 8);
    set("failed_logins_24h", 7);
    set("merchant_risk", 0.91);
    set("card_present", 0);
    set("cards_per_device_7d", 13);
    set("geo_mismatch", 1);
    set("account_age_days", 22);
  }
  setValues(values);
}

async function init() {
  const response = await fetch(MODEL_URL);
  if (!response.ok) {
    throw new Error(`Unable to load ${MODEL_URL}`);
  }
  state.model = await response.json();
  renderFields();
  renderMetrics();
  update();
}

$("#risk-form").addEventListener("submit", (event) => event.preventDefault());
$("#normal-example").addEventListener("click", () => loadExample("normal"));
$("#suspicious-example").addEventListener("click", () => loadExample("suspicious"));

init().catch((error) => {
  console.error(error);
  $("#decision-chip").textContent = "Model load failed";
});
