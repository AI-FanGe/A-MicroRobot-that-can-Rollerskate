const UPPER_BODY_MOTORS = [
  { id: 13, label: "右肩前后", min: -150, max: 150 },
  { id: 14, label: "右肩侧摆", min: 0, max: 160 },
  { id: 15, label: "右肘", min: 0, max: 85 },
  { id: 16, label: "左肩前后", min: -150, max: 150 },
  { id: 17, label: "左肩侧摆", min: -160, max: 0 },
  { id: 18, label: "左肘", min: -85, max: 0 },
  { id: 19, label: "头部左右", min: -40, max: 40 },
  { id: 20, label: "头部俯仰", min: -10, max: 30 },
];

const state = {
  robots: [],
  selectedId: null,
  expressions: [],
  selectedExpr: "happy",
  loop: [],
  frame: 0,
  walk: false,
  vx: 0,
  vy: 0,
  vtheta: 0,
  hold: null,
  holdTimer: null,
  sending: false,
  upperBody: {
    started: false,
    starting: null,
    sending: false,
    pending: false,
    timer: null,
    touched: new Set(),
    values: Object.fromEntries(UPPER_BODY_MOTORS.map((item) => [item.id, 0])),
  },
  lead: { running: false, phase: "stopped", frames: 0, ids: [], error: null },
};

const $ = (id) => document.getElementById(id);

function log(message) {
  const box = $("log");
  if (!box) return;
  const time = new Date().toLocaleTimeString();
  box.textContent = `[${time}] ${message}\n` + box.textContent;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function roundedRect(ctx, x, y, size, radius) {
  const r = Math.min(radius, size / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + size, y, x + size, y + size, r);
  ctx.arcTo(x + size, y + size, x, y + size, r);
  ctx.arcTo(x, y + size, x, y, r);
  ctx.arcTo(x, y, x + size, y, r);
  ctx.closePath();
}

function drawMatrix(canvas, cols, rows, pixels, { glow = true } = {}) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
  const pitch = Math.min(width / cols, height / rows);
  const ox = (width - (cols - 1) * pitch) / 2;
  const oy = (height - (rows - 1) * pitch) / 2;
  const unlit = Math.max(1.1, pitch * 0.40);
  const core = Math.max(1.8, pitch * 0.72);

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      roundedRect(ctx, ox + x * pitch - unlit / 2, oy + y * pitch - unlit / 2, unlit, unlit * 0.22);
      ctx.fillStyle = "#121416";
      ctx.fill();
    }
  }

  if (glow && pixels.length) {
    const stamp = (dx, dy, color) => {
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      ctx.filter = "blur(1.6px)";
      ctx.fillStyle = color;
      ctx.translate(dx, dy);
      for (const [x, y] of pixels) {
        roundedRect(ctx, ox + x * pitch - core / 2, oy + y * pitch - core / 2, core, core * 0.22);
        ctx.fill();
      }
      ctx.restore();
    };
    stamp(0, 0, "rgba(255, 236, 210, 0.45)");
  }

  ctx.filter = "none";
  ctx.globalCompositeOperation = "source-over";
  for (const [x, y] of pixels) {
    roundedRect(ctx, ox + x * pitch - core / 2, oy + y * pitch - core / 2, core, core * 0.5);
    ctx.fillStyle = "#ffe8c8";
    ctx.fill();
  }
}

function currentExpression() {
  return state.expressions.find((item) => item.name === state.selectedExpr) || state.expressions[1];
}

function currentPixels() {
  if (state.loop.length) {
    return state.loop[state.frame % state.loop.length];
  }
  return currentExpression()?.pixels || [];
}

function renderFace() {
  const item = currentExpression();
  if (!item) return;
  drawMatrix($("faceCanvas"), item.cols, item.rows, currentPixels());
  $("faceCaption").textContent = `${String(item.index).padStart(2, "0")} ${item.label}`;
}

function renderExpressions() {
  const grid = $("exprGrid");
  grid.innerHTML = "";
  for (const item of state.expressions) {
    const button = document.createElement("button");
    button.className = `expr-btn${item.name === state.selectedExpr ? " selected" : ""}`;
    button.type = "button";
    const canvas = document.createElement("canvas");
    canvas.width = 152;
    canvas.height = 112;
    drawMatrix(canvas, item.cols, item.rows, item.pixels, { glow: true });
    const label = document.createElement("small");
    label.textContent = `${String(item.index).padStart(2, "0")} ${item.label}`;
    button.append(canvas, label);
    button.addEventListener("click", () => setExpression(item.name));
    grid.append(button);
  }
}

function setUpperBodyStatus(text) {
  const status = $("upperBodyStatus");
  if (status) status.textContent = text;
}

function upperBodyPayload() {
  return {
    targets_deg: Object.fromEntries(
      UPPER_BODY_MOTORS
        .filter((item) => state.upperBody.touched.has(item.id))
        .map((item) => [String(item.id), Number(state.upperBody.values[item.id] || 0)])
    ),
  };
}

function renderUpperBodySliders() {
  const grid = $("upperBodyGrid");
  if (!grid) return;
  grid.innerHTML = "";
  for (const item of UPPER_BODY_MOTORS) {
    const row = document.createElement("label");
    row.className = "motor-slider";
    row.innerHTML = `
      <span class="motor-name">${item.id} ${item.label}</span>
      <input type="range" min="${item.min ?? -120}" max="${item.max ?? 120}" step="1" value="${state.upperBody.values[item.id]}" data-motor-id="${item.id}" />
      <output>${state.upperBody.values[item.id]}°</output>
    `;
    const input = row.querySelector("input");
    const output = row.querySelector("output");
    input.addEventListener("input", () => {
      const value = Number(input.value);
      state.upperBody.values[item.id] = value;
      state.upperBody.touched.add(item.id);
      output.textContent = `${value}°`;
      scheduleUpperBodySend();
    });
    input.addEventListener("change", () => scheduleUpperBodySend(true));
    grid.append(row);
  }
}

async function ensureUpperBodyStarted() {
  const robot = selectedRobot();
  if (!robot) throw new Error("请先选择机器人");
  if (state.upperBody.started) return;
  if (!state.upperBody.starting) {
    setUpperBodyStatus("启动中…");
    state.upperBody.starting = api(`/api/robots/${robot.id}/upper-body-start`, { method: "POST", body: {} })
      .then((result) => {
        if (result.error && !result.ok) throw new Error(result.error);
        state.upperBody.started = true;
        setUpperBodyStatus("已启动");
        log("上半身 slider 已启动");
      })
      .finally(() => {
        state.upperBody.starting = null;
      });
  }
  await state.upperBody.starting;
}

async function sendUpperBodyTargets() {
  const robot = selectedRobot();
  if (!robot) return;
  if (state.upperBody.sending) {
    state.upperBody.pending = true;
    return;
  }
  state.upperBody.sending = true;
  try {
    await ensureUpperBodyStarted();
    if (state.upperBody.touched.size) {
      await api(`/api/robots/${robot.id}/upper-body`, { method: "POST", body: upperBodyPayload() });
      setUpperBodyStatus("已下发");
    }
  } catch (error) {
    setUpperBodyStatus("失败");
    log(`上半身控制失败：${error.message}`);
  } finally {
    state.upperBody.sending = false;
    if (state.upperBody.pending) {
      state.upperBody.pending = false;
      scheduleUpperBodySend(true);
    }
  }
}

function scheduleUpperBodySend(immediate = false) {
  if (state.upperBody.timer) clearTimeout(state.upperBody.timer);
  state.upperBody.timer = setTimeout(() => {
    state.upperBody.timer = null;
    sendUpperBodyTargets();
  }, immediate ? 0 : 33);
}

async function startUpperBody() {
  try {
    await ensureUpperBodyStarted();
    await sendUpperBodyTargets();
  } catch (error) {
    setUpperBodyStatus("失败");
    log(`启动上半身失败：${error.message}`);
  }
}

async function stopUpperBody() {
  const robot = selectedRobot();
  if (!robot) return;
  try {
    const result = await api(`/api/robots/${robot.id}/upper-body-stop`, { method: "POST", body: {} });
    if (result.error && !result.ok) throw new Error(result.error);
    state.upperBody.started = false;
    setUpperBodyStatus("未启动");
    log("上半身 slider 已停止");
  } catch (error) {
    log(`停止上半身失败：${error.message}`);
  }
}

function zeroUpperBody() {
  for (const item of UPPER_BODY_MOTORS) {
    state.upperBody.values[item.id] = 0;
    state.upperBody.touched.add(item.id);
  }
  renderUpperBodySliders();
  scheduleUpperBodySend(true);
}

function renderRobots() {
  const list = $("robotList");
  list.innerHTML = "";
  if (!state.robots.length) {
    list.innerHTML = '<p class="muted">还没有机器人。点右上角「添加」。</p>';
    return;
  }
  for (const robot of state.robots) {
    const card = document.createElement("div");
    card.className = `robot-card${robot.id === state.selectedId ? " active" : ""}`;
    card.innerHTML = `
      <div class="row">
        <b>${robot.name}</b>
        <span class="dot ${robot.online ? "online" : ""}"></span>
      </div>
      <span>${robot.user}@${robot.host}:${robot.port}</span>
      <span>${robot.has_password ? "已保存密码" : "需要密码"}</span>
    `;
    card.addEventListener("click", () => selectRobot(robot.id));
    list.append(card);
  }
  fillLeadSelects();
}

function guessLeadFollow() {
  const duck = state.robots.find((item) => /duck/i.test(`${item.name} ${item.host}`));
  const ban = state.robots.find((item) => /ban/i.test(`${item.name} ${item.host}`) && item !== duck);
  return {
    lead: duck?.id || state.robots[0]?.id || "",
    follow: ban?.id || state.robots.find((item) => item.id !== duck?.id)?.id || "",
  };
}

function fillLeadSelects() {
  const lead = $("leadRobot");
  const follow = $("followRobot");
  if (!lead || !follow) return;
  const previous = { lead: lead.value, follow: follow.value };
  const guess = guessLeadFollow();
  const options = state.robots
    .map((robot) => `<option value="${robot.id}">${robot.name} · ${robot.host}</option>`)
    .join("");
  lead.innerHTML = options;
  follow.innerHTML = options;
  lead.value = previous.lead || guess.lead;
  follow.value = previous.follow || guess.follow;
}

function leadLabel(status) {
  if (status.error) return `出错：${status.error}`;
  if (status.phase === "starting") return "正在推送并接通…";
  if (status.phase === "streaming") {
    const ids = (status.ids || []).join(",") || "—";
    return `跟随中 ${status.frames || 0} 帧 · ID ${ids}`;
  }
  if (status.phase === "stopping") return "正在停止…";
  return "未开始";
}

async function refreshLead() {
  try {
    const status = await api("/api/lead");
    state.lead = status;
    $("leadStatus").textContent = leadLabel(status);
  } catch (error) {
    $("leadStatus").textContent = error.message;
  }
}

async function startLead() {
  const leadId = $("leadRobot").value;
  const followId = $("followRobot").value;
  if (!leadId || !followId) {
    log("请先添加示教机和真机两台机器人");
    return;
  }
  if (leadId === followId) {
    log("示教机和真机不能是同一台");
    return;
  }
  log("正在开始示教…会先停掉两边的走路控制循环");
  try {
    const result = await api("/api/lead/start", {
      method: "POST",
      body: { lead_id: leadId, follow_id: followId, ids: $("leadIds").value || "13-18" },
    });
    if (result.error && !result.ok) throw new Error(result.error);
    await refreshLead();
    log("示教正在接通：先推代码、停走路，然后用手扳 microduck 即可。进度看上面状态。");
  } catch (error) {
    log(`示教启动失败：${error.message}`);
  }
}

async function stopLead() {
  try {
    await api("/api/lead/stop", { method: "POST", body: {} });
    await refreshLead();
    log("已停止示教，真机手臂扭矩已松开");
  } catch (error) {
    log(`停止示教失败：${error.message}`);
  }
}

function selectedRobot() {
  return state.robots.find((item) => item.id === state.selectedId) || null;
}

function updateDeskChrome(status = {}) {
  const robot = selectedRobot();
  if (!robot) return;
  $("robotTitle").textContent = robot.name;
  $("robotHost").textContent = `${robot.user}@${robot.host}`;
  const online = Boolean(status.online ?? robot.online);
  const running = Boolean(status.running ?? robot.running);
  $("onlinePill").textContent = online ? "已连接" : "离线";
  $("onlinePill").className = `pill ${online ? "ok" : ""}`;
  $("runPill").textContent = running ? "控制循环运行中" : "控制未启动";
  $("runPill").className = `pill ${running ? "ok" : "warn"}`;
  $("latencyPill").textContent = status.latency_ms != null ? `${status.latency_ms} ms` : "—";
  $("authPill").textContent = robot.has_password ? "已保存密码" : "未保存密码";
  $("authPill").className = `pill ${robot.has_password ? "ok" : "warn"}`;
  if ($("authHost") && !document.activeElement?.closest("#authForm")) {
    $("authHost").value = robot.host || "xiaoxiaobai.local";
    $("authUser").value = robot.user || "user";
  }
}

async function loadRobots() {
  const data = await api("/api/robots");
  state.robots = data.robots || [];
  renderRobots();
}

async function loadExpressions() {
  const data = await api("/api/expressions");
  state.expressions = data.items || [];
  $("exprCount").textContent = `${state.expressions.length} 个`;
  renderExpressions();
  renderFace();
}

async function selectRobot(id) {
  state.selectedId = id;
  state.upperBody.started = false;
  state.upperBody.touched = new Set();
  setUpperBodyStatus("未启动");
  $("emptyState").classList.add("hidden");
  $("desk").classList.remove("hidden");
  renderRobots();
  updateDeskChrome();
  try {
    const status = await api(`/api/robots/${id}/status`);
    const robot = selectedRobot();
    if (robot) {
      robot.online = status.online;
      robot.running = status.running;
    }
    if (status.expression) {
      state.selectedExpr = status.expression;
      renderExpressions();
      renderFace();
    }
    updateDeskChrome(status);
    log(status.online ? `已连上 ${robot?.name || id}` : `连不上：${status.error || "未知错误"}`);
  } catch (error) {
    log(error.message);
  }
}

async function loadLoop(name) {
  const data = await api(`/api/expressions/${name}/loop`);
  state.loop = data.frames || [];
  state.frame = 0;
  renderFace();
}

async function setExpression(name) {
  state.selectedExpr = name;
  renderExpressions();
  try {
    await loadLoop(name);
  } catch (error) {
    state.loop = [];
    renderFace();
    log(error.message);
  }
  const robot = selectedRobot();
  if (!robot) return;
  try {
    const result = await api(`/api/robots/${robot.id}/expression`, {
      method: "POST",
      body: { name },
    });
    log(`表情切换为 ${result.expression || name}`);
  } catch (error) {
    log(`表情下发失败：${error.message}`);
  }
}

function velocityPayload() {
  return {
    walk: state.walk,
    vx: state.vx,
    vy: state.vy,
    vtheta: state.vtheta,
    expression: state.selectedExpr,
  };
}

async function sendControl() {
  const robot = selectedRobot();
  if (!robot || state.sending) return;
  state.sending = true;
  try {
    await api(`/api/robots/${robot.id}/control`, { method: "POST", body: velocityPayload() });
  } catch (error) {
    log(`控制失败：${error.message}`);
  } finally {
    state.sending = false;
  }
}

function setVelocity({ vx = state.vx, vy = state.vy, vtheta = state.vtheta, walk = state.walk } = {}) {
  state.vx = vx;
  state.vy = vy;
  state.vtheta = vtheta;
  state.walk = walk;
  $("walkToggle").checked = walk;
  $("velReadout").textContent = `vx ${vx.toFixed(2)} · vθ ${vtheta.toFixed(2)}`;
}

function startHold(kind) {
  stopHold();
  const apply = () => {
    if (kind === "forward") setVelocity({ walk: true, vx: 0.55, vtheta: 0 });
    if (kind === "back") setVelocity({ walk: true, vx: -0.4, vtheta: 0 });
    if (kind === "left") setVelocity({ walk: true, vx: 0, vtheta: 0.55 });
    if (kind === "right") setVelocity({ walk: true, vx: 0, vtheta: -0.55 });
    if (kind === "stop") setVelocity({ walk: false, vx: 0, vy: 0, vtheta: 0 });
    sendControl();
  };
  apply();
  if (kind !== "stop") {
    state.hold = kind;
    state.holdTimer = setInterval(apply, 220);
  }
}

function stopHold() {
  if (state.holdTimer) clearInterval(state.holdTimer);
  state.holdTimer = null;
  if (state.hold) {
    state.hold = null;
    setVelocity({ vx: 0, vy: 0, vtheta: 0 });
    sendControl();
  }
}

function attachStick() {
  const stick = $("stick");
  const knob = $("stickKnob");
  const radius = 63;
  let dragging = false;

  const setFromPoint = (event) => {
    const rect = stick.getBoundingClientRect();
    const dx = event.clientX - (rect.left + rect.width / 2);
    const dy = event.clientY - (rect.top + rect.height / 2);
    const mag = Math.min(radius, Math.hypot(dx, dy));
    const ang = Math.atan2(dy, dx);
    const x = Math.cos(ang) * mag;
    const y = Math.sin(ang) * mag;
    knob.style.left = `${63 + x}px`;
    knob.style.top = `${63 + y}px`;
    const nx = x / radius;
    const ny = -y / radius;
    setVelocity({
      walk: true,
      vx: Math.max(-1, Math.min(1, ny)),
      vtheta: Math.max(-1, Math.min(1, -nx)),
    });
    sendControl();
  };

  stick.addEventListener("pointerdown", (event) => {
    dragging = true;
    stick.setPointerCapture(event.pointerId);
    setFromPoint(event);
  });
  stick.addEventListener("pointermove", (event) => {
    if (dragging) setFromPoint(event);
  });
  const end = () => {
    if (!dragging) return;
    dragging = false;
    knob.style.left = "63px";
    knob.style.top = "63px";
    setVelocity({ vx: 0, vtheta: 0 });
    sendControl();
  };
  stick.addEventListener("pointerup", end);
  stick.addEventListener("pointercancel", end);
}

async function runAction(action) {
  const robot = selectedRobot();
  if (!robot) return;
  log(`正在 ${action} …`);
  try {
    const result = await api(`/api/robots/${robot.id}/${action}`, { method: "POST", body: {} });
    if (result.error && !result.ok) throw new Error(result.error);
    log(result.output || result.message || `${action} 完成`);
    if (action !== "shutdown") {
      await selectRobot(robot.id);
    }
  } catch (error) {
    log(`${action} 失败：${error.message}`);
  }
}

function bindEvents() {
  $("toggleFormBtn").addEventListener("click", () => $("robotForm").classList.toggle("hidden"));
  $("cancelFormBtn").addEventListener("click", () => $("robotForm").classList.add("hidden"));
  $("robotForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    const body = Object.fromEntries(form.entries());
    const existingId = body.id;
    delete body.id;
    try {
      const data = existingId
        ? await api(`/api/robots/${existingId}`, { method: "POST", body })
        : await api("/api/robots", { method: "POST", body });
      event.target.reset();
      event.target.port.value = "22";
      event.target.user.value = "user";
      event.target.host.value = "xiaoxiaobai.local";
      $("robotForm").classList.add("hidden");
      await loadRobots();
      await selectRobot(data.robot.id);
    } catch (error) {
      log(error.message);
    }
  });
  $("authForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const robot = selectedRobot();
    if (!robot) return;
    const password = $("authPassword").value;
    const host = $("authHost").value.trim();
    const user = $("authUser").value.trim() || "user";
    if (!host) {
      log("请填写主机名，例如 xiaoxiaobai.local");
      $("authHost").focus();
      return;
    }
    if (!password && !robot.has_password) {
      log("请先输入这台 Pi 的 SSH 密码");
      $("authPassword").focus();
      return;
    }
    try {
      const body = { host, user };
      if (password) body.password = password;
      await api(`/api/robots/${robot.id}`, { method: "POST", body });
      $("authPassword").value = "";
      await loadRobots();
      state.selectedId = robot.id;
      log("密码已保存，正在连接 …");
      await selectRobot(robot.id);
    } catch (error) {
      log(error.message);
    }
  });

  $("discoverBtn").addEventListener("click", async () => {
    log("正在扫描 .local …");
    try {
      const data = await api("/api/discover");
      if (!data.hosts.length) {
        log("没有扫到 SSH 主机，请手动添加。");
        return;
      }
      for (const host of data.hosts) {
        const exists = state.robots.some((item) => item.host === host.host);
        if (!exists) {
          await api("/api/robots", {
            method: "POST",
            body: { name: host.name, host: host.host, user: "user", port: 22 },
          });
        }
      }
      await loadRobots();
      log(`发现 ${data.hosts.length} 台：${data.hosts.map((item) => item.host).join(", ")}`);
    } catch (error) {
      log(error.message);
    }
  });

  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => runAction(button.dataset.action));
  });
  $("upperBodyStartBtn").addEventListener("click", startUpperBody);
  $("upperBodyZeroBtn").addEventListener("click", zeroUpperBody);
  $("upperBodyStopBtn").addEventListener("click", stopUpperBody);
  document.querySelectorAll("[data-walk]").forEach((button) => {
    button.addEventListener("pointerdown", () => startHold(button.dataset.walk));
    button.addEventListener("pointerup", stopHold);
    button.addEventListener("pointerleave", stopHold);
  });
  $("walkToggle").addEventListener("change", (event) => {
    setVelocity({ walk: event.target.checked, vx: 0, vtheta: 0 });
    sendControl();
  });

  window.addEventListener("keydown", (event) => {
    if (event.repeat) return;
    const map = { KeyW: "forward", ArrowUp: "forward", KeyS: "back", ArrowDown: "back", KeyA: "left", ArrowLeft: "left", KeyD: "right", ArrowRight: "right", Space: "stop" };
    if (map[event.code]) {
      event.preventDefault();
      startHold(map[event.code]);
    }
  });
  window.addEventListener("keyup", (event) => {
    const map = { KeyW: 1, ArrowUp: 1, KeyS: 1, ArrowDown: 1, KeyA: 1, ArrowLeft: 1, KeyD: 1, ArrowRight: 1, Space: 1 };
    if (map[event.code]) stopHold();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      setVelocity({ walk: false, vx: 0, vy: 0, vtheta: 0 });
      sendControl();
    }
  });
  attachStick();
  $("leadStartBtn").addEventListener("click", startLead);
  $("leadStopBtn").addEventListener("click", stopLead);
}

function tickClock() {
  $("clock").textContent = new Date().toLocaleTimeString();
}

async function main() {
  tickClock();
  setInterval(tickClock, 1000);
  renderUpperBodySliders();
  bindEvents();
  setInterval(() => {
    if (state.walk && selectedRobot()) sendControl();
  }, 250);
  setInterval(refreshLead, 1000);
  refreshLead();
  setInterval(() => {
    if (!state.loop.length) return;
    state.frame = (state.frame + 1) % state.loop.length;
    renderFace();
  }, 100);
  await loadExpressions();
  await loadRobots();
  if (state.robots[0]) await selectRobot(state.robots[0].id);
  if (state.selectedExpr) {
    try {
      await loadLoop(state.selectedExpr);
    } catch (error) {
      log(error.message);
    }
  }
}

main().catch((error) => log(error.message));
