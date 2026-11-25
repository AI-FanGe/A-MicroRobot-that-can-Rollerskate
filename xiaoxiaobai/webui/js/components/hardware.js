// Hardware callouts on the virtual robot: leader lines and tags for every motor, board and
// sensor, and a detail panel with live readings and whatever control each part supports.

import * as api from "../api.js";
import { action, on, store, UPPER_MOTORS } from "../store.js";
import { clear, confirmDialog, duration, h, icon, rangeEl, throttle, timeAgo, toast } from "../ui.js";
export const JOINT_LABEL = {
  right_hip_yaw: "右髋偏航", right_hip_roll: "右髋侧摆", right_hip_pitch: "右髋俯仰",
  right_knee: "右膝", right_ankle_pitch: "右踝俯仰", right_ankle_roll: "右踝侧摆",
  left_hip_yaw: "左髋偏航", left_hip_roll: "左髋侧摆", left_hip_pitch: "左髋俯仰",
  left_knee: "左膝", left_ankle_pitch: "左踝俯仰", left_ankle_roll: "左踝侧摆",
  right_shoulder_pitch: "右肩俯仰", right_shoulder_roll: "右肩侧摆", right_elbow: "右肘",
  left_shoulder_pitch: "左肩俯仰", left_shoulder_roll: "左肩侧摆", left_elbow: "左肘",
  head: "头部偏航", head_pitch: "头部俯仰",
};

const INFO = {
  pi: { title: "主控板", model: "50 Hz 控制循环", icon: "cpu",
    desc: "运行 50 Hz 控制循环、强化学习策略、本网页服务和大模型对话。" },
  imu: { title: "IMU", model: "I2C-1", icon: "activity",
    desc: "加速度计 + 陀螺仪，给走路、滑板等策略提供机身姿态。" },
  battery: { title: "电池", model: "2S 锂电（2×18650）", icon: "battery",
    desc: "满电 8.4 V。控制循环运行时每 5 秒经 19 号电机读一次电压。" },
  esp32: { title: "头部板", model: "摄像头 · 麦克风", icon: "camera",
    desc: "驱动表情屏、扬声器、麦克风和摄像头，通过 UART 与主控通信。" },
  screen: { title: "表情屏", model: "LCD", icon: "smile", desc: "显示表情动画，可以从这里直接切换。" },
  speaker: { title: "扬声器", model: "I2S 功放", icon: "volume", desc: "提示音和语音对话的输出。" },
  motor: { title: "电机", model: "总线舵机", icon: "zap", desc: "" },
};

const UPPER_IDS = new Set([13, 14, 15, 16, 17, 18, 19, 20]);
const FILTERS = [
  { id: "electronics", label: "电子" },
  { id: "motors", label: "电机" },
  { id: "all", label: "全部" },
  { id: "off", label: "隐藏" },
];
const deg = (rad) => (rad == null ? null : (rad * 180) / Math.PI);
const fmtDeg = (rad, digits = 1) => (rad == null ? "—" : `${deg(rad).toFixed(digits)}°`);

function titleOf(c) {
  if (c.kind === "motor") return `${c.motor_id} · ${JOINT_LABEL[c.joint] || c.joint}`;
  return INFO[c.kind]?.title || c.id;
}

/**
 * Callout layer for a Twin inside `stage`. Returns { toolbar, dispose }.
 * opts.setAction(name) switches the motion mode (shared with the page).
 */
export function hardwareLayer(stage, twin, opts = {}) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("hw-lines");
  const tagsEl = h("div.hw-tags");
  const panel = h("div.hw-panel", { hidden: true });
  const layer = h("div.hw-layer", svg, tagsEl);
  stage.append(layer, panel);

  let filter = localStorage.getItem("xiaoxiaobai.hwFilter") || "electronics";
  let selected = null;
  let hover = null;
  const tags = new Map();
  const live = { joints: null, jointsAt: 0, info: null, imu: null };
  const offs = [];
  const timers = [];

  const seg = h("div.seg.sm", FILTERS.map((f) => h(`button${f.id === filter ? ".active" : ""}`, {
    onclick: (e) => {
      filter = f.id;
      localStorage.setItem("xiaoxiaobai.hwFilter", filter);
      seg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === e.currentTarget));
      rebuildTags();
    },
  }, f.label)));
  const toolbar = h("div.hw-toolbar", seg);

  // ------------------------------------------------------------------ tags

  function shown(c) {
    if (c.id === hover) return true;
    // While a part is open in the panel only its own callout stays, so the robot is not buried in labels.
    if (selected) return c.id === selected;
    if (filter === "off") return false;
    if (filter === "motors") return c.kind === "motor";
    if (filter === "electronics") return c.kind !== "motor";
    return true;
  }

  // Add/remove only what changed so existing tags keep their place.
  function rebuildTags() {
    const want = new Set(twin.componentList().filter(shown).map((c) => c.id));
    for (const [id, t] of tags) {
      if (want.has(id)) continue;
      t.el.remove();
      t.line.remove();
      t.dot.remove();
      tags.delete(id);
    }
    for (const c of twin.componentList()) {
      if (!want.has(c.id) || tags.has(c.id)) continue;
      const value = h("span.v");
      const el = h(`button.hw-tag.k-${c.kind}`, {
        onclick: () => select(c.id === selected ? null : c.id),
        onmouseenter: () => setHover(c.id),
        onmouseleave: () => setHover(null),
      }, h("i"), h("span.t", titleOf(c)), value);
      const line = document.createElementNS(svg.namespaceURI, "polyline");
      const dot = document.createElementNS(svg.namespaceURI, "circle");
      dot.setAttribute("r", "2.6");
      svg.append(line, dot);
      tagsEl.append(el);
      tags.set(c.id, { c, el, value, line, dot, y: null, side: null, w: 0, anchor: {} });
    }
    refreshValues();
    syncTwin();
  }

  function refreshValues() {
    const s = store.snapshot;
    const joints = live.joints ? Object.fromEntries(live.joints.joints.map((j) => [j.name, j])) : {};
    for (const t of tags.values()) {
      const { c } = t;
      let text = "";
      if (c.kind === "motor") {
        const j = joints[c.joint];
        text = j?.position != null ? fmtDeg(j.position) : j?.target != null ? `→${fmtDeg(j.target, 0)}` : "";
      } else if (c.kind === "battery") {
        text = s?.battery?.voltage_v ? `${s.battery.percent}%` : "";
      } else if (c.kind === "imu") {
        const loop = s?.action?.loop;
        text = loop?.roll_deg != null ? `${loop.roll_deg.toFixed(1)}° / ${loop.pitch_deg.toFixed(1)}°` : "";
      } else if (c.kind === "pi") {
        text = live.info?.cpu_temp_c != null ? `${live.info.cpu_temp_c.toFixed(0)}°C` : "";
      } else if (c.kind === "screen") {
        text = store.expressions?.find((e) => e.name === s?.expression)?.label || "";
      }
      if (t.value.textContent !== text) {
        t.value.textContent = text;
        t.w = 0;
      }
    }
  }

  // Two columns hugging the robot; tags keep their vertical order and never overlap.
  function layout() {
    const H = stage.clientHeight;
    if (panel.hidden) twin.setInset();
    else if (panel.offsetTop < H / 2) twin.setInset({ right: panel.offsetWidth + 16 });
    else twin.setInset({ bottom: H - panel.offsetTop + 8 });
    // Keep the right column clear of the open detail panel (desktop layout only).
    const W = stage.clientWidth - (!panel.hidden && panel.offsetTop < H / 2 ? panel.offsetWidth + 16 : 0);
    if (!W || !H || !tags.size) return;
    let minX = Infinity;
    let maxX = -Infinity;
    for (const c of twin.componentList()) {
      const p = twin.project(c.id);
      if (!p?.visible) continue;
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
    }
    if (!Number.isFinite(minX)) return;
    const cx = (minX + maxX) / 2;
    const top = 62;
    const bottom = !panel.hidden && panel.offsetTop >= H / 2 ? panel.offsetTop - 14 : H - 52;
    const cols = { left: [], right: [] };
    for (const t of tags.values()) {
      twin.project(t.c.id, t.anchor);
      if (!t.w) t.w = t.el.offsetWidth;
      const want = t.anchor.x < cx ? "left" : "right";
      if (t.side !== want && (t.side == null || Math.abs(t.anchor.x - cx) > 10)) t.side = want;
      cols[t.side].push(t);
    }
    const gap = 30;
    for (const [side, list] of Object.entries(cols)) {
      if (!list.length) continue;
      list.sort((a, b) => a.anchor.y - b.anchor.y);
      const step = Math.min(27, (bottom - top) / list.length);
      const ys = list.map((t) => Math.min(bottom, Math.max(top, t.anchor.y)));
      for (let i = 1; i < ys.length; i++) ys[i] = Math.max(ys[i], ys[i - 1] + step);
      if (ys[ys.length - 1] > bottom) {
        ys[ys.length - 1] = bottom;
        for (let i = ys.length - 2; i >= 0; i--) ys[i] = Math.min(ys[i], ys[i + 1] - step);
      }
      list.forEach((t, i) => {
        t.y = t.y == null ? ys[i] : t.y + (ys[i] - t.y) * 0.3;
        const x = side === "left"
          ? Math.max(8, Math.min(minX - gap, W / 2) - t.w)
          : Math.min(W - 8 - t.w, Math.max(maxX + gap, W / 2));
        t.el.style.transform = `translate(${x.toFixed(1)}px, ${(t.y - 11).toFixed(1)}px)`;
        const edge = side === "left" ? x + t.w : x;
        const elbow = side === "left" ? edge + 12 : edge - 12;
        t.line.setAttribute("points", `${t.anchor.x.toFixed(1)},${t.anchor.y.toFixed(1)} ${elbow.toFixed(1)},${t.y.toFixed(1)} ${edge.toFixed(1)},${t.y.toFixed(1)}`);
        t.dot.setAttribute("cx", t.anchor.x.toFixed(1));
        t.dot.setAttribute("cy", t.anchor.y.toFixed(1));
        const on = t.c.id === selected || t.c.id === hover;
        t.el.classList.toggle("on", on);
        t.line.classList.toggle("on", on);
        t.dot.classList.toggle("on", on);
      });
    }
  }

  function syncTwin() {
    twin.highlight([selected, hover]);
  }

  function setHover(id) {
    if (hover === id) return;
    const previous = hover;
    hover = id;
    stage.classList.toggle("hw-pointer", !!id);
    if ((id && !tags.has(id)) || (previous && tags.has(previous))) rebuildTags();
    else syncTwin();
  }

  // ------------------------------------------------------------------ picking on the canvas

  let down = null;
  let lastMove = 0;
  const canvas = stage.querySelector("canvas");
  const local = (e) => {
    const r = stage.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const onDown = (e) => (down = [e.clientX, e.clientY]);
  const onUp = (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    down = null;
    const id = twin.pick(...local(e));
    select(id && id === selected ? null : id);
  };
  const onMove = (e) => {
    if (e.buttons || performance.now() - lastMove < 60) return;
    lastMove = performance.now();
    setHover(twin.pick(...local(e)));
  };
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointermove", onMove);
  const onLeave = () => setHover(null);
  canvas.addEventListener("pointerleave", onLeave);

  // ------------------------------------------------------------------ detail panel

  let panelTick = null;

  function select(id) {
    selected = id;
    panelTick = null;
    const c = id && twin.components.get(id);
    rebuildTags();
    if (!c) {
      panel.hidden = true;
      clear(panel);
      return;
    }
    if (c.kind === "motor") twin.previewJoint(c.joint);
    renderPanel(c);
    panel.hidden = false;
  }

  function renderPanel(c) {
    const info = INFO[c.kind] || INFO.motor;
    const body = h("div.hw-body");
    clear(panel,
      h("div.hw-head",
        h(`span.hw-ico.k-${c.kind}`, icon(info.icon, "sm")),
        h("div.hw-title", h("b", c.kind === "motor" ? `${c.motor_id} 号电机` : info.title), h("small", info.model)),
        h("button.btn.ghost.sm.icon-only", { title: "关闭", onclick: () => select(null) }, icon("x"))),
      body);
    if (info.desc) body.append(h("p.hw-desc", info.desc));
    const builder = { motor: motorPanel, pi: piPanel, imu: imuPanel, battery: batteryPanel, esp32: esp32Panel, screen: screenPanel, speaker: speakerPanel }[c.kind];
    builder?.(c, body);
  }

  const kv = (rows) => h("dl.hw-kv", rows.flatMap(([k, v]) => [h("dt", k), h("dd", v)]));

  function motorPanel(c, body) {
    const joint = twin.joints.get(c.joint)?.def;
    const range = joint?.range;
    const side = c.joint.startsWith("right") ? "右侧" : c.joint.startsWith("left") ? "左侧" : "中线";
    body.append(kv([
      ["关节", h("span", JOINT_LABEL[c.joint] || c.joint, h("code.hw-code", c.joint))],
      ["位置", side],
      ["工作范围", range ? `${fmtDeg(range[0], 0)} ~ ${fmtDeg(range[1], 0)}` : "±45°（仅上肢工具）"],
    ]));
    const liveBox = h("div.hw-live");
    body.append(h("div.hw-section", h("h4", "实时反馈"), liveBox));
    const paint = () => {
      const j = live.joints?.joints.find((x) => x.name === c.joint);
      if (!j || (j.position == null && j.target == null)) {
        clear(liveBox, h("p.muted.small", live.joints?.note || "读取中…"));
        return;
      }
      const err = j.position != null && j.target != null ? j.target - j.position : null;
      clear(liveBox, h("div.hw-metrics",
        metric("当前", fmtDeg(j.position)),
        metric("目标", fmtDeg(j.target)),
        metric("误差", err == null ? "—" : fmtDeg(err, 2), err != null && Math.abs(deg(err)) > 8 ? "warn" : ""),
        metric("速度", j.velocity == null ? "—" : `${deg(j.velocity).toFixed(0)}°/s`)),
        live.joints.stamp ? h("small.muted", `更新于 ${timeAgo(live.joints.stamp)}`) : null);
    };
    panelTick = paint;
    paint();

    const ctl = h("div.hw-section", h("h4", "控制"));
    body.append(ctl);
    const actions = h("div.hw-actions",
      h("button.btn.sm", { onclick: () => twin.previewJoint(c.joint) }, icon("play", "sm"), "演示转动方向"));
    if (UPPER_IDS.has(c.motor_id)) {
      if (action() === "BIPED_CUSTOM_UPPER") {
        const m = UPPER_MOTORS.find((u) => u.id === c.motor_id);
        const current = Math.max(m.min, Math.min(m.max, twin.state.upper?.[c.motor_id] ?? 0));
        const send = throttle((v) => api.post("/mc/upper_body", { targets_deg: { [c.motor_id]: v } })
          .catch((e) => toast(e.message, "error")), 40);
        ctl.append(rangeEl({
          min: m.min, max: m.max, step: 1, value: current, format: (v) => `${v}°`,
          oninput: (v) => {
            twin.update({ upper: { ...twin.state.upper, [c.motor_id]: v } });
            send(v);
          },
        }));
      } else {
        const allowed = store.snapshot?.action.allowed.some((a) => a.name === "BIPED_CUSTOM_UPPER");
        ctl.append(h("p.muted.small", "进入「上肢控制」后可以直接拖动这个电机（腿部无力矩，先把机器人放平或悬挂）。"));
        actions.append(h("button.btn.sm", {
          disabled: !allowed,
          title: allowed ? "" : "需要先回到零力矩",
          onclick: () => opts.setAction?.("BIPED_CUSTOM_UPPER"),
        }, icon("bot", "sm"), "进入上肢控制"));
      }
    } else {
      ctl.append(h("p.muted.small", "腿部电机由站立 / 走路 / 滑板等策略统一控制，不支持单独下发角度。"));
    }
    actions.append(h("a.btn.sm.ghost", { href: "#/motion" }, "运动页", icon("external", "sm")));
    ctl.append(actions);
  }

  function piPanel(c, body) {
    const box = h("div.hw-live", h("p.muted.small", "读取中…"));
    body.append(box);
    const paint = () => {
      const i = live.info;
      if (!i) return;
      clear(box,
        h("div.hw-metrics",
          metric("CPU 温度", i.cpu_temp_c != null ? `${i.cpu_temp_c.toFixed(1)}°C` : "—", i.cpu_temp_c > 75 ? "warn" : ""),
          metric("负载", i.load ? i.load[0].toFixed(2) : "—"),
          metric("可用内存", i.mem_available_mb != null ? `${i.mem_available_mb} MB` : "—"),
          metric("磁盘剩余", i.disk_free_gb != null ? `${i.disk_free_gb} GB` : "—")),
        kv([
          ["主机名", i.hostname],
          ["IP", (i.ips || []).join("  ") || "—"],
          ["Wi-Fi", i.wifi_ssid || "—"],
          ["运行时长", duration(i.uptime_s)],
          ["版本", `API ${i.api_version} · ${i.git || "—"}`],
        ]));
    };
    panelTick = paint;
    loadInfo().then(paint);
    body.append(h("div.hw-actions",
      h("a.btn.sm", { href: "#/developer/logs" }, icon("terminal", "sm"), "日志"),
      h("button.btn.sm", { onclick: () => power("reboot") }, icon("refresh", "sm"), "重启"),
      h("button.btn.sm.danger", { onclick: () => power("shutdown") }, icon("power", "sm"), "关机")));
  }

  async function power(kind) {
    const ok = await confirmDialog({
      title: kind === "reboot" ? "重启主控？" : "关闭主控？",
      message: kind === "reboot" ? "控制循环会先停止并断开力矩，约 1 分钟后恢复。" : "控制循环会先停止并断开力矩。关机后需要手动重新上电。",
      okText: kind === "reboot" ? "重启" : "关机",
      danger: true,
    });
    if (!ok) return;
    api.post("/system/power", { action: kind }).then((r) => toast(r.message || "已发送")).catch((e) => toast(e.message, "error"));
  }

  function imuPanel(c, body) {
    const horizon = h("canvas.hw-horizon", { width: 240, height: 140 });
    const box = h("div.hw-live");
    body.append(horizon, box);
    const paint = () => {
      const i = live.imu;
      drawHorizon(horizon, i?.roll_deg, i?.pitch_deg);
      if (!i?.live) {
        clear(box, h("p.muted.small", i?.note || "读取中…"));
        return;
      }
      const g = (i.gyro || []).map((v) => `${((v * 180) / Math.PI).toFixed(1)}`);
      clear(box, h("div.hw-metrics",
        metric("横滚", `${i.roll_deg.toFixed(1)}°`, Math.abs(i.roll_deg) > 25 ? "warn" : ""),
        metric("俯仰", `${i.pitch_deg.toFixed(1)}°`, Math.abs(i.pitch_deg) > 25 ? "warn" : ""),
        metric("角速度 x/y/z", g.length ? `${g.join(" / ")} °/s` : "—")));
    };
    panelTick = paint;
    paint();
  }

  function batteryPanel(c, body) {
    const box = h("div.hw-live");
    body.append(box, h("div.hw-actions", h("a.btn.sm.ghost", { href: "#/settings/system" }, "电压阈值设置", icon("external", "sm"))));
    const paint = () => {
      const b = store.snapshot?.battery;
      if (!b?.voltage_v) {
        clear(box, h("p.muted.small", b?.note || "暂无读数"));
        return;
      }
      const tone = { critical: "danger", low: "warn", ok: "ok" }[b.level] || "";
      clear(box,
        h("div.hw-gauge", h(`div.fill.${tone}`, { style: { width: `${b.percent}%` } })),
        h("div.hw-metrics",
          metric("电量", `${b.percent}%`, tone === "ok" ? "" : tone),
          metric("电压", `${b.voltage_v.toFixed(2)} V`),
          metric("单节", `${(b.voltage_v / 2).toFixed(2)} V`)),
        b.measured_at ? h("small.muted", `测量于 ${timeAgo(b.measured_at)}`) : null);
    };
    panelTick = paint;
    paint();
  }

  function esp32Panel(c, body) {
    const box = h("div.hw-live", h("p.muted.small", "读取中…"));
    body.append(box, h("div.hw-actions",
      h("a.btn.sm", { href: "#/vision" }, icon("camera", "sm"), "摄像头"),
      h("a.btn.sm", { href: "#/interaction" }, icon("smile", "sm"), "表情与语音"),
      h("button.btn.sm", { onclick: () => api.post("/interaction/sound", { name: "chime", volume: 60 }).then(() => toast("已播放测试音")).catch((e) => toast(e.message, "error")) },
        icon("volume", "sm"), "测试扬声器")));
    const paint = () => {
      const i = live.info;
      if (!i) return;
      clear(box, kv([
        ["显示进程", h(`span.badge.${i.display_running ? "ok" : ""}`, i.display_running ? "运行中" : "未运行")],
        ["Wi-Fi 地址", i.esp32?.ip || i.esp32 || "—"],
        ["语音对话", store.snapshot?.voice?.running ? "进行中" : "未开启"],
      ]));
    };
    panelTick = paint;
    loadInfo().then(paint);
  }

  function screenPanel(c, body) {
    const grid = h("div.hw-faces");
    body.append(grid, h("div.hw-actions", h("a.btn.sm.ghost", { href: "#/interaction" }, "全部表情", icon("external", "sm"))));
    const paint = () => {
      const list = (store.expressions || []).slice(0, 12);
      const current = store.snapshot?.expression;
      clear(grid, list.map((e) => h(`button.btn.xs${e.name === current ? ".active" : ""}`, {
        onclick: () => api.post("/interaction/emotion", { name: e.name }).catch((err) => toast(err.message, "error")),
      }, e.label || e.name)));
    };
    panelTick = paint;
    paint();
  }

  function speakerPanel(c, body) {
    const grid = h("div.hw-faces", h("p.muted.small", "读取中…"));
    body.append(grid);
    api.get("/interaction/sounds").then((sounds) => clear(grid, sounds.map((s) => h("button.btn.xs", {
      title: s.label,
      onclick: () => api.post("/interaction/sound", { name: s.name, volume: 60 }).catch((e) => toast(e.message, "error")),
    }, icon("play", "sm"), s.label.replace(/（.*）/, ""))))).catch((e) => clear(grid, h("p.muted.small", e.message)));
  }

  function metric(label, value, tone = "") {
    return h(`div.hw-metric${tone ? `.${tone}` : ""}`, h("small", label), h("b", value));
  }

  // ------------------------------------------------------------------ data

  let infoAt = 0;
  async function loadInfo() {
    if (Date.now() - infoAt > 8000) {
      infoAt = Date.now();
      live.info = await api.get("/system/info").catch(() => live.info);
    }
    return live.info;
  }

  async function poll() {
    const wantsJoints = [...tags.values()].some((t) => t.c.kind === "motor") || twin.components.get(selected)?.kind === "motor";
    const running = ["BIPED_STAND_DEFAULT", "BIPED_WALK_RUN", "BIPED_WALK_UPPER", "BIPED_SKATE", "BIPED_GYMNASTICS", "BIPED_PEEK", "BIPED_PROPUP", "BIPED_CHILL_LIE", "BIPED_CHILL_SIT", "BIPED_CHILL_CHAIR", "BIPED_WAKE_STRETCH", "BIPED_CUSTOM_UPPER"].includes(action());
    if (wantsJoints && (running || Date.now() - live.jointsAt > 5000)) {
      live.jointsAt = Date.now();
      live.joints = await api.get("/hal/joint/state").catch(() => live.joints);
    }
    if (twin.components.get(selected)?.kind === "imu") live.imu = await api.get("/hal/imu/state").catch(() => live.imu);
    if ([...tags.values()].some((t) => t.c.kind === "pi")) await loadInfo();
    refreshValues();
    panelTick?.();
  }
  timers.push(setInterval(poll, 400));
  offs.push(on("status", () => {
    refreshValues();
    if (["battery", "screen"].includes(twin.components.get(selected)?.kind)) panelTick?.();
  }));
  // The upper-body control needs a rebuild when the mode changes.
  let lastMode = action();
  offs.push(on("status", () => {
    if (action() === lastMode) return;
    lastMode = action();
    const c = twin.components.get(selected);
    if (c?.kind === "motor") renderPanel(c);
  }));

  twin.onFrame = layout;
  rebuildTags();
  poll();

  return {
    toolbar,
    select,
    dispose() {
      twin.onFrame = null;
      offs.forEach((off) => off());
      timers.forEach(clearInterval);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      layer.remove();
      panel.remove();
    },
  };
}

function drawHorizon(canvas, roll, pitch) {
  const ctx = canvas.getContext("2d");
  const { width: w, height: hgt } = canvas;
  const css = getComputedStyle(document.documentElement);
  ctx.clearRect(0, 0, w, hgt);
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(0, 0, w, hgt, 10);
  ctx.clip();
  ctx.translate(w / 2, hgt / 2);
  ctx.rotate(((roll || 0) * Math.PI) / 180);
  const offset = (pitch || 0) * 2.2;
  ctx.fillStyle = "#3b82c4";
  ctx.fillRect(-w, -hgt * 2 + offset, w * 2, hgt * 2);
  ctx.fillStyle = "#7a5534";
  ctx.fillRect(-w, offset, w * 2, hgt * 2);
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(-w, offset);
  ctx.lineTo(w, offset);
  ctx.stroke();
  ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(255,255,255,0.8)";
  ctx.font = "10px system-ui";
  for (const p of [-20, -10, 10, 20]) {
    const y = offset - p * 2.2;
    ctx.beginPath();
    ctx.moveTo(-18, y);
    ctx.lineTo(18, y);
    ctx.stroke();
    ctx.fillText(String(p), 22, y + 3);
  }
  ctx.restore();
  ctx.strokeStyle = css.getPropertyValue("--warn").trim() || "#f59e0b";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(w / 2 - 44, hgt / 2);
  ctx.lineTo(w / 2 - 14, hgt / 2);
  ctx.lineTo(w / 2, hgt / 2 + 9);
  ctx.lineTo(w / 2 + 14, hgt / 2);
  ctx.lineTo(w / 2 + 44, hgt / 2);
  ctx.stroke();
  if (roll == null) {
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, 0, w, hgt);
  }
}
