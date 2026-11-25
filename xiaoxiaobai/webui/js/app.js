// App shell: navigation, top status bar, emergency stop, event stream, page router.

import * as api from "./api.js";
import * as native from "./native.js";
import { ACTION_STYLE, emit, on, store } from "./store.js";
import { $, clear, h, icon, toast } from "./ui.js";

// `primary` pages sit in the phone's bottom bar; the others are under 更多.
const PAGES = [
  { path: "home", label: "首页", icon: "home", primary: true, load: () => import("./pages/home.js") },
  { path: "motion", label: "运动", icon: "gamepad", primary: true, load: () => import("./pages/motion.js") },
  { path: "interaction", label: "表情声音", icon: "smile", primary: true, load: () => import("./pages/interaction.js") },
  { path: "music", label: "音乐", icon: "music", primary: true, load: () => import("./pages/music.js") },
  { path: "computer", label: "电脑", icon: "monitor", primary: true, load: () => import("./pages/computer.js") },
  { path: "vision", label: "视觉", icon: "camera", load: () => import("./pages/vision.js") },
  { path: "memo", label: "备忘录", icon: "calendar", load: () => import("./pages/memo.js") },
  { path: "settings", label: "设置", icon: "settings", load: () => import("./pages/settings.js") },
  { path: "developer", label: "开发者", icon: "code", load: () => import("./pages/developer.js") },
];

let cleanup = null;
let routeToken = 0;

const currentPage = () => location.hash.replace(/^#\/?/, "").split("/")[0] || "home";

function renderNav() {
  const current = currentPage();
  const inMore = PAGES.some((p) => !p.primary && p.path === current);
  clear($("nav"),
    PAGES.map((p) => h(`a${p.path === current ? ".active" : ""}${p.primary ? "" : ".nav-secondary"}`, { href: `#/${p.path}` },
      icon(p.icon), h("span", p.label))),
    h(`button.nav-more${inMore ? ".active" : ""}`, { type: "button", onclick: openMore }, icon("layers"), h("span", "更多")));
}

// Phone layout: the pages that don't fit the bottom bar, the theme and (in the app) the robot's link.
function openMore() {
  const back = h("div.sheet-back");
  const close = () => back.remove();
  const current = currentPage();
  const theme = document.documentElement.dataset.theme;
  const row = (iconName, title, hint, onclick) =>
    h("button.sheet-row", { type: "button", onclick: () => (close(), onclick()) }, icon(iconName),
      h("span.text", h("b", title), hint ? h("small", hint) : null), icon("chevronR", "sm"));
  back.append(h("div.sheet", { role: "dialog", "aria-modal": "true" },
    h("div.sheet-grip"),
    h("div.sheet-grid", PAGES.filter((p) => !p.primary).map((p) =>
      h(`a.sheet-tile${p.path === current ? ".active" : ""}`, { href: `#/${p.path}`, onclick: close }, icon(p.icon, "lg"), h("span", p.label)))),
    h("div.sheet-list",
      native.inApp ? row("target", "跟随模式", "认出你的腿跟着走，熄屏也能继续", () => native.openDevices("#/follow")) : null,
      native.inApp ? row("wifi", "设备与网络", "切换机器人、蓝牙配对、给机器人配 Wi-Fi", () => native.openDevices("#/robot")) : null,
      native.inApp ? row("gamepad", "蓝牙遥控", "没有 Wi-Fi 时只用蓝牙控制动作和表情", () => native.openDevices("#/remote")) : null,
      row(theme === "dark" ? "sun" : "moon", theme === "dark" ? "浅色模式" : "深色模式", null, toggleTheme)),
    h("div.sheet-foot", h("span", "API"), h("span.mono", store.apiVersion ? `v${store.apiVersion}` : "—"))));
  back.addEventListener("click", (event) => event.target === back && close());
  document.body.append(back);
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("xiaoxiaobai.theme", next);
  native.setTheme(next);
  emit("theme", { theme: next });
  renderFoot();
}

async function route() {
  const [name, ...rest] = (location.hash.replace(/^#\/?/, "") || "home").split("/");
  const page = PAGES.find((p) => p.path === name) || PAGES[0];
  const token = ++routeToken;
  renderNav();
  cleanup?.();
  cleanup = null;
  const container = $("page");
  container.className = "page";
  const spinner = h("div.empty", h("span.spinner"), "加载中…");
  clear(container, spinner);
  $("pageTitle").textContent = page.label === "首页" ? "总览" : page.label;
  try {
    const mod = (await page.load()).default;
    if (token !== routeToken) return;
    $("pageTitle").textContent = mod.title || page.label;
    container.className = `page${mod.fill ? " fill" : ""}`;
    spinner.hidden = !!mod.fill;
    // A page that is still mounting when the user navigates away writes into a detached host.
    const host = h("div", { style: { display: "contents" } });
    container.append(host);
    const dispose = (await mod.mount(host, { args: rest })) || null;
    spinner.remove();
    if (token !== routeToken) {
      dispose?.();
      return;
    }
    cleanup = dispose;
  } catch (error) {
    console.error(error);
    clear(container, h("div.empty", icon("alert"), h("b", "页面加载失败"), h("div.small", String(error.message || error))));
  }
}

// ---------------------------------------------------------------- status bar

function renderStatus() {
  const s = store.snapshot;
  const strip = $("statusStrip");
  if (!s) {
    clear(strip, h("span.chip.busy", h("span.dot"), "连接中…"));
    return;
  }
  const act = s.action.current_action;
  const style = ACTION_STYLE[act.name] || {};
  const target = s.action.target_action;
  const switching = target && target !== act.name;
  const bat = s.battery;
  const batTone = bat.level === "critical" ? "danger" : bat.level === "low" ? "warn" : bat.voltage_v ? "ok" : "";
  const voice = s.voice || {};
  const music = s.music || {};
  const voiceTone = voice.running ? (voice.state === "error" ? "danger" : "busy") : "";
  const viaBle = store.connected && store.link === "ble";
  clear(strip,
    h(`span.chip.${!store.connected ? "danger" : viaBle ? "warn" : "ok"}.optional`, h("span.dot"),
      !store.connected ? "已断开" : viaBle ? "仅蓝牙" : s.mock ? "模拟模式" : "已连接"),
    native.inApp ? bleChip() : null,
    h(`span.chip.${switching ? "busy" : style.tone || ""}`, { title: `${act.name}（${act.value}）` },
      h("span.dot"), "模式 ", h("b", act.label),
      switching ? h("span.muted", ` → ${labelOf(target)}`) : null),
    h(`span.chip.${batTone}`, { title: bat.note || `${bat.voltage_v?.toFixed(2)} V` }, icon("battery", "sm"),
      bat.voltage_v ? h("b", `${bat.percent}%`) : h("span.muted", "—"),
      bat.voltage_v ? h("span.muted", `${bat.voltage_v.toFixed(2)}V`) : null),
    s.input_source ? h("span.chip.optional", icon("gamepad", "sm"), "控制源 ", h("b", s.input_source)) : null,
    voice.running ? h(`span.chip.${voiceTone}.optional`, h("span.dot"), "语音 ", h("b", VOICE_STATE[voice.state] || voice.state)) : null,
    music.track && ["loading", "playing", "paused"].includes(music.state)
      ? h(`a.chip.${music.state === "playing" ? "ok" : "busy"}.optional`, { href: "#/music", title: music.track.artist },
        icon(music.state === "paused" ? "pause" : "music", "sm"), h("b", music.track.name))
      : null,
  );
  $("brandName").textContent = s.robot_name || "小小白";
  document.title = `${s.robot_name || "小小白"} 控制台`;
}

const VOICE_STATE = { starting: "连接中", ready: "聆听中", thinking: "思考中", speaking: "说话中", idle: "空闲", error: "出错" };
const LINK_STATE = { ready: "已连接", connecting: "连接中", pairing: "待配对", scanning: "搜索中", idle: "未连接", off: "已关闭", denied: "无权限" };

function bleChip() {
  const link = native.linkState();
  const tone = link.state === "ready" ? "ok" : ["connecting", "scanning"].includes(link.state) ? "busy" : "";
  return h(`button.chip.${tone || "plain"}.chip-btn`, { type: "button", title: link.error || "蓝牙控制通道", onclick: () => native.openDevices("#/robot") },
    h("span.dot"), "蓝牙 ", h("b", LINK_STATE[link.state] || link.state));
}

function labelOf(name) {
  const allowed = store.snapshot?.action?.allowed || [];
  return allowed.find((a) => a.name === name)?.label || ACTION_STYLE[name]?.short || name;
}

function renderBanner() {
  const s = store.snapshot;
  const banner = $("banner");
  banner.className = "banner";
  if (!store.connected) {
    banner.classList.add("danger");
    clear(banner, icon("wifi"), "与机器人的连接已断开，正在重连…",
      native.inApp ? h("button.btn.sm", { onclick: () => native.openDevices("#/robot") }, "设备与网络") : null);
    banner.hidden = false;
  } else if (store.link === "ble") {
    clear(banner, icon("wifi"), "Wi-Fi 连不上机器人，已改用蓝牙：动作和表情照常，画面、对话和列表要等 Wi-Fi 恢复",
      h("button.btn.sm", { onclick: () => native.openDevices("#/wifi") }, "配置 Wi-Fi"));
    banner.hidden = false;
  } else if (s?.battery?.level === "critical") {
    banner.classList.add("danger");
    clear(banner, icon("battery"), `电池电压 ${s.battery.voltage_v.toFixed(2)} V，已禁止走路和做动作，请立即充电`);
    banner.hidden = false;
  } else if (s?.action?.last_stop && !s.action.last_stop.expected && Date.now() / 1000 - s.action.last_stop.ts < 60) {
    clear(banner, icon("alert"), `控制循环意外退出：${s.action.last_stop.text}`,
      h("a.btn.sm", { href: "#/developer/logs" }, "查看日志"));
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }
}

function renderFoot() {
  const theme = document.documentElement.dataset.theme;
  clear($("sidebarFoot"),
    h("button.btn.ghost.sm", { onclick: toggleTheme }, icon(theme === "dark" ? "sun" : "moon", "sm"), theme === "dark" ? "浅色模式" : "深色模式"),
    h("div.row", h("span", "API"), h("span.mono", store.apiVersion ? `v${store.apiVersion}` : "—")),
  );
}

// ---------------------------------------------------------------- emergency stop

async function estop() {
  native.haptic("heavy");
  try {
    const res = await api.post("/mc/estop", { source: "app_proxy" });
    toast(res.message || "已急停", "warn");
  } catch (error) {
    toast(`急停失败：${error.message}`, "error", 8000);
  }
}

let lastEsc = 0;
function bindShortcuts() {
  $("estopBtn").addEventListener("click", estop);
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || document.querySelector(".modal-back")) return;
    if (Date.now() - lastEsc < 600) {
      lastEsc = 0;
      estop();
    } else {
      lastEsc = Date.now();
    }
  });
}

// ---------------------------------------------------------------- events

function handleEvent(event) {
  switch (event.type) {
    case "hello":
      store.apiVersion = event.api_version;
      store.snapshot = event.snapshot;
      renderFoot();
      break;
    case "status": {
      const { type, ts, ...snap } = event;
      store.snapshot = snap;
      break;
    }
    case "alert":
      toast(event.message, event.level === "error" ? "error" : event.level === "warning" ? "warn" : "info", 7000);
      break;
    case "settings":
      store.settings = null;
      break;
    default:
      break;
  }
  if (event.type === "hello" || event.type === "status") {
    renderStatus();
    renderBanner();
    emit("status", store.snapshot);
  } else {
    emit(event.type, event);
  }
}

function connect() {
  api.subscribe(handleEvent, (connected, via) => {
    if (store.connected === connected && store.link === via) return;
    store.connected = connected;
    store.link = via;
    renderStatus();
    renderBanner();
    emit("connection", { connected, via });
  });
  native.on("state", () => {
    renderStatus();
    renderBanner();
  });
}

// ---------------------------------------------------------------- boot

on("navigate", ({ path }) => {
  location.hash = `#/${path}`;
});

window.addEventListener("hashchange", route);
native.setTheme(document.documentElement.dataset.theme);
renderNav();
renderStatus();
renderFoot();
bindShortcuts();
connect();
route();
