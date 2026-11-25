// Developer: API reference with try-it, live events, logs, async tasks.

import * as api from "../api.js";
import { on } from "../store.js";
import { clear, clock, copyText, empty, h, icon, select, switchEl, toast } from "../ui.js";

const STREAMING = new Set(["/api/v1/events", "/api/v1/hal/camera/stream", "/api/v1/chat"]);
const SECTIONS = [
  { id: "apis", label: "接口", icon: "code" },
  { id: "events", label: "事件流", icon: "activity" },
  { id: "logs", label: "日志", icon: "terminal" },
  { id: "tasks", label: "任务", icon: "clock" },
];

export default {
  title: "开发者",
  async mount(container, { args }) {
    let cleanupSection = null;
    const tabs = h("div.tabs");
    const body = h("div.stack");

    function show(section) {
      history.replaceState(null, "", `#/developer/${section.id}`);
      clear(tabs, SECTIONS.map((s) => h(`button${s.id === section.id ? ".active" : ""}`, { onclick: () => show(s) }, icon(s.icon, "sm"), s.label)));
      cleanupSection?.();
      clear(body);
      cleanupSection = { apis: apisSection, events: eventsSection, logs: logsSection, tasks: tasksSection }[section.id](body) || null;
    }

    container.append(h("div.settings", tabs, body));
    show(SECTIONS.find((s) => s.id === args[0]) || SECTIONS[0]);
    return () => cleanupSection?.();
  },
};

// =================================================================== APIs

function apisSection(root) {
  const origin = location.origin;
  const list = h("div.card-body.flush", h("div.empty", h("span.spinner")));
  root.append(
    h("div.card",
      h("div.card-head", h("h3", icon("code", "sm"), "机器人 API"), h("span.sub", "接口命名与返回结构对齐 PrimeBot SDK")),
      h("div.card-body.stack",
        h("dl.kv",
          h("dt", "Base URL"), h("dd.mono", `${origin}/api/v1`),
          h("dt", "返回结构"), h("dd.mono", "{ header: { code }, state: { value, name, reason, reason_text }, message, task_id, data }"),
          h("dt", "判断成功"), h("dd", "header.code == 0 表示请求被接受；state.value：1 成功，400 执行中（用 task_id 查询），2 失败，5 参数无效"),
          h("dt", "POST 要求"), h("dd", "Content-Type: application/json（防止其他网站的表单直接操控机器人）"),
          h("dt", "运动指令"), h("dd", "速度用 -1..1 比例；输入源超时后自动归零。自定义输入源优先级 0–59，低于手柄（80）和本页面（60）")),
        h("div.small.muted", "提示：服务只在局域网监听，没有鉴权；不要把端口暴露到公网。"))),
    h("div.card", list),
  );

  api.get("/meta/apis").then((routes) => {
    const groups = new Map();
    for (const r of routes) {
      if (!groups.has(r.group)) groups.set(r.group, []);
      groups.get(r.group).push(r);
    }
    clear(list, [...groups].map(([group, items]) => h("div.api-group", h("h4", group), items.map((r) => apiRow(r)))));
  }).catch((error) => clear(list, empty("alert", "读取失败", error.message)));

  function apiRow(r) {
    let panel = null;
    const row = h("div.api-row", {
      onclick: () => {
        if (panel) {
          panel.remove();
          panel = null;
          row.classList.remove("open");
          return;
        }
        row.classList.add("open");
        panel = tryPanel(r);
        row.after(panel);
      },
    }, h(`span.method.${r.method}`, r.method), h("span.path", r.path), h("span.name", r.name), h("span.summary", r.summary));
    return row;
  }

  function tryPanel(r) {
    const params = [...r.path.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);
    const paramInputs = Object.fromEntries(params.map((p) => [p, h("input.input.mono", { placeholder: p, style: { maxWidth: "220px" } })]));
    const hasBody = r.method === "POST" || r.method === "PATCH";
    const bodyInput = h("textarea.textarea.mono", { rows: 5, value: hasBody ? JSON.stringify(r.example ?? {}, null, 2) : "" });
    const queryInput = h("input.input.mono", { placeholder: "查询参数，例如 q=充电&k=3" });
    const output = h("div");
    const streaming = STREAMING.has(r.path);

    const resolvedPath = () => {
      let path = r.path;
      for (const p of params) path = path.replace(`{${p}}`, encodeURIComponent(paramInputs[p].value.trim() || p));
      const q = queryInput.value.trim();
      return q ? `${path}?${q}` : path;
    };
    const snippet = () => {
      const url = `${origin}${resolvedPath()}`;
      const data = hasBody ? bodyInput.value.trim() || "{}" : null;
      const curl = `curl ${streaming ? "-N " : ""}-X ${r.method} '${url}'${data ? ` \\\n  -H 'Content-Type: application/json' \\\n  -d '${data.replace(/\s*\n\s*/g, " ")}'` : ""}`;
      const py = `import requests\n\nr = requests.${r.method.toLowerCase()}("${url}"${data ? `, json=${data.replace(/\s*\n\s*/g, " ").replace(/\btrue\b/g, "True").replace(/\bfalse\b/g, "False").replace(/\bnull\b/g, "None")}` : ""}, timeout=10)\nresp = r.json()\nassert resp["header"]["code"] == 0, resp["message"]\nprint(resp["data"])`;
      return { curl, py };
    };

    async function send(button) {
      const started = performance.now();
      button.disabled = true;
      try {
        let parsed;
        if (hasBody) {
          try {
            parsed = JSON.parse(bodyInput.value || "{}");
          } catch (error) {
            toast(`请求体不是合法 JSON：${error.message}`, "error");
            return;
          }
        }
        const res = await fetch(resolvedPath(), {
          method: r.method,
          headers: hasBody ? { "Content-Type": "application/json" } : {},
          body: hasBody ? JSON.stringify(parsed) : undefined,
        });
        const text = await res.text();
        let pretty = text;
        try {
          pretty = JSON.stringify(JSON.parse(text), null, 2);
        } catch {
          /* not JSON */
        }
        clear(output, h("div.row.small", h(`span.badge.${res.ok ? "ok" : "danger"}`, `HTTP ${res.status}`), h("span.muted", `${Math.round(performance.now() - started)} ms`)),
          h("pre.code", { style: { marginTop: "8px" } }, pretty.length > 20000 ? `${pretty.slice(0, 20000)}\n…（已截断）` : pretty));
      } catch (error) {
        clear(output, h("div.test-result.fail", error.message));
      } finally {
        button.disabled = false;
      }
    }

    const codeView = h("pre.code");
    let lang = "curl";
    const renderCode = () => (codeView.textContent = snippet()[lang]);
    const langSeg = h("div.seg", ["curl", "py"].map((l) => h(`button${l === lang ? ".active" : ""}`, {
      onclick: (e) => {
        lang = l;
        langSeg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === e.currentTarget));
        renderCode();
      },
    }, l === "py" ? "Python" : "curl")));
    bodyInput.addEventListener("input", renderCode);
    queryInput.addEventListener("input", renderCode);
    Object.values(paramInputs).forEach((i) => i.addEventListener("input", renderCode));
    renderCode();

    return h("div.api-try",
      params.length ? h("div.row.wrap", params.map((p) => paramInputs[p])) : null,
      r.method === "GET" ? queryInput : null,
      hasBody ? bodyInput : null,
      h("div.row",
        streaming
          ? h("span.small.muted", "流式接口（SSE / MJPEG），请用下面的命令或浏览器直接打开。")
          : h("button.btn.sm.primary", { onclick: (e) => send(e.currentTarget) }, icon("play", "sm"), "发送请求"),
        h("span.spacer"), langSeg,
        h("button.btn.sm.ghost", { onclick: () => copyText(codeView.textContent).then(() => toast("已复制", "ok", 1500)) }, icon("copy", "sm"), "复制")),
      codeView,
      output);
  }
}

// =================================================================== events

function eventsSection(root) {
  let paused = false;
  let showStatus = false;
  const logEl = h("div.event-log");
  const add = (ev) => {
    if (paused || (!showStatus && ev.type === "status")) return;
    const { type, ts, ...rest } = ev;
    logEl.prepend(h("div", { title: JSON.stringify(ev) }, h("span.t", clock(ts || Date.now() / 1000)), h("span.k", type), JSON.stringify(rest)));
    while (logEl.children.length > 300) logEl.lastChild.remove();
  };
  root.append(h("div.card",
    h("div.card-head", h("h3", icon("activity", "sm"), "事件流"), h("span.sub.mono", "GET /api/v1/events（Server-Sent Events）"),
      h("div.actions",
        h("label.row.small", "包含 status", switchEl(false, (v) => (showStatus = v))),
        h("button.btn.sm", { onclick: (e) => {
          paused = !paused;
          e.currentTarget.textContent = paused ? "继续" : "暂停";
        } }, "暂停"),
        h("button.btn.sm.ghost", { onclick: () => clear(logEl) }, "清空"))),
    h("div.card-body", logEl, h("div.small.muted", { style: { marginTop: "10px" } },
      "事件类型：status（每秒一次的整体状态）、mode、action_request、alert、expression、voice、voice_user、voice_reply(_delta)、input_source、settings。"))));
  return on("*", add);
}

// =================================================================== logs

function logsSection(root) {
  let name = "run";
  let auto = true;
  const pre = h("pre.code.log", "加载中…");
  const lines = h("input.input", { type: "number", value: 300, min: 10, max: 2000, style: { width: "100px" } });

  async function load() {
    try {
      const data = await api.get(`/system/logs/${name}?lines=${lines.value}`);
      const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
      pre.textContent = data.text || "（空）";
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    } catch (error) {
      pre.textContent = error.message;
    }
  }

  const names = [["run", "run · 控制循环"], ["display", "display · 表情/音频/摄像头"], ["voice", "voice · 语音对话"], ["upper_body_slider", "upper_body_slider · 上肢"], ["api", "api · 本服务"]];
  root.append(h("div.card",
    h("div.card-head", h("h3", icon("terminal", "sm"), "日志"),
      h("div.actions",
        select(names, name, (v) => {
          name = v;
          pre.textContent = "加载中…";
          load().then(() => (pre.scrollTop = pre.scrollHeight));
        }),
        lines,
        h("label.row.small", "自动刷新", switchEl(auto, (v) => (auto = v))),
        h("button.btn.sm.icon-only", { title: "刷新", onclick: load }, icon("refresh", "sm")))),
    h("div.card-body", pre)));
  load().then(() => (pre.scrollTop = pre.scrollHeight));
  const timer = setInterval(() => auto && !document.hidden && load(), 2000);
  return () => clearInterval(timer);
}

// =================================================================== tasks

function tasksSection(root) {
  const bodyEl = h("div.card-body.flush");
  async function load() {
    try {
      const tasks = await api.get("/tasks");
      if (!tasks.length) {
        clear(bodyEl, empty("clock", "还没有异步任务", "切换运动模式、定时行走都会产生任务。"));
        return;
      }
      const tone = { SUCCESS: "ok", RUNNING: "accent", FAILURE: "danger", TIMEOUT: "warn", ABORTED: "" };
      clear(bodyEl, h("table.table",
        h("thead", h("tr", h("th", "ID"), h("th", "类型"), h("th", "参数"), h("th", "状态"), h("th", "说明"), h("th", "时间"))),
        h("tbody", tasks.map((t) => {
          const { task_id, kind, state, message, created, updated, ...rest } = t;
          return h("tr",
            h("td.mono", task_id),
            h("td.mono", kind),
            h("td.mono.small", Object.entries(rest).map(([k, v]) => `${k}=${typeof v === "number" ? +v.toFixed?.(2) : v}`).join(" ")),
            h("td", h(`span.badge.${tone[state] ?? ""}`, state)),
            h("td", message || "—"),
            h("td.small.muted", `${clock(created)} · ${(updated - created).toFixed(1)} s`));
        }))));
    } catch (error) {
      clear(bodyEl, empty("alert", "读取失败", error.message));
    }
  }
  root.append(h("div.card",
    h("div.card-head", h("h3", icon("clock", "sm"), "异步任务"), h("span.sub.mono", "GET /api/v1/tasks/{task_id}"),
      h("div.actions", h("button.btn.sm.icon-only", { onclick: load, title: "刷新" }, icon("refresh", "sm")))),
    bodyEl));
  load();
  const timer = setInterval(load, 2000);
  return () => clearInterval(timer);
}
