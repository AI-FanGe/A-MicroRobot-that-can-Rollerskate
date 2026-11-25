// Small DOM toolkit: element builder, icons, toasts, dialogs, form controls, formatters.

const PROPS = new Set(["value", "checked", "disabled", "selected", "hidden", "textContent", "indeterminate", "multiple", "readOnly"]);

export function h(selector, attrs, ...children) {
  const [tag, ...parts] = selector.split(/(?=[.#])/);
  const el = document.createElement(tag || "div");
  for (const part of parts) {
    if (part.length < 2) continue;
    if (part[0] === ".") el.classList.add(part.slice(1));
    else el.id = part.slice(1);
  }
  if (attrs == null || typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs)) {
    children.unshift(attrs);
    attrs = {};
  }
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "class") el.className = `${el.className} ${value}`.trim();
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "html") el.innerHTML = value;
    else if (PROPS.has(key)) el[key] = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children) {
    if (child == null || child === false || child === true) continue;
    if (Array.isArray(child)) append(el, child);
    else el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function clear(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

export const $ = (id) => document.getElementById(id);

// Icons drawn in the Lucide style (24×24, stroked).
const ICONS = {
  home: '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>',
  gamepad: '<path d="M6 11h4M8 9v4M15 12h.01M18 10h.01"/><path d="M17.32 5H6.68a4 4 0 0 0-3.98 3.59C2.6 9.42 2 14.46 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.41-1.41A2 2 0 0 1 9.83 16h4.34a2 2 0 0 1 1.41.59L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.55-.6-6.58-.69-7.26A4 4 0 0 0 17.32 5z"/>',
  smile: '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2M9 9h.01M15 9h.01"/>',
  camera: '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3z"/><circle cx="12" cy="13" r="3"/>',
  settings: '<path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3M14 2v4M8 10v4M16 18v4"/>',
  code: '<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',
  send: '<path d="m22 2-7 20-4-9-9-4z"/><path d="M22 2 11 13"/>',
  mic: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v3"/>',
  stop: '<rect width="14" height="14" x="5" y="5" rx="2"/>',
  plus: '<path d="M5 12h14M12 5v14"/>',
  trash: '<path d="M3 6h18M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>',
  edit: '<path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5z"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  refresh: '<path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/>',
  battery: '<rect width="16" height="10" x="2" y="7" rx="2"/><path d="M22 11v2"/>',
  power: '<path d="M12 2v10"/><path d="M18.4 6.6a9 9 0 1 1-12.77.04"/>',
  zap: '<path d="M13 2 3 14h9l-1 8 10-12h-9z"/>',
  book: '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
  sparkles: '<path d="M12 3l1.9 5.8L20 10.5l-6.1 1.7L12 18l-1.9-5.8L4 10.5l6.1-1.7z"/><path d="M19 3v4M17 5h4"/>',
  message: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
  play: '<path d="M6 3l14 9-14 9z"/>',
  pause: '<rect width="4" height="16" x="6" y="4" rx="1"/><rect width="4" height="16" x="14" y="4" rx="1"/>',
  skipF: '<path d="m5 4 10 8-10 8z"/><path d="M19 5v14"/>',
  skipB: '<path d="m19 20-10-8 10-8z"/><path d="M5 19V5"/>',
  music: '<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>',
  phone: '<rect width="14" height="20" x="5" y="2" rx="2"/><path d="M12 18h.01"/>',
  logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  pin: '<path d="M12 17v5M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24z"/>',
  alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4M12 17h.01"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  cpu: '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
  user: '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
  external: '<path d="M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
  volume: '<path d="M11 5 6 9H2v6h4l5 4z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07M19.07 4.93a10 10 0 0 1 0 14.14"/>',
  bot: '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2M20 14h2M15 13v2M9 13v2"/>',
  target: '<circle cx="12" cy="12" r="10"/><path d="M22 12h-4M6 12H2M12 6V2M12 22v-4"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  wifi: '<path d="M12 20h.01M2 8.82a15 15 0 0 1 20 0M5 12.86a10 10 0 0 1 14 0M8.5 16.43a5 5 0 0 1 7 0"/>',
  activity: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
  terminal: '<path d="m4 17 6-6-6-6M12 19h8"/>',
  monitor: '<rect width="20" height="14" x="2" y="3" rx="2"/><path d="M8 21h8M12 17v4"/>',
  copy: '<rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
  clock: '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
  calendar: '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4M8 2v4M3 10h18M8 14h.01M12 14h.01M16 14h.01M8 18h.01M12 18h.01"/>',
  chevronL: '<path d="m15 18-6-6 6-6"/>',
  chevronR: '<path d="m9 18 6-6-6-6"/>',
  arrowUp: '<path d="m5 12 7-7 7 7M12 19V5"/>',
  arrowDown: '<path d="M12 5v14M19 12l-7 7-7-7"/>',
  rotateL: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
  rotateR: '<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>',
  moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
  file: '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/>',
  shield: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
  layers: '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65M22 12.65l-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
  bluetooth: '<path d="m7 7 10 10-5 5V2l5 5L7 17"/>',
  lock: '<rect width="18" height="11" x="3" y="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
};

export function icon(name, cls = "") {
  return h(`span.icon${cls ? "." + cls.split(" ").join(".") : ""}`, {
    html: `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ICONS.info}</svg>`,
  });
}

// ---- toasts ----
const TOAST_ICON = { ok: "check", error: "alert", warn: "alert", warning: "alert", info: "info" };

export function toast(message, level = "info", ms = 3800) {
  const el = h(`div.toast.${level}`, icon(TOAST_ICON[level] || "info"), h("div", message));
  document.getElementById("toasts").append(el);
  const close = () => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 220);
  };
  el.addEventListener("click", close);
  setTimeout(close, ms);
  return el;
}

// ---- modal ----
export function modal({ title, body, footer, wide = false, onClose } = {}) {
  const back = h("div.modal-back");
  const close = () => {
    back.remove();
    document.removeEventListener("keydown", onKey);
    onClose?.();
  };
  const onKey = (event) => {
    if (event.key === "Escape") close();
  };
  const dialog = h(
    `div.modal${wide ? ".wide" : ""}`,
    { role: "dialog", "aria-modal": "true" },
    h("div.modal-head", h("h3", title), h("button.btn.ghost.sm.icon-only", { onclick: close, title: "关闭" }, icon("x"))),
    h("div.modal-body", body),
    footer ? h("div.modal-foot", typeof footer === "function" ? footer(close) : footer) : null,
  );
  back.append(dialog);
  back.addEventListener("mousedown", (event) => {
    if (event.target === back) close();
  });
  document.addEventListener("keydown", onKey);
  document.body.append(back);
  return { close, el: dialog };
}

export function confirmDialog({ title = "确认", message, okText = "确定", danger = false }) {
  return new Promise((resolve) => {
    let result = false;
    const m = modal({
      title,
      body: h("div", { style: { whiteSpace: "pre-wrap" } }, message),
      footer: (close) => [
        h("button.btn", { onclick: close }, "取消"),
        h(`button.btn.${danger ? "danger" : "primary"}`, {
          onclick: () => {
            result = true;
            close();
          },
        }, okText),
      ],
      onClose: () => resolve(result),
    });
    m.el.querySelector(".modal-foot .btn:last-child")?.focus();
  });
}

// ---- busy button ----
export async function busy(button, task) {
  if (!button) return task();
  const original = [...button.childNodes];
  button.disabled = true;
  button.replaceChildren(h("span.spinner"), ...original.filter((n) => n.nodeType === 3));
  try {
    return await task();
  } finally {
    button.disabled = false;
    button.replaceChildren(...original);
  }
}

// ---- form controls ----
export function switchEl(checked, onchange, { disabled = false } = {}) {
  const input = h("input", { type: "checkbox", checked: !!checked, disabled, onchange: (e) => onchange(e.target.checked) });
  return h("label.switch", input, h("span"));
}

export function toggleRow(title, hint, checked, onchange, opts) {
  return h("div.toggle-row", h("div.text", h("b", title), hint ? h("small", hint) : null), switchEl(checked, onchange, opts));
}

export function field(label, control, hint) {
  return h("div.field", h("label", label), control, hint ? h("div.hint", hint) : null);
}

export function rangeEl({ min, max, step = 1, value, format = (v) => v, oninput, onchange }) {
  const out = h("output", format(value));
  const input = h("input", {
    type: "range", min, max, step, value,
    oninput: (e) => {
      out.textContent = format(Number(e.target.value));
      oninput?.(Number(e.target.value));
    },
    onchange: (e) => onchange?.(Number(e.target.value)),
  });
  const wrap = h("div.range", input, out);
  wrap.input = input;
  wrap.set = (v) => {
    input.value = v;
    out.textContent = format(Number(v));
  };
  return wrap;
}

export function select(options, value, onchange, cls = "") {
  return h(`select.select${cls}`, { onchange: (e) => onchange?.(e.target.value) },
    options.map((o) => {
      const [v, label] = Array.isArray(o) ? o : [o, o];
      return h("option", { value: v, selected: String(v) === String(value) }, label);
    }));
}

export function empty(iconName, title, text, action) {
  return h("div.empty", icon(iconName), h("b", title), text ? h("div.small", text) : null, action || null);
}

// ---- formatting ----
export function timeAgo(ts) {
  if (!ts) return "—";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 10) return "刚刚";
  if (s < 60) return `${Math.floor(s)} 秒前`;
  if (s < 3600) return `${Math.floor(s / 60)} 分钟前`;
  if (s < 86400) return `${Math.floor(s / 3600)} 小时前`;
  return new Date(ts * 1000).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

export function clock(ts) {
  return new Date(ts * 1000).toLocaleTimeString("zh-CN", { hour12: false });
}

export function dateTime(ts) {
  return new Date(ts * 1000).toLocaleString("zh-CN", { hour12: false, month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function duration(s) {
  if (s == null) return "—";
  s = Math.floor(s);
  const d = Math.floor(s / 86400);
  const hh = Math.floor((s % 86400) / 3600);
  const mm = Math.floor((s % 3600) / 60);
  if (d) return `${d} 天 ${hh} 小时`;
  if (hh) return `${hh} 小时 ${mm} 分`;
  if (mm) return `${mm} 分 ${s % 60} 秒`;
  return `${s} 秒`;
}

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Minimal Markdown for chat replies: code blocks, inline code, bold, headings, lists, [n] citations.
export function markdown(text, sources = []) {
  const blocks = [];
  let src = escapeHtml(text).replace(/```(\w*)\n?([\s\S]*?)(```|$)/g, (_, lang, code) => {
    blocks.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  const inline = (s) =>
    s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/\[(\d{1,2})\]/g, (m, n) => {
        const item = sources[Number(n) - 1];
        return item ? `<span class="cite" title="${escapeHtml(item.title)}">${n}</span>` : m;
      });
  const out = [];
  let list = null;
  for (const line of src.split("\n")) {
    const bullet = line.match(/^\s*(?:[-*•]|(\d+)[.、)])\s+(.*)$/);
    if (bullet) {
      const kind = bullet[1] ? "ol" : "ul";
      if (!list || list.kind !== kind) {
        if (list) out.push(`</${list.kind}>`);
        list = { kind };
        out.push(`<${kind}>`);
      }
      out.push(`<li>${inline(bullet[2])}</li>`);
      continue;
    }
    if (list) {
      out.push(`</${list.kind}>`);
      list = null;
    }
    const heading = line.match(/^#{1,4}\s+(.*)$/);
    if (heading) out.push(`<h4>${inline(heading[1])}</h4>`);
    else if (/^\u0000\d+\u0000$/.test(line.trim())) out.push(line.trim());
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push(`</${list.kind}>`);
  return out.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

// At most one call per `ms` while called continuously (e.g. a slider drag), the last
// arguments always delivered: the robot follows the drag instead of waiting for a pause.
export function throttle(fn, ms) {
  let last = 0;
  let timer = null;
  let pending = null;
  const run = () => {
    timer = null;
    last = Date.now();
    const args = pending;
    pending = null;
    fn(...args);
  };
  return (...args) => {
    pending = args;
    if (!timer) timer = setTimeout(run, Math.max(0, last + ms - Date.now()));
  };
}

export function copyText(text) {
  if (window.XxbNative?.copy) {
    window.XxbNative.copy(String(text));
    return Promise.resolve();
  }
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const ta = h("textarea", { value: text, style: { position: "fixed", opacity: "0" } });
  document.body.append(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
  return Promise.resolve();
}
