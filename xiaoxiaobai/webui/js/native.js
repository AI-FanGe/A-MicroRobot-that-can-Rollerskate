// Bridge to the 小小白 Android app. Its WebView injects window.XxbNative; in a browser nothing
// here is active and every call goes over HTTP as before.
//
// In the app, control commands go over the robot's Bluetooth link (see src/api/ble.py) and
// everything else (lists, chat, camera, audio, files) over HTTP on the Wi-Fi LAN. When Wi-Fi
// drops, requests that fit fall back to Bluetooth and the event stream follows.

const bridge = window.XxbNative || null;
export const inApp = !!bridge;

// Writes that steer the robot: small, frequent, and the ones that must not wait for Wi-Fi.
const CONTROL = /^\/(mc\/|interaction\/(emotion|sound|voice\/(start|stop)|heard)$|music\/(pause|resume|stop|next|previous|volume)$|memo\/(show|hide)$|system\/power$)/;
// Streams and files: HTTP only (server.py HTTP_ONLY).
const HTTP_ONLY = /^\/(events|chat|hal\/camera\/(stream|snapshot)|interaction\/voice\/preview|memo\/export|system\/walk_rec\/(file\/[^?]+|bundle))(\?|$)/;

const pending = new Map();
const late = new Map();
const handlers = { state: new Set(), event: new Set(), scan: new Set() };
let seq = 0;
let link = parse(bridge?.status?.()) || { state: inApp ? "idle" : "off" };

function parse(text) {
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    return null;
  }
}

if (inApp) {
  window.__xxb = {
    resolve(id, text) {
      const msg = parse(text);
      const p = pending.get(id);
      if (p) {
        pending.delete(id);
        clearTimeout(p.timer);
        if (msg?.error) p.reject(new Error(msg.error));
        else p.resolve(msg);
        return;
      }
      const l = late.get(id);
      if (l) {
        late.delete(id);
        l(msg?.r || msg);
      }
    },
    state(text) {
      link = parse(text) || link;
      handlers.state.forEach((fn) => fn(link));
    },
    event(text) {
      const event = parse(text);
      if (event) handlers.event.forEach((fn) => fn(event));
    },
    scan(text) {
      const found = parse(text);
      if (found) handlers.scan.forEach((fn) => fn(found));
    },
  };
  document.documentElement.classList.add("in-app");
  document.addEventListener("click", saveDownloads, true);
}

export function on(kind, fn) {
  handlers[kind].add(fn);
  return () => handlers[kind].delete(fn);
}

export const linkState = () => link;
export const bleReady = () => inApp && link.state === "ready";
export const prefersBle = (method, path) => method !== "GET" && CONTROL.test(path);
export const bleCapable = (method, path) => (method === "GET" || CONTROL.test(path)) && !HTTP_ONLY.test(path);

// One protocol message to the robot (t = req / pair / auth / sub); resolves with its answer.
export function send(msg, { timeout = 12000, quiet = false, onLateError } = {}) {
  return new Promise((resolve, reject) => {
    if (!inApp) return reject(new Error("不在 App 里"));
    const id = ++seq;
    const error = bridge.send(id, JSON.stringify(msg));
    if (error) return reject(new Error(error));
    if (quiet) {
      // Quiet requests are only answered when they fail.
      if (onLateError) {
        late.set(id, onLateError);
        setTimeout(() => late.delete(id), 3000);
      }
      return resolve(null);
    }
    pending.set(id, {
      resolve,
      reject,
      timer: setTimeout(() => {
        pending.delete(id);
        reject(new Error("蓝牙请求超时"));
      }, timeout),
    });
  });
}

// An API call over Bluetooth: resolves with the CommonResponse (null for quiet requests).
export async function request(method, path, body, opts = {}) {
  const res = await send({ t: "req", m: method, p: path, b: body ?? null, q: opts.quiet ? 1 : 0 }, opts);
  return res ? res.r : null;
}

export function subscribe(enabled) {
  return send({ t: "sub", on: !!enabled }).catch(() => null);
}

// GET a robot URL from native code (the app's bundled https pages can't fetch http).
// Resolves with { ok, status, ms, body?, error? }; never rejects.
export function httpGet(url, timeout = 2500) {
  return new Promise((resolve) => {
    if (!bridge?.httpCheck) return resolve({ ok: false, error: "不在 App 里" });
    const id = ++seq;
    const fail = (error) => resolve({ ok: false, error: error?.message || String(error) });
    pending.set(id, {
      resolve,
      reject: fail,
      timer: setTimeout(() => {
        pending.delete(id);
        fail("超时");
      }, timeout + 3000),
    });
    bridge.httpCheck(id, url, timeout);
  });
}

// Native helpers; each is a no-op in a browser.
export const call = (name, ...args) => (bridge && typeof bridge[name] === "function" ? bridge[name](...args) : null);
export const setTheme = (theme) => call("setTheme", theme);
export const haptic = (kind = "heavy") => call("haptic", kind);
export const openDevices = (route = "") => call("openSetup", route);

// <a download href="blob:…"> (camera snapshots) can't be downloaded by the WebView itself.
async function saveDownloads(event) {
  const a = event.target.closest?.("a[download]");
  if (!a || !/^(blob|data):/.test(a.href)) return;
  event.preventDefault();
  event.stopPropagation();
  try {
    const blob = await (await fetch(a.href)).blob();
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
    call("saveFile", a.getAttribute("download") || "download", blob.type || "application/octet-stream", data);
  } catch (error) {
    console.warn("save failed", error);
  }
}
