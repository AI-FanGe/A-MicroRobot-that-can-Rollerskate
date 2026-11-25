// Client for the robot API (/api/v1). Every endpoint answers with a CommonResponse:
// { header: { code }, state: { value, name, reason, reason_text }, message, task_id, data }.
// In the phone app, control commands go over Bluetooth (native.js); the rest over HTTP.

import * as native from "./native.js";

const BASE = "/api/v1";
const QUEUED = { header: { code: 0 }, state: { value: 1, name: "SUCCESS", reason: 0, reason_text: "" }, message: "", task_id: null, data: null };

export class ApiError extends Error {
  constructor(message, response = null, status = 0) {
    super(message);
    this.response = response;
    this.status = status;
  }
}

function checked(json, status = 200) {
  if (!json || !json.header || json.header.code !== 0) {
    throw new ApiError(json?.message || json?.state?.reason_text || `请求失败（HTTP ${status}）`, json, status);
  }
  return json;
}

async function viaBle(method, path, body, quiet, onError, timeout) {
  const json = await native.request(method, path, body, {
    quiet,
    timeout,
    onLateError: onError && ((res) => onError(new ApiError(res?.message || "请求失败", res))),
  });
  return json === null ? QUEUED : checked(json);
}

// `quiet` (Bluetooth only): don't wait for an answer; `onError` still hears a failure.
// `ble`: in the app, send over Bluetooth whenever it is up (calls that cut the robot's Wi-Fi).
export async function call(path, { method = "GET", body, signal, quiet = false, onError, ble = false, timeout } = {}) {
  if (native.bleReady() && (ble || native.prefersBle(method, path))) {
    try {
      return await viaBle(method, path, body, quiet, onError, timeout);
    } catch (error) {
      if (error instanceof ApiError) throw error;
      // the Bluetooth link failed: HTTP below
    }
  }
  const init = { method, signal, headers: {} };
  if (body !== undefined || method === "POST") {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body ?? {});
  }
  let res;
  try {
    res = await fetch(BASE + path, init);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    if (native.bleReady() && native.bleCapable(method, path)) return viaBle(method, path, body, false, null, timeout);
    throw new ApiError("无法连接机器人，请检查网络", null, 0);
  }
  let json = null;
  try {
    json = await res.json();
  } catch {
    throw new ApiError(`HTTP ${res.status}`, null, res.status);
  }
  return checked(json, res.status);
}

export async function get(path, opts) {
  return (await call(path, opts)).data;
}

export const post = (path, body, opts) => call(path, { ...opts, method: "POST", body });

// POST that answers with a file (audio, image); errors still come back as a CommonResponse.
export async function postBlob(path, body) {
  let res;
  try {
    res = await fetch(BASE + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
  } catch {
    throw new ApiError("无法连接机器人，请检查网络", null, 0);
  }
  if ((res.headers.get("Content-Type") || "").startsWith("application/json")) {
    const json = await res.json().catch(() => null);
    throw new ApiError(json?.message || `请求失败（HTTP ${res.status}）`, json, res.status);
  }
  if (!res.ok) throw new ApiError(`HTTP ${res.status}`, null, res.status);
  return res.blob();
}
export const patch = (path, body) => call(path, { method: "PATCH", body });
export const del = (path) => call(path, { method: "DELETE" });

// POST that answers with Server-Sent Events (chat). Yields parsed events.
export async function* stream(path, body, signal) {
  let res;
  try {
    res = await fetch(BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") return;
    throw new ApiError("无法连接机器人，请检查网络");
  }
  const type = res.headers.get("Content-Type") || "";
  if (!type.startsWith("text/event-stream")) {
    let json = null;
    try {
      json = await res.json();
    } catch {
      /* not JSON */
    }
    throw new ApiError(json?.message || `HTTP ${res.status}`, json, res.status);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const data = chunk.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
        if (data) yield JSON.parse(data);
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") throw error;
  } finally {
    reader.releaseLock();
  }
}

// Robot event stream with automatic reconnect. onConnection(connected, via): via is "wifi",
// or "ble" while the app carries the events over Bluetooth because Wi-Fi is down.
export function subscribe(onEvent, onConnection) {
  let source = null;
  let closed = false;
  let retry = null;
  let sseOpen = false;
  let bleOn = false;
  const useBle = (enabled) => {
    if (bleOn === enabled) return;
    bleOn = enabled;
    native.subscribe(enabled);
    if (enabled) onConnection(true, "ble");
  };
  const offs = native.inApp
    ? [
      native.on("event", (event) => bleOn && onEvent(event)),
      native.on("state", (link) => {
        if (sseOpen) return;
        if (link.state === "ready") useBle(true);
        else if (bleOn) {
          bleOn = false;
          onConnection(false, "ble");
        }
      }),
    ]
    : [];
  const open = () => {
    source = new EventSource(BASE + "/events");
    source.onopen = () => {
      sseOpen = true;
      useBle(false);
      onConnection(true, "wifi");
    };
    source.onmessage = (msg) => {
      try {
        onEvent(JSON.parse(msg.data));
      } catch (error) {
        console.warn("bad event", error);
      }
    };
    source.onerror = () => {
      sseOpen = false;
      if (native.bleReady()) useBle(true);
      else onConnection(false, "wifi");
      source.close();
      if (!closed) retry = setTimeout(open, 2000);
    };
  };
  open();
  return () => {
    closed = true;
    clearTimeout(retry);
    source?.close();
    offs.forEach((off) => off());
    if (bleOn) native.subscribe(false);
  };
}
