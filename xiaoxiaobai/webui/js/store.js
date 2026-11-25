// Shared state fed by the event stream, plus a tiny pub/sub.

import * as api from "./api.js";

const listeners = new Map();

export const store = {
  connected: false,
  snapshot: null,
  expressions: null,
  settings: null,
  // What the UI last commanded; drives the virtual robot (it does not mirror live sensor data).
  command: { velocity: { vx: 0, vy: 0, vtheta: 0 }, upper: {}, walkUntil: 0, walkVelocity: null },
};

export function on(type, fn) {
  if (!listeners.has(type)) listeners.set(type, new Set());
  listeners.get(type).add(fn);
  return () => listeners.get(type)?.delete(fn);
}

export function emit(type, data) {
  for (const fn of listeners.get(type) || []) {
    try {
      fn(data);
    } catch (error) {
      console.error(error);
    }
  }
  for (const fn of listeners.get("*") || []) fn({ type, ...data });
}

export const action = () => store.snapshot?.action?.current_action?.name || "PASSIVE_DEFAULT";
export const targetAction = () => store.snapshot?.action?.target_action || action();

export async function loadExpressions() {
  if (!store.expressions) store.expressions = await api.get("/interaction/emotions");
  return store.expressions;
}

export async function loadSettings(force = false) {
  if (!store.settings || force) store.settings = await api.get("/settings");
  return store.settings;
}

export const ACTION_STYLE = {
  PASSIVE_DEFAULT: { tone: "", short: "零力矩" },
  STOPPING: { tone: "busy", short: "停止中" },
  STARTING: { tone: "busy", short: "启动中" },
  BIPED_STAND_DEFAULT: { tone: "ok", short: "站立" },
  BIPED_WALK_RUN: { tone: "ok", short: "走跑" },
  BIPED_CUSTOM_UPPER: { tone: "warn", short: "上肢" },
  BIPED_WALK_UPPER: { tone: "ok", short: "走跑·上肢" },
  BIPED_SKATE: { tone: "ok", short: "滑板" },
  BIPED_GYMNASTICS: { tone: "busy", short: "广播体操" },
  BIPED_PEEK: { tone: "busy", short: "偷看" },
  BIPED_PROPUP: { tone: "busy", short: "撑起/躺下" },
  BIPED_CHILL_LIE: { tone: "busy", short: "躺着休闲" },
  BIPED_CHILL_SIT: { tone: "busy", short: "坐地休闲" },
  BIPED_CHILL_CHAIR: { tone: "busy", short: "坐凳休闲" },
  BIPED_WAKE_STRETCH: { tone: "busy", short: "起床伸懒腰" },
};

// Raw motor angles for BIPED_CUSTOM_UPPER (upper_body_slider.MOTOR_LIMITS_DEG).
export const UPPER_MOTORS = [
  { id: 13, label: "右肩前后", min: -150, max: 150 },
  { id: 14, label: "右肩侧摆", min: 0, max: 160 },
  { id: 15, label: "右肘", min: 0, max: 85 },
  { id: 16, label: "左肩前后", min: -150, max: 150 },
  { id: 17, label: "左肩侧摆", min: -160, max: 0 },
  { id: 18, label: "左肘", min: -85, max: 0 },
  { id: 19, label: "头部左右", min: -40, max: 40 },
  { id: 20, label: "头部俯仰", min: -10, max: 30 },
];

// Modes in which a locomotion policy runs and velocity commands are taken.
export const WALKING_ACTIONS = new Set(["BIPED_WALK_RUN", "BIPED_WALK_UPPER", "BIPED_SKATE"]);
export const isWalking = (name = action()) => WALKING_ACTIONS.has(name);

// Velocity sender shared by every joystick: repeats at 8 Hz while non-zero (the API's
// input source watchdog drops commands older than 1 s), then sends one final zero.
let velocityTimer = null;
let lastSent = 0;
let velocityError = 0;

export function setVelocity(v, onError) {
  store.command.velocity = { vx: v.vx || 0, vy: v.vy || 0, vtheta: v.vtheta || 0 };
  emit("command", { velocity: store.command.velocity });
  const moving = Object.values(store.command.velocity).some((x) => Math.abs(x) > 0.01);
  const report = (error) => {
    if (Date.now() - velocityError > 3000) {
      velocityError = Date.now();
      onError?.(error);
    }
  };
  const send = () => {
    lastSent = Date.now();
    api.post("/mc/locomotion/velocity", { source: "app_proxy", ...store.command.velocity }, { quiet: true, onError: report })
      .catch(report);
  };
  if (!isWalking()) return;
  if (moving && !velocityTimer) {
    send();
    velocityTimer = setInterval(send, 125);
  } else if (!moving && velocityTimer) {
    clearInterval(velocityTimer);
    velocityTimer = null;
    send();
  } else if (Date.now() - lastSent > 60) {
    send();
  }
}
