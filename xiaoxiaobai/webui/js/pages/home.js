// Home: virtual robot (driven by commands) + chat with the robot.

import * as api from "../api.js";
import { chatPanel } from "../components/chat.js";
import { hardwareLayer } from "../components/hardware.js";
import { joystick } from "../components/joystick.js";
import { Twin } from "../components/twin.js";
import { ACTION_STYLE, action, isWalking, on, setVelocity, store } from "../store.js";
import { clear, h, icon, toast } from "../ui.js";

const QUICK = [
  { name: "BIPED_STAND_DEFAULT", label: "站立", icon: "arrowUp" },
  { name: "BIPED_WALK_RUN", label: "走跑", icon: "gamepad" },
  { name: "BIPED_WALK_UPPER", label: "上肢自由", icon: "bot" },
  { name: "BIPED_SKATE", label: "滑板", icon: "zap" },
  { name: "BIPED_GYMNASTICS", label: "广播体操", icon: "music" },
  { name: "BIPED_PEEK", label: "偷看", icon: "search", hint: "约 15 秒，身体斜出去探头张望，只靠双脚站稳" },
  { name: "BIPED_PROPUP", label: "撑起", icon: "arrowUp" },
  { name: "BIPED_CHILL_LIE", label: "躺着休闲", icon: "moon", hint: "仰面躺着时循环做放松的小动作" },
  { name: "BIPED_CHILL_SIT", label: "坐地休闲", icon: "sun", hint: "撑起坐着时晃脚、四处张望" },
  { name: "BIPED_CHILL_CHAIR", label: "坐凳休闲", icon: "sun", hint: "先把机器人放到约 11 cm 高的无靠背凳子上扶稳" },
  { name: "BIPED_WAKE_STRETCH", label: "起床伸懒腰", icon: "sun", hint: "仰面躺着时做一次约 9 秒：伸懒腰、全身发抖、放松，做完回到站立" },
  { name: "PASSIVE_DEFAULT", label: "零力矩", icon: "power" },
];

// 撑起/躺下: what the loop reports it is doing.
const PROPUP_PHASE = { rising: "撑起中", up: "已撑起", lowering: "躺下中", lying: "躺着", refused: "需要仰躺" };
// 休闲 modes: their phase, or the section of the clip playing.
const CHILL_PHASE = { easing: "进入姿势", settling: "回到起始姿势", settled: "回到起始姿势", refused: "姿势不对", done: "做完了" };

export async function setAction(name) {
  try {
    const res = await api.post("/mc/action", { action: name, source: "app_proxy" });
    toast(res.message || "已发送", res.state.reason === 16 ? "info" : "ok");
  } catch (error) {
    toast(error.message, "error", 6000);
  }
}

export default {
  title: "总览",
  fill: true,
  async mount(container) {
    const offs = [];
    const stage = h("div.twin-stage");
    const loading = h("div.twin-loading", h("span.spinner"), "正在加载 3D 模型…");
    const modeTag = h("div.twin-mode");
    const dock = h("div.twin-dock");
    const cards = h("div.status-cards");
    const stick = joystick({
      small: true,
      onChange: ({ x, y }) => setVelocity({ vx: y, vy: 0, vtheta: -x }, (e) => toast(e.message, "warn")),
    });
    const stickWrap = h("div.twin-stick", { hidden: true }, stick);
    const hwSlot = h("span.hw-slot");

    stage.append(
      loading,
      h("div.twin-overlay",
        h("div.title", h("b", "虚拟机器人"), h("small", "演示你发出的指令，不是真机实时姿态；点部件看详情")),
        h("div.actions",
          hwSlot,
          h("button.btn.sm.icon-only", { title: "回到原点", onclick: () => twin?.resetPosition() }, icon("target")),
          h("button.btn.sm.icon-only", { title: "重置视角", onclick: () => twin?.resetView() }, icon("refresh")))),
      modeTag,
      stickWrap,
    );

    const chat = chatPanel();
    // Phones show one of the two at a time (CSS hides the switch on wide screens).
    const home = h("div.home", h("div.card.twin-card", stage, dock, cards), h("div.card.chat-card", chat.el));
    const switcher = h("div.seg.home-switch");
    const showPane = (pane) => {
      sessionStorage.setItem("xiaoxiaobai.homePane", pane);
      home.classList.toggle("show-chat", pane === "chat");
      clear(switcher, [["robot", "bot", "机器人"], ["chat", "message", "对话"]].map(([id, ic, label]) =>
        h(`button${pane === id ? ".active" : ""}`, { type: "button", onclick: () => showPane(id) }, icon(ic, "sm"), label)));
    };
    showPane(sessionStorage.getItem("xiaoxiaobai.homePane") || "robot");
    container.append(switcher, home);

    let twin = new Twin(stage);
    let hw = null;
    twin.init().then(() => {
      loading.remove();
      if (twin.disposed) return;
      hw = hardwareLayer(stage, twin, { setAction });
      hwSlot.append(hw.toolbar);
    }).catch((error) => {
      console.error(error);
      clear(loading, icon("alert"), `3D 模型加载失败：${error.message}`);
      twin = null;
    });

    let lastSig = "";
    const render = () => {
      const s = store.snapshot;
      if (!s) return;
      const current = s.action.current_action;
      const allowed = new Set(s.action.allowed.map((a) => a.name));
      const target = s.action.target_action;
      const propup = s.action.propup;
      const sig = `${current.name}|${target}|${[...allowed].join(",")}|${s.action.head_motion}|${s.action.dance_pending}|${propup?.down}`;
      if (sig !== lastSig) lastSig = sig, clear(dock,
        h("div.group", QUICK.map((q) => {
          const isCurrent = current.name === q.name;
          const pending = (target === q.name || q.name === s.action.dance_pending) && !isCurrent;
          // One button for 撑起/躺下: it says what the next press does (the API toggles).
          const toggle = q.name === "BIPED_PROPUP" && isCurrent;
          const down = toggle && !propup?.down;
          const label = toggle ? (down ? "躺下" : "撑起") : q.label;
          const btn = h(`button.btn.sm${isCurrent ? ".active" : ""}${q.name === "PASSIVE_DEFAULT" ? ".danger" : ""}`, {
            disabled: !isCurrent && !allowed.has(q.name),
            title: !isCurrent && !allowed.has(q.name) ? `当前「${current.label}」不能直接切换到「${q.label}」`
              : q.name === "BIPED_PROPUP" ? "仰面躺着时撑起上身并保持，再按一次躺回去" : q.hint || q.label,
            onclick: () => (toggle || !isCurrent) && setAction(q.name),
          }, pending ? h("span.spinner") : icon(down ? "arrowDown" : q.icon, "sm"), label);
          return btn;
        })),
        h("span.spacer"),
        ["BIPED_STAND_DEFAULT", "BIPED_WALK_RUN", "BIPED_WALK_UPPER"].includes(current.name)
          ? h("button.btn.sm", {
            title: "做一个随机的拟人动作（会切到「走跑·上肢自由」）",
            onclick: () => api.post("/mc/gesture", { name: "random" }).then((r) => toast(r.message)).catch((e) => toast(e.message, "error")),
          }, icon("sparkles", "sm"), "随机动作")
          : null,
        ["BIPED_STAND_DEFAULT", "BIPED_WALK_RUN", "BIPED_SKATE"].includes(current.name)
          ? h(`button.btn.sm${s.action.head_motion ? ".active" : ""}`, {
            onclick: () => api.post("/mc/head", { enabled: !s.action.head_motion }).then((r) => toast(r.message)).catch((e) => toast(e.message, "error")),
          }, icon("rotateR", "sm"), "头部摆动")
          : null,
      );
      const style = ACTION_STYLE[current.name] || {};
      const dance = s.action.dance;
      const chill = s.action.chill;
      const progress = dance ? ` ${dance.t_s.toFixed(0)} / ${dance.duration_s.toFixed(0)} s`
        : propup?.phase ? ` · ${PROPUP_PHASE[propup.phase] || propup.phase}`
        : chill?.phase ? ` · ${CHILL_PHASE[chill.phase] || chill.section || ""}` : "";
      clear(modeTag, h(`span.chip.${style.tone || ""}`, h("span.dot"), current.label + progress));
      const walking = isWalking(current.name);
      stickWrap.hidden = !walking;
      stick.setEnabled(walking);

      const bat = s.battery;
      const voice = s.voice || {};
      const expr = store.expressions?.find((e) => e.name === s.expression);
      clear(cards,
        h("div.metric", h("small", "运动模式"), h("b", h(`span.dot.${style.tone || ""}`), current.label + progress)),
        h("div.metric", h("small", "电池"), h("b", bat.voltage_v ? `${bat.percent}%` : "—", bat.voltage_v ? h("small", ` ${bat.voltage_v.toFixed(2)} V`) : h("small", " 站立后读取"))),
        h("div.metric", h("small", "表情"), h("b", expr?.label || s.expression || "—")),
        h("div.metric", h("small", "语音对话"), h("b", h(`span.dot.${voice.running ? "busy" : ""}`), voice.running ? "进行中" : "未开启")),
      );
      twin?.update({ mode: current.name, head: s.action.head_motion, expression: s.expression, propup });
    };

    const pushVelocity = () => {
      const cmd = store.command;
      const v = Date.now() < cmd.walkUntil ? cmd.walkVelocity : cmd.velocity;
      twin?.update({ velocity: isWalking() ? v : { vx: 0, vy: 0, vtheta: 0 } });
    };
    const velocityTimer = setInterval(pushVelocity, 100);

    // Free upper body: show the robot's own arms and head, polled at the loop's status rate.
    let polling = false;
    const pollUpper = async () => {
      if (polling || !twin || action() !== "BIPED_WALK_UPPER") return;
      polling = true;
      try {
        const state = await api.get("/hal/joint/state");
        const live = {};
        for (const j of state.joints) {
          const value = j.position ?? j.target;
          if (j.motor_id >= 13 && value != null) live[j.name] = value;
        }
        twin?.update({ liveUpper: live });
      } catch {
        // the next poll retries
      } finally {
        polling = false;
      }
    };
    const upperTimer = setInterval(pollUpper, 200);

    offs.push(on("status", render), on("command", pushVelocity), on("theme", () => twin?.onTheme?.()));
    api.get("/interaction/emotions").then((list) => {
      store.expressions = list;
      render();
    }).catch(() => {});
    render();

    return () => {
      offs.forEach((off) => off());
      clearInterval(velocityTimer);
      clearInterval(upperTimer);
      setVelocity({ vx: 0, vy: 0, vtheta: 0 });
      chat.dispose();
      hw?.dispose();
      twin?.dispose();
    };
  },
};
