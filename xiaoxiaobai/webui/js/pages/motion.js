// Motion control: McAction state machine, driving, telemetry, input sources, upper body.

import * as api from "../api.js";
import * as native from "../native.js";
import { joystick } from "../components/joystick.js";
import { moveEditor } from "../components/move-editor.js";
import { action, emit, isWalking, on, setVelocity, store, UPPER_MOTORS } from "../store.js";
import { clear, empty, h, icon, rangeEl, switchEl, throttle, timeAgo, toast, toggleRow } from "../ui.js";
import { setAction } from "./home.js";

const UPPER_LABEL = Object.fromEntries(UPPER_MOTORS.map((m) => [m.id, m.label]));
const GESTURE_FROM = new Set(["BIPED_STAND_DEFAULT", "BIPED_WALK_RUN", "BIPED_WALK_UPPER"]);
const KEYS = { w: [1, 0, 0], s: [-1, 0, 0], a: [0, 0, 1], d: [0, 0, -1], q: [0, 1, 0], e: [0, -1, 0] };
const REVERT_MS = 10000;

// A slider that only takes effect after「保存」; an unsaved drag snaps back to the saved value.
// onsave(v) may return a promise; resolving to false keeps the old value.
function savedRange({ value, format = (v) => v, onsave, ...opts }) {
  let saved = value;
  let dirty = false;
  let timer = 0;
  const hint = h("span.small.muted");
  const btn = h("button.btn.xs.primary", { disabled: true, onclick: () => save() }, "保存");
  const range = rangeEl({ ...opts, value, format, oninput: () => markDirty() });
  const current = () => Number(range.input.value);
  const settle = () => {
    dirty = false;
    clearTimeout(timer);
    btn.disabled = true;
    hint.textContent = "";
  };
  const revert = () => {
    range.set(saved);
    settle();
  };
  function markDirty() {
    dirty = current() !== saved;
    clearTimeout(timer);
    btn.disabled = !dirty;
    hint.textContent = dirty ? `未保存，${REVERT_MS / 1000} 秒内不点保存会恢复为 ${format(saved)}` : "";
    if (dirty) timer = setTimeout(revert, REVERT_MS);
  }
  async function save() {
    const v = current();
    clearTimeout(timer);
    btn.disabled = true;
    if ((await onsave(v)) === false) return revert();
    saved = v;
    settle();
  }
  const wrap = h("div.stack", { style: { gap: "4px" } }, range, h("div.row", { style: { gap: "8px" } }, btn, hint));
  wrap.input = range.input;
  wrap.set = (v) => {
    saved = v;
    if (!dirty) range.set(v);
  };
  wrap.dispose = () => clearTimeout(timer);
  return wrap;
}

export default {
  title: "运动控制",
  async mount(container) {
    const offs = [];
    const [catalog, gestureInfo, tuningInfo, editor] = await Promise.all([
      api.get("/mc/actions").catch(() => []),
      api.get("/mc/gestures").catch(() => ({ gestures: [], joints: [] })),
      api.get("/mc/walk_tuning").catch(() => null),
      moveEditor().catch((error) => {
        console.error(error);
        return null;
      }),
    ]);
    let scale = Number(localStorage.getItem("xiaoxiaobai.speedScale") || 0.6);
    let stickV = { vx: 0, vy: 0, vtheta: 0 };
    const held = new Set();

    // ---------------------------------------------------------------- mode card
    const modeGrid = h("div.mode-flow");
    const modeInfo = h("div.small.muted");
    const modeCard = h("div.card",
      h("div.card-head", h("h3", icon("layers", "sm"), "运动模式"), h("span.sub", "SDK McAction 状态机，切换为异步任务"), h("div.actions", modeInfo)),
      h("div.card-body", modeGrid));

    // The app's follow mode (android FollowService): runs on the phone, drives 「走跑」 itself.
    const FOLLOW_FROM = new Set(["BIPED_STAND_DEFAULT", "BIPED_WALK_RUN", "BIPED_WALK_UPPER"]);
    // 跟屁虫 is the same, but walks right up and hugs the leg once its camera is covered.
    const followStatus = () => JSON.parse(native.call("followStatus") || "{}");
    const followButton = (cling, label, idle) => {
      const text = h("small");
      const btn = h("button.mode-btn", {
        onclick: () => {
          const f = followStatus();
          if (f.running && !!f.cling === cling) return native.call("followStop");
          const { fast = 0.6, slow = 0.3 } = JSON.parse(native.call("followSettings") || "{}");
          const error = native.call("followStart", JSON.stringify({ fast, slow, cling }));
          if (error) toast(error, "error", 6000);
        },
      }, h("span.code", "App"), h("b", label), text);
      return { btn, text, cling, idle };
    };
    const followBtns = native.inApp ? [
      followButton(false, "跟随", "认出你的腿跟着走，熄屏也继续；速度在「更多 → 跟随模式」里调"),
      followButton(true, "跟屁虫", "一路跟到你脚边，贴上去抱住你的腿不放，你走开就松手接着跟"),
    ] : [];
    const [followBtn, clingBtn] = followBtns.map((b) => b.btn);
    function renderFollow() {
      const f = followStatus();
      const current = store.snapshot?.action?.current_action?.name;
      for (const b of followBtns) {
        const mine = !!f.running && !!f.cling === b.cling;
        b.btn.classList.toggle("current", mine);
        b.btn.disabled = !f.running && !FOLLOW_FROM.has(current);
        b.text.textContent = mine ? `${f.message}（点一下停止）`
          : f.running ? `点一下换成「${b.btn.querySelector("b").textContent}」`
          : FOLLOW_FROM.has(current) ? b.idle : "先站立，再点这里开始";
      }
    }

    // ---------------------------------------------------------------- drive card
    const bars = h("div.vel-bars");
    const stick = joystick({
      onChange: ({ x, y }) => {
        stickV = { vx: y * scale, vy: 0, vtheta: -x * scale };
        pushVelocity();
      },
    });
    const hold = (vec) => ({
      onpointerdown: () => {
        held.add(vec);
        pushVelocity();
      },
      onpointerup: () => {
        held.delete(vec);
        pushVelocity();
      },
      onpointerleave: () => {
        if (held.delete(vec)) pushVelocity();
      },
    });
    const dpad = h("div.dpad",
      h("button.btn", { title: "左移（Q）", ...hold(KEYS.q) }, "左移"),
      h("button.btn", { title: "前进（W）", ...hold(KEYS.w) }, icon("arrowUp")),
      h("button.btn", { title: "右移（E）", ...hold(KEYS.e) }, "右移"),
      h("button.btn", { title: "左转（A）", ...hold(KEYS.a) }, icon("rotateL")),
      h("button.btn", { title: "后退（S）", ...hold(KEYS.s) }, icon("arrowDown")),
      h("button.btn", { title: "右转（D）", ...hold(KEYS.d) }, icon("rotateR")));
    const scaleEl = savedRange({
      min: 0.1, max: 1, step: 0.05, value: scale, format: (v) => `${Math.round(v * 100)}%`,
      onsave: (v) => {
        scale = v;
        localStorage.setItem("xiaoxiaobai.speedScale", String(v));
        toast(`速度上限已保存为 ${Math.round(v * 100)}%`);
      },
    });
    const driveHint = h("div.small.muted");
    const driveCard = h("div.card",
      h("div.card-head", h("h3", icon("gamepad", "sm"), "行走控制"), h("span.sub", "摇杆、方向键或键盘 WASD / QE")),
      h("div.card-body.stack",
        h("div.drive", stick, h("div.stack", bars, h("div.row.wrap", dpad, h("div.field", { style: { flex: 1, minWidth: "180px" } }, h("label", "速度上限"), scaleEl)))),
        driveHint));

    function pushVelocity() {
      let keyV = [0, 0, 0];
      for (const vec of held) keyV = keyV.map((x, i) => x + vec[i]);
      const v = held.size
        ? { vx: keyV[0] * scale, vy: keyV[1] * scale, vtheta: keyV[2] * scale }
        : stickV;
      setVelocity(v, (e) => toast(e.message, "warn"));
      renderBars();
    }

    function renderBars() {
      const v = store.command.velocity;
      const bar = (label, value) => {
        const pct = Math.min(50, Math.abs(value) * 50);
        return h("div.vel-bar", h("span.muted", label),
          h("div.track", h("div.fill", { style: { left: value >= 0 ? "50%" : `${50 - pct}%`, width: `${pct}%` } })),
          h("output", value.toFixed(2)));
      };
      clear(bars, bar("前后 vx", v.vx), bar("横移 vy", v.vy), bar("转向 vθ", v.vtheta));
    }

    // ---------------------------------------------------------------- telemetry
    const telemetry = h("div.metrics");
    const telemetryCard = h("div.card",
      h("div.card-head", h("h3", icon("activity", "sm"), "控制循环"), h("span.sub", "来自 scheduler 的状态文件（5 Hz）")),
      h("div.card-body", telemetry));

    // ---------------------------------------------------------------- walk policy / IMU correction
    const agentSeg = h("div.seg.sm");
    const skateSeg = h("div.seg.sm");
    const corrMax = tuningInfo?.imu_corr_max_deg ?? 6;
    const sendTuning = (body) => api.post("/mc/walk_tuning", body).catch((e) => toast(e.message, "error"));
    const corrRange = (key, label) => savedRange({
      min: -corrMax, max: corrMax, step: 0.5, value: tuningInfo?.[key] ?? 0,
      format: (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}°`,
      onsave: (v) => api.post("/mc/walk_tuning", { [key]: v }).then(
        () => toast(`${label}已保存为 ${v > 0 ? "+" : ""}${v.toFixed(1)}°`),
        (e) => (toast(e.message, "error"), false)),
    });
    const rollRange = corrRange("imu_roll_corr_deg", "IMU 横滚校正");
    const pitchRange = corrRange("imu_pitch_corr_deg", "IMU 俯仰校正");
    const headSwitch = switchEl(false, (on) => api.post("/mc/head", { enabled: on }).then((r) => toast(r.message)).catch((e) => {
      toast(e.message, "error");
      headSwitch.querySelector("input").checked = false;
    }));
    const mirrorRow = toggleRow("策略左右镜像（诊断）",
      "把走路 / 滑板策略左右翻过来跑，先把 IMU 横滚校正调回 0 再比。走歪换了边是策略的问题，没换边是机器人本身（重心、装配）。不保存，重启后自动关闭。",
      !!tuningInfo?.mirror_policy,
      (on) => sendTuning({ mirror_policy: on }).then((r) => {
        if (r) toast(r.message);
        else mirrorRow.querySelector("input").checked = !on;
      }));
    const switchRow = (key, title, hint) => {
      const row = toggleRow(title, hint, tuningInfo?.[key] ?? true,
        (on) => sendTuning({ [key]: on }).then((r) => {
          if (r) toast(r.message);
          else row.querySelector("input").checked = !on;
        }));
      return row;
    };
    const kpVinRow = switchRow("kp_vin_comp", "电压补偿 KP",
      "电量下降时舵机变软、身体慢慢歪：走动时按电池电压调走路 KP，让舵机一直像 7.0 V 时一样硬（满电时略低于 125，最多加到 1.25 倍）。"
      + "原地站着不动时不会高于 125，避免腿太硬发抖。走路中切换立即生效，保存在机器人上。");
    const autoTrimRow = switchRow("imu_auto_trim", "IMU 自动修正",
      "以 ≥0.1 m/s 稳定直走（前进或后退，来回走的短段会累计）时每攒够 2 秒修一次，站着或原地踏步时不学（那时策略不扶正身体，只会越修越多）：横滚拉回左右直立（策略没扶正的那部分由它补上，最多 ±3°，需要先做下面的「IMU 安装标定」），"
      + "俯仰保持本次开机刚走时的前后姿态、只抵消之后的慢漂移（最多 ±2°）。每次起步 2 秒内渐入；某个方向顶到上限还修不动就自动清零停用。关掉即清零。");
    const calibInfo = h("small");
    const calibBtn = h("button.btn.sm", {
      onclick: async () => {
        calibBtn.disabled = true;
        calibBtn.textContent = "标定中，别碰…";
        try {
          const r = await api.post("/mc/imu_calibrate", {}, { timeout: 20000 });
          toast(r.message);
          if (r.data) Object.assign(tuningInfo, r.data);
          tuningSig = "";
        } catch (e) {
          toast(e.message, "error");
        } finally {
          calibBtn.disabled = false;
          calibBtn.textContent = "开始标定";
        }
      },
    }, "开始标定");
    const calibRow = h("div.toggle-row", h("div.text", h("b", "IMU 安装标定"), calibInfo), calibBtn);
    const tuneLive = h("div.small.muted");
    const compLive = h("div.small.muted");
    const tuneCard = h("div.card",
      h("div.card-head", h("h3", icon("settings", "sm"), "走路调校"), h("span.sub", "走路、滑板中实时生效，保存在机器人上"),
        h("div.actions",
          h("label.row.small", { style: { gap: "8px" }, title: "站立 / 走跑 / 滑板时让头左右摆动" }, "头部摆动", headSwitch),
          h("button.btn.sm", { onclick: () => sendTuning({ reset: true }).then((r) => r && toast("已恢复默认")) }, "恢复默认"))),
      h("div.card-body.stack",
        h("div.field", h("label", "走路策略"), agentSeg),
        h("div.field", h("label", "滑板策略"), skateSeg),
        h("div.slider-grid",
          h("div.slider-item", h("div.lbl", h("span", "IMU 横滚校正"), h("span.muted.small", "+ 往左扶正")), rollRange),
          h("div.slider-item", h("div.lbl", h("span", "IMU 俯仰校正"), h("span.muted.small", "+ 往前扶正")), pitchRange)),
        kpVinRow,
        calibRow,
        autoTrimRow,
        compLive,
        mirrorRow,
        tuneLive));
    let tuningSig = "";
    const fmtDeg = (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}°`;
    function renderTuning(a) {
      const t = a.tuning;
      if (!t || !tuningInfo) return;
      const sig = `${t.walk_agent}|${t.skate_agent}|${t.imu_roll_corr_deg}|${t.imu_pitch_corr_deg}|${t.mirror_policy}`
        + `|${t.kp_vin_comp}|${t.imu_auto_trim}|${t.imu_mount_roll_deg}|${t.imu_mount_pitch_deg}|${t.imu_calib_ts}`;
      if (sig !== tuningSig) {
        tuningSig = sig;
        const agentButtons = (agents, key) => (agents || []).map((ag) => h(`button${ag.index === t[key] ? ".active" : ""}`, {
          title: ag.file,
          onclick: () => ag.index !== t[key] && sendTuning({ [key]: ag.index }).then((r) => r && toast(r.message)),
        }, ag.label, ag.index === tuningInfo.defaults[key] ? h("span.muted", "（默认）") : null));
        clear(agentSeg, agentButtons(tuningInfo.agents, "walk_agent"));
        clear(skateSeg, agentButtons(tuningInfo.skate_agents, "skate_agent"));
        rollRange.set(t.imu_roll_corr_deg);
        pitchRange.set(t.imu_pitch_corr_deg);
        mirrorRow.querySelector("input").checked = !!t.mirror_policy;
        kpVinRow.querySelector("input").checked = t.kp_vin_comp !== false;
        autoTrimRow.querySelector("input").checked = t.imu_auto_trim !== false;
        const mount = `横滚 ${fmtDeg(t.imu_mount_roll_deg ?? 0)} / 俯仰 ${fmtDeg(t.imu_mount_pitch_deg ?? 0)}`;
        calibInfo.textContent = (t.imu_calib_ts
          ? `已标定（${new Date(t.imu_calib_ts * 1000).toLocaleString()}）：安装偏差 ${mount}。`
          : "还没标定：IMU 自动修正需要它才准。")
          + "先切到「位控站立」，放在平地上别碰，点「开始标定」，约 3 秒。换 IMU、拆装或摔过后再做一次；手动校正会归零。";
      }
      headSwitch.querySelector("input").checked = !!a.head_motion;
      const loop = a.loop || {};
      calibBtn.disabled = calibBtn.textContent !== "开始标定" || a.current_action?.name !== "BIPED_STAND_DEFAULT";
      const trim = loop.imu_trim_deg;
      compLive.textContent = loop.supply_v == null && loop.kp_written == null
        ? "站立后显示供电电压、实际写入的 KP 和自动修正量。"
        : `供电 ${loop.supply_v != null ? loop.supply_v.toFixed(2) + " V" : "—"}，走路 KP ${loop.kp_written ?? "—"}`
          + (t.imu_auto_trim !== false && trim ? `；自动修正 横滚 ${fmtDeg(trim[0])} / 俯仰 ${fmtDeg(trim[1])}` : "")
          + ((loop.imu_trim_given_up || []).length
            ? `（${loop.imu_trim_given_up.map((x) => (x === "roll" ? "横滚" : "俯仰")).join("、")}修不动，已停用；检查安装标定或关掉再开重置）`
            : "");
      tuneLive.textContent = loop.roll_deg == null
        ? "站立后显示实时姿态。「走路策略」用于「走跑」，「滑板策略」用于「滑板」；「走跑·上肢自由」固定用只控制腿 1-12 的策略，不受这里影响。"
        : `当前姿态：横滚 ${loop.roll_deg.toFixed(1)}° / 俯仰 ${loop.pitch_deg.toFixed(1)}°。走路往哪边歪或跑偏，就往反方向扶。`
          + "广播体操、偷看也用这组 IMU 校正：做动作总往前倒就把俯仰往负方向调（每次 0.5–1°，拖好后点「保存」才生效，做动作中也可以调）；它同时作用于走路。";
    }

    // ---------------------------------------------------------------- input sources
    const sourceBody = h("div.card-body.flush");
    const sourceCard = h("div.card",
      h("div.card-head", h("h3", icon("shield", "sm"), "输入源仲裁"), h("span.sub", "高优先级可抢占，超时自动释放")),
      sourceBody);

    async function renderSources() {
      try {
        const data = await api.get("/mc/input_source");
        clear(sourceBody, h("table.table",
          h("thead", h("tr", h("th", "名称"), h("th", "优先级"), h("th", "超时"), h("th", "说明"), h("th", ""))),
          h("tbody", data.sources.map((s) => h(`tr${s.active ? ".active" : ""}`,
            h("td", h("b.mono", s.name), s.active ? h("span.badge.accent", { style: { marginLeft: "6px" } }, "控制中") : null),
            h("td", s.priority),
            h("td", `${s.timeout_ms} ms`),
            h("td.muted.wrap", s.description),
            h("td", s.builtin ? h("span.badge", "内置") : h("button.btn.xs.ghost", {
              onclick: () => api.del(`/mc/input_source/${encodeURIComponent(s.name)}`).then(renderSources).catch((e) => toast(e.message, "error")),
            }, icon("trash", "sm"))))))));
      } catch (error) {
        clear(sourceBody, empty("alert", "读取失败", error.message));
      }
    }

    // ---------------------------------------------------------------- upper body
    const upperValues = JSON.parse(localStorage.getItem("xiaoxiaobai.upper") || "{}");
    const upperGrid = h("div.slider-grid");
    const upperHint = h("div.small.muted");
    const sendUpper = throttle(async () => {
      if (action() !== "BIPED_CUSTOM_UPPER") return;
      try {
        await api.post("/mc/upper_body", { targets_deg: upperValues });
      } catch (error) {
        toast(error.message, "error");
      }
    }, 40);
    for (const m of UPPER_MOTORS) {
      const r = rangeEl({
        min: m.min, max: m.max, value: Math.max(m.min, Math.min(m.max, upperValues[m.id] ?? 0)), format: (v) => `${v}°`,
        oninput: (v) => {
          upperValues[m.id] = v;
          localStorage.setItem("xiaoxiaobai.upper", JSON.stringify(upperValues));
          store.command.upper = { ...upperValues };
          emit("command", { upper: true });
          sendUpper();
        },
      });
      upperGrid.append(h("div.slider-item", h("div.lbl", h("span", m.label), h("span.muted.mono", `ID ${m.id}`)), r));
    }
    store.command.upper = { ...upperValues };
    const upperCard = h("div.card",
      h("div.card-head", h("h3", icon("bot", "sm"), "上肢控制"), h("span.sub", "BIPED_CUSTOM_UPPER：13–20 号舵机直接给目标角"),
        h("div.actions",
          h("button.btn.sm", {
            onclick: () => {
              for (const m of UPPER_MOTORS) upperValues[m.id] = 0;
              localStorage.setItem("xiaoxiaobai.upper", "{}");
              upperGrid.querySelectorAll(".range").forEach((r) => r.set(0));
              store.command.upper = { ...upperValues };
              sendUpper();
            },
          }, "全部归零"))),
      h("div.card-body.stack", upperHint, upperGrid));

    // ---------------------------------------------------------------- gestures (walk + free upper body)
    const gestureButtons = new Map();
    const gestureGrid = h("div.gesture-grid", gestureInfo.gestures.map((g) => {
      const btn = h("button.gesture-btn", { title: g.name, onclick: () => playGesture(g.name) },
        h("span.code", String(g.index).padStart(2, "0")), h("b", g.label));
      gestureButtons.set(g.name, btn);
      return btn;
    }));
    const gestureHint = h("div.small.muted");
    const myGrid = h("div.gesture-grid");
    const myWrap = h("div.stack", { style: { gap: "8px" }, hidden: true }, h("div.small.muted", "我的动作（在下方「动作编辑器」里教的）"), myGrid);
    function renderMine(moves) {
      myWrap.hidden = !moves.length;
      clear(myGrid, moves.map((m, i) => h("button.gesture-btn", { title: `${m.frames} 帧 · ${m.seconds.toFixed(1)} 秒`, onclick: () => editor.play(m) },
        h("span.code", `M${i + 1}`), h("b", m.name))));
    }
    if (editor) {
      renderMine(editor.moves());
      offs.push(editor.onMoves(renderMine));
    }
    const autoSwitch = switchEl(true, (on) => api.post("/mc/upper", { auto: on }).catch((e) => toast(e.message, "error")));
    const leadSwitch = switchEl(false, async (on) => {
      try {
        const res = await api.post("/mc/lead", { enabled: on });
        toast(res.message || (on ? "牵手走已开启" : "牵手走已关闭"));
      } catch (error) {
        toast(error.message, "error");
        leadSwitch.querySelector("input").checked = !on;
      }
    });
    const leadHint = h("div.small.lead-hint", { hidden: true });
    async function playGesture(name) {
      try {
        const res = await api.post("/mc/gesture", { name });
        toast(res.message || "已发送");
      } catch (error) {
        toast(error.message, "error");
      }
    }

    // Hold single joints while the rest stays alive (joint angles, as the model and the twin).
    const heldJoints = new Map();
    const holdGrid = h("div.slider-grid");
    const sendHold = throttle(async () => {
      if (action() !== "BIPED_WALK_UPPER") return;
      try {
        await api.post("/mc/upper", { manual_deg: Object.fromEntries(heldJoints) });
      } catch (error) {
        toast(error.message, "error");
      }
    }, 50);
    const holdRanges = new Map();
    for (const j of gestureInfo.joints) {
      const tag = h("button.btn.xs.ghost", {
        title: "松开，交回给动作和待机小动作",
        onclick: () => {
          heldJoints.delete(j.motor_id);
          holdRanges.get(j.motor_id).set(Math.round(j.rest_deg));
          api.post("/mc/upper", { release: [j.motor_id] }).catch((e) => toast(e.message, "error"));
          renderHeld();
        },
      }, "松开");
      const r = rangeEl({
        min: j.min_deg, max: j.max_deg, value: Math.round(j.rest_deg), format: (v) => `${v}°`,
        oninput: (v) => {
          heldJoints.set(j.motor_id, v);
          renderHeld();
          sendHold();
        },
      });
      holdRanges.set(j.motor_id, r);
      r.tag = tag;
      holdGrid.append(h("div.slider-item",
        h("div.lbl", h("span", UPPER_LABEL[j.motor_id] || j.name), h("span.row", { style: { gap: "6px" } }, tag, h("span.muted.mono", `ID ${j.motor_id}`))), r));
    }
    function renderHeld() {
      for (const [id, r] of holdRanges) r.tag.hidden = !heldJoints.has(id);
    }
    renderHeld();

    const gestureCard = h("div.card",
      h("div.card-head", h("h3", icon("sparkles", "sm"), "拟人动作"),
        h("span.sub", "走跑·上肢自由：走路策略管腿，13–20 号自由。站立或走跑中点一下会自动切过来"),
        h("div.actions",
          h("label.row.small", { style: { gap: "8px" }, title: "手臂放软，牵着它的手带它走：手臂往前拉就前进、往后推就后退，往侧上方抬就原地往那边转，两样一起就走弧线；松手后手臂慢慢回到原位，它也慢慢停下。站立或走跑中打开会自动切到「走跑·上肢自由」" }, "牵手走", leadSwitch),
          h("label.row.small", { style: { gap: "8px" }, title: "站着不动时，隔十几秒随机做个小动作（点头、挠头、伸懒腰…）" }, "空闲小动作", autoSwitch),
          h("button.btn.sm", { onclick: () => playGesture("random") }, icon("sparkles", "sm"), "随机"),
          h("button.btn.sm", { onclick: () => api.post("/mc/gesture/stop").catch((e) => toast(e.message, "error")) }, icon("stop", "sm"), "停止"))),
      h("div.card-body.stack", leadHint, gestureHint, gestureGrid, myWrap,
        h("details.hold",
          h("summary", "手动保持关节角（拖动即保持，其余关节照常活动）",
            h("button.btn.xs", {
              style: { marginLeft: "10px" },
              onclick: (e) => {
                e.preventDefault();
                heldJoints.clear();
                for (const j of gestureInfo.joints) holdRanges.get(j.motor_id).set(Math.round(j.rest_deg));
                renderHeld();
                api.post("/mc/upper", { release: "all" }).catch((err) => toast(err.message, "error"));
              },
            }, "全部松开")),
          holdGrid)));

    // ---------------------------------------------------------------- walk recordings
    const recList = h("div.stack", { style: { gap: "6px" } });
    const recCount = h("select.input", { style: { width: "auto" } },
      ...[1, 3, 5, 10].map((n) => h("option", { value: n, selected: n === 5 }, `最近 ${n} 段`)));
    const download = (url) => {
      window.location.href = url;
      toast(native.inApp ? "开始下载，存到手机「下载」目录" : "开始下载");
    };
    const recLabel = (name) => {
      const m = name.match(/^(.+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.(npz|part)$/);
      if (!m) return name;
      return `${m[1]} ${m[3]}-${m[4]} ${m[5]}:${m[6]}:${m[7]}` + (m[8] === "part" ? "（中断/录制中）" : "");
    };
    async function loadRecs() {
      try {
        const d = await api.get("/system/walk_rec");
        const items = (d.items || []).slice(0, 15);
        clear(recList, items.length ? items.map((it) => h("div.row", { style: { justifyContent: "space-between", gap: "8px" } },
          h("span.small", recLabel(it.name)),
          h("span.small.muted", `${(it.size / 1048576).toFixed(1)} MB · ${timeAgo(it.mtime)}`),
          h("button.btn.sm", { title: it.name, onclick: () => download(`/api/v1/system/walk_rec/file/${encodeURIComponent(it.name)}`) },
            icon("download", "sm"))))
          : h("div.small.muted", "还没有录像：走一段路停下来，就会生成一段。"));
      } catch (e) {
        clear(recList, h("div.small.muted", `读不到录像列表：${e.message}`));
      }
    }
    const recCard = h("div.card",
      h("div.card-head", h("h3", icon("download", "sm"), "走路数据记录"), h("span.sub", "每次走路 / 做动作自动逐帧记录在机器人上"),
        h("div.actions", h("button.btn.sm", { onclick: loadRecs }, "刷新"))),
      h("div.card-body.stack",
        h("div.small.muted", "停下后录像写成完整一段；走路中每 5 秒先存一次，断电、重启也只丢最后几秒（标「中断」）。"
          + "诊断包 = 最近几段录像 + 控制循环日志 + 当前调校和状态，发给开发者看就够了。"
          + "下载在后台以最低优先级读文件，不影响走路；最好停下来再下载。"),
        h("div.row", { style: { gap: "8px" } }, recCount,
          h("button.btn.sm.primary", { onclick: () => download(`/api/v1/system/walk_rec/bundle?count=${recCount.value}`) },
            icon("download", "sm"), "下载诊断包")),
        recList));
    loadRecs();
    let wasWalking = false;
    offs.push(on("status", () => {
      const walking = isWalking();
      if (wasWalking && !walking) setTimeout(loadRecs, 3000);
      wasWalking = walking;
    }));

    container.append(h("div.stack",
      modeCard,
      gestureCard,
      editor?.el,
      h("div.grid.cols-2", driveCard, telemetryCard),
      h("div.grid.cols-2", tuneCard, sourceCard),
      recCard,
      upperCard));

    // ---------------------------------------------------------------- render from status
    let lastSig = "";
    function render() {
      const s = store.snapshot;
      if (!s) return;
      editor?.render(s);
      const a = s.action;
      const current = a.current_action.name;
      const allowed = new Set(a.allowed.map((x) => x.name));
      const sig = `${current}|${a.target_action}|${[...allowed].join(",")}|${a.dance_pending}`;
      if (sig !== lastSig) lastSig = sig, clear(modeGrid, catalog.filter((c) => c.settable || c.name === current).map((c) => {
        const isCurrent = c.name === current;
        const pending = (a.target_action === c.name || c.name === a.dance_pending) && !isCurrent;
        const reachable = isCurrent || allowed.has(c.name);
        return h(`button.mode-btn${isCurrent ? ".current" : ""}${pending ? ".pending" : ""}${c.name === "PASSIVE_DEFAULT" ? ".danger" : ""}`, {
          disabled: !reachable || !c.settable,
          title: reachable ? c.description : `当前「${a.current_action.label}」不能直接切换；可从：${c.from.join("、")}`,
          onclick: () => (!isCurrent || c.name === "BIPED_PROPUP") && setAction(c.name),
        },
        h("span.code", c.value),
        h("b", pending ? h("span.spinner", { style: { width: "12px", height: "12px", marginRight: "6px" } }) : null, c.label),
        h("small", c.description));
      }), followBtn, clingBtn);
      renderFollow();
      clear(modeInfo,
        `已持续 ${timeAgo(a.since).replace("前", "")}`,
        a.last_stop ? ` · 上次停止：${a.last_stop.text}（${timeAgo(a.last_stop.ts)}）` : "");

      renderTuning(a);
      const walking = isWalking(current);
      stick.setEnabled(walking);
      dpad.querySelectorAll("button").forEach((b) => (b.disabled = !walking));
      driveHint.textContent = current === "BIPED_SKATE"
        ? "滑板只能向前滑和转弯，后退即停、不能横移。松开即停。"
        : current === "BIPED_GYMNASTICS"
        ? "做操中不接收速度指令，做完自动回到「走跑」原地平衡。"
        : current === "BIPED_PEEK"
        ? "偷看中不接收速度指令，做完自动回到「走跑」原地平衡。"
        : current === "BIPED_PROPUP"
        ? "撑起/躺下不接收速度指令：再点一次「撑起/躺下」在撑起和躺下之间切换；动作做到一半时点击会等它做完再执行。"
        : current === "BIPED_CHILL_CHAIR"
        ? "坐凳休闲不接收速度指令。结束前请先扶住机器人，再切回「位控站立」。"
        : current === "BIPED_CHILL_LIE" || current === "BIPED_CHILL_SIT"
        ? "休闲动作不接收速度指令；切走时会先回到起始姿势（约 1–2 秒）再切换。"
        : current === "BIPED_WAKE_STRETCH"
        ? "起床伸懒腰不接收速度指令，做完（约 9 秒）自动回到「位控站立」，仍是躺着的姿势。"
        : walking
        ? "松开即停。网页的控制源为 app_proxy（优先级 60），手柄（rc，80）会抢占。"
        : current === "BIPED_STAND_DEFAULT" ? "先切换到「走跑」才能控制移动。" : "先站立，再切换到「走跑」。";

      const loop = a.loop || {};
      const n = (v, digits = 1, unit = "") => (v == null ? "—" : `${Number(v).toFixed(digits)}${unit}`);
      clear(telemetry,
        h("div.metric", h("small", "进程"), h("b", loop.pid ? h("span.mono", loop.pid) : "未运行")),
        h("div.metric", h("small", "电机总电流"), h("b", n(loop.current_a, 2), h("small", " A"))),
        h("div.metric", h("small", "电池电压"), h("b", n(s.battery.voltage_v, 2), h("small", " V"))),
        h("div.metric", h("small", "横滚 / 俯仰"), h("b", `${n(loop.roll_deg)}° / ${n(loop.pitch_deg)}°`)),
        h("div.metric", h("small", "单周期耗时"), h("b", n(loop.loop_ms, 1), h("small", " ms"))),
        h("div.metric", h("small", "超时次数"), h("b", loop.overruns ?? "—")),
        h("div.metric", { style: { gridColumn: "span 2" } }, h("small", "动作层"),
          h("b", { style: { fontSize: "14px" } }, Object.entries(loop.moves || {}).map(([k, v]) =>
            h(`span.badge.${v === "active" ? "ok" : v === "inactive" ? "" : "accent"}`, { style: { marginRight: "6px" } }, `${k}: ${v}`)))),
      );

      const free = current === "BIPED_WALK_UPPER";
      const live = a.upper || {};
      const canGesture = GESTURE_FROM.has(current);
      for (const [name, btn] of gestureButtons) {
        btn.disabled = !canGesture;
        btn.classList.toggle("playing", free && live.playing === name);
      }
      holdGrid.querySelectorAll("input").forEach((i) => (i.disabled = !free));
      if (free && autoSwitch.querySelector("input").checked !== live.auto) autoSwitch.querySelector("input").checked = !!live.auto;
      const leading = free && !!live.lead;
      leadSwitch.querySelector("input").checked = leading;
      leadSwitch.querySelector("input").disabled = !canGesture;
      leadHint.hidden = !leading;
      if (leading) {
        const vx = Number(live.lead_vx) || 0;
        const vt = Number(live.lead_vtheta) || 0;
        const parts = [];
        if (Math.abs(vx) > 0.005) parts.push(`${vx > 0 ? "前进" : "后退"} ${Math.abs(vx).toFixed(2)} m/s`);
        if (Math.abs(vt) > 0.01) parts.push(`${vt > 0 ? "左转" : "右转"} ${Math.round(Math.abs(vt) * 180 / Math.PI)}°/s`);
        clear(leadHint, h("b", "牵手走中："),
          !live.lead_active ? "手臂正在放软…" : parts.length ? parts.join("，") : "站着等你：拉手臂往前走，往后推后退，侧抬原地转弯",
          h("span.muted", "（手柄或网页摇杆一动就听它们的）"));
      }
      if (free) {
        for (const [id, deg] of Object.entries(live.manual_deg || {})) {
          if (!heldJoints.has(Number(id))) heldJoints.set(Number(id), deg), holdRanges.get(Number(id))?.set(deg);
        }
        renderHeld();
      }
      gestureHint.textContent = free
        ? `${live.playing ? `正在做：${gestureInfo.gestures.find((g) => g.name === live.playing)?.label || live.playing}。` : "待机中：会呼吸、轻微晃动、四处看看。"}`
          + (live.scale != null && live.scale < 0.99 ? ` 当前动作幅度 ${Math.round(live.scale * 100)}%（按走路策略和速度自动收小，防止摔倒）。` : "")
        : canGesture ? "点任意动作即可（会切换到「走跑·上肢自由」，走路策略继续平衡）。" : "先站立，才能做动作。";

      const upper = current === "BIPED_CUSTOM_UPPER";
      upperGrid.querySelectorAll("input").forEach((i) => (i.disabled = !upper));
      clear(upperHint, upper
        ? "拖动滑块实时下发目标角。完成后点「零力矩」退出。"
        : h("span", "需要先在零力矩状态下进入「上肢控制」模式。", allowed.has("BIPED_CUSTOM_UPPER")
          ? h("button.btn.xs", { style: { marginLeft: "8px" }, onclick: () => setAction("BIPED_CUSTOM_UPPER") }, "进入上肢控制")
          : null));
    }

    // ---------------------------------------------------------------- keyboard
    const keyDown = (event) => {
      if (event.target.closest("input, textarea, select") || event.repeat) return;
      const vec = KEYS[event.key.toLowerCase()];
      if (vec && isWalking()) {
        held.add(vec);
        pushVelocity();
      }
    };
    const keyUp = (event) => {
      const vec = KEYS[event.key.toLowerCase()];
      if (vec && held.delete(vec)) pushVelocity();
    };
    const blur = () => {
      held.clear();
      pushVelocity();
    };
    document.addEventListener("keydown", keyDown);
    document.addEventListener("keyup", keyUp);
    window.addEventListener("blur", blur);

    offs.push(on("status", render), on("input_source", renderSources), on("command", renderBars));
    render();
    renderBars();
    renderSources();
    const sourcesTimer = setInterval(renderSources, 3000);
    const followTimer = followBtn ? setInterval(renderFollow, 1000) : null;

    return () => {
      offs.forEach((off) => off());
      [scaleEl, rollRange, pitchRange].forEach((r) => r.dispose());
      editor?.dispose();
      clearInterval(sourcesTimer);
      clearInterval(followTimer);
      document.removeEventListener("keydown", keyDown);
      document.removeEventListener("keyup", keyUp);
      window.removeEventListener("blur", blur);
      setVelocity({ vx: 0, vy: 0, vtheta: 0 });
    };
  },
};
