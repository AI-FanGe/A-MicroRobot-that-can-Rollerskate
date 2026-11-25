// Move editor: pose the arms and head (sliders, or by hand in teach mode), record keyframes,
// preview, and save the move under a name the robot then answers to ("做一下螃蟹舞").

import * as api from "../api.js";
import { action, on } from "../store.js";
import { busy, clear, confirmDialog, empty, h, icon, rangeEl, select, switchEl, throttle, toast } from "../ui.js";
import { setAction } from "../pages/home.js";

const DRAFT_KEY = "xiaoxiaobai.moveDraft";
const POSE_MODES = new Set(["BIPED_CUSTOM_UPPER", "BIPED_WALK_UPPER"]);
const PLAY_MODES = new Set(["BIPED_CUSTOM_UPPER", "BIPED_STAND_DEFAULT", "BIPED_WALK_RUN", "BIPED_WALK_UPPER"]);
const TEACH_POLL_MS = 200;

const round = (v) => Math.round(v * 10) / 10;
const total = (keys) => keys.reduce((s, k) => s + Number(k.dur) + Number(k.hold), 0);

export async function moveEditor() {
  const info = await api.get("/mc/editor");
  const joints = info.joints;
  const limits = info.limits;
  const rest = Object.fromEntries(joints.map((j) => [j.key, j.rest]));
  let moves = info.moves;
  let draft = loadDraft();
  let pose = { ...rest, ...(draft.pose || {}) };
  let selected = -1;
  let live = true;
  let teachTimer = null;
  const offs = [];
  const listeners = new Set();

  function loadDraft() {
    try {
      const d = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
      if (d && Array.isArray(d.keys)) return d;
    } catch { /* bad JSON: start over */ }
    return { id: null, name: "", keys: [] };
  }
  const saveDraft = () => localStorage.setItem(DRAFT_KEY, JSON.stringify({ ...draft, pose }));

  // ---------------------------------------------------------------- pose sliders
  const sendPose = throttle(async () => {
    if (!live || !POSE_MODES.has(action()) || teachTimer) return;
    try {
      await api.post("/mc/editor/pose", { pose });
    } catch (error) {
      toast(error.message, "error");
    }
  }, 60);
  const ranges = new Map();
  const sliderGrid = h("div.slider-grid", joints.map((j) => {
    const r = rangeEl({
      min: j.min, max: j.max, step: 1, value: pose[j.key], format: (v) => `${Math.round(v)}°`,
      oninput: (v) => {
        pose[j.key] = v;
        saveDraft();
        sendPose();
      },
    });
    ranges.set(j.key, r);
    return h("div.slider-item", h("div.lbl", h("span", j.label), h("span.muted.mono", j.key)), r);
  }));
  function setPose(next, { send = true } = {}) {
    pose = { ...pose, ...next };
    for (const [key, r] of ranges) r.set(round(pose[key]));
    saveDraft();
    if (send) sendPose();
  }

  const liveSwitch = switchEl(live, (v) => {
    live = v;
    if (v) sendPose();
  });
  const teachSwitch = switchEl(false, async (v) => {
    try {
      const res = await api.post("/mc/editor/teach", { enabled: v });
      toast(res.message, v ? "warn" : "ok");
      if (!v) readPose().catch(() => {});
    } catch (error) {
      toast(error.message, "error");
      teachSwitch.querySelector("input").checked = false;
    }
  });
  const teachLabel = h("label.row.small", { style: { gap: "8px" }, title: "放松 13–20 号电机，用手把手臂和头掰到想要的位置，滑块会跟着读数走（走跑·上肢自由下腿会自己保持平衡）" },
    "手动示教", teachSwitch);

  async function readPose() {
    const data = await api.get("/mc/editor/pose");
    if (data.source) setPose(data.pose, { send: false });
    return data;
  }

  function startTeachPoll() {
    if (teachTimer) return;
    teachTimer = setInterval(() => readPose().catch(() => {}), TEACH_POLL_MS);
  }
  function stopTeachPoll() {
    clearInterval(teachTimer);
    teachTimer = null;
  }

  // ---------------------------------------------------------------- keyframes
  const frameList = h("div.kf-list");
  const totals = h("span.sub");
  const nameInput = h("input.input", { placeholder: "动作名字，例如：螃蟹舞", maxlength: 16, value: draft.name || "",
    oninput: (e) => {
      draft.name = e.target.value;
      saveDraft();
    } });
  const editing = h("span.badge.accent", { hidden: true });

  function record() {
    if (draft.keys.length >= limits.max_keys) return toast(`最多 ${limits.max_keys} 个关键帧`, "warn");
    const key = { dur: 0.8, hold: 0.3, ease: "io", pose: Object.fromEntries(joints.map((j) => [j.key, round(pose[j.key])])) };
    const at = selected >= 0 ? selected + 1 : draft.keys.length;
    draft.keys.splice(at, 0, key);
    selected = at;
    changed();
  }

  function changed() {
    saveDraft();
    renderFrames();
  }

  function numberInput(value, [lo, hi], onchange) {
    return h("input.input.kf-num", {
      type: "number", min: lo, max: hi, step: 0.05, value,
      onchange: (e) => {
        const v = Math.min(hi, Math.max(lo, Number(e.target.value) || lo));
        e.target.value = v;
        onchange(v);
      },
    });
  }

  function renderFrames() {
    const seconds = total(draft.keys);
    totals.textContent = `${draft.keys.length} 帧 · ${seconds.toFixed(1)} 秒（上限 ${limits.max_keys} 帧 / ${limits.max_s} 秒）`;
    totals.style.color = seconds > limits.max_s ? "var(--danger)" : "";
    editing.hidden = !draft.id;
    editing.textContent = draft.id ? `正在修改「${moves.find((m) => m.id === draft.id)?.name || draft.name}」` : "";
    if (!draft.keys.length) {
      clear(frameList, empty("sparkles", "还没有关键帧", "摆好一个姿势，点「记录关键帧」；再摆下一个，再记录……"));
      return;
    }
    clear(frameList, draft.keys.map((k, i) => {
      const move = (d) => {
        const j = i + d;
        if (j < 0 || j >= draft.keys.length) return;
        [draft.keys[i], draft.keys[j]] = [draft.keys[j], draft.keys[i]];
        selected = j;
        changed();
      };
      const summary = joints.filter((j) => Math.abs(k.pose[j.key] - j.rest) >= 3)
        .map((j) => `${j.label.replace(/（.*）/, "")} ${Math.round(k.pose[j.key])}°`).join(" · ") || "自然站姿";
      return h(`div.kf-row${i === selected ? ".selected" : ""}`, { onclick: () => { selected = i; renderFrames(); } },
        h("span.kf-index", String(i + 1)),
        h("div.kf-main",
          h("div.kf-summary", { title: summary }, summary),
          h("div.kf-params",
            h("label", "用时", numberInput(k.dur, limits.dur_s, (v) => { k.dur = v; changed(); }), "秒"),
            h("label", "停留", numberInput(k.hold, limits.hold_s, (v) => { k.hold = v; changed(); }), "秒"),
            select(info.eases, k.ease, (v) => { k.ease = v; changed(); }, ".sm"))),
        h("div.kf-actions",
          h("button.btn.xs", { title: "把机器人摆到这一帧", onclick: (e) => { e.stopPropagation(); setPose(k.pose); } }, icon("target", "sm")),
          h("button.btn.xs", { title: "用当前姿势替换这一帧", onclick: (e) => {
            e.stopPropagation();
            k.pose = Object.fromEntries(joints.map((j) => [j.key, round(pose[j.key])]));
            changed();
            toast(`第 ${i + 1} 帧已更新`);
          } }, icon("refresh", "sm")),
          h("button.btn.xs", { title: "上移", disabled: i === 0, onclick: (e) => { e.stopPropagation(); move(-1); } }, icon("arrowUp", "sm")),
          h("button.btn.xs", { title: "下移", disabled: i === draft.keys.length - 1, onclick: (e) => { e.stopPropagation(); move(1); } }, icon("arrowDown", "sm")),
          h("button.btn.xs.ghost", { title: "删除这一帧", onclick: (e) => {
            e.stopPropagation();
            draft.keys.splice(i, 1);
            selected = Math.min(selected, draft.keys.length - 1);
            changed();
          } }, icon("trash", "sm"))));
    }));
  }

  async function preview(button) {
    if (!draft.keys.length) return toast("先记录几个关键帧", "warn");
    await busy(button, async () => {
      try {
        const res = await api.post("/mc/editor/preview", { keys: draft.keys });
        toast(res.message);
      } catch (error) {
        toast(error.message, "error");
      }
    });
  }

  async function save(button, asNew = false) {
    const name = nameInput.value.trim();
    if (!name) return toast("给动作起个名字", "warn");
    await busy(button, async () => {
      try {
        const res = await api.post("/mc/custom_gestures", { id: asNew ? null : draft.id, name, keys: draft.keys });
        draft.id = res.data.id;
        draft.name = res.data.name;
        saveDraft();
        toast(`${res.message}。之后对机器人说「${res.data.name}」它就会做`, "ok", 6000);
        await reloadMoves();
      } catch (error) {
        toast(error.message, "error");
      }
    });
  }

  async function clearDraft() {
    if (draft.keys.length && !(await confirmDialog({ title: "清空关键帧", message: "清空当前编辑的全部关键帧？已保存的动作不受影响。", okText: "清空", danger: true }))) return;
    draft = { id: null, name: "", keys: [] };
    nameInput.value = "";
    selected = -1;
    changed();
  }

  // ---------------------------------------------------------------- saved moves
  const savedList = h("div.list");
  async function reloadMoves() {
    moves = await api.get("/mc/custom_gestures");
    renderSaved();
    for (const fn of listeners) fn(moves);
  }
  async function playMove(m) {
    try {
      const res = await api.post(`/mc/custom_gestures/${m.id}/play`, {});
      toast(res.message);
    } catch (error) {
      toast(error.message, "error");
    }
  }
  function renderSaved() {
    if (!moves.length) {
      clear(savedList, h("div.small.muted", { style: { padding: "12px 16px" } }, "还没有保存的动作。"));
      return;
    }
    clear(savedList, moves.map((m) => h("div.list-item",
      h("div.main-col",
        h("div.title", m.name, m.id === draft.id ? h("span.badge.accent", "编辑中") : null),
        h("div.meta", h("span", `${m.frames} 帧`), h("span", `${m.seconds.toFixed(1)} 秒`), h("span.mono", m.id), h("span", `语音：「做一下${m.name}」`))),
      h("div.item-actions",
        h("button.btn.xs", { title: "做一次", onclick: () => playMove(m) }, icon("play", "sm"), "做一次"),
        h("button.btn.xs", { title: "载入编辑器修改", onclick: () => {
          draft = { id: m.id, name: m.name, keys: JSON.parse(JSON.stringify(m.keys)) };
          nameInput.value = m.name;
          selected = -1;
          changed();
          renderSaved();
          toast(`已载入「${m.name}」`);
        } }, icon("edit", "sm")),
        h("button.btn.xs.ghost", { title: "删除", onclick: async () => {
          if (!(await confirmDialog({ title: "删除动作", message: `删除「${m.name}」？`, okText: "删除", danger: true }))) return;
          try {
            const res = await api.del(`/mc/custom_gestures/${m.id}`);
            if (draft.id === m.id) draft.id = null;
            toast(res.message);
            await reloadMoves();
            renderFrames();
          } catch (error) {
            toast(error.message, "error");
          }
        } }, icon("trash", "sm"))))));
  }

  // ---------------------------------------------------------------- layout
  const hint = h("div.small.muted");
  const modeButtons = h("div.row.wrap", { style: { gap: "6px" } });
  const releaseBtn = h("button.btn.sm", { title: "走跑·上肢自由：松开编辑器保持的关节，交回给待机小动作",
    onclick: () => api.post("/mc/editor/release").then((r) => toast(r.message)).catch((e) => toast(e.message, "error")) }, "松开关节");
  const previewBtn = h("button.btn.sm", { onclick: (e) => preview(e.currentTarget) }, icon("play", "sm"), "预览");
  const saveBtn = h("button.btn.sm.primary", { onclick: (e) => save(e.currentTarget) }, icon("check", "sm"), "保存动作");
  const saveNewBtn = h("button.btn.sm", { onclick: (e) => save(e.currentTarget, true), title: "保存成一个新动作，原来的不变" }, "另存为新动作");

  const el = h("div.card",
    h("div.card-head", h("h3", icon("edit", "sm"), "动作编辑器"),
      h("span.sub", "摆姿势 → 记录关键帧 → 再摆 → 再记录 → 起名保存，之后说出名字机器人就会做"),
      h("div.actions", teachLabel,
        h("label.row.small", { style: { gap: "8px" }, title: "拖动滑块时机器人实时跟着动" }, "实时同步", liveSwitch))),
    h("div.card-body.stack",
      h("div.row.wrap", { style: { justifyContent: "space-between" } }, hint, modeButtons),
      sliderGrid,
      h("div.row.wrap", { style: { gap: "8px" } },
        h("button.btn.sm.primary", { onclick: record }, icon("plus", "sm"), "记录关键帧"),
        h("button.btn.sm", { title: "把滑块设成机器人现在的姿势", onclick: () => readPose().then((d) => toast(d.source ? "已读取机器人姿态" : "机器人没有可读的姿态（先进入上肢控制或站立）", d.source ? "ok" : "warn")).catch((e) => toast(e.message, "error")) }, icon("download", "sm"), "读取机器人姿态"),
        h("button.btn.sm", { onclick: () => setPose(rest) }, "回到自然姿势"),
        releaseBtn),
      h("div.kf-head", h("b", "关键帧"), totals, editing,
        h("div.row", { style: { marginLeft: "auto", gap: "6px" } },
          previewBtn,
          h("button.btn.sm", { onclick: () => api.post("/mc/gesture/stop").catch((e) => toast(e.message, "error")) }, icon("stop", "sm"), "停止"),
          h("button.btn.sm.ghost", { onclick: clearDraft }, icon("trash", "sm"), "清空"))),
      frameList,
      h("div.input-group", nameInput, saveNewBtn, saveBtn)),
    h("div.card-head", { style: { paddingTop: "0" } }, h("h3", icon("sparkles", "sm"), "我的动作"), h("span.sub", "保存在机器人上，语音、文字聊天和接口都能调用")),
    h("div.card-body.flush", savedList));

  let lastSig = "";
  function render(snapshot) {
    const a = snapshot?.action;
    if (!a) return;
    const current = a.current_action.name;
    const teach = !!a.editor?.teach;
    const allowed = new Set(a.allowed.map((x) => x.name));
    const sig = `${current}|${teach}|${[...allowed].join(",")}`;
    teachSwitch.querySelector("input").checked = teach;
    teachSwitch.querySelector("input").disabled = current !== "BIPED_CUSTOM_UPPER" && current !== "BIPED_WALK_UPPER";
    if (teach) startTeachPoll();
    else stopTeachPoll();
    previewBtn.disabled = !PLAY_MODES.has(current);
    releaseBtn.hidden = current !== "BIPED_WALK_UPPER";
    saveNewBtn.hidden = !draft.id;
    if (sig === lastSig) return;
    lastSig = sig;
    const goto = (name, label) => allowed.has(name) ? h("button.btn.xs", { onclick: () => setAction(name) }, label) : null;
    const teaching = "示教中：手臂和头已放松，用手掰到想要的姿势，滑块跟着读数走；摆好点「记录关键帧」。关掉示教会锁定在当前姿势。";
    if (current === "BIPED_WALK_UPPER") {
      hint.textContent = teach
        ? `${teaching}腿由走路策略保持平衡，别推它的身体。`
        : "走跑·上肢自由：腿自己保持平衡，上肢不参与。拖滑块实时摆姿势，或打开「手动示教」直接用手掰。";
      clear(modeButtons);
    } else if (current === "BIPED_CUSTOM_UPPER") {
      hint.textContent = teach
        ? teaching
        : "上肢控制：腿不出力，要先把机器人架起来。站着示教请用「走跑·上肢自由」。";
      clear(modeButtons);
    } else if (current === "BIPED_STAND_DEFAULT" || current === "BIPED_WALK_RUN") {
      hint.textContent = "站立中可以预览和播放动作；要摆姿势或手动示教，切到「走跑·上肢自由」（腿保持平衡，上肢随你摆）。";
      clear(modeButtons, goto("BIPED_WALK_UPPER", "进入走跑·上肢自由"));
    } else {
      hint.textContent = "现在可以先离线编辑关键帧。要让机器人跟着摆：站立后进入「走跑·上肢自由」（腿保持平衡，可以直接用手示教），或零力矩下进入「上肢控制」（腿不出力，要架起来）。";
      clear(modeButtons, goto("BIPED_STAND_DEFAULT", "站立"), goto("BIPED_CUSTOM_UPPER", "进入上肢控制"));
    }
  }

  offs.push(on("custom_gestures", () => reloadMoves().catch(() => {})));
  renderFrames();
  renderSaved();

  return {
    el,
    render,
    moves: () => moves,
    onMoves: (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    play: playMove,
    dispose: () => {
      stopTeachPoll();
      offs.forEach((off) => off());
    },
  };
}
