// Face expressions, speaker tones and the realtime voice session.

import * as api from "../api.js";
import { decodeLevels, drawLevels, drawMatrix, FaceAnimator } from "../components/face.js";
import { isVolc, labelOf, voicePicker } from "../components/voice-picker.js";
import { loadExpressions, on, store } from "../store.js";
import { busy, clear, clock, empty, h, icon, rangeEl, toast } from "../ui.js";

const VOICE_STATE = { starting: "连接中", ready: "聆听中", thinking: "思考中", speaking: "说话中", idle: "未开启", error: "出错" };

export default {
  title: "表情与声音",
  async mount(container) {
    const offs = [];
    const expressions = await loadExpressions();

    // ---------------------------------------------------------------- face
    const hero = h("canvas.face-hero", { width: 640, height: 480 });
    const animator = new FaceAnimator(hero);
    const heroLabel = h("b");
    const tiles = new Map();
    const grid = h("div.face-grid", expressions.map((e) => {
      const canvas = h("canvas", { width: 160, height: 120 });
      if (e.levels) drawLevels(canvas, e.cols, e.rows, decodeLevels(e.levels, e.cols * e.rows), { glow: false });
      else drawMatrix(canvas, e.cols, e.rows, e.pixels, { glow: false });
      const tile = h("button.face-tile", { title: e.name, onclick: () => setExpression(e.name, tile) }, canvas, h("span", e.label));
      tiles.set(e.name, tile);
      return tile;
    }));

    async function setExpression(name, tile) {
      tile?.classList.add("active");
      try {
        const res = await api.post("/interaction/emotion", { name });
        if (!res.data.display_running) toast("表情已保存，但显示进程没有运行，屏幕暂时不会变化", "warn");
      } catch (error) {
        toast(error.message, "error");
      }
    }

    let current = null;
    function renderFace() {
      const name = store.snapshot?.expression;
      if (!name || name === current) return;
      current = name;
      animator.show(name);
      const item = expressions.find((e) => e.name === name);
      heroLabel.textContent = item ? `${item.label} · ${item.name}` : name;
      for (const [n, tile] of tiles) tile.classList.toggle("active", n === name);
    }

    // ---------------------------------------------------------------- screen brightness
    const display = (await api.get("/settings").catch(() => null))?.display || {};
    let brightnessTimer = null;
    let brightnessSent = display.brightness ?? 100;
    const sendBrightness = (v) => {
      if (v === brightnessSent) return;
      brightnessSent = v;
      api.patch("/settings/display", { brightness: v }).catch((error) => toast(error.message, "error"));
    };
    const brightness = rangeEl({
      min: 10, max: 100, step: 5, value: brightnessSent, format: (v) => `${v}%`,
      oninput: () => {
        if (brightnessTimer) return;
        brightnessTimer = setTimeout(() => {
          brightnessTimer = null;
          sendBrightness(Number(brightness.input.value));
        }, 150);
      },
      onchange: (v) => {
        clearTimeout(brightnessTimer);
        brightnessTimer = null;
        sendBrightness(v);
      },
    });

    // ---------------------------------------------------------------- voice
    const voiceBody = h("div.card-body.stack");
    const transcript = h("div.stack", { style: { gap: "8px", maxHeight: "260px", overflow: "auto" } });
    let settings = null;
    let voiceSig = "";
    let catalog = null;
    let picker = null;
    let loading = null;

    async function loadVoiceSettings() {
      settings = await api.get("/settings").catch(() => null);
      catalog ??= await api.get("/interaction/voice").then((s) => s.voices).catch(() => null);
      picker?.destroy();
      picker = settings && catalog ? voicePicker({
        groups: catalog,
        value: settings.voice.voice,
        compact: true,
        hasVolcKey: () => !!settings?.voice?.has_volc_key,
        speechRate: () => settings?.voice?.speech_rate,
        onChange: changeVoice,
      }) : null;
      voiceSig = "";
    }

    async function changeVoice(id) {
      try {
        await api.patch("/settings/voice", { voice: id });
        const name = labelOf(id, catalog);
        if (!store.snapshot?.voice?.running) {
          toast(`音色已切换为「${name}」`, "ok", 2000);
          return;
        }
        toast(`正在用「${name}」重新连接语音对话…`, "info", 2500);
        await api.post("/interaction/voice/stop", {});
        for (let i = 0; i < 30 && (await api.get("/interaction/voice")).running; i++) await new Promise((r) => setTimeout(r, 200));
        await api.post("/interaction/voice/start", {});
      } catch (error) {
        toast(error.message, "error", 7000);
      }
    }

    async function renderVoice() {
      const voice = store.snapshot?.voice || {};
      if (!settings) {
        loading ??= loadVoiceSettings().finally(() => (loading = null));
        await loading;
      }
      const sig = JSON.stringify([voice.running, voice.state, voice.error, voice.has_key]);
      if (sig === voiceSig) return;
      voiceSig = sig;
      const cfg = settings?.voice || {};
      const btn = h(`button.btn${voice.running ? ".danger" : ".primary"}`, {
        onclick: (e) => busy(e.currentTarget, async () => {
          try {
            const res = await api.post(voice.running ? "/interaction/voice/stop" : "/interaction/voice/start", {});
            toast(res.message);
          } catch (error) {
            toast(error.message, "error", 7000);
          }
        }),
      }, icon(voice.running ? "stop" : "mic", "sm"), voice.running ? "结束对话" : "开始语音对话");
      clear(voiceBody,
        h("div.row",
          h(`span.chip.${voice.running ? (voice.state === "error" ? "danger" : "busy") : voice.error ? "danger" : ""}`, h("span.dot"),
            VOICE_STATE[voice.state] || voice.state || "未开启"),
          h("span.spacer"), btn),
        voice.error ? h("div.test-result.fail", voice.error) : null,
        !voice.has_key && !store.snapshot?.mock
          ? h("div.test-result.fail", "还没有配置千问（DashScope）API Key。", h("a", { href: "#/settings/voice" }, " 去设置"))
          : null,
        picker ? h("div.field", h("label", "音色", h("span.spacer"), h("a.small", { href: "#/settings/voice" }, "语速与 API Key")), picker) : null,
        h("dl.kv",
          h("dt", "音频链路"), h("dd", cfg.route === "pi" || isVolc(cfg.voice || "") ? "树莓派转发（UART）" : "ESP32 直连云端"),
          h("dt", "音量 / 麦克风增益"), h("dd", `${cfg.volume ?? "—"} / ${cfg.mic_gain ?? "—"}`),
          h("dt", "摄像头"), h("dd", cfg.camera ? "对话时发送画面" : "仅语音"),
          h("dt", "人设与记忆"), h("dd", cfg.share_persona ? "与文字聊天共用" : "使用默认")),
        h("div.small.muted", "语音对话由 voice_chat.py 驱动（千问 Omni 实时模型听懂与回答，所选音色负责说话），对话内容会自动保存到「设置 → 记忆 → 对话历史」。"),
        h("div.divider"),
        h("div.field-label", "实时字幕"),
        transcript);
      if (!transcript.children.length) transcript.append(h("div.small.muted.placeholder", "开始对话后，这里会显示双方说的话。"));
    }

    function addLine(role, text, extra = "") {
      transcript.querySelector(".placeholder")?.remove();
      const line = h("div.small", h("span.muted.mono", `${clock(Date.now() / 1000)} `), h("b", role === "user" ? "你：" : "机器人："), text, extra);
      transcript.append(line);
      transcript.scrollTop = transcript.scrollHeight;
      return line;
    }
    let replyLine = null;

    // ---------------------------------------------------------------- sounds
    let volume = Number(localStorage.getItem("xiaoxiaobai.toneVolume") || 60);
    const sounds = await api.get("/interaction/sounds").catch(() => []);
    const soundCard = h("div.card",
      h("div.card-head", h("h3", icon("volume", "sm"), "提示音"), h("span.sub", "经 ESP32 扬声器播放")),
      h("div.card-body.stack",
        sounds.length
          ? h("div.btn-group", sounds.map((s) => h("button.btn.sm", {
            onclick: (e) => busy(e.currentTarget, () => api.post("/interaction/sound", { name: s.name, volume })
              .catch((error) => toast(error.message, "error"))),
          }, icon("play", "sm"), s.label)))
          : empty("volume", "没有提示音"),
        h("div.field", h("label", "音量"), rangeEl({
          min: 0, max: 100, value: volume, format: (v) => `${v}`,
          onchange: (v) => {
            volume = v;
            localStorage.setItem("xiaoxiaobai.toneVolume", String(v));
          },
        }))));

    container.append(h("div.grid.split-faces",
      h("div.card",
        h("div.card-head", h("h3", icon("smile", "sm"), "表情"), h("span.sub", `${expressions.length} 种，点击切换到机器人屏幕`)),
        h("div.card-body", grid)),
      h("div.stack",
        h("div.card", h("div.card-body.stack", hero, h("div.row", h("span.muted", "当前表情"), heroLabel),
          h("div.field", h("label", icon("sun", "sm"), "屏幕亮度"), brightness))),
        h("div.card", h("div.card-head", h("h3", icon("mic", "sm"), "实时语音对话")), voiceBody),
        soundCard)));

    offs.push(
      on("status", () => {
        renderFace();
        renderVoice();
      }),
      on("settings", () => {
        settings = null;
      }),
      on("voice_user", (ev) => addLine("user", ev.text)),
      on("voice_reply_delta", (ev) => {
        if (!replyLine) {
          replyLine = addLine("assistant", "");
          replyLine.append(h("span.reply"));
        }
        replyLine.querySelector(".reply").textContent += ev.text;
      }),
      on("voice_reply", (ev) => {
        if (replyLine) replyLine.querySelector(".reply").textContent = ev.text;
        else addLine("assistant", ev.text);
        if (ev.interrupted) (replyLine || transcript.lastChild).append(h("span.muted", "（被打断）"));
        replyLine = null;
      }),
    );
    renderFace();
    renderVoice();

    return () => {
      offs.forEach((off) => off());
      picker?.destroy();
      animator.dispose();
    };
  },
};
