// Voice dropdown (Qwen Omni, Qwen TTS, Volcengine Doubao) with a preview button,
// shared by the voice settings and the voice chat card.

import * as api from "../api.js";
import { h, icon, toast } from "../ui.js";

const CUSTOM = "__custom__";
export const ENGINE_LABEL = { omni: "千问 Omni", qwen_tts: "千问 TTS", volc: "火山引擎" };
const ENGINE_HINT = {
  omni: "Omni 模型自己出声，延迟最低，可以 ESP32 直连。",
  qwen_tts: "Omni 想好回答后，由千问 TTS 朗读。",
  volc: "Omni 想好回答后，由火山引擎豆包语音朗读；音频会自动改走树莓派转发。",
};

export const isVolc = (voice) => /_bigtts$/.test(voice) || /^(S_|saturn_)/.test(voice);

export function engineOf(voice, groups = []) {
  const group = groups.find((g) => g.voices.some((v) => v.id === voice));
  return group?.engine || (isVolc(voice) ? "volc" : "qwen_tts");
}

export function labelOf(voice, groups = []) {
  for (const g of groups) {
    const hit = g.voices.find((v) => v.id === voice);
    if (hit) return hit.name;
  }
  return voice;
}

let player = null;

function stopPreview() {
  if (!player) return;
  player.audio.pause();
  URL.revokeObjectURL(player.url);
  player.button.classList.remove("playing");
  player = null;
}

// opts: { groups, value, onChange(voice), hasVolcKey: () => bool, speechRate: () => number, compact }
export function voicePicker({ groups, value, onChange, hasVolcKey = () => true, speechRate = () => undefined, compact = false }) {
  let current = value || "Mochi";
  const known = () => groups.some((g) => g.voices.some((v) => v.id === current));

  const custom = h("input.input.mono", {
    value: known() ? "" : current,
    placeholder: "火山音色 ID，如 zh_female_xxx_moon_bigtts 或复刻音色 S_xxx",
    onchange: (e) => {
      const id = e.target.value.trim();
      if (id) commit(id);
    },
  });
  const select = h("select.select", {
    onchange: (e) => {
      if (e.target.value === CUSTOM) {
        custom.hidden = false;
        custom.focus();
        if (custom.value.trim()) commit(custom.value.trim());
        else sync();
        return;
      }
      custom.value = "";
      commit(e.target.value);
    },
  },
  groups.map((g) => h("optgroup", { label: g.label },
    g.voices.map((v) => h("option", { value: v.id, selected: v.id === current }, v.name)))),
  h("optgroup", { label: "其他" }, h("option", { value: CUSTOM, selected: !known() }, "自定义音色 ID…")));

  const badge = h("span.chip.voice-engine");
  const hint = h("div.hint.voice-hint");
  const play = h("button.btn.voice-play", { type: "button", title: "试听这个音色", onclick: () => preview() }, icon("volume", "sm"), compact ? null : "试听");

  function sync() {
    const customMode = select.value === CUSTOM;
    custom.hidden = !customMode;
    const engine = engineOf(current, groups);
    badge.className = `chip voice-engine ${engine}`;
    badge.textContent = ENGINE_LABEL[engine];
    const warn = engine === "volc" && !hasVolcKey();
    hint.className = `hint voice-hint${warn ? " warn" : ""}`;
    hint.textContent = warn ? "这是火山引擎音色：请先在「设置 → 语音」填写火山引擎 API Key。" : ENGINE_HINT[engine];
  }

  function commit(voice) {
    if (voice === current) return sync();
    current = voice;
    stopPreview();
    sync();
    onChange?.(voice);
  }

  async function preview() {
    if (player?.button === play) return stopPreview();
    stopPreview();
    const voice = select.value === CUSTOM ? custom.value.trim() : select.value;
    if (!voice) return toast("先填写音色 ID", "warn");
    play.disabled = true;
    play.classList.add("loading");
    try {
      const blob = await api.postBlob("/interaction/voice/preview", { voice, speech_rate: speechRate() });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      player = { audio, url, button: play };
      play.classList.add("playing");
      audio.onended = stopPreview;
      await audio.play();
    } catch (error) {
      stopPreview();
      toast(error.message, "error", 6000);
    } finally {
      play.disabled = false;
      play.classList.remove("loading");
    }
  }

  sync();
  const el = h(`div.voice-picker${compact ? ".compact" : ""}`,
    h("div.voice-row", select, play),
    custom,
    h("div.voice-meta", badge, hint));
  el.refresh = sync;
  el.destroy = stopPreview;
  return el;
}
