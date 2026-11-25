// Settings: LLM providers, voice, persona, knowledge base, memory, network, safety & system.

import * as api from "../api.js";
import { loadSettings, store } from "../store.js";
import {
  busy, clear, confirmDialog, duration, empty, field, h, icon, modal, rangeEl, select, toast, toggleRow,
} from "../ui.js";
import { isVolc, labelOf, voicePicker } from "../components/voice-picker.js";
import computerTab from "./settings-computer.js";
import knowledgeTab from "./settings-knowledge.js";
import memoryTab from "./settings-memory.js";
import networkTab from "./settings-network.js";

const TABS = [
  { id: "llm", label: "大模型", icon: "sparkles", render: llmTab },
  { id: "voice", label: "语音", icon: "mic", render: voiceTab },
  { id: "persona", label: "人设", icon: "user", render: personaTab },
  { id: "computer", label: "电脑助手", icon: "monitor", render: computerTab },
  { id: "knowledge", label: "知识库", icon: "book", render: knowledgeTab },
  { id: "memory", label: "记忆", icon: "pin", render: memoryTab },
  { id: "network", label: "网络", icon: "wifi", render: networkTab },
  { id: "system", label: "安全与系统", icon: "shield", render: systemTab },
];

const PRESET_STYLE = {
  qianwen: ["#615ced", "千"], dashscope: ["#615ced", "通"], deepseek: ["#4d6bfe", "DS"], moonshot: ["#16191f", "K"], zhipu: ["#3859ff", "智"],
  doubao: ["#1f6fff", "豆"], siliconflow: ["#7c3aed", "硅"], openai: ["#10a37f", "AI"], anthropic: ["#d97757", "C"],
  gemini: ["#4285f4", "G"], openrouter: ["#6467f2", "OR"], ollama: ["#3f3f46", "OL"], custom: ["#64748b", "API"],
};

const logo = (preset, cls = "logo") => {
  const [color, text] = PRESET_STYLE[preset] || PRESET_STYLE.custom;
  return h(`div.${cls}`, { style: { background: color } }, text);
};

async function save(section, values, quiet = false) {
  try {
    const res = await api.patch(`/settings/${section}`, values);
    if (store.settings) store.settings[section] = { ...store.settings[section], ...res.data };
    if (!quiet) toast(res.message || "已保存", "ok", 1800);
    return res.data;
  } catch (error) {
    toast(error.message, "error");
    throw error;
  }
}

export default {
  title: "设置",
  async mount(container, { args }) {
    let active = TABS.find((t) => t.id === args[0]) || TABS[0];
    let cleanupTab = null;
    const tabs = h("div.tabs");
    const body = h("div.stack");

    async function show(tab) {
      active = tab;
      history.replaceState(null, "", `#/settings/${tab.id}`);
      clear(tabs, TABS.map((t) => h(`button${t.id === tab.id ? ".active" : ""}`, { onclick: () => show(t) }, icon(t.icon, "sm"), t.label)));
      cleanupTab?.();
      cleanupTab = null;
      clear(body, h("div.empty", h("span.spinner")));
      try {
        const settings = await loadSettings(true);
        clear(body);
        cleanupTab = (await tab.render(body, settings)) || null;
      } catch (error) {
        clear(body, empty("alert", "加载失败", error.message));
      }
    }

    container.append(h("div.settings", tabs, body));
    await show(active);
    return () => cleanupTab?.();
  },
};

// =================================================================== LLM

async function llmTab(root, settings) {
  const presets = await api.get("/llm/presets");
  const list = h("div.card-body.flush");

  async function renderList() {
    const s = await loadSettings(true);
    const providers = s.llm.providers || [];
    if (!providers.length) {
      clear(list, empty("sparkles", "还没有配置大模型", "添加一个服务商的 API Key，机器人就能用它聊天和执行指令。",
        h("button.btn.primary", { onclick: () => editProvider() }, icon("plus", "sm"), "添加模型")));
      return;
    }
    clear(list, providers.map((p) => {
      const isDefault = p.id === s.llm.active;
      return h("div.provider",
        logo(p.preset),
        h("div.info",
          h("b", p.name, isDefault ? h("span.badge.accent", "默认") : null, p.enabled === false ? h("span.badge", "已停用") : null,
            !p.has_key && p.preset !== "ollama" ? h("span.badge.warn", "缺少 Key") : null),
          h("small", `${p.model || "未选择模型"} · ${p.base_url}`),
          p.has_key ? h("small.mono", p.api_key) : null),
        h("div.btn-group",
          h("button.btn.sm", { onclick: (e) => testProvider(e.currentTarget, { id: p.id }) }, icon("zap", "sm"), "测试"),
          !isDefault ? h("button.btn.sm", {
            onclick: () => api.post("/llm/active", { id: p.id }).then((r) => { toast(r.message, "ok"); renderList(); }).catch((e) => toast(e.message, "error")),
          }, "设为默认") : null,
          h("button.btn.sm.icon-only", { title: "编辑", onclick: () => editProvider(p) }, icon("edit", "sm")),
          h("button.btn.sm.icon-only.ghost", {
            title: "删除",
            onclick: async () => {
              if (!(await confirmDialog({ title: "删除模型配置", message: `确定删除「${p.name}」？`, okText: "删除", danger: true }))) return;
              api.del(`/llm/providers/${p.id}`).then(() => renderList()).catch((e) => toast(e.message, "error"));
            },
          }, icon("trash", "sm"))));
    }));
  }

  async function testProvider(button, body, resultEl = null) {
    return busy(button, async () => {
      try {
        const r = await api.post("/llm/test", body);
        const text = `连接成功 · 首字 ${r.data.first_token_ms} ms · 回复：${r.data.reply || "（空）"}`;
        if (resultEl) clear(resultEl, h("div.test-result.ok", text));
        else toast(text, "ok", 6000);
      } catch (error) {
        if (resultEl) clear(resultEl, h("div.test-result.fail", error.message));
        else toast(error.message, "error", 8000);
      }
    });
  }

  function editProvider(existing = null) {
    const form = {
      preset: existing?.preset || "dashscope",
      name: existing?.name || "",
      base_url: existing?.base_url || "",
      api_key: existing?.api_key || "",
      model: existing?.model || "",
      enabled: existing?.enabled !== false,
    };
    const applyPreset = (id, keepModel = false) => {
      const p = presets.find((x) => x.id === id);
      form.preset = id;
      form.name = p.name;
      form.base_url = p.base_url;
      if (!keepModel) form.model = p.models[0] || "";
    };
    if (!existing) applyPreset(form.preset);

    const presetGrid = h("div.preset-grid");
    const nameInput = h("input.input", { value: form.name, oninput: (e) => (form.name = e.target.value) });
    const urlInput = h("input.input.mono", { value: form.base_url, placeholder: "https://…/v1", oninput: (e) => (form.base_url = e.target.value) });
    const keyInput = h("input.input.mono", {
      type: "password", value: form.api_key, autocomplete: "off", placeholder: "sk-…",
      oninput: (e) => (form.api_key = e.target.value),
    });
    const modelList = h("datalist", { id: "modelOptions" });
    const modelInput = h("input.input.mono", { value: form.model, list: "modelOptions", placeholder: "模型名", oninput: (e) => (form.model = e.target.value) });
    const keyLink = h("a.small", { target: "_blank", rel: "noopener" });
    const modelHint = h("div.hint");
    const result = h("div");

    const payload = () => ({ ...form, ...(existing ? { id: existing.id } : {}) });
    const refreshPresetUI = () => {
      const p = presets.find((x) => x.id === form.preset);
      clear(presetGrid, presets.map((x) => h(`button.preset${x.id === form.preset ? ".active" : ""}`, {
        type: "button",
        onclick: () => {
          applyPreset(x.id);
          nameInput.value = form.name;
          urlInput.value = form.base_url;
          modelInput.value = form.model;
          refreshPresetUI();
        },
      }, logo(x.id), h("span", x.name))));
      clear(modelList, (p?.models || []).map((m) => h("option", { value: m })));
      keyLink.href = p?.key_url || "#";
      keyLink.textContent = p?.key_url ? "获取 API Key" : "";
      keyInput.placeholder = p?.no_key ? "本地服务通常不需要" : "sk-…";
      modelHint.textContent = p?.model_hint || "可直接输入，或点右侧从服务商拉取列表";
    };
    refreshPresetUI();

    const fetchModels = (button) => busy(button, async () => {
      try {
        const r = await api.post("/llm/models", payload());
        clear(modelList, r.data.models.map((m) => h("option", { value: m })));
        toast(`获取到 ${r.data.models.length} 个模型，点模型输入框选择`, "ok");
        modelInput.focus();
      } catch (error) {
        toast(error.message, "error", 7000);
      }
    });

    modal({
      title: existing ? `编辑 ${existing.name}` : "添加大模型",
      wide: true,
      body: h("div.stack",
        existing ? null : h("div.field", h("label", "服务商"), presetGrid),
        h("div.form-grid",
          field("显示名称", nameInput),
          field("Base URL（OpenAI 兼容）", urlInput),
          h("div.field.span-2", h("label", "API Key", h("span.spacer"), keyLink), keyInput,
            h("div.hint", existing ? "留空或不修改表示沿用已保存的 Key。Key 只保存在机器人上，不会再显示明文。" : "Key 只保存在机器人上（权限 600），页面只显示打码后的内容。")),
          h("div.field.span-2", h("label", "模型"),
            h("div.input-group", modelInput, h("button.btn", { type: "button", onclick: (e) => fetchModels(e.currentTarget) }, icon("refresh", "sm"), "获取模型列表")),
            modelHint, modelList)),
        result),
      footer: (close) => [
        h("button.btn", { onclick: (e) => testProvider(e.currentTarget, payload(), result) }, icon("zap", "sm"), "测试连接"),
        h("span.spacer"),
        h("button.btn", { onclick: close }, "取消"),
        h("button.btn.primary", {
          onclick: (e) => busy(e.currentTarget, async () => {
            try {
              const r = existing ? await api.patch(`/llm/providers/${existing.id}`, form) : await api.post("/llm/providers", form);
              toast(r.message || "已保存", "ok");
              close();
              renderList();
            } catch (error) {
              toast(error.message, "error");
            }
          }),
        }, "保存"),
      ],
    });
  }

  const cfg = settings.llm;
  const saveLlm = (key) => (value) => save("llm", { [key]: value }, true);
  root.append(
    h("div.card",
      h("div.card-head", h("h3", icon("sparkles", "sm"), "模型服务"), h("span.sub", "任何 OpenAI 兼容接口：国内外主流服务商、本地 Ollama"),
        h("div.actions", h("button.btn.sm.primary", { onclick: () => editProvider() }, icon("plus", "sm"), "添加模型"))),
      list),
    h("div.card",
      h("div.card-head", h("h3", "对话参数")),
      h("div.card-body.stack",
        h("div.form-grid",
          field("温度", rangeEl({ min: 0, max: 2, step: 0.1, value: cfg.temperature, format: (v) => v.toFixed(1), onchange: saveLlm("temperature") }), "越高越发散，越低越稳定"),
          field("最大回复长度", rangeEl({ min: 128, max: 4096, step: 128, value: cfg.max_tokens, format: (v) => `${v} tokens`, onchange: saveLlm("max_tokens") })),
          field("携带历史轮数", rangeEl({ min: 0, max: 30, value: cfg.history_turns, format: (v) => `${v} 轮`, onchange: saveLlm("history_turns") })),
          field("知识库检索条数", rangeEl({ min: 0, max: 10, value: cfg.kb_top_k, format: (v) => (v ? `${v} 条` : "关闭"), onchange: saveLlm("kb_top_k") }))),
        h("div",
          toggleRow("允许控制身体", "模型可以调用站立、行走、做动作、停止（急停永远只由人触发）", cfg.allow_motion_tools, saveLlm("allow_motion_tools")),
          toggleRow("语音指挥动作", "语音里说“往前走几步”“抬一下头”“跟我打个招呼”，机器人照做：现成动作优先，没有就自己编关节动作（限位内）；做完回到原来的模式",
            cfg.voice_motion !== false, saveLlm("voice_motion")),
          toggleRow("允许表情和提示音", "模型可以切换表情、播放提示音", cfg.allow_expression_tools, saveLlm("allow_expression_tools")),
          toggleRow("联网搜索", "需要实时信息时自动搜索，说“联网搜一下…”时一定搜（千问按次计费；非千问模型借用千问 Key 搜索）",
            cfg.web_search !== false, saveLlm("web_search")),
          toggleRow("自动记忆", "每轮对话（文字和语音）后让模型提取关于你的长期信息（会多一次模型调用）", cfg.auto_memory, saveLlm("auto_memory")),
          toggleRow("回忆以前的对话", "问到“还记得…吗”“上次说的…”时，先翻长期记忆和以前所有的聊天记录（文字和语音）再回答",
            cfg.memory_recall !== false, saveLlm("memory_recall"))))),
  );
  await renderList();
}

// =================================================================== voice

async function voiceTab(root, settings) {
  const cfg = settings.voice;
  const status = await api.get("/interaction/voice").catch(() => ({ voices: [], volc_resources: [] }));
  const set = (key) => (value) => save("voice", { [key]: value }, true);
  const keyInput = h("input.input.mono", { type: "password", value: cfg.api_key, placeholder: cfg.has_key ? "已从环境变量或密钥文件读取" : "sk-…", autocomplete: "off" });
  const volcInput = h("input.input.mono", {
    type: "password", value: cfg.volc_api_key, autocomplete: "off",
    placeholder: cfg.has_volc_key ? "已从环境变量 VOLC_TTS_API_KEY 读取" : "火山引擎豆包语音 API Key",
  });
  const source = { settings: "本页保存", env: "环境变量 DASHSCOPE_API_KEY", file: "~/.config/xiaoxiaobai/dashscope_api_key", none: "未配置" }[cfg.key_source];
  const volcSource = () => ({ settings: "已保存", env: "环境变量 VOLC_TTS_API_KEY", none: "未配置" })[cfg.volc_key_source];
  const volcState = h("div.hint");
  const routeHint = h("div.hint");

  const syncRoute = () => {
    const volc = isVolc(cfg.voice);
    routeSelect.disabled = volc;
    routeSelect.value = volc ? "pi" : cfg.route;
    routeHint.textContent = volc
      ? "当前是火山引擎音色：语音会自动改走树莓派转发（ESP32 固件只能直连千问）。"
      : "直连时 ESP32 自己连 WebSocket，树莓派只负责下发配置";
  };
  const routeSelect = select([["esp32", "ESP32 直连云端（推荐）"], ["pi", "树莓派转发"]], cfg.route, (route) => {
    cfg.route = route;
    set("route")(route);
  });
  const picker = voicePicker({
    groups: status.voices,
    value: cfg.voice,
    hasVolcKey: () => !!cfg.has_volc_key,
    speechRate: () => cfg.speech_rate,
    onChange: async (voice) => {
      await save("voice", { voice }, true);
      cfg.voice = voice;
      syncRoute();
      toast(`音色已切换为「${labelOf(voice, status.voices)}」，下次开始语音对话时生效`, "ok", 2200);
    },
  });
  const syncVolc = () => {
    volcState.textContent = `当前：${volcSource()}。新版控制台的 API Key 直接填；旧版控制台可以填「APP ID:Access Token」。`;
    picker.refresh();
  };
  syncRoute();
  syncVolc();

  const saveVolc = (button) => busy(button, async () => {
    Object.assign(cfg, await save("voice", { volc_api_key: volcInput.value.trim() }));
    volcInput.value = cfg.volc_api_key;
    syncVolc();
  });

  root.append(
    h("div.card",
      h("div.card-head", h("h3", icon("volume", "sm"), "机器人的声音"), h("span.sub", "选哪个音色，小小白就用哪个声音说话")),
      h("div.card-body.stack",
        field("音色", picker),
        h("div.form-grid",
          field("语速", rangeEl({
            min: 0.5, max: 2, step: 0.1, value: cfg.speech_rate ?? 1, format: (v) => `${v.toFixed(1)}×`,
            onchange: (v) => {
              cfg.speech_rate = v;
              set("speech_rate")(v);
            },
          }), "千问 TTS 与火山引擎音色生效；Omni 原生音色固定语速"),
          field("扬声器音量", rangeEl({ min: 0, max: 100, value: cfg.volume, onchange: set("volume") }))))),
    h("div.card",
      h("div.card-head", h("h3", icon("mic", "sm"), "千问实时语音"), h("span.sub", "听懂你、想回答都由 Qwen-Omni Realtime 完成")),
      h("div.card-body.stack",
        h("div.field",
          h("label", "DashScope API Key", h("span.spacer"), h("a.small", { href: "https://bailian.console.aliyun.com/?apiKey=1", target: "_blank", rel: "noopener" }, "获取 API Key")),
          h("div.input-group", keyInput, h("button.btn", {
            onclick: (e) => busy(e.currentTarget, () => save("voice", { api_key: keyInput.value.trim() })),
          }, "保存")),
          h("div.hint", `当前来源：${source}。语音对话始终需要它；语音和文字聊天可以使用不同的服务商。`)),
        h("div.field",
          h("label", "火山引擎 API Key（豆包语音）", h("span.spacer"),
            h("a.small", { href: "https://console.volcengine.com/speech/new/setting/apikeys?projectName=default", target: "_blank", rel: "noopener" }, "获取 API Key")),
          h("div.input-group", volcInput, h("button.btn", { onclick: (e) => saveVolc(e.currentTarget) }, "保存")),
          volcState,
          h("div.hint", "只在使用火山引擎音色时需要。先在豆包语音控制台「开通管理」里开通「豆包语音合成模型 2.0」。")),
        h("div.form-grid",
          h("div.field", h("label", "音频链路"), routeSelect, routeHint),
          field("麦克风增益", rangeEl({ min: 0.5, max: 8, step: 0.1, value: cfg.mic_gain, format: (v) => `${v.toFixed(1)}×`, onchange: set("mic_gain") }))),
        h("div",
          toggleRow("对话时发送摄像头画面", "你说到要看的时候才拍一张给模型（看看、这是什么…），拍完就关，平时摄像头不工作", cfg.camera, set("camera")),
          toggleRow("全双工", "机器人说话时也开着麦克风，可以随时打断（可能听到自己的声音）", cfg.full_duplex, set("full_duplex")),
          toggleRow("联网搜索", "问到时事、天气、价格等或说“联网搜一下…”时模型自己去搜", cfg.search, set("search")),
          toggleRow("转写我说的话", "在字幕和对话历史里显示你说的内容", cfg.show_user_text, set("show_user_text")),
          toggleRow("共用人设与记忆", "语音对话使用「人设」页的设定和已保存的记忆", cfg.share_persona, set("share_persona"))),
        h("details.disclosure",
          h("summary", icon("chevron", "sm"), "高级"),
          h("div", { style: { marginTop: "10px" } },
            h("div.form-grid",
              field("自定义 WebSocket 地址", h("input.input.mono", {
                value: cfg.url, placeholder: "留空使用默认（北京地域）", onchange: (e) => set("url")(e.target.value.trim()),
              }), "千问国际站等其他地域时填写"),
              field("火山引擎 Resource ID", select(
                [["", "自动（按音色判断）"], ...(status.volc_resources || []).map((r) => [r, r])],
                cfg.volc_resource || "", set("volc_resource"),
              ), "2.0 音色用 seed-tts-2.0，1.0 音色用 seed-tts-1.0，复刻音色用 seed-icl-*")))),
        h("div.small.muted", "修改会在下次开始语音对话时生效。"))),
  );
  return () => picker.destroy();
}

// =================================================================== persona

async function personaTab(root, settings) {
  const robot = settings.robot;
  const name = h("input.input", { value: robot.name, maxlength: 24 });
  const persona = h("textarea.textarea", { value: robot.persona, rows: 10, style: { minHeight: "220px" } });
  const counter = h("span.small.muted");
  const count = () => (counter.textContent = `${persona.value.length} 字`);
  persona.addEventListener("input", count);
  count();
  root.append(h("div.card",
    h("div.card-head", h("h3", icon("user", "sm"), "人设"), h("span.sub", "文字聊天与语音对话共用的系统提示词")),
    h("div.card-body.stack",
      field("机器人名字", name, "显示在页面顶部，也会告诉模型"),
      h("div.field", h("label", "性格与说话方式", h("span.spacer"), counter), persona,
        h("div.hint", "系统会自动在后面附上：机器人当前状态、可用的身体动作、关于你的记忆、知识库检索结果。")),
      h("div.row",
        h("button.btn", {
          onclick: () => {
            persona.value = robot.default_persona;
            count();
          },
        }, "恢复默认"),
        h("span.spacer"),
        h("button.btn.primary", {
          onclick: (e) => busy(e.currentTarget, () => save("robot", { name: name.value.trim(), persona: persona.value.trim() })),
        }, "保存")))));
}

// =================================================================== system

async function systemTab(root, settings) {
  const safety = settings.safety;
  const low = h("input.input", { type: "number", step: 0.1, min: 5.5, max: 8, value: safety.low_battery_v });
  const warn = h("input.input", { type: "number", step: 0.1, min: 5.5, max: 8.4, value: safety.warn_battery_v });
  const info = h("div.card-body", h("span.spinner"));

  async function renderInfo() {
    try {
      const s = await api.get("/system/info");
      const mem = s.mem_total_mb ? `${((s.mem_total_mb - s.mem_available_mb) / 1024).toFixed(1)} / ${(s.mem_total_mb / 1024).toFixed(1)} GB` : "—";
      clear(info, h("dl.kv",
        h("dt", "主机名"), h("dd", s.hostname),
        h("dt", "IP 地址"), h("dd.mono", (s.ips || []).join("  ") || "—"),
        h("dt", "Wi-Fi"), h("dd", s.wifi_ssid || "—"),
        h("dt", "ESP32"), h("dd", s.esp32 ? `${s.esp32.ip || "—"}${s.esp32.rssi != null ? ` · ${s.esp32.rssi} dBm` : ""}` : "未连接"),
        h("dt", "显示进程"), h("dd", s.display_running ? h("span.badge.ok", "运行中") : h("span.badge", "未运行")),
        h("dt", "手柄"), h("dd", s.gamepad ? "已连接" : "未检测到"),
        h("dt", "CPU 温度"), h("dd", s.cpu_temp_c != null ? `${s.cpu_temp_c} °C` : "—"),
        h("dt", "负载"), h("dd", s.load ? s.load.map((x) => x.toFixed(2)).join(" / ") : "—"),
        h("dt", "内存"), h("dd", mem),
        h("dt", "磁盘剩余"), h("dd", `${s.disk_free_gb} GB`),
        h("dt", "系统运行"), h("dd", duration(s.uptime_s)),
        h("dt", "API 运行"), h("dd", duration(s.api_uptime_s)),
        h("dt", "版本"), h("dd.mono", `API ${s.api_version} · ${s.git || "—"} · Python ${s.python}`),
        h("dt", "模式"), h("dd", s.mock ? h("span.badge.warn", "模拟模式") : "真机")));
    } catch (error) {
      clear(info, empty("alert", "读取失败", error.message));
    }
  }

  const power = async (action) => {
    const label = action === "reboot" ? "重启" : "关机";
    const ok = await confirmDialog({
      title: `${label}树莓派`,
      message: `会先急停（电机断开力矩），然后${label}。\n${action === "shutdown" ? "关机后需要手动重新上电。" : "大约 1 分钟后重新连接。"}`,
      okText: label,
      danger: true,
    });
    if (!ok) return;
    try {
      const r = await api.post("/system/power", { action });
      toast(r.message, "warn", 8000);
    } catch (error) {
      toast(error.message, "error");
    }
  };

  root.append(
    h("div.grid.cols-2", { style: { alignItems: "start" } },
      h("div.stack",
        h("div.card",
          h("div.card-head", h("h3", icon("battery", "sm"), "电池保护"), h("span.sub", "2S 锂电池，满电约 8.4 V")),
          h("div.card-body.stack",
            h("div.form-grid", field("低电提醒（V）", warn, "低于此电压时弹出提醒"), field("低电保护（V）", low, "低于此电压时禁止走路和做动作")),
            h("div.row", h("span.spacer"), h("button.btn.primary", {
              onclick: (e) => busy(e.currentTarget, () => save("safety", { low_battery_v: Number(low.value), warn_battery_v: Number(warn.value) })),
            }, "保存")))),
        h("div.card",
          h("div.card-head", h("h3", icon("power", "sm"), "电源")),
          h("div.card-body.stack",
            h("div.small.muted", "需要 API 服务用户有免密 sudo 权限（树莓派默认用户通常有）。"),
            h("div.btn-group",
              h("button.btn", { onclick: () => power("reboot") }, icon("refresh", "sm"), "重启"),
              h("button.btn.danger", { onclick: () => power("shutdown") }, icon("power", "sm"), "关机"))))),
      h("div.card",
        h("div.card-head", h("h3", icon("cpu", "sm"), "系统信息"), h("div.actions", h("button.btn.sm.ghost.icon-only", { onclick: renderInfo, title: "刷新" }, icon("refresh", "sm")))),
        info)),
  );
  renderInfo();
}
