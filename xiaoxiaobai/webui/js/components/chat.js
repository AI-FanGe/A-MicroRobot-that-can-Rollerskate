// Chat panel: streams replies from /api/v1/chat, shows tool calls (robot actions),
// knowledge-base citations and memories, and mirrors the realtime voice session.

import * as api from "../api.js";
import { emit, loadSettings, on, store } from "../store.js";
import { clear, dateTime, h, icon, markdown, toast } from "../ui.js";

const CONV_KEY = "xiaoxiaobai.conversation";
const SUGGESTIONS = ["介绍一下你自己", "站起来", "往前走两秒", "跟我挥挥手", "做个开心的表情", "你现在电量多少？"];
const TOOL_LABEL = {
  robot_set_mode: "切换模式",
  robot_walk: "行走",
  robot_stop: "停止",
  robot_gesture: "做动作",
  robot_motion: "编排动作",
  robot_set_expression: "换表情",
  robot_play_sound: "播放提示音",
  robot_get_status: "读取状态",
  memory_save: "记住",
  memory_search: "翻记忆",
  knowledge_search: "检索知识库",
  computer_task: "交给电脑",
  web_search: "联网搜索",
  memo_add: "记到备忘录",
  memo_list: "查备忘录",
  memo_update: "改备忘录",
  memo_show: "屏幕显示清单",
};

function describeArgs(name, args) {
  if (!args) return "";
  if (name === "robot_walk") {
    const parts = [];
    if (args.vx) parts.push(`${args.vx > 0 ? "前进" : "后退"} ${Math.abs(args.vx)}`);
    if (args.vy) parts.push(`${args.vy > 0 ? "左移" : "右移"} ${Math.abs(args.vy)}`);
    if (args.vtheta) parts.push(`${args.vtheta > 0 ? "左转" : "右转"} ${Math.abs(args.vtheta)}`);
    return `${parts.join("，") || "原地"} · ${args.duration_s ?? 2} 秒`;
  }
  if (name === "robot_set_mode") return { stand: "站立", walk: "走跑", skate: "滑板", gymnastics: "广播体操", peek: "偷看", propup: "撑起", liedown: "躺下" }[args.mode] || args.mode;
  if (name === "memory_save") return args.text;
  if (name === "robot_motion") {
    const steps = Array.isArray(args.steps) ? args.steps : [];
    return steps.map((s) => (s.walk ? "走路" : s.gesture ? s.gesture : "自编动作")).join(" → ");
  }
  return Object.values(args).join(" ");
}

function previewTool(name, args) {
  if (name === "robot_walk") {
    store.command.walkVelocity = { vx: args.vx || 0, vy: args.vy || 0, vtheta: args.vtheta || 0 };
    store.command.walkUntil = Date.now() + 1000 * Math.min(10, Number(args.duration_s) || 2);
    emit("command", { walk: true });
  }
}

export function chatPanel() {
  let conversationId = localStorage.getItem(CONV_KEY) || null;
  let conversations = [];
  let controller = null;
  let voiceMsg = null;
  const offs = [];

  const select = h("select.select", { title: "会话", onchange: (e) => openConversation(e.target.value || null) });
  const modelInfo = h("div.model");
  const log = h("div.chat-log");
  const textarea = h("textarea", { rows: 1, placeholder: "和机器人说点什么… 例如“站起来，往前走两秒”", "aria-label": "消息" });
  const sendBtn = h("button.btn.primary.icon-only", { title: "发送（Enter）" }, icon("send"));
  const voiceBtn = h("button.btn.icon-only", { title: "实时语音对话" }, icon("mic"));
  const voiceBar = h("div.voice-live", { hidden: true });
  const foot = h("div.chat-foot", h("span", "Enter 发送 · Shift+Enter 换行"), h("span.spacer"));

  const el = h("div.chat",
    h("div.chat-head",
      select,
      h("button.btn.ghost.sm.icon-only", { title: "新对话", onclick: () => openConversation(null) }, icon("plus")),
      modelInfo),
    log,
    voiceBar,
    h("div.chat-input", h("div.chat-box", textarea, voiceBtn, sendBtn), foot),
  );

  // ---------------------------------------------------------------- header

  async function refreshList() {
    try {
      conversations = await api.get("/conversations?channel=chat");
    } catch {
      conversations = [];
    }
    clear(select,
      h("option", { value: "" }, "新对话"),
      conversations.slice(0, 40).map((c) => h("option", { value: c.id, selected: c.id === conversationId }, c.title)));
    if (conversationId && !conversations.some((c) => c.id === conversationId)) select.value = "";
  }

  async function refreshModel() {
    let settings = null;
    try {
      settings = await loadSettings();
    } catch {
      /* offline */
    }
    const providers = (settings?.llm?.providers || []).filter((p) => p.enabled !== false);
    const active = providers.find((p) => p.id === settings?.llm?.active) || providers[0];
    if (active) {
      clear(modelInfo, h("span.dot.ok"), h("span.ellipsis", { title: active.base_url }, `${active.name} · ${active.model}`));
    } else if (store.snapshot?.mock) {
      clear(modelInfo, h("span.dot.warn"), h("span", "离线演示"), h("a", { href: "#/settings/llm" }, "配置模型"));
    } else {
      clear(modelInfo, h("span.dot.danger"), h("a", { href: "#/settings/llm" }, "未配置大模型，去设置"));
    }
  }

  // ---------------------------------------------------------------- messages

  function welcome() {
    return h("div.chat-welcome",
      h("img", { src: "img/logo.svg", width: 44, height: 44, alt: "" }),
      h("h2", `你好，我是 ${store.snapshot?.robot_name || "小小白"}`),
      h("div.small", "可以和我聊天，也可以让我站起来、走路、换表情。我会参考知识库和记住的事情。"),
      h("div.chat-suggest", SUGGESTIONS.map((s) => h("button", { onclick: () => send(s) }, s))));
  }

  function userMessage(text, meta = {}) {
    return h("div.msg.user",
      h("div.avatar", icon("user", "sm")),
      h("div.content",
        h("div.bubble", text),
        meta.voice ? h("div.meta-line", h("span.tag-voice", icon("mic", "sm"), "语音")) : null));
  }

  function toolChip(name, args, state = "pending", result = null) {
    const stateEl = state === "pending" ? h("span.spinner") : icon(state === "ok" ? "check" : "x", "sm");
    const resultText = result ? (result.error || result.message || result.expression || result.saved || "") : "";
    return h(`div.tool-call.${state === "pending" ? "" : state}`,
      { title: resultText || JSON.stringify(args || {}) },
      icon(name.startsWith("memory") ? "pin" : name.startsWith("memo_") ? "calendar" : name.startsWith("knowledge") ? "book"
        : name === "web_search" ? "search" : "bot", "sm"),
      h("span.name", TOOL_LABEL[name] || name),
      h("span.args", describeArgs(name, args)),
      h("span.state", stateEl));
  }

  function assistantMessage(content, meta = {}) {
    const bubble = h("div.bubble");
    const tools = h("div.tool-calls");
    const extras = h("div");
    const metaLine = h("div.meta-line");
    const msg = h("div.msg.assistant", h("div.avatar", icon("bot", "sm")), h("div.content", tools, bubble, extras, metaLine));
    const view = {
      el: msg,
      text: content || "",
      sources: (meta.sources || []).map((s) => (typeof s === "string" ? { title: s } : s)),
      reasoning: meta.reasoning || "",
      renderQueued: false,
      render(final = false) {
        bubble.innerHTML = markdown(this.text, this.sources);
        bubble.classList.toggle("caret", !final);
        bubble.hidden = !this.text && final;
      },
      queue() {
        if (this.renderQueued) return;
        this.renderQueued = true;
        requestAnimationFrame(() => {
          this.renderQueued = false;
          this.render(false);
          scroll();
        });
      },
      setExtras(memories = []) {
        const parts = [];
        if (this.reasoning) {
          parts.push(h("details.disclosure", h("summary", icon("sparkles", "sm"), "思考过程"), h("div.inner", this.reasoning)));
        }
        if (this.sources.length || memories.length) {
          const label = [this.sources.length ? `${this.sources.length} 条知识` : "", memories.length ? `${memories.length} 条记忆` : ""]
            .filter(Boolean).join(" · ");
          parts.push(h("details.disclosure", h("summary", icon("book", "sm"), `参考了 ${label}`),
            h("div.inner",
              this.sources.map((s, i) => h("div.source-item", h("b", `[${i + 1}] ${s.title}`), s.text ? h("div", s.text) : null)),
              memories.map((m) => h("div.source-item", icon("pin", "sm"), " ", m)))));
        }
        clear(extras, parts);
      },
      setMeta(info) {
        clear(metaLine,
          info.voice ? h("span.tag-voice", icon("mic", "sm"), "语音") : null,
          info.interrupted ? h("span", "被打断") : null,
          info.model ? h("span", info.model) : null,
          info.elapsed_ms ? h("span", `${(info.elapsed_ms / 1000).toFixed(1)} 秒`) : null,
          info.ts ? h("span", dateTime(info.ts)) : null);
      },
      tools,
      bubble,
    };
    for (const a of meta.actions || []) tools.append(toolChip(a.name, a.arguments, a.ok ? "ok" : "fail", a.result));
    view.render(true);
    view.setExtras([]);
    view.setMeta({ ...meta, ts: meta.ts });
    return view;
  }

  function errorMessage(text) {
    return h("div.msg.assistant.error", h("div.avatar", icon("alert", "sm")), h("div.content", h("div.bubble", text)));
  }

  const scroll = () => {
    const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 160;
    if (nearBottom) log.scrollTop = log.scrollHeight;
  };
  const scrollEnd = () => {
    log.scrollTop = log.scrollHeight;
  };

  async function openConversation(id) {
    controller?.abort();
    conversationId = id;
    if (id) localStorage.setItem(CONV_KEY, id);
    else localStorage.removeItem(CONV_KEY);
    select.value = id || "";
    if (!id) {
      clear(log, welcome());
      return;
    }
    clear(log, h("div.empty", h("span.spinner")));
    try {
      const conv = await api.get(`/conversations/${id}`);
      clear(log);
      for (const m of conv.messages) {
        if (m.role === "user") log.append(userMessage(m.content));
        else if (m.role === "assistant") log.append(assistantMessage(m.content, { ...(m.meta || {}), ts: m.ts }).el);
      }
      if (!conv.messages.length) log.append(welcome());
      scrollEnd();
    } catch {
      conversationId = null;
      localStorage.removeItem(CONV_KEY);
      clear(log, welcome());
    }
  }

  // ---------------------------------------------------------------- sending

  function setSending(sending) {
    clear(sendBtn, icon(sending ? "stop" : "send"));
    sendBtn.title = sending ? "停止" : "发送（Enter）";
    sendBtn.classList.toggle("danger", sending);
    sendBtn.classList.toggle("primary", !sending);
  }

  async function send(text) {
    text = (text ?? textarea.value).trim();
    if (!text || controller) return;
    textarea.value = "";
    autosize();
    log.querySelector(".chat-welcome")?.remove();
    log.append(userMessage(text));
    const view = assistantMessage("", {});
    view.render(false);
    log.append(view.el);
    scrollEnd();
    controller = new AbortController();
    setSending(true);
    const chips = new Map();
    let memories = [];
    let finished = false;
    try {
      for await (const ev of api.stream("/chat", { conversation_id: conversationId, message: text }, controller.signal)) {
        if (ev.type === "meta") {
          if (ev.conversation_id !== conversationId) {
            conversationId = ev.conversation_id;
            localStorage.setItem(CONV_KEY, conversationId);
            refreshList();
          }
          view.sources = ev.sources || [];
          memories = ev.memories || [];
          view.setMeta({ model: ev.model });
        } else if (ev.type === "delta") {
          view.text += ev.text;
          view.queue();
        } else if (ev.type === "reasoning") {
          view.reasoning += ev.text;
        } else if (ev.type === "tool_call") {
          const chip = toolChip(ev.name, ev.arguments);
          chips.set(ev.id, { chip, args: ev.arguments });
          view.tools.append(chip);
          previewTool(ev.name, ev.arguments);
          scroll();
        } else if (ev.type === "tool_result") {
          const entry = chips.get(ev.id);
          const chip = toolChip(ev.name, entry?.args, ev.ok ? "ok" : "fail", ev.result);
          if (entry) entry.chip.replaceWith(chip);
          else view.tools.append(chip);
          if (!ev.ok) toast(`${TOOL_LABEL[ev.name] || ev.name}失败：${ev.result?.error || ""}`, "warn");
        } else if (ev.type === "notice") {
          view.tools.append(h("div.small.muted", ev.text));
        } else if (ev.type === "error") {
          view.el.after(errorMessage(ev.message));
        } else if (ev.type === "done") {
          finished = true;
          view.setMeta({ ...(ev.message?.meta || {}), ts: ev.message?.ts });
        }
      }
    } catch (error) {
      view.el.after(errorMessage(error.message));
      if (error.response?.state?.reason === 101) {
        view.el.nextSibling.querySelector(".bubble").append(" ", h("a", { href: "#/settings/llm" }, "去配置"));
      }
    } finally {
      controller = null;
      setSending(false);
      if (!view.text && !view.tools.children.length) view.el.remove();
      else {
        view.render(true);
        view.setExtras(memories);
      }
      if (!finished) refreshList();
      scroll();
    }
  }

  // ---------------------------------------------------------------- voice

  function renderVoice() {
    const voice = store.snapshot?.voice || {};
    voiceBtn.classList.toggle("active", !!voice.running);
    voiceBtn.title = voice.running ? "结束语音对话" : "开始实时语音对话（用机器人的麦克风和扬声器）";
    voiceBar.hidden = !voice.running;
    if (voice.running) {
      const label = { starting: "正在连接语音服务…", ready: "正在聆听，直接对机器人说话", thinking: "思考中…", speaking: "机器人正在说话" }[voice.state] || voice.state;
      clear(voiceBar, h("span.voice-bars", h("i"), h("i"), h("i"), h("i")), h("span", label), h("span.spacer"),
        h("button.btn.xs", { onclick: toggleVoice }, icon("stop", "sm"), "结束"));
    }
  }

  async function toggleVoice() {
    const running = store.snapshot?.voice?.running;
    try {
      const res = await api.post(running ? "/interaction/voice/stop" : "/interaction/voice/start", {});
      toast(res.message, "info");
    } catch (error) {
      toast(error.message, "error", 7000);
      if (error.response?.state?.reason === 101) location.hash = "#/settings/voice";
    }
  }

  function voiceUser(ev) {
    log.querySelector(".chat-welcome")?.remove();
    log.append(userMessage(ev.text, { voice: true }));
    scroll();
  }

  function voiceDelta(ev) {
    if (!voiceMsg) {
      log.querySelector(".chat-welcome")?.remove();
      voiceMsg = assistantMessage("", {});
      voiceMsg.setMeta({ voice: true });
      log.append(voiceMsg.el);
    }
    voiceMsg.text += ev.text;
    voiceMsg.queue();
  }

  function voiceReply(ev) {
    if (!voiceMsg) voiceDelta({ text: "" });
    voiceMsg.text = ev.text;
    voiceMsg.render(true);
    voiceMsg.setMeta({ voice: true, interrupted: ev.interrupted, ts: ev.ts });
    voiceMsg = null;
    scroll();
  }

  // ---------------------------------------------------------------- input

  function autosize() {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(160, textarea.scrollHeight)}px`;
  }
  textarea.addEventListener("input", autosize);
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      send();
    }
  });
  sendBtn.addEventListener("click", () => (controller ? controller.abort() : send()));
  voiceBtn.addEventListener("click", toggleVoice);

  offs.push(
    on("status", renderVoice),
    on("voice_user", voiceUser),
    on("voice_reply_delta", voiceDelta),
    on("voice_reply", voiceReply),
    on("settings", () => setTimeout(refreshModel, 100)),
  );

  refreshList();
  refreshModel();
  renderVoice();
  openConversation(conversationId);

  return {
    el,
    focus: () => textarea.focus(),
    dispose() {
      controller?.abort();
      offs.forEach((off) => off());
    },
  };
}
