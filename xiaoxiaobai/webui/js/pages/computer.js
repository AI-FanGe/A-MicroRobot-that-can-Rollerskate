// Computer: hand tasks to WorkBuddy on a PC and watch it work, step by step; answer what it asks,
// and carry on a finished task (tasks share one WorkBuddy session, so a follow-up has context).

import * as api from "../api.js";
import { on } from "../store.js";
import { busy, clear, duration, empty, h, icon, markdown, timeAgo, toast } from "../ui.js";

const STATUS = {
  queued: ["排队中", "accent"], running: ["执行中", "accent"], done: ["完成", "ok"], failed: ["失败", "danger"], cancelled: ["已取消", ""],
};
const ORIGIN = { voice: "语音", chat: "聊天", api: "接口", web: "网页" };
const SUGGEST = ["打开记事本，写一句“你好，我是小小白”", "把桌面截个图保存到桌面", "看看电脑还剩多少磁盘空间", "用浏览器打开 bilibili"];
const live = (t) => t.status === "queued" || t.status === "running";

export default {
  title: "电脑",
  async mount(container) {
    const tasks = new Map();
    const cards = new Map();
    const expanded = new Set(); // finished tasks the user opened
    const openOutputs = new Set(); // "<task>:<step>" whose output is unfolded
    const drafts = new Map(); // typed text by field, kept across re-renders
    const picks = new Map(); // "<task>:<question>" -> options chosen, not sent yet
    let newest = null;

    const statusEl = h("div.small.muted", h("span.spinner"), " 正在连接电脑…");
    const input = h("textarea.textarea", { rows: 2, placeholder: "想让电脑做什么？例如：打开记事本，写一句“你好，我是小小白”" });
    const sendBtn = h("button.btn.primary", { onclick: (e) => send(e.currentTarget) }, icon("send", "sm"), "交给电脑");
    const list = h("div.stack");

    async function check() {
      try {
        const s = await api.get("/workbuddy");
        const who = s.peer?.name ? `「${s.peer.name}」` : s.url ? s.url.replace(/^https?:\/\//, "") : "";
        const tone = s.authorized ? "ok" : s.reachable ? "warn" : "danger";
        const label = !s.enabled ? "未启用" : s.authorized ? "已连接" : s.reachable ? "密码不对" : "未连接";
        clear(statusEl, h(`span.badge.${s.enabled ? tone : ""}`, label), " ",
          s.authorized ? `${who} 在线，说“让电脑帮我……”也行` : s.message || "",
          s.authorized ? null : h("a", { href: "#/settings/computer", style: { marginLeft: "6px" } }, "去连接电脑 ›"));
      } catch (error) {
        clear(statusEl, h("span.badge.danger", "出错"), " ", error.message);
      }
    }

    async function send(button, text = input.value.trim()) {
      if (!text) return;
      await busy(button, async () => {
        try {
          const res = await api.post("/workbuddy/tasks", { text });
          input.value = "";
          upsert(res.data);
        } catch (error) {
          toast(error.message, "error");
        }
      });
    }

    function field(key, attrs) {
      return h("input.input", {
        ...attrs, value: drafts.get(key) || "", dataset: { draft: key },
        oninput: (e) => drafts.set(key, e.target.value),
      });
    }

    async function answer(button, t, answers) {
      await busy(button, async () => {
        try {
          const res = await api.post(`/workbuddy/tasks/${t.id}/answer`, { answers });
          for (const id of Object.keys(answers)) {
            drafts.delete(`${t.id}:${id}`);
            picks.delete(`${t.id}:${id}`);
          }
          upsert(res.data);
        } catch (error) {
          toast(error.message, "error");
        }
      });
    }

    function questionForm(t, q) {
      const open = q.items.filter((item) => !(item.id in q.answers));
      const collect = () => {
        const answers = {};
        for (const item of open) {
          const typed = (drafts.get(`${t.id}:${item.id}`) || "").trim();
          const chosen = [...(picks.get(`${t.id}:${item.id}`) || [])];
          if (typed) answers[item.id] = typed;
          else if (chosen.length) answers[item.id] = item.multi ? chosen : chosen[0];
        }
        return answers;
      };
      const submit = h("button.btn.primary.sm", {
        onclick: (e) => {
          const answers = collect();
          if (!Object.keys(answers).length) return toast("先选一个或写一句回答", "warn");
          answer(e.currentTarget, t, answers);
        },
      }, icon("send", "sm"), "回答");
      const instant = open.length === 1 && !open[0].multi; // one choice to make: a tap answers
      return h("div.wb-ask-form",
        open.map((item) => {
          const key = `${t.id}:${item.id}`;
          const chosen = picks.get(key) || new Set();
          return h("div.wb-ask-item",
            item.header ? h("div.wb-ask-header", item.header) : null,
            h("div.wb-ask-q", item.question),
            item.options.length ? h("div.wb-opts", item.options.map((o) => h(`button.btn.sm.wb-opt${chosen.has(o.label) ? ".picked" : ""}`, {
              title: o.description || null,
              onclick: (e) => {
                if (instant) return answer(e.currentTarget, t, { [item.id]: o.label });
                const next = new Set(item.multi ? chosen : []);
                next.has(o.label) ? next.delete(o.label) : next.add(o.label);
                picks.set(key, next);
                render(t);
              },
            }, h("span", o.label), o.description ? h("small", o.description) : null))) : null,
            field(key, {
              placeholder: item.options.length ? "或者写你自己的回答…" : "写下你的回答…",
              onkeydown: (e) => e.key === "Enter" && !e.isComposing && submit.click(),
            }));
        }),
        h("div.row", h("span.small.muted", "小小白开着语音对话时会把问题念出来，直接说答案也行"), h("span.spacer"), submit));
    }

    function askStep(t, s) {
      const q = t.question && t.question.id === s.id ? t.question : null;
      const said = (id) => {
        const a = (q ? q.answers : s.answers || {})[id];
        return Array.isArray(a) ? a.join("、") : a;
      };
      return h(`div.wb-ask.${q ? "waiting" : s.status}`,
        h("div.wb-ask-title", icon("message", "sm"), q ? "电脑在问你" : "电脑问了你"),
        q ? questionForm(t, q) : (s.items || []).map((item) => h("div.wb-ask-item",
          h("div.wb-ask-q", item.question),
          said(item.id) ? h("div.wb-ask-a", "你的回答：", h("b", said(item.id)))
            : h("div.small.muted", s.status === "waiting" ? "等你回答…" : "没有回答（超时或任务已停止），电脑会自己决定"))));
    }

    function replyBox(t) {
      const key = `${t.id}:reply`;
      const go = (button) => {
        const text = (drafts.get(key) || "").trim();
        if (!text) return;
        busy(button, async () => {
          try {
            const res = await api.post("/workbuddy/tasks", { text, reply_to: t.id });
            drafts.delete(key);
            upsert(res.data);
          } catch (error) {
            toast(error.message, "error");
          }
        });
      };
      const btn = h("button.btn.sm", { onclick: (e) => go(e.currentTarget) }, icon("send", "sm"), "接着说");
      return h("div.row.wb-reply",
        field(key, {
          placeholder: /[？?]\s*$/.test(t.result || "") ? "回答电脑的问题，或者接着吩咐…" : "接着这个任务说点什么…",
          onkeydown: (e) => e.key === "Enter" && !e.isComposing && btn.click(),
        }),
        btn);
    }

    function step(task, s, i) {
      if (s.type === "say") return h("div.wb-say", { html: markdown(s.text) });
      if (s.type === "ask") return askStep(task, s);
      const key = `${task.id}:${s.id || i}`;
      const running = live(task) && !["completed", "failed"].includes(s.status);
      const mark = s.status === "failed" ? icon("x", "sm") : running ? h("span.spinner") : s.status === "completed" ? icon("check", "sm") : icon("clock", "sm");
      const label = s.detail || s.title || "执行操作";
      const code = s.code && s.code !== label ? s.code : s.title && s.title !== label ? s.title : "";
      const took = s.finished && s.started ? `${Math.max(0.1, s.finished - s.started).toFixed(1)} 秒` : "";
      const out = s.output ? h("details.wb-out", {
        open: openOutputs.has(key),
        ontoggle: (e) => (e.target.open ? openOutputs.add(key) : openOutputs.delete(key)),
      }, h("summary", "输出"), h("pre", s.output)) : null;
      return h(`div.wb-step.${s.status === "failed" ? "failed" : running ? "running" : "done"}`,
        h("span.wb-mark", mark),
        h("div.wb-body",
          h("div.wb-label", h("span", label), took ? h("small.muted", took) : null),
          code ? h("code.wb-code", code) : null,
          out));
    }

    function card(t) {
      const [label, tone] = t.question ? ["等你回答", "warn"] : STATUS[t.status] || [t.status, ""];
      const isLive = live(t);
      const parent = t.reply_to && tasks.get(t.reply_to);
      const open = isLive || t.id === newest || expanded.has(t.id);
      const elapsed = (t.finished || Date.now() / 1000) - t.created;
      const tools = t.steps.filter((s) => s.type === "tool").length;
      const head = h("div.wb-head", {
        onclick: () => {
          if (isLive || t.id === newest) return;
          expanded.has(t.id) ? expanded.delete(t.id) : expanded.add(t.id);
          render(t);
        },
      },
        h(`span.badge${tone ? `.${tone}` : ""}`, isLive && !t.question ? h("span.spinner") : null, label),
        h("div.wb-title",
          h("b", t.text),
          h("div.meta", h("span", ORIGIN[t.origin] || t.origin), h("span", timeAgo(t.created)),
            parent ? h("span", `接着「${parent.text.slice(0, 12)}${parent.text.length > 12 ? "…" : ""}」`) : null,
            isLive ? h("span.wb-elapsed", { dataset: { created: t.created } }, `已用 ${duration(elapsed)}`)
              : h("span", `用时 ${duration(Math.max(1, elapsed))}`),
            tools ? h("span", `${tools} 步操作`) : null)),
        isLive ? h("button.btn.xs", {
          onclick: (e) => {
            e.stopPropagation();
            busy(e.currentTarget, async () => {
              try {
                upsert(await api.post(`/workbuddy/tasks/${t.id}/cancel`, {}));
              } catch (error) {
                toast(error.message, "error");
              }
            });
          },
        }, "停止") : open ? null : icon("chevronR", "sm"));
      if (!open) {
        const summary = (t.error && !t.result ? t.error : t.result || "").replace(/\s+/g, " ").slice(0, 90);
        return h("div.card.wb-task", head, summary ? h("div.wb-summary", summary) : null);
      }
      const waiting = isLive && !t.steps.length && !t.thinking && !t.result;
      return h(`div.card.wb-task${isLive ? ".live" : ""}`, head,
        h("div.wb-timeline",
          waiting ? h("div.wb-thinking", h("span.spinner"), t.status === "queued" ? "排队中，等前一个任务做完…" : "已发给电脑，等它开始…") : null,
          t.steps.map((s, i) => step(t, s, i)),
          isLive && t.thinking ? h("div.wb-thinking", icon("sparkles", "sm"), h("span", "思考中：", t.thinking.replace(/\s+/g, " ").slice(-120))) : null,
          t.result ? h(`div.wb-result${isLive ? ".streaming" : ""}`, { html: markdown(t.result) }) : null,
          t.error && t.error !== t.result ? h("div.wb-error", icon("alert", "sm"), h("span", t.error)) : null,
          !isLive && t.via === "runs" && !t.steps.length ? h("div.small.muted", "这台电脑上的 WorkBuddy 版本较旧，只能看到结果，看不到过程。") : null),
        isLive ? null : replyBox(t));
    }

    function render(t) {
      const el = card(t);
      const old = cards.get(t.id);
      const typing = old?.contains(document.activeElement) ? document.activeElement.dataset.draft : null;
      if (old?.isConnected) old.replaceWith(el);
      else list.prepend(el);
      cards.set(t.id, el);
      const again = typing && [...el.querySelectorAll("[data-draft]")].find((f) => f.dataset.draft === typing);
      if (again) {
        again.focus();
        again.setSelectionRange(again.value.length, again.value.length);
      }
    }

    function upsert(t) {
      if (!t?.id) return;
      const isNew = !tasks.has(t.id);
      tasks.set(t.id, t);
      if (isNew && (!newest || t.created >= tasks.get(newest).created)) {
        const prev = newest && tasks.get(newest);
        newest = t.id;
        if (prev) render(prev);
      }
      if (list.querySelector(".empty")) clear(list);
      render(t);
    }

    async function load() {
      try {
        const items = await api.get("/workbuddy/tasks");
        clear(list);
        if (!items.length) {
          list.append(empty("monitor", "还没有交给电脑的任务", "在上面输入一句，或对小小白说“让电脑帮我……”。"));
          return;
        }
        newest = items[0].id;
        for (const t of [...items].reverse()) {
          tasks.set(t.id, t);
          render(t);
        }
      } catch (error) {
        clear(list, empty("alert", "读取失败", error.message));
      }
    }

    container.append(h("div.stack.wb-page",
      h("div.card",
        h("div.card-body.stack",
          h("div.row", statusEl, h("span.spacer"), h("a.btn.ghost.sm", { href: "#/settings/computer" }, icon("settings", "sm"), "连接设置")),
          input,
          h("div.row.wb-suggest", SUGGEST.map((s) => h("button.btn.xs", { onclick: () => (input.value = s, input.focus()) }, s))),
          h("div.row", h("span.small.muted", "直接发给电脑，不经过大模型"), h("span.spacer"), sendBtn))),
      list));

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send(sendBtn);
    });
    const offs = [
      on("workbuddy", (e) => upsert(e.task)),
      on("workbuddy_peer", () => check()),
    ];
    const timer = setInterval(() => {
      for (const el of list.querySelectorAll(".wb-elapsed")) el.textContent = `已用 ${duration(Date.now() / 1000 - Number(el.dataset.created))}`;
    }, 1000);
    check();
    await load();
    return () => {
      clearInterval(timer);
      offs.forEach((off) => off());
    };
  },
};
