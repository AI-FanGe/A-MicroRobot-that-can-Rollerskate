// Settings → memory: long-term facts about the user, and the conversation history.

import * as api from "../api.js";
import { busy, clear, confirmDialog, dateTime, debounce, empty, field, h, icon, markdown, modal, timeAgo, toast } from "../ui.js";

const SOURCE = { manual: "手动添加", chat: "聊天中记住", auto: "自动提取", voice: "语音" };

export default async function memoryTab(root) {
  let query = "";
  let channel = "";
  const memList = h("div.card-body.flush");
  const convList = h("div.card-body.flush");
  const memCount = h("span.sub");

  async function renderMemories() {
    try {
      const items = await api.get(`/memory${query ? `?q=${encodeURIComponent(query)}` : ""}`);
      memCount.textContent = `${items.length} 条`;
      if (!items.length) {
        clear(memList, empty("pin", query ? "没有匹配的记忆" : "还没有记忆", "聊天时说“记住……”，或在上面手动添加。置顶的记忆每次对话都会带上。"));
        return;
      }
      clear(memList, items.map((m) => h("div.list-item",
        h("button.btn.xs.ghost.icon-only", {
          title: m.pinned ? "取消置顶" : "置顶（每次对话都带上）",
          style: { color: m.pinned ? "var(--accent)" : "" },
          onclick: () => api.patch(`/memory/${m.id}`, { pinned: !m.pinned }).then(renderMemories).catch((e) => toast(e.message, "error")),
        }, icon("pin", "sm")),
        h("div.main-col",
          h("div.memory-text", m.text),
          h("div.meta", h("span", SOURCE[m.source] || m.source), h("span", timeAgo(m.updated)))),
        h("div.item-actions",
          h("button.btn.xs.ghost.icon-only", { title: "编辑", onclick: () => editMemory(m) }, icon("edit", "sm")),
          h("button.btn.xs.ghost.icon-only", {
            title: "删除",
            onclick: () => api.del(`/memory/${m.id}`).then(renderMemories).catch((e) => toast(e.message, "error")),
          }, icon("trash", "sm"))))));
    } catch (error) {
      clear(memList, empty("alert", "读取失败", error.message));
    }
  }

  function editMemory(m) {
    const text = h("textarea.textarea", { value: m.text, rows: 4 });
    modal({
      title: "编辑记忆",
      body: field("内容", text),
      footer: (close) => [
        h("button.btn", { onclick: close }, "取消"),
        h("button.btn.primary", {
          onclick: (e) => busy(e.currentTarget, () => api.patch(`/memory/${m.id}`, { text: text.value })
            .then(() => {
              close();
              renderMemories();
            })
            .catch((err) => toast(err.message, "error"))),
        }, "保存"),
      ],
    });
  }

  async function renderConversations() {
    try {
      const items = await api.get(`/conversations${channel ? `?channel=${channel}` : ""}`);
      if (!items.length) {
        clear(convList, empty("message", "暂无对话", "首页聊天和语音对话都会保存在这里。"));
        return;
      }
      clear(convList, items.map((c) => h("div.list-item",
        icon(c.channel === "voice" ? "mic" : "message"),
        h("div.main-col",
          h("div.title", h("span.ellipsis", c.title), h(`span.badge${c.channel === "voice" ? ".accent" : ""}`, c.channel === "voice" ? "语音" : "文字")),
          h("div.meta", h("span", `${c.count} 条消息`), h("span", dateTime(c.updated)))),
        h("div.item-actions",
          h("button.btn.xs", { onclick: () => viewConversation(c) }, "查看"),
          c.channel === "chat" ? h("button.btn.xs", {
            onclick: () => {
              localStorage.setItem("xiaoxiaobai.conversation", c.id);
              location.hash = "#/home";
            },
          }, "继续聊") : null,
          h("button.btn.xs.ghost.icon-only", {
            title: "删除",
            onclick: async () => {
              if (!(await confirmDialog({ title: "删除对话", message: `确定删除「${c.title}」？`, okText: "删除", danger: true }))) return;
              api.del(`/conversations/${c.id}`).then(renderConversations).catch((e) => toast(e.message, "error"));
            },
          }, icon("trash", "sm"))))));
    } catch (error) {
      clear(convList, empty("alert", "读取失败", error.message));
    }
  }

  async function viewConversation(c) {
    try {
      const conv = await api.get(`/conversations/${c.id}`);
      const title = h("input.input", { value: conv.title });
      modal({
        title: "对话记录",
        wide: true,
        body: h("div.stack",
          h("div.input-group", title, h("button.btn", {
            onclick: () => api.patch(`/conversations/${c.id}`, { title: title.value }).then(() => {
              toast("已重命名", "ok");
              renderConversations();
            }).catch((e) => toast(e.message, "error")),
          }, "重命名")),
          h("div.conv-messages", conv.messages.length ? conv.messages.map((m) => h(`div.msg.${m.role}`,
            h("div.avatar", icon(m.role === "user" ? "user" : "bot", "sm")),
            h("div.content",
              m.role === "user" ? h("div.bubble", m.content) : h("div.bubble", { html: markdown(m.content) }),
              h("div.meta-line", h("span", dateTime(m.ts)),
                m.meta?.actions?.length ? h("span", `执行了 ${m.meta.actions.length} 个动作`) : null,
                m.meta?.interrupted ? h("span", "被打断") : null))))
            : empty("message", "没有消息"))),
      });
    } catch (error) {
      toast(error.message, "error");
    }
  }

  const addInput = h("input.input", { placeholder: "例如：主人叫小明，喜欢恐龙，晚上九点睡觉" });
  const add = () => {
    const text = addInput.value.trim();
    if (!text) return;
    api.post("/memory", { text, pinned: false }).then(() => {
      addInput.value = "";
      renderMemories();
    }).catch((e) => toast(e.message, "error"));
  };
  addInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.isComposing) add();
  });
  const seg = h("div.seg");
  const renderSeg = () => clear(seg, [["", "全部"], ["chat", "文字"], ["voice", "语音"]].map(([v, label]) =>
    h(`button${channel === v ? ".active" : ""}`, {
      onclick: () => {
        channel = v;
        renderSeg();
        renderConversations();
      },
    }, label)));
  renderSeg();

  root.append(h("div.grid.cols-2", { style: { alignItems: "start" } },
    h("div.card",
      h("div.card-head", h("h3", icon("pin", "sm"), "长期记忆"), memCount,
        h("div.actions", h("button.btn.sm.ghost", {
          onclick: async () => {
            if (!(await confirmDialog({ title: "清空记忆", message: "删除全部长期记忆？此操作不能撤销。", okText: "清空", danger: true }))) return;
            api.del("/memory").then((r) => {
              toast(r.message, "ok");
              renderMemories();
            }).catch((e) => toast(e.message, "error"));
          },
        }, "清空"))),
      h("div.card-body.stack",
        h("div.input-group", addInput, h("button.btn.primary", { onclick: add }, icon("plus", "sm"), "添加")),
        h("input.input", { placeholder: "搜索记忆…", oninput: debounce((e) => {
          query = e.target.value.trim();
          renderMemories();
        }, 250) })),
      memList),
    h("div.card",
      h("div.card-head", h("h3", icon("clock", "sm"), "对话历史"), h("div.actions", seg)),
      convList)));
  await Promise.all([renderMemories(), renderConversations()]);
}
