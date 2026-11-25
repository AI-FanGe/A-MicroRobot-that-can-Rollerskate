// Settings → knowledge base: documents the assistant searches before answering.

import * as api from "../api.js";
import { busy, clear, confirmDialog, dateTime, debounce, empty, field, h, icon, modal, switchEl, toast } from "../ui.js";

const TEXT_TYPES = /\.(txt|md|markdown|csv|json|log|ya?ml|ini|html?)$/i;

export default async function knowledgeTab(root) {
  const list = h("div.card-body.flush");
  const stats = h("span.sub");
  const results = h("div");
  const fileInput = h("input", { type: "file", multiple: true, accept: ".txt,.md,.markdown,.csv,.json,.log,.yaml,.yml,.html", hidden: true });

  async function render() {
    try {
      const data = await api.get("/knowledge");
      stats.textContent = `${data.stats.documents} 篇文档 · ${data.stats.chunks} 个片段 · ${(data.stats.chars / 1000).toFixed(1)}k 字`;
      if (!data.documents.length) {
        clear(list, empty("book", "知识库是空的", "添加说明书、家庭信息、常见问答等，机器人回答时会先检索这里。"));
        return;
      }
      clear(list, data.documents.map((d) => h("div.list-item",
        icon("file"),
        h("div.main-col",
          h("div.title", h("span.ellipsis", d.title), d.enabled ? null : h("span.badge", "已停用")),
          h("div.meta", h("span", `${d.chars} 字`), h("span", `${d.chunks} 个片段`), h("span", dateTime(d.created)), d.source !== "manual" ? h("span", d.source) : null)),
        h("div.item-actions",
          switchEl(d.enabled, (on) => api.patch(`/knowledge/${d.id}`, { enabled: on }).then(render).catch((e) => toast(e.message, "error"))),
          h("button.btn.xs.ghost.icon-only", { title: "查看 / 编辑", onclick: () => openDoc(d.id) }, icon("edit", "sm")),
          h("button.btn.xs.ghost.icon-only", {
            title: "删除",
            onclick: async () => {
              if (!(await confirmDialog({ title: "删除文档", message: `确定删除「${d.title}」？`, okText: "删除", danger: true }))) return;
              api.del(`/knowledge/${d.id}`).then(render).catch((e) => toast(e.message, "error"));
            },
          }, icon("trash", "sm"))))));
    } catch (error) {
      clear(list, empty("alert", "读取失败", error.message));
    }
  }

  function editor({ title = "", text = "", id = null } = {}) {
    const titleInput = h("input.input", { value: title, placeholder: "例如：小小白使用说明" });
    const textArea = h("textarea.textarea", { value: text, rows: 16, placeholder: "粘贴纯文本或 Markdown。空行分段，检索以段落为单位。", style: { minHeight: "320px" } });
    modal({
      title: id ? "编辑文档" : "新建文档",
      wide: true,
      body: h("div.stack", field("标题", titleInput), field("内容", textArea)),
      footer: (close) => [
        h("button.btn", { onclick: close }, "取消"),
        h("button.btn.primary", {
          onclick: (e) => busy(e.currentTarget, async () => {
            try {
              const body = { title: titleInput.value, text: textArea.value };
              if (id) await api.patch(`/knowledge/${id}`, body);
              else await api.post("/knowledge", body);
              toast("已保存", "ok");
              close();
              render();
            } catch (error) {
              toast(error.message, "error");
            }
          }),
        }, "保存"),
      ],
    });
  }

  async function openDoc(id) {
    try {
      const doc = await api.get(`/knowledge/${id}`);
      editor({ title: doc.title, text: doc.text, id });
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function upload(files) {
    let added = 0;
    for (const file of files) {
      if (!TEXT_TYPES.test(file.name)) {
        toast(`${file.name}：暂只支持纯文本 / Markdown 等文本文件`, "warn");
        continue;
      }
      try {
        const text = await file.text();
        await api.post("/knowledge", { title: file.name.replace(/\.[^.]+$/, ""), text, source: file.name });
        added += 1;
      } catch (error) {
        toast(`${file.name}：${error.message}`, "error");
      }
    }
    if (added) toast(`已添加 ${added} 篇文档`, "ok");
    render();
  }

  fileInput.addEventListener("change", () => {
    upload([...fileInput.files]);
    fileInput.value = "";
  });
  const drop = h("div.dropzone", { onclick: () => fileInput.click() }, icon("upload", "lg"),
    h("b", "拖入文件或点击上传"), h("div.small", "支持 .txt / .md / .csv / .json 等文本文件，单篇最多 40 万字"));
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    upload([...e.dataTransfer.files]);
  });

  const search = debounce(async (q) => {
    if (!q.trim()) {
      clear(results);
      return;
    }
    try {
      const hits = await api.get(`/knowledge/search?q=${encodeURIComponent(q)}&k=5`);
      clear(results, hits.length
        ? hits.map((x, i) => h("div.hit", h("div.head", h("span.cite", i + 1), x.title, h("span.spacer"), h("span.badge", `相关度 ${x.score}`)), h("div.body", x.text)))
        : h("div.small.muted", "没有匹配的片段"));
    } catch (error) {
      clear(results, h("div.small.muted", error.message));
    }
  }, 250);

  root.append(
    h("div.grid.cols-2", { style: { alignItems: "start" } },
      h("div.card",
        h("div.card-head", h("h3", icon("book", "sm"), "文档"), stats,
          h("div.actions", h("button.btn.sm.primary", { onclick: () => editor() }, icon("plus", "sm"), "新建"))),
        list),
      h("div.stack",
        h("div.card", h("div.card-body", drop, fileInput)),
        h("div.card",
          h("div.card-head", h("h3", icon("search", "sm"), "检索测试"), h("span.sub", "看看机器人会参考哪些内容")),
          h("div.card-body.stack",
            h("input.input", { placeholder: "输入一个问题…", oninput: (e) => search(e.target.value) }),
            results)),
        h("div.small.muted", "检索方式：本地 BM25（中文按字二元组切分），不需要向量模型，也不会把文档上传到任何地方；只有命中的片段会随问题发给大模型。"))),
  );
  await render();
}
