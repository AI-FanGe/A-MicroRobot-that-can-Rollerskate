// Memo: the robot's calendar and notebook. Everything the user asked it to remember, by day,
// with the undated notes; shown on the face screen on request, exported as CSV / Markdown / JSON / ICS.

import * as api from "../api.js";
import { on } from "../store.js";
import { busy, clear, confirmDialog, debounce, empty, field, h, icon, modal, select, switchEl, timeAgo, toast } from "../ui.js";

const WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"];
const SOURCE = { manual: "网页", chat: "文字聊天", voice: "语音", auto: "自动", api: "接口" };
const EXPORTS = [["csv", "CSV（Excel）"], ["md", "Markdown"], ["json", "JSON"], ["ics", "日历 ICS"]];

// Dates are "YYYY-MM-DD" strings in the robot's time zone; Date objects only in UTC to step days.
const parse = (s) => new Date(`${s}T00:00:00Z`);
const iso = (d) => d.toISOString().slice(0, 10);
const addDays = (s, n) => iso(new Date(parse(s).getTime() + n * 86400000));
const weekday = (s) => `周${WEEKDAYS[(parse(s).getUTCDay() + 6) % 7]}`;

function dayLabel(s, today) {
  const diff = Math.round((parse(s) - parse(today)) / 86400000);
  const near = { "-2": "前天", "-1": "昨天", 0: "今天", 1: "明天", 2: "后天" }[diff];
  const d = parse(s);
  return `${d.getUTCMonth() + 1}月${d.getUTCDate()}日 ${weekday(s)}${near ? `（${near}）` : ""}`;
}

export default {
  title: "备忘录",
  async mount(container) {
    let items = [];
    let today = "";
    let nowText = "";
    let selected = "";
    let month = "";
    let query = "";
    let settings = null;
    const offs = [];

    const nowLine = h("span.sub");
    const calTitle = h("b");
    const calGrid = h("div.cal-grid");
    const dayTitle = h("h3");
    const dayCount = h("span.sub");
    const dayList = h("div.card-body.flush");
    const notesList = h("div.card-body.flush");
    const notesCount = h("span.sub");
    const overdueCard = h("div.card", { hidden: true });
    const searchCard = h("div.card", { hidden: true });
    const settingsBody = h("div.card-body.stack");

    async function load() {
      try {
        const data = await api.get("/memo");
        items = data.items;
        today = data.today;
        nowText = data.now;
        nowLine.textContent = `现在 ${nowText}（${data.timezone || "系统时区"}）`;
        if (!selected) selected = today;
        if (!month) month = selected.slice(0, 7);
        render();
      } catch (error) {
        clear(dayList, empty("alert", "读取失败", error.message));
      }
    }

    const reload = debounce(load, 300);

    function render() {
      renderCalendar();
      renderDay();
      renderNotes();
      renderOverdue();
      renderSearch();
    }

    // ---- calendar ----
    function renderCalendar() {
      const first = `${month}-01`;
      const [y, m] = month.split("-").map(Number);
      calTitle.textContent = `${y} 年 ${m} 月`;
      const start = addDays(first, -((parse(first).getUTCDay() + 6) % 7));
      const byDay = new Map();
      for (const it of items) {
        if (!it.date) continue;
        if (!byDay.has(it.date)) byDay.set(it.date, []);
        byDay.get(it.date).push(it);
      }
      const cells = WEEKDAYS.map((d) => h("div.cal-dow", d));
      for (let i = 0; i < 42; i++) {
        const day = addDays(start, i);
        if (i >= 35 && day.slice(0, 7) !== month) break;
        const list = byDay.get(day) || [];
        const open = list.filter((x) => !x.done);
        const cls = [day.slice(0, 7) !== month && "other", day === today && "today", day === selected && "selected"].filter(Boolean);
        cells.push(h(`button.cal-day${cls.map((c) => "." + c).join("")}`, {
          title: list.map((x) => `${x.time ? x.time + " " : ""}${x.text}${x.done ? "（已完成）" : ""}`).join("\n"),
          onclick: () => {
            selected = day;
            if (day.slice(0, 7) !== month) month = day.slice(0, 7);
            render();
          },
        },
        h("span.num", String(parse(day).getUTCDate())),
        open.length ? h("span.peek", open[0].text) : null,
        list.length ? h("span.dots", list.slice(0, 6).map((x) => h(`span.dot-memo${x.done ? ".done" : ""}`))) : null));
      }
      clear(calGrid, cells);
    }

    function shiftMonth(n) {
      const [y, m] = month.split("-").map(Number);
      const d = new Date(Date.UTC(y, m - 1 + n, 1));
      month = iso(d).slice(0, 7);
      renderCalendar();
    }

    // ---- lists ----
    function memoRow(m, { showDate = false } = {}) {
      const check = h("input.memo-check", {
        type: "checkbox", checked: !!m.done, title: m.done ? "标记为未完成" : "标记为完成",
        onchange: (e) => api.patch(`/memo/${m.id}`, { done: e.target.checked }).then(load).catch((err) => toast(err.message, "error")),
      });
      return h(`div.list-item.memo-item${m.done ? ".done" : ""}`,
        check,
        h("div.main-col",
          h("div.memory-text",
            showDate && m.date ? h("span.badge", dayLabel(m.date, today)) : null,
            m.time ? h("span.badge.accent", m.time) : null,
            " ", m.text),
          h("div.meta", h("span", SOURCE[m.source] || m.source || ""), h("span", `记于 ${timeAgo(m.created)}`))),
        h("div.item-actions",
          h("button.btn.xs.ghost.icon-only", { title: "编辑", onclick: () => edit(m) }, icon("edit", "sm")),
          h("button.btn.xs.ghost.icon-only", {
            title: "删除",
            onclick: () => api.del(`/memo/${m.id}`).then(load).catch((e) => toast(e.message, "error")),
          }, icon("trash", "sm"))));
    }

    function sortKey(m) {
      return `${m.date || "9999-99-99"} ${m.time || "99:99"} ${String(m.created).padStart(14, "0")}`;
    }

    function renderDay() {
      const list = items.filter((m) => m.date === selected).sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
      dayTitle.replaceChildren(icon("calendar", "sm"), dayLabel(selected, today));
      dayCount.textContent = list.length ? `${list.filter((m) => !m.done).length} 项待办 / 共 ${list.length} 项` : "";
      clear(dayList, list.length ? list.map((m) => memoRow(m))
        : empty("calendar", "这天没有安排", "在上面添加，或者对机器人说“提醒我明天下午三点开会”。"));
    }

    function renderNotes() {
      const list = items.filter((m) => !m.date).sort((a, b) => (a.done - b.done) || (b.created - a.created));
      notesCount.textContent = `${list.length} 条`;
      clear(notesList, list.length ? list.map((m) => memoRow(m))
        : empty("pin", "还没有随手记", "没有日期的事情放在这里，比如“记住车停在 B2-115”。"));
    }

    function renderOverdue() {
      const list = items.filter((m) => m.date && m.date < today && !m.done).sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
      overdueCard.hidden = !list.length;
      clear(overdueCard,
        h("div.card-head", h("h3", icon("alert", "sm"), "过期未完成"), h("span.sub", `${list.length} 项`)),
        h("div.card-body.flush", list.map((m) => memoRow(m, { showDate: true }))));
    }

    function renderSearch() {
      searchCard.hidden = !query;
      if (!query) return;
      const needle = query.toLowerCase();
      const list = items.filter((m) => m.text.toLowerCase().includes(needle)).sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
      clear(searchCard,
        h("div.card-head", h("h3", icon("search", "sm"), `搜索“${query}”`), h("span.sub", `${list.length} 条`)),
        h("div.card-body.flush", list.length ? list.map((m) => memoRow(m, { showDate: true })) : empty("search", "没有匹配的备忘")));
    }

    // ---- editing ----
    function edit(m) {
      const text = h("textarea.textarea", { value: m.text, rows: 3 });
      const date = h("input.input", { type: "date", value: m.date || "" });
      const time = h("input.input", { type: "time", value: m.time || "" });
      modal({
        title: "编辑备忘",
        body: h("div.stack", field("内容", text),
          h("div.grid.cols-2", field("日期", date, "清空就变成随手记"), field("时间", time, "可以不填"))),
        footer: (close) => [
          h("button.btn", { onclick: close }, "取消"),
          h("button.btn.primary", {
            onclick: (e) => busy(e.currentTarget, () => api.patch(`/memo/${m.id}`, { text: text.value, date: date.value, time: time.value })
              .then(() => {
                close();
                load();
              })
              .catch((err) => toast(err.message, "error"))),
          }, "保存"),
        ],
      });
    }

    function addForm(dated) {
      const text = h("input.input", { placeholder: dated ? "要做的事，例如：和小王开会" : "要记住的事，例如：车停在 B2-115" });
      const time = dated ? h("input.input", { type: "time", style: { maxWidth: "120px" }, title: "时间（可不填）" }) : null;
      const add = (button) => {
        if (!text.value.trim()) return;
        busy(button, () => api.post("/memo", { text: text.value, date: dated ? selected : null, time: time?.value || null })
          .then(() => {
            text.value = "";
            if (time) time.value = "";
            load();
          })
          .catch((e) => toast(e.message, "error")));
      };
      const button = h("button.btn.primary", { onclick: (e) => add(e.currentTarget) }, icon("plus", "sm"), "添加");
      text.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.isComposing) add(button);
      });
      return h("div.input-group", text, time, button);
    }

    // ---- screen ----
    async function show(body, button) {
      await busy(button, () => api.post("/memo/show", body).then((r) => toast(r.message || "已显示", "ok"))
        .catch((e) => toast(e.message, "error")));
    }

    // ---- settings ----
    async function renderSettings() {
      try {
        settings = (await api.get("/settings")).memo;
      } catch (error) {
        clear(settingsBody, empty("alert", "读取设置失败", error.message));
        return;
      }
      const save = (values) => api.patch("/settings/memo", values).then((r) => {
        settings = r.data;
        toast("已保存", "ok");
        load();
      }).catch((e) => {
        toast(e.message, "error");
        renderSettings();
      });
      const row = (title, hint, key) => h("div.toggle-row", h("div.text", h("b", title), h("small", hint)),
        switchEl(settings[key] !== false, (v) => save({ [key]: v })));
      const tz = h("input.input", { value: settings.timezone || "", placeholder: "Asia/Shanghai" });
      clear(settingsBody,
        row("启用备忘录", "关掉后聊天和语音都不再记事、也不带备忘录", "enabled"),
        row("语音记事", "对机器人说“提醒我…”“记一下…”“…做完了”时自动记进来（用大模型理解）", "voice_capture"),
        row("在屏幕上显示", "问“今天要做什么”时把清单显示在脸上的点阵屏", "show_on_screen"),
        row("到点提醒", "带时间的备忘到点时响提示音、屏幕显示并在网页弹出提醒", "remind"),
        field("时区", h("div.input-group", tz, h("button.btn", { onclick: () => save({ timezone: tz.value.trim() }) }, "保存")),
          "“今天”按这个时区算（机器人系统时钟可能是别的时区），例如 Asia/Shanghai"),
        field("屏幕显示时长", select([[15, "15 秒"], [30, "30 秒"], [45, "45 秒"], [60, "60 秒"], [90, "90 秒"]],
          settings.board_seconds || 30, (v) => save({ board_seconds: Number(v) }))));
    }

    // ---- layout ----
    const exportSelect = select([["", "导出…"], ...EXPORTS], "", (fmt) => {
      if (!fmt) return;
      window.location.href = `/api/v1/memo/export?format=${fmt}`;
      exportSelect.value = "";
    });
    exportSelect.style.width = "160px";
    const search = h("input.input", { placeholder: "搜索备忘…", style: { maxWidth: "220px" }, oninput: debounce((e) => {
      query = e.target.value.trim();
      renderSearch();
    }, 200) });

    container.append(
      h("div.card", { style: { marginBottom: "16px" } },
        h("div.card-head", h("h3", icon("clock", "sm"), "时间"), nowLine,
          h("div.actions",
            h("button.btn.sm", { onclick: (e) => show({ clock: true }, e.currentTarget) }, icon("clock", "sm"), "屏幕显示时钟"),
            h("button.btn.sm", { onclick: (e) => show({ date: today }, e.currentTarget) }, icon("monitor", "sm"), "屏幕显示今天"),
            h("button.btn.sm.ghost", {
              onclick: (e) => busy(e.currentTarget, () => api.post("/memo/hide").then(() => toast("已收起", "ok")).catch((err) => toast(err.message, "error"))),
            }, "收起"))),
        h("div.card-body.row.wrap", search, exportSelect,
          h("button.btn.sm.ghost", {
            onclick: async () => {
              if (!(await confirmDialog({ title: "清除已完成", message: "删除全部已完成的备忘？", okText: "清除", danger: true }))) return;
              api.del("/memo").then((r) => {
                toast(r.message, "ok");
                load();
              }).catch((e) => toast(e.message, "error"));
            },
          }, icon("trash", "sm"), "清除已完成"),
          h("span.muted.small", "对机器人说“看一下今天要做的事情”“前天要做的事情”，清单会显示在脸上的屏幕上。"))),
      h("div.grid.cols-2", { style: { alignItems: "start" } },
        h("div.stack",
          h("div.card",
            h("div.card-head", h("h3", icon("calendar", "sm"), calTitle),
              h("div.actions",
                h("button.btn.xs.ghost.icon-only", { title: "上个月", onclick: () => shiftMonth(-1) }, icon("chevronL", "sm")),
                h("button.btn.xs", {
                  onclick: () => {
                    selected = today;
                    month = today.slice(0, 7);
                    render();
                  },
                }, "今天"),
                h("button.btn.xs.ghost.icon-only", { title: "下个月", onclick: () => shiftMonth(1) }, icon("chevronR", "sm")))),
            h("div.card-body", calGrid)),
          h("div.card", h("div.card-head", h("h3", icon("settings", "sm"), "设置")), settingsBody)),
        h("div.stack",
          searchCard,
          h("div.card",
            h("div.card-head", dayTitle, dayCount,
              h("div.actions", h("button.btn.xs", { onclick: (e) => show({ date: selected }, e.currentTarget) }, icon("monitor", "sm"), "在屏幕上显示"))),
            h("div.card-body", addForm(true)),
            dayList),
          overdueCard,
          h("div.card",
            h("div.card-head", h("h3", icon("pin", "sm"), "随手记"), notesCount, h("span.sub", "没有日期的事")),
            h("div.card-body", addForm(false)),
            notesList))),
    );

    offs.push(on("memo", reload));
    await Promise.all([load(), renderSettings()]);
    const timer = setInterval(load, 60000);
    return () => {
      clearInterval(timer);
      offs.forEach((off) => off());
    };
  },
};
