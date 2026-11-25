// Settings → computer assistant: connect WorkBuddy on a PC. Tasks themselves are on pages/computer.js.

import * as api from "../api.js";
import { on, store } from "../store.js";
import { busy, clear, copyText, empty, field, h, icon, toast, toggleRow } from "../ui.js";

async function save(values, quiet = true) {
  try {
    const res = await api.patch("/settings/workbuddy", values);
    if (store.settings) store.settings.workbuddy = { ...store.settings.workbuddy, ...res.data };
    if (!quiet) toast(res.message || "已保存", "ok", 1800);
    return res.data;
  } catch (error) {
    toast(error.message, "error");
    throw error;
  }
}

function command(text) {
  return h("div.input-group",
    h("input.input.mono", { value: text, readOnly: true, onfocus: (e) => e.target.select() }),
    h("button.btn", { onclick: () => copyText(text).then(() => toast("已复制", "ok", 1500)) }, icon("copy", "sm"), "复制"));
}

function peerText(peer) {
  if (!peer) return "还没有电脑来报到";
  const who = peer.name ? `「${peer.name}」` : "电脑";
  return `${who}（${peer.url.replace(/^http:\/\//, "")}）${peer.online ? `${peer.age_s} 秒前报到` : "很久没有报到了：电脑可能关机或休眠"}`;
}

export default async function computerTab(root, settings) {
  const cfg = settings.workbuddy;
  const urlInput = h("input.input.mono", { value: cfg.url, placeholder: "电脑报到后自动填写，例如 192.168.2.20:8321", autocomplete: "off" });
  const pwInput = h("input.input.mono", { type: "password", value: cfg.password, placeholder: "打开本页时自动生成", autocomplete: "off" });
  const installBox = h("div.stack", h("div.small.muted", h("span.spinner"), " 正在生成安装命令…"));
  const peerEl = h("div.small.muted");
  const enabledRow = toggleRow("启用电脑助手", "把“操作电脑”的要求转交给电脑上的 WorkBuddy 智能体（电脑第一次报到时自动打开）",
    cfg.enabled, (v) => save({ enabled: v }).then(check));
  const triggers = h("input.input", { value: (cfg.triggers || []).join("，") });
  const statusEl = h("div.small.muted", "—");

  async function check() {
    clear(statusEl, h("span.spinner"), " 正在连接电脑…");
    try {
      const s = await api.get("/workbuddy");
      const tone = s.authorized ? "ok" : s.reachable ? "warn" : "danger";
      clear(statusEl, h(`span.badge.${tone}`, s.authorized ? "已连接" : s.reachable ? "密码不对" : "未连接"), " ", s.message || "");
      peerEl.textContent = peerText(s.peer);
    } catch (error) {
      clear(statusEl, h("span.badge.danger", "出错"), " ", error.message);
    }
  }

  // The robot fills its address and the gateway password into the script it serves, so the
  // commands carry neither and are fine to send to the PC over WeChat and the like.
  async function loadInstall() {
    try {
      const { password, peer } = (await api.post("/workbuddy/pairing", {})).data;
      const robot = location.origin;
      const local = !location.protocol.startsWith("http") || /^(localhost|127\.|\[::1\])/.test(location.hostname);
      // Saving the form must not blank the password the PC was given.
      pwInput.value = password;
      peerEl.textContent = peerText(peer);
      const auto = (q) => (autoBox.checked ? q : "");
      const winCmd = () => `powershell -NoExit -ExecutionPolicy Bypass -Command "irm '${robot}/api/v1/workbuddy/install.ps1${auto("?auto=1")}' | iex"`;
      const macCmd = () => `curl -fsSL '${robot}/api/v1/workbuddy/install.sh${auto("?auto=1")}' | bash`;
      const autoBox = h("input", { type: "checkbox", checked: true, onchange: () => {
        winBox.replaceWith((winBox = command(winCmd())));
        macBox.replaceWith((macBox = command(macCmd())));
      } });
      let winBox = command(winCmd());
      let macBox = command(macCmd());
      clear(installBox,
        local ? h("div.hint", { style: { color: "var(--danger)" } },
          "你现在是用 ", h("code", location.host || location.protocol), " 打开的网页，电脑连不到这个地址。请用机器人的局域网地址（例如 http://192.168.x.x:8000）打开本页。") : null,
        h("div.small.muted", "电脑上先装好并登录 WorkBuddy，然后只要做下面一件事。装好后电脑开机自动连上小小白，换了 IP 也会自动更新。"),
        field("最简单：在电脑浏览器里打开这个网址，照着点", command(`${robot}/pc`),
          "页面会认出是 Windows 还是 Mac，给出一行命令并显示是否连上。可以复制后用微信 / QQ 发到电脑上。"),
        h("details",
          h("summary.small.muted", { style: { cursor: "pointer" } }, "或者直接在电脑上运行命令"),
          h("div.stack", { style: { marginTop: "8px" } },
            field("Windows：按 Win+R，粘贴，回车", winBox),
            field("macOS / Linux：在“终端”里粘贴，回车", macBox))),
        h("label.small", { style: { display: "flex", gap: "8px", alignItems: "flex-start" } }, autoBox,
          h("span", "全自动执行：电脑上不弹窗等人批准，人不在电脑前也能做完。只用于你信任的电脑。")),
        h("div.hint", "卸载命令在电脑上安装完成时会显示。"));
    } catch (error) {
      clear(installBox, empty("alert", "生成安装命令失败", error.message));
    }
  }

  // What is in the fields is what gets tested: check() only reads the saved settings.
  function saveAndCheck(e) {
    return busy(e.currentTarget, async () => {
      const data = await save({ url: urlInput.value.trim(), password: pwInput.value.trim() }, false);
      urlInput.value = data.url;
      pwInput.value = data.password;
      await Promise.all([check(), loadInstall()]);
    });
  }

  root.append(
    h("div.grid.cols-2", { style: { alignItems: "start" } },
      h("div.stack",
        h("div.card",
          h("div.card-head", h("h3", icon("monitor", "sm"), "连接电脑上的 WorkBuddy"), h("span.sub", "小小白负责听，WorkBuddy 负责操作电脑")),
          h("div.card-body.stack",
            enabledRow,
            h("div.form-grid",
              field("电脑地址", urlInput, "电脑开机报到时自动更新，一般不用手动改"),
              field("访问密码", pwInput, "机器人生成，已经写在右边的安装命令里")),
            peerEl,
            h("div.row", statusEl, h("span.spacer"),
              h("button.btn", { onclick: saveAndCheck }, icon("refresh", "sm"), "测试连接"),
              h("button.btn.primary", { onclick: saveAndCheck }, "保存")))),
        h("div.card",
          h("div.card-head", h("h3", icon("mic", "sm"), "怎么叫它")),
          h("div.card-body.stack",
            toggleRow("语音口令", "语音对话里说到下面的词，就把这句话交给电脑", cfg.voice_trigger, (v) => save({ voice_trigger: v })),
            field("口令词", triggers, "用逗号分隔。例：“让电脑帮我把桌面截图发到微信文件传输助手”"),
            toggleRow("文字聊天也能用", "首页聊天时，大模型判断需要操作电脑就会自动转交", cfg.chat_tool, (v) => save({ chat_tool: v })),
            toggleRow("完成提示音", "电脑做完后小小白播放提示音（失败时是低沉的音）", cfg.announce, (v) => save({ announce: v })),
            h("div.row", h("span.spacer"), h("button.btn", {
              onclick: (e) => busy(e.currentTarget, () => save({ triggers: triggers.value }, false).then((d) => (triggers.value = d.triggers.join("，")))),
            }, "保存口令词"))))),
      h("div.stack",
        h("div.card",
          h("div.card-head", h("h3", icon("terminal", "sm"), "电脑端：装一次，开机自动连")),
          h("div.card-body", installBox)),
        h("div.card",
          h("div.card-body.row",
            h("span.small.muted", "连好以后，到底部的「电脑」页发任务，能看到电脑一步步在做什么。"), h("span.spacer"),
            h("a.btn.primary", { href: "#/computer" }, icon("monitor", "sm"), "去发任务"))))),
  );

  const offs = [
    on("workbuddy_peer", (e) => {
      urlInput.value = e.url;
      const box = enabledRow.querySelector("input[type=checkbox]");
      if (box) box.checked = true;
      if (store.settings) store.settings.workbuddy = { ...store.settings.workbuddy, url: e.url, enabled: true };
      toast(`电脑${e.peer?.name ? `「${e.peer.name}」` : ""}已报到：${e.url}`, "ok", 3000);
      check();
    }),
  ];
  loadInstall();
  if (cfg.enabled) check();
  else clear(statusEl, h("span.badge", "未启用"));
  return () => offs.forEach((off) => off());
}
