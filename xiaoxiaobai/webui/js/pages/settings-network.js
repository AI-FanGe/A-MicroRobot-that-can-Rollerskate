// Settings → 网络: the robot's Wi-Fi (scan, join, forget) and the phones paired over Bluetooth.

import * as api from "../api.js";
import * as native from "../native.js";
import { on } from "../store.js";
import { busy, clear, confirmDialog, empty, field, h, icon, modal, timeAgo, toast, toggleRow } from "../ui.js";

const bars = (signal) => (signal >= 70 ? "强" : signal >= 45 ? "中" : "弱");
const band = (mhz) => (mhz >= 5000 ? "5 GHz" : mhz ? "2.4 GHz" : "");

export default async function networkTab(root) {
  const statusBody = h("div.card-body", h("span.spinner"));
  const nearby = h("div.card-body.flush", empty("wifi", "还没有扫描", "点「扫描」看看机器人附近有哪些 Wi-Fi。"));
  const savedBody = h("div.card-body.flush");
  const bleBody = h("div.card-body", h("span.spinner"));
  let wifi = null;
  let disposed = false;

  async function loadWifi() {
    try {
      wifi = await api.get("/system/wifi");
    } catch (error) {
      clear(statusBody, empty("alert", "读取 Wi-Fi 状态失败", error.message));
      return;
    }
    if (!wifi.available) {
      clear(statusBody, empty("wifi", "这台机器人不能管理 Wi-Fi", "需要 NetworkManager（树莓派系统 Bookworm 默认自带）。"));
      clear(savedBody);
      return;
    }
    clear(statusBody, h("dl.kv",
      h("dt", "状态"), h("dd", wifi.connected ? h("span.badge.ok", "已连接") : h("span.badge.warn", "未连接")),
      h("dt", "网络"), h("dd", wifi.ssid || "—"),
      h("dt", "IP 地址"), h("dd.mono", wifi.ip || "—"),
      h("dt", "信号"), h("dd", wifi.signal != null ? `${wifi.signal}%（${bars(wifi.signal)}）` : "—"),
      h("dt", "网卡"), h("dd.mono", wifi.device || "—")));
    const saved = wifi.saved || [];
    clear(savedBody, saved.length
      ? saved.map((n) => h("div.provider",
        h("div.info", h("b", n.name, n.active ? h("span.badge.ok", "使用中") : null), h("small.mono", n.uuid)),
        h("button.btn.sm.ghost.icon-only", { title: "删除", onclick: () => forget(n) }, icon("trash", "sm"))))
      : empty("wifi", "没有保存的网络"));
  }

  async function scan(button) {
    return busy(button, async () => {
      clear(nearby, h("div.empty", h("span.spinner"), "正在扫描…（约 5 秒）"));
      try {
        const { networks } = await api.get("/system/wifi/scan", { ble: true, timeout: 40000 });
        clear(nearby, networks.length
          ? networks.map((n) => h("button.wifi-row", { type: "button", onclick: () => join(n) },
            icon("wifi"),
            h("span.text", h("b", n.ssid, n.in_use ? h("span.badge.ok", "已连接") : null),
              h("small", [`信号 ${bars(n.signal)}`, band(n.freq_mhz), n.secure ? n.security : "开放网络"].filter(Boolean).join(" · "))),
            n.secure ? icon("lock", "sm") : null))
          : empty("wifi", "附近没有找到 Wi-Fi", "主控只能连 2.4 GHz 网络。"));
        nearby.append(h("button.wifi-row", { type: "button", onclick: () => join(null) }, icon("plus"), h("span.text", h("b", "其他网络…"), h("small", "隐藏的网络，手动输入名称"))));
      } catch (error) {
        clear(nearby, empty("alert", "扫描失败", error.message));
      }
    });
  }

  function join(net) {
    const ssid = h("input.input", { value: net?.ssid || "", placeholder: "Wi-Fi 名称", maxlength: 32, autocomplete: "off" });
    const password = h("input.input", { type: "password", placeholder: net && !net.secure ? "开放网络，不需要密码" : "8–63 个字符", autocomplete: "off", disabled: net && !net.secure });
    let hidden = !net;
    const overWifi = !native.bleReady();
    modal({
      title: net ? `连接「${net.ssid}」` : "连接其他网络",
      body: h("div.stack",
        net ? null : field("网络名称（SSID）", ssid),
        net && !net.secure ? null : field("密码", password),
        net ? null : toggleRow("隐藏网络", "这个网络不广播名称", hidden, (v) => (hidden = v)),
        overWifi && wifi?.connected
          ? h("div.test-result.fail", "你现在是通过 Wi-Fi 连着机器人的：机器人换网络的这几十秒里页面会断开。换到别的网络后，手机也要连到同一个网络，再用新的 IP 打开。")
          : null,
        h("div.hint", "机器人会保存这个网络，以后开机自动连接。")),
      footer: (close) => [
        h("button.btn", { onclick: close }, "取消"),
        h("button.btn.primary", {
          onclick: (e) => busy(e.currentTarget, async () => {
            const body = { ssid: (net?.ssid || ssid.value).trim(), password: password.value, hidden: !net && hidden };
            if (!body.ssid) return toast("请输入网络名称", "warn");
            try {
              const r = await api.post("/system/wifi/connect", body, { ble: true, timeout: 70000 });
              toast(r.message, "ok", 6000);
              close();
              loadWifi();
            } catch (error) {
              toast(error.message, "error", 8000);
            }
          }),
        }, "连接"),
      ],
    });
    setTimeout(() => (net ? password : ssid).focus(), 50);
  }

  async function forget(n) {
    const warn = n.active ? "\n这是机器人正在用的网络，删除后会马上断开。" : "";
    if (!(await confirmDialog({ title: "删除网络", message: `删除「${n.name}」？${warn}`, okText: "删除", danger: true }))) return;
    try {
      await api.call(`/system/wifi/${encodeURIComponent(n.uuid)}`, { method: "DELETE", ble: true });
      toast("已删除", "ok");
      loadWifi();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function loadBle() {
    let s;
    try {
      s = await api.get("/system/ble");
    } catch (error) {
      clear(bleBody, empty("alert", "读取蓝牙状态失败", error.message));
      return;
    }
    const state = s.running ? h("span.badge.ok", "广播中") : s.enabled === false ? h("span.badge", "已关闭") : h("span.badge.warn", "未就绪");
    const sessions = s.sessions || [];
    const clients = s.clients || [];
    const online = new Set(sessions.filter((x) => x.authed).map((x) => x.client));
    clear(bleBody,
      h("dl.kv",
        h("dt", "状态"), h("dd", state),
        h("dt", "蓝牙名称"), h("dd.mono", s.name || "—"),
        h("dt", "连着的手机"), h("dd", sessions.length ? sessions.map((x) => `${x.name || x.device}${x.authed ? "" : "（未配对）"}`).join("、") : "无"),
        s.error ? h("dt", "问题") : null, s.error ? h("dd", h("span.text-danger", s.error)) : null),
      h("div.small.muted", { style: { margin: "12px 0 6px" } }, "配对过的手机（机器人连上 Wi-Fi 后，新手机要输入屏幕上的配对码）"),
      clients.length
        ? h("div", clients.map((c) => h("div.provider",
          h("div.info", h("b", icon("phone", "sm"), c.name || "手机", online.has(c.id) ? h("span.badge.ok", "在线") : null),
            h("small", `配对于 ${timeAgo(c.created)} · 最近 ${timeAgo(c.last_seen)}`)),
          h("button.btn.sm", { onclick: () => revoke(c) }, "取消配对"))))
        : h("div.small.muted", "还没有手机配对过。"),
      clients.length > 1 ? h("div.row", { style: { marginTop: "10px" } }, h("span.spacer"), h("button.btn.sm.danger", { onclick: () => revoke(null) }, "全部取消配对")) : null);
  }

  async function revoke(client) {
    const what = client ? `「${client.name || "这台手机"}」` : "全部手机";
    if (!(await confirmDialog({ title: "取消配对", message: `取消${what}的配对？下次连接要重新输入机器人屏幕上的配对码。`, okText: "取消配对", danger: true }))) return;
    try {
      await api.del(client ? `/system/ble/clients/${encodeURIComponent(client.id)}` : "/system/ble/clients");
      toast("已取消配对", "ok");
      loadBle();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  const scanBtn = h("button.btn.sm.primary", { onclick: (e) => scan(e.currentTarget) }, icon("search", "sm"), "扫描");
  root.append(
    h("div.grid.cols-2", { style: { alignItems: "start" } },
      h("div.stack",
        h("div.card",
          h("div.card-head", h("h3", icon("wifi", "sm"), "Wi-Fi"), h("span.sub", "语音、画面和文件走局域网"),
            h("div.actions", h("button.btn.sm.ghost.icon-only", { title: "刷新", onclick: () => loadWifi() }, icon("refresh", "sm")))),
          statusBody),
        h("div.card",
          h("div.card-head", h("h3", "附近的网络"), h("div.actions", scanBtn)),
          nearby),
        h("div.card",
          h("div.card-head", h("h3", "已保存的网络"), h("span.sub", "开机时自动连接")),
          savedBody)),
      h("div.card",
        h("div.card-head", h("h3", icon("bluetooth", "sm"), "手机 App 蓝牙"), h("span.sub", "首次配网和动作控制走蓝牙"),
          h("div.actions", h("button.btn.sm.ghost.icon-only", { title: "刷新", onclick: () => loadBle() }, icon("refresh", "sm")))),
        bleBody,
        native.inApp
          ? h("div.card-body", h("button.btn", { onclick: () => native.openDevices("#/robot") }, icon("phone", "sm"), "这台手机的连接与配对"))
          : null)),
  );

  const offWifi = on("wifi", () => !disposed && loadWifi());
  await Promise.all([loadWifi(), loadBle()]);
  const timer = setInterval(() => !document.hidden && loadBle(), 5000);
  return () => {
    disposed = true;
    clearInterval(timer);
    offWifi?.();
  };
}
