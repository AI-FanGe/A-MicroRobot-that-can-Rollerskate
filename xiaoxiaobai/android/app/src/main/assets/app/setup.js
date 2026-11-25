// The app's own pages: robot list and Bluetooth search, pairing, first Wi-Fi setup, the robot's
// status and a Bluetooth-only remote. The robot's full console is its web UI (openConsole).

import { busy, clear, confirmDialog, empty, field, h, icon, modal, rangeEl, timeAgo, toast, toggleRow } from "../web/js/ui.js";
import * as native from "../web/js/native.js";
import { joystick } from "../web/js/components/joystick.js";

const $ = (id) => document.getElementById(id);
const LINK = { ready: "已连接", pairing: "待配对", connecting: "连接中", scanning: "搜索中", idle: "未连接", off: "蓝牙已关闭", denied: "无权限" };
const LINK_TONE = { ready: "ok", pairing: "warn", connecting: "busy", scanning: "busy" };

const robots = () => JSON.parse(native.call("robots") || "{}");
const currentRobot = () => {
  const { current, robots: list = [] } = robots();
  return list.find((r) => r.id === current) || list[0] || null;
};
const go = (hash, replace = false) => (replace ? location.replace(hash) : (location.hash = hash));
const consoleUrl = (ip, port) => `http://${ip}:${port || 8000}/`;
const bars = (rssi) => (rssi >= -60 ? "强" : rssi >= -75 ? "中" : "弱");

// One API call over Bluetooth; resolves with `data` or throws with the robot's message.
async function ble(method, path, body, timeout) {
  const res = await native.request(method, path, body, { timeout });
  if (!res || res.header?.code !== 0) throw new Error(res?.message || res?.state?.reason_text || "请求失败");
  return res.data;
}

// Is the robot's web console reachable from this phone over Wi-Fi? Returns its URL or null.
async function reach(robot, ip = robot?.ip, port = robot?.port) {
  if (!ip) return null;
  const res = await native.httpGet(`http://${ip}:${port || 8000}/api/v1/meta/ping`, 2500);
  const data = res.ok ? res.body?.data : null;
  if (!data?.pong) return null;
  if (data.id && robot?.id && data.id !== robot.id && !robot.id.includes(":")) return null; // another robot took this IP
  native.call("saveRobot", JSON.stringify({ id: data.id || robot?.id || `${ip}:${port}`, name: data.name || robot?.name, ip, port: port || 8000 }));
  return consoleUrl(ip, port);
}

function header(title, { back = null, estop = false } = {}) {
  $("title").textContent = title;
  const backBtn = $("back");
  backBtn.hidden = !back;
  clear(backBtn, icon("chevronL"));
  backBtn.onclick = () => (history.length > 1 ? history.back() : go(back, true));
  $("logo").hidden = !!back;
  $("estop").hidden = !estop;
}

$("estop").onclick = async () => {
  native.haptic("heavy");
  try {
    await ble("POST", "/mc/estop", { source: "app_proxy" }, 4000);
    toast("已急停：电机断开力矩", "warn");
  } catch (error) {
    toast(`急停没有送达：${error.message}`, "error", 6000);
  }
};

function linkChip(s = native.linkState()) {
  return h(`span.chip.${LINK_TONE[s.state] || "plain"}`, h("span.dot"), "蓝牙 ", h("b", LINK[s.state] || s.state));
}

function bluetoothBanner(s) {
  if (s.state !== "off" && s.state !== "denied") return null;
  return h("div.card", h("div.card-body.stack",
    h("div.row", icon("bluetooth"), h("b", s.state === "off" ? "手机蓝牙没有打开" : "需要蓝牙权限")),
    h("div.small.muted", "第一次给机器人联网、以及用手机遥控机器人都要用蓝牙。"),
    h("button.btn.primary", { onclick: () => native.call("enableBluetooth") }, s.state === "off" ? "打开蓝牙" : "允许")));
}

// ======================================================================= boot

function boot(page) {
  header("小小白");
  const line = h("div.small.muted", "");
  page.append(h("div.splash", h("img", { src: "../web/img/logo.svg", alt: "" }), h("b", "小小白"), line));
  const robot = currentRobot();
  if (!robot) {
    go("#/home", true);
    return null;
  }
  line.textContent = `正在连接 ${robot.name}…`;
  let done = false;
  let wifiFailed = !robot.ip;
  const finish = (fn) => {
    if (done) return;
    done = true;
    fn();
  };
  const tryWifi = (ip, port) => reach(robot, ip, port).then((url) => {
    if (url) finish(() => native.call("openConsole", url));
    else wifiFailed = true;
    return url;
  });
  if (robot.address) native.call("select", robot.id);
  tryWifi(robot.ip, robot.port);

  const onState = (s) => {
    if (done) return;
    if (s.state === "pairing") finish(() => go("#/pair", true));
    else if (s.state === "ready") {
      if (s.setup) return finish(() => go("#/wifi", true));
      const ip = s.robot?.ip;
      line.textContent = "蓝牙已连接，正在打开控制台…";
      if (ip && ip !== robot.ip) tryWifi(ip, s.robot?.port).then((url) => !url && finish(() => go("#/robot", true)));
      else if (wifiFailed) finish(() => go("#/robot", true));
    } else if ((s.state === "off" || s.state === "denied" || !robot.address) && wifiFailed) {
      finish(() => go("#/robot", true));
    }
  };
  const off = native.on("state", onState);
  onState(native.linkState());
  const timer = setTimeout(() => finish(() => go("#/robot", true)), 10000);
  return () => {
    done = true;
    off();
    clearTimeout(timer);
  };
}

// ======================================================================= robot list + search

function home(page) {
  header("小小白");
  const { current, robots: list = [] } = robots();
  const banner = h("div");
  const results = h("div.list");
  const scanBtn = h("button.btn.primary", { onclick: () => startScan() }, icon("search", "sm"), "搜索附近的机器人");
  let scanning = false;

  const renderResults = (found) => {
    if (!found.length) {
      clear(results, scanning
        ? h("div.empty", h("span.spinner"), "正在搜索…机器人开机约 1 分钟后才能被找到")
        : empty("bluetooth", "附近没有找到小小白", "确认机器人已开机、离手机不远；已经联网的机器人也能搜到。"));
      return;
    }
    clear(results, found.map((r) => h("button.item", {
      type: "button",
      onclick: () => {
        native.call("scan", false);
        native.call("connect", r.address);
        go(`#/connect?name=${encodeURIComponent(r.saved?.name || r.name)}`);
      },
    }, h("img", { src: "../web/img/logo.svg", alt: "" }),
    h("span.text", h("b", r.saved?.name || r.name, r.saved ? h("span.badge.ok", "已保存") : null),
      h("small", `${r.name} · 信号${bars(r.rssi)}`)),
    icon("chevronR", "sm"))));
  };
  const startScan = () => {
    scanning = true;
    renderResults([]);
    native.call("scan", true);
  };
  const offScan = native.on("scan", (found) => renderResults(found));
  const offState = native.on("state", (s) => {
    clear(banner, bluetoothBanner(s));
    if (scanning && !s.scanning && s.state !== "denied" && s.state !== "off") {
      scanning = false;
      scanBtn.textContent = "重新搜索";
      if (!results.querySelector(".item")) renderResults([]);
    }
  });
  clear(banner, bluetoothBanner(native.linkState()));

  const manual = () => {
    const ip = h("input.input.mono", { placeholder: "例如 192.168.1.23", inputmode: "decimal", autocomplete: "off" });
    const port = h("input.input.mono", { value: "8000", inputmode: "numeric" });
    modal({
      title: "输入机器人的地址",
      body: h("div.stack",
        h("div.small.muted", "机器人已经联网、手机和它在同一个 Wi-Fi 时可以直接输入 IP（机器人屏幕或路由器里能看到）。"),
        field("IP 地址", ip), field("端口", port)),
      footer: (close) => [
        h("button.btn", { onclick: close }, "取消"),
        h("button.btn.primary", {
          onclick: (e) => busy(e.currentTarget, async () => {
            const url = await reach(null, ip.value.trim(), Number(port.value) || 8000);
            if (!url) return toast("连不上这个地址：确认 IP 正确、手机和机器人在同一个 Wi-Fi", "error", 6000);
            close();
            native.call("openConsole", url);
          }),
        }, "连接"),
      ],
    });
    setTimeout(() => ip.focus(), 50);
  };

  page.append(h("div.stack",
    banner,
    list.length ? h("div.card",
      h("div.card-head", h("h3", "我的机器人")),
      h("div.list", list.map((r) => h("button.item", {
        type: "button",
        onclick: () => {
          native.call("select", r.id);
          go("#/robot");
        },
      }, h("img", { src: "../web/img/logo.svg", alt: "" }),
      h("span.text", h("b", r.name, r.id === current ? h("span.badge.accent", "当前") : null),
        h("small", [r.paired ? "已配对" : r.address ? "未配对" : "Wi-Fi", r.ip || null, r.last_seen ? `${timeAgo(r.last_seen / 1000)}连接过` : null].filter(Boolean).join(" · "))),
      icon("chevronR", "sm"))))) : h("div.card", h("div.card-body.stack",
      h("div.hero", h("img", { src: "../web/img/logo.svg", alt: "" }), h("div.text", h("b", "欢迎使用小小白"), h("span.small.muted", "三步完成：找到机器人 → 连上蓝牙 → 给它连 Wi-Fi"))))),
    h("div.card",
      h("div.card-head", h("h3", icon("bluetooth", "sm"), "添加机器人")),
      h("div.card-body.stack",
        h("p.lead", "打开机器人电源，等脸上出现表情（约 1 分钟）。机器人开机就会打开蓝牙，手机靠近后点下面搜索。"),
        scanBtn),
      results),
    h("button.btn.ghost", { onclick: manual }, icon("wifi", "sm"), "已经联网？输入 IP 地址连接"),
    h("div.small.muted", { style: { textAlign: "center" } }, `小小白 App ${JSON.parse(native.call("appInfo") || "{}").version || ""}`)));
  if (!list.length) startScan();
  else renderResults([]);
  return () => {
    offScan();
    offState();
    native.call("scan", false);
  };
}

// ======================================================================= connecting

function connect(page, params) {
  header("连接机器人", { back: "#/home" });
  const name = params.get("name") || "小小白";
  const steps = h("div.steps");
  const note = h("div");
  const step = (n, text, state) => h(`div.step.${state}`, h("span.num", state === "done" ? icon("check", "sm") : String(n)), h("span", text));
  const render = (s) => {
    const linked = s.state === "ready" || s.state === "pairing";
    clear(steps,
      step(1, `用蓝牙连接 ${name}`, linked ? "done" : "active"),
      step(2, "配对（机器人已联网时，输入它屏幕上的 4 位码）", s.state === "ready" ? "done" : linked ? "active" : ""),
      step(3, "给机器人连 Wi-Fi", s.state === "ready" && !s.setup ? "done" : s.state === "ready" ? "active" : ""));
    clear(note,
      s.state === "connecting" && s.error ? h("div.test-result.fail", s.error) : null,
      bluetoothBanner(s));
    if (s.state === "pairing") go("#/pair", true);
    else if (s.state === "ready") go(s.setup ? "#/wifi" : "#/robot", true);
  };
  page.append(h("div.stack",
    h("div.card", h("div.card-body.stack", h("div.row", h("span.spinner"), h("b", "正在连接…")), steps)),
    note,
    h("button.btn", { onclick: () => go("#/home", true) }, "取消")));
  const off = native.on("state", render);
  render(native.linkState());
  return off;
}

// ======================================================================= pairing

function pair(page) {
  header("配对", { back: "#/home" });
  const codeInput = h("input.input.code-input", { inputmode: "numeric", maxlength: 4, autocomplete: "one-time-code", placeholder: "····" });
  const status = h("div.small.muted", "");
  let sending = false;

  const ask = async (button) => {
    try {
      const res = await native.send({ t: "pair" });
      status.textContent = res.shown
        ? `配对码已显示在机器人的屏幕上，${Math.round((res.expires_s || 120) / 60)} 分钟内有效`
        : "机器人的屏幕没有显示出来（显示程序没运行？）。可以在电脑上打开控制台看提醒。";
    } catch (error) {
      status.textContent = error.message;
    }
    button?.blur();
  };
  const submit = async () => {
    const code = codeInput.value.replace(/\D/g, "");
    if (code.length !== 4 || sending) return;
    sending = true;
    try {
      const res = await native.send({ t: "auth", code });
      if (res.ok) {
        native.haptic("ok");
        toast("配对成功", "ok");
        const s = native.linkState();
        go(s.setup ? "#/wifi" : "#/robot", true);
      } else {
        toast(res.message || "配对码不对", "error");
        codeInput.value = "";
        codeInput.focus();
      }
    } catch (error) {
      toast(error.message, "error");
    } finally {
      sending = false;
    }
  };
  codeInput.addEventListener("input", () => {
    codeInput.value = codeInput.value.replace(/\D/g, "").slice(0, 4);
    if (codeInput.value.length === 4) submit();
  });

  page.append(h("div.stack",
    h("div.card", h("div.card-body.stack",
      h("div.hero", h("img", { src: "../web/img/logo.svg", alt: "" }),
        h("div.text", h("b", "输入机器人屏幕上的配对码"), h("span.small.muted", "机器人已经联网时，新手机第一次连接需要配对。"))),
      codeInput,
      status,
      h("div.row",
        h("button.btn", { onclick: (e) => ask(e.currentTarget) }, icon("refresh", "sm"), "重新显示配对码"),
        h("span.spacer"),
        h("button.btn.primary", { onclick: submit }, "配对")))),
    h("div.small.muted", "配对后这台手机会被记住，以后不用再输入。可以在控制台「设置 → 网络」里取消配对。")));
  const off = native.on("state", (s) => {
    if (s.state === "ready") go(s.setup ? "#/wifi" : "#/robot", true);
    else if (s.state !== "pairing" && s.state !== "connecting") go("#/robot", true);
  });
  ask();
  setTimeout(() => codeInput.focus(), 200);
  return off;
}

// ======================================================================= Wi-Fi setup over Bluetooth

function wifi(page) {
  header("给机器人连 Wi-Fi", { back: "#/robot" });
  const statusBody = h("div.card-body", h("span.spinner"));
  const list = h("div.list");
  const result = h("div");
  const scanBtn = h("button.btn.sm", { onclick: (e) => scan(e.currentTarget) }, icon("refresh", "sm"), "重新扫描");
  let robotWifi = null;

  if (native.linkState().state !== "ready") {
    page.append(h("div.stack", empty("bluetooth", "要先用蓝牙连上机器人", "配网通过蓝牙进行。",
      h("button.btn.primary", { onclick: () => go("#/home") }, "去连接"))));
    return null;
  }

  const loadStatus = async () => {
    try {
      robotWifi = await ble("GET", "/system/wifi", null, 15000);
      clear(statusBody, robotWifi.available === false
        ? h("div.small.muted", "机器人没有 NetworkManager，不能在这里配网。")
        : h("dl.kv",
          h("dt", "状态"), h("dd", robotWifi.connected ? h("span.badge.ok", "已连接") : h("span.badge.warn", "未联网")),
          h("dt", "网络"), h("dd", robotWifi.ssid || "—"),
          h("dt", "IP"), h("dd.mono", robotWifi.ip || "—")));
    } catch (error) {
      clear(statusBody, h("div.test-result.fail", error.message));
    }
  };

  const scan = (button) => busy(button, async () => {
    clear(list, h("div.empty", h("span.spinner"), "机器人正在扫描附近的 Wi-Fi…"));
    try {
      const { networks } = await ble("GET", "/system/wifi/scan", null, 40000);
      clear(list,
        networks.length ? networks.map((n) => h("button.item", { type: "button", onclick: () => join(n) },
          icon("wifi"),
          h("span.text", h("b", n.ssid, n.in_use ? h("span.badge.ok", "已连接") : null),
            h("small", [`信号 ${n.signal >= 70 ? "强" : n.signal >= 45 ? "中" : "弱"}`, n.secure ? n.security : "开放网络"].join(" · "))),
          n.secure ? icon("lock", "sm") : null)) : empty("wifi", "没有扫描到 Wi-Fi", "机器人只能连 2.4 GHz 网络；把手机热点改成 2.4 GHz 也可以。"),
        h("button.item", { type: "button", onclick: () => join(null) }, icon("plus"), h("span.text", h("b", "其他网络…"), h("small", "隐藏的网络，手动输入名称"))));
    } catch (error) {
      clear(list, h("div.card-body", h("div.test-result.fail", error.message)));
    }
  });

  const join = (net) => {
    const ssid = h("input.input", { placeholder: "Wi-Fi 名称", maxlength: 32, autocomplete: "off" });
    const password = h("input.input", { type: "password", placeholder: "8–63 个字符", autocomplete: "off" });
    const show = h("button.btn.ghost.sm", { type: "button", onclick: () => (password.type = password.type === "password" ? "text" : "password") }, "显示");
    let hidden = !net;
    modal({
      title: net ? `连接「${net.ssid}」` : "其他网络",
      body: h("div.stack",
        net ? null : field("网络名称（SSID）", ssid),
        net && !net.secure ? h("div.small.muted", "这是开放网络，不需要密码。") : h("div.field", h("label", "密码", h("span.spacer"), show), password),
        net ? null : toggleRow("隐藏网络", "这个网络不广播名称", hidden, (v) => (hidden = v)),
        h("div.small.muted", "机器人会记住这个网络，以后开机自动连接。连接需要 10–40 秒。")),
      footer: (close) => [
        h("button.btn", { onclick: close }, "取消"),
        h("button.btn.primary", {
          onclick: (e) => busy(e.currentTarget, async () => {
            const body = { ssid: (net?.ssid || ssid.value).trim(), password: password.value, hidden: !net && hidden };
            if (!body.ssid) return toast("请输入网络名称", "warn");
            try {
              const status = await ble("POST", "/system/wifi/connect", body, 75000);
              close();
              connected(status);
            } catch (error) {
              toast(error.message, "error", 8000);
            }
          }),
        }, "连接"),
      ],
    });
    setTimeout(() => (net ? password : ssid).focus(), 60);
  };

  const connected = async (status) => {
    native.haptic("ok");
    robotWifi = status;
    loadStatus();
    const box = h("div.card-body.stack", h("div.row", h("span.spinner"), "已连上，正在检查手机能不能通过 Wi-Fi 访问机器人…"));
    clear(result, h("div.card", h("div.card-head", h("h3", icon("check", "sm"), `机器人已连上 ${status.ssid}`)), box));
    result.scrollIntoView({ behavior: "smooth" });
    const check = async () => {
      const url = await reach(currentRobot(), status.ip, native.linkState().robot?.port);
      if (url) {
        clear(box, h("div.small.muted", `IP ${status.ip}，手机可以直接访问。语音、画面和文件都会走 Wi-Fi。`),
          h("button.btn.primary", { onclick: () => native.call("openConsole", url) }, "打开控制台"));
        setTimeout(() => location.hash.startsWith("#/wifi") && native.call("openConsole", url), 1500);
      } else {
        clear(box,
          h("div.test-result.fail", `手机访问不到 ${status.ip || "机器人"}：把手机也连到「${status.ssid}」，再点重试。`),
          h("div.row",
            h("button.btn.primary", { onclick: (e) => busy(e.currentTarget, check) }, icon("refresh", "sm"), "重试"),
            h("button.btn", { onclick: () => go("#/remote") }, icon("gamepad", "sm"), "先用蓝牙遥控")));
      }
    };
    check();
  };

  page.append(h("div.stack",
    h("div.card", h("div.card-head", h("h3", icon("bot", "sm"), "机器人的网络")), statusBody),
    result,
    h("div.card",
      h("div.card-head", h("h3", icon("wifi", "sm"), "选择 Wi-Fi"), h("div.actions", scanBtn)),
      list),
    h("div.small.muted", "主控只支持 2.4 GHz Wi-Fi。连好后手机要和机器人在同一个 Wi-Fi 下，语音、画面和文件才能走局域网。")));
  loadStatus();
  scan(scanBtn);
  return native.on("state", (s) => s.state !== "ready" && s.state !== "connecting" && toast("蓝牙断开了", "warn"));
}

// ======================================================================= robot status

function robot(page, params) {
  const r = currentRobot();
  if (!r) {
    go("#/home", true);
    return null;
  }
  header(r.name, { back: "#/home" });
  const offlineUrl = params.get("url");
  const chips = h("div.chips");
  const wifiChip = h("span.chip.busy", h("span.dot"), "Wi-Fi ", h("b", "检查中"));
  const openBtn = h("button.btn.primary", { disabled: true }, icon("monitor", "sm"), "打开控制台");
  const actions = h("div.list");
  let url = null;
  let checking = false;

  const check = async () => {
    if (checking) return;
    checking = true;
    const s = native.linkState();
    const ip = (s.state === "ready" && s.robot?.ip) || currentRobot()?.ip;
    url = await reach(currentRobot(), ip, s.robot?.port || r.port);
    checking = false;
    clear(wifiChip, h("span.dot"), "Wi-Fi ", h("b", url ? ip : ip ? "访问不到" : "未知"));
    wifiChip.className = `chip ${url ? "ok" : "warn"}`;
    openBtn.disabled = !url;
    openBtn.onclick = () => url && native.call("openConsole", url);
    renderActions(native.linkState());
  };

  const action = (ic, title, hint, onclick, disabled = false) =>
    h("button.item", { type: "button", onclick, disabled }, icon(ic), h("span.text", h("b", title), hint ? h("small", hint) : null), icon("chevronR", "sm"));

  const renderActions = (s) => {
    const linked = s.state === "ready";
    clear(chips, linkChip(s), wifiChip, s.robot?.ssid ? h("span.chip.plain", icon("wifi", "sm"), s.robot.ssid) : null);
    clear(actions,
      action("gamepad", "蓝牙遥控", "没有 Wi-Fi 也能控制动作、表情和音乐", () => go("#/remote"), !linked),
      action("target", "跟随模式", "认出你的腿跟着走，熄屏也能跑（要 Wi-Fi / 热点）", () => go("#/follow")),
      action("wifi", "给机器人连 Wi-Fi", s.setup ? "机器人还没联网" : "换一个网络", () => go("#/wifi"), !linked),
      s.state === "pairing" ? action("lock", "输入配对码", "这台手机还没和机器人配对", () => go("#/pair")) : null,
      r.address && !["ready", "pairing", "connecting"].includes(s.state)
        ? action("bluetooth", "连接蓝牙", s.error || "", () => native.call("select", r.id)) : null,
      action("layers", "切换或添加机器人", null, () => go("#/home")),
      action("trash", "忘记这台机器人", "删除保存的配对，下次要重新配对", async () => {
        if (!(await confirmDialog({ title: "忘记机器人", message: `从这台手机上删除「${r.name}」？`, okText: "删除", danger: true }))) return;
        native.call("forget", r.id);
        go("#/home", true);
      }));
  };

  page.append(h("div.stack",
    offlineUrl ? h("div.card", h("div.card-body.stack",
      h("div.row", icon("alert"), h("b", "控制台打不开")),
      h("div.small.muted", `连不上 ${offlineUrl}：机器人可能换了网络、关机了，或者手机不在同一个 Wi-Fi。`))) : null,
    h("div.card", h("div.card-body.stack",
      h("div.hero", h("img", { src: "../web/img/logo.svg", alt: "" }), h("div.text", h("b", r.name), chips)),
      openBtn)),
    h("div.card", actions)));
  const off = native.on("state", (s) => {
    renderActions(s);
    if (s.state === "ready" && !url) check();
  });
  if (r.address && !["ready", "pairing", "connecting"].includes(native.linkState().state)) native.call("select", r.id);
  renderActions(native.linkState());
  check();
  const timer = setInterval(() => !document.hidden && check(), 6000);
  return () => {
    off();
    clearInterval(timer);
  };
}

// ======================================================================= Bluetooth remote

function remote(page) {
  header("蓝牙遥控", { back: "#/robot", estop: true });
  const statusLine = h("div.row.wrap");
  const modes = h("div.remote-grid");
  const gestures = h("div.chips-wrap", h("span.spinner"));
  const faces = h("div.chips-wrap", h("span.spinner"));
  const sounds = h("div.chips-wrap", h("span.spinner"));
  let snapshot = null;
  let velocity = { vx: 0, vy: 0, vtheta: 0 };
  let timer = null;
  const walking = () => ["BIPED_WALK_RUN", "BIPED_WALK_UPPER", "BIPED_SKATE"].includes(snapshot?.action?.current_action?.name);

  const run = (method, path, body, okText) => async (event) => {
    const button = event?.currentTarget;
    native.haptic("light");
    try {
      const res = await ble(method, path, body, 15000);
      if (okText) toast(okText, "ok", 1500);
      return res;
    } catch (error) {
      toast(error.message, "error", 5000);
    } finally {
      button?.blur();
    }
  };

  const sendVelocity = () => native.request("POST", "/mc/locomotion/velocity", { source: "app_proxy", ...velocity }, { quiet: true })
    .catch(() => null);
  const stick = joystick({
    onChange: ({ x, y }) => {
      velocity = { vx: y * 0.8, vy: 0, vtheta: -x * 0.8 };
      const moving = Math.abs(velocity.vx) > 0.01 || Math.abs(velocity.vtheta) > 0.01;
      if (!walking()) return;
      if (moving && !timer) {
        sendVelocity();
        timer = setInterval(sendVelocity, 125);
      } else if (!moving && timer) {
        clearInterval(timer);
        timer = null;
        sendVelocity();
      }
    },
  });

  const renderStatus = () => {
    const s = snapshot;
    if (!s) return;
    const act = s.action?.current_action || {};
    const target = s.action?.target_action;
    const bat = s.battery || {};
    clear(statusLine,
      h(`span.chip.${target && target !== act.name ? "busy" : "ok"}`, h("span.dot"), "模式 ", h("b", act.label || act.name || "—")),
      h(`span.chip.${bat.level === "critical" ? "danger" : bat.level === "low" ? "warn" : "plain"}`, icon("battery", "sm"),
        h("b", bat.voltage_v ? `${bat.percent}%` : "—")));
    const allowed = s.action?.allowed || [];
    clear(modes,
      h("button.btn.active", { disabled: true }, act.label || "当前"),
      allowed.map((a) => h("button.btn", { onclick: run("POST", "/mc/action", { action: a.name }, `切换到「${a.label}」`) }, a.label)));
    stick.setEnabled(walking());
  };

  page.append(h("div.stack",
    h("div.card", h("div.card-body.stack", statusLine, h("div.small.muted", "通过蓝牙控制，不需要 Wi-Fi。画面、对话和文件要等机器人联网后在控制台里用。"))),
    h("div.card", h("div.card-head", h("h3", icon("layers", "sm"), "运动模式")), h("div.card-body", modes)),
    h("div.card", h("div.card-head", h("h3", icon("gamepad", "sm"), "走跑"), h("span.sub", "在走跑模式下推动摇杆")),
      h("div.card-body.stick-wrap", stick)),
    h("div.card", h("div.card-head", h("h3", icon("bot", "sm"), "动作"),
      h("div.actions", h("button.btn.sm", { onclick: run("POST", "/mc/gesture/stop", {}, "已停止") }, icon("stop", "sm"), "停止"))),
    h("div.card-body", gestures)),
    h("div.card", h("div.card-head", h("h3", icon("smile", "sm"), "表情")), h("div.card-body", faces)),
    h("div.card", h("div.card-head", h("h3", icon("volume", "sm"), "提示音")), h("div.card-body", sounds)),
    h("div.card", h("div.card-head", h("h3", icon("music", "sm"), "音乐")), h("div.card-body", h("div.remote-grid",
      h("button.btn", { onclick: run("POST", "/music/previous", {}) }, icon("skipB", "sm"), "上一首"),
      h("button.btn", { onclick: run("POST", "/music/pause", {}) }, icon("pause", "sm"), "暂停"),
      h("button.btn", { onclick: run("POST", "/music/resume", {}) }, icon("play", "sm"), "继续"),
      h("button.btn", { onclick: run("POST", "/music/next", {}) }, icon("skipF", "sm"), "下一首"),
      h("button.btn", { onclick: run("POST", "/music/stop", {}) }, icon("stop", "sm"), "停止"),
      h("button.btn", { onclick: run("POST", "/music/volume", { volume: 40 }) }, icon("volume", "sm"), "音量 40")))),
  ));

  const load = async () => {
    try {
      const [g, e, snd, action] = await Promise.all([
        ble("GET", "/mc/gestures", null, 15000),
        ble("GET", "/interaction/emotions?brief=1", null, 15000),
        ble("GET", "/interaction/sounds", null, 15000),
        ble("GET", "/mc/action", null, 15000),
      ]);
      snapshot = { ...(snapshot || {}), action };
      renderStatus();
      clear(gestures, (g.gestures || []).map((x) => h("button.btn", { onclick: run("POST", "/mc/gesture", { name: x.name, source: "app_proxy" }) }, x.label)));
      clear(faces, e.map((x) => h("button.btn", { onclick: run("POST", "/interaction/emotion", { name: x.name }) }, x.label)));
      clear(sounds, snd.map((x) => h("button.btn", { onclick: run("POST", "/interaction/sound", { name: x.name, volume: 60 }) }, x.label.replace(/（.*?）/, ""))));
    } catch (error) {
      toast(`读取列表失败：${error.message}`, "error", 6000);
    }
  };

  const offEvent = native.on("event", (event) => {
    if (event.type !== "status") return;
    snapshot = event;
    renderStatus();
  });
  const offState = native.on("state", (s) => {
    if (s.state === "ready") native.subscribe(true);
    else if (s.state !== "connecting") go("#/robot", true);
  });
  if (native.linkState().state === "ready") {
    native.subscribe(true);
    load();
  } else {
    go("#/robot", true);
  }
  return () => {
    offEvent();
    offState();
    clearInterval(timer);
    if (timer) sendVelocity();
    native.subscribe(false);
  };
}

// ======================================================================= follow mode

// FollowService does the work (camera, leg detection, velocity commands) and keeps running with
// the screen off; this page starts / stops it and shows what it sees.
const PHASE = {
  starting: ["busy", "启动中"], waiting: ["warn", "等机器人"], no_camera: ["warn", "没有画面"], offline: ["danger", "连不上"],
  searching: ["busy", "找人中"], following: ["ok", "跟随中"], close: ["ok", "近距离"], busy: ["warn", "被接管"],
  stopping: ["busy", "停下中"], error: ["danger", "出错"],
};
const LEG_LINKS = [[11, 13], [13, 15], [12, 14], [14, 16], [11, 12]];

function follow(page) {
  const r = currentRobot();
  if (!r) {
    go("#/home", true);
    return null;
  }
  header("跟随模式", { back: "#/robot" });
  const settings = JSON.parse(native.call("followSettings") || "{}");
  let fast = Math.round((settings.fast ?? 0.6) * 100);
  let slow = Math.round((settings.slow ?? 0.3) * 100);
  let status = {};

  const chips = h("div.chips");
  const line = h("div.small.muted", "");
  const canvas = h("canvas.follow-view", { width: 320, height: 240 });
  const ctx = canvas.getContext("2d");
  const stats = h("div.small.muted.mono", "");
  const startBtn = h("button.btn.primary", { onclick: () => start(startBtn) }, icon("play", "sm"), "开始跟随");
  const stopBtn = h("button.btn.danger", { onclick: () => native.call("followStop") }, icon("stop", "sm"), "停止跟随");
  const battery = h("div.stack");

  const retune = () => status.running && native.call("followStart", JSON.stringify({ fast: fast / 100, slow: slow / 100 }));
  const fastRange = rangeEl({ min: 30, max: 80, step: 5, value: fast, format: (v) => `${v}%`, onchange: (v) => {
    fast = v;
    if (slow > fast) slowRange.set((slow = fast));
    retune();
  } });
  const slowRange = rangeEl({ min: 10, max: 60, step: 5, value: slow, format: (v) => `${v}%`, onchange: (v) => {
    slow = Math.min(v, fast);
    slowRange.set(slow);
    retune();
  } });

  async function start(button) {
    await busy(button, async () => {
      if (!(await reach(currentRobot()))) {
        toast("手机访问不到机器人：确认手机热点 / Wi-Fi 和机器人是同一个网络", "error", 6000);
        return;
      }
      const error = native.call("followStart", JSON.stringify({ fast: fast / 100, slow: slow / 100 }));
      if (error) toast(error, "error", 6000);
      else native.haptic("ok");
    });
  }

  function renderBattery() {
    const exempt = JSON.parse(native.call("followSettings") || "{}").battery_exempt;
    clear(battery,
      h("div.row", icon(exempt ? "check" : "alert", "sm"), h("b", exempt ? "已允许后台运行" : "还没允许后台运行")),
      h("div.small.muted", "熄屏后要继续跟随：允许「后台运行 / 不优化电池」。小米、华为、OPPO、vivo 还要在应用设置里打开「自启动」并把省电策略设为「无限制」。"),
      h("div.row.wrap",
        exempt ? null : h("button.btn.sm.primary", { onclick: () => native.call("askBatteryExempt") }, "允许后台运行"),
        h("button.btn.sm", { onclick: () => native.call("openAppSettings") }, icon("settings", "sm"), "应用设置")));
  }

  let drawnSeq = -1;
  function drawFrame() {
    const seq = Number(native.call("followFrameSeq")) || 0;
    if (seq === drawnSeq) return;
    drawnSeq = seq;
    const f = JSON.parse(native.call("followFrame") || "{}");
    if (!f.jpeg) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      const W = img.width;
      const H = img.height;
      ctx.drawImage(img, 0, 0);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(200,200,200,0.7)";
      for (const [x1, y1, x2, y2] of f.others || []) ctx.strokeRect(x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H);
      const t = f.target;
      if (!t) return;
      const [x1, y1, x2, y2] = t.box;
      ctx.strokeStyle = "#2dd4bf";
      ctx.strokeRect(x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H);
      const k = t.kpts;
      const seen = (i) => k[i * 3 + 2] >= 0.4;
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 3;
      for (const [a, b] of LEG_LINKS) {
        if (!seen(a) || !seen(b)) continue;
        ctx.beginPath();
        ctx.moveTo(k[a * 3] * W, k[a * 3 + 1] * H);
        ctx.lineTo(k[b * 3] * W, k[b * 3 + 1] * H);
        ctx.stroke();
      }
      ctx.fillStyle = "#facc15";
      for (let i = 11; i <= 16; i++) {
        if (seen(i)) ctx.fillRect(k[i * 3] * W - 3, k[i * 3 + 1] * H - 3, 6, 6);
      }
    };
    img.src = `data:image/jpeg;base64,${f.jpeg}`;
  }

  function render() {
    status = JSON.parse(native.call("followStatus") || "{}");
    const [tone, label] = status.running ? PHASE[status.phase] || ["plain", status.phase] : ["plain", "未开启"];
    clear(chips,
      h(`span.chip.${tone}`, h("span.dot"), h("b", label)),
      status.running && status.legs ? h("span.chip.plain", status.legs) : null,
      status.running ? h("span.chip.plain", "速度 ", h("b", `${status.vx}%`)) : null);
    line.textContent = status.running ? status.message : "开始后，机器人会认出你的腿：看到整条腿就快一点跟上，只看到小腿就放慢，只看到脚就停下等你。"
      + "跟丢了会先转头往你离开的方向看，再转身找，找到后头看着你、身子转过来再跟上。";
    const where = status.bearing == null ? "" : ` · 你在${status.bearing >= 0 ? "左" : "右"} ${Math.abs(status.bearing)}°`;
    stats.textContent = status.running
      ? `画面 ${status.camera_fps} 帧/秒 · 识别 ${status.vision_fps} 次/秒（${status.infer_ms} ms，${status.model === "n" ? "轻量" : "精准"}模型）· ${status.people} 人`
        + ` · 转向 ${status.vtheta}% · 头 ${status.head_ok ? `${status.head}°` : "朝前（机器人端没更新）"}${where}`
        + (status.cling ? ` · 画面起伏 ${status.luma_std}（低于 14 算贴住）` : "")
      : "";
    startBtn.hidden = !!status.running;
    stopBtn.hidden = !status.running;
  }

  page.append(h("div.stack",
    h("div.card", h("div.card-body.stack", chips, line, canvas, stats, h("div.row.wrap", startBtn, stopBtn))),
    h("div.card", h("div.card-head", h("h3", icon("settings", "sm"), "速度"), h("span.sub", "占满速的比例，加减速都是缓慢渐变")),
      h("div.card-body.stack",
        field("看到整条腿（离得远）", fastRange),
        field("只看到小腿（离得近）", slowRange))),
    h("div.card", h("div.card-head", h("h3", icon("power", "sm"), "熄屏继续跟随")), h("div.card-body", battery)),
    h("div.card", h("div.card-body.stack",
      h("div.small.muted", "机器人要在「走跑」模式（站着时会自动切过去）。手柄和网页摇杆随时可以接管；停止后机器人会慢慢停下。摄像头画面直接从机器人头上的 ESP32 走 Wi-Fi 发到手机，同一时间只能有一个地方看 ESP32 的直连画面。")))));

  renderBattery();
  render();
  const timer = setInterval(() => !document.hidden && render(), 250);
  const frameTimer = setInterval(() => !document.hidden && status.running && drawFrame(), 50);
  const onVisible = () => !document.hidden && renderBattery();
  document.addEventListener("visibilitychange", onVisible);
  return () => {
    clearInterval(timer);
    clearInterval(frameTimer);
    document.removeEventListener("visibilitychange", onVisible);
  };
}

// ======================================================================= router

const ROUTES = { boot, home, connect, pair, wifi, robot, offline: robot, remote, follow };
let cleanup = null;

function route() {
  const [path, query] = location.hash.replace(/^#\/?/, "").split("?");
  cleanup?.();
  cleanup = null;
  document.querySelectorAll(".modal-back").forEach((m) => m.remove());
  const page = $("page");
  clear(page);
  page.scrollTop = 0;
  cleanup = (ROUTES[path] || home)(page, new URLSearchParams(query || "")) || null;
}

window.addEventListener("hashchange", route);
route();
