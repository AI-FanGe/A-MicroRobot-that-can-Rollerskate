// Camera: live MJPEG stream straight from the ESP32 over Wi-Fi when it is on the network (the
// API's proxy goes through the Pi's 3 Mbaud UART and is much slower), plus snapshots.

import * as api from "../api.js";
import * as native from "../native.js";
import { clear, empty, h, icon, toast } from "../ui.js";

// The ESP32 serves one stream viewer at a time: the app's follow mode holds it while it runs.
const followRunning = () => !!JSON.parse(native.call("followStatus") || "{}").running;

export default {
  title: "视觉",
  async mount(container) {
    const view = h("div.camera-view");
    const info = h("dl.kv");
    const actions = h("div.actions");
    const shots = h("div.grid", { style: { gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: "10px" } });
    let streaming = false;
    let cam = {};
    const route = h("dd");

    async function load() {
      try {
        cam = await api.get("/hal/camera");
      } catch (error) {
        cam = { available: false, note: error.message };
      }
      clear(info,
        h("dt", "状态"), h("dd", cam.available || cam.esp32_ip ? h("span.badge.ok", "可用") : h("span.badge", "不可用")),
        h("dt", "传输"), route,
        h("dt", "截图"), h("dd.mono", cam.snapshot_url || "—"));
      clear(route, cam.esp32_ip ? `Wi-Fi 直连 ESP32（${cam.esp32_ip}）` : "经树莓派转发（ESP32 没连上 Wi-Fi）");
      if (!cam.available && !cam.esp32_ip) {
        streaming = false;
        clear(view, empty("camera", "摄像头未就绪", cam.note,
          h("button.btn.sm", { onclick: load }, icon("refresh", "sm"), "重试")));
        clear(actions);
        return;
      }
      showIdle();
      renderActions();
    }

    function renderActions() {
      clear(actions,
        h("button.btn.sm", { onclick: () => (streaming ? stop() : start()) },
          icon(streaming ? "stop" : "play", "sm"), streaming ? "关闭实时传输" : "打开实时传输"),
        h("button.btn.sm.primary", { onclick: snapshot }, icon("camera", "sm"), "拍照"));
    }

    function showIdle() {
      streaming = false;
      const img = view.querySelector("img");
      if (img) img.src = "";
      clear(view, empty("camera", "实时传输已关闭", "走路时摄像头保持关闭。需要画面时再打开，再点一下就关掉。"));
    }

    function start(direct = !!cam.esp32_ip && !followRunning()) {
      streaming = true;
      const proxied = `/api/v1/hal/camera/stream?t=${Date.now()}`;
      const src = direct ? `http://${cam.esp32_ip}:81/stream?size=vga&quality=12&t=${Date.now()}` : proxied;
      const img = h("img", { src, alt: "摄像头画面" });
      clear(route, direct ? `Wi-Fi 直连 ESP32（${cam.esp32_ip}）`
        : cam.esp32_ip && followRunning() ? "经树莓派转发（跟随模式正在用直连画面）" : "经树莓派转发");
      img.onerror = () => {
        if (!streaming) return;
        if (direct && cam.available) return start(false);
        streaming = false;
        clear(view, empty("alert", "视频流中断", "摄像头或显示进程可能已重启", h("button.btn.sm", { onclick: load }, "重新连接")));
        renderActions();
      };
      clear(view, img);
      renderActions();
    }

    function stop() {
      showIdle();
      renderActions();
    }

    async function snapshot() {
      try {
        const res = await fetch(`/api/v1/hal/camera/snapshot?t=${Date.now()}`);
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).message || `HTTP ${res.status}`);
        const url = URL.createObjectURL(await res.blob());
        const name = `xiaoxiaobai-${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`;
        shots.prepend(h("a", { href: url, download: name, title: "点击下载" },
          h("img", { src: url, style: { width: "100%", borderRadius: "10px", display: "block" } })));
        toast("已拍照，点击缩略图下载", "ok");
      } catch (error) {
        toast(`拍照失败：${error.message}`, "error");
      }
    }

    container.append(h("div.grid.split-media",
      h("div.card", h("div.card-head", h("h3", icon("camera", "sm"), "实时画面"), actions), h("div.card-body", view)),
      h("div.stack",
        h("div.card", h("div.card-head", h("h3", "摄像头")), h("div.card-body", info,
          h("div.small.muted", { style: { marginTop: "12px" } }, "画面来自机器人头部摄像头（ESP32）。ESP32 连上 Wi-Fi 时直接把画面发给手机，不经过树莓派，更流畅。平时不传图像；打开实时传输才连续出画面，离开本页或再点一次就关掉。语音里说到要看的时候，只拍一帧就关。"))),
        h("div.card", h("div.card-head", h("h3", "本次拍摄")), h("div.card-body", shots)))));
    await load();

    return () => {
      streaming = false;
      const img = view.querySelector("img");
      if (img) img.src = "";
    };
  },
};
