// Music: NetEase Cloud Music on the robot's speaker. Now playing + queue, search, quick lists,
// the NetEase account login (QR code / SMS code / cookie) and the music settings.

import * as api from "../api.js";
import { on } from "../store.js";
import { busy, clear, empty, field, h, icon, rangeEl, select, switchEl, toast } from "../ui.js";

const STATE = { idle: "空闲", searching: "搜索中", loading: "缓冲中", playing: "播放中", paused: "已暂停", error: "出错" };
const QUALITY = [["standard", "标准 128k"], ["higher", "较高 192k"], ["exhigh", "极高 320k"]];
const QR_POLL_MS = 2000;
const SMS_COOLDOWN_S = 60;
// NetEase Yidun's NECaptcha, the slider of NetEase's own verify page (-462 with verifyType 40 / 41).
const NECAPTCHA_SDK = "https://cstaticdun.126.net/load.min.js";

let necaptcha = null;
function loadNecaptcha() {
  if (window.initNECaptcha) return Promise.resolve();
  necaptcha ??= new Promise((resolve, reject) => {
    const script = h("script", { src: NECAPTCHA_SDK, async: true });
    script.onload = () => (window.initNECaptcha ? resolve() : reject(new Error("易盾滑块脚本里没有 initNECaptcha")));
    script.onerror = () => {
      necaptcha = null;
      script.remove();
      reject(new Error("加载不了网易易盾滑块（这台电脑连不上 cstaticdun.126.net，或被广告拦截插件挡了）"));
    };
    document.head.append(script);
  });
  return necaptcha;
}

const mmss = (s) => {
  s = Math.max(0, Math.floor(s || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

function songMeta(t) {
  return [t.artist, t.album].filter(Boolean).join(" · ");
}

function songBadges(t) {
  return [
    t.playable === false ? h("span.badge.danger", "播不了") : null,
    t.trial ? h("span.badge.warn", "试听") : t.vip ? h("span.badge.accent", "VIP") : null,
  ];
}

export default {
  title: "音乐",
  async mount(container) {
    let status = null;
    let queue = { tracks: [], index: -1, source: "" };
    let results = [];
    let resultsLabel = "";
    let account = null;
    let settings = null;
    let searching = "";
    let qr = null; // { key, timer }
    let smsTimer = null;
    let tick = null;
    const offs = [];

    const nowBody = h("div.card-body.stack");
    const queueBody = h("div.card-body.flush.music-list");
    const queueCount = h("span.sub");
    const resultsBody = h("div.card-body.flush.music-list");
    const resultsHead = h("span.sub");
    const accountBody = h("div.card-body.stack");
    const settingsBody = h("div.card-body.stack");

    // ---- now playing ----
    const cover = h("div.music-cover", icon("music"));
    const title = h("div.music-title");
    const meta = h("div.music-meta");
    const stateChip = h("span.chip");
    const note = h("div.small", { hidden: true });
    const bar = h("div.music-bar", h("span"));
    const posText = h("span.mono.small");
    const durText = h("span.mono.small");
    const playBtn = h("button.btn.primary.icon-only", { title: "播放 / 暂停", onclick: () => toggle() }, icon("play"));
    const volume = rangeEl({
      min: 0, max: 100, value: 60, format: (v) => `${v}`,
      onchange: (v) => api.post("/music/volume", { volume: v }).catch((e) => toast(e.message, "error")),
    });

    const control = (path, button) => busy(button, () => api.post(path).then((r) => {
      if (r.state?.reason_name === "REASON_TARGET_STATE") toast(r.message, "warn");
    }).catch((e) => toast(e.message, "error")));

    function toggle() {
      if (!status) return;
      if (status.state === "playing") control("/music/pause", playBtn);
      else if (status.state === "paused") control("/music/resume", playBtn);
      else if (queue.tracks.length) play({ index: Math.max(0, status.index) }, playBtn);
      else play({ playlist: "hot" }, playBtn);
    }

    nowBody.append(
      h("div.music-now", cover,
        h("div.main-col", h("div.row", stateChip), title, meta, note)),
      h("div.music-progress", posText, bar, durText),
      h("div.row.wrap", { style: { justifyContent: "space-between" } },
        h("div.row",
          h("button.btn.icon-only", { title: "上一首", onclick: (e) => control("/music/previous", e.currentTarget) }, icon("skipB")),
          playBtn,
          h("button.btn.icon-only", { title: "下一首", onclick: (e) => control("/music/next", e.currentTarget) }, icon("skipF")),
          h("button.btn.icon-only", { title: "停止", onclick: (e) => control("/music/stop", e.currentTarget) }, icon("stop"))),
        h("div.row", { style: { minWidth: "200px", flex: "1", maxWidth: "320px" } }, icon("volume", "sm"), volume)),
    );

    function renderNow() {
      const s = status || {};
      const t = s.track;
      const state = searching ? "searching" : s.state || "idle";
      stateChip.className = `chip ${state === "playing" ? "ok" : state === "error" ? "danger" : state === "idle" ? "" : "busy"}`;
      clear(stateChip, h("span.dot"), STATE[state] || state, s.source ? h("span.muted", ` · ${s.source}`) : null);
      clear(cover, t?.cover ? h("img", { src: `${t.cover}?param=160y160`, alt: "", referrerpolicy: "no-referrer" }) : icon("music"));
      title.textContent = searching ? `正在搜索「${searching}」…` : t ? t.name : "还没有放歌";
      meta.textContent = t ? songMeta(t) : "搜一首歌，或者对机器人说“放一首晴天”";
      const lines = [];
      if (s.ffmpeg === false) lines.push(["danger", "机器人上没有 ffmpeg，放不了：在电脑上运行 make music-setup"]);
      if (s.error) lines.push(["danger", s.error]);
      if (s.notice) lines.push(["warn", s.notice]);
      if (s.voice_paused) lines.push(["muted", "语音对话已暂停，音乐停了会自动恢复"]);
      note.hidden = !lines.length;
      clear(note, lines.map(([tone, text]) => h(`div.${tone === "muted" ? "muted" : `text-${tone}`}`, text)));
      playBtn.replaceChildren(icon(s.state === "playing" ? "pause" : "play"));
      if (document.activeElement !== volume.input && s.volume != null) volume.set(s.volume);
      renderProgress();
    }

    function renderProgress() {
      const t = status?.track;
      const dur = t?.duration_s || 0;
      const pos = Math.min(status?.position_s || 0, dur || Infinity);
      posText.textContent = mmss(pos);
      durText.textContent = dur ? mmss(dur) : "--:--";
      bar.firstChild.style.width = dur ? `${Math.min(100, (pos / dur) * 100)}%` : "0%";
    }

    // ---- queue ----
    function renderQueue() {
      queueCount.textContent = queue.tracks.length ? `${queue.tracks.length} 首 · ${queue.source}` : "";
      if (!queue.tracks.length) {
        clear(queueBody, empty("music", "播放列表是空的", "搜歌后点播放，搜到的歌会排进来"));
        return;
      }
      clear(queueBody, queue.tracks.map((t, i) => h(`div.list-item.music-row${i === queue.index ? ".current" : ""}${t.playable === false ? ".off" : ""}`,
        h("span.music-num", i === queue.index && status?.state === "playing" ? icon("volume", "sm") : String(i + 1)),
        h("div.main-col", h("div.title", t.name, songBadges(t)), h("div.meta", songMeta(t))),
        h("span.mono.small.muted", mmss(t.duration_s)),
        h("div.item-actions",
          h("button.btn.xs.ghost.icon-only", { title: "播放这首", onclick: (e) => play({ index: i }, e.currentTarget) }, icon("play", "sm"))))));
    }

    // ---- search ----
    const searchInput = h("input.input", { placeholder: "歌名、歌手…（例如：晴天 周杰伦）" });
    const searchBtn = h("button.btn.primary", { onclick: () => search() }, icon("search", "sm"), "搜索");
    searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.isComposing) search();
    });

    async function search() {
      const q = searchInput.value.trim();
      if (!q) return;
      await busy(searchBtn, async () => {
        try {
          const data = await api.get(`/music/search?q=${encodeURIComponent(q)}`);
          results = data.tracks;
          resultsLabel = `搜索「${q}」`;
          renderResults();
        } catch (e) {
          toast(e.message, "error");
        }
      });
    }

    function renderResults() {
      resultsHead.textContent = results.length ? `${resultsLabel} · ${results.length} 首` : "";
      if (!results.length) {
        clear(resultsBody, empty("search", resultsLabel ? "没搜到" : "搜一首歌", resultsLabel ? "换个关键词试试" : "灰色的是网易云不让播的（版权或 VIP）"));
        return;
      }
      clear(resultsBody, results.map((t, i) => h(`div.list-item.music-row${t.playable === false ? ".off" : ""}`,
        h("div.main-col", h("div.title", t.name, songBadges(t)), h("div.meta", songMeta(t))),
        h("span.mono.small.muted", mmss(t.duration_s)),
        h("div.item-actions",
          h("button.btn.xs.icon-only", {
            title: t.playable === false ? "播不了（版权或 VIP）" : "播放（搜索结果排进播放列表）",
            disabled: t.playable === false,
            onclick: (e) => play({ tracks: results, index: i, source: resultsLabel }, e.currentTarget),
          }, icon("play", "sm"))))));
    }

    async function play(body, button) {
      await busy(button, () => api.post("/music/play", body).then((r) => toast(r.message, "ok")).catch((e) => toast(e.message, "error")));
    }

    // ---- account ----
    function stopQr() {
      if (qr?.timer) clearTimeout(qr.timer);
      qr = null;
    }

    // NetEase risk control (-462) wants a verification first: a slider to drag here, or a QR code
    // for its app. Once done, the robot runs the refused step again and done() gets what it
    // answered (a CommonResponse, or { error }).
    let verifyTimer = null;
    let slider = null;
    function stopVerify() {
      clearTimeout(verifyTimer);
      verifyTimer = null;
      try {
        slider?.destroy?.();
      } catch {
        // already gone with its element
      }
      slider = null;
    }
    function verifyBox(verify, done) {
      stopVerify();
      return verify.kind === "slider" ? sliderBox(verify, done) : verifyQrBox(verify, done);
    }
    function sliderBox(verify, done) {
      const text = h("div.small.muted", "加载网易易盾的滑块…");
      const host = h("div.music-slider");
      const box = h("div.stack", h("div.small.text-warn", "网易云要求先做一次安全验证：拖动下面的滑块拼好拼图"), host, text);
      const mount = async (again = false) => {
        try {
          await loadNecaptcha();
        } catch (e) {
          text.textContent = e.message;
          return;
        }
        if (!box.isConnected) return;
        stopVerify();
        const el = h("div", { id: `necaptcha-${Date.now()}` });
        host.replaceChildren(el);
        window.initNECaptcha({
          captchaId: verify.captcha_id,
          element: `#${el.id}`,
          mode: "embed",
          width: "320px",
          lang: "zh-CN",
          // NECaptcha only calls the onVerify given here, not one passed with the instance.
          onVerify: async (err, data) => {
            const validate = data?.validate || data?.data?.validate;
            if (err || !validate) return; // a failed drag: NECaptcha resets itself
            text.textContent = "提交验证…";
            try {
              const r = await api.post("/music/login/verify", { key: verify.key, validate });
              stopVerify();
              done(r.data?.retry || {});
            } catch (e) {
              text.textContent = e.message;
              mount(true); // a validate is good for one try
            }
          },
        }, (instance) => {
          slider = instance;
          if (!again) text.textContent = verify.text || "";
        }, (e) => {
          text.textContent = `滑块加载失败：${e?.message || e}`;
        });
      };
      queueMicrotask(mount);
      return box;
    }
    function verifyQrBox(verify, done) {
      const text = h("div.small", verify.text ? `${verify.text}（网易云 App → 扫一扫）` : "用网易云 App 扫一扫完成安全验证");
      const box = h("div.stack",
        h("div.small.text-warn", "网易云要求先做一次安全验证"),
        h("div.music-qr", h("div", { html: verify.svg })), text);
      const poll = () => {
        verifyTimer = setTimeout(async () => {
          if (!box.isConnected) return;
          try {
            const r = await api.get(`/music/login/verify/${encodeURIComponent(verify.key)}`);
            text.textContent = r.text;
            if (r.status === 20) {
              verifyTimer = null;
              done(r.retry || {});
              return;
            }
            if (r.status === 21) return;
          } catch (e) {
            text.textContent = e.message;
            return;
          }
          poll();
        }, QR_POLL_MS);
      };
      poll();
      return box;
    }

    async function loadAccount(refresh = false) {
      try {
        account = await api.get(`/music/account${refresh ? "?refresh=1" : ""}`);
      } catch (e) {
        account = { logged_in: false, error: e.message };
      }
      renderAccount();
    }

    let loginTab = "qr";
    function renderAccount() {
      if (account?.logged_in) {
        stopQr();
        clear(accountBody,
          h("div.row",
            account.avatar ? h("img.music-avatar", { src: `${account.avatar}?param=80y80`, alt: "", referrerpolicy: "no-referrer" }) : icon("user"),
            h("div.main-col",
              h("b", account.nickname || "网易云用户", account.vip ? h("span.badge.accent", { style: { marginLeft: "8px" } }, "黑胶 VIP") : null),
              h("span.muted.small", account.vip ? "VIP 歌曲可以完整播放" : "不是 VIP：VIP 歌曲只能放试听片段")),
            h("button.btn.sm", {
              onclick: (e) => busy(e.currentTarget, () => api.post("/music/logout").then(() => loadAccount(true)).catch((err) => toast(err.message, "error"))),
            }, icon("logout", "sm"), "退出")),
          h("div.row.wrap",
            h("button.btn.sm", { onclick: (e) => play({ playlist: "liked" }, e.currentTarget) }, icon("music", "sm"), "放我喜欢的音乐"),
            h("button.btn.sm", { onclick: (e) => play({ playlist: "daily" }, e.currentTarget) }, icon("sparkles", "sm"), "放每日推荐")));
        return;
      }
      const tabs = h("div.seg.sm",
        [["qr", "扫码登录"], ["sms", "验证码登录"], ["cookie", "Cookie 登录"]].map(([key, label]) =>
          h(`button${loginTab === key ? ".active" : ""}`, {
            onclick: () => {
              loginTab = key;
              if (key !== "qr") stopQr();
              renderAccount();
            },
          }, label)));
      const pane = loginTab === "qr" ? qrPane() : loginTab === "sms" ? smsPane() : cookiePane();
      clear(accountBody,
        h("div.muted.small", account?.expired ? "登录已过期，请重新登录。" : "不登录也能搜歌、放免费歌曲；登录后能放 VIP 歌曲（需黑胶 VIP）、「我喜欢的音乐」和每日推荐。"),
        tabs, pane,
        h("div.muted.small", "登录信息只保存在机器人上（~/.config/xiaoxiaobai/netease_cookies.json，仅本机用户可读）。"));
    }

    function qrPane() {
      const box = h("div.music-qr", empty("phone", "用网易云音乐 App 扫码", "点下面的按钮生成二维码"));
      const text = h("div.small.muted");
      const verifyArea = h("div");
      const start = async (button) => {
        stopQr();
        stopVerify();
        await busy(button, async () => {
          try {
            const data = await api.post("/music/login/qr", {});
            box.classList.remove("expired");
            box.replaceChildren(h("div", { html: data.data.svg }));
            verifyArea.replaceChildren();
            text.textContent = "打开网易云音乐 App → 左上角菜单 / 我的 → 扫一扫";
            qr = { key: data.data.key, timer: null };
            poll(data.data.key);
          } catch (e) {
            toast(e.message, "error");
          }
        });
      };
      // r: the state of a QR code (polled, or checked again after a verification); true: poll on.
      const follow = (key, r) => {
        text.textContent = r.text;
        if (r.code === 803) {
          toast("网易云登录成功", "ok");
          stopQr();
          loadAccount(true);
          return false;
        }
        if (r.verify) {
          stopQr();
          box.classList.add("expired");
          verifyArea.replaceChildren(verifyBox(r.verify, (retry) => {
            verifyArea.replaceChildren();
            if (retry.error || !retry.data) {
              text.textContent = retry.error || retry.message || "验证后网易云没有回应，请刷新二维码";
              return;
            }
            box.classList.remove("expired");
            qr = { key, timer: null };
            if (follow(key, retry.data)) poll(key);
          }));
          return false;
        }
        if (![801, 802].includes(r.code)) {
          stopQr();
          box.classList.add("expired");
          return false;
        }
        return true;
      };
      const poll = (key) => {
        if (!qr || qr.key !== key) return;
        qr.timer = setTimeout(async () => {
          if (!qr || qr.key !== key) return;
          try {
            const r = await api.get(`/music/login/qr/${encodeURIComponent(key)}`);
            if (!box.isConnected || !follow(key, r)) return;
          } catch (e) {
            text.textContent = e.message;
          }
          poll(key);
        }, QR_POLL_MS);
      };
      return h("div.stack", box, text, verifyArea,
        h("div.row", h("button.btn.primary.sm", { onclick: (e) => start(e.currentTarget) }, icon("refresh", "sm"), qr ? "刷新二维码" : "生成二维码")));
    }

    function smsPane() {
      const phone = h("input.input", { placeholder: "手机号", inputmode: "numeric", autocomplete: "tel" });
      const code = h("input.input", { placeholder: "验证码", inputmode: "numeric", autocomplete: "one-time-code" });
      const sendBtn = h("button.btn", "发送验证码");
      const verifyArea = h("div");
      const cooldown = () => {
        let left = SMS_COOLDOWN_S;
        clearInterval(smsTimer);
        smsTimer = setInterval(() => {
          left -= 1;
          sendBtn.disabled = left > 0;
          sendBtn.textContent = left > 0 ? `${left} 秒后重发` : "发送验证码";
          if (left <= 0) clearInterval(smsTimer);
        }, 1000);
      };
      // r: the CommonResponse of a send / login, straight or run again after a verification.
      const sent = (r) => {
        if (r.error) return toast(r.error, "error");
        if (r.data?.verify) return verifyArea.replaceChildren(verifyBox(r.data.verify, sent));
        verifyArea.replaceChildren();
        toast(r.message || "验证码已发送", "ok");
        cooldown();
      };
      const loggedIn = (r) => {
        if (r.error) return toast(r.error, "error");
        if (r.data?.verify) return verifyArea.replaceChildren(verifyBox(r.data.verify, loggedIn));
        verifyArea.replaceChildren();
        toast("网易云登录成功", "ok");
        loadAccount(true);
      };
      sendBtn.addEventListener("click", () => busy(sendBtn, () => api.post("/music/login/sms/send", { phone: phone.value })
        .then(sent)
        .catch((e) => toast(e.message, "error"))));
      const loginBtn = h("button.btn.primary", {
        onclick: (e) => busy(e.currentTarget, () => api.post("/music/login/sms", { phone: phone.value, code: code.value })
          .then(loggedIn)
          .catch((err) => toast(err.message, "error"))),
      }, "登录");
      return h("div.stack",
        h("div.input-group", phone, sendBtn),
        h("div.input-group", code, loginBtn),
        verifyArea);
    }

    function cookiePane() {
      const text = h("textarea.input", { rows: 3, placeholder: "MUSIC_U=…; __csrf=…（或者只粘贴 MUSIC_U 的值）" });
      return h("div.stack",
        h("div.muted.small", "扫码和验证码被网易云风控拦住时用这个：电脑浏览器登录 music.163.com → F12 → 应用(Application) → Cookie → 复制 MUSIC_U 的值。"),
        text,
        h("div.row", h("button.btn.primary.sm", {
          onclick: (e) => busy(e.currentTarget, () => api.post("/music/login/cookie", { cookie: text.value })
            .then(() => {
              toast("网易云登录成功", "ok");
              loadAccount(true);
            })
            .catch((err) => toast(err.message, "error"))),
        }, "登录")));
    }

    // ---- settings ----
    async function renderSettings() {
      try {
        settings = (await api.get("/settings")).music;
      } catch (e) {
        clear(settingsBody, empty("alert", "读取设置失败", e.message));
        return;
      }
      const save = (values) => api.patch("/settings/music", values).then((r) => {
        settings = r.data;
        toast("已保存", "ok");
      }).catch((e) => {
        toast(e.message, "error");
        renderSettings();
      });
      const row = (label, hint, key) => h("div.toggle-row", h("div.text", h("b", label), h("small", hint)),
        switchEl(settings[key] !== false, (v) => save({ [key]: v })));
      clear(settingsBody,
        row("启用音乐", "关掉后语音和聊天都不会放歌", "enabled"),
        row("语音点歌", "语音对话里说“放一首晴天”“来首歌”“放我喜欢的歌”就会放", "voice_control"),
        row("放完恢复语音对话", "放歌时语音对话会先暂停（喇叭只有一个），放完或停止后自动重新开始", "resume_voice"),
        row("屏幕显示歌名", "开始放一首歌时在脸上的点阵屏显示歌名几秒", "show_on_screen"),
        row("跟着节拍晃动", "在「走跑·上肢自由」模式下放歌时，手臂和头（13–20 号关节）跟着歌的结构跳：歌下载完会分析出每一拍、小节和段落（前奏、主歌、副歌、间奏），动作在段落开头切换、段落结尾加花，重复的段落跳同一套，最后一拍收住", "groove"),
        field("晃动幅度", rangeEl({
          min: 0, max: 100, step: 5, value: settings.groove_size ?? 70, format: (v) => `${v}%`,
          onchange: (v) => save({ groove_size: v }),
        }), "副歌比主歌跳得大，前奏、间奏轻轻晃；走得越快幅度越小，做动作时让给动作"),
        field("动作时机", rangeEl({
          min: -300, max: 300, step: 10, value: settings.groove_offset_ms ?? 0,
          format: (v) => (v > 0 ? `提前 ${v} ms` : v < 0 ? `推后 ${-v} ms` : "不调"),
          onchange: (v) => save({ groove_offset_ms: v }),
        }), "动作总比音乐慢半拍就往「提前」调，抢拍就往「推后」调（广播体操配乐也按这个对齐）"),
        row("动作配乐", "「广播体操」先放配乐，第一拍响起后准时开始；配乐按动作的节拍合成", "dance_music"),
        field("配乐音量", rangeEl({
          min: 0, max: 100, step: 5, value: settings.dance_volume ?? 80, format: (v) => `${v}`,
          onchange: (v) => save({ dance_volume: v }),
        }), "做操时舵机声音大，单独设得比平时听歌响一些"),
        field("音质", select(QUALITY, settings.quality || "exhigh", (v) => save({ quality: v })),
          "机器人喇叭很小，标准音质就够；网络慢时选低一点开播更快"));
    }

    // ---- data ----
    async function loadStatus() {
      try {
        status = await api.get("/music");
        renderNow();
      } catch (e) {
        clear(nowBody, empty("alert", "读取失败", e.message));
      }
    }

    async function loadQueue() {
      try {
        queue = await api.get("/music/queue");
        renderQueue();
      } catch {
        /* the status card shows connection problems */
      }
    }

    offs.push(on("music", (ev) => {
      if (ev.state === "searching") {
        searching = ev.query || "歌";
        renderNow();
        return;
      }
      searching = "";
      const indexChanged = !status || status.index !== ev.index || status.track?.id !== ev.track?.id;
      status = { ...(status || {}), state: ev.state, track: ev.track, index: ev.index, position_s: ev.position_s, error: ev.error, notice: ev.notice };
      renderNow();
      if (indexChanged || ev.finished) {
        loadStatus();
        loadQueue();
      } else {
        renderQueue();
      }
    }));
    offs.push(on("music_account", () => loadAccount(true)));
    offs.push(on("settings", (ev) => ev.section === "music" && loadStatus()));

    container.append(
      h("div.grid.cols-2", { style: { alignItems: "start" } },
        h("div.stack",
          h("div.card", h("div.card-head", h("h3", icon("music", "sm"), "正在播放")), nowBody),
          h("div.card", h("div.card-head", h("h3", icon("layers", "sm"), "播放列表"), queueCount), queueBody),
          h("div.card",
            h("div.card-head", h("h3", icon("mic", "sm"), "语音点歌")),
            h("div.card-body.small.muted",
              "语音对话时对机器人说：“放一首晴天”“我想听周杰伦的歌”“来首歌”“放我喜欢的歌”“放每日推荐”。",
              h("br"),
              "放歌时语音对话会暂停（机器人会听到自己的歌声），想切歌、停止就在这个页面操作，或者在文字聊天里说“下一首”“别放了”。"))),
        h("div.stack",
          h("div.card",
            h("div.card-head", h("h3", icon("search", "sm"), "找歌"), resultsHead),
            h("div.card-body.stack",
              h("div.input-group", searchInput, searchBtn),
              h("div.row.wrap",
                h("button.btn.sm", { onclick: (e) => play({ playlist: "hot" }, e.currentTarget) }, icon("zap", "sm"), "热歌榜"),
                h("button.btn.sm", { onclick: (e) => play({ playlist: "daily" }, e.currentTarget) }, icon("sparkles", "sm"), "每日推荐"),
                h("button.btn.sm", { onclick: (e) => play({ playlist: "liked" }, e.currentTarget) }, icon("music", "sm"), "我喜欢的音乐"))),
            resultsBody),
          h("div.card", h("div.card-head", h("h3", icon("user", "sm"), "网易云账号")), accountBody),
          h("div.card", h("div.card-head", h("h3", icon("settings", "sm"), "设置")), settingsBody))),
    );

    renderResults();
    await Promise.all([loadStatus(), loadQueue(), loadAccount(), renderSettings()]);
    tick = setInterval(() => {
      if (status?.state === "playing") {
        status.position_s = (status.position_s || 0) + 0.5;
        renderProgress();
      }
    }, 500);
    return () => {
      clearInterval(tick);
      clearInterval(smsTimer);
      stopQr();
      stopVerify();
      offs.forEach((off) => off());
    };
  },
};
