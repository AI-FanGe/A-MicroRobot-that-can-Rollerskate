# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Hands computer tasks to WorkBuddy on a PC.

WorkBuddy's agent is the CodeBuddy Code CLI; started with `codebuddy --serve` it exposes the
Gateway HTTP API: POST /api/v1/runs starts a run, /api/v1/runs/{id}/stream streams it (SSE).
webui/downloads/workbuddy-gateway.* install it on the PC so it starts at login and reports its
address here (register) every REGISTER_EVERY_S, with the password the robot chose (pairing).
"""

from __future__ import annotations

import hmac
import itertools
import json
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from typing import Any, Callable, Iterator

from api.events import EventBus
from api.protocol import ApiError, CommonState, Reason

DEFAULT_TRIGGERS = ["让电脑", "叫电脑", "电脑帮我", "在电脑上", "用电脑", "WorkBuddy", "work buddy"]
DEFAULT_PORT = 8321
TERMINAL = {"done", "completed", "complete", "succeeded", "success", "finished", "failed", "error", "cancelled", "canceled", "timeout", "stopped"}
FAILED = {"failed", "error", "timeout"}
CANCELLED = {"cancelled", "canceled", "stopped"}
# Keys that carry reply text in Gateway outbound events; checked in this order.
TEXT_KEYS = ("text", "delta", "content", "markdown", "result", "output", "reply", "message")
# The gateway reports a signed-out agent as a completed run whose reply is this error.
SIGNED_OUT = re.compile(r"AuthenticationRequiredError|Authentication (required|failed)\. Please use /login")
LOGIN_HINT = ("电脑上的 WorkBuddy 智能体还没登录。在电脑终端里运行 codebuddy（装了 WorkBuddy 的话先设 "
              "CODEBUDDY_CONFIG_DIR=~/.workbuddy），输入 /login 登录一次，再重新发任务。")
KEEP_TASKS = 30
KEEP_RAW = 40
KEEP_STEPS = 60
OUTPUT_TAIL = 1500  # characters of a tool's output kept for the page
THINK_TAIL = 300
PUBLISH_EVERY_S = 0.3
QUIET_OUTPUT = {"Stderr: (empty)", "Signal: (none)", "Exit Code: 0"}
REGISTER_EVERY_S = 30
PREFIX = "（以下是用户对桌面机器人「{name}」说的语音指令，经语音识别转写，可能有同音错字。请在这台电脑上完成，最后用一两句话说明结果。）\n"
QUESTION_WAIT_S = 290  # the agent takes a question as declined after 300 s
LISTEN_S = 90  # after the robot reads a question out, what the user says next within this time is the answer
ORDINALS = ("一", "二", "三", "四", "五", "六")
ASK_PROMPT = ("（这不是我说的话，是系统消息：电脑上的 WorkBuddy 正在帮我做「{task}」，它需要我回答一个问题。"
              "请用一两句话把下面的问题问我，有选项就把选项都念出来，然后停下来等我回答；不要替我回答，不要加别的内容。"
              "我回答以后，你只需要说一句“好的，我告诉电脑”。）\n问题：{question}")
FOLLOWUP_PROMPT = ("（这不是我说的话，是系统消息：电脑上的 WorkBuddy 做完「{task}」后反问了我。请用一两句话把它的问题问我，"
                   "然后停下来等我回答；不要替我回答。我回答以后，你只需要说一句“好的，我告诉电脑”。）\n它说：{text}")


def normalize_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if url.startswith("https://"):
        return url  # e.g. the gateway's Cloudflare tunnel, on 443
    if not url.startswith("http://"):
        url = "http://" + url
    if not re.search(r":\d+$", url.split("//", 1)[1].split("/", 1)[0]):
        url += f":{DEFAULT_PORT}"
    return url


def _text_of(obj: Any) -> tuple[str, bool]:
    """(text, is_delta) found in one stream event; nested payload/data objects are searched too."""
    if isinstance(obj, str):
        return obj, False
    if not isinstance(obj, dict):
        return "", False
    kind = str(obj.get("type") or obj.get("event") or "")
    for key in TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value, key == "delta" or kind.endswith(("delta", "chunk"))
    for key in ("payload", "data", "update", "params", "content"):
        inner = obj.get(key)
        if isinstance(inner, list):
            inner = next((i for i in reversed(inner) if isinstance(i, dict) and i.get("type") == "text"), None)
        if isinstance(inner, dict):
            text, delta = _text_of(inner)
            if text:
                return text, delta or kind.endswith(("delta", "chunk"))
    return "", False


def _is_side_event(obj: Any) -> bool:
    """Tool calls and echoes of the prompt: their status/text say nothing about the run's reply."""
    if not isinstance(obj, dict):
        return False
    kind = str(obj.get("type") or obj.get("event") or obj.get("sessionUpdate") or "").lower()
    if "tool" in kind or any(k in obj for k in ("toolCallId", "tool_call_id", "toolName", "tool_name")):
        return True
    if str(obj.get("role") or "").lower() == "user" or "user_message" in kind:
        return True
    return any(_is_side_event(obj[k]) for k in ("payload", "data", "update", "params") if isinstance(obj.get(k), dict))


def _status_of(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    for key in ("status", "state"):
        value = obj.get(key)
        if isinstance(value, str):
            return value.lower()
    kind = str(obj.get("type") or obj.get("event") or "").lower()
    for word in ("done", "completed", "finished", "failed", "error", "cancelled", "end"):
        if kind == word or kind.endswith(("." + word, "_" + word, ":" + word)):
            return "done" if word == "end" else word
    for key in ("data", "payload"):
        if isinstance(obj.get(key), dict):
            status = _status_of(obj[key])
            if status:
                return status
    return ""


def _question_items(params: dict[str, Any]) -> list[dict[str, Any]]:
    """The questions of a _codebuddy.ai/question request, as the web page and the voice read them."""
    items = []
    for i, q in enumerate(((params.get("schema") or {}).get("questions") or [])[:6]):
        if not isinstance(q, dict):
            continue
        options = [{"label": str(o.get("label") or "")[:120], "description": str(o.get("description") or "")[:300]}
                   for o in q.get("options") or [] if isinstance(o, dict) and o.get("label")]
        items.append({"id": str(q.get("id") or f"q_{i}"), "question": str(q.get("question") or "")[:600],
                      "header": str(q.get("header") or "")[:60], "options": options[:8], "multi": bool(q.get("multiSelect"))})
    return items


def _pick(item: dict[str, Any], text: str) -> str | list[str]:
    """A spoken answer to one question: the options it names (or "第二个"), else the words themselves."""
    labels = [o["label"] for o in item["options"]]
    squeezed = text.replace(" ", "").lower()
    named = [label for label in labels if label.replace(" ", "").lower() in squeezed]
    if not named:
        named = [labels[i] for i, word in enumerate(ORDINALS[:len(labels)]) if f"第{word}" in text or f"第{i + 1}" in text]
    if item["multi"] and named:
        return named
    return named[0] if len(named) == 1 else text


def _asks_back(text: str) -> str:
    """The question a finished reply ends with ("要我再……吗？"), or ""."""
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    last = lines[-1].rstrip("*_` ") if lines else ""
    return last[-300:] if last.endswith(("？", "?")) else ""


class WorkBuddy:
    def __init__(self, settings: Any, events: EventBus, notify: Callable[[dict[str, Any]], None] | None = None, mock: bool = False,
                 speak: Callable[[str], bool] | None = None) -> None:
        self.settings = settings
        self.events = events
        self.notify = notify
        self.speak = speak  # has the robot say a question to the user; False when it cannot
        self.mock = mock
        self._listening: dict[str, Any] | None = None  # the question the user's next utterance answers
        self._tasks: deque[dict[str, Any]] = deque(maxlen=KEEP_TASKS)
        self._lock = threading.Lock()
        self._done = threading.Condition(self._lock)
        self._peer: dict[str, Any] | None = None
        self._acp_lock = threading.Lock()
        self._acp_sessions: dict[str, str] = {}  # gateway URL -> the WorkBuddy session the robot's tasks share

    # ---- config ----------------------------------------------------------------------------

    def config(self) -> dict[str, Any]:
        return self.settings.section("workbuddy")

    def _simulated(self, cfg: dict[str, Any]) -> bool:
        return self.mock and not cfg.get("url")

    def _endpoint(self) -> tuple[str, str]:
        cfg = self.config()
        url = normalize_url(cfg.get("url", ""))
        if not cfg.get("enabled"):
            raise ApiError("电脑助手没有开启（设置 → 电脑助手）", CommonState.FAILURE, Reason.NOT_CONFIGURED)
        if not url and not self._simulated(cfg):
            raise ApiError("还没有填写电脑上 WorkBuddy 的地址（设置 → 电脑助手）", CommonState.FAILURE, Reason.NOT_CONFIGURED)
        return url, (cfg.get("password") or "").strip()

    def _request(self, method: str, url: str, password: str, body: Any = None, timeout: float = 10, stream: bool = False,
                 extra: dict[str, str] | None = None):
        headers = {"X-CodeBuddy-Request": "1", "Accept": "text/event-stream" if stream else "application/json", **(extra or {})}
        parts = urllib.parse.urlsplit(url)
        if parts.scheme == "http" and parts.port:
            # The gateway answers runs only for Host localhost / 127.0.0.1 (DNS-rebinding guard for
            # browsers); the robot reaches it by LAN IP and authenticates with the password instead.
            headers["Host"] = f"localhost:{parts.port}"
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode()
            headers["Content-Type"] = "application/json"
        if password:
            headers["Authorization"] = f"Bearer {password}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code == 401:
                raise ApiError("WorkBuddy 拒绝了密码：请重新填写电脑上启动时显示的 Password", CommonState.FAILURE, Reason.UPSTREAM, 502) from None
            if exc.code == 403:
                raise ApiError(f"WorkBuddy 拒绝了请求（403）：{detail}", CommonState.FAILURE, Reason.UPSTREAM, 502) from None
            err = ApiError(f"WorkBuddy 返回 HTTP {exc.code}：{detail}", CommonState.FAILURE, Reason.UPSTREAM, 502)
            err.upstream_status = exc.code
            raise err from None
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ApiError(f"连不上电脑上的 WorkBuddy（{url.split('/api/')[0]}）：{reason}", CommonState.FAILURE, Reason.UPSTREAM, 502) from None

    def _json(self, method: str, path: str, body: Any = None, timeout: float = 10) -> Any:
        url, password = self._endpoint()
        with self._request(method, url + path, password, body, timeout) as res:
            raw = res.read().decode(errors="replace")
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {"raw": raw[:500]}

    # ---- status ----------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Configuration plus a live check: health, then a call that needs the password."""
        cfg = self.config()
        out: dict[str, Any] = {
            "enabled": bool(cfg.get("enabled")),
            "url": normalize_url(cfg.get("url", "")),
            "has_password": bool(cfg.get("password")),
            "reachable": False,
            "authorized": False,
            "simulated": self._simulated(cfg),
            "running": sum(1 for t in self._tasks if t["status"] in ("queued", "running")),
            "peer": self.peer(),
        }
        if out["simulated"]:
            out.update(reachable=True, authorized=True, message="模拟模式：不连接真实电脑")
            return out
        if not out["url"]:
            out["message"] = "还没有电脑来报到：在电脑上运行右边的安装命令"
            return out
        out.update(self._probe(out["url"], (cfg.get("password") or "").strip()))
        return out

    def _probe(self, url: str, password: str) -> dict[str, Any]:
        """Health, then a call that needs the password."""
        out: dict[str, Any] = {"reachable": False, "authorized": False}
        try:
            # Newer gateways want the password on /health too.
            with self._request("GET", url + "/api/v1/health", password, timeout=4) as res:
                health = json.loads(res.read().decode(errors="replace") or "{}")
            out["reachable"] = True
            out["health"] = health.get("data", health) if isinstance(health, dict) else health
            with self._request("GET", url + "/api/v1/sessions", password, timeout=4):
                pass
            out["authorized"] = True
            out["message"] = "已连接"
        except ApiError as exc:
            out["message"] = exc.message
        return out

    # ---- pairing ---------------------------------------------------------------------------

    def peer(self) -> dict[str, Any] | None:
        """The PC that last reported in, with how long ago."""
        peer = self._peer
        if peer is None:
            return None
        age = time.time() - peer["seen"]
        return {**peer, "age_s": round(age), "online": age < REGISTER_EVERY_S * 3}

    def pairing(self) -> dict[str, Any]:
        """The gateway password the PC installer is given; chosen here the first time."""
        password = (self.config().get("password") or "").strip()
        if not password:
            password = secrets.token_urlsafe(18)
            self.settings.update("workbuddy", {"password": password})
        return {"password": password, "port": DEFAULT_PORT, "peer": self.peer()}

    def register(self, client: str, body: dict[str, Any]) -> dict[str, Any]:
        """A PC's gateway reporting in: its address is where the request came from."""
        password = (self.config().get("password") or "").strip()
        given = str(body.get("password") or "")
        if not password or not hmac.compare_digest(given.encode(), password.encode()):
            raise ApiError("配对密码不对：在小小白网页 设置 → 电脑助手 里重新复制安装命令，在电脑上再运行一次",
                           CommonState.FAILURE, Reason.NOT_CONFIGURED, 403)
        if not client:
            raise ApiError("电脑需要经 Wi-Fi（HTTP）报到", CommonState.INVALID, Reason.NOT_SUPPORTED)
        try:
            port = int(body.get("port") or DEFAULT_PORT)
        except (TypeError, ValueError):
            raise ApiError("port 需要是数字", CommonState.INVALID) from None
        if not 0 < port < 65536:
            raise ApiError("port 超出范围", CommonState.INVALID)
        host = client.removeprefix("::ffff:")
        url = f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"
        moved = self.config().get("url") != url
        if moved:
            # A new address is a new pairing: switch the assistant on. Later reports from the same
            # address leave the switch alone, so turning it off in the web UI sticks.
            self.settings.update("workbuddy", {"url": url, "enabled": True})
        self._peer = {"name": str(body.get("name") or "").strip()[:60], "url": url, "seen": time.time()}
        result = {"url": url, **self._probe(url, password)}
        if not result["reachable"]:
            result["message"] = (f"机器人连不上 {url}：请在电脑防火墙里允许端口 {port} 的传入连接"
                                 f"（Windows 选“专用网络”），并确认电脑和机器人在同一个 Wi-Fi。详情：{result['message']}")
        elif not result["authorized"]:
            result["message"] = ("电脑上的网关没有用机器人给的密码：删掉 WorkBuddy 配置里 settings.json 的 gateway.password 后"
                                 f"重新运行安装命令。详情：{result['message']}")
        if moved:
            self.events.publish("workbuddy_peer", peer=self.peer(), url=url)
        return result

    # ---- triggers --------------------------------------------------------------------------

    def match_voice(self, text: str) -> bool:
        cfg = self.config()
        if not (cfg.get("enabled") and cfg.get("voice_trigger", True)):
            return False
        lowered = text.lower().replace(" ", "")
        return any(t and t.lower().replace(" ", "") in lowered for t in cfg.get("triggers") or DEFAULT_TRIGGERS)

    # ---- tasks -----------------------------------------------------------------------------

    def tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public(t, raw=False) for t in reversed(self._tasks)]

    def get(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = next((t for t in self._tasks if t["id"] == task_id), None)
            if task is None:
                raise ApiError("没有这个电脑任务", CommonState.INVALID, Reason.NOT_FOUND, 404)
            return self._public(task)

    @staticmethod
    def _public(task: dict[str, Any], raw: bool = True) -> dict[str, Any]:
        out = {k: v for k, v in task.items() if not k.startswith("_") and (raw or k != "raw")}
        out["steps"] = [dict(s) for s in task["steps"]]
        if task["question"] is not None:
            out["question"] = {**task["question"], "answers": dict(task["question"]["answers"])}
        return out

    def submit(self, text: str, origin: str = "web", robot_name: str = "", reply_to: str | None = None) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ApiError("任务内容不能为空", CommonState.INVALID)
        self._listening = None
        url, password = ("", "") if self._simulated(self.config()) and self.config().get("enabled") else self._endpoint()
        task = {
            "id": uuid.uuid4().hex[:12],
            "text": text,
            "origin": origin,
            "status": "queued",
            "run_id": None,
            "result": "",
            "error": None,
            "created": time.time(),
            "finished": None,
            "steps": [],  # what the PC is doing: tool calls and what the agent says between them
            "thinking": "",  # the tail of the agent's current reasoning, while it has not acted yet
            "via": None,  # "acp" (live steps) or "runs" (result only)
            "question": None,  # what the agent is waiting for the user to answer
            "reply_to": reply_to if isinstance(reply_to, str) else None,  # the task this one answers or continues
            "raw": [],
            "_cancel": threading.Event(),
            "_answered": threading.Event(),
            "_acp": None,
            "_sent": 0.0,
        }
        with self._lock:
            self._tasks.append(task)
        self._publish(task)
        prompt = PREFIX.format(name=robot_name or "小小白") + text if origin == "voice" else text
        threading.Thread(target=self._run, args=(task, url, password, prompt), name="xiaoxiaobai-api-workbuddy", daemon=True).start()
        return self._public(task)

    def wait(self, task_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._done:
            while True:
                task = next((t for t in self._tasks if t["id"] == task_id), None)
                if task is None or task["status"] not in ("queued", "running"):
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._done.wait(left)
        return self.get(task_id)

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = next((t for t in self._tasks if t["id"] == task_id), None)
        if task is None:
            raise ApiError("没有这个电脑任务", CommonState.INVALID, Reason.NOT_FOUND, 404)
        if task["status"] not in ("queued", "running"):
            return self._public(task)
        task["_cancel"].set()
        if task["_acp"] is not None:
            task["_acp"].cancel()
        elif task["run_id"]:
            try:
                self._json("POST", f"/api/v1/runs/{task['run_id']}/cancel", {}, timeout=5)
            except ApiError:
                pass
        self._finish(task, "cancelled", error="已取消")
        return self._public(task)

    # ---- questions from the agent ----------------------------------------------------------

    def answer(self, task_id: str, answers: Any = None, text: str = "") -> dict[str, Any]:
        """The user's answer to the question a task waits on: {question id: label(s) or words}, or
        `text` for the next unanswered question (spoken answers are matched to its options)."""
        with self._lock:
            task = next((t for t in self._tasks if t["id"] == task_id), None)
            if task is None:
                raise ApiError("没有这个电脑任务", CommonState.INVALID, Reason.NOT_FOUND, 404)
            question = task["question"]
            if question is None:
                raise ApiError("电脑现在没有在等这个任务的回答", CommonState.INVALID, Reason.TARGET_STATE, 409)
            given: dict[str, Any] = {}
            if isinstance(answers, dict):
                ids = {item["id"] for item in question["items"]}
                for key, value in answers.items():
                    values = [str(v).strip() for v in value] if isinstance(value, list) else [str(value or "").strip()]
                    values = [v for v in values if v]
                    if key in ids and values:
                        given[key] = values if isinstance(value, list) else values[0]
            elif (text or "").strip():
                item = next((i for i in question["items"] if i["id"] not in question["answers"]), None)
                if item is not None:
                    given[item["id"]] = _pick(item, text.strip())
            if not given:
                raise ApiError("回答不能为空", CommonState.INVALID)
            question["answers"].update(given)
            complete = all(item["id"] in question["answers"] for item in question["items"])
            if complete:
                task["_answered"].set()
        self._publish(task)
        if not complete:
            self._speak_question(task)
        return self._public(task)

    def voice_reply(self, text: str, robot_name: str = "") -> bool:
        """What the user said right after the robot read out a question: its answer (a question the
        agent ended its reply with gets the answer as a new task in the same session)."""
        listening = self._listening
        if listening is None or not listening["armed"] or time.monotonic() > listening["until"] or not text.strip():
            return False
        if self.match_voice(text):
            return False  # "让电脑……": a new task, not the answer
        self._listening = None
        try:
            if listening["kind"] == "ask":
                self.answer(listening["task"], text=text)
            else:
                self.submit(text, "voice", robot_name, reply_to=listening["task"])
        except ApiError:
            return False
        return True

    def voice_said(self) -> None:
        """The robot finished saying something: a question read out is now waiting for its answer."""
        listening = self._listening
        if listening is not None and not listening["armed"]:
            listening.update(armed=True, until=time.monotonic() + LISTEN_S)

    def _listen(self, task: dict[str, Any], kind: str) -> None:
        # Armed once the robot has said it (voice_said); it may wait for the user to finish first.
        self._listening = {"task": task["id"], "kind": kind, "armed": False, "until": time.monotonic() + LISTEN_S}

    def _ask(self, task: dict[str, Any], qid: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Shows the agent's question on the task (and has the robot read it out), then waits for the
        answer: {question id: answer}, or None when it was not answered in time."""
        question = {"id": qid, "items": items, "answers": {}, "asked": time.time(), "until": time.time() + QUESTION_WAIT_S}
        step = next((s for s in task["steps"] if s.get("id") == qid), None)
        if step is None:
            step = {"id": qid}
            task["steps"].append(step)
        step.update(type="ask", items=items, answers=None, status="waiting", started=time.time(), finished=None)
        with self._lock:
            task["_answered"].clear()
            task["question"] = question
            task["thinking"] = ""
        self._publish(task)
        first = items[0]["question"] if items else ""
        self.events.alert("info", f"电脑在问你：{first[:80]}（在 电脑 页面回答）")
        self._speak_question(task)
        deadline = time.monotonic() + QUESTION_WAIT_S
        while not task["_answered"].wait(1.0):
            if task["_cancel"].is_set() or time.monotonic() > deadline:
                break
        with self._lock:
            answers = dict(question["answers"]) if task["_answered"].is_set() else None
            task["question"] = None
        if self._listening is not None and self._listening["task"] == task["id"]:
            self._listening = None
        step.update(answers=answers, status="answered" if answers else "skipped", finished=time.time())
        self._publish(task)
        return answers

    def _speak_question(self, task: dict[str, Any]) -> None:
        question = task["question"]
        item = next((i for i in question["items"] if i["id"] not in question["answers"]), None) if question else None
        if item is None or self.speak is None:
            return
        text = item["question"]
        if item["options"]:
            text += "\n选项：" + "；".join(o["label"] for o in item["options"]) + ("（可以选多个）" if item["multi"] else "")
        if self.speak(ASK_PROMPT.format(task=task["text"][:60], question=text)):
            self._listen(task, "ask")

    def _speak_followup(self, task: dict[str, Any]) -> None:
        asked = _asks_back(task["result"])
        if not asked or self.speak is None:
            return
        if self.speak(FOLLOWUP_PROMPT.format(task=task["text"][:60], text=asked)):
            self._listen(task, "followup")

    # ---- worker ----------------------------------------------------------------------------

    def _publish(self, task: dict[str, Any], throttle: bool = False) -> None:
        """Every change goes to the web pages; streamed text at most every PUBLISH_EVERY_S."""
        now = time.monotonic()
        if throttle and now - task["_sent"] < PUBLISH_EVERY_S:
            return
        task["_sent"] = now
        self.events.publish("workbuddy", task=self._public(task, raw=False))

    def _finish(self, task: dict[str, Any], status: str, result: str | None = None, error: str | None = None) -> None:
        with self._done:
            if task["status"] not in ("queued", "running"):
                return
            task["status"] = status
            if result is not None:
                task["result"] = result.strip()
            task["error"] = error
            task["finished"] = time.time()
            self._done.notify_all()
        self._publish(task)
        if self.notify:
            try:
                self.notify(self._public(task))
            except Exception:  # noqa: BLE001 - a notification never fails the task
                pass
        # Typed tasks are answered on the page; a spoken one may end on a question the user hears.
        if status == "done" and task["origin"] == "voice":
            self._speak_followup(task)

    def _run(self, task: dict[str, Any], url: str, password: str, prompt: str) -> None:
        task["status"] = "running"
        self._publish(task)
        if not url:
            self._simulate(task)
            return
        timeout_s = float(self.config().get("timeout_s", 600) or 600)
        try:
            try:
                status, text = self._run_acp(task, url, password, prompt, timeout_s)
            except NoAcp:
                status, text = self._run_gateway(task, url, password, prompt, timeout_s)
            if task["_cancel"].is_set():
                return
            if SIGNED_OUT.search(text or ""):
                self._finish(task, "failed", text, error=LOGIN_HINT)
            elif status in FAILED:
                self._finish(task, "failed", text, error=text or "WorkBuddy 执行失败")
            elif status in CANCELLED:
                self._finish(task, "cancelled", text, error="已在电脑上取消")
            else:
                self._finish(task, "done", text or "已完成（WorkBuddy 没有返回文字说明）")
        except ApiError as exc:
            self._finish(task, "failed", error=LOGIN_HINT if SIGNED_OUT.search(exc.message) else exc.message)
        except Exception as exc:  # noqa: BLE001 - surfaced on the task instead of killing the thread
            self._finish(task, "failed", error=f"{type(exc).__name__}: {exc}")

    def _simulate(self, task: dict[str, Any]) -> None:
        """Mock mode: a made-up run with the same kind of steps as a real one."""
        task["via"] = "acp"
        script = [
            ("thinking", "用户想让我在电脑上完成这件事，先看看需要哪些步骤……"),
            ("say", "好的，我先确认一下电脑的状态。"),
            ("tool", ("查询当前系统时间", "date", "2026-11-12 21:30:00 CST")),
            ("tool", ("列出桌面上的文件", "ls ~/Desktop", "截图.png\n笔记.txt")),
        ]
        for kind, value in script:
            time.sleep(1.2)
            if task["_cancel"].is_set():
                return
            if kind == "thinking":
                task["thinking"] = value
            elif kind == "say":
                task["steps"].append({"type": "say", "text": value})
                task["thinking"] = ""
            else:
                detail, code, output = value
                step = {"type": "tool", "id": uuid.uuid4().hex[:8], "kind": "execute", "title": code, "detail": detail, "code": code,
                        "status": "in_progress", "output": "", "started": time.time(), "finished": None}
                task["steps"].append(step)
                self._publish(task)
                time.sleep(1.0)
                step.update(status="completed", output=output, finished=time.time())
            self._publish(task)
        if "问我" in task["text"]:
            # A task that says "ask me" asks, like a real agent using its AskUserQuestion tool.
            answers = self._ask(task, uuid.uuid4().hex[:8], [{
                "id": "q_0", "question": "桌面上有两个文件，要整理哪一个？", "header": "选文件", "multi": False,
                "options": [{"label": "截图.png", "description": "今天的截图"}, {"label": "笔记.txt", "description": "上周的笔记"}],
            }])
            if task["_cancel"].is_set():
                return
            if answers is None:
                self._finish(task, "done", "（模拟）没等到回答，先不整理了。")
                return
            task["steps"].append({"type": "say", "text": f"好的，就整理「{answers['q_0']}」。"})
            self._publish(task)
            time.sleep(1.0)
        ending = "要我再把它发到你手机上吗？" if "反问" in task["text"] else ""
        self._finish(task, "done", f"（模拟）已在电脑上完成：{task['text']}。{ending}")

    def _run_acp(self, task: dict[str, Any], url: str, password: str, prompt: str, timeout_s: float) -> tuple[str, str]:
        """One prompt over the gateway's ACP endpoint, which streams every tool call. Tasks take
        turns on one WorkBuddy session, so a follow-up ("save it to the desktop") has context."""
        with self._acp_lock:
            if task["_cancel"].is_set():
                return "cancelled", ""
            run = AcpRun(self, task, url, password)
            task["_acp"] = run
            task["via"] = "acp"
            try:
                run.open()
                sid = run.session(self._acp_sessions.get(url))
                self._acp_sessions[url] = sid
                return run.prompt(prompt, timeout_s)
            finally:
                run.close()

    def _run_gateway(self, task: dict[str, Any], url: str, password: str, prompt: str, timeout_s: float) -> tuple[str, str]:
        """Older gateways without ACP: POST /api/v1/runs, then its stream (the result only)."""
        task["via"] = "runs"
        body = {
            "id": f"xxb-{task['id']}",
            "type": "message",
            "source": {
                "platform": "generic",
                "sender": {"id": "xiaoxiaobai", "name": "小小白机器人"},
                "conversation": {"id": self.config().get("conversation") or "xiaoxiaobai", "type": "direct"},
            },
            "payload": {"text": prompt},
            "timeoutMs": int(timeout_s * 1000),
        }
        with self._request("POST", url + "/api/v1/runs", password, body, timeout=15) as res:
            started = json.loads(res.read().decode(errors="replace") or "{}")
        data = started.get("data", started)
        task["run_id"] = data.get("runId") or data.get("run_id") or data.get("id")
        if not task["run_id"]:
            raise ApiError(f"WorkBuddy 没有返回 runId：{json.dumps(started, ensure_ascii=False)[:200]}", CommonState.FAILURE, Reason.UPSTREAM)
        self._publish(task)
        return self._follow(task, url, password, timeout_s)

    def _follow(self, task: dict[str, Any], url: str, password: str, timeout_s: float) -> tuple[str, str]:
        """Read the run's SSE stream until it ends; fall back to polling the run."""
        deadline = time.monotonic() + timeout_s + 30
        deltas: list[str] = []
        last_message = ""
        status = ""
        try:
            with self._request("GET", f"{url}/api/v1/runs/{task['run_id']}/stream", password, timeout=timeout_s + 30, stream=True) as res:
                for name, data in _sse(res):
                    if task["_cancel"].is_set():
                        return "cancelled", ""
                    try:
                        obj: Any = json.loads(data)
                    except json.JSONDecodeError:
                        obj = data
                    if isinstance(obj, dict) and name and "type" not in obj:
                        obj = {"type": name, **obj}
                    task["raw"] = (task["raw"] + [f"{name or 'message'}: {data[:600]}"])[-KEEP_RAW:]
                    if _is_side_event(obj):
                        continue
                    text, is_delta = _text_of(obj)
                    if text:
                        if is_delta:
                            deltas.append(text)
                        else:
                            last_message = text
                        task["result"] = (last_message or "".join(deltas))[-4000:]
                        self._publish(task, throttle=True)
                    status = _status_of(obj) or (name.lower() if name.lower() in TERMINAL else "") or status
                    if status in TERMINAL:
                        break
        except ApiError:
            if not task["run_id"]:
                raise
        text = last_message or "".join(deltas)
        while status not in TERMINAL and time.monotonic() < deadline and not task["_cancel"].is_set():
            info = self._json("GET", f"/api/v1/runs/{task['run_id']}", timeout=10)
            data = info.get("data", info) if isinstance(info, dict) else {}
            status = _status_of(data) or status
            found, _ = _text_of(data)
            text = found or text
            if status in TERMINAL:
                break
            time.sleep(2.0)
        if status not in TERMINAL:
            return "timeout", text or "等待 WorkBuddy 超时"
        return status, text


def _sse(res: Any) -> Iterator[tuple[str, str]]:
    name, data = "", []
    for raw in res:
        line = raw.decode(errors="replace").rstrip("\r\n")
        if not line:
            if data:
                yield name, "\n".join(data)
            name, data = "", []
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield name, "\n".join(data)


class NoAcp(Exception):
    """The gateway has no ACP endpoint (an older CodeBuddy): use /api/v1/runs."""


def _chunk_text(update: dict[str, Any]) -> str:
    content = update.get("content")
    return str(content.get("text") or "") if isinstance(content, dict) else ""


def _replayed(update: dict[str, Any]) -> bool:
    """session/load replays a session's history as updates; they are not this task's steps."""
    meta = update.get("_meta") or {}
    return "codebuddy.ai/historyReplay" in meta or (meta.get("codebuddy.ai") or {}).get("mode") == "history"


def _clean_output(text: str) -> str:
    lines = [line for line in text.replace("\r\n", "\n").split("\n") if line.strip() not in QUIET_OUTPUT]
    return "\n".join(lines).removeprefix("Stdout: ").strip()


class AcpRun:
    """One task over the gateway's Agent Client Protocol endpoint (JSON-RPC 2.0): POST
    /api/v1/acp/connect opens a connection, GET /api/v1/acp streams the agent's notifications (SSE),
    POST /api/v1/acp carries requests; replies come back on the POST or on the stream. Every tool
    call and the agent's text arrive as session/update notifications, which become task["steps"]."""

    def __init__(self, wb: WorkBuddy, task: dict[str, Any], url: str, password: str) -> None:
        self.wb = wb
        self.task = task
        self.url = url
        self.password = password
        self.headers: dict[str, str] = {}
        self.sid: str | None = None
        self._ids = itertools.count(1)
        self._replies: dict[int, dict[str, Any]] = {}
        self._cond = threading.Condition()
        self._closed = False
        self._live = False  # set when our prompt goes out: earlier updates belong to session/load
        self._stream: Any = None
        self._text = ""  # what the agent has said since its last tool call

    def open(self) -> None:
        try:
            with self.wb._request("POST", self.url + "/api/v1/acp/connect", self.password, {}, timeout=10) as res:
                conn = json.loads(res.read().decode(errors="replace") or "{}")
        except ApiError as exc:
            if getattr(exc, "upstream_status", None) in (404, 405):
                raise NoAcp from None
            raise
        if not conn.get("connectionId"):
            raise NoAcp
        self.headers = {"acp-connection-id": conn["connectionId"], "acp-session-token": conn.get("sessionToken") or ""}
        timeout_s = float(self.wb.config().get("timeout_s", 600) or 600)
        self._stream = self.wb._request("GET", self.url + "/api/v1/acp", self.password, timeout=timeout_s + 60,
                                        stream=True, extra=self.headers)
        threading.Thread(target=self._listen, name="xiaoxiaobai-api-acp", daemon=True).start()
        # The "question" capability routes the agent's AskUserQuestion tool here (_codebuddy.ai/question).
        self.call("initialize", {"protocolVersion": 1, "clientCapabilities": {
            "fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False,
            "_meta": {"codebuddy.ai": {"question": True}}}}, 20)

    def session(self, previous: str | None) -> str:
        cwd = "/"
        try:
            with self.wb._request("GET", self.url + "/api/v1/info", self.password, timeout=5) as res:
                info = json.loads(res.read().decode(errors="replace") or "{}")
            cwd = (info.get("data") or info).get("cwd") or cwd
        except (ApiError, ValueError, AttributeError):
            pass
        if previous:
            try:
                self.call("session/load", {"sessionId": previous, "cwd": cwd, "mcpServers": []}, 30)
                self.sid = previous
                return previous
            except (ApiError, TimeoutError):
                pass  # gone (gateway restarted): start a new one
        result = self.call("session/new", {"cwd": cwd, "mcpServers": []}, 30)
        self.sid = result.get("sessionId")
        if not self.sid:
            raise ApiError("WorkBuddy 没有返回会话 ID", CommonState.FAILURE, Reason.UPSTREAM)
        return self.sid

    def prompt(self, text: str, timeout_s: float) -> tuple[str, str]:
        self._live = True
        try:
            result = self.call("session/prompt", {"sessionId": self.sid, "prompt": [{"type": "text", "text": text}]}, timeout_s)
        except TimeoutError:
            self.cancel()
            return "timeout", "等待 WorkBuddy 超时"
        said = self._text.strip() or next((s["text"] for s in reversed(self.task["steps"]) if s["type"] == "say"), "")
        self.task["thinking"] = ""
        return ("cancelled" if result.get("stopReason") == "cancelled" else "done"), said

    def cancel(self) -> None:
        if self.sid and not self._closed:
            threading.Thread(target=self._post, args=({"jsonrpc": "2.0", "method": "session/cancel",
                                                       "params": {"sessionId": self.sid}}, 10), daemon=True).start()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self.headers:
            try:
                self.wb._request("DELETE", self.url + "/api/v1/acp", self.password, timeout=3, extra=self.headers).close()
            except ApiError:
                pass
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass

    # ---- JSON-RPC ----------------------------------------------------------------------------

    def call(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        rid = next(self._ids)
        msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        threading.Thread(target=self._post, args=(msg, timeout), name="xiaoxiaobai-api-acp-post", daemon=True).start()
        deadline = time.monotonic() + timeout
        with self._cond:
            while rid not in self._replies:
                if self._closed:
                    raise ApiError("和电脑上 WorkBuddy 的连接断开了", CommonState.FAILURE, Reason.UPSTREAM)
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(method)
                self._cond.wait(min(left, 1.0))
            reply = self._replies.pop(rid)
        if "error" in reply:
            err = reply.get("error") or {}
            raise ApiError(f"WorkBuddy：{err.get('message') if isinstance(err, dict) else err}", CommonState.FAILURE, Reason.UPSTREAM)
        return reply.get("result") or {}

    def _post(self, msg: dict[str, Any], timeout: float) -> None:
        try:
            with self.wb._request("POST", self.url + "/api/v1/acp", self.password, msg, timeout=timeout + 30,
                                  extra={**self.headers, "Accept": "application/json, text/event-stream"}) as res:
                if "event-stream" in (res.headers.get("Content-Type") or ""):
                    for _, data in _sse(res):
                        self._received(data)
                else:
                    raw = res.read().decode(errors="replace").strip()
                    if raw:
                        self._received(raw)
        except (ApiError, OSError) as exc:
            if "method" in msg and "id" in msg:
                self._deliver({"id": msg["id"], "error": {"message": getattr(exc, "message", str(exc))}})

    def _listen(self) -> None:
        try:
            for _, data in _sse(self._stream):
                self._received(data)
        except (OSError, ValueError):
            pass
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def _received(self, data: str) -> None:
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return
        for msg in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(msg, dict):
                continue
            if "method" in msg and "id" in msg:
                self._answer(msg)
            elif msg.get("method") == "session/update":
                self._update(msg.get("params") or {})
            elif "id" in msg and "method" not in msg:
                self._deliver(msg)

    def _deliver(self, reply: dict[str, Any]) -> None:
        with self._cond:
            self._replies[reply["id"]] = reply
            self._cond.notify_all()

    def _answer(self, msg: dict[str, Any]) -> None:
        """Requests from the agent. Permission is granted, as /api/v1/runs does for the whole run;
        a question waits for the user, off the thread that reads the stream."""
        if msg["method"] == "_codebuddy.ai/question":
            threading.Thread(target=self._ask, args=(msg,), name="xiaoxiaobai-api-acp-ask", daemon=True).start()
            return
        if msg["method"] == "session/request_permission":
            options = (msg.get("params") or {}).get("options") or []
            pick = (next((o for o in options if o.get("kind") == "allow_once"), None)
                    or next((o for o in options if str(o.get("kind", "")).startswith("allow")), None))
            outcome = {"outcome": "selected", "optionId": pick["optionId"]} if pick else {"outcome": "cancelled"}
            reply: dict[str, Any] = {"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": outcome}}
        else:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"{msg['method']} is not supported"}}
        threading.Thread(target=self._post, args=(reply, 15), daemon=True).start()

    def _ask(self, msg: dict[str, Any]) -> None:
        params = msg.get("params") or {}
        items = _question_items(params)
        answers = self.wb._ask(self.task, str(params.get("toolCallId") or msg["id"]), items) if items else None
        # The agent reads any other outcome as "User declined to answer questions".
        result = {"outcome": "submitted", "answers": answers} if answers else {"outcome": "cancelled"}
        self._post({"jsonrpc": "2.0", "id": msg["id"], "result": result}, 15)

    # ---- progress ----------------------------------------------------------------------------

    def _update(self, params: dict[str, Any]) -> None:
        update = params.get("update") or {}
        if not self._live or params.get("sessionId") not in (None, self.sid) or _replayed(update):
            return
        kind = update.get("sessionUpdate")
        task = self.task
        if kind == "agent_message_chunk":
            self._text += _chunk_text(update)
            task["result"] = self._text[-4000:]
            task["thinking"] = ""
            self.wb._publish(task, throttle=True)
        elif kind == "agent_thought_chunk":
            task["thinking"] = (task["thinking"] + _chunk_text(update))[-THINK_TAIL:]
            self.wb._publish(task, throttle=True)
        elif kind in ("tool_call", "tool_call_update"):
            self._tool(update)

    def _tool(self, update: dict[str, Any]) -> None:
        task = self.task
        steps = task["steps"]
        tid = update.get("toolCallId")
        step = next((s for s in steps if s.get("id") == tid), None)
        if step is not None and step["type"] == "ask":
            return  # the AskUserQuestion call: its question and answer are the step
        if step is None:
            # What the agent said before acting ("先打开文本编辑") is a step of its own.
            if self._text.strip():
                steps.append({"type": "say", "text": self._text.strip()[-2000:]})
                self._text = ""
                task["result"] = ""
            step = {"type": "tool", "id": tid, "kind": "", "title": "", "detail": "", "code": "", "status": "pending",
                    "output": "", "started": time.time(), "finished": None}
            steps.append(step)
            del steps[:-KEEP_STEPS]
            task["thinking"] = ""
            task["raw"] = (task["raw"] + [f"tool_call: {json.dumps(update, ensure_ascii=False)[:600]}"])[-KEEP_RAW:]
        raw_in = update.get("rawInput") if isinstance(update.get("rawInput"), dict) else {}
        if raw_in.get("description"):
            step["detail"] = str(raw_in["description"])[:200]
        code = next((raw_in[k] for k in ("command", "file_path", "path", "url", "pattern", "query") if raw_in.get(k)), None)
        if code:
            step["code"] = str(code)[:500]
        if isinstance(update.get("title"), str) and update["title"]:
            step["title"] = update["title"].strip("`")[:200]
        if update.get("kind"):
            step["kind"] = str(update["kind"])
        status = update.get("status")
        if status:
            step["status"] = str(status)
            if status in ("completed", "failed") and not step["finished"]:
                step["finished"] = time.time()
        output = update.get("rawOutput")
        if output:
            text = output.get("text") if isinstance(output, dict) else output
            step["output"] = _clean_output(str(text or ""))[-OUTPUT_TAIL:]
        # Status changes go out right away; the arguments streaming in are throttled.
        self.wb._publish(task, throttle=not status and not output)
