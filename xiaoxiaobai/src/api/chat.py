# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Text chat with the robot: persona + robot state + memories + knowledge base, with robot tools."""

from __future__ import annotations

import itertools
import json
import re
import threading
import time
from datetime import date, timedelta
from typing import Any, Iterator

from api import llm, memo
from api.interaction import SOUNDS, Interaction
from api.knowledge import KnowledgeBase
from api.memo import MemoStore
from api.memory import RECALL_HINT, ConversationStore, MemoryStore, recall_text
import gestures
from api.motion import (ACTIONS, CHILL_CHAIR, CHILL_LIE, CHILL_SIT, GYMNASTICS, PEEK, PROPUP, SKATE, STAND,
                        WAKE_STRETCH, WALK, WALKING, MotionController, clamp_unit)
from api.protocol import ApiError, CommonState, Reason
from api.store import ROBOT_NAME, Settings, qwen_api_key
from api.workbuddy import WorkBuddy

COMPUTER_WAIT_S = 90

MAX_TOOL_ROUNDS = 4
MODE_NAMES = {"stand": STAND, "walk": WALK, "skate": SKATE,
              "gymnastics": GYMNASTICS, "peek": PEEK, "propup": PROPUP, "liedown": PROPUP,
              "chill_lie": CHILL_LIE, "chill_sit": CHILL_SIT, "chill_chair": CHILL_CHAIR, "wake_stretch": WAKE_STRETCH}

# Web search (llm.web_search setting). "native": the chat model is a Qwen model and searches by
# itself (enable_search); "tool": any other model gets a web_search tool, run on a Qwen provider.
# Asking for it in so many words forces a search, otherwise the model decides.
SEARCH_TRIGGER = re.compile(r"联网|上网|网上|搜一下|搜一搜|搜搜|搜索一下|查一下网|百度一下")
SEARCH_NOTES = {
    "native": "\n[联网搜索] 你可以联网搜索：遇到新闻、天气、价格、比分、最新版本等实时信息或你不确定的事实会自动搜索，"
              "用户说“联网搜一下…”时一定要搜；结合前面的对话理解“他”“这个”指什么。搜索已经在回答前完成，"
              "直接说结果，不要说“正在搜索”。只挑和问题直接相关的重点，一两句话说清，不要把搜到的内容全部罗列出来，"
              "用户想知道更多会接着问。",
    "tool": "\n[联网搜索] 需要实时信息（新闻、天气、价格、比分、最新消息）或你不确定的事实时调用 web_search，"
            "query 写成完整具体的搜索词（结合前面的对话补全“他”“这个”）；平常闲聊不要搜索。"
            "拿到结果后只挑和问题直接相关的重点，一两句话回答。",
}
SEARCHED_NOTE = "\n[联网搜索结果]（用户要求联网搜索，这是刚搜到的内容：只挑和问题直接相关的重点，一两句话回答）\n"
RECALL_GUIDE = ("\n[记忆] 你有长期记忆，还保存着和用户以前所有的聊天记录（文字和语音）。用户问起以前的事、问你记不记得时，"
                "不要直接说不记得，也不要说自己没有记忆：先看下面的记忆和检索结果{tool}，确实找不到再如实说没找到。")
RECALL_TOOL = "，不够就调用 memory_search 搜（可以换几个关键词多搜几次）"
RECALL_VOICE = ("\n你记得和用户以前的聊天：用户问起以前的事（还记得吗、上次说的……）时，系统会自动翻长期记忆和聊天记录给你，"
                "照着说；不要说你没有记忆或者记不住，确实没找到就请用户提醒一下。")
# Earlier dialogue a new conversation starts with (voice <-> text), if this recent.
RECENT_WITHIN_S = 12 * 3600
VOICE_INSTRUCTIONS_MAX = 1800
MEMO_NOTE_SECONDS = 10
MEMO_EXTRACT = (
    "现在是 {now}。用户刚才对机器人说：「{heard}」\n现有未完成的备忘：\n{open}\n"
    "判断用户是不是要新增、完成或取消备忘。只输出一个 JSON 对象，例如 "
    '{{"add":[{{"text":"开会","date":"2026-01-02","time":"15:00"}}],"done":["m1"],"delete":["m2"]}}\n'
    "规则：text 只写事情本身，简短，不带日期、时间和“提醒我”“记一下”；明天、下周三、15号等换算成具体日期；"
    "上午/下午/晚上换算成 24 小时制（下午三点→15:00，晚上八点半→20:30）；只说了时间没说哪天就是今天，"
    "今天这个时间已经过了就是明天；只说“记住…”、没有日期的事 date 和 time 都为 null；没说具体时间 time 为 null；"
    "说某件事做完了、搞定了，把它的编号放进 done；说不用了、取消、删掉，放进 delete；改时间就是 delete 旧的再 add 新的；"
    "一句话里有几件事就分几条；和备忘无关就输出三个空数组。"
)
# Said to the robot, worth asking the model whether it is about moving (motion_heard).
MOTION_HINT = re.compile(
    r"走|前进|后退|退后|过来|转身|转个|转一|左转|右转|转圈|掉头|抬|低头|点头|摇头|扭头|转头|往左看|往右看|看左边|看右边|"
    r"举|挥|招手|招呼|鞠躬|敬礼|跳舞|跳个|鼓掌|拍手|比心|比个|做个动作|做动作|摆个|姿势|伸懒腰|抱抱|拥抱|耸肩|摊手|"
    r"加油|欢呼|秀肌肉|叉腰|握拳|动一动|动一下|活动一下|手臂|胳膊|双手|左手|右手"
)
# Said around a taught move's name when only that move is asked for ("给我做一下螃蟹舞吧").
TAUGHT_FILLER = re.compile(r"[\s,，。.!！?？、~～]|请|你|给我|给大家|帮我|再|来|做|跳|摆|演|表演|展示|一下|一个|一遍|一次|个|下|那个|这个|的|动作|吧|啊|呀|嘛|呗|好吗|好不好")
MOTION_GUIDE = (
    "步骤按顺序执行，有三种：\n"
    '1. 走路 {{"walk":{{"vx":0.5,"vy":0,"vtheta":0,"seconds":2}}}}：vx 前进(正)/后退(负)，vy 向左(正)/向右(负)，'
    "vtheta 左转(正)/右转(负)，都是 -1..1 的比例（vx 1 = 0.5 米/秒，vtheta 1 ≈ 每秒 85°），seconds 0.5–10。"
    "往前走几步 ≈ vx 0.5 走 2 秒；转 90° ≈ vtheta 0.6 走 1.8 秒；转身 180° ≈ vtheta 0.6 走 3.5 秒。\n"
    '2. 现成动作 {{"gesture":"wave"}}，有合适的就优先用：{catalog}。\n'
    '3. 自编动作 {{"keys":[{{"dur":0.4,"pose":{{"tilt":10}},"hold":1.0,"ease":"io"}}]}}：现成动作里没有时才用。'
    "关键帧依次执行，pose 只写要动的关节，其余保持；做完自动回到自然姿态，不用写回去。dur 0.25–3 秒（移到位的用时），"
    "hold 0–3 秒（到位后停留），ease 可选 io 平滑、out 快起、back 冲过头再回、lin 匀速；整段不超过 15 秒、最多 12 帧。"
    "关节（度，左右对称，两边数值意义相同）：rp/lp 右/左肩前举（0 自然下垂，90 向前平举，150 举过头顶，负数向后摆，"
    "别超过 -40）；rr/lr 右/左肩侧抬（0 贴身，90 侧平举，最大 160，不能往里收）；re/le 右/左肘弯曲（0 伸直，最大 85）；"
    "yaw 头左右（正数向左看，-40..40）；tilt 头上下（正数抬头，最多 10；负数低头，最多 -30）。"
    "自然站姿 rp 0、rr 6、re 12、yaw 0、tilt 0。超出范围的会被截掉。"
)
MOTION_PLAN = (
    "用户刚才对机器人说：「{heard}」\n"
    "如果是让机器人自己做身体动作（走、转身、抬头、点头、挥手、打招呼、跳舞、摆姿势等），编排成动作步骤，只输出一个 JSON："
    '{{"steps":[...]}}；不是让它动（聊天、提问、说的是别人或别的东西、已经结束的事）就输出 {{"steps":[]}}。\n'
    + MOTION_GUIDE
    + "\n“抬一下头”这类简短动作：移到位置、停留一两秒就行。不要解释。"
)


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


class ChatService:
    def __init__(
        self,
        settings: Settings,
        motion: MotionController,
        interaction: Interaction,
        memory: MemoryStore,
        conversations: ConversationStore,
        knowledge: KnowledgeBase,
        memos: MemoStore,
        workbuddy: WorkBuddy | None = None,
        demo: bool = False,
    ) -> None:
        self.memos = memos
        self.workbuddy = workbuddy
        self.settings = settings
        self.motion = motion
        self.interaction = interaction
        self.memory = memory
        self.conversations = conversations
        self.knowledge = knowledge
        self.demo = demo
        # MusicService, set by the app once the voice session it hands the speaker over to exists.
        self.music: Any = None
        self._busy: set[str] = set()
        self._busy_lock = threading.Lock()
        self._voice_heard = ""

    # ---- providers -------------------------------------------------------------------------

    def providers(self) -> list[dict[str, Any]]:
        return self.settings.section("llm").get("providers", [])

    def provider(self, provider_id: str | None = None) -> dict[str, Any] | None:
        cfg = self.settings.section("llm")
        if not cfg.get("providers") and not cfg.get("qwen_key_adopted"):
            key, _ = qwen_api_key(self.settings.section("voice"))
            if key:
                # Once only: a provider deleted afterwards stays deleted.
                auto = llm.provider_for_qwen_key(key)
                self.settings.update("llm", {"providers": [auto], "active": auto["id"], "qwen_key_adopted": True})
                cfg = self.settings.section("llm")
        wanted = provider_id or cfg.get("active")
        enabled = [p for p in cfg.get("providers", []) if p.get("enabled", True)]
        for p in enabled:
            if p["id"] == wanted:
                return p
        return enabled[0] if enabled and not provider_id else None

    def search_provider(self) -> dict[str, Any] | None:
        """A provider with built-in web search: a configured one, else one on the voice key."""
        for p in self.providers():
            if p.get("enabled", True) and llm.native_search(p):
                return p
        key, _ = qwen_api_key(self.settings.section("voice"))
        auto = llm.provider_for_qwen_key(key) if key else None
        return auto if llm.native_search(auto) else None

    def search_mode(self, provider: dict[str, Any] | None) -> str | None:
        if provider is None or not self.settings.section("llm").get("web_search", True):
            return None
        if llm.native_search(provider):
            return "native"
        return "tool" if self.search_provider() else None

    # ---- prompt ----------------------------------------------------------------------------

    def robot_state_text(self) -> str:
        action = self.motion.get_action()["current_action"]
        battery = self.motion.battery()
        parts = [f"运动模式：{action['label']}（{action['name']}）"]
        if battery.get("voltage_v"):
            parts.append(f"电量：{battery['percent']}%（{battery['voltage_v']:.2f} V）")
        parts.append(f"当前表情：{self.interaction.current_expression()}")
        return "；".join(parts)

    def recent_dialogue(self, channels: tuple[str, ...], limit_chars: int) -> str:
        """The end of the latest conversation in `channels`, if it is recent."""
        convs = sorted((c for ch in channels for c in self.conversations.list(ch)), key=lambda c: -c["updated"])
        if not convs or time.time() - convs[0]["updated"] > RECENT_WITHIN_S:
            return ""
        lines = [f"{'用户' if m['role'] == 'user' else '你'}：{m['content'][:80]}" for m in self.conversations.history(convs[0]["id"], 3)]
        return "\n".join(lines)[-limit_chars:]

    def memo_enabled(self) -> bool:
        return bool(self.settings.section("memo").get("enabled", True))

    def recall_enabled(self) -> bool:
        return bool(self.settings.section("llm").get("memory_recall", True))

    def now_text(self) -> str:
        now = self.memos.now()
        return f"{now:%Y-%m-%d %H:%M} {memo.weekday(now.date())}"

    def system_prompt(
        self, query: str, use_tools: bool, recent: bool = False, memo_query: tuple[date | None, date | None, str] | None = None,
        recall_before: float | None = None,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        robot = self.settings.section("robot")
        cfg = self.settings.section("llm")
        lines = [robot.get("persona", "").strip(), f"你的名字是 {robot.get('name') or ROBOT_NAME}。"]
        lines.append(f"\n[机器人当前状态] {self.robot_state_text()}；现在时间：{self.now_text()}")
        if use_tools:
            lines.append(
                "\n[身体控制] 你可以调用工具让自己站立、行走、换表情、播放提示音。"
                "行走前机器人必须先站立；只有用户明确要求时才移动身体，动作完成后用一句话说明你做了什么。"
                "需要记住关于用户的长期信息（名字、喜好、约定）时调用 memory_save。"
            )
            if self.memo_enabled():
                lines.append(
                    "\n[备忘录] 用户让你记事、提醒、安排日程时调用 memo_add（一件事一条，日期按现在时间换算成 YYYY-MM-DD，"
                    "时间 24 小时制 HH:MM）；说某件事做完了或不用了，先 memo_list 找到它再 memo_update；"
                    "问某天的安排用 memo_list，要在脸上的屏幕显示清单用 memo_show。"
                )
        memories = self.memory.for_prompt(query)
        if self.recall_enabled():
            lines.append(RECALL_GUIDE.format(tool=RECALL_TOOL if use_tools else ""))
        if memories:
            lines.append("\n[关于用户的记忆]\n" + "\n".join(f"- {m['text']}" for m in memories))
        if self.recall_enabled() and RECALL_HINT.search(query):
            lines.append(recall_text(self.memory, self.conversations, query, recall_before))
        if self.memo_enabled():
            upcoming = self.memos.for_prompt(600)
            if upcoming:
                lines.append("\n[备忘录·近期]（用户让你记住的事和日程，聊到相关的事时可以提醒）\n" + upcoming)
            if memo_query is not None:
                first, last, label = memo_query
                lines.append(f"\n[备忘录·{label}]（用户正在问这些，清单已显示在你脸上的屏幕上：逐条简短地说，没有就说没有安排）\n"
                             + self.memos.days_text(first, last, label))
        voice = self.recent_dialogue(("voice",), 400) if recent else ""
        if voice:
            lines.append("\n[刚才的语音对话]（用户可能接着问；没提到就不用管）\n" + voice)
        hits = self.knowledge.search(query, int(cfg.get("kb_top_k", 4))) if query.strip() else []
        if hits:
            lines.append("\n[知识库检索结果]（回答涉及这些内容时，在句末用 [编号] 标注来源；与问题无关就忽略）")
            for i, hit in enumerate(hits, 1):
                lines.append(f"[{i}] 《{hit['title']}》{hit['text'][:600]}")
        return "\n".join(line for line in lines if line), memories, hits

    def voice_instructions(self) -> str:
        robot = self.settings.section("robot")
        text = f"{robot.get('persona', '').strip()}\n你的名字是 {robot.get('name') or ROBOT_NAME}。这是语音对话，回答要口语化、简短，不要使用列表或表情符号。"
        memories = self.memory.for_prompt("", limit_chars=800)
        if self._computer_enabled("voice_trigger"):
            text += "\n用户让你操作电脑时（比如说“让电脑…”），电脑上的助手会自动去做，你只需简短回答“好的，交给电脑了”，不要说你做不到。"
        if self.recall_enabled():
            text += RECALL_VOICE
        if memories:
            text += "\n关于用户的记忆：" + "；".join(m["text"] for m in memories)
        if self.music_enabled("voice_control"):
            text += ("\n用户让你放歌、听歌、唱首歌时，系统会自动从网易云搜歌并用你的喇叭播放（播放时语音对话暂停，放完自动回来）："
                     "简短回应一句（比如“好嘞，这就给你放《晴天》”），不要说你不能放歌，也不要自己唱。")
        if self.voice_motion_enabled():
            text += ("\n用户让你走路、转身、抬头、点头、挥手、打招呼、跳舞或者摆个动作时，你的身体会自动去做："
                     "简短回应一句（比如“好嘞，看我的”），不要说你动不了。")
            taught = self.motion.custom.names()
            if taught:
                text += "主人还教过你这些动作，说到名字你也会做：" + "、".join(taught[:20]) + "。"
        if self.memo_enabled():
            now = self.memos.now()
            text += (
                f"\n今天是{now.month}月{now.day}日{memo.weekday(now.date())}。用户让你记事、提醒、安排日程，或者说某件事做完了、"
                "不用了的时候，系统会自动记进备忘录：你简短确认并复述时间和事情就行（比如“好，明天下午三点开会，记下了”），"
                "不要说你记不住或者设不了提醒。"
            )
            upcoming = self.memos.for_prompt(300)
            if upcoming:
                text += "\n备忘录（近期）：" + "；".join(upcoming.splitlines())
        recent = self.recent_dialogue(("voice", "chat"), 360)
        if recent:
            # Last, so that the length cap cuts it first.
            text += "\n上次聊到这里（用户接着问就接上，没提就别主动重复）：\n" + recent
        return text[:VOICE_INSTRUCTIONS_MAX]

    def voice_turn(self, role: str, text: str, interrupted: bool) -> None:
        """A finished voice turn: memos asked about go on the screen, memos told get noted down; with
        auto_memory on, long-term facts are taken from it like from the chat."""
        if role == "user":
            self._voice_heard = text
            if self.music is not None and self.music.heard(text, "voice"):
                return
            if not self.memo_heard(text, "voice"):
                self.motion_heard(text, "voice")
            return
        heard, self._voice_heard = self._voice_heard, ""
        if not heard or interrupted or not self.settings.section("llm").get("auto_memory"):
            return
        provider = self.provider()
        if provider is not None:
            threading.Thread(target=self._extract_memories, args=(provider, heard, text), daemon=True).start()

    def music_enabled(self, key: str = "enabled") -> bool:
        cfg = self.settings.section("music")
        return self.music is not None and bool(cfg.get("enabled", True)) and bool(cfg.get(key, True))

    def _computer_enabled(self, key: str) -> bool:
        cfg = self.settings.section("workbuddy")
        return self.workbuddy is not None and bool(cfg.get("enabled")) and bool(cfg.get(key, True))

    # ---- memos -----------------------------------------------------------------------------

    def show_memos(self, first: date | None, last: date | None, label: str, source: str = "interaction",
                   title: str = "", title_color: str = "", items: list[dict[str, Any]] | None = None,
                   seconds: float | None = None) -> bool:
        """A day's memos (or `items`) on the face screen."""
        cfg = self.settings.section("memo")
        if not cfg.get("show_on_screen", True):
            return False
        today = self.memos.today()
        if items is None:
            items = memo.in_days(self.memos.all(), first, last, today)
        spec = memo.board_spec(items, first, last, label, today, title, seconds or float(cfg.get("board_seconds", 30)))
        if title_color:
            spec["title_color"] = title_color
        return self.interaction.show_board(spec, source)

    def show_range(self, start: date | None, end: date | None, source: str = "interaction") -> bool:
        """The memos of start..end (default today) on the face screen."""
        today = self.memos.today()
        first = start or today
        last = max(first, end or first)
        label = memo.day_name(first, today) if first == last else f"{memo.short_date(first)}-{memo.short_date(last)}"
        return self.show_memos(first, last, label, source)

    def show_clock(self, source: str = "interaction") -> bool:
        cfg = self.settings.section("memo")
        if not cfg.get("show_on_screen", True):
            return False
        now = self.memos.now()
        today = now.date()
        upcoming = [m for m in memo.in_days(self.memos.all(), today, today, today)
                    if not m.get("done") and (m.get("time") or "99:99") >= f"{now:%H:%M}"]
        spec = {"kind": "clock", "tz": cfg.get("timezone", ""), "seconds": 15, "items": memo.board_rows(upcoming, today, False)}
        return self.interaction.show_board(spec, source)

    def memo_heard(self, text: str, origin: str) -> bool:
        """Something the user said: the memos of the day asked about (or the time) go on the
        screen; memos told, finished or cancelled are noted down by the model in the background.
        True when it was about memos (or the board)."""
        if not self.memo_enabled() or not text.strip():
            return False
        if self.interaction.board_up() and memo.wants_dismiss(text):
            self.interaction.show_board({"kind": "clear"}, origin)
            return True
        query = memo.parse_query(text, self.memos.today())
        if query is not None:
            self.show_memos(*query, source=origin)
        elif memo.asks_clock(text):
            self.show_clock(origin)
        elif memo.wants_capture(text, self.memos.today()) and self.settings.section("memo").get("voice_capture", True):
            provider = self.provider()
            if provider is not None:
                threading.Thread(target=self._capture_memos, args=(provider, text, origin), daemon=True).start()
        else:
            return False
        return True

    # ---- moving on request -------------------------------------------------------------------

    def voice_motion_enabled(self) -> bool:
        cfg = self.settings.section("llm")
        return bool(cfg.get("allow_motion_tools", True) and cfg.get("voice_motion", True))

    def motion_heard(self, text: str, origin: str) -> None:
        """Something said that may ask the robot to move: a taught move named on its own is done
        right away; anything else the model turns into walks, gestures, taught moves and made-up
        keyframes (MOTION_PLAN) in the background, and the robot does them."""
        if not self.voice_motion_enabled():
            return
        taught = self.motion.custom.heard(text)
        if taught is not None and len(TAUGHT_FILLER.sub("", text.replace(taught["name"], ""))) <= 2:
            try:
                self.motion.play_custom(taught["id"], source="interaction")
                self.interaction.events.alert("info", f"语音动作：{taught['name']}")
            except ApiError as exc:
                self.interaction.events.alert("warning", f"想做「{taught['name']}」，但{exc.message}")
            return
        if taught is None and not MOTION_HINT.search(text):
            return
        provider = self.provider()
        if provider is not None:
            threading.Thread(target=self._plan_motion, args=(provider, text, origin), daemon=True).start()

    def _plan_motion(self, provider: dict[str, Any], heard: str, origin: str) -> None:
        prompt = MOTION_PLAN.format(heard=heard, catalog=self._gesture_list())
        try:
            reply = llm.complete(provider, [{"role": "user", "content": prompt}], max_tokens=800, temperature=0.3)
            found = re.search(r"\{.*\}", reply, re.S)
            plan = json.loads(found.group()) if found else {}
        except (llm.LlmError, ValueError, OSError) as exc:
            self.interaction.events.alert("warning", f"动作没做成：{exc}")
            return
        steps = plan.get("steps") if isinstance(plan, dict) else None
        if not steps:
            return
        try:
            self.motion.run_sequence(steps, source="interaction")
        except ApiError as exc:
            self.interaction.events.alert("warning", f"想动一下，但{exc.message}")
            return
        self.interaction.events.alert("info", "语音动作：" + " → ".join(self._describe_step(s) for s in steps if isinstance(s, dict)))

    def _gesture_list(self) -> str:
        items = [f"{g['name']}={g['label']}" for g in gestures.catalog()]
        items += [f"{m['id']}={m['name']}（主人教的）" for m in self.motion.custom.list()]
        return "、".join(items)

    def _describe_step(self, step: dict[str, Any]) -> str:
        if isinstance(step.get("walk"), dict):
            w = step["walk"]
            turn, move = clamp_unit(w.get("vtheta")), clamp_unit(w.get("vx"))
            kind = "后退" if move < -0.05 else "前进" if move > 0.05 else "左转" if turn > 0.05 else "右转" if turn < -0.05 else "走"
            return f"{kind} {w.get('seconds', '')} 秒"
        name = step.get("gesture")
        if name in gestures.GESTURES:
            return gestures.GESTURES[name].label
        taught = self.motion.custom.find(str(name)) if isinstance(name, str) else None
        if taught is not None:
            return taught["name"]
        return "随机动作" if name == "random" else "自编动作"

    def _capture_memos(self, provider: dict[str, Any], heard: str, origin: str) -> None:
        today = self.memos.today()
        since = (today - timedelta(days=memo.OVERDUE_DAYS)).isoformat()
        open_items = [m for m in self.memos.list(status="open") if not m.get("date") or m["date"] >= since][:40]
        aliases = {f"m{i}": m for i, m in enumerate(open_items, 1)}
        listing = "\n".join(f"{alias}: {memo.describe(m, today)}" for alias, m in aliases.items()) or "（无）"
        prompt = MEMO_EXTRACT.format(now=self.now_text(), heard=heard, open=listing)
        try:
            reply = llm.complete(provider, [{"role": "user", "content": prompt}], max_tokens=300, temperature=0)
            found = re.search(r"\{.*\}", reply, re.S)
            plan = json.loads(found.group()) if found else {}
        except (llm.LlmError, ValueError, OSError) as exc:
            self.interaction.events.alert("warning", f"备忘录没记上：{exc}")
            return
        if not isinstance(plan, dict):
            return
        added, finished, removed = [], [], []
        for item in (plan.get("add") or [])[:5]:
            if isinstance(item, dict):
                try:
                    added.append(self.memos.add(str(item.get("text") or ""), item.get("date"), item.get("time"), source=origin))
                except ApiError:
                    pass
        for key, bucket in (("done", finished), ("delete", removed)):
            for alias in (plan.get(key) or [])[:10]:
                target = aliases.get(str(alias))
                if target is None:
                    continue
                try:
                    bucket.append(self.memos.update(target["id"], {"done": True}) if key == "done" else self.memos.delete(target["id"]))
                except ApiError:
                    pass
        if not (added or finished or removed):
            return
        title, color = ("已记下", "green") if added else ("完成啦", "green") if finished else ("已删除", "pink")
        shown = added + finished + [{**m, "done": True} for m in removed]
        self.show_memos(None, None, title, origin, title=title, title_color=color, items=shown, seconds=MEMO_NOTE_SECONDS)
        parts = [f"记下 {memo.describe(m, today)}" for m in added] + [f"完成 {m['text']}" for m in finished]
        parts += [f"删除 {m['text']}" for m in removed]
        self.interaction.events.alert("info", "备忘录：" + "；".join(parts))

    # ---- tools -----------------------------------------------------------------------------

    def tools(self, web_search: bool = False) -> list[dict[str, Any]]:
        cfg = self.settings.section("llm")
        tools = [
            _fn("robot_get_status", "读取机器人当前的运动模式、电量和表情。", {}),
            _fn("memory_save", "把关于用户的长期有用信息记下来（名字、喜好、约定等），一次一条。",
                {"text": {"type": "string", "description": "要记住的一句话"}}, ["text"]),
            _fn("knowledge_search", "在本地知识库里检索资料。", {"query": {"type": "string"}}, ["query"]),
        ]
        if self.recall_enabled():
            tools.append(_fn("memory_search", "在长期记忆和以前的聊天记录（文字和语音）里搜索：用户问起以前说过、聊过的事时用。",
                             {"query": {"type": "string", "description": "要找的事的关键词，比如“生日”“喜欢的动物”"}}, ["query"]))
        if self.memo_enabled():
            day = {"type": "string", "description": "YYYY-MM-DD"}
            tools += [
                _fn("memo_add", "记一条备忘：待办、约会、要提醒的事，或者用户让你记住的事（没有日期就不填 date）。一件事一条。",
                    {"text": {"type": "string", "description": "事情本身，简短，不带日期时间"}, "date": day,
                     "time": {"type": "string", "description": "HH:MM，24 小时制，没说具体时间就不填"}}, ["text"]),
                _fn("memo_list", "列出备忘：start..end 之间的（都不填就是近期未完成的和随手记），返回 id。",
                    {"start": day, "end": day}),
                _fn("memo_update", "修改一条备忘：标记完成（done）、改内容/日期/时间，或 delete=true 删除。",
                    {"id": {"type": "string"}, "done": {"type": "boolean"}, "delete": {"type": "boolean"},
                     "text": {"type": "string"}, "date": day, "time": {"type": "string"}}, ["id"]),
                _fn("memo_show", "在脸上的点阵屏显示 start..end 的备忘清单（不填就是今天）；clock=true 显示时钟。",
                    {"start": day, "end": day, "clock": {"type": "boolean"}}),
            ]
        if web_search:
            tools.append(_fn("web_search", "联网搜索实时信息（新闻、天气、价格、比分、最新消息等），返回搜到的要点。",
                             {"query": {"type": "string", "description": "完整具体的搜索词"}}, ["query"]))
        if cfg.get("allow_motion_tools", True):
            tools += [
                _fn("robot_set_mode", "切换运动模式：stand 站立保持，walk 进入行走模式（之后用 robot_walk 移动），"
                    "skate 穿滑板鞋时的滑行模式（只能向前和转弯），"
                    "gymnastics 做一套约 46 秒的广播体操（做完自动回到行走模式原地站着），"
                    "peek 做一段约 15 秒鬼鬼祟祟的偷看：身体斜出去探头张望再缩回（做完同样回到行走模式），"
                    "propup 仰面躺着时用手肘和手撑起上身并保持，liedown 从撑起躺回去，"
                    "chill_lie 躺着时循环做放松的小动作（翘腿、晃脚、看看四周），chill_sit 撑起坐着时晃脚张望，"
                    "chill_chair 坐在凳子上时双手撑凳、晃腿张望（要先把机器人放到凳子上），"
                    "wake_stretch 仰面躺着时做一次约 9 秒的起床伸懒腰（举手伸腿、全身发抖再放松，做完回到站立模式）。",
                    {"mode": {"type": "string", "enum": list(MODE_NAMES)}}, ["mode"]),
                _fn("robot_walk", "以给定速度走一段时间后停下。速度是 -1..1 的比例，正 vx 向前，正 vy 向左，正 vtheta 逆时针转。",
                    {
                        "vx": {"type": "number", "minimum": -1, "maximum": 1},
                        "vy": {"type": "number", "minimum": -1, "maximum": 1},
                        "vtheta": {"type": "number", "minimum": -1, "maximum": 1},
                        "duration_s": {"type": "number", "minimum": 0.5, "maximum": 10},
                    }, ["duration_s"]),
                _fn("robot_stop", "立即停止行走，原地站立。", {}),
                _fn("robot_gesture",
                    "用双臂和头做一个拟人动作（站着或走着都行，会自动切到「走跑·上肢自由」，做完回到原来的模式）。适合打招呼、"
                    "表达情绪、回应夸奖等。可选：" + self._gesture_list() + "，random=随机。",
                    {"name": {"type": "string", "enum": [*gestures.GESTURE_NAMES, "random",
                                                         *(m["id"] for m in self.motion.custom.list())]}}, ["name"]),
                _fn("robot_motion",
                    "按顺序做一串身体动作：走路、现成动作，或者现成动作里没有时自己编的关节动作（抬头、举左手、摆姿势等）。"
                    "做动作时自动切到「走跑·上肢自由」，做完回到原来的模式。" + MOTION_GUIDE.format(catalog=self._gesture_list()),
                    {"steps": {"type": "array", "items": {"type": "object"}, "description": "动作步骤，按顺序执行"}}, ["steps"]),
            ]
        if cfg.get("allow_expression_tools", True):
            tools += [
                _fn("robot_set_expression", "切换脸部点阵屏表情。",
                    {"name": {"type": "string", "enum": self.interaction.expression_names()}}, ["name"]),
                _fn("robot_play_sound", "用扬声器播放一段提示音。", {"name": {"type": "string", "enum": list(SOUNDS)}}, ["name"]),
            ]
        if self.music_enabled():
            tools += [
                _fn("music_play", "用机器人的喇叭放音乐（网易云）：query 是歌名/歌手（比如“晴天 周杰伦”）；不指定歌时用 playlist："
                    "hot 热歌榜，daily 每日推荐，liked 用户的「我喜欢的音乐」（后两个要登录网易云）。返回正在放的歌。",
                    {"query": {"type": "string"}, "playlist": {"type": "string", "enum": ["hot", "daily", "liked"]}}),
                _fn("music_control", "控制正在放的音乐。",
                    {"action": {"type": "string", "enum": ["stop", "pause", "resume", "next", "previous", "louder", "quieter"]}},
                    ["action"]),
            ]
        if self._computer_enabled("chat_tool"):
            tools.append(_fn(
                "computer_task",
                "让用户电脑上的 WorkBuddy 智能体去操作电脑完成任务（打开软件、整理文件、查资料、写文档、发消息等）。"
                "把用户的要求整理成一条完整、具体的中文指令。返回执行结果；耗时长的任务会在后台继续，完成后提醒用户。",
                {"instruction": {"type": "string", "description": "交给电脑的完整指令"}}, ["instruction"]))
        return tools

    def run_tool(self, name: str, args: dict[str, Any], recall_before: float | None = None) -> dict[str, Any]:
        motion_ok = self.settings.section("llm").get("allow_motion_tools", True)
        if name == "robot_get_status":
            return {"state": self.robot_state_text()}
        if name == "memory_save":
            return {"saved": self.memory.add(str(args.get("text", "")), source="chat")["text"]}
        if name == "memory_search" and self.recall_enabled():
            query = str(args.get("query", ""))
            found = {
                "memories": [m["text"] for m in self.memory.search(query)],
                "conversations": [
                    {"time": time.strftime("%Y-%m-%d %H:%M", time.localtime(h["ts"])), "channel": "语音" if h["channel"] == "voice" else "文字",
                     "messages": [f"{'用户' if m['role'] == 'user' else '你'}：{m['content']}" for m in h["messages"]]}
                    for h in self.conversations.search(query, limit=5, before=recall_before)
                ],
            }
            if not found["memories"] and not found["conversations"]:
                found["message"] = "没有找到，可以换个关键词再搜"
            return found
        if name == "knowledge_search":
            return {"results": [{"title": h["title"], "text": h["text"][:500]} for h in self.knowledge.search(str(args.get("query", "")), 3)]}
        if name == "web_search":
            return self.web_search(str(args.get("query", "")))
        if name.startswith("memo_") and self.memo_enabled():
            return self._memo_tool(name, args)
        if name == "robot_set_mode" and motion_ok:
            mode = MODE_NAMES.get(str(args.get("mode")))
            if mode is None:
                raise ApiError(f"mode 只能是 {' / '.join(MODE_NAMES)}", CommonState.INVALID)
            if mode == PROPUP:
                result = self.motion.set_propup(args.get("mode") == "liedown", source="interaction")
            else:
                result = self.motion.set_action(mode, source="interaction")
            return {"message": result["message"] or "已开始切换", "target": ACTIONS[mode].label}
        if name == "robot_walk" and motion_ok:
            result = self.motion.walk_for(
                float(args.get("vx", 0) or 0), float(args.get("vy", 0) or 0), float(args.get("vtheta", 0) or 0),
                float(args.get("duration_s", 2) or 2), source="interaction",
            )
            return {"message": result["message"] or "开始行走", "task_id": result["task_id"]}
        if name == "robot_gesture" and motion_ok:
            gesture = str(args.get("name", ""))
            if self.motion.custom.find(gesture) is not None:
                result = self.motion.play_custom(gesture, source="interaction")
            else:
                result = self.motion.run_sequence([{"gesture": gesture}], source="interaction")
            return {"message": result["message"], "task_id": result["task_id"]}
        if name == "robot_motion" and motion_ok:
            result = self.motion.run_sequence(args.get("steps"), source="interaction")
            return {"message": result["message"], "task_id": result["task_id"],
                    "steps": [self._describe_step(s) for s in args.get("steps") or [] if isinstance(s, dict)]}
        if name == "robot_stop" and motion_ok:
            self.motion.stop_gesture()
            if self.motion.actual() in WALKING:
                self.motion.velocity("interaction", 0, 0, 0)
                self.motion.set_action(STAND, source="interaction")
            return {"message": "已停止"}
        if name == "robot_set_expression":
            return {"expression": self.interaction.set_expression(str(args.get("name")), source="interaction")["data"]["label"]}
        if name == "robot_play_sound":
            self.interaction.play_sound(str(args.get("name")), None, 60)
            return {"played": args.get("name")}
        if name == "music_play" and self.music_enabled():
            query = str(args.get("query") or "").strip()
            result = self.music.play({"query": query} if query else {"playlist": str(args.get("playlist") or "hot")}, "interaction")
            track = result["data"]["track"] or {}
            return {"message": result["message"], "song": track.get("name"), "artist": track.get("artist"),
                    "notice": result["data"].get("notice")}
        if name == "music_control" and self.music_enabled():
            result = self.music.do(str(args.get("action", "")))
            return {"message": result["message"]}
        if name == "computer_task" and self.workbuddy is not None and self._computer_enabled("chat_tool"):
            robot = self.settings.section("robot").get("name") or ROBOT_NAME
            task = self.workbuddy.submit(str(args.get("instruction", "")), "chat", robot)
            task = self.workbuddy.wait(task["id"], COMPUTER_WAIT_S)
            if task["status"] in ("queued", "running"):
                return {"status": "running", "message": "电脑还在执行，完成后会弹出提醒", "partial": task["result"][-500:]}
            return {"status": task["status"], "result": task["result"][-1500:], "error": task["error"]}
        raise ApiError(f"工具 {name} 不可用", CommonState.INVALID, Reason.NOT_FOUND)

    def _memo_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        today = self.memos.today()

        def day(key: str) -> date | None:
            value = str(args.get(key) or "").strip()
            try:
                return date.fromisoformat(value) if value else None
            except ValueError:
                raise ApiError(f"{key} 要写成 YYYY-MM-DD", CommonState.INVALID) from None

        def brief(m: dict[str, Any]) -> dict[str, Any]:
            return {"id": m["id"], "text": m["text"], "date": m.get("date"), "time": m.get("time"), "done": m.get("done", False)}

        if name == "memo_add":
            m = self.memos.add(str(args.get("text", "")), args.get("date"), args.get("time"), source="chat")
            return {"saved": memo.describe(m, today)}
        if name == "memo_list":
            first, last = day("start"), day("end")
            if first and not last:
                last = first
            items = memo.in_days(self.memos.all(), first, last or first, today)
            return {"today": self.now_text(), "memos": [brief(m) for m in items[:40]]}
        if name == "memo_update":
            memo_id = str(args.get("id", ""))
            if args.get("delete"):
                return {"deleted": self.memos.delete(memo_id)["text"]}
            values = {k: args[k] for k in ("text", "date", "time", "done") if k in args}
            return {"updated": brief(self.memos.update(memo_id, values))}
        if name == "memo_show":
            if args.get("clock"):
                return {"shown": self.show_clock("interaction")}
            return {"shown": self.show_range(day("start"), day("end"), "interaction")}
        raise ApiError(f"工具 {name} 不可用", CommonState.INVALID, Reason.NOT_FOUND)

    def web_search(self, query: str, context: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        provider = self.search_provider()
        if provider is None or not query.strip():
            raise ApiError("联网搜索不可用（需要千问的 Key）", CommonState.FAILURE)
        try:
            text = llm.web_search(provider, query.strip(), context)
        except llm.LlmError as exc:
            raise ApiError(f"搜索失败：{exc}", CommonState.FAILURE) from None
        return {"message": text or "没有搜到相关内容"}

    # ---- chat ------------------------------------------------------------------------------

    def stream(self, conversation_id: str | None, message: str, provider_id: str | None = None) -> Iterator[dict[str, Any]]:
        message = (message or "").strip()
        if not message:
            raise ApiError("消息不能为空", CommonState.INVALID)
        provider = self.provider(provider_id)
        if provider is None and not self.demo:
            raise ApiError("还没有配置大模型，请先到 设置 → 大模型 添加一个", CommonState.FAILURE, Reason.NOT_CONFIGURED)
        if not conversation_id or not self.conversations.exists(conversation_id):
            conversation_id = self.conversations.create("chat")["id"]
        with self._busy_lock:
            if conversation_id in self._busy:
                raise ApiError("这个会话还在回复中", CommonState.FAILURE, Reason.INPUT_SOURCE_BUSY, 409)
            self._busy.add(conversation_id)
        try:
            yield from self._stream(conversation_id, message, provider)
        finally:
            with self._busy_lock:
                self._busy.discard(conversation_id)

    def _stream(self, conversation_id: str, message: str, provider: dict[str, Any] | None) -> Iterator[dict[str, Any]]:
        cfg = self.settings.section("llm")
        history = self.conversations.history(conversation_id, int(cfg.get("history_turns", 8)))
        recall_before = self.conversations.window_start(conversation_id, int(cfg.get("history_turns", 8)))
        self.conversations.append(conversation_id, "user", message)
        search = self.search_mode(provider)
        asked_search = bool(SEARCH_TRIGGER.search(message))
        tools = self.tools(web_search=search == "tool")
        memo_query = memo.parse_query(message, self.memos.today()) if self.memo_enabled() else None
        if memo_query is not None:
            self.show_memos(*memo_query, source="chat")
        elif self.memo_enabled() and memo.asks_clock(message):
            self.show_clock("chat")
        system, memories, hits = self.system_prompt(message, use_tools=bool(tools), recent=not history, memo_query=memo_query,
                                                    recall_before=recall_before)
        yield {
            "type": "meta",
            "conversation_id": conversation_id,
            "provider": (provider or {}).get("name", "离线演示"),
            "model": (provider or {}).get("model", "demo"),
            "memories": [m["text"] for m in memories],
            "sources": [{"title": h["title"], "text": h["text"][:240], "doc_id": h["doc_id"]} for h in hits],
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system + SEARCH_NOTES.get(search or "", "")}, *history, {"role": "user", "content": message},
        ]
        reply, reasoning, actions, usage = "", "", [], None
        started = time.monotonic()
        try:
            if search == "tool" and asked_search:
                yield from self._search_first(message, history, messages, actions)
            for round_index in range(MAX_TOOL_ROUNDS + 1):
                round_text = ""
                done: dict[str, Any] = {"tool_calls": []}
                forced = search == "native" and asked_search and round_index == 0
                if provider is None:
                    events = self._demo_round(message, messages)
                else:
                    # A forced search answers from its results: with tools offered the model may
                    # call one first and the later rounds no longer search.
                    events = self._round(provider, messages, [] if forced else tools,
                                         "force" if forced else "auto" if search == "native" else None, cfg)
                    try:
                        events = itertools.chain([next(events)], events)
                    except StopIteration:
                        events = iter(())
                    except llm.SearchUnsupported:
                        # Rejected before replying: the web_search tool from now on, and this round again.
                        search = "tool" if self.search_provider() else None
                        tools = self.tools(web_search=search == "tool") if tools else []
                        messages[0]["content"] = system + SEARCH_NOTES.get(search or "", "")
                        if search == "tool" and asked_search and round_index == 0:
                            yield from self._search_first(message, history, messages, actions)
                        forced = False
                        events = self._round(provider, messages, tools, None, cfg)
                if forced:
                    yield {"type": "tool_call", "id": "web_search_native", "name": "web_search", "arguments": {"query": message}}
                for kind, value in events:
                    if kind == "delta":
                        round_text += value
                        yield {"type": "delta", "text": value}
                    elif kind == "reasoning":
                        reasoning += value
                        yield {"type": "reasoning", "text": value}
                    elif kind == "notice":
                        tools = []
                        yield {"type": "notice", "text": value}
                    elif kind == "done":
                        done = value
                        usage = value.get("usage") or usage
                if forced:
                    result = {"message": "千问内置联网搜索"}
                    actions.append({"name": "web_search", "arguments": {"query": message}, "ok": True, "result": result})
                    yield {"type": "tool_result", "id": "web_search_native", "name": "web_search", "ok": True, "result": result}
                reply += round_text
                if not done["tool_calls"]:
                    break
                messages.append({
                    "role": "assistant",
                    "content": round_text or None,
                    "tool_calls": [
                        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                        for c in done["tool_calls"]
                    ],
                })
                for call in done["tool_calls"]:
                    try:
                        args = json.loads(call["arguments"] or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("参数不是对象")
                    except ValueError as exc:
                        args, result, ok = {}, {"error": f"参数解析失败：{exc}"}, False
                    else:
                        yield {"type": "tool_call", "id": call["id"], "name": call["name"], "arguments": args}
                        try:
                            result, ok = self.run_tool(call["name"], args, recall_before), True
                        except ApiError as exc:
                            result, ok = {"error": exc.message}, False
                    actions.append({"name": call["name"], "arguments": args, "ok": ok, "result": result})
                    yield {"type": "tool_result", "id": call["id"], "name": call["name"], "ok": ok, "result": result}
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
        except llm.LlmError as exc:
            yield {"type": "error", "message": f"大模型调用失败：{exc}"}
            if reply:
                self.conversations.append(conversation_id, "assistant", reply, error=str(exc), actions=actions)
            return
        except Exception as exc:  # noqa: BLE001 - network reads can fail in many ways mid-stream
            yield {"type": "error", "message": f"回复中断：{exc}"}
            if reply:
                self.conversations.append(conversation_id, "assistant", reply, error=str(exc), actions=actions)
            return
        meta = {
            "model": (provider or {}).get("model", "demo"),
            "elapsed_ms": round(1000 * (time.monotonic() - started)),
            "actions": actions,
            "sources": [h["title"] for h in hits],
        }
        if reasoning:
            meta["reasoning"] = reasoning
        if usage:
            meta["usage"] = usage
        saved = self.conversations.append(conversation_id, "assistant", reply, **meta)
        yield {"type": "done", "conversation_id": conversation_id, "message": saved}
        if cfg.get("auto_memory") and provider is not None:
            threading.Thread(target=self._extract_memories, args=(provider, message, reply), daemon=True).start()

    def _round(
        self, provider: dict[str, Any], messages: list[dict[str, Any]], tools: list[dict[str, Any]], search: str | None, cfg: dict[str, Any],
    ) -> Iterator[tuple[str, Any]]:
        return llm.stream_chat(
            provider,
            messages,
            tools=tools or None,
            temperature=float(cfg.get("temperature", 0.7)),
            max_tokens=int(cfg.get("max_tokens", 1024)),
            search=search,
        )

    def _search_first(
        self, message: str, history: list[dict[str, Any]], messages: list[dict[str, Any]], actions: list[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """Asked to search, with a model that cannot itself: search before it answers, results into its prompt."""
        args = {"query": message}
        yield {"type": "tool_call", "id": "web_search_first", "name": "web_search", "arguments": args}
        try:
            result, ok = self.web_search(message, history), True
        except ApiError as exc:
            result, ok = {"error": exc.message}, False
        actions.append({"name": "web_search", "arguments": args, "ok": ok, "result": result})
        yield {"type": "tool_result", "id": "web_search_first", "name": "web_search", "ok": ok, "result": result}
        if ok:
            messages[0]["content"] += SEARCHED_NOTE + result["message"]

    def _extract_memories(self, provider: dict[str, Any], user: str, reply: str) -> None:
        prompt = (
            "从下面这轮对话中提取值得长期记住的、关于用户本人的事实（名字、喜好、习惯、约定、重要日期）。"
            "只输出 JSON 字符串数组，每条不超过 30 字；没有就输出 []。\n"
            f"用户：{user}\n机器人：{reply}"
        )
        try:
            text = llm.complete(provider, [{"role": "user", "content": prompt}], max_tokens=200, temperature=0)
            match = re.search(r"\[.*\]", text, re.S)
            facts = json.loads(match.group()) if match else []
        except (llm.LlmError, ValueError, OSError):
            return
        for fact in facts[:3]:
            if isinstance(fact, str) and fact.strip():
                self.memory.add(fact.strip(), source="auto")

    def _demo_round(self, message: str, messages: list[dict[str, Any]]) -> Iterator[tuple[str, Any]]:
        """Offline stand-in for an LLM so the mock server can be tried without any API key."""
        if messages[-1]["role"] == "tool":
            text = "好的，已经照做了。（离线演示模式：配置真实大模型后我就能正常聊天）"
        else:
            calls = []
            if "站" in message:
                calls.append(("robot_set_mode", {"mode": "stand"}))
            if "走" in message:
                calls.append(("robot_walk", {"vx": 0.5, "vy": 0, "vtheta": 0, "duration_s": 3}))
            if "转" in message:
                calls.append(("robot_walk", {"vx": 0, "vy": 0, "vtheta": 0.6, "duration_s": 3}))
            if "停" in message:
                calls.append(("robot_stop", {}))
            for word, gesture in (("挥手", "wave"), ("打招呼", "wave"), ("比心", "heart"), ("鼓掌", "clap"), ("跳舞", "dance")):
                if word in message:
                    calls.append(("robot_gesture", {"name": gesture}))
                    break
            for word, expr in (("开心", "happy"), ("笑", "laugh"), ("难过", "sad"), ("生气", "angry"), ("睡", "sleeping")):
                if word in message:
                    calls.append(("robot_set_expression", {"name": expr}))
                    break
            if calls:
                yield ("done", {
                    "tool_calls": [{"id": f"demo_{i}", "name": n, "arguments": json.dumps(a)} for i, (n, a) in enumerate(calls)],
                    "finish_reason": "tool_calls",
                    "usage": None,
                })
                return
            kb = "[1]" if "知识库检索结果" in messages[0]["content"] else ""
            text = (
                f"（离线演示）你说的是「{message[:40]}」。我现在没有连接大模型，"
                f"可以试试“站起来”“往前走”“开心一点”来控制我。{kb}"
            )
        for i in range(0, len(text), 3):
            time.sleep(0.03)
            yield ("delta", text[i : i + 3])
        yield ("done", {"tool_calls": [], "finish_reason": "stop", "usage": None})
