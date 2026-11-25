# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Stand-in for voice_chat.py in mock mode: prints the same console lines it does.

A {"prompt": ...} line on stdin (a question a PC task asks) is "said", and a made-up user
answers it a few seconds later: the first option offered, or a plain yes."""

import json
import re
import signal
import sys
import threading
import time

TURNS = [
    ("让电脑帮我打开记事本，写一句你好。", "好的，交给电脑了。"),
    ("你好，你是谁？", "你好呀，我是小小白，一个会走路的桌面小机器人。"),
    ("今天天气怎么样？", "我现在在模拟模式，看不到外面，不过希望是个好天气。"),
    ("给我跳个舞吧", "好呀，不过要先让我站起来，再切到走路模式。"),
]
lock = threading.Lock()
asking = threading.Event()  # a question is being asked and answered: no made-up turns meanwhile


def say(reply: str) -> None:
    with lock:
        print("机器人: ", end="", flush=True)
        for ch in reply:
            print(ch, end="", flush=True)
            time.sleep(0.03)
        print(flush=True)


def hear(text: str) -> None:
    with lock:
        print(f"你: {text}", flush=True)


def commands() -> None:
    for line in sys.stdin:
        try:
            text = json.loads(line).get("prompt") or ""
        except (json.JSONDecodeError, AttributeError):
            continue
        asking.set()
        asked = re.search(r"(?:问题|它说)：(.*)", text, re.S)
        question = (asked.group(1) if asked else text).strip()
        say(question.replace("\n", "，"))
        options = re.search(r"选项：([^（\n]*)", text)
        time.sleep(3)
        hear(options.group(1).split("；")[0] if options else "好的，发吧")
        time.sleep(1)
        say("好的，我告诉电脑。")
        asking.clear()


def main() -> None:
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    print("[ready] 模拟语音会话（未连接真机，麦克风和扬声器不工作）", flush=True)
    threading.Thread(target=commands, daemon=True).start()
    index = 0
    while True:
        time.sleep(9)
        if asking.is_set():
            continue
        user, reply = TURNS[index % len(TURNS)]
        index += 1
        hear(user)
        time.sleep(1.2)
        say(reply)


if __name__ == "__main__":
    main()
