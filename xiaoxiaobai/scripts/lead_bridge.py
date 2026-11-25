#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""PC-side bridge: stream microduck lead joints to xiaoxiaobai follow joints."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSOLE = ROOT / "console"
if str(CONSOLE) not in sys.path:
    sys.path.insert(0, str(CONSOLE))

from sshutil import SshSession  # noqa: E402

REMOTE_CMD = (
    'bash -l -c \'cd "$HOME/xiaoxiaobai" && '
    'if [ ! -x .venv/bin/python ]; then echo MISSING_VENV; exit 2; fi && '
    'PYTHONPATH=src .venv/bin/python -u src/lead_teach.py {role} --ids {ids}\''
)
RSYNC_EXCLUDES = [".git", ".venv", "__pycache__", "cad", "docs", "logs", "src/debug", "src/sim", "src/model/mjcf"]


def _split_target(value: str) -> tuple[str, str]:
    if "@" in value:
        user, host = value.split("@", 1)
        return host, user
    return value, "user"


def _session(target: str, password: str | None) -> SshSession:
    host, user = _split_target(target)
    return SshSession(host=host, user=user, password=password, connect_timeout=20)


def _rsync_with_retry(session: SshSession, attempts: int = 3) -> subprocess.CompletedProcess[str]:
    last = None
    for attempt in range(1, attempts + 1):
        last = session.rsync_to_remote(ROOT, "xiaoxiaobai", RSYNC_EXCLUDES, timeout=180)
        if last.returncode == 0:
            return last
        err = (last.stderr or last.stdout or "rsync 失败").strip()
        print(f"推送到 {session.target} 第 {attempt} 次失败：{err}", flush=True)
        if attempt < attempts:
            time.sleep(2 * attempt)
    return last if last is not None else subprocess.CompletedProcess([], 1, "", "rsync 失败")


def _stop_proc(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except OSError:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_bridge(lead: str, follow: str, ids: str, lead_password: str | None, follow_password: str | None) -> int:
    lead_ssh = _session(lead, lead_password)
    follow_ssh = _session(follow, follow_password)
    lead_proc = None
    follow_proc = None

    def shutdown(*_args: object) -> None:
        _stop_proc(lead_proc)
        _stop_proc(follow_proc)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"示教机 {lead}  →  真机 {follow}", flush=True)
    print("先推送 lead_teach.py，并停掉两边的走路控制循环 …", flush=True)
    for session in (lead_ssh, follow_ssh):
        result = _rsync_with_retry(session)
        if result.returncode != 0:
            print(f"连不上 {session.target}。请确认这台已开机、和电脑同一 Wi-Fi，然后重试。", flush=True)
            print((result.stderr or result.stdout or "rsync 失败"), flush=True)
            lead_ssh.close()
            follow_ssh.close()
            return 1
        stop = session.run(
            'bash -l -c \'cd "$HOME/xiaoxiaobai" && '
            'if [ -x .venv/bin/python ]; then PYTHONPATH=src .venv/bin/python src/stop.py; fi\'',
            timeout=15,
        )
        if stop.returncode != 0:
            print((stop.stderr or stop.stdout or "停控制循环失败"), flush=True)

    follow_proc = follow_ssh.popen(REMOTE_CMD.format(role="follow", ids=ids))
    lead_proc = lead_ssh.popen(REMOTE_CMD.format(role="lead", ids=ids))
    assert follow_proc.stdin is not None and lead_proc.stdout is not None

    def pump_stderr(prefix: str, proc: subprocess.Popen[str]) -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            print(f"[{prefix}] {line.rstrip()}", flush=True)

    threading.Thread(target=pump_stderr, args=("lead", lead_proc), daemon=True).start()
    threading.Thread(target=pump_stderr, args=("follow", follow_proc), daemon=True).start()

    print("已经接通。用手扳 microduck 的手臂和头，xiaoxiaobai 会跟着动。Ctrl+C 结束。", flush=True)
    frames = 0
    last_report = time.time()
    try:
        for line in lead_proc.stdout:
            if follow_proc.poll() is not None:
                break
            follow_proc.stdin.write(line)
            follow_proc.stdin.flush()
            frames += 1
            now = time.time()
            if now - last_report >= 2.0:
                try:
                    payload = json.loads(line)
                    ids_now = payload.get("ids") or []
                    print(f"{frames} 帧  ids={ids_now}", flush=True)
                except json.JSONDecodeError:
                    print(line.rstrip(), flush=True)
                last_report = now
    finally:
        shutdown()
        lead_ssh.close()
        follow_ssh.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="把 microduck 当示教机，同步到 xiaoxiaobai")
    parser.add_argument("--lead", default="user@microduck.local", help="示教机 SSH，例如 user@microduck.local")
    parser.add_argument("--follow", default="user@xiaoxiaobai.local", help="真机 SSH，例如 user@192.168.2.111")
    parser.add_argument("--ids", default="13-18")
    parser.add_argument("--lead-password", default=None)
    parser.add_argument("--follow-password", default=None)
    parser.add_argument("--password", default=None, help="两台共用的密码")
    args = parser.parse_args()
    shared = args.password
    raise SystemExit(
        run_bridge(
            args.lead,
            args.follow,
            args.ids,
            args.lead_password or shared,
            args.follow_password or shared,
        )
    )


if __name__ == "__main__":
    main()
