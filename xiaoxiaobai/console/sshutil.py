# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _resolve_config_dir() -> Path:
    preferred = Path.home() / ".xiaoxiaobai" / "console"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        os.chmod(preferred, 0o700)
        return preferred
    except OSError:
        fallback = Path(__file__).resolve().parents[1] / ".console-data"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


CONFIG_DIR = _resolve_config_dir()
SSH_DIR = CONFIG_DIR / "ssh"


def _ensure_dirs() -> None:
    SSH_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, 0o700)
        os.chmod(SSH_DIR, 0o700)
    except OSError:
        pass


@contextmanager
def _askpass_env(password: str | None) -> Iterator[tuple[dict[str, str], list[str]]]:
    env = os.environ.copy()
    extra: list[str] = []
    askpass = None
    if password:
        handle, path = tempfile.mkstemp(prefix="xiaoxiaobai_askpass_", text=True)
        os.close(handle)
        Path(path).write_text(
            "#!/bin/sh\ncat <<'EOF'\n" + password + "\nEOF\n",
            encoding="utf-8",
        )
        os.chmod(path, 0o700)
        askpass = path
        env.pop("DISPLAY", None)
        env["SSH_ASKPASS"] = path
        env["SSH_ASKPASS_REQUIRE"] = "force"
        extra += [
            "-o",
            "PreferredAuthentications=password",
            "-o",
            "PubkeyAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "NumberOfPasswordPrompts=1",
        ]
    else:
        extra += ["-o", "BatchMode=yes"]
    try:
        yield env, extra
    finally:
        if askpass is not None:
            try:
                os.remove(askpass)
            except OSError:
                pass


class SshSession:
    def __init__(
        self,
        host: str,
        user: str = "user",
        port: int = 22,
        password: str | None = None,
        connect_timeout: int = 8,
    ) -> None:
        self.host = host.strip()
        self.user = (user or "user").strip()
        self.port = int(port or 22)
        self.password = password or None
        self.connect_timeout = int(connect_timeout)
        self._askpass_files: list[str] = []
        _ensure_dirs()

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def _ssh_opts(self, extra: list[str]) -> list[str]:
        # Password logins skip ControlMaster: a previous failed master (wrong
        # user / host key) will otherwise keep failing silently.
        opts = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            "-o",
            "ServerAliveInterval=5",
            "-p",
            str(self.port),
        ]
        if not self.password:
            opts += [
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={SSH_DIR}/%C",
                "-o",
                "ControlPersist=90",
            ]
        return opts + extra

    def run(
        self,
        remote: str,
        timeout: float = 20.0,
        tty: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if shutil.which("ssh") is None:
            raise RuntimeError("本机找不到 ssh 命令")
        with _askpass_env(self.password) as (env, extra):
            cmd = self._ssh_opts(extra)
            if tty:
                cmd.insert(1, "-tt")
            else:
                cmd.insert(1, "-T")
            cmd += [self.target, remote]
            return subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=timeout,
                env=env,
                stdin=subprocess.DEVNULL,
            )

    def rsync_to_remote(
        self,
        local_dir: Path,
        remote_dir: str,
        excludes: list[str],
        timeout: float = 180.0,
    ) -> subprocess.CompletedProcess[str]:
        if shutil.which("rsync") is None:
            raise RuntimeError("本机找不到 rsync 命令")
        with _askpass_env(self.password) as (env, extra):
            # rsync appends the host itself; -e must be ssh options only.
            ssh_cmd = self._ssh_opts(extra)
            cmd = [
                "rsync",
                "-az",
                "--partial",
                *sum((["--exclude", item] for item in excludes), []),
                "-e",
                " ".join(shlex.quote(part) for part in ssh_cmd),
                str(local_dir) + "/",
                f"{self.target}:{remote_dir}",
            ]
            return subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=timeout,
                env=env,
            )

    def popen(
        self,
        remote: str,
        *,
        stdin: int | None = subprocess.PIPE,
        stdout: int | None = subprocess.PIPE,
        stderr: int | None = subprocess.PIPE,
    ) -> subprocess.Popen[str]:
        """Long-lived SSH process. Caller must terminate it; askpass is cleaned in close()."""
        if shutil.which("ssh") is None:
            raise RuntimeError("本机找不到 ssh 命令")
        env = os.environ.copy()
        extra: list[str] = []
        if self.password:
            handle, path = tempfile.mkstemp(prefix="xiaoxiaobai_askpass_", text=True)
            os.close(handle)
            Path(path).write_text(
                "#!/bin/sh\ncat <<'EOF'\n" + self.password + "\nEOF\n",
                encoding="utf-8",
            )
            os.chmod(path, 0o700)
            self._askpass_files.append(path)
            env.pop("DISPLAY", None)
            env["SSH_ASKPASS"] = path
            env["SSH_ASKPASS_REQUIRE"] = "force"
            extra += [
                "-o",
                "PreferredAuthentications=password",
                "-o",
                "PubkeyAuthentication=no",
                "-o",
                "KbdInteractiveAuthentication=no",
                "-o",
                "NumberOfPasswordPrompts=1",
            ]
        else:
            extra += ["-o", "BatchMode=yes"]
        cmd = self._ssh_opts(extra) + ["-T", self.target, remote]
        return subprocess.Popen(
            cmd,
            text=True,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=env,
            bufsize=1,
        )

    def close(self) -> None:
        for path in self._askpass_files:
            try:
                os.remove(path)
            except OSError:
                pass
        self._askpass_files.clear()
