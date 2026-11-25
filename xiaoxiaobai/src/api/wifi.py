# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""The Pi's Wi-Fi through NetworkManager (nmcli): status, scan, join, forget.

Used by the web UI's network settings and by the phone app's Bluetooth setup, which writes
the first network of a new robot. The ESP32 follows the Pi's networks on its own
(systemd/xiaoxiaobai-wifi-export.py), so only the Pi is configured here.

nmcli runs as the service user; when NetworkManager refuses (no polkit rule), the call is
retried through `sudo -n`, like power off / reboot.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from typing import Any

from api.protocol import ApiError, CommonState, Reason

CONNECT_TIMEOUT_S = 40
DENIED = ("not authorized", "insufficient privileges", "未授权", "权限不足")


def split_terse(line: str) -> list[str]:
    """One line of `nmcli -t` output: fields separated by ':', with '\\:' and '\\\\' escaped."""
    fields, cur, i = [], [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            cur.append(line[i + 1])
            i += 2
            continue
        if c == ":":
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    fields.append("".join(cur))
    return fields


def explain(output: str) -> str:
    """nmcli's error in words a user can act on."""
    low = output.lower()
    if "secrets were required" in low or "802-11-wireless-security.psk" in low or "psk: property is invalid" in low:
        return "密码不对（或者这个网络需要别的认证方式）"
    if "no network with ssid" in low or "key-mgmt" in low or "no suitable" in low:
        return ("机器人扫描不到这个网络：只支持 2.4 GHz。手机热点请切到 2.4 GHz 频段"
                "（iPhone 打开「最大兼容性」），连接时保持热点设置页面亮着")
    if "timeout" in low or "timed out" in low:
        return "连接超时：信号太弱，或者路由器没有分配地址"
    if any(d in low for d in DENIED) or "a password is required" in low:
        return "机器人没有修改网络的权限：在机器人上运行一次 make app-enable"
    return output.strip().splitlines()[-1] if output.strip() else "连接失败"


class Wifi:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sudo = False

    @staticmethod
    def available() -> bool:
        return shutil.which("nmcli") is not None

    def _run(self, *args: str, timeout: float = 15) -> str:
        if not self.available():
            raise ApiError("这台机器没有 NetworkManager（nmcli）", CommonState.NOT_READY, Reason.NOT_SUPPORTED, 503)
        base = ["nmcli", *args]
        for attempt in range(2):
            cmd = (["sudo", "-n", *base] if self._sudo else base)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                raise ApiError("NetworkManager 没有响应", CommonState.TIMEOUT, Reason.HARDWARE, 504) from None
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0:
                return proc.stdout
            if attempt == 0 and not self._sudo and any(d in out.lower() for d in DENIED):
                self._sudo = True
                continue
            raise ApiError(explain(out), CommonState.FAILURE, Reason.HARDWARE)
        raise AssertionError

    def _device(self) -> str | None:
        for line in self._run("-t", "-f", "DEVICE,TYPE,STATE", "device").splitlines():
            dev, kind, *_ = split_terse(line) + ["", ""]
            if kind == "wifi":
                return dev
        return None

    def status(self) -> dict[str, Any]:
        if not self.available():
            return {"available": False, "connected": False, "ssid": None, "ip": None, "signal": None, "saved": []}
        dev = self._device()
        info: dict[str, Any] = {"available": dev is not None, "device": dev, "connected": False, "ssid": None,
                                "ip": None, "signal": None, "state": None}
        if dev:
            fields = {}
            for line in self._run("-t", "-f", "GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS", "device", "show", dev).splitlines():
                key, _, value = line.partition(":")
                fields.setdefault(key.split("[")[0], value)
            state = fields.get("GENERAL.STATE", "")
            info["state"] = state.split("(")[-1].rstrip(")") if "(" in state else state
            info["connected"] = state.startswith("100")
            ip = fields.get("IP4.ADDRESS", "")
            info["ip"] = ip.split("/")[0] or None
            if info["connected"]:
                for net in self._list(rescan="no"):
                    if net["in_use"]:
                        info["ssid"], info["signal"] = net["ssid"], net["signal"]
                        break
                if not info["ssid"]:
                    info["ssid"] = fields.get("GENERAL.CONNECTION") or None
        info["saved"] = self.saved()
        return info

    def saved(self) -> list[dict[str, Any]]:
        rows = []
        for line in self._run("-t", "-f", "NAME,UUID,TYPE,ACTIVE", "connection", "show").splitlines():
            name, uuid, kind, active = (split_terse(line) + ["", "", "", ""])[:4]
            if kind == "802-11-wireless":
                rows.append({"name": name, "uuid": uuid, "active": active == "yes"})
        return rows

    def _list(self, rescan: str) -> list[dict[str, Any]]:
        out = self._run("-t", "-f", "IN-USE,SSID,SIGNAL,SECURITY,FREQ", "device", "wifi", "list", "--rescan", rescan,
                        timeout=25)
        best: dict[str, dict[str, Any]] = {}
        for line in out.splitlines():
            in_use, ssid, signal, security, freq = (split_terse(line) + [""] * 5)[:5]
            if not ssid:
                continue
            net = {
                "ssid": ssid,
                "signal": int(signal) if signal.isdigit() else 0,
                "security": security.strip() or "",
                "secure": bool(security.strip() and security.strip() != "--"),
                "freq_mhz": int(re.sub(r"\D", "", freq) or 0),
                "in_use": in_use.strip() == "*",
            }
            old = best.get(ssid)
            if old is None or net["in_use"] or (not old["in_use"] and net["signal"] > old["signal"]):
                best[ssid] = net
        return sorted(best.values(), key=lambda n: (not n["in_use"], -n["signal"]))

    def scan(self) -> dict[str, Any]:
        with self._lock:
            try:
                networks = self._list(rescan="yes")
            except ApiError:
                networks = self._list(rescan="auto")  # a scan already running refuses a new one
        return {"networks": networks}

    def connect(self, ssid: str, password: str = "", hidden: bool = False) -> dict[str, Any]:
        ssid = (ssid or "").strip()
        if not ssid or len(ssid.encode()) > 32:
            raise ApiError("SSID 为 1–32 个字节", CommonState.INVALID)
        if password and not 8 <= len(password) <= 63:
            raise ApiError("Wi-Fi 密码为 8–63 个字符", CommonState.INVALID)
        with self._lock:
            self._join(ssid, password, hidden)
        deadline = time.monotonic() + 10
        status = self.status()
        while not status["ip"] and time.monotonic() < deadline:
            time.sleep(0.5)
            status = self.status()
        return status

    def _visible(self, ssid: str) -> dict[str, Any] | None:
        for rescan in ("yes", "auto"):  # a scan already running refuses a new one
            try:
                return next((n for n in self._list(rescan=rescan) if n["ssid"] == ssid), None)
            except ApiError:
                continue
        return None

    def _join(self, ssid: str, password: str, hidden: bool) -> None:
        """A profile with the security spelled out, then up. `nmcli device wifi connect` guesses
        the key management from the scan, and leaves it out (802-11-wireless-security.key-mgmt:
        property is missing) when the network is not in the scan list, as phone hotspots often
        are (5 GHz, or not broadcasting while the phone is locked)."""
        net = self._visible(ssid)
        if net is None and not hidden:
            time.sleep(2.0)  # a hotspot just switched on shows up in the next scan
            net = self._visible(ssid)
        security = (net or {}).get("security", "").upper()
        if net is not None and not net["secure"] and not password:
            key_mgmt = None
        elif not password:
            raise ApiError("这个网络需要密码", CommonState.INVALID)
        else:
            key_mgmt = "sae" if "WPA3" in security and "WPA2" not in security and "WPA1" not in security else "wpa-psk"
        for s in self.saved():
            if s["name"] == ssid:
                self._run("connection", "delete", "uuid", s["uuid"])
        args = ["connection", "add", "type", "wifi", "con-name", ssid, "ssid", ssid]
        dev = self._device()
        if dev:
            args += ["ifname", dev]
        if key_mgmt:
            args += ["wifi-sec.key-mgmt", key_mgmt, "wifi-sec.psk", password]
        if hidden or net is None:
            args += ["802-11-wireless.hidden", "yes"]
        self._run(*args)
        try:
            self._run("--wait", str(CONNECT_TIMEOUT_S), "connection", "up", "id", ssid, timeout=CONNECT_TIMEOUT_S + 10)
        except ApiError as exc:
            try:
                self._run("connection", "delete", "id", ssid)
            except ApiError:
                pass
            if net is None and not hidden:
                raise ApiError(explain("no network with ssid"), CommonState.FAILURE, Reason.HARDWARE) from exc
            raise

    def forget(self, uuid: str) -> dict[str, Any]:
        if not any(n["uuid"] == uuid for n in self.saved()):
            raise ApiError("没有这个已保存的网络", CommonState.INVALID, Reason.NOT_FOUND, 404)
        self._run("connection", "delete", "uuid", uuid)
        return {"saved": self.saved()}


class MockWifi(Wifi):
    """Pretend networks for `make api-dev`: any password of 8+ characters joins."""

    def __init__(self) -> None:
        super().__init__()
        self._networks = [
            {"ssid": "Home-2.4G", "signal": 82, "security": "WPA2", "secure": True, "freq_mhz": 2437, "in_use": False},
            {"ssid": "小小白的家", "signal": 64, "security": "WPA2 WPA3", "secure": True, "freq_mhz": 2412, "in_use": False},
            {"ssid": "Guest", "signal": 40, "security": "", "secure": False, "freq_mhz": 2462, "in_use": False},
        ]
        self._saved: list[dict[str, Any]] = []
        self._current: str | None = None

    @staticmethod
    def available() -> bool:
        return True

    def status(self) -> dict[str, Any]:
        net = next((n for n in self._networks if n["ssid"] == self._current), None)
        return {"available": True, "device": "wlan0", "connected": net is not None, "ssid": self._current,
                "ip": "127.0.0.1" if net else None, "signal": net["signal"] if net else None,
                "state": "connected" if net else "disconnected",
                "saved": [{**s, "active": s["name"] == self._current} for s in self._saved]}

    def saved(self) -> list[dict[str, Any]]:
        return self.status()["saved"]

    def scan(self) -> dict[str, Any]:
        time.sleep(1.0)
        return {"networks": [{**n, "in_use": n["ssid"] == self._current} for n in self._networks]}

    def connect(self, ssid: str, password: str = "", hidden: bool = False) -> dict[str, Any]:
        time.sleep(2.0)
        net = next((n for n in self._networks if n["ssid"] == ssid), None)
        if net is None and not hidden:
            raise ApiError(explain("No network with SSID"), CommonState.FAILURE, Reason.HARDWARE)
        if (net is None or net["secure"]) and len(password) < 8:
            raise ApiError(explain("Secrets were required"), CommonState.FAILURE, Reason.HARDWARE)
        self._current = ssid
        if not any(s["name"] == ssid for s in self._saved):
            self._saved.append({"name": ssid, "uuid": f"mock-{len(self._saved)}"})
        return self.status()

    def forget(self, uuid: str) -> dict[str, Any]:
        gone = next((s for s in self._saved if s["uuid"] == uuid), None)
        if gone is None:
            raise ApiError("没有这个已保存的网络", CommonState.INVALID, Reason.NOT_FOUND, 404)
        self._saved.remove(gone)
        if gone["name"] == self._current:
            self._current = None
        return {"saved": self.saved()}
