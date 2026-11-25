# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Bluetooth LE link for the 小小白 phone app: first Wi-Fi setup and control commands.

The Pi advertises a GATT service through BlueZ (D-Bus, with dbus-fast, which bleak already
installs). The app sends the same requests the web UI sends over HTTP; they go through the
same route table, so both ways of control share one robot and one input-source arbitration.
Video, audio, file transfers and chat streams stay on HTTP over the Wi-Fi LAN.

GATT service  f3a50001-6b2c-4e5d-8f9a-786962726f74
  RX    …0002  write, write without response   app -> robot
  TX    …0003  notify                           robot -> app
  INFO  …0004  read                             JSON: id, name, ip, port, setup (no Wi-Fi yet)

Every message is one JSON object cut into packets of [flags, seq, bytes…]: flags bit 0 marks
the first packet, bit 1 the last, bit 2 a zlib-compressed message. Messages (`t`):

  hello {client, name, token?, mtu}  -> hello {auth, setup, token?, robot}
  pair                               -> pair {shown}: a 4-digit code on the face screen
  auth  {code}                       -> auth {ok, token?}
  req   {id, m, p, b?, q?}           -> res  {re, s, r}: r is the API's CommonResponse; q=1 only
                                        answers failures (velocity streams)
  sub   {on}                         -> evt  {e}: status every second, alerts

A phone with a saved token is let in at hello. A new phone is let in without a code while
the robot has no Wi-Fi (first setup); after that, it needs the code shown on the screen.
The emergency stop works without pairing.
"""

import asyncio
import hashlib
import json
import logging
import os
import queue
import secrets
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from api.protocol import ApiError, CommonState, error_response
from api.store import CONFIG_DIR, read_json, write_json_atomic

SERVICE_UUID = "f3a50001-6b2c-4e5d-8f9a-786962726f74"
RX_UUID = "f3a50002-6b2c-4e5d-8f9a-786962726f74"
TX_UUID = "f3a50003-6b2c-4e5d-8f9a-786962726f74"
INFO_UUID = "f3a50004-6b2c-4e5d-8f9a-786962726f74"

FIRST, LAST, DEFLATE = 0x01, 0x02, 0x04
MAX_MESSAGE = 256 * 1024
COMPRESS_OVER = 160
DEFAULT_MTU = 23
MAX_VALUE = 512  # BlueZ's longest attribute value
PAIR_CODE_S = 120
PAIR_ATTEMPTS = 5
SESSION_IDLE_S = 600
RETRY_S = 10
# Streams whose order matters (a stale velocity must not land after the stop); one worker.
ORDERED = ("/mc/locomotion/velocity", "/mc/upper_body", "/mc/editor/pose", "/mc/upper")
OPEN_PATHS = ("/mc/estop", "/meta/ping")
FORWARD_EVENTS = {"alert", "expression", "settings", "wifi"}

CLIENTS_PATH = CONFIG_DIR / "ble_clients.json"
APP_PATH = "/com/xiaoxiaobai/xxb"
log = logging.getLogger("xiaoxiaobai.api.ble")

Call = Callable[[str, str, str, Any], tuple[int, dict[str, Any]]]


def encode(obj: dict[str, Any], seq: int, chunk: int) -> list[bytes]:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    flags = 0
    if len(raw) > COMPRESS_OVER:
        packed = zlib.compress(raw, 6)
        if len(packed) < len(raw):
            raw, flags = packed, DEFLATE
    parts = [raw[i:i + chunk] for i in range(0, len(raw), chunk)] or [b""]
    out = []
    for i, part in enumerate(parts):
        f = (flags | FIRST if i == 0 else 0) | (LAST if i == len(parts) - 1 else 0)
        out.append(bytes((f, seq & 0xFF)) + part)
    return out


class Assembler:
    """Packets of one sender back into messages."""

    def __init__(self) -> None:
        self._buf: bytearray | None = None
        self._seq = -1
        self._deflate = False

    def feed(self, packet: bytes) -> dict[str, Any] | None:
        if len(packet) < 2:
            return None
        flags, seq = packet[0], packet[1]
        if flags & FIRST:
            self._buf, self._seq, self._deflate = bytearray(), seq, bool(flags & DEFLATE)
        elif self._buf is None or seq != self._seq:
            self._buf = None  # the start of this message was lost
            return None
        self._buf += packet[2:]
        if len(self._buf) > MAX_MESSAGE:
            self._buf = None
            return None
        if not flags & LAST:
            return None
        data, self._buf = bytes(self._buf), None
        try:
            if self._deflate:
                data = zlib.decompress(data)
            msg = json.loads(data.decode("utf-8"))
        except (zlib.error, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return msg if isinstance(msg, dict) else None


class Clients:
    """Phones paired with this robot: a random token each, kept as a hash (file mode 600)."""

    def __init__(self, path: Path = CLIENTS_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._items: list[dict[str, Any]] = read_json(path, {"clients": []}).get("clients", [])

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def _save(self) -> None:
        write_json_atomic(self._path, {"clients": self._items}, private=True)

    def check(self, client: str, token: Any) -> bool:
        if not client or not isinstance(token, str) or not token:
            return False
        digest = self._hash(token)
        with self._lock:
            for item in self._items:
                if item["id"] == client and secrets.compare_digest(item["token_sha256"], digest):
                    item["last_seen"] = time.time()
                    self._save()
                    return True
        return False

    def issue(self, client: str, name: str) -> str:
        token = secrets.token_urlsafe(24)
        with self._lock:
            self._items = [i for i in self._items if i["id"] != client]
            self._items.append({"id": client, "name": name[:40], "token_sha256": self._hash(token),
                                "created": time.time(), "last_seen": time.time()})
            self._save()
        return token

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{k: v for k, v in i.items() if k != "token_sha256"} for i in self._items]

    def revoke(self, client: str | None = None) -> int:
        with self._lock:
            before = len(self._items)
            self._items = [i for i in self._items if client is not None and i["id"] != client]
            self._save()
            return before - len(self._items)


class Session:
    def __init__(self, device: str) -> None:
        self.device = device
        self.assembler = Assembler()
        self.client = ""
        self.name = ""
        self.authed = False
        self.subscribed = False
        self.link_mtu: int | None = None  # what BlueZ negotiated (WriteValue "mtu", BlueZ 5.62+)
        self.app_mtu: int | None = None  # what the phone reports in hello
        self.since = time.time()
        self.seen = time.monotonic()
        self.code: str | None = None
        self.code_until = 0.0
        self.attempts = 0

    @property
    def mtu(self) -> int:
        known = [m for m in (self.link_mtu, self.app_mtu) if m]
        return min(known) if known else DEFAULT_MTU

    def public(self) -> dict[str, Any]:
        return {"device": self.device.rsplit("/", 1)[-1].replace("dev_", "").replace("_", ":"), "client": self.client,
                "name": self.name, "authed": self.authed, "since": self.since, "mtu": self.mtu}


class BleLink:
    """The GATT peripheral, on its own thread and asyncio loop; retries while BlueZ is missing."""

    def __init__(self, call: Call, events: Any, snapshot: Callable[[], dict[str, Any]],
                 info: Callable[[], dict[str, Any]], setup_mode: Callable[[], bool],
                 show_code: Callable[[str | None], bool], clients: Clients | None = None) -> None:
        self._call = call
        self._events = events
        self._snapshot = snapshot
        self._info_fn = info
        self._setup_mode = setup_mode
        self._show_code = show_code
        self.clients = clients or Clients()
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="xiaoxiaobai-ble")
        self._ordered: queue.Queue = queue.Queue(maxsize=64)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._outbox: asyncio.Queue | None = None
        self._tx: Any = None
        self._seq = 0
        self._info_cache = b"{}"
        self._stop = threading.Event()
        self.enabled = False
        self.running = False
        self.name = ""
        self.address = ""
        self.error: str | None = None

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        self.enabled = True
        threading.Thread(target=self._thread, name="xiaoxiaobai-ble", daemon=True).start()
        threading.Thread(target=self._ordered_worker, name="xiaoxiaobai-ble-ordered", daemon=True).start()
        threading.Thread(target=self._housekeeping, name="xiaoxiaobai-ble-house", daemon=True).start()

    def close(self) -> None:
        self._stop.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            sessions = [s.public() for s in self._sessions.values()]
        return {"enabled": self.enabled, "running": self.running, "name": self.name, "address": self.address,
                "error": self.error, "service_uuid": SERVICE_UUID, "sessions": sessions, "clients": self.clients.list()}

    def summary(self) -> dict[str, Any]:
        """For the status snapshot: is the link up, how many paired phones are connected."""
        with self._lock:
            phones = sum(1 for s in self._sessions.values() if s.authed)
        return {"running": self.running, "name": self.name, "phones": phones}

    def revoke(self, client: str | None = None) -> int:
        count = self.clients.revoke(client)
        with self._lock:
            for s in self._sessions.values():
                if client is None or s.client == client:
                    s.authed = s.subscribed = False
        return count

    def _thread(self) -> None:
        try:
            import dbus_fast  # noqa: F401
        except ImportError:
            self.error = "缺少 dbus-fast（uv sync 后再试）"
            log.warning("BLE off: %s", self.error)
            return
        while not self._stop.is_set():
            loop = asyncio.new_event_loop()
            self._loop = loop
            try:
                loop.run_until_complete(self._serve())
            except Exception as exc:  # noqa: BLE001 - D-Bus/BlueZ errors only delay the next try
                self.error = describe(exc)
                log.warning("BLE: %s; retrying in %d s", self.error, RETRY_S)
            finally:
                self.running = False
                self._tx = None
                self._outbox = None
                with self._lock:
                    self._sessions.clear()
                loop.close()
            self._stop.wait(RETRY_S)

    async def _serve(self) -> None:
        from dbus_fast import BusType, DBusError, Message, MessageType, Variant
        from dbus_fast.aio import MessageBus

        from api import ble_gatt, ble_mgmt

        bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        kernel_adv: tuple[int, int] | None = None
        try:
            adapter = await self._adapter(bus)
            props = (await self._proxy(bus, adapter)).get_interface("org.freedesktop.DBus.Properties")
            powered = await props.call_get("org.bluez.Adapter1", "Powered")
            if not powered.value:
                await props.call_set("org.bluez.Adapter1", "Powered", Variant("b", True))
                await asyncio.sleep(1.0)
            self.address = (await props.call_get("org.bluez.Adapter1", "Address")).value
            self.name = "XXB-" + self.address.replace(":", "")[-4:].upper()

            self._outbox = asyncio.Queue()
            objects = ble_gatt.build(APP_PATH, SERVICE_UUID, RX_UUID, TX_UUID, INFO_UUID, self.name,
                                     on_write=self._on_write, on_read=lambda: self._info_cache)
            for path, iface in objects.exports:
                bus.export(path, iface)
            self._tx = objects.tx
            proxy = await self._proxy(bus, adapter)
            await proxy.get_interface("org.bluez.GattManager1").call_register_application(APP_PATH, {})
            advertising = proxy.get_interface("org.bluez.LEAdvertisingManager1")
            try:
                await advertising.call_register_advertisement(objects.adv_path, {})
            except DBusError as exc:
                if "Failed to register advertisement" not in str(exc):
                    raise
                index = int(adapter.rsplit("hci", 1)[1])
                kernel_adv = (index, int(await advertising.get_supported_instances() or 1))
                await asyncio.get_running_loop().run_in_executor(
                    None, ble_mgmt.add, *kernel_adv, SERVICE_UUID, self.name)
                log.info("BLE: bluetoothd could not add the advertisement; added it through the kernel instead")
            self.running, self.error = True, None
            log.info("BLE: advertising as %s on %s (%s)", self.name, adapter, self.address)

            gone = asyncio.Event()

            def on_message(msg: Message) -> None:
                if msg.message_type != MessageType.SIGNAL:
                    return
                if msg.member == "NameOwnerChanged" and msg.body and msg.body[0] == "org.bluez" and not msg.body[2]:
                    gone.set()
                elif (msg.member == "PropertiesChanged" and msg.body and msg.body[0] == "org.bluez.Device1"
                      and "Connected" in msg.body[1] and not msg.body[1]["Connected"].value):
                    self._drop(msg.path)

            bus.add_message_handler(on_message)
            for rule in ("type='signal',sender='org.freedesktop.DBus',member='NameOwnerChanged',arg0='org.bluez'",
                         "type='signal',sender='org.bluez',interface='org.freedesktop.DBus.Properties',"
                         "member='PropertiesChanged',arg0='org.bluez.Device1'"):
                await bus.call(Message(destination="org.freedesktop.DBus", path="/org/freedesktop/DBus",
                                       interface="org.freedesktop.DBus", member="AddMatch", signature="s", body=[rule]))
            async def stopped() -> None:
                while not self._stop.is_set():
                    await asyncio.sleep(0.5)

            tasks = [asyncio.ensure_future(t) for t in (self._sender(), stopped(), gone.wait(), bus.wait_for_disconnect())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if gone.is_set():
                raise RuntimeError("bluetoothd 重启了")
            for task in done:
                if task.exception() is not None:
                    raise task.exception()
        finally:
            if kernel_adv is not None:
                try:
                    ble_mgmt.remove(*kernel_adv)
                except (OSError, ble_mgmt.MgmtError):
                    pass
            bus.disconnect()

    @staticmethod
    async def _proxy(bus: Any, path: str) -> Any:
        return bus.get_proxy_object("org.bluez", path, await bus.introspect("org.bluez", path))

    async def _adapter(self, bus: Any) -> str:
        from dbus_fast import Variant

        manager = (await self._proxy(bus, "/")).get_interface("org.freedesktop.DBus.ObjectManager")
        wanted = os.environ.get("XIAOXIAOBAI_BLE_ADAPTER", "")
        for attempt in range(2):
            objects = await manager.call_get_managed_objects()
            adapters = sorted(p for p, ifaces in objects.items() if "org.bluez.Adapter1" in ifaces)
            if wanted:
                adapters = [p for p in adapters if p.endswith("/" + wanted)]
            if not adapters:
                raise RuntimeError("没有找到蓝牙适配器")
            ready = [p for p in adapters if "org.bluez.LEAdvertisingManager1" in objects[p]
                     and "org.bluez.GattManager1" in objects[p]]
            if ready:
                return ready[0]
            if attempt == 0:  # the LE managers only appear once the adapter is powered
                props = (await self._proxy(bus, adapters[0])).get_interface("org.freedesktop.DBus.Properties")
                await props.call_set("org.bluez.Adapter1", "Powered", Variant("b", True))
                await asyncio.sleep(1.5)
        raise RuntimeError("蓝牙适配器不支持低功耗外设（BLE 广播）")

    # ---- packets ---------------------------------------------------------------------------

    def _on_write(self, data: bytes, options: dict[str, Any]) -> None:
        device = str(options.get("device") or "?")
        with self._lock:
            sess = self._sessions.get(device)
            if sess is None:
                sess = self._sessions[device] = Session(device)
            if options.get("mtu"):
                sess.link_mtu = int(options["mtu"])
            sess.seen = time.monotonic()
        msg = sess.assembler.feed(data)
        if msg is None:
            return
        path = str(msg.get("p") or "")
        if msg.get("t") == "req" and path.split("?")[0] in ORDERED:
            try:
                self._ordered.put_nowait((sess, msg))
            except queue.Full:
                pass  # the next velocity sample replaces this one
        elif msg.get("t") == "req" and path.split("?")[0] == "/mc/estop":
            threading.Thread(target=self._handle, args=(sess, msg), daemon=True).start()
        else:
            self._pool.submit(self._handle, sess, msg)

    def _drop(self, device: str) -> None:
        with self._lock:
            sess = self._sessions.pop(device, None)
        if sess is not None:
            log.info("BLE: %s (%s) disconnected", sess.name or "phone", sess.public()["device"])

    def _chunk(self) -> int:
        with self._lock:
            mtus = [s.mtu for s in self._sessions.values()] or [DEFAULT_MTU]
        return max(DEFAULT_MTU - 5, min(min(mtus) - 5, MAX_VALUE - 2))

    def send(self, obj: dict[str, Any]) -> None:
        loop, outbox = self._loop, self._outbox
        if loop is None or outbox is None or not self.running:
            return
        loop.call_soon_threadsafe(outbox.put_nowait, obj)

    async def _sender(self) -> None:
        assert self._outbox is not None
        while True:
            obj = await self._outbox.get()
            if self._tx is None:
                continue
            self._seq = (self._seq + 1) & 0xFF
            for packet in encode(obj, self._seq, self._chunk()):
                self._tx.notify(packet)
                await asyncio.sleep(0.004)

    # ---- messages --------------------------------------------------------------------------

    def _ordered_worker(self) -> None:
        while not self._stop.is_set():
            try:
                sess, msg = self._ordered.get(timeout=1.0)
            except queue.Empty:
                continue
            self._handle(sess, msg)

    def _handle(self, sess: Session, msg: dict[str, Any]) -> None:
        try:
            kind = msg.get("t")
            if kind == "hello":
                self._hello(sess, msg)
            elif kind == "pair":
                self._pair(sess, msg)
            elif kind == "auth":
                self._auth(sess, msg)
            elif kind == "sub":
                sess.subscribed = bool(msg.get("on")) and sess.authed
                self.send({"t": "sub", "re": msg.get("id"), "on": sess.subscribed})
            elif kind == "req":
                self._request(sess, msg)
        except Exception:  # noqa: BLE001 - one bad message never stops the link
            log.exception("BLE: message %r failed", msg.get("t"))

    def _hello(self, sess: Session, msg: dict[str, Any]) -> None:
        sess.client = str(msg.get("client") or "")[:64]
        sess.name = str(msg.get("name") or "手机")[:40]
        if msg.get("mtu"):
            sess.app_mtu = max(DEFAULT_MTU, int(msg["mtu"]))
        setup = self._setup_mode()
        token = None
        if self.clients.check(sess.client, msg.get("token")):
            sess.authed = True
        elif setup and sess.client:
            sess.authed = True
            token = self.clients.issue(sess.client, sess.name)
        log.info("BLE: hello from %s (%s), %s", sess.name, sess.public()["device"],
                 "paired" if sess.authed else "needs the pairing code")
        self.send({"t": "hello", "re": msg.get("id"), "c": sess.client, "auth": sess.authed, "setup": setup,
                   "token": token, "robot": self._info()})

    def _pair(self, sess: Session, msg: dict[str, Any]) -> None:
        sess.code = f"{secrets.randbelow(10000):04d}"
        sess.code_until = time.monotonic() + PAIR_CODE_S
        sess.attempts = 0
        shown = self._show_code(sess.code)
        self.send({"t": "pair", "re": msg.get("id"), "c": sess.client, "shown": shown, "expires_s": PAIR_CODE_S})

    def _auth(self, sess: Session, msg: dict[str, Any]) -> None:
        code = str(msg.get("code") or "").strip()
        ok = bool(sess.code and time.monotonic() < sess.code_until and sess.client
                  and secrets.compare_digest(code, sess.code))
        token = None
        if ok:
            sess.authed, sess.code = True, None
            token = self.clients.issue(sess.client, sess.name)
            self._show_code(None)
            log.info("BLE: paired %s", sess.name)
        else:
            sess.attempts += 1
            if sess.attempts >= PAIR_ATTEMPTS:
                sess.code = None
        self.send({"t": "auth", "re": msg.get("id"), "c": sess.client, "ok": ok, "token": token,
                   "message": "" if ok else ("配对码错误次数太多，请重新获取" if sess.code is None else "配对码不对")})

    def _request(self, sess: Session, msg: dict[str, Any]) -> None:
        path = str(msg.get("p") or "/")
        route, _, query = path.partition("?")
        method = str(msg.get("m") or "GET").upper()
        if not sess.authed and route not in OPEN_PATHS:
            status, payload = 401, error_response(ApiError("手机还没有和机器人配对", CommonState.INVALID, http_status=401))
        else:
            status, payload = self._call(method, "/api/v1" + route, query, msg.get("b"))
        if msg.get("q") and payload.get("header", {}).get("code") == 0:
            return
        self.send({"t": "res", "re": msg.get("id"), "c": sess.client, "s": status, "r": payload})

    # ---- background ------------------------------------------------------------------------

    def _info(self) -> dict[str, Any]:
        try:
            return json.loads(self._info_cache)
        except json.JSONDecodeError:
            return {}

    def _refresh_info(self) -> None:
        try:
            info = {"v": 1, **self._info_fn(), "ble": self.name}
        except Exception as exc:  # noqa: BLE001 - nmcli hiccups keep the last value
            log.debug("BLE info: %s", exc)
            return
        data = json.dumps(info, ensure_ascii=False, separators=(",", ":")).encode()
        if len(data) > MAX_VALUE:
            info.pop("ips", None)
            data = json.dumps(info, ensure_ascii=False, separators=(",", ":")).encode()
        self._info_cache = data[:MAX_VALUE]

    def _housekeeping(self) -> None:
        """INFO every few seconds; status and selected events to subscribed phones."""
        sub = None
        last_info = last_status = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_info > 3.0:
                self._refresh_info()
                last_info = now
                with self._lock:
                    for device, s in list(self._sessions.items()):
                        if now - s.seen > SESSION_IDLE_S:
                            self._sessions.pop(device, None)
            with self._lock:
                wanted = self.running and any(s.subscribed for s in self._sessions.values())
            if wanted and sub is None:
                sub = self._events.subscribe()
            elif not wanted and sub is not None:
                self._events.unsubscribe(sub)
                sub = None
            if sub is None:
                self._stop.wait(0.5)
                continue
            try:
                event = sub.get(timeout=0.5)
                if event.get("type") in FORWARD_EVENTS and event.get("ble", True):
                    self.send({"t": "evt", "e": event})
            except queue.Empty:
                pass
            if time.monotonic() - last_status >= 1.0:
                self.send({"t": "evt", "e": {"type": "status", **self._snapshot()}})
                last_status = time.monotonic()


def describe(exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    low = text.lower()
    if "org.bluez.error.blocked" in low or "rfkill" in low:
        return "蓝牙被 rfkill 关掉了（sudo rfkill unblock bluetooth）"
    if "accessdenied" in low or "not allowed" in low or "rejected send message" in low:
        return "没有使用蓝牙的权限：运行一次 make app-enable（把用户加入 bluetooth 组）"
    if "org.freedesktop.dbus.error.serviceunknown" in low or "org.bluez was not provided" in low:
        return "蓝牙服务 bluetoothd 没有运行"
    if "no such file" in low and "bus" in low:
        return "连不上系统 D-Bus"
    if "already exists" in low:
        return "蓝牙广播已被占用（另一个 API 进程还在运行？）"
    if "maximum advertisements reached" in low:
        return "蓝牙广播数量已满"
    return text
