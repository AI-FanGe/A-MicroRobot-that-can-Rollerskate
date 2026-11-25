# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""BlueZ D-Bus objects of the app's GATT service and its advertisement (see api/ble.py).

No `from __future__ import annotations` here: dbus-fast reads the D-Bus signatures from the
annotations ("ay", "a{sv}", …).
"""

from dataclasses import dataclass
from typing import Any, Callable

from dbus_fast.service import PropertyAccess, ServiceInterface, dbus_method, dbus_property

READ = PropertyAccess.READ


class Service(ServiceInterface):
    def __init__(self, uuid: str) -> None:
        super().__init__("org.bluez.GattService1")
        self._uuid = uuid

    @dbus_property(access=READ)
    def UUID(self) -> "s":  # noqa: N802, F821 - BlueZ property names and D-Bus signatures
        return self._uuid

    @dbus_property(access=READ)
    def Primary(self) -> "b":  # noqa: N802, F821
        return True


class Characteristic(ServiceInterface):
    def __init__(self, uuid: str, service: str, flags: list[str],
                 on_write: Callable[[bytes, dict[str, Any]], None] | None = None,
                 on_read: Callable[[], bytes] | None = None) -> None:
        super().__init__("org.bluez.GattCharacteristic1")
        self._uuid = uuid
        self._service = service
        self._flags = flags
        self._on_write = on_write
        self._on_read = on_read
        self._value = b""
        self._notifying = False

    @dbus_property(access=READ)
    def UUID(self) -> "s":  # noqa: N802, F821
        return self._uuid

    @dbus_property(access=READ)
    def Service(self) -> "o":  # noqa: N802, F821
        return self._service

    @dbus_property(access=READ)
    def Flags(self) -> "as":  # noqa: N802, F821
        return self._flags

    @dbus_property(access=READ)
    def Value(self) -> "ay":  # noqa: N802, F821
        return self._value

    @dbus_property(access=READ)
    def Notifying(self) -> "b":  # noqa: N802, F821
        return self._notifying

    @dbus_method()
    def ReadValue(self, options: "a{sv}") -> "ay":  # noqa: N802, F821
        data = self._on_read() if self._on_read else self._value
        offset = int(options["offset"].value) if "offset" in options else 0
        return data[offset:]

    @dbus_method()
    def WriteValue(self, value: "ay", options: "a{sv}"):  # noqa: N802, F821
        if self._on_write:
            self._on_write(bytes(value), {k: v.value for k, v in options.items()})

    @dbus_method()
    def StartNotify(self):  # noqa: N802
        if not self._notifying:
            self._notifying = True
            self.emit_properties_changed({"Notifying": True})

    @dbus_method()
    def StopNotify(self):  # noqa: N802
        if self._notifying:
            self._notifying = False
            self.emit_properties_changed({"Notifying": False})

    def notify(self, data: bytes) -> None:
        self._value = data
        if self._notifying:
            self.emit_properties_changed({"Value": data})


class Advertisement(ServiceInterface):
    def __init__(self, service_uuid: str, name: str) -> None:
        super().__init__("org.bluez.LEAdvertisement1")
        self._uuid = service_uuid
        self._name = name

    @dbus_property(access=READ)
    def Type(self) -> "s":  # noqa: N802, F821
        return "peripheral"

    @dbus_property(access=READ)
    def ServiceUUIDs(self) -> "as":  # noqa: N802, F821
        return [self._uuid]

    @dbus_property(access=READ)
    def LocalName(self) -> "s":  # noqa: N802, F821
        return self._name

    @dbus_property(access=READ)
    def Discoverable(self) -> "b":  # noqa: N802, F821
        return True

    @dbus_method()
    def Release(self):  # noqa: N802
        pass


@dataclass
class Objects:
    exports: list[tuple[str, ServiceInterface]]
    tx: Characteristic
    adv_path: str


def build(root: str, service_uuid: str, rx_uuid: str, tx_uuid: str, info_uuid: str, name: str,
          on_write: Callable[[bytes, dict[str, Any]], None], on_read: Callable[[], bytes]) -> Objects:
    service = f"{root}/service0"
    rx = Characteristic(rx_uuid, service, ["write", "write-without-response"], on_write=on_write)
    tx = Characteristic(tx_uuid, service, ["notify"])
    info = Characteristic(info_uuid, service, ["read"], on_read=on_read)
    adv_path = f"{root}/advertisement0"
    return Objects(
        exports=[
            (service, Service(service_uuid)),
            (f"{service}/char0", rx),
            (f"{service}/char1", tx),
            (f"{service}/char2", info),
            (adv_path, Advertisement(service_uuid, name)),
        ],
        tx=tx,
        adv_path=adv_path,
    )
