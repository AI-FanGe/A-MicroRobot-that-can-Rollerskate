# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import os
import signal
import time
from pathlib import Path

STOP_FLAG_FILE = Path("/tmp/xiaoxiaobai_scheduler.stop")
PID_FILE = Path("/tmp/xiaoxiaobai_scheduler.pid")


def force_disable_torque() -> None:
	try:
		from constants import MOTOR_TO_ID
		from robot_controller import RobotController

		controller = RobotController()
		try:
			motor_ids = list(MOTOR_TO_ID.values())
			controller.sync_write_torque_enable(motor_ids, [False] * len(motor_ids))
			print("Torque force-disabled on all motors.")
		finally:
			controller.shutdown()
	except Exception as exc:
		print(f"Warning: direct torque disable failed: {exc}")


def process_exists(pid: int) -> bool:
	try:
		os.kill(pid, 0)
		return True
	except ProcessLookupError:
		return False
	except PermissionError:
		return True


def main() -> None:
	STOP_FLAG_FILE.write_text("stop\n", encoding="ascii")

	if not PID_FILE.exists():
		print("No scheduler PID file found. Stop flag has been set.")
		force_disable_torque()
		return

	pid = int(PID_FILE.read_text(encoding="ascii").strip())
	if not process_exists(pid):
		print(f"Scheduler process {pid} is not running.")
		if PID_FILE.exists():
			PID_FILE.unlink()
		force_disable_torque()
		return

	print(f"Sending SIGINT to scheduler process {pid}")
	os.kill(pid, signal.SIGINT)

	timeout_s = 3.0
	start = time.perf_counter()
	while process_exists(pid) and (time.perf_counter() - start) < timeout_s:
		time.sleep(0.1)

	if process_exists(pid):
		print(f"Scheduler {pid} did not stop after SIGINT, sending SIGTERM")
		os.kill(pid, signal.SIGTERM)
		time.sleep(0.5)

	force_disable_torque()
	print("Stop request sent.")


if __name__ == "__main__":
	main()