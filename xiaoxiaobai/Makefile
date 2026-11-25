.PHONY: sync setup image run stop fetch-walk-rec analyze-walk-rec motor-check shutdown voltage imu imu-ui bno08x-ui sim direction-check joint-compare policy-joint-compare training-direction joint-limits display-enable display-disable gamepad-headless-enable gamepad-headless-disable console lead speaker-test camera camera-wifi voice voice-key music-setup music-test esp32-ip firmware firmware-upload firmware-uart firmware-ota wifi-sync-enable wifi-sync-disable api api-dev api-enable api-disable api-logs webui-model app-enable app-disable app-icon app-apk

HOST ?= xiaoxiaobai
ID ?=
ARGS ?=
ENV ?=
LEAD ?= user@microduck.local
FOLLOW ?= user@xiaoxiaobai.local
PASSWORD ?= password
ARDUINO_CLI ?= $(or $(shell command -v arduino-cli),$(lastword $(sort $(wildcard $(HOME)/Applications/arduino-ide-*/resources/app/lib/backend/resources/arduino-cli))),arduino-cli)
ESP32_FQBN ?= esp32:esp32:XIAO_ESP32S3:PSRAM=opi
ESP32_PORT ?= /dev/ttyACM0
ESP32_HOST ?= xiaoxiaobai-face.local
ESP32_OTA_PASSWORD ?= xiaoxiaobai
ESPOTA ?= $(lastword $(sort $(wildcard $(HOME)/.arduino15/packages/esp32/hardware/esp32/*/tools/espota.py)))
FIRMWARE_BUILD ?= build/xiao_face
CAMERA_PORT ?= 8090
API_PORT ?= 8000

sync:
	rsync -avz \
		--exclude='.git' \
		--exclude='.venv' \
		--exclude='__pycache__' \
		--exclude='cad' \
		--exclude='docs' \
		--exclude='firmware' \
		--exclude='build' \
		--exclude='logs' \
		--exclude='src/debug' \
		--exclude='src/sim' \
		--exclude='src/model/mjcf' \
		--exclude='android' \
		--exclude='image' \
		./ $(HOST):xiaoxiaobai

setup: sync
	ssh $(HOST) "bash -l -c 'cd xiaoxiaobai && uv sync --frozen'"

# Ready-to-flash SD-card image, build/image/xiaoxiaobai-<date>.img.xz, from the official Raspberry Pi
# OS Lite with this code installed (asks for sudo; an x86 computer also needs qemu-user-static).
image:
	sudo bash image/build.sh

sim:
	PYTHONPATH=src uv run --group sim src/sim/sim_main.py --hz 50

direction-check:
	PYTHONPATH=src uv run --group sim src/debug/mujoco_motor_direction_check.py --host $(HOST)

joint-compare:
	PYTHONPATH=src uv run --group sim src/debug/mujoco_joint_slider_compare.py --host $(HOST) $(ARGS)

policy-joint-compare:
	PYTHONPATH=src uv run --group sim src/debug/policy_joint_slider_compare.py --host $(HOST) $(ARGS)

# Training-model joint sign vs the real robot. Positive slider = training policy positive.
training-direction:
	PYTHONPATH=src uv run --group sim src/debug/training_direction_slider.py --host $(HOST) $(ARGS)

# Unlimited sliders (training MuJoCo + real robot) to measure joint limits for the get-up training.
# Records to calibration/getup_joint_limits.json. ARGS="--no-robot" for MuJoCo only, "--kp 250" to push softer.
joint-limits:
	PYTHONPATH=src uv run --group sim src/debug/joint_limit_slider.py --host $(HOST) $(ARGS)

display-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && sudo bash systemd/install-display-daemon.sh'"

display-disable:
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && sudo bash systemd/install-display-daemon.sh --uninstall'"

# Everything is driven from the web UI: syncs, starts the API + web UI on the robot and opens it in
# this computer's browser (SSH tunnel). Ctrl-C stops the robot (torque off). ENV="VAR=value" for the robot.
run: sync
	@python3 scripts/webui_run.py --host $(HOST) --port $(API_PORT) --env "$(ENV)"

stop:
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && PYTHONPATH=src .venv/bin/python src/stop.py'"

# Every walk and dance session of `make run` is recorded on the robot (logs/walk_rec/walk_*.npz, dance_*.npz).
fetch-walk-rec:
	mkdir -p logs/walk_rec
	rsync -avz $(HOST):xiaoxiaobai/logs/walk_rec/ logs/walk_rec/

# Hang the robot (feet off the ground) first. ARGS="--no-step" to only hold the neutral pose.
motor-check: sync
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && PYTHONPATH=src .venv/bin/python src/motor_health_check.py $(ARGS)'"

analyze-walk-rec: fetch-walk-rec
	PYTHONPATH=src uv run --group sim src/debug/analyze_walk_rec.py $(ARGS)

# Beep the speaker amplifier on the head board; works while `make run` is up. ARGS="--freq 440 --ms 1000 --volume 50"
speaker-test:
	ssh $(HOST) "bash -l -c 'cd xiaoxiaobai && PYTHONPATH=src .venv/bin/python src/speaker.py $(ARGS)'"

# Live view of the head-board camera in the browser, tunnelled over SSH; works with or without `make run`.
# ARGS="--size vga --quality 10" (sizes: qqvga qvga hvga vga)
camera: sync
	python3 scripts/camera_view.py --host $(HOST) --port $(CAMERA_PORT) $(ARGS)

# Real-time voice chat with Qwen through the robot's microphone, speaker and camera; needs `make run`
# or the display service. The ESP32 talks to Qwen directly; ARGS="--route pi" relays through the Pi.
# ARGS="--voice Momo --full-duplex --verbose", ARGS="--check" tests the key only.
# `uv sync` only when the voice stack's packages are missing, so a fresh robot gets
# websockets without reinstalling everything on every start.
voice: sync
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && { .venv/bin/python -c \"import websockets\" || uv sync --frozen; } && PYTHONPATH=src .venv/bin/python src/voice_chat.py $(ARGS)'"

# Store the Qwen API key on the robot (~/.config/xiaoxiaobai/dashscope_api_key, mode 600): copies this
# computer's ~/.config/xiaoxiaobai/dashscope_api_key if there is one, otherwise asks for it.
# It is also the key for text chat until another language model is configured in the web UI.
voice-key:
	@bash -c 'f=~/.config/xiaoxiaobai/dashscope_api_key; if [ -s "$$f" ]; then key=$$(cat "$$f"); echo "using $$f"; else read -rsp "Qwen API key: " key && echo; fi; printf "%s" "$$key" | ssh $(HOST) "umask 077 && mkdir -p ~/.config/xiaoxiaobai && cat > ~/.config/xiaoxiaobai/dashscope_api_key" && echo "saved on $(HOST)"'

# ffmpeg on the robot, which decodes the songs of the web UI's music page (NetEase Cloud Music)
# for the ESP32 speaker. ARGS="--url https://..." with `make music-test` plays one file.
music-setup:
	ssh -tt $(HOST) "command -v ffmpeg >/dev/null || { sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg; }; ffmpeg -version | head -1"

music-test: sync
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && PYTHONPATH=src .venv/bin/python src/music_player.py $(ARGS)'"

# Head-board face/speaker/camera firmware (talks to the controller over its UART).
firmware:
	$(ARDUINO_CLI) compile --fqbn $(ESP32_FQBN) firmware/xiao_face

# Default install: this computer builds it, the Pi (any network) writes it over the UART.
# The ESP32 does not need Wi-Fi. Wi-Fi OTA remains `make firmware-ota`, USB remains `make firmware-upload`.
firmware-uart: sync
	$(ARDUINO_CLI) compile --fqbn $(ESP32_FQBN) --output-dir $(FIRMWARE_BUILD) firmware/xiao_face
	rsync -az $(FIRMWARE_BUILD)/xiao_face.ino.bin $(HOST):/tmp/xiao_face.ino.bin
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && PYTHONPATH=src .venv/bin/python src/esp32_uart_ota.py /tmp/xiao_face.ino.bin'"

firmware-upload:
	$(ARDUINO_CLI) compile --upload --port $(ESP32_PORT) --fqbn $(ESP32_FQBN) firmware/xiao_face

# Same firmware over the ESP32's own Wi-Fi (ArduinoOTA).
firmware-ota:
	$(ARDUINO_CLI) compile --fqbn $(ESP32_FQBN) --output-dir $(FIRMWARE_BUILD) firmware/xiao_face
	python3 scripts/esp32_wifi.py --esp32 $(ESP32_HOST) --host $(HOST) ota --espota "$(ESPOTA)" --password $(ESP32_OTA_PASSWORD) $(FIRMWARE_BUILD)/xiao_face.ino.bin

# Camera straight from the ESP32 over Wi-Fi (the Pi only hands it the Wi-Fi list). ARGS="--size qvga"
camera-wifi:
	python3 scripts/esp32_wifi.py --esp32 $(ESP32_HOST) --host $(HOST) camera $(ARGS)

esp32-ip:
	python3 scripts/esp32_wifi.py --esp32 $(ESP32_HOST) --host $(HOST) ip

# Let the ESP32 follow the Pi's saved Wi-Fi networks (installs a root export service, asks for sudo).
wifi-sync-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && sudo bash systemd/install-wifi-export.sh'"

wifi-sync-disable:
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && sudo bash systemd/install-wifi-export.sh --uninstall'"

voltage: sync
	ssh $(HOST) "bash -l -c 'cd xiaoxiaobai && PYTHONPATH=src .venv/bin/python src/voltage.py $(ID)'"

imu: sync
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && PYTHONPATH=src .venv/bin/python src/imu.py'"

imu-ui:
	python3 scripts/imu_ui.py --host $(HOST)

# Temporary I2C IMU test UI. Does not touch robot control code.
bno08x-ui:
	python3 scripts/bno08x_ui.py --host $(HOST)

console:
	python3 console/server.py --restart $(ARGS)

# Robot API + web UI (http://<robot-ip>:$(API_PORT)/), in the foreground on the robot.
api: sync
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && PYTHONPATH=src .venv/bin/python -u -m api --port $(API_PORT) $(ARGS)'"

# Same service on this computer with a simulated robot (no hardware), for UI work.
api-dev:
	PYTHONPATH=src python3 -m api --mock --port $(API_PORT) $(ARGS)

# Start the API at boot so phones/browsers can connect without SSH (asks for sudo).
api-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && sudo bash systemd/install-api-daemon.sh --port $(API_PORT)'"

api-disable:
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && sudo bash systemd/install-api-daemon.sh --uninstall'"

api-logs:
	ssh -tt $(HOST) "journalctl -u xiaoxiaobai-api -f -n 100"

# 小小白 phone app: Bluetooth at boot, Wi-Fi setup permission, ESP32 Wi-Fi follow, API at boot (asks for sudo).
app-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && sudo bash systemd/install-app.sh --port $(API_PORT)'"

app-disable:
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && sudo bash systemd/install-app.sh --uninstall'"

# Redraw the logo (webui/img/logo.svg and the Android launcher icon) from scripts/build_app_icon.py.
app-icon:
	python3 scripts/build_app_icon.py

# Build the Android app: android/app/build/outputs/apk/release/app-release.apk (needs JDK 17+ and ANDROID_HOME).
app-apk:
	cd android && ./gradlew assembleRelease

# Regenerate webui/model/ (the web UI's virtual robot) from ../modelforRLV6.3dm, using the link
# layout of ../mjlab_xiaoxiaobai/scripts/rhino3dm_to_mjcf.py. ARGS="--src path.3dm --keep 0.4"
webui-model:
	uv run --no-project --python 3.12 --with rhino3dm --with fast-simplification --with numpy python scripts/build_webui_model.py $(ARGS)

# microduck 当示教机：手扳 13-18，xiaoxiaobai 同号电机跟随。
# 例: make lead FOLLOW=user@192.168.2.111 PASSWORD=password
lead:
	python3 scripts/lead_bridge.py --lead $(LEAD) --follow $(FOLLOW) --ids 13-18 $(if $(PASSWORD),--password $(PASSWORD),)

shutdown:
	ssh -tt $(HOST) "sudo shutdown -h now"

# Opt-in headless mode: a service launches the control loop when START is held 2s on
# the gamepad (no SSH needed); B stops it.
gamepad-headless-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && sudo bash systemd/install-gamepad-daemon.sh'"

gamepad-headless-disable:
	ssh -tt $(HOST) "bash -l -c 'cd xiaoxiaobai && sudo bash systemd/install-gamepad-daemon.sh --uninstall'"
