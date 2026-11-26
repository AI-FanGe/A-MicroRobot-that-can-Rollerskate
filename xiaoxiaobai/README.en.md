<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md"><strong>English</strong></a>
</p>

# Xiaoxiaobai Robot-Side Code

Control code running on the main controller, together with the head firmware, Web UI, and Android app.

| Directory | Contents |
| :--- | :--- |
| `src/` | Control loop, motions (`src/moves/`), IMU / servo drivers, voice, and music |
| `src/agents/` | ONNX policies and reference motions for deployment |
| `src/api/` | Robot HTTP API and Web UI backend (`http://<robot-ip>:8000/`) |
| `src/sim/`, `src/debug/` | MuJoCo simulation runtime and sim-to-real debugging tools |
| `src/model/mjcf/` | Robot MJCF model (references the same mesh files in `../mjlab_xiaoxiaobai`; used only for simulation / debugging) |
| `webui/` | Browser-based control panel |
| `console/` | Multi-robot console for computers |
| `firmware/xiao_face/` | Firmware for the head's face display / speaker / camera |
| `android/` | Xiaoxiaobai Android app |
| `systemd/` | Services that start automatically at boot |
| `image/` | Build scripts for the Raspberry Pi SD card image (`make image`) |
| `calibration/` | Servo zero-position and joint-limit calibration |

## Usage

All operations are performed from your computer through `Makefile` over SSH. The default robot hostname is `xiaoxiaobai` (override with `HOST=...`). The code directory on the robot is `~/xiaoxiaobai`.

Remote commands below that omit `HOST` require the `xiaoxiaobai` alias to be configured first, including the robot's IP address and username, as described in the [main guide's SSH setup](../README.en.md#2-configure-ssh). Without an alias, explicitly specify the target from this directory on your computer:

```bash
make setup HOST=user@192.168.1.100
make run HOST=user@192.168.1.100
```

Replace the example IP with the robot's actual IP. The Xiaoxiaobai image uses the username `user`; with the official OS image, use the username you configured. `HOST=...` applies only to the current command and must be specified each time you run a remote command.

| Command | Purpose |
| :--- | :--- |
| `make setup` | Sync code and install dependencies on the robot (`uv sync --frozen`) |
| `make image` | Build a flashable Raspberry Pi image (`build/image/xiaoxiaobai-<date>.img.xz`; requires sudo) |
| `make run` | Sync code, start the API + Web UI on the robot, and open it in your computer's browser |
| `make stop` | Stop the control loop and disable servo torque |
| `make api-enable` | Start the API service automatically at boot |
| `make api-dev` | Run the API / Web UI locally in mock mode (no hardware required) |
| `make sim` | Run the control code in MuJoCo |
| `make firmware-uart` | Compile and flash the head firmware through the main controller's UART |
| `make app-apk` | Build the Android app (requires JDK 17+ and the Android SDK) |
| `make shutdown` | Shut down safely (always run before cutting power) |

See `Makefile` for the full command list.

## Policy Models

The Web UI currently uses the Chinese labels shown alongside their English translations below.

| File | Web UI name | Description |
| :--- | :--- | :--- |
| `walk_low_battery_1.onnx` | Low-Battery Gait 1 (低电量步态1) | Default walking policy, trained over a supply voltage range of 5–8 V |
| `walk_low_battery_2.onnx` | Low-Battery Gait 2 (低电量步态2) | Builds on Gait 1 with training that includes battery voltage drops during walking, improving stability as voltage falls |
| `walk_free_arms.onnx` | Walk/Run · Free Upper Body (走跑·上肢自由) | Walking policy for free-upper-body mode; controls only the legs |
| `skate_glide.onnx` | Two-Foot Gliding (双脚滑行) | Default roller-skating policy |
| `skate_single_leg.onnx` | Single-Leg Push-Off (单脚蹬滑) | Roller-skating policy with one leg pushing off |

Switch walking and skating policies on the Web UI's **Motion (运动)** page. The lists are defined by `WALK_AGENTS` / `SKATE_AGENTS` in `src/constants.py`.

## Android App

The app is a wrapper around the robot's Web UI (a WebView loading `http://<robot-ip>:8000`), with local YOLO pose estimation added. Changes to the Web UI do not require rebuilding the app. Prebuilt packages are in `release/` at the repository root.

## License

GNU GPL v3; see [LICENSE](LICENSE). The Noto Sans SC font in `src/fonts/` uses the SIL Open Font License; see `src/fonts/OFL.txt`.
