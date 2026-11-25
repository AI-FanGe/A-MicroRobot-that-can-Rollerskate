# 小小白机器人端代码

运行在主控板上的控制代码，以及配套的头部固件、Web UI 和安卓 App。

| 目录 | 内容 |
| :--- | :--- |
| `src/` | 控制循环、各动作（`src/moves/`）、IMU / 舵机驱动、语音、音乐 |
| `src/agents/` | 部署用的 ONNX 策略和参考动作 |
| `src/api/` | 机器人 HTTP API 与 Web UI 后端（`http://<robot-ip>:8000/`） |
| `src/sim/`, `src/debug/` | MuJoCo 仿真运行与 sim-to-real 调试工具 |
| `src/model/mjcf/` | 机器人 MJCF 模型（网格引用 `../mjlab_xiaoxiaobai` 中的同一份文件，仅仿真 / 调试使用） |
| `webui/` | 浏览器控制面板 |
| `console/` | 电脑端多机器人控制台 |
| `firmware/xiao_face/` | 头部表情屏 / 扬声器 / 摄像头固件 |
| `android/` | 小小白安卓 App |
| `systemd/` | 开机自启服务 |
| `image/` | 树莓派 SD 卡镜像的构建脚本（`make image`） |
| `calibration/` | 舵机零位与关节限位标定 |

## 使用

所有操作都在电脑上通过 `Makefile` 经 SSH 完成，默认机器人主机名为 `xiaoxiaobai`（可用 `HOST=...` 覆盖）。机器人上的代码目录是 `~/xiaoxiaobai`。

下表中省略 `HOST` 的远程命令，要求先按[主教程的 SSH 配置](../README.md#2-配置-ssh)设置 `xiaoxiaobai` 别名（包含机器人的 IP 和用户名）。不配置别名时，请在电脑的本目录下明确指定目标：

```bash
make setup HOST=user@192.168.1.100
make run HOST=user@192.168.1.100
```

把示例 IP 换成机器人实际 IP；小小白镜像的用户名是 `user`，官方系统使用自己设置的用户名。`HOST=...` 只对当前命令生效，每次运行远程命令都要指定。


| 命令 | 作用 |
| :--- | :--- |
| `make setup` | 同步代码并在机器人上安装依赖（`uv sync --frozen`） |
| `make image` | 打包可直接烧录的树莓派镜像（`build/image/xiaoxiaobai-<日期>.img.xz`，需要 sudo） |
| `make run` | 同步代码，在机器人上启动 API + Web UI，并在本机浏览器打开 |
| `make stop` | 停止控制循环并卸力 |
| `make api-enable` | 开机自启 API 服务 |
| `make api-dev` | 在本机以模拟模式运行 API / Web UI（无需硬件） |
| `make sim` | 在 MuJoCo 中运行控制代码 |
| `make firmware-uart` | 编译头部固件并经主控串口烧录 |
| `make app-apk` | 构建安卓 App（需要 JDK 17+ 和 Android SDK） |
| `make shutdown` | 安全关机（断电前务必执行） |

完整命令见 `Makefile`。

## 策略模型

| 文件 | Web UI 名称 | 说明 |
| :--- | :--- | :--- |
| `walk_low_battery_1.onnx` | 低电量步态1 | 默认行走策略，在 5-8 V 供电电压范围内训练 |
| `walk_low_battery_2.onnx` | 低电量步态2 | 在 1 的基础上加入行走中电池掉电的训练，电压下降时更稳 |
| `walk_free_arms.onnx` | （走跑·上肢自由） | 上肢自由模式下的行走策略，只控制腿 |
| `skate_glide.onnx` | 双脚滑行 | 默认滑板策略 |
| `skate_single_leg.onnx` | 单脚蹬滑 | 单腿蹬地的滑板策略 |

步态和滑板策略可以在 Web UI「运动」页切换，列表定义在 `src/constants.py` 的 `WALK_AGENTS` / `SKATE_AGENTS`。

## 安卓 App

App 只是机器人 Web UI 的壳（WebView 加载 `http://<robot-ip>:8000`），外加本地 YOLO 姿态识别，所以 Web UI 的改动不需要重新打包。编译好的安装包在仓库根目录的 `release/`。

## License

GNU GPL v3，见 [LICENSE](LICENSE)。`src/fonts/` 中的 Noto Sans SC 字体使用 SIL Open Font License，见 `src/fonts/OFL.txt`。
