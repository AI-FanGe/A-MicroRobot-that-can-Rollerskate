<p align="center">
  <a href="README.md"><strong>中文</strong></a> | <a href="README.en.md">English</a>
</p>

# MjLab Xiaoxiaobai

小小白的强化学习训练环境，基于 [MjLab](https://github.com/mujocolab/mjlab)。执行器模型使用 [BAM](https://github.com/Rhoban/bam)，配合域随机化实现 sim-to-real。

## 安装

需要 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和一块 NVIDIA GPU：

```
uv sync
```

## 任务

| 任务 ID | 说明 |
| :--- | :--- |
| `Mjlab-Velocity-Xiaoxiaobai` | 速度跟踪行走 |
| `Mjlab-Velocity-Xiaoxiaobai-FreeArms` | 行走，手臂可自由动作 |
| `Mjlab-Drag-Xiaoxiaobai` | 被牵着手行走 |
| `Mjlab-Getup-Xiaoxiaobai` / `Mjlab-Rollover-Xiaoxiaobai` | 摔倒后起身 / 翻身 |
| `Mjlab-Sit-Xiaoxiaobai` / `Mjlab-SitUp-Xiaoxiaobai` | 坐下 / 坐起 |
| `Mjlab-PropUp-Xiaoxiaobai` | 躺姿撑起与躺下 |
| `Mjlab-Skate-Xiaoxiaobai` / `Mjlab-Skate-Stroke-Xiaoxiaobai` | 滑板 |
| `Mjlab-Dance-Xiaoxiaobai` / `Mjlab-Gymnastics-Xiaoxiaobai` / `Mjlab-Peek-Xiaoxiaobai` | 动作跟踪：舞蹈 / 广播体操 / 偷看 |

环境配置在 `src/mjlab_xiaoxiaobai/tasks/`，机器人模型（MJCF + 网格）在 `src/mjlab_xiaoxiaobai/robot/xiaoxiaobai/`。

## 训练与回放

先用零动作或随机动作检查环境：

```
uv run play Mjlab-Velocity-Xiaoxiaobai --agent zero
```

训练：

```
uv run train Mjlab-Velocity-Xiaoxiaobai --env.scene.num-envs 4096
```

回放 checkpoint（一般在 `logs/rsl_rl/<experiment>/<date>/model_<n>.pt`）：

```
uv run play Mjlab-Velocity-Xiaoxiaobai --checkpoint-file <checkpoint>
```

## 导出到机器人

行走策略导出为 ONNX，直接写进机器人仓库：

```
uv run python src/mjlab_xiaoxiaobai/scripts/export_onnx.py --checkpoint <checkpoint> --output ../xiaoxiaobai/src/agents/walk_low_battery_1.onnx
```

动作跟踪类任务（舞蹈、体操、偷看、撑起）用 `scripts/export_dance.py` 同时导出策略和参考动作。

## 动作数据

`data/motions/xiaoxiaobai/` 是动作跟踪任务使用的参考动作（50 Hz，已重定向到小小白）。`scripts/` 下是生成这些数据的工具，例如 `fetch_g1_motions.py`（下载原始动作）、`retarget_g1_to_xiaoxiaobai.py`（重定向）、`gymnastics_motion.py` / `propup_motion.py` / `peek_motion.py`（手工编排动作），以及各任务的评估、录像脚本。

## License

GNU GPL v3，见 [LICENSE](LICENSE)。部分代码源自 [mjlab_microban](https://github.com/MarcDcls/mjlab_microban)，按 Apache License 2.0 授权，原文见 [LICENSE.Apache-2.0](LICENSE.Apache-2.0)。
