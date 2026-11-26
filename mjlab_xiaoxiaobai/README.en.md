<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md"><strong>English</strong></a>
</p>

# MjLab Xiaoxiaobai

Xiaoxiaobai's reinforcement learning training environment, built on [MjLab](https://github.com/mujocolab/mjlab). It uses the [BAM](https://github.com/Rhoban/bam) actuator model together with domain randomization for sim-to-real transfer.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and an NVIDIA GPU:

```
uv sync
```

## Tasks

| Task ID | Description |
| :--- | :--- |
| `Mjlab-Velocity-Xiaoxiaobai` | Velocity-tracking walking |
| `Mjlab-Velocity-Xiaoxiaobai-FreeArms` | Walking with the arms free to move |
| `Mjlab-Drag-Xiaoxiaobai` | Walking while being led by the hand |
| `Mjlab-Getup-Xiaoxiaobai` / `Mjlab-Rollover-Xiaoxiaobai` | Getting up after a fall / rolling over |
| `Mjlab-Sit-Xiaoxiaobai` / `Mjlab-SitUp-Xiaoxiaobai` | Sitting down / sitting up |
| `Mjlab-PropUp-Xiaoxiaobai` | Propping up from a lying position and lying back down |
| `Mjlab-Skate-Xiaoxiaobai` / `Mjlab-Skate-Stroke-Xiaoxiaobai` | Roller-skating |
| `Mjlab-Dance-Xiaoxiaobai` / `Mjlab-Gymnastics-Xiaoxiaobai` / `Mjlab-Peek-Xiaoxiaobai` | Motion tracking: dance / calisthenics / peeking |

Environment configurations are in `src/mjlab_xiaoxiaobai/tasks/`. The robot model (MJCF + meshes) is in `src/mjlab_xiaoxiaobai/robot/xiaoxiaobai/`.

## Training and Playback

First check the environment with zero or random actions:

```
uv run play Mjlab-Velocity-Xiaoxiaobai --agent zero
```

Train:

```
uv run train Mjlab-Velocity-Xiaoxiaobai --env.scene.num-envs 4096
```

Replay a checkpoint (typically stored at `logs/rsl_rl/<experiment>/<date>/model_<n>.pt`):

```
uv run play Mjlab-Velocity-Xiaoxiaobai --checkpoint-file <checkpoint>
```

## Export to the Robot

Export a walking policy to ONNX directly into the robot code directory:

```
uv run python src/mjlab_xiaoxiaobai/scripts/export_onnx.py --checkpoint <checkpoint> --output ../xiaoxiaobai/src/agents/walk_low_battery_1.onnx
```

For motion-tracking tasks (dance, calisthenics, peeking, and propping up), use `scripts/export_dance.py` to export both the policy and reference motion.

## Motion Data

`data/motions/xiaoxiaobai/` contains reference motions for motion-tracking tasks (50 Hz, already retargeted to Xiaoxiaobai). `scripts/` contains tools for generating this data, including `fetch_g1_motions.py` (download source motions), `retarget_g1_to_xiaoxiaobai.py` (retarget motions), and `gymnastics_motion.py` / `propup_motion.py` / `peek_motion.py` (manually choreographed motions), as well as evaluation and video-recording scripts for each task.

## License

GNU GPL v3; see [LICENSE](LICENSE). Some code originates from [mjlab_microban](https://github.com/MarcDcls/mjlab_microban) and is licensed under the Apache License 2.0; see the original text in [LICENSE.Apache-2.0](LICENSE.Apache-2.0).
