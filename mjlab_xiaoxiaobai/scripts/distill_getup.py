"""Distil the roll-over + get-up hand-over (xiaoxiaobai GetupMove) into one get-up policy.

Teacher: from face down (or upright) the get-up policy; otherwise the roll-over policy until
the robot lies face down and still (HANDOVER_*), then the get-up policy; back to the
roll-over policy if it ends up on its back again (the hand-over alone never switches back).
Student: the get-up actor (same observations, actions and ONNX layout), started from the
get-up checkpoint and trained by DAgger: rollouts mix teacher and student actions, shifting
to the student, and every visited state is labelled with the teacher's action.

usage: distill_getup.py <rollover_ckpt> <getup_ckpt> <run_name> [iterations]
Checkpoints go to logs/rsl_rl/mjlab_xiaoxiaobai_getup/<date>_<run_name>/model_<i>.pt and load
like any get-up checkpoint (getup_eval.py, getup_chain_eval.py, export_onnx.py).
"""
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_rollover_env_cfg import HANDOVER_HOLD_S, HANDOVER_MAX_ANG_VEL, HANDOVER_MIN_GX

TASK = "Mjlab-Getup-Xiaoxiaobai"
NUM_ENVS = 4096
STEPS = 50                # env steps per iteration
BUFFER_ITERS = 10         # iterations of samples kept
EPOCHS = 2
BATCH = 16384
LR = 1e-4
BETA_ITERS = 150          # teacher share of the rollout actions goes 1 -> 0 over these
SAVE_EVERY = 100
BACK_GX = -0.5            # on the back again ...
BACK_HOLD_S = 0.5         # ... for this long: back to the roll-over

rollover_ckpt, getup_ckpt, run_name = sys.argv[1:4]
iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 600

cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = NUM_ENVS
cfg.episode_length_s = 15.0
cfg.curriculum = {}
del cfg.events["getup_assist"]
cfg.events["reset_getup"].params.update({"flat_fraction": 0.7, "supine_fraction": 0.6})
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped


def runner_from(ckpt):
    # The runner pops keys from its config, so each gets a fresh one.
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    return runner


rollover = runner_from(rollover_ckpt).get_inference_policy("cuda:0")
getup = runner_from(getup_ckpt).get_inference_policy("cuda:0")
student_runner = runner_from(getup_ckpt)
actor = student_runner.alg.actor
actor.eval()


def student(x: torch.Tensor) -> torch.Tensor:
    # The get-up's observation normalizer stays frozen; only the MLP is trained.
    return actor.mlp(actor.obs_normalizer(x))


obs = env.get_observations()
with torch.no_grad():
    assert torch.allclose(student(obs["actor"]), getup(obs), atol=1e-5)
opt = torch.optim.Adam(actor.mlp.parameters(), lr=LR)

out = Path("logs/rsl_rl/mjlab_xiaoxiaobai_getup") / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{run_name}"
out.mkdir(parents=True)
robot = u.scene["robot"]
n, dev = u.num_envs, u.device
hold, back_hold = round(HANDOVER_HOLD_S / u.step_dt), round(BACK_HOLD_S / u.step_dt)
rolling = torch.zeros(n, dtype=torch.bool, device=dev)
ticks = torch.zeros(n, dtype=torch.long, device=dev)
back_ticks = torch.zeros(n, dtype=torch.long, device=dev)
obs_dim, act_dim = obs["actor"].shape[1], u.action_manager.total_action_dim
buf_x = torch.zeros(BUFFER_ITERS * STEPS * n, obs_dim, device=dev)
buf_y = torch.zeros(BUFFER_ITERS * STEPS * n, act_dim, device=dev)
filled, cursor = 0, 0
t0 = time.time()
for it in range(1, iterations + 1):
    beta = max(0.0, 1.0 - (it - 1) / BETA_ITERS)
    stats = {"rolling": 0.0, "standing": 0.0}
    for _ in range(STEPS):
        g = robot.data.projected_gravity_b
        fresh = u.episode_length_buf <= 0
        start_roll = (g[:, 0] <= HANDOVER_MIN_GX) & (g[:, 2] >= -0.8)
        rolling = torch.where(fresh, start_roll, rolling)
        ticks[fresh], back_ticks[fresh] = 0, 0
        ready = (g[:, 0] > HANDOVER_MIN_GX) & (robot.data.root_link_ang_vel_b.norm(dim=-1) < HANDOVER_MAX_ANG_VEL)
        ticks = torch.where(rolling & ready, ticks + 1, 0)
        back_ticks = torch.where(~rolling & (g[:, 0] < BACK_GX), back_ticks + 1, 0)
        rolling = (rolling & (ticks < hold)) | (~rolling & (back_ticks >= back_hold))
        ticks[~rolling], back_ticks[rolling] = 0, 0
        with torch.no_grad():
            x = obs["actor"]
            teacher = torch.where(rolling.unsqueeze(-1), rollover(obs), getup(obs))
            mine = student(x)
            use_teacher = (torch.rand(n, device=dev) < beta).unsqueeze(-1)
            action = torch.where(use_teacher, teacher, mine)
        buf_x[cursor:cursor + n], buf_y[cursor:cursor + n] = x, teacher
        cursor = (cursor + n) % buf_x.shape[0]
        filled = min(filled + n, buf_x.shape[0])
        obs, *_ = env.step(action)
        stats["rolling"] += rolling.float().mean().item() / STEPS
        h = robot.data.root_link_pos_w[:, 2] - u.scene.env_origins[:, 2]
        stats["standing"] += ((h > 0.16) & (robot.data.projected_gravity_b[:, 2] < -0.95)).float().mean().item() / STEPS

    loss_sum, batches = 0.0, 0
    for _ in range(EPOCHS):
        perm = torch.randperm(filled, device=dev)
        for i in range(0, filled, BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.mean(torch.square(student(buf_x[idx]) - buf_y[idx]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.0)
            opt.step()
            loss_sum, batches = loss_sum + loss.item(), batches + 1
    if it % 10 == 0 or it == 1:
        print(f"it {it:4d} | beta {beta:.2f} | loss {loss_sum / batches:.4f} | rolling {stats['rolling']:.2f}"
              f" | standing {stats['standing']:.2f} | {time.time() - t0:.0f} s", flush=True)
    if it % SAVE_EVERY == 0 or it == iterations:
        # The runner's own save() needs its training logger, which only learn() sets up.
        torch.save({**student_runner.alg.save(), "iter": it, "infos": None}, out / f"model_{it}.pt")
print(f"saved to {out}", flush=True)
