"""PPO that also keeps some action dimensions close to a frozen teacher actor.

After each PPO update, a few behaviour-cloning steps on the rollout's actor observations pull the
policy's mean action on ``XIAOXIAOBAI_TEACHER_JOINTS`` toward the teacher's (``XIAOXIAOBAI_TEACHER``,
an rsl_rl checkpoint with the same actor observations and actions). The cloning has its own Adam,
so it neither shares PPO's moments nor its adaptive learning rate. Comparing mean actions keeps the
exploration noise out of it, which a reward on the sampled actions would penalise.
With ``XIAOXIAOBAI_TEACHER_WALK_ONLY=1`` only samples with a moving command (the last three actor
observations) are cloned, leaving the standing arms free.
``XIAOXIAOBAI_TEACHER_LEG_LR=<lr>`` also clones ``XIAOXIAOBAI_TEACHER_LEG_JOINTS`` (default: all but
the arms) with an Adam of its own at that learning rate, a soft prior the reward can override.
Adam steps ~lr whatever the loss scale, so a weight on the legs in the arms' loss does not
soften it: at weight 0.2 with the arms' 3e-4 the legs stayed within 1 deg of the teacher's and
the fine-tune learned nothing of its low supply voltages.
With ``XIAOXIAOBAI_TEACHER_LEG_VIN_MIN=<V>`` the legs are cloned only on samples from envs supplied
at V or more at the update (the battery may run down mid-episode): the teacher sank and fell at
the low end, so there the reward alone decides.
"""

from __future__ import annotations

import os
import re

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict


def _load_actor(path: str, device: str) -> tuple[torch.nn.Sequential, torch.Tensor, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    weights = sorted((k for k in state if k.startswith("mlp.") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    layers: list[torch.nn.Module] = []
    for i, k in enumerate(weights):
        lin = torch.nn.Linear(state[k].shape[1], state[k].shape[0])
        lin.weight.data, lin.bias.data = state[k].clone(), state[k.replace("weight", "bias")].clone()
        layers += [lin] if i == len(weights) - 1 else [lin, torch.nn.ELU()]
    mlp = torch.nn.Sequential(*layers).to(device).eval()
    for p in mlp.parameters():
        p.requires_grad_(False)
    return mlp, state["obs_normalizer._mean"], state["obs_normalizer._std"]


class TeacherArmsPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher, self._t_mean, self._t_std = _load_actor(os.environ["XIAOXIAOBAI_TEACHER"], self.device)
        env = self.symmetry["_env"].unwrapped
        names = env.action_manager.get_term("joint_pos")._target_names
        pattern = os.environ.get("XIAOXIAOBAI_TEACHER_JOINTS", r".*(shoulder_pitch|elbow).*")
        self._cols = torch.tensor([i for i, n in enumerate(names) if re.match(pattern, n)], device=self.device)
        leg_pattern = os.environ.get("XIAOXIAOBAI_TEACHER_LEG_JOINTS", r"^(?!.*(shoulder|elbow)).*$")
        self._leg_cols = torch.tensor([i for i, n in enumerate(names) if re.match(leg_pattern, n)], device=self.device)
        self._lr = float(os.environ.get("XIAOXIAOBAI_TEACHER_LR", 3e-4))
        self._steps = int(os.environ.get("XIAOXIAOBAI_TEACHER_STEPS", 4))
        self._walk_only = os.environ.get("XIAOXIAOBAI_TEACHER_WALK_ONLY") == "1"
        self._bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._lr)
        leg_lr = os.environ.get("XIAOXIAOBAI_TEACHER_LEG_LR")
        self._leg_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(leg_lr)) if leg_lr else None
        self._vin_acts = [a for a in env.scene["robot"].actuators if getattr(a, "vin_tensor", None) is not None]
        leg_vin_min = os.environ.get("XIAOXIAOBAI_TEACHER_LEG_VIN_MIN")
        self._leg_vin_min = float(leg_vin_min) if self._leg_optimizer and leg_vin_min else None
        self._leg_envs = self._leg_env_mask()
        print(f"[TeacherArmsPPO] teacher {os.environ['XIAOXIAOBAI_TEACHER']} on {[names[i] for i in self._cols.tolist()]}")
        if self._leg_optimizer:
            share = "" if self._leg_envs is None else f", {self._leg_envs.float().mean().item():.0%} of the envs"
            print(f"[TeacherArmsPPO] legs at lr {leg_lr} on {[names[i] for i in self._leg_cols.tolist()]}{share}")

    def _leg_env_mask(self) -> torch.Tensor | None:
        if self._leg_vin_min is None:
            return None
        return torch.stack([a.vin_tensor[:, 0] for a in self._vin_acts]).min(dim=0).values >= self._leg_vin_min

    def update(self) -> dict[str, float]:
        self._leg_envs = self._leg_env_mask()
        obs = self.storage.observations["actor"].flatten(0, 1).clone()
        num_envs = self.storage.observations["actor"].shape[1]
        env_ids = torch.arange(obs.shape[0], device=self.device) % num_envs
        if self._walk_only:
            moving = obs[:, -3:-1].norm(dim=1) + obs[:, -1].abs() >= 0.01
            obs, env_ids = obs[moving], env_ids[moving]
        loss_dict = super().update()
        batch = obs.shape[0] // self.num_mini_batches
        with torch.no_grad():
            target = self._teacher((obs - self._t_mean) / (self._t_std + 1e-2))
        losses = {"teacher_arms": (self._cols, self._bc_optimizer, torch.arange(obs.shape[0], device=self.device))}
        if self._leg_optimizer:
            leg_ids = torch.arange(obs.shape[0], device=self.device)
            if self._leg_envs is not None:
                leg_ids = leg_ids[self._leg_envs[env_ids]]
            losses["teacher_legs"] = (self._leg_cols, self._leg_optimizer, leg_ids)
        for name, (cols, optimizer, pool) in losses.items():
            total = 0.0
            for _ in range(self._steps):
                ids = pool[torch.randint(pool.shape[0], (batch,), device=self.device)]
                mean = self.actor(TensorDict({"actor": obs[ids]}, batch_size=[batch]))[:, cols]
                loss = torch.square(mean - target[ids][:, cols]).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                optimizer.step()
                total += loss.item()
            loss_dict[name] = total / self._steps
        return loss_dict
