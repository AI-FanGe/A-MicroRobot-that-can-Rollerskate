"""Widen a checkpoint's actor and critic inputs by observation terms appended at the end
(e.g. the stroke clock's gait_phase), so a policy trained without them starts a task that
has them. The new inputs get zero weights, normaliser mean 0 and variance 1: the widened
policy acts exactly as the original until training uses them. Adam moments are padded too,
and the saved env step counter restarts at 0 so the new task's curricula run from the start.

usage: pad_checkpoint_obs.py <in.pt> <out.pt> [extra_inputs=2]
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
extra = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ck = torch.load(src, map_location="cpu", weights_only=False)

padded = {}
for net in ("actor_state_dict", "critic_state_dict"):
    sd = ck[net]
    w = sd["mlp.0.weight"]
    padded[w.shape] = (w.shape[0], w.shape[1] + extra)
    sd["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra, dtype=w.dtype)], dim=1)
    for key, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
        v = sd[key]
        sd[key] = torch.cat([v, torch.full((v.shape[0], extra), fill, dtype=v.dtype)], dim=1)
    print(f"{net}: inputs {w.shape[1]} -> {w.shape[1] + extra}")

for state in ck["optimizer_state_dict"]["state"].values():
    for name in ("exp_avg", "exp_avg_sq"):
        t = state.get(name)
        if t is not None and tuple(t.shape) in padded:
            state[name] = torch.cat([t, torch.zeros(t.shape[0], extra, dtype=t.dtype)], dim=1)

# The runner restores the env step counter from here on resume, which would fire every
# step-based curriculum stage of the new task at once.
env_state = (ck.get("infos") or {}).get("env_state")
if env_state is not None and "common_step_counter" in env_state:
    print(f"common_step_counter {env_state['common_step_counter']} -> 0")
    env_state["common_step_counter"] = 0

torch.save(ck, dst)
print(f"saved {dst} (iter {ck.get('iter')})")
