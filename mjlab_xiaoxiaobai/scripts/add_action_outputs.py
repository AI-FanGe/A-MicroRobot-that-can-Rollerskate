"""Append zero-initialised action outputs to an rsl_rl actor checkpoint.

The new outputs start at 0 (the old behaviour) with exploration std ``--std``; the actor's
inputs and the critic are unchanged. Adam moments for the grown tensors are zero-padded.

    python scripts/add_action_outputs.py IN.pt OUT.pt --count 1 --std 0.3
"""

import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--std", type=float, default=0.3)
    args = p.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    actor = ck["actor_state_dict"]
    head = max((k for k in actor if k.startswith("mlp.") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    head_bias = head.replace(".weight", ".bias")
    std_key = next(k for k in actor if k.startswith("distribution.") and "std" in k)
    n = actor[head].shape[0]

    for key in (head, head_bias, std_key):
        old = actor[key]
        pad = torch.zeros((args.count, *old.shape[1:]), dtype=old.dtype)
        if key == std_key:
            pad[:] = args.std if "log" not in key else torch.log(torch.tensor(args.std))
        actor[key] = torch.cat([old, pad])

    # Optimizer params follow actor.parameters() order, then the critic's.
    names = [k for k in actor if not k.startswith("obs_normalizer.")]
    opt = ck["optimizer_state_dict"]
    for idx, name in enumerate(names):
        if name in (head, head_bias, std_key) and idx in opt["state"]:
            st = opt["state"][idx]
            for m in ("exp_avg", "exp_avg_sq"):
                old = st[m]
                assert old.shape[0] == n, (name, old.shape)
                st[m] = torch.cat([old, torch.zeros((args.count, *old.shape[1:]), dtype=old.dtype)])

    torch.save(ck, args.dst)
    print(f"{args.src} -> {args.dst}: actions {n} -> {n + args.count} ({head}, {std_key})")


if __name__ == "__main__":
    main()
