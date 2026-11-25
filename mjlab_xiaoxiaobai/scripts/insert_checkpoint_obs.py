"""Insert observation inputs inside a checkpoint's actor or critic input, e.g. joints a newer
model adds to a motion command (head_wobble_* in the reference joint state). Like
pad_checkpoint_obs.py, which appends at the end: zero weights, normaliser mean 0 and variance 1,
Adam moments padded, so the network computes exactly what it did.

usage: insert_checkpoint_obs.py <in.pt> <out.pt> --net critic --at 75:2 --at 96:2
       (--at <column in the original input>:<inputs inserted before it>)
"""
import argparse

import torch


def insert(t: torch.Tensor, at: list[tuple[int, int]], fill: float) -> torch.Tensor:
    parts, prev = [], 0
    for col, n in sorted(at):
        parts += [t[:, prev:col], torch.full((t.shape[0], n), fill, dtype=t.dtype)]
        prev = col
    return torch.cat(parts + [t[:, prev:]], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--net", choices=("actor", "critic"), required=True)
    ap.add_argument("--at", action="append", required=True, help="column:count")
    args = ap.parse_args()
    at = [tuple(int(v) for v in a.split(":")) for a in args.at]

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ck[f"{args.net}_state_dict"]
    w = sd["mlp.0.weight"]
    old_shape = tuple(w.shape)
    sd["mlp.0.weight"] = insert(w, at, 0.0)
    for key, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
        sd[key] = insert(sd[key], at, fill)
    for state in ck["optimizer_state_dict"]["state"].values():
        for name in ("exp_avg", "exp_avg_sq"):
            t = state.get(name)
            if t is not None and tuple(t.shape) == old_shape:
                state[name] = insert(t, at, 0.0)
    torch.save(ck, args.dst)
    print(f"{args.net}: inputs {old_shape[1]} -> {sd['mlp.0.weight'].shape[1]} -> {args.dst}")


if __name__ == "__main__":
    main()
