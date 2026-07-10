#!/usr/bin/env python3
# One-time utility: strip optimizer/scaler state from a training checkpoint
# to produce a lean inference-only checkpoint, written to the fast native fs.
#
# Usage:
#   python convert_to_inference_checkpoint.py <src.pth> <dst.pth>
import sys
import os
import torch


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_to_inference_checkpoint.py <src.pth> <dst.pth>")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    print(f"[1/3] Loading checkpoint (mmap) from:\n  {src}")
    print(f"  src size: {human(os.path.getsize(src))}")

    # mmap=True avoids materializing the whole blob in RAM during unpickling
    ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)

    print("[2/3] Top-level keys and rough sizes:")
    for k, v in ckpt.items():
        if isinstance(v, dict):
            try:
                nbytes = sum(t.numel() * t.element_size() for t in v.values())
                print(f"  {k:12s} dict[{len(v):4d}] ~ {human(nbytes)}")
            except Exception:
                print(f"  {k:12s} dict[{len(v)}]")
        else:
            print(f"  {k:12s} {type(v).__name__}")

    # Keep ONLY what load_model() actually reads: 'model' and 'args'.
    lean = {"model": ckpt["model"], "args": ckpt["args"]}
    if "epoch" in ckpt:
        lean["epoch"] = ckpt["epoch"]

    print(f"[3/3] Writing inference-only checkpoint to:\n  {dst}")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(lean, dst)
    print(f"  dst size: {human(os.path.getsize(dst))}")
    print("Done. Use this path as --weights / weights_path.")


if __name__ == "__main__":
    main()
