#!/usr/bin/env python3
"""acc_A/s{00..39} -> the c2t64 OpenVINO IR's 7-input safetensors contract.

The IR (`inputs/cams2-tokens64/openvino-dynamic-w8a8`) was traced from
lerobot pi05 `sample_actions(images, masks, tokens, token_mask, noise)` with
C=2 cameras and T=64 tokens, so its ports are, in exported order:

  '2630'      f32 [1,3,224,224]   camera 0   (base_0_rgb)
  '4265'      f32 [1,3,224,224]   camera 1   (left_wrist_0_rgb)
  '315'       bool[1]             img_mask 0
  '335'       bool[1]             img_mask 1
  'tokens'    i64 [1,64]          tokenized prompt (pi0.5 carries the robot
                                  state as discretised tokens inside it;
                                  the architecture has no state_proj)
  'masks'     bool[1,64]          prompt mask
  'input.879' f32 [1,15,32]       flow noise

acc_A carries exactly that: `siglip.in` is [2,3,224,224] already in serving
order with the pad camera dropped (manifest `drop_pad_camera: true`,
`cameras_used = [base_0_rgb, left_wrist_0_rgb]`), `tokens`/`tokens_mask` are
T=64, and `noise` is the harness's fixed per-sample [15,32].

usage: make_accA_safetensors.py <acc_root> <out_dir>
"""
import json
import pathlib
import sys

import numpy as np
from safetensors.numpy import save_file

NAMES = ("2630", "4265", "315", "335", "tokens", "masks", "input.879")


def build(sample_dir):
    images = np.load(sample_dir / "siglip.in.npy")
    assert images.shape == (2, 3, 224, 224) and images.dtype == np.float32, images.shape
    tokens = np.load(sample_dir / "tokens.npy")
    mask = np.load(sample_dir / "tokens_mask.npy")
    noise = np.load(sample_dir / "noise.npy")
    assert tokens.shape == (64,) and mask.shape == (64,) and noise.shape == (15, 32)
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    used = manifest["info"]["cameras_used"]
    assert len(used) == 2, used
    return {
        "2630": np.ascontiguousarray(images[0:1]),
        "4265": np.ascontiguousarray(images[1:2]),
        "315": np.ones((1,), dtype=bool),
        "335": np.ones((1,), dtype=bool),
        "tokens": np.ascontiguousarray(tokens[None].astype(np.int64)),
        "masks": np.ascontiguousarray(mask[None].astype(bool)),
        "input.879": np.ascontiguousarray(noise[None].astype(np.float32)),
    }, manifest


def main():
    acc = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for n in range(40):
        tag = "s%02d" % n
        values, manifest = build(acc / tag)
        path = out / ("accA-%s.safetensors" % tag)
        save_file(values, str(path))
        index[tag] = {
            "file": path.name,
            "trajectory": manifest["trajectory"],
            "control_index": manifest["control_index"],
            "frame": manifest["frame"],
            "noise_seed": manifest["noise_seed"],
            "text_valid": manifest["info"]["text_valid"],
            "cameras_used": manifest["info"]["cameras_used"],
            "drop_pad_camera": manifest["harness"]["configA_spec"]["drop_pad_camera"],
        }
    # A pure-noise control: zero images, the s00 prompt, s00 noise.
    values, _ = build(acc / "s00")
    values["2630"] = np.zeros_like(values["2630"])
    values["4265"] = np.zeros_like(values["4265"])
    save_file(values, str(out / "accA-noiseonly.safetensors"))
    index["noiseonly"] = {"file": "accA-noiseonly.safetensors",
                          "note": "s00 prompt+noise, both camera tensors zeroed"}
    (out / "accA-index.json").write_text(json.dumps(index, indent=2) + "\n")
    print("wrote", len(index), "samples to", out)


if __name__ == "__main__":
    main()
