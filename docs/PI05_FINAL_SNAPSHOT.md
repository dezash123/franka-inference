> Historical host snapshot. See the repository README and `manifests/snapshot.json` for the publication snapshot; current live state can differ.

# pi05-final — the two final pi0.5 DROID lines on rome

Installed 2026-09-24. Both lines serve openpi's `pi05_droid_jointpos` (checkpoint A,
absolute joint positions) and both are selectable in the Franka control UI
(http://100.95.186.107:8787) and served by `franka-droid-policy.service` through
`/var/lib/spring-data/workloads/robo_run/franka_droid/serve_policy.py`.

Nothing was compiled on rome. The native binary is the shipped `ServeTune` build from the
campaign (built on the B580 build hosts, sha pinned below); the OpenVINO IRs were exported on
buildfarm and gated on sleepy-joe (OvCkptA). rome only compiles OpenVINO's per-device kernel
cache on first start (`ov-a/runs/serve/cache`, ~43 s per window cold, ~8 s warm).

| line | dir | what runs | UI backend id |
|---|---|---|---|
| native compressed `ssog-A-mc2` | `ssog-a-mc2/` | `fused_mux_droid` mux (T = 64/128/256 resident, MeanCache MC2, 2 passes, W8A8 + W4A8 expert) in the `ssog-b580:neo2513` container | `ssog_mc2` |
| OpenVINO W8A8, checkpoint A | `ov-a/` | `ov_mux_server.py` with the custom GPU plugin, T = 64 + 256 resident, 5 Euler steps inside the graph | `ov_a_w8a8` |

## ssog-a-mc2/

Extracted with the archive's own `install.sh` from
`s3://old-local-compute-archives/jason/2026-09-23/pi05-ssog-a-mc2-mux-t64-t128-t256/`
(local copies in `../pi05-ssog-reproduction/archive/`, `sha256sum -c SHA256SUMS` OK):

| file | sha256 |
|---|---|
| `muxfinal/bin/fused_mux_droid` (74,802,120 B) | `db89756ca6b0451040cbbf6e3490d28606d5ed510de3d4cc371bfce3259d6edb` |
| `campaign13/RuntimeCore/CANONICAL.sha256` | `0376a60ab7c1833c6dcdfa7bd04281a8c1f0a2804e8c45743738a2c5001ff103` |
| base archive `pi05-b580-ssog-a-mc2.tar.zst` | `75faa985d62d03d4a2ffd91310c46be655b211b861fc0045f828de3893511d66` |
| mux overlay `pi05-ssog-a-mc2-mux-overlay.tar.zst` | `fd419f9fa98d306e24a73adefcd8e437418d0edbba8b53afe883c53cea0e3579` |

Weights: `campaign13/share/weights-A/` (INDEX.json, every file re-hashed by the arm adapter at
service start). Launcher: `run_mux_serve.sh` (resolves the B580 by PCI `0000:03:00.0`, so the
card number does not matter). Smoke: `smoke_mux.sh`.

rome, 2026-09-24: `smoke_mux.sh` → `actions.f32` sha256
`195fed903c4c31b9ba3e47350f2d1d946129065b06b3128c6ca9d9a052f81867`, **identical to the
archive's recorded smoke hash** (`archive/TARGET-VERIFICATION.json`).

## ov-a/

Layout is the relocatable `pi05-c2t64-continuation` tree (RomeOv) with the checkpoint-A IRs in
`model/t64` and `model/t256`, `unit/` from the OvCkptA lane, and `serve_ov_a.sh` as the
single-card launcher. `SHA256SUMS.pinned` (all OK on rome):

| file | sha256 |
|---|---|
| `model/t64/pi05_droid_dynamic_w8a8.xml` | `12ce852195a82564c3ef8f4804a1f069a5f2df9e245efe4547e9675b3a8748f8` (byte-identical to the checkpoint-B export) |
| `model/t64/pi05_droid_dynamic_w8a8.bin` (3,776,269,399 B) | `d84fcd2cf4f0f307a2218a0f94b8e546239ae6683894a13ec68af0cd6d0ddb8b` |
| `model/t256/pi05_droid_dynamic_w8a8.xml` | `3895ab861e9c70d544e0d09652fe627e2ed71c28d69ed5a0ee235385722c3f95` |
| `model/t256/pi05_droid_dynamic_w8a8.bin` (3,776,269,591 B) | `785a45d0da322245942ab4f4ad9fe7bb04bf91e42207e66ff65d7dc124bd36da` |
| `custom-plugin-cams-r11/libopenvino_intel_gpu_plugin.so` | `0dac1fd9d724ad025899dbdc6abbcdad50b1144ff7c1656a90b8bea36e493b96` |
| `ovmux/ov_mux_server.py` | `78f971a40313a9e34a3cea84d86e79e42437c20e7f111529a93fe2c77a0e3ea4` |
| `ovmux/c2t64-r11-tuning.json` / `c2t256-r11-tuning.json` | `ae24cc16…` / `45870c3c…` (retargeted into `runs/mux-tuning/` by `serve_local.sh`) |
| `unit/assets_ckptA/norm_stats.json` | `57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7` |
| `unit/assets_ckptA/paligemma_tokenizer.model` | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |
| `unit/unit_ov.py` | `c72f8f704e1ed5acdcf4140367956ee056939c3ae1da2ec756f0048a3fd0ea3c` |

Sources: IRs and `unit/` from `sleepy-joe:/home/spring/ovckptA` (the OvCkptA stage; the IRs
also exist on buildfarm), everything else from rome's own
`robo_run/scratch/pi05-c2t64-continuation` (hash-checked against its SHA256SUMS). The final
S3 bundle's `ov/` holds the checkpoint-**B** IRs; the checkpoint-A IRs are not in S3.

`venv/` = `setup_venv.sh` offline from `wheels/`: OpenVINO 2026.3.1-22476, numpy 2.2.6.
Host runtime is rome's own NEO 26.31 / xe; the bundled Level-Zero in `runtime/` is inert.

Serve by hand: `bash ov-a/serve_ov_a.sh 0` (JSON lines on stdin/stdout, `{"ready":true,...}`
first; `{"cmd":"chunk","episode":..,"dir":<case dir>}` → `actions.f32`). Two workers, one per
window, 4.58 + 4.61 GB VRAM.

### rome verification (2026-09-24, `ov-a/verification/rome/`)

* `ov_mux_client.py fidelity` over the 40 real acc_A frames vs the checkpoint-A bf16 eager
  5-step references: rel-L2 **0.0258 all-32 / 0.0857 task dims**, worst `s15` — the OvCkptA
  §5 row to the digit; **40/40 served chunks byte-identical to sleepy-joe's**
  (`verification/direct-a-t64` vs `verification/rome/direct-a-t64`). Kernel fingerprint
  550 / 107 / 107 / 90 / 35 / 185 / 17 / 0-0-107 / 764 on both windows (the shipped signature).
* `unit_ov.py --checkpoint A gate` (the CPU transform the arm adapter uses): **PASS** —
  tokens exact vs acc_A, noise exact, images within one uint8 step, acc_A bitwise vs the
  server, corridor 0.0230 mean / 0.0272 worst under the 0.05 floor.
* Device ms on rome, b2b, `min_freq` at the host default 1200 (NOT pinned): median 29.62 ms.
  The clock is not pinned by anything here; `serve_ov_a.sh` reads the real floor into the
  ready line's `clock_policy`. To pin: `PI05_CARD=card1 ./pin_clock.sh 2850` (the B580 is
  card1 on rome now; the continuation script's default `card0` is wrong on this box).

## Arm integration (`robo_run/franka_droid`)

* `ssog_policy.py` → `INSTALL = pi05-final/ssog-a-mc2` (unchanged otherwise).
* `ov_a_policy.py` (new) → `pi05-final/ov-a`: same case-directory / JSON-lines pattern as the
  SSOG adapter; `unit_ov.Transform(..., absolute_actions=True)` — quantile Unnormalize on
  actions and state, AbsoluteActions on the 7 joints, absolute gripper; verifies the
  norm_stats / plugin / server hashes on disk and the IR hashes on the server's ready line.
* `runtime_contract.py`: backend `ov_a_w8a8` (W8A8, 5 steps, absolute joint space); every
  absolute-joint backend must serve `pi05_droid_jointpos`.
* `serve_policy.py`: dispatch + `config: pi05_droid_jointpos` for absolute backends.
* `ui/index.html`, `ui/app.js`: dropdown entry and model summary.
* Backup of the edited files: `backups/before-pi05-final-20260924-190116/`.

Through the running service with a recorded real observation
(`stream-policy-1790268664239107706-initial-observation.npz`, 8 calls, warmed):
`ssog_mc2` round trip 24–26 ms (device 19–22 ms), `ov_a_w8a8` 35.4 ms (device+router
31.8 ms); both validate `runtime_contract.validate_runtime` and return finite (15, 8)
absolute chunks. The UI's live-camera warmup could not complete on 2026-09-24 because external
ZED 2i `39338267` (USB hub 3-2.3.1) returns truncated 524,096-byte USB frames (rejections began
14:05 EDT, before this install); once that camera is reseated, press **Load model** in the UI.
