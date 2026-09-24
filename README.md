# Franka inference on Rome

The deployed Franka FR3 inference and control stack: a 1 kHz libfranka torque controller, synchronous π0.5 action playback, three cameras, Robotiq gripper, selectable model backends, and a tailnet control UI with live plots.

This repository preserves the **Rome-specific setup**. It contains source and deployment files; multi-gigabyte checkpoints, IRs, GPU plugins, virtual environments, camera observations, session logs, and credentials are not checked into Git. [Artifact recovery](docs/ARTIFACTS.md) identifies their exact locations and hashes. Restoring on a fresh machine also requires the pinned dependencies, PREEMPT_RT kernel, Intel GPU runtime, ROS build, and physical hardware configuration. A clean-machine reinstall has not been validated by this publication.

## Current configuration

- Franka FR3 v2.1, firmware 5.10; local libfranka 0.21.3 with the recorded patches.
- Intel Arc B580, Ryzen 5 4500; PREEMPT_RT `6.18.53-spring-rt`.
- Control CPU 5 / FIFO 80; robot Ethernet CPU 4 / FIFO 90; housekeeping 0–3,6–9.
- ZED 2i external left eye, always paired with the ZED Mini wrist left eye. No wrist digital zoom. The selector supports randomly choosing between two external left eyes; `37266581` remains excluded from inference after the earlier shared-USB issue, while `39338267` is the active external camera. The UI streams all three devices independently over continuous MJPEG, with measured capture FPS; simultaneous live capture was verified at about 15 FPS per camera after the USB topology changed.
- Continuous sessions until Pause, Stop, or a fault. Saved selection at the latest audit: **10 actions per inference, nominal 15 Hz**, prompt `put all items on the desk in the bowl`. These are a dated snapshot; the UI can change them. See [current operations](docs/OPERATIONS.md) and the [audit receipt](manifests/setup-audit.json).
- DROID FR3 constraints and hybrid joint/Cartesian impedance. No Ruckig and no table boundary; native rate limiting, 100 Hz filtering, 40 N / 40 Nm native collision thresholds, workspace and joint checks remain. See the [operating handoff](docs/DESMOND_FRANKA_HANDOFF.md).

The selected playback Hz is not the fresh inference rate. The client plays the selected portion of a 15-action chunk, holds for inference, validates the next targets, and repeats. Holds, inference, and validation reduce effective throughput.

| Backend ID | Runtime | Checkpoint | Output contract |
|---|---|---|---|
| `torch_eager` | Torch eager FP16, 5 steps | pi05_droid | normalized joint increments |
| `torch_compile` | torch.compile FP16, 5 steps | pi05_droid | normalized joint increments |
| `openvino` | OpenVINO FP16, 5 steps | pi05_droid | normalized joint increments |
| `ssog_mc2` | native SSOG MC2, INT8/INT4, 2 passes | pi05_droid_jointpos | absolute joint positions |
| `ov_a_w8a8` | custom OpenVINO W8A8, 5 steps, T64/T256 | pi05_droid_jointpos | absolute joint positions |

All backends return absolute gripper closure (0 open, 1 closed). `runtime_contract.py` prevents mixing action decoding or checkpoints. The checkpoint-A adapter is included, with its separate provenance and validation history in [PI05_FINAL_SNAPSHOT.md](docs/PI05_FINAL_SNAPSHOT.md). The historical handoff includes both checkpoint-A backends but predates the latest camera/UI changes. The live application itself remains authoritative for readiness and robot state.

## Layout

| Path | Contents |
|---|---|
| `franka_droid/` | Active arm client, native controller, model adapters, configuration/calibration, web UI |
| `camera/` | Serial-aware camera discovery, V4L2 capture helper, idle preview server |
| `deploy/` | Rome's systemd services, RT/network configuration, ROS launch files, kernel configuration |
| `patches/` | Local changes against exact libfranka / ROS source revisions |
| `manifests/` | Source pins, package inventories, file provenance, artifact receipts, publication status |
| `artifacts/` | Small model-serving sources, launchers, tuning kernels/configuration, checksums |
| `scripts/` | Dependency checkout, native build, artifact recovery, non-actuating deployment helper |
| `docs/` | Setup and operations, historical handoff, artifact provenance |

## Restore / develop

1. Read [SETUP.md](docs/SETUP.md) and [ARTIFACTS.md](docs/ARTIFACTS.md). They describe the fixed Rome paths, shared Python environments, ROS state ownership, and external assets.
2. Fetch pinned source repositories with `python3 scripts/fetch_sources.py --destination /path/to/vendor-sources`. This clones dependencies and applies recorded patches; it does not build or run hardware.
3. On a provisioned Linux machine, build with `FRANKA_PREFIX=/path/to/franka_install bash scripts/build_native.sh /path/to/build-output`.
4. Preview the Rome deployment mapping using `python3 scripts/install_rome.py`. Applying it requires root and `--apply`, rejects active arm/model warm-up work, preserves existing setup/calibration, and leaves services stopped for explicit startup.
5. Use the UI at `http://100.95.186.107:8787/` on the tailnet. Start requires a warmed model and an Execution-ready robot. Do not start a second FCI owner or camera capture process beside an active session.

Publishing this repository did not start, stop, or reconfigure Rome's arm. Offline tests were not run. Publication validation covers source/syntax, build instructions, secret/artifact review, and repository integrity; previously recorded live-motion evidence is described separately in the handoff.

The latest completeness audit compared every recorded deployed file, all 11 pinned dependency revisions and three patches with Rome. [Coverage and verification](docs/COVERAGE.md) explains the receipts, external assets, deliberate exclusions and restore limitations. Recheck drift without touching the running session:

```bash
python3 scripts/audit_rome.py root@100.95.186.107 --output /tmp/rome-audit.json
```

## Operations

- [Current operations and troubleshooting](docs/OPERATIONS.md)
- [Publication coverage and verification](docs/COVERAGE.md)
- [UI behavior and API](franka_droid/CONTROL_UI.md)
- [Detailed operator/developer handoff](docs/DESMOND_FRANKA_HANDOFF.md)
- [Dependencies and deployment](docs/SETUP.md)
- [Model artifacts and recovery](docs/ARTIFACTS.md)
- [Third-party attribution](THIRD_PARTY_NOTICES.md)

Native faults stop the session; no automatic retry is performed. A previous misleading `FCI state timestamp stale` report was traced to a Cartesian reflex: the controller now reports its concrete fault before dumping history, and the client preserves the native cause. Genuine stale feedback and physical reflexes still stop motion.
