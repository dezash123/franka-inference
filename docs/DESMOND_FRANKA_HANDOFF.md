> Historical host snapshot. See the repository README and `manifests/snapshot.json` for the publication snapshot; current live state can differ.

# Desmond / Rome Franka handoff

Prepared 2026-09-24 from live Rome services, configuration, source, and session receipts. This is a snapshot, not an instruction to restart motion. Recheck `/api/status` before acting. No credentials are included.

## Current state

At the handoff check (2026-09-24, approximately 17:07 UTC / 13:07 EDT):

- **Arm stopped by the user; robot in Programming mode.** No reported robot errors. Brakes unlocked; FCI enabled. The UI reports `stopped`, not a controller failure.
- UI and inference services are active; selected model is loaded and ready. The idle camera viewer is active.
- Prompt: **`pick up the markers and put them in the bowl`**.
- Backend: **`ssog_mc2`**, `pi05_droid_jointpos`, W8A8/W4A8, 2 model passes.
- Generated horizon: 15 actions. Selected playback: **10 actions per inference at nominal 60 Hz**.
- Inference is **synchronous chunk refill**. The arm holds for fresh inference after the selected actions, or earlier if a target is rejected. There is no session-duration timer in the UI.
- Wrist: ZED Mini, left eye, **1.0× / no digital zoom**. External view: random selection between the two external cameras' **left eyes** on each inference. Wrist is always included.

Latest run: `evidence/stream-policy-1790268664239107706.json`:

| Result | Value |
|---|---:|
| Duration including shutdown | 712.213 s (11 min 52 s) |
| Dispatched actions | 16,550 |
| Generated chunks / inferences | 2,084 |
| Effective average action rate | 23.24 actions/s |
| Effective average inference rate | 2.93 inferences/s |
| Joint-target holds | 562 |
| Camera / validation / workspace holds | 0 / 0 / 0 |
| Minimum native command success during motion | 1.0 (100%) |
| Maximum observed joint speed | 1.110 rad/s |
| Exit | User stop; controller exit 0; `controller_complete=true` |
| Finished | 2026-09-24 17:02:59 UTC |

The receipt's `failure: "Stop requested"` and absent/false task `complete` represent an intentional interruption; they do not mean that this run hit a native fault. Task success (actually accomplishing the pick/place task) has **not** been established by these counters. The 562 target holds are real rejected targets, not completed actions.

The last 32 model calls in this run had median **37.595 ms**, p95 **40.952 ms**; latest model call **37.875 ms**, latest client round trip **42.205 ms**. These are recent samples, not a full-run latency distribution.

## Access and main locations

- Tailscale host: `rome`, `100.95.186.107`.
- Current OS hostname: `spring-edge-20260910003703`. The RT setup script validates this hostname; do not casually rename it.
- SSH used in this work: `ssh root@100.95.186.107`.
- This file: **`/root/DESMOND_FRANKA_HANDOFF.md`**, i.e. `~` for the root SSH login.
- Runtime user: `spring`, UID 1000, home `/home/spring`.
- UI: **http://100.95.186.107:8787/**, tailnet-bound.
- Robot: `172.16.0.2`, Franka Research 3 v2.1, serial `309969-2662311`, firmware 5.10.0.
- Active arm workspace: **`/var/lib/spring-data/workloads/robo_run/franka_droid`**.
- Convenient alias: `/home/spring/robo_run/franka_droid` (symlink to the active workspace).
- OpenPI source: `/home/spring/robo_run/openpi`.
- FP16 checkpoint: `/home/spring/robo_run/checkpoints/pi05_droid_pytorch`.
- libfranka 0.21.3 install: `/var/lib/spring-data/workloads/robo_run/vendor/franka_install`.
- Robot network/ROS setup: `/var/lib/spring-data/franka-network-20260923`.
- Idle camera viewer source: `/home/spring/zed_camera_setup`; HTTP loopback port 8765.
- Final model lines (2026-09-24): `/var/lib/spring-data/workloads/pi05-final/` — `ssog-a-mc2/` (native mux, backend `ssog_mc2`) and `ov-a/` (OpenVINO W8A8 checkpoint A, backend `ov_a_w8a8`); see its `README.md`. The earlier `pi05-ssog-reproduction` workspace is kept for provenance but no longer serves the arm.

For Franka Desk from a client machine, establish a loopback-only tunnel:

```bash
ssh -N -L 127.0.0.1:17443:172.16.0.2:443 root@100.95.186.107
```

Then open `https://127.0.0.1:17443/` (settings: `/admin`). Use the existing authorized Desk login; no credentials are recorded here. Only create the tunnel if that local port is free.

## Services and ownership

| Service | Scope | Purpose / verified state |
|---|---|---|
| `franka-control-ui.service` | root systemd | Python web panel on 8787; active |
| `franka-droid-policy.service` | spring user systemd | Selected policy WebSocket server on 127.0.0.1:8000; active |
| `zed-camera-viewer.service` | spring user systemd | Idle camera previews on 127.0.0.1:8765; active while stopped |
| `franka-ros-moveit.service` | root systemd, process user spring | MoveIt state/path validity; active |
| `franka-ros-state.service` | root systemd | ROS hardware state owner; **inactive for direct FCI operation** |
| `franka-droid-run-<timestamp>.service` | transient root systemd, `franka.slice` | A live arm session launched by the UI / launcher |

The policy server alone does not actuate the arm. Its metadata says `robot_commands_enabled=false` because motion belongs to the separate arm client/controller. This is expected even during real arm execution.

**Only one process may own the robot FCI connection.** The direct launcher refuses to run while the ROS hardware-state service is active. Do not start a standalone FCI reader or warm-up reader during arm control. Do not blindly restart MoveIt: its unit declares a dependency on the ROS state service and may bring that FCI owner back. Check the live arrangement first.

The idle viewer is stopped while the arm client or model warm-up owns the cameras and restored on cleanup. Do not open competing V4L2 streams to diagnose an active session.

**Restarting `franka-control-ui.service` stops its owned arm client or warm-up process.** Stop motion cleanly before deploying changes that require a UI service restart. A service restart restores the last completed session for display; it never resumes motion automatically.

Read-only commands from a root shell on Rome:

```bash
cd /var/lib/spring-data/workloads/robo_run/franka_droid
systemctl status franka-control-ui.service --no-pager
runuser -u spring -- env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user status franka-droid-policy.service --no-pager
runuser -u spring -- env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user status zed-camera-viewer.service --no-pager
python3 /var/lib/spring-data/franka-network-20260923/read-franka.py
curl -fsS http://100.95.186.107:8787/api/status | python3 -c 'import json,sys; s=json.load(sys.stdin); s.pop("csrf",None); print(json.dumps(s,indent=2))'
```

The Desk status reader is the usual non-FCI status check. Avoid publishing the `/api/status` CSRF token in logs or handoff material.

## Normal operation

1. Recheck the actual workspace and robot state. The current handoff state is **Programming**, so an operator must put it into Execution before arm motion can start. Resolve any real contact, cable tension, or native fault first.
2. Use the UI to edit prompt, model, action count, and playback Hz while stopped. Save/load, and wait for readiness. Backend switches can trigger model loading and live-camera warm-up.
3. Press **Start** to run continuously. Runtime readiness must match the current server PID and backend; robot readiness requires Execution, unlocked brakes, enabled FCI, and no reported errors.
4. **Pause** holds the arm and gripper. **Play** discards the remaining chunk and takes a fresh observation. **Stop** terminates the session. Faults terminate execution rather than automatically retrying.

Changing `Playback Hz` changes nominal **action dispatch timing**, not the frequency of fresh model calls. At 60 Hz, 10 actions would nominally take 167 ms, followed by inference and validation overhead. The measured delivery rate can be substantially lower. Camera streams remain 15 fps. Input accepts any finite positive Hz; there is no 5–30 Hz clamp. This is not a promise that arbitrary physical playback rates can be achieved.

Actions per inference is bounded 1–15 because the model generates 15 actions. With the current value 10, the remaining 5 are discarded. Rejected targets, Pause/Play, and some holds can cause an earlier refill.

For an explicitly authorized CLI run while the UI is idle and the robot is ready:

```bash
cd /var/lib/spring-data/workloads/robo_run/franka_droid
./run-policy-stream.sh --execute-confirmed --duration 0
```

`--duration 0` means continuous. A positive duration is an explicit timed-run override; `--horizon`, `--action-hz`, and `--prompt` can override settings for that invocation. Prefer the UI for ordinary runs so its ownership and controls remain coherent. Do not launch a CLI session alongside a UI session.

## Model contracts: do not conflate the runtimes

`runtime_contract.py` is the authoritative runtime/precision/action-space contract. The client checks server metadata before motion.

| Backend | Checkpoint / precision | Sampler | Client action interpretation |
|---|---|---|---|
| `torch_eager` | pi05_droid / FP16 | 5 denoise steps | Normalized joint increment + absolute gripper closure |
| `torch_compile` | pi05_droid / FP16 | 5 denoise steps | Same normalized-increment contract |
| `openvino` | pi05_droid / FP16 | 5 denoise steps | Same normalized-increment contract |
| `ssog_mc2` | pi05_droid_jointpos / W8A8 + W4A8 | Fixed MC2, 2 forwards on a 10-node integration grid | Absolute joint targets in radians + absolute gripper closure |
| `ov_a_w8a8` | pi05_droid_jointpos / W8A8 (custom OpenVINO GPU plugin, T = 64 + 256 resident) | 5 Euler steps inside the graph | Absolute joint targets in radians + absolute gripper closure (same transform family as SSOG: quantile Unnormalize + AbsoluteActions) |

For the normalized path, `delta = 0.2 * action[:7] / max(1, max(abs(action[:7])))`, then target = measured joints + delta. **Do not apply this decoding to SSOG.** The SSOG adapter applies its own checkpoint transforms and outputs absolute joint positions. Its target must remain within the same existing 0.2-rad increment check and joint limits.

Gripper closure is `0=open`, `1=closed`. Model position dimensions beyond the first seven joints and one gripper value are not robot commands.

SSOG details:

- Requested archive: `s3://old-local-compute-archives/jason/2026-09-23/pi05-ssog-a-mc2-mux-t64-t128-t256/`.
- Installed at `/var/lib/spring-data/workloads/pi05-final/ssog-a-mc2` (moved 2026-09-24 from `pi05-ssog-reproduction/install`; same archive, re-extracted and re-verified, smoke hash identical).
- Adapter: `ssog_policy.py`; resident mux is launched through the archive's Docker launcher, `install/run_mux_serve.sh`.
- Binary: `install/muxfinal/bin/fused_mux_droid`, expected SHA-256 `db89756ca6b0451040cbbf6e3490d28606d5ed510de3d4cc371bfce3259d6edb`.
- W8A8 SigLIP/prefix/flow, W4A8 expert, 50% visual pruning, checkpoint quantile normalization, tokenizer, and BF16 embedding table.
- Selects the smallest 64/128/256 text window that fits the complete prompt and state. Overlong input is rejected rather than silently truncated.
- Visual-pruning history resets when session, prompt, or external view changes.
- Archive and smoke-output hashes were verified. The archive's manifest does not match every included source file and omits some mux build files: exact shipped-binary replay is supported; a source-identical rebuild from the archive alone is not established.
- Prior reproduction: 100 fresh-episode samples/window gave resident-server medians 17.714 / 20.687 / 25.585 ms for 64 / 128 / 256 tokens. This boundary includes NPY I/O, conversions, transfers, graph execution, and action-file writing; it excludes camera acquisition and the CPU frontend. These are **historical benchmark measurements**, not the current full arm-loop latency.
- Reproduction proof: `/var/lib/spring-data/workloads/pi05-ssog-reproduction/results/integration-summary.json`.

`SSOG_SETUP.md` and the historical integration summary still contain the original eight-action, Programming-mode, pre-motion snapshot. They are useful for provenance, but **their old selected settings and “motion awaits Execution” statements are superseded by the current configuration and later session receipts**.

The user asked to keep 5 denoise steps on the FP16 paths. SSOG's 2-pass configuration is a different archived recipe, not a five-step FP16 implementation. A low warmed model latency alone does not establish 30 Hz fresh inference while also playing a full chunk synchronously.

## Cameras and gripper

**Open hardware issue (2026-09-24 ~14:05 EDT onward):** external ZED 2i `39338267` on USB hub `3-2.3.1` returns only truncated 524,096-byte USB frames (V4L2 error flag); the idle viewer shows 0 fps and a climbing `rejected_frames` count, and the UI live-camera warm-up fails with `No fresh camera frame` for every backend. Re-enumerating the device and its hub over sysfs did not help; reseat the cable/hub. `37266581` and the wrist ZED Mini are healthy.

| Role | Device / serial | Stream |
|---|---|---|
| External 1 | ZED 2i `39338267` | Left eye, 672×376, 15 fps |
| External 2 | ZED 2i `37266581` | Left eye, 672×376, 15 fps |
| Wrist | ZED Mini / `ZED-M` `12607895` | Left eye, 1280×720, 15 fps, zero rotation, digital zoom 1.0 |

Each inference independently draws one external camera uniformly; both left eyes are available, but only the selected external image plus wrist enter that observation. Do not re-enable random right-eye selection without a new user request.

The wrist ZED replaced the D405. Its device discovery pairs the USB video device with the serial-bearing HID sibling on the same physical route; `/dev/videoN` numbering is not stable. Factory calibration files are in `config/calibration/SN<serial>.conf`; `cameras.py` handles rectification and source selection. `camera_crop.py` remains available, but the user explicitly removed the 1.25× wrist zoom.

Gripper: Robotiq 2F-85, Modbus RTU, 115200 baud, slave 9, FTDI serial `BG04SJCN`:

```text
/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG04SJCN-if00-port0
```

Open/closed position registers are 3/230; maximum width is 0.085 m. USB disconnections previously caused gripper and camera failures; the FTDI adapter is present at this handoff. Startup handles the known communication-timeout/activation recovery; cleanup still closes cameras/ROS even if gripper cleanup fails. Do not treat a disconnected physical adapter as an inference problem.

Camera freshness: maximum age 250 ms; external/wrist skew maximum 100 ms. Existing capture holds/checks stay enabled. The idle viewer status is `http://127.0.0.1:8765/api/status` on Rome; production UI previews use captured observations during motion instead of opening another camera stream.

## Controller and active constraints

This is a **local direct-libfranka torque controller**, not the complete standard DROID/Polymetis transport. DROID FR3 limit values and hybrid feedback are ported locally, with a live FR3 firmware model and additional validation/watchdogs.

Active source: `stream_fci.cpp`, `droid_stream_common.hpp`, `droid_constraints.hpp`; profile: `config/droid-fr3-active.json`.

| Constraint | Current value |
|---|---|
| Joint velocity J1–J4 | 2.075 rad/s |
| Joint velocity J5–J7 | 2.51 rad/s |
| Elbow velocity | 2.075 rad/s |
| Additional Ruckig speed/acceleration/jerk caps | **None; Ruckig removed** |
| Native rate limiter / filter | Enabled / 100 Hz |
| Native Cartesian collision thresholds | 40 N on force axes; 40 Nm on torque axes |
| Native external joint collision thresholds | 40 Nm per joint |
| Commanded torque caps | J1–J4 86 Nm; J5–J7 11.5 Nm |
| Maximum target increment | 0.2 rad; target/state checks retained |
| Workspace | Base-frame box [-1, +1] m on each axis |
| Table / fingertip boundary | **Removed; no active table boundary** |

Joint position bounds (radians):

```text
lower = [-2.65, -1.68, -2.80, -2.95, -2.70, 0.45, -2.90]
upper = [ 2.65,  1.68,  2.80, -0.16,  2.70, 4.40,  2.90]
```

Protective feedback starts within joint-position margin 0.2 rad, Cartesian margin 0.05 m, and joint-velocity margin 0.5 rad/s. The recorded 215 mm flange-to-fingertip measurement is metadata only; `used_as_motion_boundary=false`. The older 500 mm sphere, gentle caps, application force guard, and historical table boundary are not the active profile.

Hybrid feedback: `(Kq + Jᵀ Kx J)(target - q) - (Kqd + Jᵀ Kxd J)dq + coriolis`, with native gravity compensation:

```text
Kq  = [40, 30, 50, 25, 35, 25, 10]
Kqd = [ 4,  6,  5,  5,  3,  2,  1]
Kx  = [400, 400, 400, 15, 15, 15]
Kxd = [ 37,  37,  37,  2,  2,  2]
```

Payload: 0.9 kg, COM `[0,0,0.057]` m; diagonal inertia `[0.002768,0.003149,0.000564]`. Desk end-effector mass must not double-count the load. The controller checks load/transform readback before motion.

MoveIt self-collision checks and native target-workspace validation remain active. Joint target rejection holds position and discards/refills the chunk. Validation rechecks a moving starting state; after repeated mismatch it holds until settled, with a two-second bound. **Matching DROID's limit numbers does not make the entire implementation identical to DROID.**

Reference commits recorded in the active profile: DROID `33ae6a67274f36d2e29525b86f23a56616ef43a7`, fairo `0a01a7fa7a7c65b2f9a3aebf5e79040940daf9d2`.

## Real-time scheduling and watchdogs

- Kernel: **6.18.53-spring-rt**, verified `CONFIG_PREEMPT_RT=y`.
- CPU: AMD Ryzen 5 4500, 6 cores / 12 logical CPUs; inference GPU Intel Arc B580.
- Control thread: **CPU 5, SCHED_FIFO priority 80, locked memory**.
- Robot Ethernet IRQ/threaded NAPI: CPU 4, FIFO priority 90.
- Housekeeping/model/UI/I/O: CPUs **0–3,6–9**. SMT siblings 10 and 11 are excluded from ordinary work.
- Isolation is through affinity/cgroups and IRQ configuration. Do not describe it as boot-time `isolcpus`/`nohz_full` isolation.
- `/etc/franka/realtime.json`, `/etc/franka/realtime.env`, `/usr/local/sbin/franka-realtime-setup`, `/run/franka/realtime-applied.json` describe/apply the layout. The launcher reapplies it before a session. The setup script refuses to tune an active controller.
- UI graphs read logs on housekeeping CPUs. No plot work was added to the 1 kHz torque callback.

Preserved timing checks: native command watchdog 350 ms; control-cycle gap 10 ms; client feedback/timestamp freshness 150 ms; inference round-trip deadline 250 ms. Cameras have their separate 250 ms age / 100 ms skew checks.

Low-Hz playback uses `KEEPALIVE` about every 100 ms while waiting, alongside feedback/camera/gripper checks. It refreshes command liveness without advancing an action or changing the validated target. Do not remove the watchdog to support low frequencies. `stream_fci` must advertise `command_keepalive_supported=true` for the current client.

The last run's active control thread was verified on CPU 5 at FIFO 80 with ~86 MiB locked memory; its I/O thread was SCHED_OTHER on 0–3,6–9. The control thread is named `franka-control` and the I/O thread `franka-io`, so `pgrep -x stream_fci` alone can miss the renamed process. Inspect `/proc/*/exe` or the transient service/cgroup when checking ownership.

## The stale-timestamp incident and deployed fix

Preceding run `stream-policy-1790268166784479081.json` stopped after about 206.4 s. The UI said `FCI state timestamp stale`, but matching stderr and the native history showed:

```text
libfranka: Move command aborted: motion aborted by reflex! ["cartesian_reflex"]
control_command_success_rate: 1
```

The first fault record had stiffness-frame/tool-Z force **40.336 N** against a **40 N** threshold, with Cartesian collision flag on Z. This was evidence of a native force-triggered reflex, not packet-loss evidence. Do not infer a specific table contact solely from the message: the user had corrected an earlier table-strike claim, and later confirmed the workspace clear before this run was resumed.

The reporting race was that the native process dumped historical diagnostics before sending its failure record. The Python client could time out waiting for fresh control state during that dump, masking the real fault.

Deployed changes:

1. Native code emits a JSON-escaped concrete error immediately after the control worker exits, **before** the history dump.
2. Python feedback handling latches the first native error before log writes. `control_error_record` remains diagnostic history, never a live policy observation.
3. Final receipt reconciles generic exit/stale-feedback symptoms with a recorded controller error, preserving the original symptom as `client_failure` and adding `controller_error` / `controller_fault` when available.
4. The UI resolves equivalent legacy receipt messages through the matching stderr, so the older failed session also displays `cartesian_reflex` correctly.

This fixes error attribution. It does **not** remove the 40 N reflex, establish that contact can never recur, or disable genuine stale-feedback stops. A future stale error with no recorded native cause needs an actual timing/transport investigation; do not blanket-suppress it or reuse this historical diagnosis.

The operator confirmed clearance, motion resumed, and the subsequent 712-second run ended cleanly on user Stop. No intentional collision/fault was induced to test error handling. The native build, Python/JS syntax, legacy fault display, and live arm/UI behavior were verified; **no offline test suite was run**.

## Frontend, telemetry, and evidence

Files: `control_ui.py`, `live_telemetry.py`, `ui/index.html`, `ui/app.js`, `ui/graphs.js`, `ui/style.css`, `CONTROL_UI.md`.

The UI now has a compact control column, session metrics, model latency, inference/action/hold timeline, measured-versus-commanded joint plots, camera previews, and session log. Choose J1–J7 and a 10/30/60-second graph window. Joint selection and window changes affect visualization only.

- Timeline inference bars = completed inference **round-trip intervals**; action ticks = dispatched actions; amber = logged rejected joint targets / validation holds; pause regions = logged pauses. The timeline is retrospective, refreshed from recorded events, not a prediction of upcoming inference.
- Joint plot = dispatched targets in radians and timestamp-aligned measured joints. Do not substitute native `q_d` for our torque-controller target, or graph normalized model increments as absolute positions.
- Graph endpoint `/api/telemetry`: 400 ms browser polling while visible; 250 ms server cache; at most 2 MiB read per journal per refresh; bounded deques retaining recent history (about 60 seconds). Measured controller feedback is downsampled for display. Old receipts without `control_started_monotonic` use measured joints at action dispatch instead of guessing clock alignment.
- Status endpoint `/api/status`: selected settings, phase, robot status, run counters, latency, and per-process CSRF token. Status polls about 700 ms; receipt-based latency can lag by about two seconds.
- `/api/frame?view=external` and `/api/frame?view=wrist`: preview images, refreshed about every two seconds.
- `/api/log`: bounded displayed logs.
- POST `/api/settings`, `/api/start`, `/api/pause`, `/api/play`, `/api/stop` require the current `X-Control-Token` and a matching allowed `Origin`. Use the UI unless programmatic control is specifically needed. Never hard-code the token.

Evidence is under the active workspace's `evidence/`:

| File/pattern | Meaning |
|---|---|
| `stream-policy-<id>.json` | Session metadata, counters, recent chunks/actions, final state/failure |
| `stream-policy-<id>.events.jsonl` | Full continuous-run action/chunk/hold/pause journal |
| `stream-policy-<id>.states.jsonl` | Native feedback, RT metadata, and clearly separated fault-history records |
| `stream-policy-<id>.stderr.log` | Native controller stderr; inspect on any stop |
| `ui-arm-*.log`, `ui-runtime-*.log` | Launch/run and warm-up logs |
| `policy-readiness.json`, `policy-readiness-*.json` | Warm-up evidence; readiness tied to current policy PID |
| `policy-runtime.json`, `policy-service.log`, `ssog-policy-engine.log` | Runtime contract and backend logs |
| `live-graphs-install.json`, `live-graphs-validation.json` | Latest deployment and initial live validation |
| `unbounded-playback-*.json` | Prior frequency/keepalive installation and real 1 Hz validation |

Continuous receipts deliberately retain only the latest 128 actions and 32 chunks; full history is in the adjacent event journal. Avoid repeatedly loading entire long-running journals into memory. A total action/chunk count is not the length of the bounded receipt array.

Latest combined UI/fault-fix backup: `backups/before-live-graphs-1790268646062142247`. Staging/source: `backups/live-graphs-staging`. Installation receipt has SHA-256 values. Preserve current files and evidence before any rollback; restore compatible client/controller versions together and restart only when stopped. The binary has no added file capabilities; the existing launcher provides RT privileges.

Other docs (`README.md`, `SSOG_SETUP.md`) contain historical experiments, superseded caps and snapshots. Prioritize live `config/setup.json`, `config/droid-fr3-active.json`, active source, runtime metadata, and the latest matching session evidence.

## User preferences and remaining work

- Do not run offline tests. Use appropriate syntax/build checks and explicitly authorized real workload verification.
- Current continuous sessions have no five-minute timer; the user requested running until Pause/Stop. A documentation change does not warrant restarting a manually stopped arm.
- Preserve the 5-step FP16 recipes, SSOG's distinct archived recipe, full-view wrist image, left-eye external selection, and current saved prompt/action settings unless asked to change them.
- The user asked that fast-model work be done directly, not delegated to a subagent.
- Do not silently remove native safety, increase thresholds, or promise that force-triggered reflexes cannot recur. Investigate the concrete cause and distinguish a model target rejection from a native controller stop.
- Current stopped/Programming state is intentional. No motion or runtime configuration was changed to write this handoff.
- Remaining quality issue: frequent joint-target holds (especially joint 7 in observed live graphs) reduce effective action rate. They are visible, counted, and preserved. No change to their constraints was requested or made in the latest update.
- Manipulation task success and freedom from future contact/reflex stops remain unverified. Successful inference/FCI timing is not proof of successful pick-and-place behavior.
