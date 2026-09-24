> Historical host snapshot. See [current operations](OPERATIONS.md) and the [latest audit](../manifests/setup-audit.json) for the later publication state. Original content below is preserved.

# Rome: pi0.5 DROID

## Action playback — all 15 generated actions, 2026-09-24

The streaming client now defaults to `policy.execution_horizon: 15`, matching the model's 15-action output. Each chunk is played through at nominal 15 Hz before synchronous inference refill, unless a safety/freshness interruption or the run-duration deadline ends it. The previous default was eight actions per chunk. The 1 kHz torque controller is unchanged. Inference and validation pauses mean elapsed-time throughput can be lower than 15 actions/second. Explicit `--horizon` remains available as a run-specific override.

## Current DROID profile — experimental table boundary removed, 2026-09-24

The user corrected the earlier contact report: the arm did not strike the table. That prior table-contact diagnosis is withdrawn; the earlier native Cartesian-reflex event remains logged, with its physical cause unresolved.

The active controller has no experimental table-height floor, target-height buffer, or downward-velocity lookahead. Its workspace is the DROID +/-1 m box. The 215 mm user tool measurement is retained as metadata and does not impose a motion boundary. DROID joint/velocity/torque constraints, protective torque feedback, native 40 N / 40 Nm collision thresholds, native rate limiting and 100 Hz filtering remain configured. The exact constraint arrays are in `config/droid-fr3-active.json`. Repeated HOLD commands now keep a fixed captured hold pose instead of recapturing the measured pose on every pause.

Prompt: `Put the marker in the mug`; five denoising steps; OpenVINO FP16. MoveIt self-collision checks, camera/state freshness, command watchdog and CPU isolation remain active. A five-minute real-arm run is requested after installation. No offline tests were run. Older dated sections below describe superseded settings.

## Active DROID constraints and 215 mm tool clearance — 2026-09-24

The active streaming controller now uses a direct external hybrid joint/Cartesian torque callback with the published DROID FR3 gains, zero desired joint velocity, native FR3 Coriolis compensation, and the exact published constraint arrays in `config/droid-fr3-active.json`. Native collision thresholds are set to 40 on all Cartesian and joint channels on explicit START. Native torque rate limiting and the 100 Hz filter remain enabled. Ruckig and its custom speed/acceleration/jerk caps, the old application wrench/torque guards, the sphere workspace, and the motion-generator tracking guard are no longer in this execution path.

DROID protective torques act within 0.2 rad of joint bounds, 0.5 rad/s of speed bounds and 50 mm of the +/-1 m workspace box. Summed controller and protective torques are clamped to 86 Nm for joints 1–4 and 11.5 Nm for joints 5–7. Any measured hard limit violation stops control. This is a local port using this FR3's native model, not the original Polymetis server or its legacy Panda URDF.

The user measured 215 mm from flange to furthest fingertip with the gripper closed and reports that the robot base is mounted directly on the table. An additional conservative 30 mm clearance sets the flange floor at base z=0.245 m. Sampled target paths must stay above 0.265 m; a 250 ms downward-velocity lookahead stops before predicted crossing of the 0.245 m floor. This is a measured tool-reach proxy, not a calibrated tool mesh. Existing self-collision validation, fresh camera/state checks, command watchdog and CPU isolation remain active. No automatic error recovery or motion restart is performed.

The default prompt is `pick up the marker`, with five denoising steps and a requested 300-second motion run. Deployment and live-validation evidence is kept under `evidence/droid-table-215mm-install.json` and the subsequent stream-policy receipts. Older sections below describe historical controller settings.

## Current task prompt

The default task prompt is `Put the marker in the mug`, stored in `config/setup.json` under `task.prompt`. The streaming launcher uses it unless an explicit `--prompt` overrides it. Changing this prompt does not start inference or arm motion.

## Requested DROID constraints — staged, inactive

The exact published FR3 limit arrays, native collision thresholds, safety-controller margins and stiffnesses are saved in `config/reference/droid-fr3-constraints.staged.json`, with pinned source files beside it. The active launcher does not load this reference profile. The active gentle native-impedance controller remains unchanged. A previous marker run made confirmed fingertip-table contact; fingertip clearance and the table are not yet modeled. DROID also requires torque safety feedback near its limits, which the active native joint-impedance callback does not implement. Higher limits and another marker motion run are pending resolution of this known contact path. The requested next run is 300 seconds of real arm inference with five denoising steps and prompt `pick up the marker`.

## Gentle operating profile for Cartesian-reflex diagnosis — 2026-09-24

The active Ruckig trajectory uses 0.04 rad/s speed, 0.15 rad/s² acceleration and 1 rad/s³ jerk, matching the earlier completed 128-second run's motion profile. Hard ceilings remain 1 rad/s, 1.5 rad/s² and 10 rad/s³; the measured-speed stop stays at 1 rad/s. The lower profile reduces dynamic demand while diagnosing native Cartesian-reflex stops. It does not establish that acceleration was their only cause.

Native joint impedance, five denoising steps, the fixed 500 mm workspace, the existing force/torque and tracking guards, native collision behavior, rate limiting, 100 Hz filtering, and CPU isolation remain unchanged. The active task prompt is `pick up the marker`. Action scaling and chunk timing are unchanged.

Live telemetry now includes both wrench frames, external joint torques, desired acceleration, contact/collision flags, tool transforms, and configured payload. A libfranka ControlException also emits its retained pre-stop states after control exits. These historical states are isolated from live policy feedback. No offline tests are run.

## Native joint impedance restored — 2026-09-24

The live streaming controller uses Franka's native joint-impedance mode again. The custom DROID hybrid torque callback was removed from the active build after three motion attempts exceeded the existing 0.03 rad tracking-error guard within 0.33–0.78 seconds. Its source and installation history are retained in backups.

Ruckig continues to generate joint-velocity commands with caps of 1 rad/s, 1.5 rad/s² and 10 rad/s³. The additional measured-speed stop at 1 rad/s is retained. The launcher verifies `feedback_controller: franka_native_joint_impedance` before START. Five denoising steps, the fixed 500 mm workspace, 30 N / 15 Nm per-axis wrench guards, 10 Nm external joint torque guard, tracking and joint-margin checks, watchdogs, native rate limiter and 100 Hz filter remain configured. CPU isolation is unchanged.

No custom torques are sent in this mode; torque control is handled by Franka. This rollback addresses the hybrid controller's tracking regression. Previous native Cartesian-reflex stops still require independent diagnosis. Compilation and syntax checks precede authorized short live motion, followed by a five-minute attempt only if the short run completes cleanly. No offline tests are run.

## Current workspace radius — 2026-09-24T03:08:51.973492-04:00

The streaming launcher now defaults to a **500 mm radius sphere** around the existing fixed center **[0.472761541605, -0.14038155973, 0.25467634201] m** in the robot base frame. This is saved in `config/setup.json` under `workspace`. The prior run used a 250 mm radius. Both Python and C++ now accept radii from 150 through 500 mm; the controller reports the active radius and anchor, and the launcher checks them before START. On a boundary stop, the controller logs actual distance, radius, and robot timestamp. A CLI workspace option explicitly overrides the saved default.

For the next authorized run, use `bash run-policy-stream.sh --execute-confirmed --duration 300 --horizon 8 --prompt "pick up the marker and place it in the mug"`; no workspace flags are needed. If specifying them, use `--workspace-anchor 0.472761541605 -0.14038155973 0.25467634201 --workspace-radius 0.5`.

Ruckig caps remain 1 rad/s, 1.5 rad/s² and 10 rad/s³. The 30 N / 15 Nm wrench guard, native protections, CPU isolation and camera settings are unchanged. Source, configuration and binary were backed up and rebuilt while stopped. No offline tests or robot motion were run. Receipt: `evidence/workspace-500mm-install.json`. Earlier dated sections below describe earlier settings.

## Restored Ruckig motion profile — 2026-09-24T02:53:52.040286-04:00

The active streaming controller uses Ruckig at 1 kHz to generate joint-velocity commands, with per-joint speed **1 rad/s**, acceleration **1.5 rad/s²**, and jerk **10 rad/s³** caps. It retains the prior time-synchronized Ruckig setup, including controlled braking for STOP, watchdog expiry, and run completion. Every generated sample is checked for finite position/velocity/acceleration and compliance with all three limits, allowing only numerical tolerance. The Python launcher requires matching Ruckig mode and caps in the controller's startup metadata before sending START. These are generated-trajectory caps; measured tracking has not been validated in a new physical run.

The 30 N / 15 Nm per-axis application wrench guard, 10 Nm external-joint-torque guard, native rate limiter and 100 Hz filter, CPU isolation, payload readback wait, joint-bound re-inference, camera configuration, collision checks and other guards are retained. This velocity controller links the existing standard libfranka installation; the temporary position-command precision patch remains archived but is no longer loaded.

The controller and launcher were backed up and rebuilt/parsed with the arm stopped. No offline tests, trajectory simulations, or robot motion were run. Receipt: `evidence/restore-ruckig-install.json`. Earlier dated sections below describe earlier settings.

## Current application wrench guard — 2026-09-24T02:47:53.253862-04:00

The active streaming controller stops when any estimated Cartesian force component exceeds **30 N** or torque component exceeds **15 Nm** in magnitude (50% above 20 N / 10 Nm). Nonfinite values also stop control. Its ready metadata records both limits; a stop now logs the axis, exact estimated value, threshold, and robot timestamp. The separate external-joint-torque guard remains 10 Nm. This change does not call `setCollisionBehavior`, modify native self-collision protection, restore Ruckig, or change speed/acceleration/jerk behavior.

This is an application threshold adjustment, not proof that abrupt policy motion or the tool-load model is correct. The previous run stopped on the wrench guard and its exact crossing sample was not recorded. The source and binary were backed up, rebuilt, and installed with the controller stopped. No offline tests or arm motion were run. Earlier dated sections below describe earlier settings. Receipt: `evidence/wrench-30N-15Nm-install.json`.

## Current real-time CPU configuration — 2026-09-24T02:17:54-04:00

The controller worker is pinned to logical CPU 5 with SCHED_FIFO priority 80. CPU 11, its SMT sibling, is excluded from ordinary userspace. Robot Ethernet enp15s0 uses CPU 4 for interrupts and threaded NAPI reception at FIFO priority 90; sibling CPU 10 is likewise excluded. Ordinary system/user work, OpenVINO, MoveIt, camera processing, and controller pipe/JSON I/O use CPUs 0–3 and 6–9. A top-level franka.slice gives the arm worker access to CPU 5 while the rest of the application stays on housekeeping CPUs. The launch wrapper enters that slice automatically, reapplies the host profile, exports /etc/franka/realtime.env, and forwards stop requests to its supervised unit.

The controller locks its memory, prefaults the worker stack, explicitly configures its own priority/affinity, and fails before control if configuration cannot be applied. Startup state logs include the actual control thread scheduling readback. franka-realtime.service applies IRQ affinity, threaded Ethernet reception, performance governors, and housekeeping workqueue affinity at boot; no boot-parameter changes or reboot were performed. This is userspace/workload separation with IRQ tuning, not full tickless kernel isolation. Unmovable managed IRQs are recorded in evidence/realtime-host-profile.json. No offline tests were run. The arm was not moved during installation.

## Current control mode — 2026-09-24T02:09:53-04:00

Ruckig and its custom joint speed, acceleration, and jerk limits have been removed at the user's request. DROID actions still decode as `q + 0.2 * clip(action[:7], -1, 1)`. The adapter now forwards those joint-position targets directly through libfranka `JointPositions`, with native rate limiting enabled and its 100 Hz low-pass filter retained. The former 1 rad/s, 1.5 rad/s², and 10 rad/s³ application limits are no longer enforced; native limits are those used by libfranka for this robot and its current configuration. The controller reports this mode explicitly, and the Python launcher validates it and records that there is no application motion-limit profile.

HOLD, STOP, watchdog expiry, and run-duration expiry request zero position advance from each current desired robot state; libfranka handles deceleration. Completion waits for desired joint speed at or below 0.001 rad/s and measured speed at or below 0.02 rad/s, with a 0.2-second minimum stop interval and the existing one-second stopping deadline. The 20 N / 10 Nm Cartesian guard, 10 Nm external joint torque guard, workspace envelope, joint margins, tracking guard, command watchdog, and camera/state checks remain. The previously removed C++ 0.005 rad reference guard remains absent.

This revision was compiled and installed while the arm was stopped. No offline tests or physical execution were performed. Earlier sections below describe historical profiles and runs. `decrease_limits.py` belongs to the retired Ruckig path and is not applicable to this control mode.

## Target-reference guard revision — 2026-09-24T02:06:30-04:00

The C++ check that rejected targets when the expected joint reference differed from the controller state by more than 0.005 rad has been removed at the user's request. Its obsolete self-test assertion and result field were removed as well; no offline tests were executed. The controller was rebuilt and installed while idle, without starting robot motion. Python still refreshes and revalidates targets after state changes, and retains its 150 ms feedback-age check. Target increment and joint-range checks, tracking, workspace, force/torque guards, and command watchdog remain in the execution path.

## Current application wrench guard — 2026-09-24T01:59:07-04:00

The streaming controller now stops if an estimated Cartesian force component exceeds **20 N** or a Cartesian torque component exceeds **10 Nm** in magnitude. The separate external-joint-torque guard is 10 Nm. Joint motion caps are 1 rad/s, 1.5 rad/s², and 10 rad/s³. This revision changes the application wrench guard; it does not call `setCollisionBehavior` or change onboard collision settings. The controller was rebuilt and installed with the arm idle. No offline tests or robot motion were run for this revision.

## Current installed motion profile — 2026-09-24T01:56:12-04:00

The streaming controller is configured for **1.0 rad/s joint speed, 1.5 rad/s² acceleration, and 10.0 rad/s³ jerk**, per joint. The Python client records and requires the same 1.0 rad/s controller speed. The installed binary passed its no-hardware trajectory and target-validation self-test; full-speed braking completed in 0.816 seconds under the existing one-second stopping deadline. The representative action test reached 0.609342 rad/s; 1.0 rad/s is the configured ceiling, not a claim about achieved task speed. No physical motion was run for this installation.

The ±0.2 rad action range, workspace envelope, force/torque guards, joint margins, tracking checks, collision validity, camera/state freshness, watchdog, and 300-second run bound remain enforced. Descriptions of lower limits below refer to earlier revisions and physical-run evidence.

Installed at `/home/spring/robo_run/franka_droid` (persistent data under `/var/lib/spring-data/workloads/robo_run/franka_droid`).

The active inference server uses **OpenVINO 2026.4.0 FP16 on the Intel Arc B580 with 5 denoising steps**. Thirty warmed localhost requests measured **79.79 ms median / 80.31 ms p95** per complete 15 x 8 action chunk, after three warmups across ten saved real camera/state observations. This includes preprocessing, vision encoding, language prefill, all five denoising steps, postprocessing and localhost transport; it excludes camera capture and robot motion. The arm stayed idle during the latency benchmark. A subsequent authorized physical run is recorded below; marker placement remains incomplete. Raw five-step measurements are in `evidence/openvino-fp16-5step-server-latency.json` on Rome and `openvino-fp16-5step-latency.json` in the local deliverables.

At **10 denoising steps**, OpenVINO measured **100.94 ms median / 101.38 ms p95**, versus **521.47 / 561.22 ms** for the prior PyTorch BF16 server at the same ten steps: **5.17x faster** by median. Reducing OpenVINO from ten to five steps lowers latency by **21.0%**. Five steps changes the sampling schedule; physical task success at five steps has not been evaluated.

With the five-step OpenVINO policy, a subsequent live run completed **10 bounded direct-libfranka control steps**, with no controller failure and exit code 0. Live inference measured **89.05 ms median** (88.58–89.65 ms), excluding camera capture and arm movement. The flange moved 58.13 mm net, with maximum joint tracking error 0.0001533 rad. The gripper remained open and the marker stayed on the table. The controller stopped after the bounded run. Evidence: `openvino-5step-arm-run.json` locally and `evidence/fci-policy-1790226251629338176.json` on Rome, with matching final external/wrist images.

The latest requested physical execution accumulated **300.08 seconds across four velocity-control segments**, issuing **3,754 actions** with OpenVINO FP16 and five denoising steps. It was **not an uninterrupted five-minute run**. The final **128.18-second segment completed cleanly** with controller exit code 0; one brief camera hold recovered without ending the run. The arm **grasped and lifted the marker but did not place it in the mug**. It is stopped and still holding the marker. Current Desk status reports no robot errors and no execution running; no streaming control process remains.

The action scheduler targets **15 Hz within eight-action chunks**, with synchronous inference refill. Across those four segments the effective action rate was **12.51 Hz**, and 471 inference requests measured **88.46 ms median / 90.48 ms p95**. Logged post-gripper action intervals within chunks averaged 68.52 ms; the scheduler is nominally 15 Hz, not a guarantee of exact per-command timing. Measurements are in `streaming-run-summary.json` with the four `stream-policy-*.json` receipts and final camera images.

The continuous adapter is `run_policy_stream.py` / `run-policy-stream.sh` plus `stream_fci.cpp`. The low-level controller streams joint velocities at the libfranka control rate using Ruckig trajectory smoothing; DROID actions still decode to joint-position targets before smoothing. Limits are 0.04 rad/s, 0.15 rad/s² and 1 rad/s³, with force/torque, joint, tracking, command-watchdog and camera guards retained. Only the control worker has real-time priority; pipe and JSON handling use normal scheduling. The trajectory advances by the actual number of robot ticks to account for missed packets. Slow validity responses trigger fresh-state replanning; brief camera mismatches hold and wait for a matching fresh pair. Session duration is optional; the control UI now runs continuously until Pause or Stop (see CONTROL_UI.md).

Bring-up stops exposed a 40 ms validity-service timeout and position-command acceleration discontinuities. The position stream was replaced with the smoothed velocity stream. The first velocity segment reached the original 150 mm initial-test travel boundary after picking up the marker. Following the user's request to relax software constraints, the travel boundary alone was enlarged to **250 mm around the same fixed origin [0.472761541605, -0.14038155973, 0.25467634201] m**. It was not recentered on each resume. Later state-freshness and camera-pair interruptions were handled by bounded holds and fresh checks. Earlier short bring-up attempts and full state logs remain in Rome's `evidence/` directory.

To repeat the bounded chunked run from the configured tabletop workspace, use a root shell on Rome:

```bash
cd /home/spring/robo_run/franka_droid
bash run-policy-stream.sh --execute-confirmed --duration 300 --horizon 8 \
  --workspace-anchor 0.472761541605 -0.14038155973 0.25467634201 \
  --workspace-radius 0.25 \
  --prompt 'pick up the marker and place it in the mug'
```

A later request to remove all application-added limits was handled as a **read-only raw-inference rerun**: the active motion limits were **not removed**, and the arm/gripper received **no actuation commands**. Thirty requests after three warmups on one fresh camera/FCI/gripper observation measured **79.23 ms median / 79.64 ms p95** with five-step OpenVINO FP16. Raw output was retained without the motion adapter's joint-increment scaling: the largest DROID-decoded joint increment was **0.12649 rad**. This was not a physical trial of those actions. Evidence: `raw-inference-rerun.json` and `raw-inference-actions.npy`. The separate `raw-inference-only.sh` / `raw_inference_only.py` path uses read-only FCI and gripper reads and restores the camera viewer afterward.

The installed IRs and GPU compilation cache are in `openvino_fp16/`; runtime dependencies are in `ov-venv/`. `config/setup.json` selects OpenVINO by default. All three compiled graphs run on GPU.0 with the FP16 precision hint and dynamic quantization disabled. Execution graphs show 452 of 454 matrix/convolution compute nodes in FP16; the two FP32 matrix operations generate rotary position embeddings. Host Euler integration remains FP32. Vision embeddings and KV buffers stay in GPU remote tensors. The cache uses four dimensions to avoid a GPU compiler error with the original five-dimensional stack. Attention masks use a finite FP16-safe negative value to avoid infinities on fully padded queries.

At ten denoising steps, three saved real observations were checked against fixed-noise outputs from the PyTorch BF16 model: action RMSE 0.00248–0.00640, maximum absolute action difference 0.02086, and all 45 gripper decisions matched. This numerical comparison is not a completed physical manipulation evaluation. Raw measurements and outputs are retained in `evidence/openvino-*.json`.

Start inference with `source ./env.sh && ov-venv/bin/python serve_policy.py --start`. The already-running transient user unit is `franka-droid-policy.service`, listening on `127.0.0.1:8000`; it does not launch motion or start at boot. The original backend remains available with `venv/bin/python serve_policy.py --start --backend pytorch` after stopping the server. Do not run two servers on the same port.

The current execution path is `run_policy_stream.py` plus `stream_fci`, using direct **libfranka 0.21.3** and OpenVINO FP16 with five denoising steps on Rome. It dispatches eight actions from each 15-action chunk at a nominal 15 Hz and holds during synchronous inference refill. Previous physical runs grasped and lifted the marker; mug placement has not been verified. The older single-step adapter described below remains available separately.

The streaming motion revision removes the extra 0.025 rad uniform action shrink. Joint targets now use the OpenPI/DROID conversion `q + 0.2 * clip(action[:7], -1, 1)`, without additional rescaling. This permits a decoded target increment of up to 0.2 rad; it does not move that entire distance instantaneously. The joint-speed ceiling is 0.10 rad/s, increased from 0.04. Acceleration remains 0.15 rad/s² and jerk 1 rad/s³. The no-hardware trajectory test reached the revised speed, respected acceleration and jerk, and braked from full speed in 0.816 seconds, within the existing one-second stopping deadline.

Collision validity, force/torque stops, joint margins, tracking error, camera/state freshness, command watchdog, and the fixed workspace envelope remain active. Collision-path sampling now adds samples for larger decoded targets (at most 0.01 rad per joint between samples). Franka rate limiting and the 100 Hz filter remain enabled. The adapter records actual target-dispatch timestamps, additional action scale, normalized clipping, and controller limit readback in each receipt. These settings still shape physical trajectories and do not establish exact native DROID tracking behavior.

The revised physical run on September 24 lasted 23.839 seconds before the D405 disconnected from USB (kernel event at 01:35:12 EDT). The stale-camera protection stopped the controller cleanly, and Desk reported no robot errors. Recorded targets reached 0.10346 rad with no extra shrink, measured joint speed reached 0.10202 rad/s, and maximum tracking error was 0.0003884 rad. The requested five-minute rerun was not completed. The camera was still absent afterward and must be reconnected before further execution. `changed-caps-run.json` records the outcome; the final portion of the Python action log was lost during camera cleanup, but the complete controller state log contains the clean stop. Final receipts are now saved before camera cleanup to preserve future failure evidence.

The exact Robotiq configuration linked by DROID is applied by the direct helper on connection and verified by robot readback: mass 0.9 kg, COM [0, 0, 0.057] m, diagonal inertia [0.002768, 0.003149, 0.000564] kg m². The helper rejects nonzero existing Desk end-effector mass and a nonidentity flange-to-EE transform to avoid silently combining incompatible configurations. Starting the policy server alone does not move the arm or gripper.

This is an FCI load setting, **not a saved Desk end-effector profile**. Safari automation failed when opening the import dialog. `config/droid-robotiq-endeffector.json` is the exact official import file; `apply_droid_load.py --apply` reapplies it without arm motion while the arm controller is inactive and the existing end-effector mass is zero. Avoid double counting: when importing/activating a persistent 0.9 kg end-effector profile in Desk, clear the separate FCI load and verify the combined mass remains 0.9 kg. The DROID baseline does not account for the D405 or custom mounts.

## Hardware and policy inputs

| Input | Selected hardware/configuration |
|---|---|
| External image | ZED 39338267, left eye, factory rectification, 672 x 376 at 15 FPS |
| Wrist image | Intel RealSense D405 427622273688, RGB 1280 x 720 at 15 FPS, USB 3, SDK geometry rectification |
| Arm state | FR3 serial 309969-2662311, seven ordered joint positions directly from libfranka |
| Gripper state | Robotiq 2F-85, FTDI BG04SJCN USB serial, Modbus 115200 baud, slave 9 |
| Checkpoint | `/home/spring/robo_run/checkpoints/pi05_droid_pytorch`, source BF16 converted to OpenVINO FP16, Intel Arc B580 GPU |

Images remain RGB and are padded/resized to 224 x 224 using OpenPI. The ZED rectification uses all eight rational distortion coefficients from the current factory file; using only the older Open Capture five-coefficient model caused excessive cropping and was corrected before the physical attempt. The unused image slot is masked. The configured checkpoint produces 15 x 8 action chunks; the old example's 10 x 8 assertion is not used. Official OpenPI transforms handle DROID normalization, discrete state tokens, and action unnormalization.

DROID gripper state is closure (0 open, 1 closed), derived from the Polymetis 85 mm / encoder 3-to-230 mapping. DROID's joint actions represent normalized joint increments scaled by 0.2, **not raw ROS radians/second**. `contract.py` decodes them for inspection only.

The camera pair delivered approximately 14.98 FPS each for ten seconds, with 150 fresh pairs and maximum observed timestamp difference 33.5 ms. These are host timestamps, not hardware synchronization. USB 2 limits the selected ZED to the configured VGA feed. The D405 differs from DROID's ZED Mini; matching inputs does not establish equal task success. Wrist mounting/orientation was inspected in captured images.

## Run the inference server

After the user's explicit run request, a transient user service was started:

```bash
systemctl --user status franka-droid-policy.service
tail -f ~/robo_run/franka_droid/evidence/openvino-server.log
```

It listens only on `127.0.0.1:8000` after model loading. It has no robot command interface and does not start on boot. To stop:

```bash
systemctl --user stop franka-droid-policy.service
```

To start manually when the service is stopped:

```bash
cd ~/robo_run/franka_droid
source ./env.sh
ov-venv/bin/python serve_policy.py --start
```

## Capture and infer without motion

Capture normally requires a healthy, already-activated gripper and fresh ROS state. It never activates the gripper. The explicit `--inference-only-allow-unready-gripper` option allows measured gripper state to be captured for inference without motion even while the communication watchdog is faulted. The wrapper temporarily releases the existing camera viewer and restores it on exit.

```bash
cd ~/robo_run/franka_droid
source ./env.sh
bash capture-observation.sh --prompt 'YOUR TASK' --output evidence/observation.npz
python infer_observation.py --infer --observation evidence/observation.npz --output evidence/actions.npy
```

The client saves actions only. It does not execute them.

## Direct physical execution

`run_policy_fci.py` captures fresh camera images, reads the actual joints and Robotiq closure, infers a fresh action, and passes a bounded joint target to `direct_fci`. Only the first action from each chunk is executed. DROID's 0.2 rad scale is uniformly reduced to the configured maximum increment. This initial adapter is deliberately slower than native DROID's 15 Hz loop.

`direct_fci.cpp` uses `Robot::control` with the onboard joint impedance controller, rate limiting enabled, and a 100 Hz command filter. Each step uses a 1.2 second quintic ramp and 0.5 second hold. The C++ callback checks forces/torques, joint tracking, control-cycle gaps, and a 150 mm session envelope. Targets must match the fresh initial state, remain within 0.03 rad of it, and respect the joint limits retrieved from this robot's URDF. Python also monitors camera/gripper health. MoveIt is used only for joint/self-collision validity; **ROS does not command the arm**. Table and custom-tool collision geometry are not modeled.

Run from a root shell on Rome, with the policy server and MoveIt validity service already running:

```bash
cd /home/spring/robo_run/franka_droid
bash run-policy-fci.sh --execute-confirmed \
  --prompt 'pick up the marker and place it in the mug' \
  --steps 1 --max-increment 0.005
```

The wrapper requires the ROS hardware service to be stopped, grants the child process realtime resource limits, then runs the policy client/helper as `spring`. It temporarily stops and restores the camera viewer. `--infer-only` performs observation/inference and load setup without an arm trajectory; the gripper keepalive remains active.

The gripper is activated and its command keepalive runs during execution. On exit the helper ends control and the gripper receives a hold command. There is no automatic motion retry, error recovery, or boot-time execution.

The older ROS position-controller path failed with velocity/acceleration discontinuities and later a tracking-tolerance error. Its configuration was changed to Franka's published effort-controller gains but that alternative was not physically validated; the user chose direct libfranka instead. The force thresholds were not raised.

The persistent Desk profile remains "No End Effector". The user authorized the marker task, confirmed the clear workspace/local stop operator, and enabled Execution and unlocked joints.
## Evidence

`evidence/setup_validation.json` records preprocessing-only checks, checkpoint header, package versions and XPU availability. `evidence/camera_pair_check.json` records live camera timing before the rectification correction. `evidence/gripper_activation.json` records the subsequently authorized activation. `evidence/marker_mug_actions.npy` is the first inference using the excessively cropped external image and must not be executed. `marker_mug_observation_corrected.npz` / `marker_mug_actions_corrected.npy` use the corrected full view. `marker-mug-first-run.log` records the initial force stop. `ros-policy-*.json` record the unsuccessful ROS trials. `fci-policy-*.json`, matching image snapshots, and stderr logs record direct control, tracking, camera observations, and outcomes. The checkpoint was not loaded during the setup-only validation; the subsequent run request explicitly authorized starting the server and then executing the specified task.

The environment reuses the existing `robo_run/venvs/convert` dependencies via a `.pth` file. Keep that environment and `zed_camera_setup` (native capture library/discovery) in place.

## Sources

- [OpenPI DROID integration](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README.md)
- [DROID's Robotiq load setup](https://droid-dataset.github.io/droid/software-setup/host-installation.html#updating-inertia-parameters-for-robotiq-gripper)
- [Exact Franka configuration linked by DROID](https://github.com/frankarobotics/external_gripper_example/blob/master/panda_with_robotiq_gripper_example/config/endeffector-config.json)
- [DROID action conversion](https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py)
- [DROID ZED camera path](https://github.com/droid-dataset/droid/blob/main/droid/camera_utils/camera_readers/zed_camera.py)
- [RealSense D405 specifications](https://www.realsenseai.com/product-family/d405-series/)
- [Polymetis Robotiq implementation](https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/python/polymetis/robot_client/robotiq_gripper/third_party/robotiq_2finger_grippers/robotiq_2f_gripper.py)
