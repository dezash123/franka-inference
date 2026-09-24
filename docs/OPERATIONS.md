# Current Rome operations

Checked 2026-09-24 against deployed files and the running UI. See [setup-audit.json](../manifests/setup-audit.json) for the exact UTC time. Settings and session state are mutable; this document is not a command to resume motion.

## Saved settings at the audit

| Setting | Value |
|---|---|
| Backend | `ssog_mc2`, checkpoint `pi05_droid_jointpos`, INT8/INT4, two model passes |
| Prompt | `put all items on the desk in the bowl` |
| Generated / played actions | 15 generated / first 10 played per inference |
| Nominal playback | 15 Hz; synchronous refill with holds for inference and validation |
| Session duration | Unlimited until Pause, Stop or fault |
| External policy view | `39338267`, left eye |
| Additional external | `37266581`, left eye, preview only |
| Wrist policy view | ZED Mini `12607895`, left eye, no digital zoom |
| Camera capture | All three approximately 15 FPS |
| UI | Three live MJPEG previews with measured FPS, latency, inference timeline and joint trajectories; no Session log panel |

The audit initially observed an already-running arm session; a subsequent read showed it stopped, still in Execution with the model ready and all three preview feeds healthy. This audit did not start, stop or modify the session. The latest receipt records the final observed state; it does not certify task success or future robot readiness.

The DROID constraint profile, controller gains, payload, bounds and watchdogs are described in the [historical handoff](DESMOND_FRANKA_HANDOFF.md#controller-and-active-constraints) and preserved in active source/configuration. There is no Ruckig or tabletop boundary. Matching DROID's numeric constraints does not make the custom controller, observation pipeline or runtime identical to DROID.

The client update installed on Rome at 22:44 UTC rescales absolute-joint model targets toward the measured posture when any joint step exceeds 0.2 rad, preserving the direction of the seven-joint delta. The old step-size-only HOLD/discard path is removed. Joint-position bounds, path validation and native controller checks still apply; a rescaled target is not guaranteed to dispatch. Normalized-action decoding is unchanged. Action journals now record the applied `scale` and `absolute_target_clamped`. This file is loaded on session startup; the publication audit did not run motion to validate the new behavior.

## Normal use

Open [the control panel](http://100.95.186.107:8787/) on the tailnet. Stop the current session before editing prompt, model, actions or playback Hz. Saving a model change loads and warms it using live observations; readiness belongs to that exact backend and policy-server process. Start requires readiness and the robot in Execution. Pause holds the arm and gripper; Play discards the old chunk and takes a fresh observation; Stop ends the session. A fault does not automatically retry.

The Hz setting controls action playback, not independent background inference. Camera FPS, model latency, effective actions/s and effective inferences/s measure different parts of the pipeline. A faster playback setting does not increase camera acquisition frequency. Model latency excludes image preparation; RPC timing includes preparation and transport. Neither includes action playback. See [CONTROL_UI.md](../franka_droid/CONTROL_UI.md).

## Services and ownership

| Owner | Role | Expected arrangement during arm motion |
|---|---|---|
| root `franka-control-ui.service` | Tailnet UI, port 8787 | Active |
| spring `franka-droid-policy.service` | Model server, loopback port 8000 | Active |
| transient root `franka-droid-run-*.service` | Launcher, client and native controller | Active in `franka.slice` |
| root `franka-ros-state.service` | Alternative ROS hardware/FCI owner | Inactive |
| root `franka-ros-moveit.service` / planner process | Self-collision checking | Planner active; no ROS hardware ownership |
| spring `zed-camera-viewer.service` | Idle camera capture, loopback port 8765 | Inactive while warm-up/session owns cameras |
| root `franka-realtime.service` | CPU/IRQ layout | Active (exited); reapplied by launcher |
| root `franka-c2-dhcp.service` | Dedicated robot link DHCP | Active |

Read status without touching hardware:

```bash
systemctl status franka-control-ui.service franka-ros-state.service franka-ros-moveit.service
runuser -u spring -- env XDG_RUNTIME_DIR=/run/user/1000 \
  systemctl --user status franka-droid-policy.service zed-camera-viewer.service
```

The native worker renames its threads to `franka-control` and `franka-io`; `pgrep -x stream_fci` alone can miss it. Inspect `/proc/*/exe` or the transient service. The [read-only audit](../scripts/audit_rome.py) reports the current worker's affinity and scheduler without connecting to FCI. The expected control thread uses CPU 5 / FIFO 80; housekeeping uses 0–3,6–9. Network NAPI uses CPU 4 / FIFO 90. The kernel is PREEMPT_RT; this is affinity/cgroup partitioning, not boot-time `isolcpus` isolation.

## Startup after maintenance or reboot

The UI, RT and DHCP units are enabled. The two ROS units are disabled for automatic boot; spring's user manager has lingering enabled. The repository installer leaves UI/model/idle-camera services stopped and does not restart them. On a provisioned, stopped host, start the UI with `systemctl start franka-control-ui.service`, then use the UI to load/warm the selected model. Starting the UI does not start arm motion.

MoveIt must be available before motion. The captured MoveIt unit has `Requires=franka-ros-state.service`: starting that unit normally starts the competing FCI owner. If neither planner nor arm is running and only the planner is needed, the existing launcher can run directly in a dedicated terminal as spring:

```bash
sudo -u spring bash /var/lib/spring-data/franka-network-20260923/ros/start-ros-moveit.sh
```

Its launch file starts `move_group` only. Keep that terminal alive, and do not launch a second planner alongside an existing one. Do not start the ROS hardware service or use an FCI diagnostic executable while the direct controller owns the robot. The service dependency has been documented, not modified by this publication audit.

For Franka Desk, from your client machine with local port 17443 free:

```bash
ssh -N -L 127.0.0.1:17443:172.16.0.2:443 root@100.95.186.107
```

Open `https://127.0.0.1:17443/` using the existing authorized login. Tailnet enrollment, SSH and Desk credentials are not in this repository.

## Diagnose a stop

Read the current UI error, matching `evidence/stream-policy-*.json`, native `.stderr.log` and `.states.jsonl`. The controller reports the native cause before its history dump. `cartesian_reflex` is a native force/torque stop; a stale timestamp without a recorded native cause needs a fresh scheduling/transport investigation. Do not assume every stale error is the old reporting race.

For camera errors, inspect `GET /api/cameras`: serial, inclusion, source, freshness and measured FPS. The idle viewer being stopped during arm motion is expected. Required policy cameras are checked independently of the preview-only external. Do not run another capture process beside the active owner. `external_cameras_note` in the saved configuration describes the earlier shared USB link; it does not reflect the subsequently successful three-camera preview check.

Logs remain on Rome even though the Session log panel is gone. Full action history lives in `.events.jsonl`; the JSON receipt retains only bounded recent action/chunk arrays. Preserve the matching receipt and journals before a rollback. No software change in this audit altered the motion profile, runtime, camera selection or watchdogs.
