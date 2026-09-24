# Restore and operate the Rome deployment

This is a source/configuration export of the deployed host, not a disk image or a validated unattended fresh-host installer. The native programs were built from the exported sources on Rome during publication, in a separate temporary directory. No arm programs or offline test suites were executed for publication. Preserve the distinction between that build check and the historical live-run receipts.

## Paths and prerequisites

| Component | Deployed location |
|---|---|
| Runtime | `/var/lib/spring-data/workloads/robo_run/franka_droid` |
| Workspace alias | `/home/spring/robo_run` → `/var/lib/spring-data/workloads/robo_run` |
| OpenPI | `/home/spring/robo_run/openpi` |
| Direct libfranka install | `/var/lib/spring-data/workloads/robo_run/vendor/franka_install` |
| ROS overlay | `/var/lib/spring-data/workloads/robo_run/ros2_ws` |
| Network / ROS launch helpers | `/var/lib/spring-data/franka-network-20260923` |
| Camera implementation | `/home/spring/zed_camera_setup` |
| Native SSOG | `/var/lib/spring-data/workloads/pi05-final/ssog-a-mc2` |
| OpenVINO checkpoint A | `/var/lib/spring-data/workloads/pi05-final/ov-a` |

These paths occur in services and Python adapters; cloning the repository elsewhere does not redirect a live installation. The installation helper targets this layout and this host only. Model assets are covered in [ARTIFACTS.md](ARTIFACTS.md).

Rome uses Ubuntu 24.04.4 / Spring Edge OS 0.1.2, Python 3.12, ROS 2 Jazzy, GCC 13.3, CMake 3.28, an Intel Arc B580 at PCI `0000:03:00.0`, and the `xe` GPU driver. Host packages are `intel-opencl-icd` / `libze-intel-gpu1` `26.31.39395.13-1~24.04~ppa1`, and `libze1` `1.32.0-1~24.04~ppa1`. The runtime account is `spring` (UID 1000), with access to the camera, serial, render, video and Docker devices. Docker is needed for the native SSOG line. The custom OpenVINO line runs directly on the host. Upstream software and model terms remain applicable.

## Native dependencies and build

`manifests/sources.json` pins the direct libfranka checkout, OpenPI, and all captured ROS workspace checkouts. Fetch into a **new** directory:

```bash
python3 scripts/fetch_sources.py --destination /path/to/new-sources
```

The helper checks out detached commits, initializes submodules, and checks/applies local patches. It does not install packages or start services. `--only openpi` or `--only libfranka` restricts the selection; the option may be repeated. The `ros2/` prefix in manifest keys creates the corresponding subdirectory. Place/link the ROS sources under the deployed workspace's `src/` directory when restoring it.

`manifests/native-dependencies.json` records the native package versions, libfranka build options and Pinocchio/urdfdom revisions used on Rome. Build these dependencies using the pinned libfranka `.ci` recipes and patches into `vendor/franka_install`, then build libfranka with that prefix, `BUILD_TESTS=OFF` and `GENERATE_PYLIBFRANKA=OFF`. Preserve `patches/libfranka.patch`: it contains the mainline PREEMPT_RT detection change and the recorded JointPositions history change. The ROS overlay uses a separate libfranka revision and patch. Do not replace the direct install with the ROS library accidentally.

Once the prefix exists:

```bash
FRANKA_PREFIX=/var/lib/spring-data/workloads/robo_run/vendor/franka_install \
  bash scripts/build_native.sh /path/to/build-output
```

This builds the torque controller, four diagnostic/recovery helpers, and the V4L2 shared library. It never executes them. In particular, `recover_fci` and the FCI readers must not be used as build checks: they connect to physical hardware, and only one process may own FCI.

## Python and ROS environments

The existing environments share packages through `.pth` files; they are not three independent pip locks:

- `franka_droid/venv` includes packages from `robo_run/venvs/convert/lib/python3.12/site-packages`.
- The conversion environment includes OpenPI and its client source, and packages from `/home/spring/pi05-up/venvs/torch-xpu/lib/python3.12/site-packages`.
- `franka_droid/ov-venv` includes the arm environment and supplies OpenVINO FP16.
- `pi05-final/ov-a/venv` is a separate small environment, created by its shipped `setup_venv.sh` from pinned wheels.

The effective inventories in `requirements/` and `manifests/python-*-effective.json` resolve package precedence; the raw inventories also contain shadowed duplicate versions. `manifests/*-module-paths.json` shows actual module origins. The inventories are a record, **not** a standalone `pip install -r` recipe: local OpenPI/ROS packages, XPU wheels, package indexes and the Transformers replacement files must be supplied separately.

Key effective versions at export: Torch `2.12.1+xpu`, torchvision `0.27.1+xpu`, triton-xpu `3.7.1`, Transformers `4.53.2`, OpenVINO FP16 `2026.4.0`; the checkpoint-A worker uses OpenVINO `2026.3.1` and NumPy `2.2.6`. Install the XPU builds from the corresponding PyTorch XPU distribution, not CPU/CUDA wheels. OpenPI's `src/openpi/models_pytorch/transformers_replace/` must be copied into the installed Transformers package as described by the pinned OpenPI README. Its local model has patched AdaRMS/cache behavior and is not compatible with an arbitrary stock Transformers release.

Restore/build the pinned ROS workspace with ROS Jazzy and its package dependencies (`rosdep`/`colcon`, without running test targets). Source `/opt/ros/jazzy/setup.bash` before building. The deployment's `env.sh` sources the ROS overlay, then exposes OpenPI and the selected venv. `rclpy`, MoveIt interfaces and generated ROS messages come from the ROS installation/overlay, not the pip inventories. The network environment sets `ROS_DOMAIN_ID=83` and local-host discovery.

## Kernel, scheduling and network

`deploy/kernel/config-6.18.53-spring-rt` preserves the installed kernel configuration, including `CONFIG_PREEMPT_RT=y`. The kernel binary, modules, initramfs and bootloader are not included; they must be rebuilt/provisioned separately before the real-time layout can be restored. A `.config` file alone does not install PREEMPT_RT. The local libfranka patch accepts the mainline PREEMPT_RT version signature even when `/sys/kernel/realtime` is absent.

The captured layout reserves CPU 5 for control (FIFO 80) and CPU 4 for robot Ethernet IRQ/NAPI work (FIFO 90). Ordinary processes use CPUs 0–3,6–9; SMT siblings 10 and 11 are excluded. This is affinity/cgroup isolation, not `isolcpus` or `nohz_full`. `deploy/host/etc/franka/realtime.json` names the host and `enp15s0`; reusing it on different CPU topology or a different NIC is invalid.

`deploy/host/usr/local/sbin/franka-realtime-setup`, slice constraints, limits and systemd units preserve the host setup. The session launcher reapplies the layout, creates a transient `franka.slice` service with RT/memlock limits, and drops to `spring`. Native helpers do not require file capabilities.

The robot is `172.16.0.2` on Rome's dedicated `enp15s0` link. `c2-dhcp.conf` reserves that address for the recorded robot MAC. The host interface has static address `172.16.0.1/24` in NetworkManager connection `franka-c2`; restore it before enabling DHCP. Do not copy a guessed NetworkManager profile or expose the robot subnet publicly. The UI binds Rome's tailnet address `100.95.186.107:8787`; inference and idle-camera services use loopback. Tailnet enrollment, SSH keys and Desk credentials are intentionally external.

## Cameras and configuration

`config/setup.json`, `config/droid-fr3-active.json`, UI defaults and calibration files preserve the installed settings. At publication, the external camera set contains only `39338267`; `37266581` is recorded as disabled because both share one 480 Mbps upstream link. The random external selector still works when multiple cameras are enabled. The wrist is ZED Mini `12607895`, left eye, no digital zoom. Resolve the physical USB topology before re-enabling concurrent external streams.

The Robotiq 2F-85 uses the serial-by-id FTDI path in `setup.json`. The controller profile has no additional Ruckig caps or table boundary. Native collisions, target limits, workspace bounds, MoveIt self-collision validation and feedback watchdogs remain active. The earlier handoff documents their numeric values and the historical contact/reflex incidents; it is not proof of the current physical workspace state.

## Installing an update on existing Rome

The publication did not run this installer. First preview the mapping:

```bash
python3 scripts/install_rome.py --build-dir /path/to/build-output
```

An explicit `--apply` as root writes to the existing Rome layout. It refuses active controller/session/warm-up processes and a busy UI, checks all build outputs, backs up overwritten files, stops UI/model/idle-camera services, installs sources, and leaves those services stopped. Existing configuration/calibration is preserved, so code updates do not silently replace the operator's settings. `--host-config` additionally installs the captured systemd/RT/network files; inspect their hardware-specific values first. The helper does not provision OS packages, model assets, Python environments, ROS, the kernel or host IP addressing.

Ordinary startup uses the existing services and the UI. Check ownership first: `franka-ros-state.service` must be inactive for direct FCI control. **Restarting `franka-ros-moveit.service` can start ROS state via its `Requires=` dependency.** Do not blindly start every supplied unit. Read [the handoff](DESMOND_FRANKA_HANDOFF.md) for the owner arrangement and UI readiness workflow. Starting the UI never automatically resumes motion; an operator starts the session after the model is ready and the robot is in Execution.

The previous stale-timestamp fault was improved by reporting native controller errors before history dumping, distinguishing genuine stale feedback from a terminated controller. It does not prevent physical reflexes or eliminate all possible feedback loss. Current fault handlers stop rather than retrying motion automatically.
