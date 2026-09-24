# SSOG A MC2 MUX on Rome

The requested archive is installed at
`/var/lib/spring-data/workloads/pi05-final/ssog-a-mc2` (2026-09-24; the earlier copy under
`pi05-ssog-reproduction/install` is no longer referenced by the arm adapter). The OpenVINO W8A8
checkpoint-A line lives beside it in `pi05-final/ov-a` as backend `ov_a_w8a8`; see
`/var/lib/spring-data/workloads/pi05-final/README.md`.
Its shipped mux binary and the two recovery archives are SHA-256 verified.

The deployed `ssog_mc2` backend uses the archived `pi05_droid_jointpos` checkpoint,
W8A8 SigLIP/prefix/flow plus W4A8 expert, 50% visual pruning, and the fixed MC2
sampler with two model forwards on a ten-node integration grid. The existing
OpenVINO and Torch FP16 backends retain five Euler denoising steps.

Inputs use the archived quantile state normalization, tokenizer and BF16
embedding table. The mux selects the smallest 64/128/256 text window that fits
the complete prompt and state; overlong prompts are rejected. The model's
32-dimensional output is unnormalized using its own checkpoint statistics;
the first seven dimensions are added to the observed joint positions and
the first eight dimensions are returned as absolute joint positions and
absolute gripper closure.

The existing FP16 backends retain their normalized-increment decoding.
SSOG absolute joint targets are checked directly. Both paths retain the
native controller, 0.2-rad maximum target increment, joint/workspace checks,
self-collision validation, and 250-ms inference deadline. SSOG resets its
visual-pruning history when the external camera, task prompt or session
changes. The wrist camera remains included.

The new backend has no five-minute timer. Production sessions remain
continuous until Pause, Stop or a controller/inference fault.

The archived 29.410-ms smoke measurement includes NPY reads, conversions,
GPU transfers, graph execution and action-file writes. It excludes CPU
tokenization, image processing and action unnormalization. It is not a
five-step FP16 latency. First use of a new actual token count may record a
new graph; report first-use and warmed timings separately.

The provided source manifest does not match all included source files and
references missing mux build files. The exact shipped binary can be replayed;
the archive alone does not support a source-identical mux rebuild.

The archive's saved smoke action hash matched exactly on Rome. With 100
samples per window, warmed resident-server median latencies were 17.714,
20.687 and 25.585 ms for fresh episodes at 64/128/256 tokens. These exclude
the camera and CPU frontend as described above.

Eight live-camera inferences passed with the current bowl prompt, alternating
the two external left-eye cameras and always including the wrist. The first
cold request took 523.140 ms; subsequent requests took 23.155–25.969 ms. The
last-five median was 23.949 ms, below the retained 250-ms motion deadline.

The UI has SSOG selected, eight-action playback and continuous sessions.
Actual SSOG arm motion awaits Execution mode; the live readiness check
sent no robot or gripper actuation commands. Original FP16 backends remain
available in the dropdown with five denoising steps.

During installation, the preceding OpenVINO arm session stopped after
251.7 seconds because the gripper's FTDI USB adapter disconnected. The
adapter is connected and responding again. Gripper fault 9 from that
disconnect is handled by the existing startup recovery. Client cleanup now
records gripper-close errors and still closes the cameras and ROS node.

Evidence is in `results/integration-summary.json` under the isolated install
workspace and `evidence/ssog-ui-install.json` under the active arm workspace.
