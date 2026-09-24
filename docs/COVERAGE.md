# Publication coverage and verification

The repository is a source/configuration export of the deployed Rome system. The latest audit is [setup-audit.json](../manifests/setup-audit.json); [OPERATIONS.md](OPERATIONS.md) describes the corresponding current arrangement. Historical receipts retain their original dates and results.

## What is preserved

| Component | In Git | External recovery / remaining requirement |
|---|---|---|
| Arm client, native controller and UI | Active Python/C++/headers, browser assets, settings, calibration, build helper | Compile against the pinned native dependencies |
| Camera capture and previews | V4L2 C helper, serial discovery, capture/viewer/publisher and calibration | Build the shared library; physical USB devices and wiring |
| Real-time host | Kernel `.config`, CPU/IRQ setup, RT limits, slices and units | Pinned kernel Debian packages via `fetch_host_assets.py`; host boot provisioning |
| Robot network and ROS | DHCP configuration, URDF, launch files, pinned ROS repositories and patches | ROS Jazzy packages/overlay build, NetworkManager address, physical robot and Desk configuration |
| Python | Effective/raw version inventories, module origins, exact `.pth` bridges, source pins | Python 3.12 environments, XPU wheels, OpenPI Transformers replacements, ROS packages |
| Torch / OpenVINO FP16 | Adapters, export/conversion scripts, 12-file checksum manifest | `fetch_fp16.py` from Rome; upstream checkpoint alternative documented |
| Native SSOG A MC2 | Adapter, launcher, CPU transforms, archive hashes, weight inventory and historical receipts | `fetch_ssog.sh` from S3; archive contains weights/binary and Docker recipe |
| OpenVINO W8A8 A | Adapter, server, transforms, tuning kernels, launchers and pinned checksums | `fetch_ov_a.sh` from Rome for checkpoint-A IRs, plugin, wheels and embeddings |
| OpenPI tokenizer | Exact cached file hash and recovery mapping | `fetch_host_assets.py`; upstream anonymous GCS path documented |
| Runtime evidence | Sanitized receipts and provenance summaries | Full sessions, images, model caches and logs remain on Rome |
| Access | Hostnames, ports and service arrangement | SSH keys, Tailscale identity, AWS authentication and Desk login remain private/external |

`manifests/rome-files.json` maps each exported file to its original Rome path and hash. `published_sha256` identifies the checked-in content. Differences are explicitly annotated for repository documentation edits and historical banners. Source revisions and all three local patches are in `manifests/sources.json` and `patches/`.

The exported configuration is the saved snapshot, not a command to overwrite the live operator's choices. `install_rome.py` preserves existing configuration and calibration. Historical host documents may disagree about prompt, action count, camera selection, previews or stopped/running state; the current operations guide and dated latest audit take precedence for this publication.

## Verification and its limits

The current audit reads the host on housekeeping CPUs. It compares every recorded source/configuration file, all 11 upstream revisions, the complete local patch bytes, artifact presence and known sizes, small artifact checksums, service state and active controller scheduling. It reads existing HTTP status endpoints; it does not open cameras, gripper, GPU inference or FCI. Large model weights are not rehashed as part of this lightweight audit; their original checksum receipts remain authoritative for the earlier verification time. Recovery helpers verify their downloaded pinned artifacts in full.

The current saved settings were refreshed from Rome. A concurrent update to `run_policy_stream.py` landed during the audit; the deployed absolute-target step rescaling was copied and documented, including its next-session activation. Other recorded runtime sources matched. Repository-only documentation changes are recorded separately. The active-directory inventory was reviewed against native includes, Python imports and launchers: old `run_policy_fci.py`/`run_policy_ros.py`, `direct_fci.cpp`, `stream_fci_common.hpp`, `droid_hybrid_feedback.hpp`, `decrease_limits.py` and previous diagnostic/test experiments are superseded paths, not dependencies of `run_policy_stream.py` / `stream_fci.cpp`. The one-time ROS load helper is unnecessary for this controller: `stream_fci.cpp` sets and checks the payload itself.

| Receipt | Meaning |
|---|---|
| `publication.json` | Initial source export: successful isolated native compile, syntax/package integrity checks; no programs executed |
| `camera-ui-validation.json` | Earlier three-camera UI update |
| `live-preview-validation.json` | Subsequent camera-only MJPEG check, approximately 15 FPS for all three; no arm/model run |
| `setup-audit.json` | Latest read-only deployed-file, dependency, artifact and status comparison |
| `repository-validation.json` | Current syntax, documentation links, pinned inputs and publication integrity checks |
| `ssog-*.json`, `artifacts/ssog/TARGET-VERIFICATION.json` | Historical model integration/latency receipts, not new measurements |

No offline test suite, model benchmark, arm restart or fresh-machine installation was performed for the completeness audit. The native source was unchanged since its recorded successful build, so it was not rebuilt again. The new host-asset recovery helper was exercised as a read-only download and full checksum verification; no downloaded package was installed.

## Remaining restore limitations

This Git repository alone cannot restore every byte of the deployment. OpenVINO checkpoint-A IRs/plugin and the captured kernel packages still need an existing host; they are not covered by the SSOG S3 archive. The converted FP16 checkpoint/IRs can be copied from Rome or rebuilt from upstream with the recorded tooling. Python inventories are not standalone dependency locks. Native dependencies, ROS and boot setup require provisioning; an unattended clean-machine restore has not been demonstrated.

The SSOG archive supports replay of its shipped executable but lacks some original mux build inputs, so an identical rebuild from source is not established. Model latency does not establish manipulation success. These limitations are preserved explicitly rather than presenting the source export as a disk backup or a newly validated robot deployment.

Bundled standalone license files were removed at the user's request. [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) retains upstream provenance references. No model binaries, camera imagery, private credentials or full session logs are pushed.
