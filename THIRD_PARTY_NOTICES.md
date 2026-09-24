# Third-party sources and artifacts

This export does not relicense third-party software or model assets. Their original terms apply. No blanket license is asserted for all contents of this deployment snapshot.

- **OpenPI**, Physical Intelligence: pinned repository and commit in `manifests/sources.json`; Apache-2.0 text in `third_party/licenses/openpi-LICENSE`. Model assets are recovered separately and remain subject to their upstream terms.
- **libfranka**, Franka Robotics: direct and ROS revisions are pinned separately. Corresponding license text is retained in `third_party/licenses/`; local patches are in `patches/`.
- **DROID**, `droid-dataset/droid`, commit `33ae6a67274f36d2e29525b86f23a56616ef43a7`: source URLs for hardware constraints and action mapping are recorded in `franka_droid/config/reference/sources.json`. No root LICENSE file was found in the queried upstream repository. This export makes no license grant on DROID's behalf. Upstream reference code can be retrieved from those pinned URLs; it is not required at runtime.
- **fairo / Polymetis**, Meta: reference implementation at `0a01a7fa7a7c65b2f9a3aebf5e79040940daf9d2`; MIT license in `third_party/licenses/fairo-LICENSE`. The local controller's DROID-style feedback and constraints retain their source references.
- **ROS dependencies**: repository URLs and revisions appear in `manifests/sources.json`. License files captured from the installed checkouts are retained under `third_party/licenses/`. Dependency checkouts, not a merged vendor tree, are recovered by `scripts/fetch_sources.py`.
- **SSOG and custom OpenVINO model runtimes**: small launchers, transforms, server and kernel sources are preserved from the user-designated archive and Rome installation. Artifact manifests record provenance and hashes; binaries/weights are external. No additional distribution or model license is inferred from the archive's availability.
- **Intel / PyTorch / OpenVINO / Stereolabs**: runtime dependencies and device calibration data retain their respective upstream terms. The repository does not include vendor driver packages, ZED SDK binaries or Docker images.

For file-level provenance, see `manifests/rome-files.json`. Historical documents retain their original paths and dates, with a publication-snapshot notice added.
