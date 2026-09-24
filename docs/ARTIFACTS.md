# Model assets and recovery

Git contains the adapters, serving code, transforms, launchers, tuning kernels, manifests and checksums. It does not contain multi-gigabyte checkpoints/IRs, compiled GPU plugins, native mux binaries, Python environments, Docker images, camera captures or credentials. The paths below distinguish assets recoverable from the requested S3 archive from assets that still require access to an existing host.

## Native SSOG A MC2

Source: `s3://old-local-compute-archives/jason/2026-09-23/pi05-ssog-a-mc2-mux-t64-t128-t256/`.

Use an independently authenticated AWS CLI identity authorized to read that prefix:

```bash
bash scripts/fetch_ssog.sh /path/to/archive-downloads
```

This downloads the base and mux-overlay archives and verifies `artifacts/ssog/SHA256SUMS`. It does not extract them or run inference. Never put AWS keys, SSO caches or signed URLs in Git.

The copied archive `install.sh` verifies the downloads, extracts the layout, verifies the shipped binary, and builds the `ssog-b580:neo2513` image if absent. Its default behavior can replace an installation and delete `.installing`/`.previous` directories: use a **new empty destination**, with neither suffix present. It prints a smoke-test command but does not execute that test. For a restore, invoke it from the download directory with an explicit destination, then place that installation at the path expected by the adapter:

```bash
bash /path/to/archive-downloads/install.sh /path/to/new-ssog-install
# Deployed adapter expects /var/lib/spring-data/workloads/pi05-final/ssog-a-mc2
```

The archive's `/work` symlinks are intentional container paths. Preserve them. The base archive contains weights, tokenizer, embedding table, unit transforms and runtime; the overlay supplies the resident T64/T128/T256 mux. `artifacts/ssog/` preserves Rome's small launcher/transform files for comparison and restoration after extraction.

| Item | SHA-256 |
|---|---|
| Base archive | `75faa985d62d03d4a2ffd91310c46be655b211b861fc0045f828de3893511d66` |
| Mux overlay | `fd419f9fa98d306e24a73adefcd8e437418d0edbba8b53afe883c53cea0e3579` |
| `muxfinal/bin/fused_mux_droid` | `db89756ca6b0451040cbbf6e3490d28606d5ed510de3d4cc371bfce3259d6edb` |
| Canonical manifest file | `0376a60ab7c1833c6dcdfa7bd04281a8c1f0a2804e8c45743738a2c5001ff103` |

The Dockerfile pins the archive's NEO 25.13 user-space runtime; the host's newer Intel GPU stack serves the other backends. Do not flatten those runtimes into one environment. Exact replay of the shipped mux binary is supported; the archive omits some mux build inputs and contains source/manifest discrepancies, so an identical rebuild from the available sources is not established.

Historical latency and smoke receipts are in `manifests/ssog-*.json`, `artifacts/ssog/TARGET-VERIFICATION.json` and the handoff. Their timing boundary excludes camera acquisition and the CPU frontend. No benchmark or offline inference was rerun for this publication.

## OpenVINO W8A8, checkpoint A

Active path: `/var/lib/spring-data/workloads/pi05-final/ov-a`. This is the `ov_a_w8a8` backend: `pi05_droid_jointpos`, five Euler steps, T64 and T256 workers, custom Intel GPU plugin. **Its checkpoint-A IRs are not in the SSOG S3 archive.** A similarly named continuation bundle contains checkpoint B; substituting its weights changes the policy even when XML geometry matches.

Recover the exact assets from an authorized Rome SSH account into a new destination:

```bash
bash scripts/fetch_ov_a.sh root@100.95.186.107 /path/to/new-ov-a
```

The helper combines repository sources with Rome's `model/`, `custom-plugin-cams-r11/`, `wheels/`, tokenizer and embedding assets, then checks `SHA256SUMS.pinned`. It reads remote files only and excludes runs/camera observations. The source host must still retain these assets. Original IR provenance is `sleepy-joe:/home/spring/ovckptA`; [the final-model snapshot](PI05_FINAL_SNAPSHOT.md) records the validation lineage. This Git repository alone is not a complete disaster-recovery copy of checkpoint A.

In the recovered tree, `setup_venv.sh` installs the pinned Python 3.12 wheels. It recreates that tree's `venv`, so use it on the new restore only. `env-local.sh` regenerates `plugins.local.xml` and prepends the custom plugin; `serve_local.sh` retargets the shipped `/workflow` kernel paths into the recovered tree. The referenced tuning/kernel sources are included. On Rome, use host GPU runtime defaults; the optional `PI05_USE_BUNDLED_RUNTIME=1` paths require runtime files beyond this recovery helper.

`SHA256SUMS.pinned` identifies IRs, plugin, server, tuning, normalization, tokenizer and transform implementation. Critical weight hashes:

- T64 `.bin`: `d84fcd2cf4f0f307a2218a0f94b8e546239ae6683894a13ec68af0cd6d0ddb8b`
- T256 `.bin`: `785a45d0da322245942ab4f4ad9fe7bb04bf91e42207e66ff65d7dc124bd36da`
- GPU plugin: `0dac1fd9d724ad025899dbdc6abbcdad50b1144ff7c1656a90b8bea36e493b96`

T128 is not resident in the active OpenVINO line because three workers do not fit the B580's memory. The source/tuning set also includes other geometry files for provenance; that does not mean they are active. The launcher records the current GPU clock floor without changing it.

## Torch and original OpenVINO FP16

These backends use **`pi05_droid`**, distinct from the absolute-joint checkpoint-A model. Existing assets:

- Original Orbax checkpoint: `/home/spring/robo_run/checkpoints/pi05_droid`
- Converted PyTorch checkpoint and normalization assets: `/home/spring/robo_run/checkpoints/pi05_droid_pytorch`
- FP16 IRs: `/var/lib/spring-data/workloads/robo_run/franka_droid/openvino_fp16`

`manifests/fp16-artifacts.json` records freshly verified hashes and sizes for the converted checkpoint, normalization/configuration, and the three FP16 IRs. Recover those pinned files from Rome into a new directory (about 13.7 GB):

```bash
python3 scripts/fetch_fp16.py root@100.95.186.107 /path/to/new-fp16-restore
```

The helper recreates the `checkpoints/` and `franka_droid/openvino_fp16/` layout beneath that destination, then verifies every file. It omits generated runtime caches and does not install into the active workspace. For recovery from upstream, the original checkpoint is in the public `gs://openpi-assets/checkpoints/pi05_droid/` prefix. `scripts/download_pi05_droid.py` downloads to the invoking user's `~/robo_run/checkpoints/pi05_droid`; run it as `spring` for the deployed layout. `scripts/convert_pi05_droid.py` calls the pinned OpenPI Orbax-to-PyTorch converter with BF16 storage using fixed Rome paths. This is a large conversion job, not a service-start command.

`franka_droid/export_openvino.py` exports `vision`, `prefix`, and `denoise` IRs separately from that checkpoint, compressing weights to FP16. Run those named export stages in the correctly provisioned environment only when rebuilding the assets. Its separate `reference` stage is a historical offline comparison utility, uses saved observations and a different sample-step setting, and is not part of deployment or publication validation. The deployed adapters use five denoise steps.

`runtime_contract.py` preserves checkpoint and action interpretation across model selection. The FP16 paths output normalized joint increments; both checkpoint-A paths apply their checkpoint-specific transforms and return absolute joint targets. All use an absolute gripper-closure channel. Matching tensor shapes alone is insufficient for switching models.

## Kernel packages and OpenPI tokenizer

`manifests/host-artifacts.json` records full SHA-256 hashes of Rome's exact `6.18.53-spring-rt` image/header/libc Debian packages (package version `6.18.53-1`) and the cached OpenPI tokenizer. Recover them without installation or reboot:

```bash
python3 scripts/fetch_host_assets.py root@100.95.186.107 /path/to/new-host-assets
```

The helper copies about 83 MB into `kernel/` and `openpi-cache/big_vision/`, verifies every file, and refuses an existing destination. Kernel packages are currently retained under `/var/lib/spring-data/kernel-rt-20260923`; they are not in the SSOG S3 prefix. Kernel installation, initramfs generation and boot selection are separate provisioning steps, and were not performed during publication.

The FP16 adapters use OpenPI's `PaligemmaTokenizer`, which fetches `gs://big_vision/paligemma_tokenizer.model` anonymously if absent. Rome's cached file is `/home/spring/.cache/openpi/big_vision/paligemma_tokenizer.model`, SHA-256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`. Restore the verified file to that cache path, or set `OPENPI_DATA_HOME` to the recovered `openpi-cache` directory for the service environment. This avoids an implicit first-load network dependency. SSOG and OpenVINO A have their own tokenizer copies in their recovered asset trees.

`manifests/python-path-bridges.json` records the exact environment-linking `.pth` contents; it does not contain packages. `manifests/ssog-weight-inventory.json` records the archive index's expected served weight hashes and sizes. Its `unserved.safetensors` entry is explicitly not served and is absent from the deployed tree; it is not a missing inference dependency.
