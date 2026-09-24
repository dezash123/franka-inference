#!/usr/bin/env bash
# Download and hash-check the archived assets. Does not extract or run them.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cache_dir=${1:?Usage: fetch_ssog.sh /path/to/download-directory}
mkdir -p "$cache_dir"
cache_dir=$(cd "$cache_dir" && pwd)
archive_prefix=s3://old-local-compute-archives/jason/2026-09-23/pi05-ssog-a-mc2-mux-t64-t128-t256
for name in pi05-b580-ssog-a-mc2.tar.zst pi05-ssog-a-mc2-mux-overlay.tar.zst; do
  aws s3 cp "$archive_prefix/$name" "$cache_dir/$name"
done
cp "$repo_dir/artifacts/ssog/"{SHA256SUMS,CANONICAL.sha256,install.sh} "$cache_dir/"
(cd "$cache_dir" && sha256sum -c SHA256SUMS)
printf 'Verified downloads in %s. Read docs/ARTIFACTS.md before extraction.\n' "$cache_dir"
