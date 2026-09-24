#!/usr/bin/env bash
# Recover the exact checkpoint-A assets from an authorized existing Rome host.
# It reads remote files only; does not change clocks, run inference, or restart services.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_host=${1:?Usage: fetch_ov_a.sh root@rome /path/to/new-ov-a}
dest_dir=${2:?Usage: fetch_ov_a.sh root@rome /path/to/new-ov-a}
case "$source_host" in -*|*[!a-zA-Z0-9@._-]*) echo 'Invalid SSH host' >&2; exit 2;; esac
if [ -e "$dest_dir" ]; then echo 'Choose a new destination; existing data will not be overwritten.' >&2; exit 2; fi
mkdir -p "$dest_dir"
source_dir=/var/lib/spring-data/workloads/pi05-final/ov-a
# Preserve the small shipped server, transforms, tuning and kernel sources.
rsync -a "$repo_dir/artifacts/ov-a/" "$dest_dir/"
# Omit runs, observations, verification cases, virtualenvs and secrets.
for part in model custom-plugin-cams-r11 wheels; do
  rsync -a -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' "$source_host:$source_dir/$part" "$dest_dir/"
done
mkdir -p "$dest_dir/unit/assets_ckptA"
rsync -a -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' \
  "$source_host:$source_dir/unit/assets_ckptA/" "$dest_dir/unit/assets_ckptA/"
(cd "$dest_dir" && sha256sum -c SHA256SUMS.pinned)
printf 'Recovered and verified checkpoint-A assets in %s. No inference was run.\n' "$dest_dir"
