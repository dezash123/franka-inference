#!/usr/bin/env python3
"""Recover the pinned PyTorch checkpoint and FP16 IRs; never run inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('host', help='Authorized SSH destination, e.g. root@100.95.186.107')
parser.add_argument('destination', type=Path, help='New directory; existing paths are rejected')
args = parser.parse_args()
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9@._-]*', args.host):
    parser.error('Invalid SSH destination')
dest = args.destination.resolve()
if dest.exists():
    parser.error('Choose a new destination; existing data will not be overwritten')
repo = Path(__file__).resolve().parent.parent
manifest = json.loads((repo / 'manifests/fp16-artifacts.json').read_text())
prefix = '/var/lib/spring-data/workloads/robo_run/'
dest.mkdir(parents=True)
for item in manifest['files']:
    source = item['path']
    if not source.startswith(prefix):
        raise SystemExit('Unexpected artifact path in manifest')
    relative = Path(source[len(prefix):])
    if '..' in relative.parts or relative.is_absolute():
        raise SystemExit('Invalid artifact path in manifest')
    target = dest / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes',
                    f'{args.host}:{source}', str(target)], check=True)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if target.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise SystemExit(f'Artifact checksum mismatch: {relative}')
    print(f'Verified {relative}', flush=True)
print(f'Recovered to {dest}. Nothing installed or executed.')
