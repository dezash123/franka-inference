#!/usr/bin/env python3
"""Recover pinned Rome kernel packages and tokenizer; does not install anything."""
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
manifest = json.loads((repo / 'manifests/host-artifacts.json').read_text())
for item in manifest['files']:
    relative = Path(item['destination'])
    if relative.is_absolute() or '..' in relative.parts:
        raise SystemExit('Invalid artifact destination')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', item['path']):
        raise SystemExit('Invalid source path')
dest.mkdir(parents=True)
for item in manifest['files']:
    target = dest / item['destination']
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes',
                    f'{args.host}:{item["path"]}', str(target)], check=True)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if target.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise SystemExit(f'Artifact checksum mismatch: {item["destination"]}')
    print(f'Verified {item["destination"]}', flush=True)
print(f'Recovered to {dest}. Nothing installed or executed; no reboot performed.')
