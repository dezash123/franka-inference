#!/usr/bin/env python3
"""Clone exact dependency revisions and apply the recorded local patches."""
import argparse
import json
import pathlib
import subprocess

repo = pathlib.Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--destination', type=pathlib.Path, required=True)
parser.add_argument('--only', action='append', help='Manifest key; repeat to select several')
args = parser.parse_args()
pins = json.loads((repo / 'manifests/sources.json').read_text())
if args.only and set(args.only) - pins.keys():
    parser.error('Unknown dependency key')
for name, pin in pins.items():
    if args.only and name not in args.only:
        continue
    dst = args.destination.resolve() / name
    if dst.exists():
        raise SystemExit(f'Refusing to overwrite existing dependency checkout: {dst}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', pin['url'], str(dst)], check=True)
    subprocess.run(['git', '-C', str(dst), 'checkout', '--detach', pin['commit']], check=True)
    subprocess.run(['git', '-C', str(dst), 'submodule', 'update', '--init', '--recursive'], check=True)
    if pin.get('patch'):
        subprocess.run(['git', '-C', str(dst), 'apply', '--check', str(repo / pin['patch'])], check=True)
        subprocess.run(['git', '-C', str(dst), 'apply', str(repo / pin['patch'])], check=True)
    print(f'{name}: {pin["commit"]}')
