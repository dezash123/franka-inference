#!/usr/bin/env python3
"""Preview/install source and built helpers on the existing Rome layout; no motion."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from urllib.request import urlopen

repo = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--apply', action='store_true', help='Write files after idle/host checks')
parser.add_argument('--host-config', action='store_true', help='Also install captured RT/network/systemd configuration')
parser.add_argument('--build-dir', type=Path, default=repo / '.build/native')
args = parser.parse_args()
app = Path('/var/lib/spring-data/workloads/robo_run/franka_droid')
paths = []
for srcdir, dst in [('franka_droid', app), ('camera', Path('/home/spring/zed_camera_setup')),
                    ('deploy/robot-network', Path('/var/lib/spring-data/franka-network-20260923')),
                    ('deploy/user-systemd', Path('/home/spring/.config/systemd/user'))]:
    for src in (repo / srcdir).rglob('*'):
        if src.is_file() and '__pycache__' not in src.parts:
            target = dst / src.relative_to(repo / srcdir)
            if srcdir == 'franka_droid' and 'config' in src.relative_to(repo / srcdir).parts and target.exists():
                continue  # Preserve the installed robot calibration/limits/settings.
            paths.append((src, target))
for name in ['stream_fci', 'read_fci_errors', 'read_fci_snapshot', 'read_fci_realtime', 'recover_fci']:
    paths.append((args.build_dir / name, app / name))
paths.append((args.build_dir / 'camera/libcapture_v4l2.so', Path('/home/spring/zed_camera_setup/build/libcapture_v4l2.so')))
if args.host_config:
    for src in (repo / 'deploy/host').rglob('*'):
        if src.is_file():
            paths.append((src, Path('/') / src.relative_to(repo / 'deploy/host')))
print(json.dumps({'apply': args.apply, 'host_config': args.host_config, 'files': [str(dst) for _, dst in paths]}, indent=2))
if not args.apply:
    raise SystemExit(0)
if os.geteuid() != 0:
    raise SystemExit('Run --apply as root')
expected = json.loads((repo / 'deploy/host/etc/franka/realtime.json').read_text())['hostname']
if socket.gethostname() != expected:
    raise SystemExit('Rome-only deployment: review paths, hardware identities and CPU topology for another host')
missing = [str(src) for src, _ in paths if not src.is_file()]
if missing:
    raise SystemExit('Build/recover the missing inputs first: ' + ', '.join(missing))
for proc in Path('/proc').glob('[0-9]*'):
    try:
        if (proc / 'exe').resolve().name == 'stream_fci':
            raise SystemExit('Arm controller active; stop through its owner first')
        argv = (proc / 'cmdline').read_bytes().split(b'\0')
        if any(Path(os.fsdecode(a)).name in ('run_policy_stream.py', 'warmup_policy_live.py') for a in argv if a):
            raise SystemExit('Arm session or warm-up active; wait or stop through its owner')
    except OSError:
        pass
try:
    with urlopen('http://100.95.186.107:8787/api/status', timeout=3) as response:
        state = json.load(response)
except OSError:
    state = None
if state and state.get('busy'):
    raise SystemExit('UI operation active; wait until idle')
backup = app / 'backups' / ('repository-install-' + str(time.time_ns()))
backup.mkdir(parents=True)
subprocess.run(['systemctl', 'stop', 'franka-control-ui.service'], check=True)
for name in ('franka-droid-policy.service', 'zed-camera-viewer.service'):
    subprocess.run(['runuser', '-u', 'spring', '--', 'env', 'XDG_RUNTIME_DIR=/run/user/1000',
                    'systemctl', '--user', 'stop', name], check=True)
for src, dst in paths:
    if dst.exists():
        saved = backup / dst.relative_to('/')
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, saved)
    dst.parent.mkdir(parents=True, exist_ok=True)
    staged = dst.with_name('.repo-install-' + dst.name)
    shutil.copy2(src, staged)
    if str(dst).startswith(('/home/spring/', str(app))):
        os.chown(staged, 1000, 1000)
    os.replace(staged, dst)
for folder in [app / 'evidence', app / 'logs', Path('/home/spring/zed_camera_setup/evidence')]:
    folder.mkdir(parents=True, exist_ok=True);os.chown(folder, 1000, 1000)
subprocess.run(['systemctl', 'daemon-reload'], check=True)
subprocess.run(['runuser', '-u', 'spring', '--', 'env', 'XDG_RUNTIME_DIR=/run/user/1000',
                'systemctl', '--user', 'daemon-reload'], check=True)
print(f'Installed; backup at {backup}. Services remain stopped. No robot commands sent.')
