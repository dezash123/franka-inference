#!/usr/bin/env python3
"""Read-only comparison with Rome. No inference, device access or service changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


REMOTE = r'''
import hashlib,json,os,pathlib,subprocess,sys,urllib.request
x=json.load(sys.stdin)
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=15)
    return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
files=[]
for item in x['files']:
    p=pathlib.Path(item['source'])
    row={'path':item['path'],'exists':p.is_file()}
    if p.is_file():
        row.update(sha256=digest(p),bytes=p.stat().st_size)
        row['matches']=row['sha256']==item['sha256']
    files.append(row)
sources={}
for name,item in x['sources'].items():
    base=['runuser','-u','spring','--','git','-C',item['path']]
    head=command(base+['rev-parse','HEAD'])
    diff=command(base+['diff','HEAD','--binary'])
    patch=diff['stdout']
    status=command(base+['status','--porcelain'])
    sources[name]={'commit':head['stdout'].strip(),'status':status['stdout'].splitlines(),
        'matches':head['returncode']==0 and diff['returncode']==0 and status['returncode']==0
            and head['stdout'].strip()==item['commit']
            and (hashlib.sha256(patch.encode()).hexdigest() if patch else None)==item['patch_sha256']}
assets=[]
for item in x['assets']:
    p=pathlib.Path(item['path']);row={'path':str(p),'exists':p.is_file()}
    if p.is_file():
        row['bytes']=p.stat().st_size
        row['size_matches']=item.get('bytes',row['bytes'])==row['bytes']
        if row['bytes']<=16*1024*1024 and item.get('sha256'):
            row['sha256']=digest(p);row['hash_matches']=row['sha256']==item['sha256']
        else:row['hash_check']='Not rehashed: large asset or no pinned digest'
    assets.append(row)
live={}
for name,url in [('ui','http://100.95.186.107:8787/api/status'),('cameras','http://100.95.186.107:8787/api/cameras')]:
    try:
        with urllib.request.urlopen(url,timeout=4) as response:obj=json.load(response)
        if name=='ui':
            robot=obj.get('robot') or {}
            obj={k:obj.get(k) for k in ['phase','busy','ready','settings','server_unix']}
            obj['robot']={k:robot.get(k) for k in ['operating_mode','fci_active','robot_errors']}
        else:
            obj={'cameras':[{k:c.get(k) for k in ['serial','role','eye','inference_enabled','fps','healthy','source']} for c in obj.get('cameras',[])]}
        live[name]=obj
    except Exception as e:live[name]={'error':str(e)}
units=['franka-control-ui.service','franka-ros-state.service','franka-ros-moveit.service','franka-realtime.service','franka-c2-dhcp.service']
services=command(['systemctl','show',*units,'-p','Id,ActiveState,SubState,UnitFileState,FragmentPath,DropInPaths'])
user_services=command(['runuser','-u','spring','--','env','XDG_RUNTIME_DIR=/run/user/1000','systemctl','--user','show','franka-droid-policy.service','zed-camera-viewer.service','-p','Id,ActiveState,SubState,UnitFileState,FragmentPath,DropInPaths'])
threads=[]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        if (proc/'exe').resolve().name!='stream_fci':continue
        for task in (proc/'task').iterdir():
            tid=int(task.name)
            threads.append({'name':(task/'comm').read_text().strip(),'cpus':sorted(os.sched_getaffinity(tid)),
                'scheduler':os.sched_getscheduler(tid),'priority':os.sched_getparam(tid).sched_priority})
    except (OSError,ValueError):pass
print(json.dumps({'files':files,'sources':sources,'assets':assets,'live':live,
    'services':services,'user_services':user_services,'kernel':os.uname().release,'controller_threads':threads}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', nargs='?', default='root@100.95.186.107')
    parser.add_argument('--output', type=Path, help='Write the sanitized report locally')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9@._-]*', args.host):
        parser.error('Invalid SSH destination')
    repo = Path(__file__).resolve().parent.parent
    def load(name):
        return json.loads((repo / 'manifests' / name).read_text())
    files = load('rome-files.json')
    local_mismatches = [x['path'] for x in files if not (repo / x['path']).is_file()
                        or hashlib.sha256((repo / x['path']).read_bytes()).hexdigest() != x['published_sha256']]
    sources = load('sources.json')
    for item in sources.values():
        item['patch_sha256'] = hashlib.sha256((repo / item['patch']).read_bytes()).hexdigest() if item.get('patch') else None
    assets = load('fp16-artifacts.json')['files'] + load('host-artifacts.json')['files']
    ov = '/var/lib/spring-data/workloads/pi05-final/ov-a/'
    for line in (repo / 'artifacts/ov-a/SHA256SUMS.pinned').read_text().splitlines():
        sha, path = line.split(maxsplit=1)
        assets.append({'path': ov + path, 'sha256': sha})
    ssog = '/var/lib/spring-data/workloads/pi05-final/ssog-a-mc2/'
    for item in load('ssog-weight-inventory.json')['files']:
        if item['served']:
            assets.append({**item, 'path': ssog + 'campaign13/share/weights-A/' + item['file']})
    assets.append({'path': ssog + 'muxfinal/bin/fused_mux_droid',
                   'sha256': 'db89756ca6b0451040cbbf6e3490d28606d5ed510de3d4cc371bfce3259d6edb'})
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', args.host,
                             'taskset -c 0-3,6-9 python3 -c ' + shlex.quote(REMOTE)],
                            input=json.dumps({'files': files, 'sources': sources, 'assets': assets}),
                            text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise SystemExit(result.stderr)
    report = json.loads(result.stdout)
    problems = {'local_manifest_mismatches': local_mismatches,
                'remote_file_changes': [x['path'] for x in report['files'] if not x.get('matches')],
                'source_revision_or_patch_changes': [k for k, v in report['sources'].items() if not v['matches']],
                'artifact_problems': [x['path'] for x in report['assets'] if not x['exists']
                                      or x.get('size_matches') is False or x.get('hash_matches') is False]}
    report.update(checked_utc=datetime.now(timezone.utc).isoformat(), host=args.host, problems=problems,
                  source_and_asset_checks_passed=not any(problems.values()),
                  scope='Read-only file/revision comparison, asset presence/small hashes, existing UI status and process scheduling. No offline tests, model execution, device access, service changes or robot commands. Large asset hashes retain their original verification dates.')
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
        print(json.dumps({'output': str(args.output), 'problems': problems,
                          'files': len(files), 'sources': len(sources), 'assets': len(assets)}, indent=2))
    else:
        print(output, end='')
    return int(any(problems.values()))


if __name__ == '__main__':
    raise SystemExit(main())
