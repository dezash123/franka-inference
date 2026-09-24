"""Bounded, read-only view of the existing policy journals. No robot commands."""
import json
import math
import threading
import time
from collections import deque


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


class JournalTail:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.identity = None
        self.pending = b''

    def read(self):
        try:
            with self.path.open('rb') as stream:
                stat = self.path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.offset = 0
                    self.pending = b''
                    self.identity = identity
                # Opening an old session or reconnecting never scans the full log.
                limit = 2 * 1024 * 1024
                skipped = stat.st_size - self.offset > limit
                if skipped:
                    self.offset = stat.st_size - limit
                    self.pending = b''
                stream.seek(self.offset)
                data = stream.read(limit)
                self.offset += len(data)
            if skipped:
                data = data.partition(b'\n')[2]
            lines = (self.pending + data).split(b'\n')
            self.pending = lines.pop()[-65536:]
            result = []
            for line in lines:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        result.append(item)
                except (ValueError, UnicodeError):
                    pass
            return result
        except OSError:
            return []


class LiveTelemetry:
    def __init__(self):
        self.lock = threading.Lock()
        self.path = None
        self.updated = 0
        self.cached = {}

    def reset(self, path):
        self.path = path
        self.events = JournalTail(path.with_suffix('.events.jsonl'))
        self.states = JournalTail(path.with_suffix('.states.jsonl'))
        self.actions = deque(maxlen=6000)
        self.chunks = deque(maxlen=1000)
        self.holds = deque(maxlen=1000)
        self.measured = deque(maxlen=2000)
        self.pauses = {}
        self.last_sample = float('-inf')
        self.updated = 0

    def event(self, kind, entry):
        t = entry.get('dispatch_t', entry.get('t'))
        if not number(t):
            return
        if kind == 'actions':
            self.actions.append({k: entry.get(k) for k in ('step', 'chunk', 'index', 'q', 'target') } | {'t': t})
        elif kind == 'chunks':
            ms = entry.get('inference_ms')
            if not number(ms):
                return
            self.chunks.append({'id': entry.get('id'), 't': t, 'start': t - ms / 1000,
                                'ms': ms, 'model_ms': (entry.get('policy_timing') or {}).get('infer_ms'),
                                'camera': (entry.get('external_view') or {}).get('serial')})
        elif kind in ('joint_limit_events', 'validation_hold_events'):
            self.holds.append({'t': t, 'duration': entry.get('duration_s', 0),
                               'reason': entry.get('reason', kind), 'joints': entry.get('joints', [])})
        elif kind in ('pause_events', 'pause_resumed', 'pause_ended'):
            self.pauses[t] = {'t': t, 'end': entry.get('resumed_t', t + entry['duration_s'] if 'duration_s' in entry else None)}

    def read(self, path, phase):
        if path is None:
            return {'session': None, 'phase': phase, 'actions': [], 'chunks': [], 'measured': [], 'holds': [], 'pauses': []}
        with self.lock:
            if path != self.path:
                self.reset(path)
            now = time.monotonic()
            if now - self.updated < .25:
                return dict(self.cached, phase=phase)
            try:
                receipt = json.loads(path.read_text())
            except (OSError, ValueError):
                return dict(self.cached, phase=phase)
            journal = path.with_suffix('.events.jsonl').exists()
            if journal:
                for item in self.events.read():
                    if isinstance(item.get('entry'), dict):
                        self.event(item.get('kind'), item['entry'])
            else:
                self.actions.clear(); self.chunks.clear(); self.holds.clear(); self.pauses.clear()
                for kind in ('actions', 'chunks', 'joint_limit_events', 'validation_hold_events', 'pause_events'):
                    for item in receipt.get(kind, []):
                        self.event(kind, item)
            base = receipt.get('control_started_monotonic')
            if number(base):
                for item in self.states.read():
                    stamp = item.get('stamp')
                    state = item.get('state')
                    if not number(stamp) or not isinstance(state, dict) or not isinstance(state.get('q'), list):
                        continue
                    t = stamp - base
                    if t < 0 or t - self.last_sample < .045:
                        continue
                    self.measured.append({'t': t, 'q': state['q']})
                    self.last_sample = t
            finished = receipt.get('finished_at_unix')
            started = receipt.get('control_started_unix')
            live = not finished and phase in ('playing', 'pausing', 'paused', 'resuming', 'stopping')
            end = receipt.get('actual_control_s', receipt.get('elapsed_s', 0)) or 0
            if live and number(base):
                end = max(0, now - base)
            elif live and number(started):
                end = max(0, time.time() - started)
            end = max(end, self.actions[-1]['t'] if self.actions else 0,
                      self.measured[-1]['t'] if self.measured else 0)
            cutoff = end - 60
            for queue in (self.actions, self.chunks, self.holds, self.measured):
                while len(queue) > 1 and queue[1]['t'] < cutoff:
                    queue.popleft()
            self.pauses = {t: p for t, p in self.pauses.items() if p['end'] is None or p['end'] >= cutoff}
            # Older receipts have aligned feedback at dispatch; never guess a clock offset.
            measured = list(self.measured) if number(base) else [{'t': a['t'], 'q': a['q']} for a in self.actions]
            self.cached = {'session': path.name, 'phase': phase, 'live': live, 'end': end,
                           'backend': receipt.get('policy_metadata', {}).get('backend'),
                           'actions': list(self.actions), 'chunks': list(self.chunks),
                           'measured': measured, 'holds': list(self.holds), 'pauses': list(self.pauses.values()),
                           'measured_source': 'controller' if number(base) else 'action_dispatch',
                           'sample_age_s': max(0, end - measured[-1]['t']) if measured else None,
                           'failure': receipt.get('failure'), 'finished': bool(finished)}
            self.updated = now
            return self.cached
