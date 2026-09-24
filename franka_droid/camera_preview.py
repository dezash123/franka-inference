"""Bounded camera telemetry/JPEG publication on a housekeeping thread.

Reuses the inference camera owner. Preview-only cameras never enter observations
or the required-camera freshness checks, and their failure cannot stop motion.
"""
import json
import logging
import os
import threading
import time

import cv2
from cameras import LatestCamera, ZedSource
from contract import ROOT


class CameraPreview:
    def __init__(self, streams, cfg):
        self.stop = threading.Event()
        self.pid = os.getpid()
        self.directory = ROOT / 'evidence'
        self.status_path = self.directory / 'camera-preview.json'
        self.entries = []
        self.owned = []
        self.files = set()
        active = [*cfg['external_cameras'], cfg['wrist_camera']]
        for index, (stream, camera) in enumerate(zip(streams, active)):
            self.entries.append(dict(serial=camera['serial'], role='wrist' if index == len(active)-1 else 'external',
                                     stream=stream, inference_enabled=True, config=camera))
        serials = {entry['serial'] for entry in self.entries}
        for camera in cfg.get('external_cameras_disabled', []):
            if camera['serial'] not in serials:
                self.entries.append(dict(serial=camera['serial'], role='external', stream=None,
                                         inference_enabled=False, config=camera, retry=0., error=None))
                serials.add(camera['serial'])
        self.thread = threading.Thread(target=self._run, name='camera-preview', daemon=True)
        self.thread.start()

    def _write(self, path, data):
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(data)
        os.replace(temporary, path)
        self.files.add(path)

    def _publish(self):
        now = time.monotonic()
        cameras = []
        for entry in self.entries:
            if entry['stream'] is None and now >= entry.get('retry', 0):
                try:
                    stream = LatestCamera(ZedSource({**entry['config'], 'eye': 'left'}))
                    entry['stream'] = stream
                    self.owned.append(stream)
                    entry['error'] = None
                except Exception as exc:
                    entry['error'] = str(exc)
                    entry['retry'] = now + 5
            camera = {key: entry[key] for key in ('serial', 'role', 'inference_enabled')}
            camera.update(fps=0., healthy=False, age_seconds=None, frame_count=0,
                          error=entry.get('error'), image=None)
            stream = entry['stream']
            if stream is not None:
                frame, metrics = stream.preview()
                camera.update(metrics)
                if frame is not None and metrics['healthy']:
                    rgb = frame[0]
                    if isinstance(rgb, dict):
                        rgb = rgb['left']
                    if rgb.shape[1] > 672:
                        rgb = cv2.resize(rgb, (672, round(rgb.shape[0] * 672 / rgb.shape[1])))
                    ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                             [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        # Serial IDs come from the configured hardware, not HTTP input.
                        name = f'camera-preview-{self.pid}-{entry["serial"]}.jpg'
                        self._write(self.directory / name, jpeg.tobytes())
                        camera['image'] = name
            cameras.append(camera)
        self._write(self.status_path, json.dumps(dict(owner_pid=self.pid, updated_monotonic=time.monotonic(),
                                                     cameras=cameras)).encode())

    def _run(self):
        try:
            while not self.stop.is_set():
                try:
                    self._publish()
                except Exception:
                    logging.exception('Camera preview update failed')
                self.stop.wait(1.)
        finally:
            for stream in self.owned:
                try:
                    stream.close()
                except Exception:
                    logging.exception('Preview-only camera cleanup failed')
            for path in self.files:
                if path == self.status_path:
                    try:
                        if json.loads(path.read_text()).get('owner_pid') != self.pid:
                            continue
                    except (OSError, ValueError):
                        continue
                path.unlink(missing_ok=True)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
