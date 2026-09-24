"""Read public Franka identity and status; never request control or send motion."""
import base64
import json
import os
import socket
import ssl
import struct
import urllib.request

host = "172.16.0.2"
context = ssl._create_unverified_context()  # Robot's local self-signed certificate.


def get(path):
    with urllib.request.urlopen("https://" + host + path, context=context, timeout=5) as r:
        return json.load(r)


def status():
    with context.wrap_socket(socket.create_connection((host, 443), timeout=5), server_hostname=host) as c:
        c.settimeout(5)
        key = base64.b64encode(os.urandom(16)).decode()
        c.sendall((f"GET /admin/api/system-status HTTP/1.1\r\nHost: {host}\r\n"
                   "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                   f"Origin: https://{host}\r\n\r\n").encode())
        stream = c.makefile("rb")
        response = stream.readline().decode().strip()
        assert "101 " in response, response
        while stream.readline().strip():
            pass
        header = stream.read(2)
        length = header[1] & 127
        if length == 126:
            length = struct.unpack("!H", stream.read(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", stream.read(8))[0]
        assert not header[1] & 128 and length < 200000
        return json.loads(stream.read(length))


state = status()
safety = state.get("safety", {})
print(json.dumps({
    "system_version": get("/admin/api/system-version").splitlines()[0],
    "arm": get("/admin/api/robot/nameplate"),
    "operating_mode": state.get("derived", {}).get("operatingMode"),
    "first_start": state.get("firstStart"),
    "fci_active": state.get("controlToken", {}).get("fciActive"),
    "execution_running": state.get("execution", {}).get("running"),
    "brakes": safety.get("brakeState"),
    "power": safety.get("powerState"),
    "safety_status": safety.get("safetyControllerStatus"),
    "safety_warnings": safety.get("activeWarnings"),
    "robot_errors": state.get("robot", {}).get("robotErrors"),
}, indent=2))
