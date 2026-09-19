"""ADB controller that shells out to the official adb binary."""

import io
import os
import shutil
import subprocess
from typing import Optional

from PIL import Image


class ADBController:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.adb_path = self._discover_adb()
        self.connected = False

    @staticmethod
    def _discover_adb() -> Optional[str]:
        adb = shutil.which("adb")
        if adb:
            return adb
        candidates = [
            os.environ.get("ADB_PATH"),
            r"C:\\platform-tools\\adb.exe",
            r"C:\\Android\\platform-tools\\adb.exe",
            "/usr/bin/adb",
            "/usr/local/bin/adb",
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def _run(self, *args, capture_output=True):
        if not self.adb_path:
            raise FileNotFoundError(
                "adb.exe/adb not found. Add it to PATH or set ADB_PATH."
            )
        command = [self.adb_path, "-s", self.device_id, *args]
        return subprocess.run(
            command,
            capture_output=capture_output,
            check=True,
        )

    def connect(self) -> bool:
        try:
            if not self.device_id:
                return False
            result = self._run("get-state")
            state = result.stdout.decode("utf-8", errors="replace").strip()
            if state != "device":
                return False
            self.connected = True
            return True
        except Exception:
            self.connected = False
            return False

    def disconnect(self):
        self.connected = False

    def get_screenshot(self):
        if not self.connected:
            return None
        try:
            result = self._run("exec-out", "screencap", "-p")
            image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            return image
        except Exception:
            return None

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 120):
        if not self.connected:
            return False
        try:
            self._run(
                "shell",
                "input",
                "swipe",
                str(int(start_x)),
                str(int(start_y)),
                str(int(end_x)),
                str(int(end_y)),
                str(int(duration_ms)),
            )
            return True
        except Exception:
            return False

    def tap(self, x: int, y: int):
        if not self.connected:
            return False
        try:
            self._run("shell", "input", "tap", str(int(x)), str(int(y)))
            return True
        except Exception:
            return False
