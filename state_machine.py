"""Simple state machine for dodge timing."""

import time


class DodgeStateMachine:
    def __init__(self, dodge_cooldown_seconds: float = 0.2):
        self.state = "searching"
        self.dodge_cooldown_seconds = float(dodge_cooldown_seconds)
        self.last_dodge_at = 0.0

    def update(self, threat_detected: bool):
        now = time.monotonic()
        if threat_detected:
            self.state = "threat_detected"
        elif now - self.last_dodge_at < self.dodge_cooldown_seconds:
            self.state = "recovering"
        else:
            self.state = "searching"

    def can_dodge(self) -> bool:
        now = time.monotonic()
        return (now - self.last_dodge_at) >= self.dodge_cooldown_seconds

    def on_dodge(self):
        self.last_dodge_at = time.monotonic()
        self.state = "dodging"

    def reset(self):
        self.state = "searching"
        self.last_dodge_at = 0.0
