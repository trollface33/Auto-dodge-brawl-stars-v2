"""Plan dodge direction based on threat position."""

from typing import Tuple


class DodgePlanner:
    def __init__(self, dodge_range: int = 320):
        self.dodge_range = int(dodge_range)

    def plan(self, player_position: Tuple[int, int], threat_position: Tuple[int, int], frame_size: Tuple[int, int]) -> Tuple[int, int]:
        player_x, player_y = player_position
        threat_x, threat_y = threat_position
        width, height = frame_size

        dx = player_x - threat_x
        dy = player_y - threat_y

        if abs(dx) >= abs(dy):
            target_x = player_x + (self.dodge_range if dx > 0 else -self.dodge_range)
            target_y = player_y
        else:
            target_x = player_x
            target_y = player_y + (self.dodge_range if dy > 0 else -self.dodge_range)

        target_x = max(50, min(target_x, width - 50))
        target_y = max(50, min(target_y, height - 50))
        return (int(target_x), int(target_y))
