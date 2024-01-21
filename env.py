import random
from typing import Tuple, Optional
import numpy as np

from config import POSITION_BOUNDS, SPEED_BOUNDS, INITIAL_POSITION_BOUNDS, BASE_REWARD
from utils import State, Action


class Env:
    def __init__(
        self,
        position_bounds: Optional[Tuple[float, float]] = POSITION_BOUNDS,
        speed_bounds: Optional[Tuple[float, float]] = SPEED_BOUNDS,
    ):
        self.position_bounds = position_bounds
        self.speed_bounds = speed_bounds

    def transition(self, state: State, action: Action):
        next_speed = np.clip(
            state.speed + 0.001 * action - 0.0025 * np.cos(3 * state.position),
            self.speed_bounds[0],
            self.speed_bounds[1],
        )
        next_position = np.clip(
            state.position + next_speed,
            self.position_bounds[0],
            self.position_bounds[1],
        )
        next_state = State(next_position, next_speed)

        if self._is_terminal(next_position):
            return next_state, BASE_REWARD, True
        return next_state, BASE_REWARD, False

    def _is_terminal(self, position: float):
        return position == self.position_bounds[1]

    @staticmethod
    def initial_state():
        return State(
            random.uniform(INITIAL_POSITION_BOUNDS[0], INITIAL_POSITION_BOUNDS[1]), 0
        )
