import random
from typing import Optional, List

import numpy as np

from config import (
    MAX_NUMBER_TILES,
    POSSIBLE_ACTIONS,
    EPSILON,
    NUMBER_OF_EPISODES,
    H_GRADIENT,
    ALPHA,
    GAMMA,
)
from env import Env
from tile_encoder import TileEncoder
from utils import State, Action


class Agent:
    def __init__(self, env: Env):
        self.env = env
        self.tile_encoder = TileEncoder(env)
        self.w = np.zeros(MAX_NUMBER_TILES)

    def sarsa_learning(self):
        for episode in range(NUMBER_OF_EPISODES):
            number_steps = 0
            is_terminal = False
            state = self.env.initial_state()
            action = self._select_eps_greedy_action(state)
            encoded_state_action = self.tile_encoder.encode(state, action)
            while not is_terminal:
                number_steps += 1
                next_state, reward, is_terminal = self.env.transition(state, action)
                next_action = self._select_eps_greedy_action(next_state)

                encoded_next_state_next_action = self.tile_encoder.encode(
                    next_state, next_action
                )

                self.w += (
                    ALPHA
                    * (
                        self.q(encoded_next_state_next_action, is_terminal=is_terminal)
                        + reward
                        - self.q(encoded_state_action)
                    )
                    * self.nabla(encoded_state_action)
                )

                state, action, encoded_state_action = (
                    next_state,
                    next_action,
                    encoded_next_state_next_action,
                )
            print(f"Episode {episode} finished after {number_steps} steps")

    def _select_eps_greedy_action(self, state: State) -> Action:
        if random.random() < EPSILON:
            return random.choice(POSSIBLE_ACTIONS)
        action_q_pairs = {
            action: self.q(self.tile_encoder.encode(state, action))
            for action in POSSIBLE_ACTIONS
        }
        return max(action_q_pairs, key=action_q_pairs.get)

    def q(
        self,
        encoded_state_action: List[int],
        w: Optional[np.array] = None,
        is_terminal: Optional[bool] = False,
    ) -> float:
        if is_terminal:
            return 0
        if w is not None:
            return sum(w[encoded_state_action])
        return sum(self.w[encoded_state_action])

    def nabla(self, encoded_state_action: List[int]):
        nabla = np.zeros(MAX_NUMBER_TILES)
        for dimension in encoded_state_action:
            nabla[dimension] = self._gradient(encoded_state_action, dimension)
        return nabla

    def _gradient(self, encoded_state_action: List[int], dimension: int):
        w_plus_h = self.w.copy()
        w_plus_h[dimension] += H_GRADIENT
        return (
            self.q(encoded_state_action, w_plus_h) - self.q(encoded_state_action)
        ) / H_GRADIENT
