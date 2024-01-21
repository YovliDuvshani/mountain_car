from config import MAX_NUMBER_TILES, NUMBER_TILINGS
from env import Env
from tile import IHT, tiles
from utils import State, Action


class TileEncoder:
    def __init__(
        self,
        env: Env,
    ):
        self.env = env
        self.iht = IHT(MAX_NUMBER_TILES)

    def encode(self, state: State, action: Action):
        return tiles(
            self.iht,
            NUMBER_TILINGS,
            [
                NUMBER_TILINGS
                * state.position
                / (self.env.position_bounds[1] - self.env.position_bounds[0]),
                NUMBER_TILINGS
                * state.speed
                / (self.env.speed_bounds[1] - self.env.speed_bounds[0]),
                action,
            ],
        )
