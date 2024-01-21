from agent import Agent
from env import Env
from tile_encoder import TileEncoder
from utils import State
from config import POSITION_BOUNDS, SPEED_BOUNDS
import plotly.graph_objects as go
import numpy as np

if __name__ == "__main__":
    env = Env()
    tile_encoder = TileEncoder(env)
    agent = Agent(env)
    agent.sarsa_learning()

    action = 0
    GRANULARITY = 100

    positions = np.linspace(*POSITION_BOUNDS, GRANULARITY)
    speeds = np.linspace(*SPEED_BOUNDS, GRANULARITY)
    q = np.zeros((GRANULARITY, GRANULARITY))
    for i in range(GRANULARITY):
        for j in range(GRANULARITY):
            q[i, j] = agent.q(
                agent.tile_encoder.encode(State(position=positions[i], speed=speeds[j]), action)
            )
    fig = go.Figure(data=[go.Surface(x=speeds, y=positions, z=-q)])
    fig.update_layout(scene=dict(
        xaxis_title='Speed',
        yaxis_title='Position',
        zaxis_title='Number of steps remaining'))
    fig.show()

