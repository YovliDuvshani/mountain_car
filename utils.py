from dataclasses import dataclass
from typing import NewType


@dataclass
class State:
    position: float
    speed: float


Action = NewType("Action", int)
