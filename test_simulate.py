import math
import numpy as np

from brain import Brain
from simulate import run_episode

brain = Brain(alpha=0.2)

result = run_episode(
    brain=brain,
    target_pos=np.array([7.5, 7.0], dtype=float),
    initial_agent_pos=np.array([2.0, 2.0], dtype=float),
    initial_agent_angle=math.radians(20.0),
    max_steps=300,
)

print(result)