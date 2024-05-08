import gym
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def create_environment() -> gym.Env:
    """
    Creates the Tetris environment with the settings used in this project.

    Returns:
        gym.Env: The Tetris environment.
    """
    env = gym_tetris.TetrisEnv(
        b_type=False,
        reward_score=True,
        reward_lines=False,
        penalize_height=False,
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env
