import gymnasium
from gymnasium.wrappers import GrayScaleObservation, TransformObservation


def create_environment(render_mode: str = "human") -> gymnasium.Env:
    """
    Creates the Tetris environment with the settings used in this project.

    Returns:
        gym.Env: The Tetris environment.
    """
    env = gymnasium.make("ALE/Tetris-v5", render_mode=render_mode)
    env = GrayScaleObservation(env, keep_dim=True)
    env = TransformObservation(env, lambda obs: obs.transpose(2, 0, 1).astype('float32') / 255.)

    return env


if __name__ == '__main__':
    env = create_environment()
    observation, info = env.reset()

    terminated = False
    while not terminated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()
