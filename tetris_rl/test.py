import equinox
import gym
import jax

from tetris_rl.environment import create_environment
from tetris_rl.model import DeepQNetwork


def test(model: DeepQNetwork, environment: gym.Env, steps: int = 10_000) -> None:
    """
    Visually evaluates the model in the environment.

    Args:
        model: The model to evaluate.
        environment: The environment to evaluate the model in.
        steps: The number of steps to evaluate the model for.

    Returns:
        None
    """
    done = False
    for step in range(steps):
        if done:
            break
        state = environment.reset()
        while not done:
            Q_values = model(state)
            action = int(Q_values.argmax())
            state, reward, done, _ = environment.step(action)
            environment.render()


if __name__ == '__main__':
    env = create_environment()
    num_actions = env.action_space.n
    key = jax.random.PRNGKey(0)
    model = DeepQNetwork(state_shape=(240, 256, 3), num_actions=num_actions, key=key)
    equinox.tree_deserialise_leaves("serialized_model.eqx", model)

    test(
        model=model,
        environment=env
    )
