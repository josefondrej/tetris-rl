import equinox as eqx
import jax

from tetris_rl.environment import create_environment
from tetris_rl.model import Model


def load_model(model_file_path: str) -> Model:
    key = jax.random.PRNGKey(0)
    model_like = Model(key=key)
    model = eqx.tree_deserialise_leaves(model_file_path, model_like)
    return model


if __name__ == '__main__':
    model = load_model(model_file_path="models/model_episode_5.eqx")
    env = create_environment(render_mode="human")

    state, info = env.reset()
    terminated = False
    while not terminated:
        Q = model(state)
        action = Q.argmax()
        state, reward, terminated, truncated, info = env.step(action)
        env.render()
