import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import Module
from jax import Array
from jaxtyping import Float

from tetris_rl.environment import create_environment


# TODO: The model has static dimensions. Try to make it more flexible.
class Model(Module):
    """
    Model that given an observation of shape `obs_shape` (channels come first)
    returns the `num_actions` Q-values for each action.
    """
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Conv2d(1, 8, stride=4, kernel_size=8, key=key1),
            eqx.nn.MaxPool2d(kernel_size=4),
            jax.nn.relu,
            eqx.nn.Conv2d(8, 4, stride=2, kernel_size=4, key=key2),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            eqx.nn.Conv2d(4, 2, stride=2, kernel_size=2, key=key3),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(140, 32, key=key4),
            jax.nn.relu,
            eqx.nn.Linear(32, 5, key=key5),
            jax.nn.relu
        ]

    def __call__(self, x: Float[Array, "1 210 160"]) -> Float[Array, "5"]:
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    env = create_environment()
    print(f"Num actions: {env.action_space.n}")
    print(f"Observation space: {env.observation_space.shape}")

    model = Model(key=jax.random.PRNGKey(0))

    state, info = env.reset()
    Q = model(state)
    print(Q)

    params, static = eqx.partition(model, eqx.is_array)
