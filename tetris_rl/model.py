import equinox as eqx  # Correct import for Equinox
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import relu


class Flatten(eqx.Module):
    """
    Flattens the input tensor.
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.reshape(x, (-1))


class DeepQNetwork(eqx.Module):
    """
    Gets the state of the game and returns the Q-values for each action.

    Args:
        state_shape: The shape of the state of the game (height, width, channels).
        num_actions: The number of actions the agent can take.
    """
    layers: list

    def __init__(self, state_shape: tuple[int, int, int], num_actions: int, key):
        key_conv_1, key_conv_2, key_conv_3, key_fc, key_output = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Conv2d(
                in_channels=state_shape[2],
                out_channels=32,
                kernel_size=(8, 8),
                stride=(4, 4),
                padding="VALID",
                key=key_conv_1
            ),
            eqx.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding="VALID",
                key=key_conv_2
            ),
            eqx.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="VALID",
                key=key_conv_3
            ),
            Flatten(),
            eqx.nn.Linear(
                in_features=46592,
                out_features=512,
                key=key_fc
            ),
            eqx.nn.Linear(
                in_features=512,
                out_features=num_actions,
                key=key_output
            )
        ]

    def __call__(self, state: np.ndarray) -> jnp.ndarray:
        """
        Forward pass of the DeepQNetwork.

        Args:
            state: The state of the game.

        Returns:
            The Q-values for each action.
        """
        x = jnp.array(state, dtype=jnp.float32) / 255.
        x = np.transpose(x, (2, 0, 1))  # JAX expects channel-first format
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, eqx.nn.Linear):
                x = relu(x)
        return x


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    model = DeepQNetwork(state_shape=(240, 253, 3), num_actions=6, key=key)
