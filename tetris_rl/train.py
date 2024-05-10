import random
from collections import deque
from copy import deepcopy

import equinox
import gym
import jax
import jax.numpy as jnp
from equinox import Module
from gym import Space

from tetris_rl.environment import create_environment
from tetris_rl.model import DeepQNetwork


class Batch:
    def __init__(self, batch_raw: list[tuple[jnp.ndarray, int, float, jnp.ndarray, bool]]):
        self._state = jnp.array([batch_item[0] for batch_item in batch_raw], dtype=jnp.float32)
        self._action = jnp.array([batch_item[1] for batch_item in batch_raw], dtype=jnp.int32)
        self._reward = jnp.array([batch_item[2] for batch_item in batch_raw], dtype=jnp.float32)
        self._next_state = jnp.array([batch_item[3] for batch_item in batch_raw], dtype=jnp.float32)
        self._done = jnp.array([batch_item[4] for batch_item in batch_raw], dtype=jnp.bool)

    @property
    def size(self) -> int:
        return len(self._state)

    @property
    def state(self) -> jnp.ndarray:
        return self._state

    @property
    def action(self) -> jnp.ndarray:
        return self._action

    @property
    def reward(self) -> jnp.ndarray:
        return self._reward

    @property
    def next_state(self) -> jnp.ndarray:
        return self._next_state

    @property
    def done(self) -> jnp.ndarray:
        return self._done


class ReplayMemory:
    def __init__(self, size: int):
        self._size = size
        self._memory = deque(maxlen=size)

    @property
    def size(self) -> int:
        return len(self._memory)

    def sample(self, batch_size: int) -> Batch:
        batch_raw = random.sample(self._memory, batch_size)
        return Batch(batch_raw)

    def append(self, state: jnp.ndarray, action: int, reward: float, next_state: jnp.ndarray, done: bool) -> None:
        self._memory.append((state, action, reward, next_state, done))


@jax.jit
def calculate_y(model: DeepQNetwork, next_state: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray,
                discount_factor: float) -> jnp.ndarray:
    """
    Calculate the target value for the Q-Learning algorithm

    Args:
        model: The model to use to calculate the target value
        next_state: The next state
        reward: The reward
        done: Whether the episode is done
        discount_factor: The discount factor to use when calculating the target value

    Returns:
        The target value (y) for the Q-Learning algorithm
    """
    next_state_q_values = jax.vmap(model)(next_state)
    next_state_max_q_values = next_state_q_values.max(axis=1)
    y = reward + (1 - done) * discount_factor * next_state_max_q_values
    return y


def select_action(model: DeepQNetwork, action_space: Space, state: jnp.ndarray, exploration_probability: float) -> int:
    """
    Pick an action to take based on the model and the state

    Args:
        model: The model to use to pick the action
        environment: The environment to pick the action for
        state: The state to pick the action for
        exploration_probability: The probability of taking a random action

    Returns:
        The action to take
    """
    take_random_action = random.random() <= exploration_probability
    if take_random_action:
        action = action_space.sample()
    else:
        Q_values = model(state)
        action = Q_values.argmax()
    return int(action)


@jax.jit
def loss_fn(model: DeepQNetwork, state: jnp.ndarray, action: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    Q_values = jax.vmap(model)(state)
    realized_Q_values = Q_values[jnp.arange(len(action)), action]
    return jnp.mean(jnp.square(y - realized_Q_values))


grad_loss_fn = jax.grad(loss_fn)


def train(
        model: Module,
        environment: gym.Env,
        steps: int = 1_000_000,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_probability: float = 0.1,
        batch_size: int = 32,
        replay_memory_size: int = 10_000,
        update_target_action_value_fn_step: int = 100
) -> None:
    """
    Train the model on the environment using the Deep Q-Learning algorithm

    Nice explanation of the algorithm itself e.g. here: https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm

    Args:
        model: The model to train
        environment: The environment to train the model on
        steps: The number of steps to train the model for
        learning_rate: The learning rate to use when training the model
        discount_factor: The discount factor to use when training the model
        exploration_probability: The probability of taking a random action
        batch_size: The size of the batch to sample from the replay memory
        replay_memory_size: The size of the replay memory
        update_target_action_value_fn_step: The number of steps to take before updating the target action value function
            with the action value function that is being trained

    Returns:
        None
    """
    replay_memory = ReplayMemory(replay_memory_size)
    target_model = deepcopy(model)

    done = True
    batch_loss_value = float("inf")

    for step_index in range(steps):
        if step_index % 10 == 0:
            print(f'Step: {step_index}, Loss: {batch_loss_value}')

        if done:
            state = environment.reset()

        action = select_action(
            model=model,
            action_space=environment.action_space,
            state=state,
            exploration_probability=exploration_probability
        )
        next_state, reward, done, info = environment.step(action)

        replay_memory.append(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )

        # Wait with training until we have enough data in the replay memory
        if replay_memory.size / 10 < batch_size:
            continue

        batch = replay_memory.sample(batch_size)
        y = calculate_y(
            model=target_model,
            next_state=batch.next_state,
            reward=batch.reward,
            done=batch.done,
            discount_factor=discount_factor
        )
        batch_loss_value = loss_fn(model, batch.state, batch.action, y)
        grads = grad_loss_fn(model, batch.state, batch.action, y)
        model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)

        if step_index % update_target_action_value_fn_step == 0:
            target_model = deepcopy(model)


if __name__ == '__main__':
    env = create_environment()
    num_actions = env.action_space.n
    key = jax.random.PRNGKey(0)
    model = DeepQNetwork(state_shape=(240, 256, 3), num_actions=num_actions, key=key)

    train(
        model=model,
        environment=env,
        steps=10_000,
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_probability=0.1,
        batch_size=32,
        replay_memory_size=10_000,
        update_target_action_value_fn_step=100
    )

    equinox.tree_serialise_leaves("serialized_model.eqx", model)

    env.close()
