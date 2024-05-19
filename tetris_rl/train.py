import os
import random
from collections import deque

import equinox as eqx
import jax
from tqdm import tqdm

from tetris_rl.agent import DoubleQAgent
from tetris_rl.environment import create_environment
from tetris_rl.model import Model


def train(
        agent,
        env,
        replay_memory_size: int,
        episodes: int,
        min_replay_memory_size_for_training: int,
        batch_size: int,
        save_period_epochs: int = 10
):
    replay_memory = deque(maxlen=replay_memory_size)
    for episode in tqdm(range(episodes)):
        previous_observation, previous_info = env.reset()
        terminated = False

        while not terminated:
            action = agent.act(previous_observation)
            observation, reward, terminated, truncated, info = env.step(action)

            replay_memory.append((previous_observation, action, reward, observation, terminated))
            previous_observation = observation
            previous_info = info

            if len(replay_memory) > min_replay_memory_size_for_training:
                batch = random.sample(replay_memory, batch_size)
                agent.train(batch)

        if episode % save_period_epochs == 0:
            os.makedirs("models", exist_ok=True)
            eqx.tree_serialise_leaves(f"models/model_episode_{episode}.eqx", agent.model)

    env.close()


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    env = create_environment(render_mode=None)
    agent = DoubleQAgent(
        model=Model(key=key),
        num_actions=env.action_space.n,
        learning_rate=1e-3,
        explore_probability=0.01,
        discount_factor=0.99,
        target_update_period_steps=1000
    )

    train(
        agent=agent,
        env=env,
        replay_memory_size=10_000,
        episodes=100,
        min_replay_memory_size_for_training=1_000,
        batch_size=32,
        save_period_epochs=5
    )
