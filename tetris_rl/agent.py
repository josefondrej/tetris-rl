import functools
import random

import equinox as eqx
import jax
import numpy as np
import optax
from equinox import Module
from jax import numpy as jnp
from jaxtyping import PyTree


@functools.partial(jax.jit, static_argnames=["static"])
def act_policy(params: Module, static: Module, observation: np.ndarray, ) -> int:
    model = eqx.combine(params, static)
    Q_values = model(observation)
    action = jnp.argmax(Q_values)
    return action


def loss(
        model_params: Module,
        model_static: Module,
        target_params: Module,
        target_static: Module,
        previous_observation_batch: jnp.ndarray,
        action_batch: jnp.ndarray,
        reward_batch: jnp.ndarray,
        observation_batch: jnp.ndarray,
        terminated_batch: jnp.ndarray,
        discount_factor: float
) -> jnp.ndarray:
    model = eqx.combine(model_params, model_static)
    target = eqx.combine(target_params, target_static)

    Q_values_realized = jax.vmap(model)(previous_observation_batch)[jnp.arange(len(action_batch)), action_batch]
    Q_values_target = jax.vmap(target)(observation_batch)
    y = reward_batch + (1 - terminated_batch) * discount_factor * jnp.max(Q_values_target, axis=1)
    loss = jnp.mean((Q_values_realized - y) ** 2)
    return loss


@functools.partial(
    jax.jit,
    static_argnames=[
        "model_static",
        "target_static",
        "discount_factor",
        "optimizer"
    ]
)
def make_step(
        model_params: Module,
        model_static: Module,
        target_params: Module,
        target_static: Module,
        previous_observation_batch: jnp.ndarray,
        action_batch: jnp.ndarray,
        reward_batch: jnp.ndarray,
        observation_batch: jnp.ndarray,
        terminated_batch: jnp.ndarray,
        discount_factor: float,
        optimizer: optax.GradientTransformation,
        optimizer_state: PyTree
):
    loss_value, d_loss_d_model_params = jax.value_and_grad(loss)(
        model_params,
        model_static=model_static,
        target_params=target_params,
        target_static=target_static,
        previous_observation_batch=previous_observation_batch,
        action_batch=action_batch,
        reward_batch=reward_batch,
        observation_batch=observation_batch,
        terminated_batch=terminated_batch,
        discount_factor=discount_factor
    )
    updates, optimizer_state = optimizer.update(d_loss_d_model_params, optimizer_state, model_params)
    model_params = optax.apply_updates(model_params, updates)
    return model_params, optimizer_state, loss_value


class DoubleQAgent:
    def __init__(self,
                 model: eqx.Module,
                 num_actions: int,
                 learning_rate: float = 1e-3,
                 explore_probability: float = 0.01,
                 discount_factor: float = 0.99,
                 target_update_period_steps: int = 100,
                 ):
        self._model_params, self._model_static = eqx.partition(model, eqx.is_array)
        self._num_actions = num_actions
        self._learning_rate = learning_rate
        self._explore_probability = explore_probability
        self._discount_factor = discount_factor
        self._target_update_period = target_update_period_steps

        self._target_params: eqx.Module = None
        self._target_static: eqx.Module = None
        self._update_target()
        self._train_step = 0

        self._optimizer = optax.adam(learning_rate)
        self._optimizer_state = self._optimizer.init(self._model_params)

    @property
    def model(self) -> eqx.Module:
        return eqx.combine(self._model_params, self._model_static)

    def _update_target(self) -> eqx.Module:
        self._target_params = self._model_params
        self._target_static = self._model_static

    def act(self, observation: np.ndarray) -> int:
        explore = random.random() < self._explore_probability
        if explore:
            return random.randint(0, self._num_actions - 1)
        else:
            return act_policy(self._model_params, self._model_static, observation)

    def train(self, batch: list):
        previous_observation_batch, action_batch, reward_batch, observation_batch, terminated_batch = zip(*batch)
        previous_observation_batch = jnp.array(previous_observation_batch, dtype=jnp.float32)
        action_batch = jnp.array(action_batch, dtype=jnp.int32)
        observation_batch = jnp.array(observation_batch, dtype=jnp.float32)
        reward_batch = jnp.array(reward_batch, dtype=jnp.float32)
        terminated_batch = jnp.array(terminated_batch, dtype=jnp.bool)

        self._model_params, self._optimizer_state, loss_value = make_step(
            model_params=self._model_params,
            model_static=self._model_static,
            target_params=self._target_params,
            target_static=self._target_static,
            previous_observation_batch=previous_observation_batch,
            action_batch=action_batch,
            reward_batch=reward_batch,
            observation_batch=observation_batch,
            terminated_batch=terminated_batch,
            discount_factor=self._discount_factor,
            optimizer=self._optimizer,
            optimizer_state=self._optimizer_state
        )

        self._train_step += 1
        if self._train_step % self._target_update_period == 0:
            self._update_target()
