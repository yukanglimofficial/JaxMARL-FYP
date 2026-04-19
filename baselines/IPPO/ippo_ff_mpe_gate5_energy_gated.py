
"""
Gate 5 training script: IPPO FF on MPE with terminal-correct step_env + masked reset
+ terminal geometry metrics + energy-gated updates + update_counter semantics
+ VarTD Path A + deterministic fixed-episode evaluation.

Built as a minimal overlay on the trusted Gate 4 collector.

Locked Gate 5 requirements implemented:
(A) Collector still uses env.step_env(...) with vmapped reset + masked reset
(B) Funding comes from terminal geometry by default:
      z_k = dbar_end / (d0 + eps)
      g_k = exp(-beta * clip(z_k, 0, zmax))
    with return-based fallback available iff geometry funding is disabled
(C) Gate uses pre-income energy:
      u_k = 1[E_k >= c_upd]
      E_{k+1} = clip(E_k - c_upd*u_k + alpha*g_k, 0, E_max)
(D) Executed-update counter increments only when an update executes and drives
    update-only RNG via jax.random.fold_in(rng_update_base, update_counter)
(E) VarTD Path A is computed every iteration from rollout data, independent of skipping
(F) Deterministic eval respects the locked episode_len = max_cycles + 1 invariant
(G) Eval debug records pre-step vs post-step terminal geometry plus CSR sweeps
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import tree_util
from omegaconf import OmegaConf


EVAL_CSR_EPS_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)


# =========================
# Gate 3 helpers (LOCKED)
# =========================
def broadcast_mask(mask_bool: jnp.ndarray, leaf_ndim: int) -> jnp.ndarray:
    """Broadcast (E,) -> (E, 1, 1, ..., 1) to match a leaf."""
    return mask_bool.reshape((mask_bool.shape[0],) + (1,) * (leaf_ndim - 1))


def tree_where(mask_bool: jnp.ndarray, tree_true, tree_false):
    """Elementwise selection across a pytree; choose from tree_true where mask_bool is True."""
    return tree_util.tree_map(
        lambda a, b: jnp.where(broadcast_mask(mask_bool, a.ndim), a, b),
        tree_true,
        tree_false,
    )


def batchify_ea(obs_dict, agents: tuple[str, ...]) -> jnp.ndarray:
    """{agent: (E, obs_dim)} -> (E, A, obs_dim) using env.agents as the static order."""
    return jnp.stack([obs_dict[a] for a in agents], axis=1)


def compute_terminal_geometry_from_obs(
    obs_arr: jnp.ndarray, N: int, eps_occ: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    obs_arr: (E, A, obs_dim)

    Validated obs layout (Gate 3):
      self_vel = obs[..., 0:2]
      self_pos = obs[..., 2:4]
      landmark_rel = obs[..., 4:4+2N] reshaped to (..., N, 2)

    Returns:
      tilde_d:  (E, N)   per-landmark min distance to closest agent
      min_dists:(E,)     sum over landmarks of tilde_d
      occupied:(E,)      number of landmarks with tilde_d < eps_occ
      success: (E,) bool occupied == N
    """
    E, A, _ = obs_arr.shape
    land_rel = obs_arr[:, :, 4 : 4 + 2 * N].reshape(E, A, N, 2)
    d = jnp.linalg.norm(land_rel, axis=-1)
    tilde_d = d.min(axis=1)
    min_dists = tilde_d.sum(axis=1)
    occupied = (tilde_d < eps_occ).sum(axis=1)
    success = occupied == N
    return tilde_d, min_dists, occupied, success


def compute_terminal_geometry_with_sweep(
    obs_arr: jnp.ndarray,
    N: int,
    eps_occ: float,
    eps_grid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns locked-threshold geometry plus a small CSR threshold sweep.

    Outputs:
      tilde_d:         (E, N)
      min_dists:       (E,)
      occupied_locked: (E,) int32 at eps_occ
      success_locked:  (E,) bool at eps_occ
      success_sweep:   (E, G) bool for eps_grid
      max_tilde_d:     (E,) max over landmarks of tilde_d
    """
    tilde_d, min_dists, occupied_locked, success_locked = compute_terminal_geometry_from_obs(
        obs_arr, N=N, eps_occ=eps_occ
    )
    success_sweep = (tilde_d[..., None] < eps_grid[None, None, :]).sum(axis=1) == N
    max_tilde_d = tilde_d.max(axis=1)
    return (
        tilde_d,
        min_dists,
        occupied_locked.astype(jnp.int32),
        success_locked,
        success_sweep,
        max_tilde_d.astype(jnp.float32),
    )


# =========================
# Gate 5 helpers
# =========================
def funding_from_geometry(
    dbar_end: jnp.ndarray,
    d0: float,
    beta: float,
    z_max: float,
    eps_norm: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (z_k, g_k) using geometry-based funding."""
    z_k = dbar_end / (jnp.asarray(d0, dtype=jnp.float32) + jnp.asarray(eps_norm, dtype=jnp.float32))
    z_clip = jnp.clip(z_k, 0.0, jnp.asarray(z_max, dtype=jnp.float32))
    g_k = jnp.exp(-jnp.asarray(beta, dtype=jnp.float32) * z_clip)
    return z_k.astype(jnp.float32), g_k.astype(jnp.float32)


def funding_from_return(
    rbar_end: jnp.ndarray,
    r0: float,
    beta: float,
    z_max: float,
    eps_norm: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (z_k, g_k) using return-based fallback funding."""
    denom = jnp.maximum(-jnp.asarray(r0, dtype=jnp.float32), jnp.asarray(eps_norm, dtype=jnp.float32))
    z_k = (-rbar_end) / denom
    z_clip = jnp.clip(z_k, 0.0, jnp.asarray(z_max, dtype=jnp.float32))
    g_k = jnp.exp(-jnp.asarray(beta, dtype=jnp.float32) * z_clip)
    return z_k.astype(jnp.float32), g_k.astype(jnp.float32)


def vartd_path_a(
    traj_batch,
    last_val: jnp.ndarray,
    num_envs: int,
    num_agents: int,
    gamma: float,
) -> jnp.ndarray:
    """
    VarTD Path A:
      delta[t,e,i] = r[t,e,i] + gamma * (1 - done_all[t,e]) * V_next[t,e,i] - V[t,e,i]
      VarTD_k = var(delta) over all (t,e,i)
    """
    t_steps = traj_batch.reward.shape[0]
    rewards = traj_batch.reward.reshape((t_steps, num_agents, num_envs)).transpose(0, 2, 1).astype(jnp.float32)
    values = traj_batch.value.reshape((t_steps, num_agents, num_envs)).transpose(0, 2, 1).astype(jnp.float32)
    dones_actor = traj_batch.done.reshape((t_steps, num_agents, num_envs)).transpose(0, 2, 1).astype(jnp.float32)
    dones_all = jnp.max(dones_actor, axis=-1).astype(jnp.float32)  # (T, E)
    last_value = last_val.reshape((num_agents, num_envs)).transpose(1, 0).astype(jnp.float32)
    values_next = jnp.concatenate([values[1:], last_value[None, ...]], axis=0)
    delta = rewards + jnp.asarray(gamma, dtype=jnp.float32) * (1.0 - dones_all[..., None]) * values_next - values
    return jnp.var(delta).astype(jnp.float32)


# =========================
# PPO model (trusted Gate 4 baseline)
# =========================
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict


def batchify(x: dict, agent_list, num_actors):
    """Baseline batchify/unbatchify: agent-major flattening for the policy net."""
    max_dim = max([x[a].shape[-1] for a in agent_list])

    def pad(z, length=max_dim):
        return jnp.concatenate(
            [z, jnp.zeros(z.shape[:-1] + (length - z.shape[-1],))], -1
        )

    x = jnp.stack(
        [x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list]
    )
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    config = dict(config)

    # -------------------------
    # Gate 5 defaults (baseline repair target)
    # -------------------------
    config.setdefault("C_UPD", 0.0)
    config.setdefault("ALPHA", 0.0)
    config.setdefault("E0", 1.0)
    config.setdefault("E_MAX", 1.0)
    config.setdefault("BETA", float(np.log(10.0)))
    config.setdefault("Z_MAX", 10.0)
    config.setdefault("EPS_NORM", 1e-8)
    config.setdefault("D0", 0.733348)
    config.setdefault("R0", -27.256548)
    config.setdefault("USE_GEOM_FUNDING", True)
    config.setdefault("DO_EVAL", True)
    config.setdefault("M_EVAL", 100)
    config.setdefault("PRINT_EVERY", 10)

    # -------------------------
    # Environment (UNWRAPPED) with alias normalization
    # -------------------------
    env_kwargs_raw = dict(config.get("ENV_KWARGS", {}))
    env_kwargs = dict(env_kwargs_raw)
    if "N" in env_kwargs and "num_agents" not in env_kwargs:
        env_kwargs["num_agents"] = env_kwargs["N"]
    if "N" in env_kwargs and "num_landmarks" not in env_kwargs:
        env_kwargs["num_landmarks"] = env_kwargs["N"]
    if "max_cycles" in env_kwargs and "max_steps" not in env_kwargs:
        env_kwargs["max_steps"] = env_kwargs["max_cycles"]
    env_kwargs.pop("N", None)
    env_kwargs.pop("max_cycles", None)

    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    if not hasattr(env, "step_env"):
        raise RuntimeError("Env has no step_env method. Gate 2 required step_env. NO-GO.")

    agents = tuple(env.agents)
    num_agents = int(env.num_agents)
    eps_occ = float(config.get("EPS_OCC", 0.1))
    eval_eps_grid = jnp.asarray(EVAL_CSR_EPS_GRID, dtype=jnp.float32)

    # Gate 4 hard requirement: collector uses step_env + masked reset
    reset_v = jax.vmap(env.reset)
    step_env_v = jax.vmap(env.step_env)

    # -------------------------
    # Derived config (LOCKED shapes)
    # -------------------------
    config["NUM_ACTORS"] = int(num_agents * config["NUM_ENVS"])
    config["NUM_UPDATES"] = int(
        int(config["TOTAL_TIMESTEPS"]) // int(config["NUM_STEPS"]) // int(config["NUM_ENVS"])
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ACTORS"] * int(config["NUM_STEPS"]) // int(config["NUM_MINIBATCHES"])
    )

    max_cycles_for_count = int(
        env_kwargs_raw.get(
            "max_cycles",
            env_kwargs_raw.get("max_steps", getattr(env, "max_steps", 25)),
        )
    )
    episode_len = int(config.get("EPISODE_LEN", max_cycles_for_count + 1))
    if max_cycles_for_count is not None:
        assert int(config["NUM_STEPS"]) >= int(max_cycles_for_count), "LOCKED: NUM_STEPS >= max_cycles/max_steps"

    c_upd = jnp.asarray(config["C_UPD"], dtype=jnp.float32)
    alpha = jnp.asarray(config["ALPHA"], dtype=jnp.float32)
    e0 = jnp.asarray(config["E0"], dtype=jnp.float32)
    e_max = jnp.asarray(config["E_MAX"], dtype=jnp.float32)
    beta = jnp.asarray(config["BETA"], dtype=jnp.float32)
    z_max = jnp.asarray(config["Z_MAX"], dtype=jnp.float32)
    eps_norm = jnp.asarray(config["EPS_NORM"], dtype=jnp.float32)
    d0 = float(config["D0"])
    r0 = float(config["R0"])
    use_geom_funding = bool(config["USE_GEOM_FUNDING"])
    do_eval = bool(config["DO_EVAL"])
    m_eval = int(config["M_EVAL"])

    if do_eval:
        if m_eval % int(config["NUM_ENVS"]) != 0:
            raise ValueError("LOCKED: M_EVAL must be divisible by NUM_ENVS for fixed-shape evaluation.")
        episodes_per_env = m_eval // int(config["NUM_ENVS"])
        t_eval = episodes_per_env * episode_len
    else:
        episodes_per_env = 0
        t_eval = 0

    def linear_schedule(count):
        frac = 1.0 - (
            (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space(agents[0]).n, activation=config["ACTIVATION"])
        rng, rng_net, rng_reset, rng_rollout0, rng_update_base, rng_eval_base = jax.random.split(rng, 6)
        init_x = jnp.zeros(env.observation_space(agents[0]).shape)
        network_params = network.init(rng_net, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV (vmapped reset)
        reset_rng = jax.random.split(rng_reset, config["NUM_ENVS"])
        obsv, env_state = reset_v(reset_rng)

        # Wrapper-free logging state
        episode_returns = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.float32)
        episode_lengths = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.float32)
        returned_episode_returns = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.float32)
        returned_episode_lengths = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.float32)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=8,
            )
            return advantages, advantages + traj_batch.value

        def _run_ppo_update(train_state_in, traj_batch, advantages, targets, rng_update):
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state_local, batch_info):
                    traj_batch_mb, advantages_mb, targets_mb = batch_info

                    def _loss_fn(params, traj_batch_mb, gae_mb, targets_mb):
                        pi, value = network.apply(params, traj_batch_mb.obs)
                        log_prob = pi.log_prob(traj_batch_mb.action)

                        value_pred_clipped = traj_batch_mb.value + (value - traj_batch_mb.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets_mb)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets_mb)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        gae_mb = (gae_mb - gae_mb.mean()) / (gae_mb.std() + 1e-8)
                        ratio = jnp.exp(log_prob - traj_batch_mb.log_prob)
                        loss_actor1 = ratio * gae_mb
                        loss_actor2 = jnp.clip(
                            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                        ) * gae_mb
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss_scalar, aux), grads = grad_fn(
                        train_state_local.params, traj_batch_mb, advantages_mb, targets_mb
                    )
                    value_loss, actor_loss, entropy, ratio = aux
                    train_state_local = train_state_local.apply_gradients(grads=grads)

                    loss_info = {
                        "total_loss": loss_scalar.astype(jnp.float32),
                        "actor_loss": actor_loss.astype(jnp.float32),
                        "critic_loss": value_loss.astype(jnp.float32),
                        "entropy": entropy.astype(jnp.float32),
                        "ratio": ratio.astype(jnp.float32),
                    }
                    return train_state_local, loss_info

                train_state_local, traj_batch_local, advantages_local, targets_local, rng_local = update_state
                rng_local, perm_key = jax.random.split(rng_local)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"], (
                    "batch size must equal NUM_STEPS * NUM_ACTORS"
                )

                permutation = jax.random.permutation(perm_key, batch_size)
                batch = (traj_batch_local, advantages_local, targets_local)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state_local, loss_info = jax.lax.scan(_update_minbatch, train_state_local, minibatches)
                update_state = (train_state_local, traj_batch_local, advantages_local, targets_local, rng_local)
                return update_state, loss_info

            update_state0 = (train_state_in, traj_batch, advantages, targets, rng_update)
            update_stateT, loss_info = jax.lax.scan(
                _update_epoch, update_state0, None, config["UPDATE_EPOCHS"]
            )
            train_state_out = update_stateT[0]
            ratio0 = loss_info["ratio"][0, 0].mean().astype(jnp.float32)
            loss_info_mean = jax.tree.map(lambda x: x.mean().astype(jnp.float32), loss_info)
            update_metrics = {
                "total_loss": loss_info_mean["total_loss"],
                "actor_loss": loss_info_mean["actor_loss"],
                "critic_loss": loss_info_mean["critic_loss"],
                "entropy": loss_info_mean["entropy"],
                "ratio0": ratio0,
                "opt_step": train_state_out.step.astype(jnp.int32),
            }
            return train_state_out, update_metrics

        def _skip_update(train_state_in):
            nan32 = jnp.array(jnp.nan, dtype=jnp.float32)
            update_metrics = {
                "total_loss": nan32,
                "actor_loss": nan32,
                "critic_loss": nan32,
                "entropy": nan32,
                "ratio0": nan32,
                "opt_step": train_state_in.step.astype(jnp.int32),
            }
            return train_state_in, update_metrics

        def _update_step(runner_state, k):
            def _env_step(carry, _t):
                (
                    train_state,
                    env_state,
                    last_obs,
                    rng_rollout,
                    episode_returns,
                    episode_lengths,
                    returned_episode_returns,
                    returned_episode_lengths,
                    sum_min_dists_end,
                    sum_occ_end,
                    sum_success_end,
                    sum_tilde_d_end,
                    done_count_end,
                    sum_team_return_end,
                    sum_team_len_end,
                ) = carry

                obs_batch = batchify(last_obs, agents, config["NUM_ACTORS"])

                rng_rollout, key_pi, key_step, key_reset = jax.random.split(rng_rollout, 4)
                keys_step = jax.random.split(key_step, config["NUM_ENVS"])
                keys_reset = jax.random.split(key_reset, config["NUM_ENVS"])

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=key_pi)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, agents, config["NUM_ENVS"], num_agents)

                obs_post, env_state_post, reward, done, _info_env = step_env_v(
                    keys_step, env_state, env_act
                )
                done_all = done["__all__"]

                r_arr = jnp.stack([reward[a] for a in agents], axis=1)
                rewardlog_arr = r_arr * float(num_agents)

                done_f = done_all.astype(jnp.float32)
                done_i = done_all.astype(jnp.int32)

                new_episode_returns = episode_returns + rewardlog_arr
                new_episode_lengths = episode_lengths + 1

                episode_returns = new_episode_returns * (1.0 - done_f)[:, None]
                episode_lengths = new_episode_lengths * (1.0 - done_f)[:, None]

                returned_episode_returns = (
                    returned_episode_returns * (1.0 - done_f)[:, None]
                    + new_episode_returns * done_f[:, None]
                )
                returned_episode_lengths = (
                    returned_episode_lengths * (1.0 - done_f)[:, None]
                    + new_episode_lengths * done_f[:, None]
                )
                returned_episode_mask = jnp.broadcast_to(
                    done_all[:, None], (config["NUM_ENVS"], num_agents)
                )

                obs_post_arr = batchify_ea(obs_post, agents)
                tilde_d, min_dists, occupied, success = compute_terminal_geometry_from_obs(
                    obs_post_arr, N=num_agents, eps_occ=eps_occ
                )

                sum_min_dists_end = sum_min_dists_end + (min_dists * done_f).sum()
                sum_occ_end = sum_occ_end + (occupied.astype(jnp.float32) * done_f).sum()
                sum_success_end = sum_success_end + (success.astype(jnp.float32) * done_f).sum()
                sum_tilde_d_end = sum_tilde_d_end + (tilde_d * done_f[:, None]).sum(axis=0)
                done_count_end = done_count_end + done_i.sum()

                team_return_end = new_episode_returns.mean(axis=1) / float(num_agents)
                sum_team_return_end = sum_team_return_end + (team_return_end * done_f).sum()

                ep_len_end = new_episode_lengths[:, 0].astype(jnp.float32)
                sum_team_len_end = sum_team_len_end + (ep_len_end * done_f).sum()

                obs_reset, env_state_reset = reset_v(keys_reset)
                obs_next = tree_where(done_all, obs_reset, obs_post)
                env_state_next = tree_where(done_all, env_state_reset, env_state_post)

                info_step = {
                    "returned_episode_returns": returned_episode_returns.reshape((config["NUM_ACTORS"],)),
                    "returned_episode_lengths": returned_episode_lengths.reshape((config["NUM_ACTORS"],)),
                    "returned_episode": returned_episode_mask.reshape((config["NUM_ACTORS"],)),
                }

                transition = Transition(
                    done=batchify(done, agents, config["NUM_ACTORS"]).squeeze(),
                    action=action,
                    value=value,
                    reward=batchify(reward, agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=info_step,
                )

                carry_out = (
                    train_state,
                    env_state_next,
                    obs_next,
                    rng_rollout,
                    episode_returns,
                    episode_lengths,
                    returned_episode_returns,
                    returned_episode_lengths,
                    sum_min_dists_end,
                    sum_occ_end,
                    sum_success_end,
                    sum_tilde_d_end,
                    done_count_end,
                    sum_team_return_end,
                    sum_team_len_end,
                )
                return carry_out, transition

            (
                train_state,
                env_state,
                last_obs,
                rng_rollout,
                episode_returns,
                episode_lengths,
                returned_episode_returns,
                returned_episode_lengths,
                energy,
                update_counter,
            ) = runner_state

            sum_min_dists_end = jnp.array(0.0, dtype=jnp.float32)
            sum_occ_end = jnp.array(0.0, dtype=jnp.float32)
            sum_success_end = jnp.array(0.0, dtype=jnp.float32)
            sum_tilde_d_end = jnp.zeros((num_agents,), dtype=jnp.float32)
            done_count_end = jnp.array(0, dtype=jnp.int32)
            sum_team_return_end = jnp.array(0.0, dtype=jnp.float32)
            sum_team_len_end = jnp.array(0.0, dtype=jnp.float32)

            rollout_carry0 = (
                train_state,
                env_state,
                last_obs,
                rng_rollout,
                episode_returns,
                episode_lengths,
                returned_episode_returns,
                returned_episode_lengths,
                sum_min_dists_end,
                sum_occ_end,
                sum_success_end,
                sum_tilde_d_end,
                done_count_end,
                sum_team_return_end,
                sum_team_len_end,
            )

            rollout_carryT, traj_batch = jax.lax.scan(
                _env_step, rollout_carry0, None, int(config["NUM_STEPS"])
            )

            (
                train_state,
                env_state,
                last_obs,
                rng_rollout,
                episode_returns,
                episode_lengths,
                returned_episode_returns,
                returned_episode_lengths,
                sum_min_dists_end,
                sum_occ_end,
                sum_success_end,
                sum_tilde_d_end,
                done_count_end,
                sum_team_return_end,
                sum_team_len_end,
            ) = rollout_carryT

            last_obs_batch = batchify(last_obs, agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

            advantages, targets = _calculate_gae(traj_batch, last_val)
            vartd_k = vartd_path_a(
                traj_batch=traj_batch,
                last_val=last_val,
                num_envs=int(config["NUM_ENVS"]),
                num_agents=num_agents,
                gamma=float(config["GAMMA"]),
            )

            denom = jnp.maximum(done_count_end.astype(jnp.float32), 1.0)
            min_dists_end_mean = sum_min_dists_end / denom
            occupied_end_mean = sum_occ_end / denom
            csr_train = sum_success_end / denom
            dbar_end = min_dists_end_mean / float(num_agents)
            team_return_end_mean = sum_team_return_end / denom
            ep_len_end_mean = sum_team_len_end / denom
            tilde_d_end_mean = sum_tilde_d_end / denom

            if use_geom_funding:
                z_k, g_k = funding_from_geometry(
                    dbar_end=dbar_end, d0=d0, beta=float(beta), z_max=float(z_max), eps_norm=float(eps_norm)
                )
            else:
                z_k, g_k = funding_from_return(
                    rbar_end=team_return_end_mean,
                    r0=r0,
                    beta=float(beta),
                    z_max=float(z_max),
                    eps_norm=float(eps_norm),
                )

            income = alpha * g_k
            e_pre = energy
            u_bool = e_pre >= c_upd
            u_f = u_bool.astype(jnp.float32)

            def _do_update(_):
                rng_update = jax.random.fold_in(rng_update_base, update_counter)
                train_state_new, update_metrics = _run_ppo_update(
                    train_state_in=train_state,
                    traj_batch=traj_batch,
                    advantages=advantages,
                    targets=targets,
                    rng_update=rng_update,
                )
                return train_state_new, update_metrics, update_counter + jnp.array(1, dtype=jnp.int32)

            def _dont_update(_):
                train_state_new, update_metrics = _skip_update(train_state)
                return train_state_new, update_metrics, update_counter

            train_state_next, update_metrics, update_counter_next = jax.lax.cond(
                u_bool,
                _do_update,
                _dont_update,
                operand=None,
            )

            e_post = jnp.clip(e_pre - c_upd * u_f + income, 0.0, e_max).astype(jnp.float32)

            tilde_d_metrics = {
                f"terminal/tilde_d_end_mean_l{i}": tilde_d_end_mean[i].astype(jnp.float32)
                for i in range(num_agents)
            }

            metric = {
                "update": jnp.asarray(k, dtype=jnp.int32),
                "episode/return_end_mean": team_return_end_mean.astype(jnp.float32),
                "episode/len_end_mean": ep_len_end_mean.astype(jnp.float32),
                "returned_episode_returns": team_return_end_mean.astype(jnp.float32),
                "returned_episode_lengths": ep_len_end_mean.astype(jnp.float32),
                "terminal/min_dists_end_mean": min_dists_end_mean.astype(jnp.float32),
                "terminal/occupied_end_mean": occupied_end_mean.astype(jnp.float32),
                "terminal/CSR_train": csr_train.astype(jnp.float32),
                "terminal/dbar_end": dbar_end.astype(jnp.float32),
                "terminal/done_count_end": done_count_end.astype(jnp.int32),
                "energy/E_pre": e_pre.astype(jnp.float32),
                "energy/E_post": e_post.astype(jnp.float32),
                "funding/z": z_k.astype(jnp.float32),
                "funding/g": g_k.astype(jnp.float32),
                "funding/income": income.astype(jnp.float32),
                "update/u": u_f.astype(jnp.float32),
                "update/update_counter": update_counter_next.astype(jnp.int32),
                "update/opt_step": update_metrics["opt_step"].astype(jnp.int32),
                "vartd/VarTD": vartd_k.astype(jnp.float32),
                "total_loss": update_metrics["total_loss"],
                "actor_loss": update_metrics["actor_loss"],
                "critic_loss": update_metrics["critic_loss"],
                "entropy": update_metrics["entropy"],
                "ratio0": update_metrics["ratio0"],
                **tilde_d_metrics,
            }

            runner_state_next = (
                train_state_next,
                env_state,
                last_obs,
                rng_rollout,
                episode_returns,
                episode_lengths,
                returned_episode_returns,
                returned_episode_lengths,
                e_post,
                update_counter_next,
            )
            return runner_state_next, metric

        def _run_eval(train_state_eval):
            if not do_eval:
                return {
                    "done_count": jnp.array(0, dtype=jnp.int32),
                    "csr": jnp.array(jnp.nan, dtype=jnp.float32),
                    "len_end_mean": jnp.array(jnp.nan, dtype=jnp.float32),
                    "return_end_mean": jnp.array(jnp.nan, dtype=jnp.float32),
                    "dbar_end": jnp.array(jnp.nan, dtype=jnp.float32),
                    "occupied_end_mean": jnp.array(jnp.nan, dtype=jnp.float32),
                    "max_tilde_end_mean": jnp.array(jnp.nan, dtype=jnp.float32),
                    "csr_pre": jnp.array(jnp.nan, dtype=jnp.float32),
                    "dbar_pre": jnp.array(jnp.nan, dtype=jnp.float32),
                    "occupied_pre_mean": jnp.array(jnp.nan, dtype=jnp.float32),
                    "max_tilde_pre_mean": jnp.array(jnp.nan, dtype=jnp.float32),
                    "csr_post": jnp.array(jnp.nan, dtype=jnp.float32),
                    "dbar_post": jnp.array(jnp.nan, dtype=jnp.float32),
                    "occupied_post_mean": jnp.array(jnp.nan, dtype=jnp.float32),
                    "max_tilde_post_mean": jnp.array(jnp.nan, dtype=jnp.float32),
                    "csr_pre_sweep": jnp.full((len(EVAL_CSR_EPS_GRID),), jnp.nan, dtype=jnp.float32),
                    "csr_post_sweep": jnp.full((len(EVAL_CSR_EPS_GRID),), jnp.nan, dtype=jnp.float32),
                    "eps_grid": jnp.asarray(EVAL_CSR_EPS_GRID, dtype=jnp.float32),
                    "episodes_per_env": jnp.array(0, dtype=jnp.int32),
                    "t_eval": jnp.array(0, dtype=jnp.int32),
                    "done_per_env_min": jnp.array(0, dtype=jnp.int32),
                    "done_per_env_max": jnp.array(0, dtype=jnp.int32),
                }

            rng_eval_reset, rng_eval_loop = jax.random.split(rng_eval_base)
            reset_keys = jax.random.split(rng_eval_reset, config["NUM_ENVS"])
            obs_eval, env_state_eval = reset_v(reset_keys)
            episode_returns_eval = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.float32)
            episode_lengths_eval = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.float32)
            done_per_env = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)

            def _eval_step(carry, _t):
                (
                    obs_dict,
                    env_state_local,
                    rng_eval_local,
                    episode_returns_local,
                    episode_lengths_local,
                    done_per_env_local,
                    sum_min_dists_pre,
                    sum_occ_pre,
                    sum_success_pre,
                    sum_max_tilde_pre,
                    sum_success_pre_sweep,
                    sum_min_dists_post,
                    sum_occ_post,
                    sum_success_post,
                    sum_max_tilde_post,
                    sum_success_post_sweep,
                    done_count_end,
                    sum_team_return_end,
                    sum_team_len_end,
                ) = carry

                rng_eval_local, key_step, key_reset = jax.random.split(rng_eval_local, 3)
                keys_step = jax.random.split(key_step, config["NUM_ENVS"])
                keys_reset = jax.random.split(key_reset, config["NUM_ENVS"])

                obs_pre_arr = batchify_ea(obs_dict, agents)
                obs_batch = batchify(obs_dict, agents, config["NUM_ACTORS"])
                pi, _ = network.apply(train_state_eval.params, obs_batch)
                action = pi.mode()
                env_act = unbatchify(action, agents, config["NUM_ENVS"], num_agents)

                obs_post, env_state_post, reward, done, _info_env = step_env_v(
                    keys_step, env_state_local, env_act
                )
                done_all = done["__all__"]
                done_f = done_all.astype(jnp.float32)
                done_i = done_all.astype(jnp.int32)

                r_arr = jnp.stack([reward[a] for a in agents], axis=1)
                rewardlog_arr = r_arr * float(num_agents)

                new_episode_returns = episode_returns_local + rewardlog_arr
                new_episode_lengths = episode_lengths_local + 1

                episode_returns_local = new_episode_returns * (1.0 - done_f)[:, None]
                episode_lengths_local = new_episode_lengths * (1.0 - done_f)[:, None]
                done_per_env_local = done_per_env_local + done_i

                obs_post_arr = batchify_ea(obs_post, agents)

                (
                    _tilde_d_pre,
                    min_dists_pre,
                    occupied_pre,
                    success_pre,
                    success_pre_sweep,
                    max_tilde_pre,
                ) = compute_terminal_geometry_with_sweep(
                    obs_pre_arr, N=num_agents, eps_occ=eps_occ, eps_grid=eval_eps_grid
                )
                (
                    _tilde_d_post,
                    min_dists_post,
                    occupied_post,
                    success_post,
                    success_post_sweep,
                    max_tilde_post,
                ) = compute_terminal_geometry_with_sweep(
                    obs_post_arr, N=num_agents, eps_occ=eps_occ, eps_grid=eval_eps_grid
                )

                sum_min_dists_pre = sum_min_dists_pre + (min_dists_pre * done_f).sum()
                sum_occ_pre = sum_occ_pre + (occupied_pre.astype(jnp.float32) * done_f).sum()
                sum_success_pre = sum_success_pre + (success_pre.astype(jnp.float32) * done_f).sum()
                sum_max_tilde_pre = sum_max_tilde_pre + (max_tilde_pre * done_f).sum()
                sum_success_pre_sweep = sum_success_pre_sweep + (
                    success_pre_sweep.astype(jnp.float32) * done_f[:, None]
                ).sum(axis=0)

                sum_min_dists_post = sum_min_dists_post + (min_dists_post * done_f).sum()
                sum_occ_post = sum_occ_post + (occupied_post.astype(jnp.float32) * done_f).sum()
                sum_success_post = sum_success_post + (success_post.astype(jnp.float32) * done_f).sum()
                sum_max_tilde_post = sum_max_tilde_post + (max_tilde_post * done_f).sum()
                sum_success_post_sweep = sum_success_post_sweep + (
                    success_post_sweep.astype(jnp.float32) * done_f[:, None]
                ).sum(axis=0)

                done_count_end = done_count_end + done_i.sum()

                team_return_end = new_episode_returns.mean(axis=1) / float(num_agents)
                sum_team_return_end = sum_team_return_end + (team_return_end * done_f).sum()
                ep_len_end = new_episode_lengths[:, 0].astype(jnp.float32)
                sum_team_len_end = sum_team_len_end + (ep_len_end * done_f).sum()

                obs_reset, env_state_reset = reset_v(keys_reset)
                obs_next = tree_where(done_all, obs_reset, obs_post)
                env_state_next = tree_where(done_all, env_state_reset, env_state_post)

                carry_out = (
                    obs_next,
                    env_state_next,
                    rng_eval_local,
                    episode_returns_local,
                    episode_lengths_local,
                    done_per_env_local,
                    sum_min_dists_pre,
                    sum_occ_pre,
                    sum_success_pre,
                    sum_max_tilde_pre,
                    sum_success_pre_sweep,
                    sum_min_dists_post,
                    sum_occ_post,
                    sum_success_post,
                    sum_max_tilde_post,
                    sum_success_post_sweep,
                    done_count_end,
                    sum_team_return_end,
                    sum_team_len_end,
                )
                return carry_out, None

            carry0 = (
                obs_eval,
                env_state_eval,
                rng_eval_loop,
                episode_returns_eval,
                episode_lengths_eval,
                done_per_env,
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.zeros((len(EVAL_CSR_EPS_GRID),), dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.zeros((len(EVAL_CSR_EPS_GRID),), dtype=jnp.float32),
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
            )
            carryT, _ = jax.lax.scan(_eval_step, carry0, None, t_eval)
            (
                _obs_eval,
                _env_state_eval,
                _rng_eval_final,
                _episode_returns_eval,
                _episode_lengths_eval,
                done_per_env_final,
                sum_min_dists_pre,
                sum_occ_pre,
                sum_success_pre,
                sum_max_tilde_pre,
                sum_success_pre_sweep,
                sum_min_dists_post,
                sum_occ_post,
                sum_success_post,
                sum_max_tilde_post,
                sum_success_post_sweep,
                done_count_end,
                sum_team_return_end,
                sum_team_len_end,
            ) = carryT

            denom = jnp.maximum(done_count_end.astype(jnp.float32), 1.0)
            csr_pre = sum_success_pre / denom
            csr_post = sum_success_post / denom
            dbar_pre = (sum_min_dists_pre / denom) / float(num_agents)
            dbar_post = (sum_min_dists_post / denom) / float(num_agents)
            occupied_pre_mean = sum_occ_pre / denom
            occupied_post_mean = sum_occ_post / denom
            max_tilde_pre_mean = sum_max_tilde_pre / denom
            max_tilde_post_mean = sum_max_tilde_post / denom
            csr_pre_sweep = sum_success_pre_sweep / denom
            csr_post_sweep = sum_success_post_sweep / denom
            return_end_mean = sum_team_return_end / denom
            len_end_mean = sum_team_len_end / denom

            return {
                "done_count": done_count_end.astype(jnp.int32),
                "csr": csr_post.astype(jnp.float32),
                "len_end_mean": len_end_mean.astype(jnp.float32),
                "return_end_mean": return_end_mean.astype(jnp.float32),
                "dbar_end": dbar_post.astype(jnp.float32),
                "occupied_end_mean": occupied_post_mean.astype(jnp.float32),
                "max_tilde_end_mean": max_tilde_post_mean.astype(jnp.float32),
                "csr_pre": csr_pre.astype(jnp.float32),
                "dbar_pre": dbar_pre.astype(jnp.float32),
                "occupied_pre_mean": occupied_pre_mean.astype(jnp.float32),
                "max_tilde_pre_mean": max_tilde_pre_mean.astype(jnp.float32),
                "csr_post": csr_post.astype(jnp.float32),
                "dbar_post": dbar_post.astype(jnp.float32),
                "occupied_post_mean": occupied_post_mean.astype(jnp.float32),
                "max_tilde_post_mean": max_tilde_post_mean.astype(jnp.float32),
                "csr_pre_sweep": csr_pre_sweep.astype(jnp.float32),
                "csr_post_sweep": csr_post_sweep.astype(jnp.float32),
                "eps_grid": eval_eps_grid.astype(jnp.float32),
                "episodes_per_env": jnp.asarray(episodes_per_env, dtype=jnp.int32),
                "t_eval": jnp.asarray(t_eval, dtype=jnp.int32),
                "done_per_env_min": done_per_env_final.min().astype(jnp.int32),
                "done_per_env_max": done_per_env_final.max().astype(jnp.int32),
            }

        runner_state0 = (
            train_state,
            env_state,
            obsv,
            rng_rollout0,
            episode_returns,
            episode_lengths,
            returned_episode_returns,
            returned_episode_lengths,
            e0,
            jnp.array(0, dtype=jnp.int32),
        )

        ks = jnp.arange(config["NUM_UPDATES"], dtype=jnp.int32)
        runner_stateT, metrics = jax.lax.scan(_update_step, runner_state0, ks)
        train_state_final = runner_stateT[0]
        eval_metrics = _run_eval(train_state_final)

        return {"runner_state": runner_stateT, "metrics": metrics, "eval": eval_metrics}

    return train


def _maybe_wandb_log_history(config, out):
    if str(config.get("WANDB_MODE", "disabled")) == "disabled":
        return

    metrics_mean = tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)).mean(axis=0), out["metrics"])
    num_updates = int(metrics_mean["update"].shape[0])

    for i in range(num_updates):
        row = {}
        for k, v in metrics_mean.items():
            if np.ndim(v) != 1:
                continue
            val = v[i]
            if np.issubdtype(np.asarray(val).dtype, np.integer):
                row[k] = int(val)
            else:
                row[k] = float(val)
        wandb.log(row, step=int(metrics_mean["update"][i]))

    eval_mean = tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)).mean(axis=0), out["eval"])
    wandb.log(
        {
            "eval/done_count": float(eval_mean["done_count"]),
            "eval/CSR": float(eval_mean["csr"]),
            "eval/len_end_mean": float(eval_mean["len_end_mean"]),
            "eval/return_end_mean": float(eval_mean["return_end_mean"]),
            "eval/dbar_end": float(eval_mean["dbar_end"]),
        },
        step=int(metrics_mean["update"][-1]),
    )


def _print_gate5_lines(config, out):
    metrics_full = tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), out["metrics"])
    metrics_mean = tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)).mean(axis=0), out["metrics"])
    eval_mean = tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)).mean(axis=0), out["eval"])

    num_updates = int(metrics_mean["update"].shape[0])
    print_every = int(config.get("PRINT_EVERY", 10))

    for i in range(0, num_updates, max(print_every, 1)):
        upd = int(metrics_mean["update"][i])
        print(
            f"[upd {upd}] "
            f"u={float(metrics_mean['update/u'][i]):.0f} "
            f"upd_ctr={int(metrics_mean['update/update_counter'][i])} "
            f"opt_step={int(metrics_mean['update/opt_step'][i])} "
            f"E={float(metrics_mean['energy/E_post'][i]):.3f} "
            f"g={float(metrics_mean['funding/g'][i]):.3f} "
            f"VarTD={float(metrics_mean['vartd/VarTD'][i]):.6f} "
            f"ret_end={float(metrics_mean['episode/return_end_mean'][i]):.3f} "
            f"dbar_end={float(metrics_mean['terminal/dbar_end'][i]):.3f} "
            f"occ_end={float(metrics_mean['terminal/occupied_end_mean'][i]):.3f} "
            f"CSR_train={float(metrics_mean['terminal/CSR_train'][i]):.3f} "
            f"done_count={int(metrics_mean['terminal/done_count_end'][i])}"
        )

    u_full = np.asarray(metrics_full["update/u"]).astype(np.int32)
    upd_ctr_full = np.asarray(metrics_full["update/update_counter"]).astype(np.int32)
    opt_step_full = np.asarray(metrics_full["update/opt_step"]).astype(np.int32)
    total_loss_full = np.asarray(metrics_full["total_loss"])

    expected_counter = np.cumsum(u_full, axis=1)
    ctr_t0_viol = int(np.sum(upd_ctr_full[:, 0] != expected_counter[:, 0])) if upd_ctr_full.size > 0 else 0
    ctr_diff_viol = int(np.sum(np.diff(upd_ctr_full, axis=1) != u_full[:, 1:])) if upd_ctr_full.shape[1] > 1 else 0

    expected_inc = int(config["NUM_MINIBATCHES"]) * int(config["UPDATE_EPOCHS"])
    opt_t0_viol = int(np.sum(opt_step_full[:, 0] != expected_inc * u_full[:, 0])) if opt_step_full.size > 0 else 0
    opt_diff_viol = int(np.sum(np.diff(opt_step_full, axis=1) != expected_inc * u_full[:, 1:])) if opt_step_full.shape[1] > 1 else 0

    nan_on_skip = True
    if np.any(u_full == 0):
        nan_on_skip = bool(np.all(np.isnan(total_loss_full[u_full == 0])))
    finite_on_update = True
    if np.any(u_full == 1):
        finite_on_update = bool(np.all(np.isfinite(total_loss_full[u_full == 1])))

    print(f"[Gate5 check] update_counter violations: t0={ctr_t0_viol} diffs={ctr_diff_viol}")
    print(f"[Gate5 check] opt_step violations: t0={opt_t0_viol} diffs={opt_diff_viol} (expected_inc={expected_inc})")
    print(f"[Gate5 check] NaN masking: nan_on_skip={nan_on_skip} finite_on_update={finite_on_update}")
    print(
        f"[Gate5 summary] mean_update_fraction={float(np.mean(u_full)):.4f} "
        f"final_executed_updates_mean={float(np.mean(upd_ctr_full[:, -1])):.2f}"
    )
    print(
        f"[Gate5 eval] done_count mean={float(eval_mean['done_count']):.1f} "
        f"CSR mean={float(eval_mean['csr']):.4f} "
        f"len_end_mean={float(eval_mean['len_end_mean']):.2f}"
    )

    if int(eval_mean["done_per_env_min"]) != int(eval_mean["episodes_per_env"]) or int(eval_mean["done_per_env_max"]) != int(eval_mean["episodes_per_env"]):
        print(
            "[Gate5 eval] WARNING: per-env eval done counts do not match episodes_per_env "
            f"(min/max={int(eval_mean['done_per_env_min'])}/{int(eval_mean['done_per_env_max'])}, "
            f"expected={int(eval_mean['episodes_per_env'])})"
        )
    print(
        f"[Gate5 eval] episodes_per_env={int(eval_mean['episodes_per_env'])} "
        f"T_eval={int(eval_mean['t_eval'])}"
    )

    eps_grid = np.asarray(eval_mean["eps_grid"], dtype=np.float32)
    csr_pre_sweep = np.asarray(eval_mean["csr_pre_sweep"], dtype=np.float32)
    csr_post_sweep = np.asarray(eval_mean["csr_post_sweep"], dtype=np.float32)
    sweep_pre = " ".join([f"CSR_pre@{eps:.2f}={val:.4f}" for eps, val in zip(eps_grid, csr_pre_sweep)])
    sweep_post = " ".join([f"CSR_post@{eps:.2f}={val:.4f}" for eps, val in zip(eps_grid, csr_post_sweep)])

    print(
        f"[Gate5 eval-debug] locked_eps_occ={float(config.get('EPS_OCC', 0.1)):.2f} "
        f"pre(step25): dbar_end={float(eval_mean['dbar_pre']):.6f} "
        f"occupied_end_mean={float(eval_mean['occupied_pre_mean']):.6f} "
        f"max_tilde_end_mean={float(eval_mean['max_tilde_pre_mean']):.6f} "
        f"CSR@locked={float(eval_mean['csr_pre']):.6f}"
    )
    print(
        f"[Gate5 eval-debug] locked_eps_occ={float(config.get('EPS_OCC', 0.1)):.2f} "
        f"post(step26): dbar_end={float(eval_mean['dbar_post']):.6f} "
        f"occupied_end_mean={float(eval_mean['occupied_post_mean']):.6f} "
        f"max_tilde_end_mean={float(eval_mean['max_tilde_post_mean']):.6f} "
        f"CSR@locked={float(eval_mean['csr_post']):.6f}"
    )
    print(f"[Gate5 eval-debug] {sweep_pre}")
    print(f"[Gate5 eval-debug] {sweep_post}")


def _save_gate5_plots(config, out):
    metrics_mean = tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)).mean(axis=0), out["metrics"])
    env_name = config["ENV_NAME"]

    plt.figure()
    plt.plot(metrics_mean["episode/return_end_mean"])
    plt.xlabel("Updates")
    plt.ylabel("Team return (episode ends)")
    plt.title(f"IPPO-FF Gate5: {env_name}")
    plt.tight_layout()
    plt.savefig(f"ippo_ff_gate5_return_{env_name}.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(metrics_mean["terminal/min_dists_end_mean"])
    plt.xlabel("Updates")
    plt.ylabel("min_dists_end_mean (sum over landmarks)")
    plt.title(f"Gate5 terminal min_dists_end_mean: {env_name}")
    plt.tight_layout()
    plt.savefig(f"ippo_ff_gate5_min_dists_end_{env_name}.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(metrics_mean["terminal/CSR_train"])
    plt.xlabel("Updates")
    plt.ylabel("CSR_train (eps_occ)")
    plt.title(f"Gate5 CSR_train (eps_occ={config.get('EPS_OCC', 0.1)}): {env_name}")
    plt.tight_layout()
    plt.savefig(f"ippo_ff_gate5_CSR_train_{env_name}.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(metrics_mean["energy/E_post"])
    plt.xlabel("Updates")
    plt.ylabel("Energy E_k")
    plt.title(f"Gate5 energy: {env_name}")
    plt.tight_layout()
    plt.savefig(f"ippo_ff_gate5_energy_{env_name}.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(metrics_mean["update/u"])
    plt.xlabel("Updates")
    plt.ylabel("Executed update fraction")
    plt.title(f"Gate5 update fraction: {env_name}")
    plt.tight_layout()
    plt.savefig(f"ippo_ff_gate5_update_fraction_{env_name}.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(metrics_mean["vartd/VarTD"])
    plt.xlabel("Updates")
    plt.ylabel("VarTD")
    plt.title(f"Gate5 VarTD: {env_name}")
    plt.tight_layout()
    plt.savefig(f"ippo_ff_gate5_VarTD_{env_name}.png", dpi=150)
    plt.close()


def _write_autosummary(config, out):
    metrics_mean = tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)).mean(axis=0), out["metrics"])
    eval_mean = tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)).mean(axis=0), out["eval"])
    eps_grid = np.asarray(eval_mean["eps_grid"], dtype=np.float32)
    csr_pre_sweep = np.asarray(eval_mean["csr_pre_sweep"], dtype=np.float32)
    csr_post_sweep = np.asarray(eval_mean["csr_post_sweep"], dtype=np.float32)
    sweep_pre = ",".join([f"{eps:.2f}:{val:.6f}" for eps, val in zip(eps_grid, csr_pre_sweep)])
    sweep_post = ",".join([f"{eps:.2f}:{val:.6f}" for eps, val in zip(eps_grid, csr_post_sweep)])
    text = "\n".join(
        [
            "Gate 5 auto-summary",
            "",
            f"ENV_NAME={config['ENV_NAME']}",
            f"NUM_UPDATES={int(metrics_mean['update'].shape[0])}",
            f"mean_update_fraction={float(np.mean(metrics_mean['update/u'])):.6f}",
            f"final_executed_updates_mean={float(np.mean(metrics_mean['update/update_counter'][-1])):.6f}",
            f"final_return_end_mean={float(metrics_mean['episode/return_end_mean'][-1]):.6f}",
            f"final_min_dists_end_mean={float(metrics_mean['terminal/min_dists_end_mean'][-1]):.6f}",
            f"final_occupied_end_mean={float(metrics_mean['terminal/occupied_end_mean'][-1]):.6f}",
            f"final_CSR_train={float(metrics_mean['terminal/CSR_train'][-1]):.6f}",
            f"final_energy={float(metrics_mean['energy/E_post'][-1]):.6f}",
            f"final_VarTD={float(metrics_mean['vartd/VarTD'][-1]):.6f}",
            f"eval_done_count_mean={float(eval_mean['done_count']):.6f}",
            f"eval_CSR_mean={float(eval_mean['csr']):.6f}",
            f"eval_len_end_mean={float(eval_mean['len_end_mean']):.6f}",
            f"eval_return_end_mean={float(eval_mean['return_end_mean']):.6f}",
            f"eval_dbar_end={float(eval_mean['dbar_end']):.6f}",
            f"eval_locked_eps_occ={float(config.get('EPS_OCC', 0.1)):.6f}",
            f"eval_csr_pre_mean={float(eval_mean['csr_pre']):.6f}",
            f"eval_dbar_pre={float(eval_mean['dbar_pre']):.6f}",
            f"eval_occupied_pre_mean={float(eval_mean['occupied_pre_mean']):.6f}",
            f"eval_max_tilde_pre_mean={float(eval_mean['max_tilde_pre_mean']):.6f}",
            f"eval_csr_post_mean={float(eval_mean['csr_post']):.6f}",
            f"eval_dbar_post={float(eval_mean['dbar_post']):.6f}",
            f"eval_occupied_post_mean={float(eval_mean['occupied_post_mean']):.6f}",
            f"eval_max_tilde_post_mean={float(eval_mean['max_tilde_post_mean']):.6f}",
            f"eval_csr_pre_sweep={sweep_pre}",
            f"eval_csr_post_sweep={sweep_post}",
            f"episodes_per_env={int(eval_mean['episodes_per_env'])}",
            f"T_eval={int(eval_mean['t_eval'])}",
        ]
    )
    Path("gate5_summary_AUTOGEN.txt").write_text(text + "\n", encoding="utf-8")


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_mpe")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    wandb.init(
        entity=config.get("ENTITY", ""),
        project=config.get("PROJECT", "jaxmarl-mpe"),
        tags=["IPPO", "FF", "Gate5", "step_env", "energy_gated"],
        config=config,
        mode=config.get("WANDB_MODE", "disabled"),
    )

    rng = jax.random.PRNGKey(int(config["SEED"]))
    rngs = jax.random.split(rng, int(config["NUM_SEEDS"]))
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    _maybe_wandb_log_history(config, out)
    _print_gate5_lines(config, out)
    _save_gate5_plots(config, out)
    _write_autosummary(config, out)


if __name__ == "__main__":
    main()
