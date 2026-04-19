"""
Gate 4 training script: IPPO FF on MPE with terminal-correct step_env + masked reset
and terminal geometry metrics computed at episode ends.

Fork of baselines/IPPO/ippo_ff_mpe.py (uploaded as baselines__IPPO__ippo_ff_mpe.py).

Key Gate 4 requirements satisfied:
(A) rollout collector steps with env.step_env (UNWRAPPED env), not env.step
(B) vmapped reset + vmapped step_env + masked reset (tree_where(done_all,...)) exactly like Gate 3
(C) episode-end accounting uses dones["__all__"] (fixed-shape JAX; no Python conditionals)
(D) terminal geometry metrics logged at episode ends: tilde_d per landmark, min_dists_end, occupied_end, CSR_train, dbar_end
(E) returned_episode_* reconstructed without wrappers (no LogWrapper dependency)
(F) outer scan outputs metrics only (traj_batch lives only inside _update_step)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import tree_util
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
from flax.training.train_state import TrainState
import distrax
import jaxmarl
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import wandb


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
    land_rel = obs_arr[:, :, 4 : 4 + 2 * N].reshape(E, A, N, 2)  # (E, A, L=N, 2)
    d = jnp.linalg.norm(land_rel, axis=-1)                      # (E, A, L)
    tilde_d = d.min(axis=1)                                     # (E, L)
    min_dists = tilde_d.sum(axis=1)                             # (E,)
    occupied = (tilde_d < eps_occ).sum(axis=1)                  # (E,)
    success = occupied == N
    return tilde_d, min_dists, occupied, success


# =========================
# PPO model (baseline)
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


# Baseline batchify/unbatchify (agent-major flattening for the policy net)
def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])

    def pad(z, length=max_dim):
        return jnp.concatenate(
            [z, jnp.zeros(z.shape[:-1] + (length - z.shape[-1],))], -1
        )

    x = jnp.stack(
        [x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list]
    )  # (A, E, obs_dim) or (A, E)
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    # -------------------------
    # Environment (UNWRAPPED)
    # -------------------------
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if not hasattr(env, "step_env"):
        raise RuntimeError("Env has no step_env method. Gate 2 required step_env. NO-GO.")

    agents = tuple(env.agents)
    N = int(env.num_agents)
    eps_occ = float(config.get("EPS_OCC", 0.1))

    # Gate 4 hard requirement: collector uses step_env + masked reset
    reset_v = jax.vmap(env.reset)
    step_env_v = jax.vmap(env.step_env)

    # -------------------------
    # Derived config (LOCKED shapes)
    # -------------------------
    config["NUM_ACTORS"] = int(env.num_agents * config["NUM_ENVS"])
    config["NUM_UPDATES"] = int(
        int(config["TOTAL_TIMESTEPS"]) // int(config["NUM_STEPS"]) // int(config["NUM_ENVS"])
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ACTORS"] * int(config["NUM_STEPS"]) // int(config["NUM_MINIBATCHES"])
    )

    # Optional sanity: NUM_STEPS must cover at least one full episode for end-metrics per update
    max_cycles = config.get("ENV_KWARGS", {}).get("max_cycles", None)
    if max_cycles is not None:
        assert int(config["NUM_STEPS"]) >= int(max_cycles), "LOCKED: NUM_STEPS >= max_cycles"

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space(agents[0]).n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(agents[0]).shape)
        network_params = network.init(_rng, init_x)

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
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = reset_v(reset_rng)

        # Wrapper-free logging state (same semantics as MPELogWrapper for returned_episode_* fields)
        episode_returns = jnp.zeros((config["NUM_ENVS"], N), dtype=jnp.float32)
        episode_lengths = jnp.zeros((config["NUM_ENVS"], N), dtype=jnp.float32)
        returned_episode_returns = jnp.zeros((config["NUM_ENVS"], N), dtype=jnp.float32)
        returned_episode_lengths = jnp.zeros((config["NUM_ENVS"], N), dtype=jnp.float32)

        # TRAIN LOOP
        def _update_step(runner_state, k):
            # -------------------------
            # Rollout collection (inner scan)
            # -------------------------
            def _env_step(carry, _t):
                (
                    train_state,
                    env_state,
                    last_obs,
                    rng,
                    episode_returns,
                    episode_lengths,
                    returned_episode_returns,
                    returned_episode_lengths,
                    # per-update accumulators (episode ends only)
                    sum_min_dists_end,
                    sum_occ_end,
                    sum_success_end,
                    sum_tilde_d_end,
                    done_count_end,
                    sum_team_return_end,
                    sum_team_len_end,
                ) = carry

                # Obs -> flat actor batch for policy/value
                obs_batch = batchify(last_obs, agents, config["NUM_ACTORS"])

                # RNG split (JAX-safe)
                rng, key_pi, key_step, key_reset = jax.random.split(rng, 4)
                keys_step = jax.random.split(key_step, config["NUM_ENVS"])
                keys_reset = jax.random.split(key_reset, config["NUM_ENVS"])

                # Policy forward + sample actions
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=key_pi)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, agents, config["NUM_ENVS"], N)

                # Terminal-correct transition (NO auto-reset)
                obs_post, env_state_post, reward, done, _info_env = step_env_v(
                    keys_step, env_state, env_act
                )
                done_all = done["__all__"]  # (E,) bool

                # -------------------------
                # returned_episode_* reconstruction (no wrappers)
                # Match MPELogWrapper reward scaling for logging only: rewardlog = reward * N
                # -------------------------
                r_arr = jnp.stack([reward[a] for a in agents], axis=1)  # (E, A)
                rewardlog_arr = r_arr * float(N)

                done_f = done_all.astype(jnp.float32)  # (E,)
                done_i = done_all.astype(jnp.int32)

                new_episode_returns = episode_returns + rewardlog_arr
                new_episode_lengths = episode_lengths + 1

                # Reset accumulators on done_all
                episode_returns = new_episode_returns * (1.0 - done_f)[:, None]
                episode_lengths = new_episode_lengths * (1.0 - done_f)[:, None]

                # Store last completed episode stats (held constant between terminals, like wrapper)
                returned_episode_returns = (
                    returned_episode_returns * (1.0 - done_f)[:, None] + new_episode_returns * done_f[:, None]
                )
                returned_episode_lengths = (
                    returned_episode_lengths * (1.0 - done_f)[:, None] + new_episode_lengths * done_f[:, None]
                )

                returned_episode_mask = jnp.broadcast_to(done_all[:, None], (config["NUM_ENVS"], N))

                # -------------------------
                # Terminal geometry metrics (episode ends only) from *post-transition obs*
                # -------------------------
                obs_post_arr = batchify_ea(obs_post, agents)  # (E, A, obs_dim)
                tilde_d, min_dists, occupied, success = compute_terminal_geometry_from_obs(
                    obs_post_arr, N=N, eps_occ=eps_occ
                )

                sum_min_dists_end = sum_min_dists_end + (min_dists * done_f).sum()
                sum_occ_end = sum_occ_end + (occupied.astype(jnp.float32) * done_f).sum()
                sum_success_end = sum_success_end + (success.astype(jnp.float32) * done_f).sum()
                sum_tilde_d_end = sum_tilde_d_end + (tilde_d * done_f[:, None]).sum(axis=0)
                done_count_end = done_count_end + done_i.sum()

                # Team return at episode end (unscaled mean over agents) for return-based fallback later
                team_return_end = new_episode_returns.mean(axis=1) / float(N)  # (E,)
                sum_team_return_end = sum_team_return_end + (team_return_end * done_f).sum()

                # Episode length at end (should be 26 = max_cycles+1)
                ep_len_end = new_episode_lengths[:, 0].astype(jnp.float32)  # (E,)
                sum_team_len_end = sum_team_len_end + (ep_len_end * done_f).sum()

                # -------------------------
                # Masked reset (vmapped reset, then select)
                # -------------------------
                obs_reset, env_state_reset = reset_v(keys_reset)
                obs_next = tree_where(done_all, obs_reset, obs_post)
                env_state_next = tree_where(done_all, env_state_reset, env_state_post)

                # Info dict for logging (shape: (NUM_ACTORS,))
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
                    rng,
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
                rng,
                episode_returns,
                episode_lengths,
                returned_episode_returns,
                returned_episode_lengths,
            ) = runner_state

            # Per-update accumulators init
            sum_min_dists_end = jnp.array(0.0, dtype=jnp.float32)
            sum_occ_end = jnp.array(0.0, dtype=jnp.float32)
            sum_success_end = jnp.array(0.0, dtype=jnp.float32)
            sum_tilde_d_end = jnp.zeros((N,), dtype=jnp.float32)
            done_count_end = jnp.array(0, dtype=jnp.int32)
            sum_team_return_end = jnp.array(0.0, dtype=jnp.float32)
            sum_team_len_end = jnp.array(0.0, dtype=jnp.float32)

            rollout_carry0 = (
                train_state,
                env_state,
                last_obs,
                rng,
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
                rng,
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

            # -------------------------
            # GAE computation (baseline)
            # -------------------------
            last_obs_batch = batchify(last_obs, agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # -------------------------
            # PPO update (baseline)
            # -------------------------
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(
                            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                        ) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)

                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "ratio": total_loss[1][3],
                    }
                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"], (
                    "batch size must be equal to number of steps * number of actors"
                )

                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            def callback(metric):
                upd = int(metric["update"])
                # Print a compact line every 10 updates into stdout.log for Gate 4 evidence.
                if upd % 10 == 0:
                    print(
                        f"[upd {upd}] return_end_mean={float(metric['episode/return_end_mean']):.3f} "
                        f"len_end_mean={float(metric['episode/len_end_mean']):.2f} "
                        f"min_dists_end_mean={float(metric['terminal/min_dists_end_mean']):.3f} "
                        f"dbar_end={float(metric['terminal/dbar_end']):.3f} "
                        f"CSR_train={float(metric['terminal/CSR_train']):.3f} "
                        f"done_count={int(metric['terminal/done_count_end'])}"
                    )
                wandb.log(metric, step=upd)

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            rng = update_state[-1]

            # -------------------------
            # Metrics (per update)
            # -------------------------
            denom = jnp.maximum(done_count_end.astype(jnp.float32), 1.0)

            min_dists_end_mean = sum_min_dists_end / denom
            occupied_end_mean = sum_occ_end / denom
            csr_train = sum_success_end / denom
            dbar_end = min_dists_end_mean / float(N)
            team_return_end_mean = sum_team_return_end / denom
            ep_len_end_mean = sum_team_len_end / denom
            tilde_d_end_mean = sum_tilde_d_end / denom  # (N,)

            tilde_d_metrics = {f"terminal/tilde_d_end_mean_l{i}": tilde_d_end_mean[i] for i in range(N)}

            r0 = {"ratio0": loss_info["ratio"][0, 0].mean()}
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            metric = {
                "update": jnp.asarray(k),
                # Standard episode metrics (wrapper-free, computed at episode ends)
                "episode/return_end_mean": team_return_end_mean,
                "episode/len_end_mean": ep_len_end_mean,
                # Backward-compatible names (existing plotting code expects these keys)
                "returned_episode_returns": team_return_end_mean,
                "returned_episode_lengths": ep_len_end_mean,
                # Terminal geometry metrics (episode ends only)
                "terminal/min_dists_end_mean": min_dists_end_mean,
                "terminal/occupied_end_mean": occupied_end_mean,
                "terminal/CSR_train": csr_train,
                "terminal/dbar_end": dbar_end,
                "terminal/done_count_end": done_count_end,
                **tilde_d_metrics,
                # PPO diagnostics
                **loss_info,
                **r0,
            }

            jax.experimental.io_callback(callback, None, metric)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                rng,
                episode_returns,
                episode_lengths,
                returned_episode_returns,
                returned_episode_lengths,
            )
            return runner_state, metric

        # Outer scan over updates (provide indices so callback can print/log with a real step)
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            _rng,
            episode_returns,
            episode_lengths,
            returned_episode_returns,
            returned_episode_lengths,
        )
        ks = jnp.arange(config["NUM_UPDATES"])
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, ks)

        return {"runner_state": runner_state, "metrics": metrics}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_mpe")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config.get("ENTITY", ""),
        project=config.get("PROJECT", "jaxmarl-mpe"),
        tags=["IPPO", "FF", "Gate4", "step_env"],
        config=config,
        mode=config.get("WANDB_MODE", "disabled"),
    )

    rng = jax.random.PRNGKey(int(config["SEED"]))
    rngs = jax.random.split(rng, int(config["NUM_SEEDS"]))
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    # --- Plots (saved into hydra.run.dir when hydra.job.chdir=True) ---
    plt.figure()
    plt.plot(out["metrics"]["episode/return_end_mean"].mean(axis=0))
    plt.xlabel("Updates")
    plt.ylabel("Team return (episode ends)")
    plt.title(f"IPPO-FF Gate4 (step_env): {config['ENV_NAME']}")
    plt.tight_layout()
    plt.savefig(f"ippo_ff_gate4_return_{config['ENV_NAME']}.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(out["metrics"]["terminal/min_dists_end_mean"].mean(axis=0))
    plt.xlabel("Updates")
    plt.ylabel("min_dists_end_mean (sum over landmarks)")
    plt.title(f"Terminal min_dists_end_mean: {config['ENV_NAME']}")
    plt.tight_layout()
    plt.savefig(f"ippo_ff_gate4_min_dists_end_{config['ENV_NAME']}.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(out["metrics"]["terminal/CSR_train"].mean(axis=0))
    plt.xlabel("Updates")
    plt.ylabel("CSR_train (eps_occ)")
    plt.title(f"Terminal CSR_train (eps_occ={config.get('EPS_OCC', 0.1)}): {config['ENV_NAME']}")
    plt.tight_layout()
    plt.savefig(f"ippo_ff_gate4_CSR_train_{config['ENV_NAME']}.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(out["metrics"]["episode/len_end_mean"].mean(axis=0))
    plt.xlabel("Updates")
    plt.ylabel("episode_len_end_mean")
    plt.title("Episode length at ends (should be 26 = max_cycles + 1)")
    plt.tight_layout()
    plt.savefig(f"ippo_ff_gate4_episode_len_{config['ENV_NAME']}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
