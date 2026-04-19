#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util

# Save a histogram PNG using a headless backend.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jaxmarl import make


# ============================================================
# Gate 3 — LOCKED configuration (matches Gate 1/2 reference point)
# ============================================================
ENV_NAME = "MPE_simple_spread_v3"
N = 3
LOCAL_RATIO = 0.5
MAX_CYCLES = 25
ACTION_TYPE = "Discrete"

NUM_ENVS = 25

# PettingZoo Simple Spread uses eps_occ=0.1 in benchmark_data for occupied landmarks.
EPS_OCC = 0.1

# Reconstruction / invariance tolerance
OBS_TOL = 1e-5

# Random-policy calibration (locked static-shape episode counting)
B_EPISODES_TOTAL = 100  # MUST be multiple of NUM_ENVS
assert B_EPISODES_TOTAL % NUM_ENVS == 0
EPISODES_PER_ENV = B_EPISODES_TOTAL // NUM_ENVS

# IMPORTANT: In JaxMARL MPE, done is computed using state.step >= max_steps
# BEFORE state.step is incremented, so one episode takes (max_cycles + 1) transitions.
EPISODE_LEN = MAX_CYCLES + 1
T_STEPS = EPISODES_PER_ENV * EPISODE_LEN

# Simple Spread Discrete action space has 5 actions: [no-op, left, right, down, up]
N_ACTIONS_DISCRETE = 5


# =========================
# Helper utilities (LOCKED)
# =========================
def broadcast_mask(mask_bool: jnp.ndarray, leaf_ndim: int) -> jnp.ndarray:
    """Broadcast (E,) -> (E, 1, 1, ..., 1) to match a leaf."""
    return mask_bool.reshape((mask_bool.shape[0],) + (1,) * (leaf_ndim - 1))


def tree_where(mask_bool: jnp.ndarray, tree_true: Any, tree_false: Any) -> Any:
    """Elementwise selection across a pytree; choose from tree_true where mask_bool is True."""
    return tree_util.tree_map(
        lambda a, b: jnp.where(broadcast_mask(mask_bool, a.ndim), a, b),
        tree_true,
        tree_false,
    )


def batchify(obs_dict: Dict[str, jnp.ndarray], agents: Tuple[str, ...]) -> jnp.ndarray:
    """{agent: (E, obs_dim)} -> (E, A, obs_dim) using env.agents as the static order."""
    return jnp.stack([obs_dict[a] for a in agents], axis=1)


def unbatchify(action_arr: jnp.ndarray, agents: Tuple[str, ...]) -> Dict[str, jnp.ndarray]:
    """(E, A) -> {agent: (E,)} using env.agents as the static order."""
    return {a: action_arr[:, i] for i, a in enumerate(agents)}


def compute_terminal_geometry_from_obs(
    obs_arr: jnp.ndarray, eps_occ: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    obs_arr: (E, A, obs_dim)

    Nominal PettingZoo-layout hypothesis (validated earlier in this script):
      self_vel = obs[..., 0:2]
      self_pos = obs[..., 2:4]
      landmark_rel = obs[..., 4:4+2N] reshaped to (..., N, 2)

    Returns:
      tilde_d:  (E, N)   per-landmark min distance to closest agent
      min_dists:(E,)     sum over landmarks of tilde_d
      occupied:(E,)      number of landmarks with tilde_d < eps_occ
      success: (E,) bool occupied == N
    """
    E = obs_arr.shape[0]
    land_rel = obs_arr[:, :, 4 : 4 + 2 * N].reshape(E, N, N, 2)   # (E, A, L=N, 2)
    d = jnp.linalg.norm(land_rel, axis=-1)                        # (E, A, L)
    tilde_d = d.min(axis=1)                                       # (E, L)
    min_dists = tilde_d.sum(axis=1)                               # (E,)
    occupied = (tilde_d < eps_occ).sum(axis=1)                    # (E,)
    success = occupied == N
    return tilde_d, min_dists, occupied, success


def get_repo_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def main() -> int:
    outdir = Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Gate 3: terminal geometry + obs-layout contract + random calibration (d0, R0) ===")
    print(f"Output dir: {outdir}")

    # ============================================================
    # 1) Instantiate the locked env
    # ============================================================
    env = make(
        ENV_NAME,
        N=N,
        local_ratio=LOCAL_RATIO,
        max_cycles=MAX_CYCLES,
        action_type=ACTION_TYPE,
    )

    agents = tuple(env.agents)
    if len(agents) != N:
        raise RuntimeError(f"Expected {N} agents, got {len(agents)}: {agents}")

    if not hasattr(env, "step_env"):
        raise RuntimeError("Env has no step_env method. Gate 2 was supposed to verify this. NO-GO.")

    print(f"Env: {ENV_NAME} N={N} local_ratio={LOCAL_RATIO} max_cycles={MAX_CYCLES} action_type={ACTION_TYPE}")
    print(f"Agents (static order): {agents}")
    print(f"NUM_ENVS={NUM_ENVS} episodes_per_env={EPISODES_PER_ENV} episode_len={EPISODE_LEN} steps={T_STEPS}")
    print(f"eps_occ={EPS_OCC}")
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    reset_v = jax.vmap(env.reset)
    step_env_v = jax.vmap(env.step_env)

    # ============================================================
    # 2) Obs-layout contract checks (batchify round-trip + reconstruction + invariance)
    # ============================================================
    key = jax.random.PRNGKey(0)
    key, key_reset0 = jax.random.split(key, 2)
    keys_reset0 = jax.random.split(key_reset0, NUM_ENVS)
    obs0_dict, state0 = reset_v(keys_reset0)

    obs0_arr = batchify(obs0_dict, agents)  # (E, A, obs_dim)
    obs_dim = int(obs0_arr.shape[-1])
    print(f"Obs dim per agent = {obs_dim}")

    # Batchify round-trip must be exact (ensures agent ordering is consistent)
    round_trip_dict = {a: obs0_arr[:, i, :] for i, a in enumerate(agents)}
    max_roundtrip_err = float(jnp.max(jnp.abs(batchify(round_trip_dict, agents) - obs0_arr)))
    print(f"Batchify round-trip max abs error = {max_roundtrip_err:.3e}")
    if max_roundtrip_err != 0.0:
        raise RuntimeError("Batchify/unbatchify round-trip failed. NO-GO.")

    # Nominal slicing hypothesis
    self_pos = obs0_arr[:, :, 2:4]  # (E, A, 2)
    land_rel = obs0_arr[:, :, 4 : 4 + 2 * N].reshape(NUM_ENVS, N, N, 2)  # (E, A, L=N, 2)

    # Reconstruction: landmark absolute position computed by all agents should agree.
    land_abs_est = self_pos[:, :, None, :] + land_rel  # (E, A, L, 2)
    land_abs_ref = land_abs_est[:, 0, :, :]            # (E, L, 2)
    recon_err = jnp.max(jnp.linalg.norm(land_abs_est - land_abs_ref[:, None, :, :], axis=-1))
    recon_err_f = float(recon_err)
    print(f"Reconstruction max error = {recon_err_f:.3e}")
    if recon_err_f > OBS_TOL:
        raise RuntimeError(f"Reconstruction check failed (>{OBS_TOL}). Obs slicing likely wrong. NO-GO.")

    # Translation invariance: distances from relative vectors vs distances from reconstructed global positions
    d_direct = jnp.linalg.norm(land_rel, axis=-1)  # (E, A, L)
    d_global = jnp.linalg.norm(land_abs_ref[:, None, :, :] - self_pos[:, :, None, :], axis=-1)
    trans_err = jnp.max(jnp.abs(d_direct - d_global))
    trans_err_f = float(trans_err)
    print(f"Translation invariance max abs diff = {trans_err_f:.3e}")
    if trans_err_f > OBS_TOL:
        raise RuntimeError(f"Translation invariance check failed (>{OBS_TOL}). NO-GO.")

    # Extra shift sanity: translate both landmark+agent positions by c; distances must not change.
    key, key_shift = jax.random.split(key, 2)
    c = jax.random.normal(key_shift, shape=(NUM_ENVS, 2)) * 0.5
    land_abs_shift = land_abs_ref + c[:, None, :]
    self_pos_shift = self_pos + c[:, None, :]
    d_shift = jnp.linalg.norm(land_abs_shift[:, None, :, :] - self_pos_shift[:, :, None, :], axis=-1)
    shift_err = jnp.max(jnp.abs(d_shift - d_global))
    shift_err_f = float(shift_err)
    print(f"Translation shift invariance max abs diff = {shift_err_f:.3e}")
    if shift_err_f > OBS_TOL:
        raise RuntimeError(f"Shift invariance check failed (>{OBS_TOL}). NO-GO.")

    obs_contract = {
        "env": {
            "name": ENV_NAME,
            "N": N,
            "local_ratio": LOCAL_RATIO,
            "max_cycles": MAX_CYCLES,
            "action_type": ACTION_TYPE,
        },
        "agents_order": list(agents),
        "obs_dim": obs_dim,
        "slicing_hypothesis": {
            "self_vel": [0, 2],
            "self_pos": [2, 4],
            "landmark_rel": [4, 4 + 2 * N],
            "landmark_rel_reshape": [N, 2],
        },
        "checks": {
            "tol": OBS_TOL,
            "batchify_roundtrip_max_abs_error": max_roundtrip_err,
            "reconstruction_max_error": recon_err_f,
            "translation_invariance_max_abs_diff": trans_err_f,
            "translation_shift_invariance_max_abs_diff": shift_err_f,
        },
        "seed": 0,
    }
    (outdir / "obs_layout_contract.json").write_text(json.dumps(obs_contract, indent=2))
    print("Wrote obs_layout_contract.json")

    # ============================================================
    # 3) Random-policy calibration (d0, R0) using step_env + vmapped reset + masked reset
    # ============================================================
    def rollout_random(key_rollout: jax.Array):
        key_rollout, key_reset = jax.random.split(key_rollout, 2)
        keys_reset = jax.random.split(key_reset, NUM_ENVS)
        obs_dict, env_state = reset_v(keys_reset)

        return_acc = jnp.zeros((NUM_ENVS,), dtype=jnp.float32)
        done_per_env = jnp.zeros((NUM_ENVS,), dtype=jnp.int32)

        done_count = jnp.array(0, dtype=jnp.int32)
        sum_min_dists_end = jnp.array(0.0, dtype=jnp.float32)
        sum_return_end = jnp.array(0.0, dtype=jnp.float32)
        sum_success_end = jnp.array(0.0, dtype=jnp.float32)

        def step_fn(carry, _t):
            (
                key_in,
                env_state_in,
                obs_dict_in,
                return_acc_in,
                done_per_env_in,
                done_count_in,
                sum_min_in,
                sum_ret_in,
                sum_succ_in,
            ) = carry

            # JAX-safe RNG splitting (no Python branching on traced values)
            key_in, key_step, key_reset, key_act = jax.random.split(key_in, 4)
            keys_step = jax.random.split(key_step, NUM_ENVS)
            keys_reset = jax.random.split(key_reset, NUM_ENVS)

            # Random discrete actions in [0, 5)
            a_arr = jax.random.randint(
                key_act,
                shape=(NUM_ENVS, N),
                minval=0,
                maxval=N_ACTIONS_DISCRETE,
                dtype=jnp.int32,
            )
            actions = unbatchify(a_arr, agents)

            # Terminal-correct step: step_env does NOT auto-reset; step() can auto-reset if done.
            obs_post, env_state_post, rewards, dones, _infos = step_env_v(keys_step, env_state_in, actions)
            done_all = dones["__all__"]  # (E,) bool

            # Team reward (mean over agents) per env-step
            r_arr = jnp.stack([rewards[a] for a in agents], axis=1)  # (E, A)
            r_mean = r_arr.mean(axis=1).astype(jnp.float32)          # (E,)
            return_acc_post = return_acc_in + r_mean

            # Terminal geometry from *post-transition obs* (before reset)
            obs_post_arr = batchify(obs_post, agents)  # (E, A, obs_dim)
            tilde_d, min_dists, _occupied, success = compute_terminal_geometry_from_obs(obs_post_arr, EPS_OCC)

            done_f = done_all.astype(jnp.float32)
            done_i = done_all.astype(jnp.int32)

            sum_min_in = sum_min_in + (done_f * min_dists).sum()
            sum_ret_in = sum_ret_in + (done_f * return_acc_post).sum()
            sum_succ_in = sum_succ_in + (done_f * success.astype(jnp.float32)).sum()
            done_count_in = done_count_in + done_i.sum()
            done_per_env_in = done_per_env_in + done_i

            # Save per-landmark terminal tilde_d for histogram (NaN on non-terminal steps)
            tilde_d_end = jnp.where(done_all[:, None], tilde_d, jnp.nan)  # (E, N)

            # Full reset for all envs, then masked overwrite where done
            obs_reset, env_state_reset = reset_v(keys_reset)
            obs_next = tree_where(done_all, obs_reset, obs_post)
            env_state_next = tree_where(done_all, env_state_reset, env_state_post)

            # Reset return accumulator on done
            return_acc_next = jnp.where(done_all, 0.0, return_acc_post)

            carry_out = (
                key_in,
                env_state_next,
                obs_next,
                return_acc_next,
                done_per_env_in,
                done_count_in,
                sum_min_in,
                sum_ret_in,
                sum_succ_in,
            )
            out = {"tilde_d_end": tilde_d_end}
            return carry_out, out

        carry0 = (
            key_rollout,
            env_state,
            obs_dict,
            return_acc,
            done_per_env,
            done_count,
            sum_min_dists_end,
            sum_return_end,
            sum_success_end,
        )

        xs = jnp.arange(T_STEPS)  # static-length scan
        carryT, outs = jax.lax.scan(step_fn, carry0, xs)

        (
            _keyT,
            _env_stateT,
            _obs_dictT,
            _return_accT,
            done_per_envT,
            done_countT,
            sum_minT,
            sum_retT,
            sum_succT,
        ) = carryT

        return {
            "done_per_env": done_per_envT,
            "done_count": done_countT,
            "sum_min_dists_end": sum_minT,
            "sum_return_end": sum_retT,
            "sum_success_end": sum_succT,
            "tilde_d_end_seq": outs["tilde_d_end"],  # (T, E, N) with NaNs on non-terminal steps
        }

    rollout_random_jit = jax.jit(rollout_random)

    key, key_rollout = jax.random.split(key, 2)

    # Compile once, then time the compiled execution
    _ = rollout_random_jit(key_rollout)["done_count"]
    jax.block_until_ready(_)

    t0 = time.time()
    result = rollout_random_jit(key_rollout)
    jax.block_until_ready(result["done_count"])
    wall = time.time() - t0

    sps = (NUM_ENVS * T_STEPS) / max(wall, 1e-9)
    print(f"Random rollout wall time (compiled) = {wall:.4f}s ; env-steps-per-second ≈ {sps:.1f}")

    done_per_env_np = jax.device_get(result["done_per_env"])
    done_count_np = int(jax.device_get(result["done_count"]))
    sum_min_np = float(jax.device_get(result["sum_min_dists_end"]))
    sum_ret_np = float(jax.device_get(result["sum_return_end"]))
    sum_succ_np = float(jax.device_get(result["sum_success_end"]))

    print(f"Done counts per env min/max = {int(done_per_env_np.min())}/{int(done_per_env_np.max())}")
    if int(done_per_env_np.min()) != EPISODES_PER_ENV or int(done_per_env_np.max()) != EPISODES_PER_ENV:
        raise RuntimeError(
            f"Masked reset loop is wrong: expected every env to finish exactly {EPISODES_PER_ENV} episodes."
        )
    if done_count_np != B_EPISODES_TOTAL:
        raise RuntimeError(f"Expected done_count={B_EPISODES_TOTAL}, got {done_count_np}. NO-GO.")

    # d0: mean per-landmark terminal distance
    d0 = (sum_min_np / done_count_np) / N
    # R0: mean team episode return (team reward defined as mean over agents)
    R0 = sum_ret_np / done_count_np
    csr_random = sum_succ_np / done_count_np
    print(f"Random baselines: d0={d0:.6f}, R0={R0:.6f}, CSR_random={csr_random:.6f}")

    # ============================================================
    # 4) eps_occ sanity histogram (terminal tilde_d values)
    # ============================================================
    import numpy as np

    tilde_d_end_seq = jax.device_get(result["tilde_d_end_seq"])  # (T, E, N)
    vals = tilde_d_end_seq.reshape(-1)
    vals = vals[np.isfinite(vals)]

    expected = B_EPISODES_TOTAL * N
    if vals.size != expected:
        print(f"WARNING: expected {expected} terminal tilde_d samples, got {vals.size}.")

    p_occ = float((vals < EPS_OCC).mean()) if vals.size > 0 else float("nan")
    print(f"Random policy: P(tilde_d < eps_occ) = {p_occ:.6f} (eps_occ={EPS_OCC})")

    plt.figure()
    plt.hist(vals, bins=50)
    plt.axvline(EPS_OCC)
    plt.title("Terminal per-landmark min distance (tilde_d) under random policy")
    plt.xlabel("tilde_d")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outdir / "eps_occ_sanity_histogram.png", dpi=150)
    plt.close()
    print("Wrote eps_occ_sanity_histogram.png")

    # ============================================================
    # 5) Write calibration JSON
    # ============================================================
    calib = {
        "env": {
            "name": ENV_NAME,
            "N": N,
            "local_ratio": LOCAL_RATIO,
            "max_cycles": MAX_CYCLES,
            "action_type": ACTION_TYPE,
        },
        "rollout": {
            "NUM_ENVS": NUM_ENVS,
            "B_EPISODES_TOTAL": B_EPISODES_TOTAL,
            "EPISODES_PER_ENV": EPISODES_PER_ENV,
            "T_STEPS": T_STEPS,
        },
        "metrics": {
            "eps_occ": EPS_OCC,
            "d0_mean_per_landmark_terminal_distance": d0,
            "R0_mean_team_episode_return": R0,
            "CSR_random": csr_random,
            "P_tilde_d_lt_eps_occ_random": p_occ,
        },
        "provenance": {
            "seed": 0,
            "git_commit": get_repo_commit_hash(),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "jax_version": getattr(jax, "__version__", None),
        },
        "timing": {
            "random_rollout_wall_s": wall,
            "approx_env_steps_per_second": sps,
        },
    }
    (outdir / "calibration_random_baselines.json").write_text(json.dumps(calib, indent=2))
    print("Wrote calibration_random_baselines.json")

    print("GO: terminal geometry metrics + calibration OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"NO-GO: {e}", file=sys.stderr)
        raise SystemExit(1)
