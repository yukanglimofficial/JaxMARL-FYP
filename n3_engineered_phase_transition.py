#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Locked N=3 engineered-transition defaults.
# This script is an EXPLICIT engineered extension. It is not the accepted thesis
# core and must not be used to rewrite the accepted Stage B/C boundary claims.
# -----------------------------------------------------------------------------
SCRIPT_VERSION = "n3_engineered_phase_transition_v2_rigorous_final"

REPO_ROOT_DEFAULT = Path.cwd()
ENV_NAME = "MPE_simple_spread_v3"
ENV_KWARGS = {
    "num_agents": 3,
    "num_landmarks": 3,
    "local_ratio": 0.5,
    "max_steps": 25,
    "action_type": "Discrete",
}
NUM_ENVS = 25
NUM_STEPS = 128
M_EVAL = 100
EPS_OCC = 0.2
E0 = 1.0
E_MAX = 1.0
D0 = 0.733348
R0 = -27.256548
BETA = float(np.log(10.0))
Z_MAX = 10.0
EPS_NORM = 1e-8
DEFAULT_LR = 2.5e-4
DEFAULT_UPDATE_EPOCHS = 4
DEFAULT_NUM_MINIBATCHES = 4
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_EPS = 0.2
DEFAULT_ENT_COEF = 0.01
DEFAULT_VF_COEF = 0.5
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_ACTIVATION = "tanh"
DEFAULT_ANNEAL_LR = True

DISCOVER_TIMESTEPS = 4_000_000
LOCKED_TIMESTEPS = 4_000_000
DISCOVER_SEED_START = 30
DISCOVER_SEED_STOP = 42  # 12 seeds by default
LOCK_SEED_START = 30
LOCK_SEED_STOP = 62      # 32 seeds by default
LATE_FRAC = 0.25
DISCOVER_ALPHA_VALUES = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]
DISCOVER_C_UPD_VALUES = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
DISCOVER_Z_CUT_VALUES = [1.00, 1.05, 1.10]
DEPR_LAMBDAS = [0.01, 0.02, 0.03]
DEPR_PSIS = [0.75, 1.00, 1.25]
DEPR_NU = 0.5
REFINE_ALPHA_STEP = 0.001
REFINE_PAD_LEFT = 0.010
REFINE_PAD_RIGHT = 0.020
DISCOVER_ALPHA_MAX = 0.12
DEFAULT_RIGHT_EDGE_LATE_UPDATES_MIN = 5.0
DEFAULT_LEFT_EDGE_LATE_UPDATES_MAX = 1.0
MIN_LATE_UPDATES_FOR_ACTIVE = 3
LATE_BLOCKS = 4
MIN_ACTIVE_BLOCKS = 2
SELECTION_LADDER = [
    {
        "name": "strict",
        "left_p_max": 0.10,
        "right_p_min": 0.90,
        "left_late_updates_max": 1.0,
        "right_late_updates_min": 5.0,
        "max_p_viol": 1,
        "max_rho_viol": 1,
        "require_chi_peak": True,
        "chi_peak_ratio_min": 1.50,
    },
    {
        "name": "relaxed_1",
        "left_p_max": 0.15,
        "right_p_min": 0.85,
        "left_late_updates_max": 1.5,
        "right_late_updates_min": 4.0,
        "max_p_viol": 2,
        "max_rho_viol": 2,
        "require_chi_peak": True,
        "chi_peak_ratio_min": 1.30,
    },
    {
        "name": "relaxed_2",
        "left_p_max": 0.20,
        "right_p_min": 0.80,
        "left_late_updates_max": 2.0,
        "right_late_updates_min": 3.0,
        "max_p_viol": 3,
        "max_rho_viol": 3,
        "require_chi_peak": False,
        "chi_peak_ratio_min": 1.10,
    },
]
STRICT_PROFILE_NAME = "strict"

TRACE_ALIASES: Dict[str, List[str]] = {
    "update": ["update"],
    "update_u": ["update/u"],
    "update_counter": ["update/update_counter"],
    "energy_E_pre": ["energy/E_pre"],
    "energy_E_post": ["energy/E_post"],
    "funding_z": ["funding/z"],
    "funding_g": ["funding/g"],
    "funding_g_raw": ["funding/g_raw"],
    "funding_cut_active": ["funding/cut_active"],
    "funding_income": ["funding/income"],
    "funding_alpha_eff": ["funding/alpha_eff"],
    "episode_return_end_mean": ["episode/return_end_mean"],
    "terminal_min_dists_end_mean": ["terminal/min_dists_end_mean"],
    "terminal_dbar_end": ["terminal/dbar_end"],
    "terminal_occupied_end_mean": ["terminal/occupied_end_mean"],
    "terminal_CSR_train": ["terminal/CSR_train"],
    "vartd_VarTD": ["vartd/VarTD"],
    "depr_F": ["depr/F"],
    "depr_F_post": ["depr/F_post"],
}

EVAL_ALIASES: Dict[str, List[str]] = {
    "eval_done_count": ["done_count"],
    "eval_return_end_mean": ["return_end_mean"],
    "eval_len_end_mean": ["len_end_mean"],
    "eval_CSR_post": ["CSR_post", "csr_post", "csr"],
    "eval_dbar_post_mean": ["dbar_post_mean", "dbar_post", "dbar_end"],
    "eval_occupied_post_mean": ["occupied_post_mean", "occupied_end_mean"],
    "eval_episodes_per_env": ["episodes_per_env"],
    "eval_t_eval": ["t_eval"],
}

PLOT_METRICS = [
    ("update_u", "Executed-update indicator $u_k$", "u"),
    ("energy_E_post", "Post-update energy", "energy"),
    ("funding_g", "Effective funding $g_k^*$", "g"),
    ("depr_F_post", "Inactivity state $F_k$", "F"),
    ("episode_return_end_mean", "Training return", "return"),
    ("vartd_VarTD", "VarTD", "VarTD"),
]

HELPER_INSERT = '''

def hard_zero_filter(g_k, z_k, hard_zero_funding: bool, z_cut: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if hard_zero_funding:
        cut_active = z_k <= z_cut
        g_eff = jnp.where(cut_active, g_k, 0.0).astype(jnp.float32)
    else:
        cut_active = jnp.asarray(True)
        g_eff = g_k.astype(jnp.float32)
    return cut_active.astype(jnp.float32), g_eff.astype(jnp.float32)
'''


@dataclass(frozen=True)
class MechanismSpec:
    c_upd: float
    z_cut: float
    depr_lambda: float = 0.0
    depr_nu: float = DEPR_NU
    depr_psi: float = 0.0
    hard_zero_funding: bool = True

    @property
    def mechanism_name(self) -> str:
        if self.depr_lambda > 0.0 and self.depr_psi > 0.0:
            return "hardzero_plus_depr"
        return "hardzero_only"

    def tag(self) -> str:
        return (
            f"{self.mechanism_name}__c{self.c_upd:.2f}__z{self.z_cut:.2f}"
            f"__lam{self.depr_lambda:.3f}__nu{self.depr_nu:.2f}__psi{self.depr_psi:.2f}"
        ).replace(".", "p")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt(x: Any, digits: int = 6) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    if np.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def maybe_to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    try:
        return pd.to_numeric(series)
    except Exception:
        return series


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = maybe_to_numeric(out[c])
    return out


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    out.columns = ["_".join(str(x) for x in col if str(x) != "") for col in df.columns]
    return out


def q25(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.nanquantile(vals, 0.25))


def q75(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.nanquantile(vals, 0.75))


def first_present(mapping: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in mapping:
            return k
    return None


def install_wandb_stub() -> None:
    try:
        import wandb  # noqa: F401
        return
    except Exception:
        pass
    m = types.ModuleType("wandb")

    class DummyRun:
        def log(self, *args: Any, **kwargs: Any) -> None:
            return None

        def finish(self, *args: Any, **kwargs: Any) -> None:
            return None

    m.init = lambda *args, **kwargs: DummyRun()  # type: ignore[attr-defined]
    m.log = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    m.finish = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["wandb"] = m


def load_module_from_path(module_name: str, path: Path, repo_root: Path):
    install_wandb_stub()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def master_seed_key(seed: int):
    import jax
    return jax.random.split(jax.random.PRNGKey(int(seed)), 1)[0]


def block_until_ready(tree: Any) -> None:
    import jax

    def _block(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return x

    jax.tree_util.tree_map(_block, tree)


def to_numpy_tree(tree: Any) -> Any:
    import jax
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def infer_repo_root(explicit: Path) -> Path:
    explicit = explicit.resolve()
    if (explicit / "baselines" / "IPPO" / "ippo_ff_mpe_gate5_energy_gated.py").exists():
        return explicit
    candidate = explicit / "JAXMARL_gate5_repair"
    if (candidate / "baselines" / "IPPO" / "ippo_ff_mpe_gate5_energy_gated.py").exists():
        return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate repo root containing baselines/IPPO/ippo_ff_mpe_gate5_energy_gated.py"
    )


# -----------------------------------------------------------------------------
# Patch accepted master into the engineered hard-zero + depreciation master.
# -----------------------------------------------------------------------------
def patch_master_source(src: str) -> str:
    # 1) insert helper after funding_from_return
    marker = "return z_k.astype(jnp.float32), g_k.astype(jnp.float32)\n\n\ndef vartd_path_a("
    if marker not in src:
        raise RuntimeError("Could not find funding_from_return marker in accepted master script.")
    src = src.replace(marker, "return z_k.astype(jnp.float32), g_k.astype(jnp.float32)\n" + HELPER_INSERT + "\n\ndef vartd_path_a(", 1)

    # 2) defaults
    old_defaults = '''    config.setdefault("C_UPD", 0.0)
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
'''
    new_defaults = '''    config.setdefault("C_UPD", 0.0)
    config.setdefault("ALPHA", 0.0)
    config.setdefault("E0", 1.0)
    config.setdefault("E_MAX", 1.0)
    config.setdefault("HARD_ZERO_FUNDING", True)
    config.setdefault("Z_CUT", 1.0)
    config.setdefault("DEPR_LAMBDA", 0.0)
    config.setdefault("DEPR_NU", 0.5)
    config.setdefault("DEPR_PSI", 0.0)
    config.setdefault("BETA", float(np.log(10.0)))
    config.setdefault("Z_MAX", 10.0)
    config.setdefault("EPS_NORM", 1e-8)
    config.setdefault("D0", 0.733348)
    config.setdefault("R0", -27.256548)
    config.setdefault("USE_GEOM_FUNDING", True)
    config.setdefault("DO_EVAL", True)
    config.setdefault("M_EVAL", 100)
    config.setdefault("PRINT_EVERY", 10)
'''
    if old_defaults not in src:
        raise RuntimeError("Could not find default-config block.")
    src = src.replace(old_defaults, new_defaults, 1)

    # 3) derived config values
    old_derived = '''    c_upd = jnp.asarray(config["C_UPD"], dtype=jnp.float32)
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
'''
    new_derived = '''    c_upd = jnp.asarray(config["C_UPD"], dtype=jnp.float32)
    alpha = jnp.asarray(config["ALPHA"], dtype=jnp.float32)
    e0 = jnp.asarray(config["E0"], dtype=jnp.float32)
    e_max = jnp.asarray(config["E_MAX"], dtype=jnp.float32)
    hard_zero_funding = bool(config.get("HARD_ZERO_FUNDING", True))
    z_cut = jnp.asarray(config.get("Z_CUT", 1.0), dtype=jnp.float32)
    depr_lambda = jnp.asarray(config.get("DEPR_LAMBDA", 0.0), dtype=jnp.float32)
    depr_nu = jnp.asarray(config.get("DEPR_NU", 0.5), dtype=jnp.float32)
    depr_psi = jnp.asarray(config.get("DEPR_PSI", 0.0), dtype=jnp.float32)
    beta = jnp.asarray(config["BETA"], dtype=jnp.float32)
    z_max = jnp.asarray(config["Z_MAX"], dtype=jnp.float32)
    eps_norm = jnp.asarray(config["EPS_NORM"], dtype=jnp.float32)
    d0 = float(config["D0"])
    r0 = float(config["R0"])
    use_geom_funding = bool(config["USE_GEOM_FUNDING"])
    do_eval = bool(config["DO_EVAL"])
    m_eval = int(config["M_EVAL"])
'''
    if old_derived not in src:
        raise RuntimeError("Could not find derived-config block.")
    src = src.replace(old_derived, new_derived, 1)

    # 4) runner-state unpack includes F
    old_unpack = '''                returned_episode_returns,
                returned_episode_lengths,
                energy,
                update_counter,
            ) = runner_state
'''
    new_unpack = '''                returned_episode_returns,
                returned_episode_lengths,
                energy,
                depr_F,
                update_counter,
            ) = runner_state
'''
    if old_unpack not in src:
        raise RuntimeError("Could not find runner_state unpack block.")
    src = src.replace(old_unpack, new_unpack, 1)

    # 5) insert hard-zero/depr block at income assignment
    old_gate = '''            income = alpha * g_k
            e_pre = energy
            u_bool = e_pre >= c_upd
            u_f = u_bool.astype(jnp.float32)
'''
    new_gate = '''            cut_active, g_eff = hard_zero_filter(g_k, z_k, hard_zero_funding, z_cut)
            alpha_eff = alpha * jnp.exp(-depr_psi * depr_F)
            income = alpha_eff * g_eff
            e_pre = energy
            u_bool = e_pre >= c_upd
            u_f = u_bool.astype(jnp.float32)
'''
    if old_gate not in src:
        raise RuntimeError("Could not find gate block.")
    src = src.replace(old_gate, new_gate, 1)

    # 6) energy update plus inactivity-state update
    old_energy = '            e_post = jnp.clip(e_pre - c_upd * u_f + income, 0.0, e_max).astype(jnp.float32)\n'
    new_energy = '''            e_post = jnp.clip(e_pre - c_upd * u_f + income, 0.0, e_max).astype(jnp.float32)
            depr_F_post = jnp.maximum((1.0 - depr_nu * u_f) * depr_F + depr_lambda * (1.0 - u_f), 0.0).astype(jnp.float32)
'''
    if old_energy not in src:
        raise RuntimeError("Could not find energy update line.")
    src = src.replace(old_energy, new_energy, 1)

    # 7) metrics add effective/raw funding + depreciation state
    old_metric_line = '                "funding/g": g_k.astype(jnp.float32),\n'
    new_metric_line = '''                "funding/g": g_eff.astype(jnp.float32),
                "funding/g_raw": g_k.astype(jnp.float32),
                "funding/cut_active": cut_active.astype(jnp.float32),
                "funding/alpha_eff": alpha_eff.astype(jnp.float32),
                "depr/F": depr_F.astype(jnp.float32),
                "depr/F_post": depr_F_post.astype(jnp.float32),
'''
    if old_metric_line not in src:
        raise RuntimeError("Could not find funding/g metric line.")
    src = src.replace(old_metric_line, new_metric_line, 1)

    # 8) runner_state_next includes F
    old_next = '''                returned_episode_returns,
                returned_episode_lengths,
                e_post,
                update_counter_next,
            )
'''
    new_next = '''                returned_episode_returns,
                returned_episode_lengths,
                e_post,
                depr_F_post,
                update_counter_next,
            )
'''
    if old_next not in src:
        raise RuntimeError("Could not find runner_state_next block.")
    src = src.replace(old_next, new_next, 1)

    # 9) runner_state0 initializes F=0
    old_state0 = '''            returned_episode_returns,
            returned_episode_lengths,
            e0,
            jnp.array(0, dtype=jnp.int32),
        )
'''
    new_state0 = '''            returned_episode_returns,
            returned_episode_lengths,
            e0,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
        )
'''
    if old_state0 not in src:
        raise RuntimeError("Could not find runner_state0 block.")
    src = src.replace(old_state0, new_state0, 1)

    return src


def write_engineered_master(repo_root: Path, force: bool = False) -> Path:
    accepted = repo_root / "baselines" / "IPPO" / "ippo_ff_mpe_gate5_energy_gated.py"
    if not accepted.exists():
        raise FileNotFoundError(f"Accepted master script not found: {accepted}")
    derived = repo_root / "baselines" / "IPPO" / "ippo_ff_mpe_n3_engineered_phase_transition.py"
    patched = patch_master_source(accepted.read_text(encoding="utf-8"))
    if derived.exists() and not force and derived.read_text(encoding="utf-8") == patched:
        return derived
    derived.write_text(patched, encoding="utf-8")
    return derived


# -----------------------------------------------------------------------------
# Running the engineered master.
# -----------------------------------------------------------------------------
def build_config(*, alpha: float, mechanism: MechanismSpec, total_timesteps: int) -> Dict[str, Any]:
    return {
        "ENV_NAME": ENV_NAME,
        "ENV_KWARGS": dict(ENV_KWARGS),
        "LR": DEFAULT_LR,
        "NUM_ENVS": NUM_ENVS,
        "NUM_STEPS": NUM_STEPS,
        "TOTAL_TIMESTEPS": int(total_timesteps),
        "UPDATE_EPOCHS": DEFAULT_UPDATE_EPOCHS,
        "NUM_MINIBATCHES": DEFAULT_NUM_MINIBATCHES,
        "GAMMA": DEFAULT_GAMMA,
        "GAE_LAMBDA": DEFAULT_GAE_LAMBDA,
        "CLIP_EPS": DEFAULT_CLIP_EPS,
        "ENT_COEF": DEFAULT_ENT_COEF,
        "VF_COEF": DEFAULT_VF_COEF,
        "MAX_GRAD_NORM": DEFAULT_MAX_GRAD_NORM,
        "ACTIVATION": DEFAULT_ACTIVATION,
        "ANNEAL_LR": DEFAULT_ANNEAL_LR,
        "NUM_SEEDS": 1,
        "SEED": 0,
        "WANDB_MODE": "disabled",
        "ENTITY": "",
        "PROJECT": "JaxMARL_N3_EngineeredPhase",
        "PRINT_EVERY": 1_000_000_000,
        "EPS_OCC": EPS_OCC,
        "D0": D0,
        "R0": R0,
        "ALPHA": float(alpha),
        "C_UPD": float(mechanism.c_upd),
        "E0": E0,
        "E_MAX": E_MAX,
        "HARD_ZERO_FUNDING": bool(mechanism.hard_zero_funding),
        "Z_CUT": float(mechanism.z_cut),
        "DEPR_LAMBDA": float(mechanism.depr_lambda),
        "DEPR_NU": float(mechanism.depr_nu),
        "DEPR_PSI": float(mechanism.depr_psi),
        "BETA": BETA,
        "Z_MAX": Z_MAX,
        "EPS_NORM": EPS_NORM,
        "USE_GEOM_FUNDING": True,
        "DO_EVAL": True,
        "M_EVAL": M_EVAL,
    }


def extract_trace_df(metrics_np: Mapping[str, np.ndarray], *, alpha: float, mechanism: MechanismSpec, seed: int) -> pd.DataFrame:
    data: Dict[str, np.ndarray] = {}
    length: Optional[int] = None
    for out_key, aliases in TRACE_ALIASES.items():
        raw_key = first_present(metrics_np, aliases)
        if raw_key is None:
            continue
        arr = np.asarray(metrics_np[raw_key])
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D metric trace for {raw_key}, got {arr.shape}")
        if length is None:
            length = int(arr.shape[0])
        if arr.shape[0] != length:
            raise ValueError(f"Mismatched trace length for {raw_key}: {arr.shape[0]} vs {length}")
        data[out_key] = arr
    if length is None:
        raise RuntimeError("No metric traces extracted from engineered master output.")
    if "update" not in data:
        data["update"] = np.arange(length, dtype=int)
    df = pd.DataFrame(data)
    df.insert(0, "seed", int(seed))
    df.insert(0, "alpha", float(alpha))
    df.insert(0, "mechanism", mechanism.mechanism_name)
    df.insert(0, "mechanism_tag", mechanism.tag())
    df["env_steps"] = (pd.to_numeric(df["update"], errors="coerce") + 1) * NUM_ENVS * NUM_STEPS
    return df


def _scalar_from_array(x: Any) -> Optional[float]:
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return None


def extract_eval_row(eval_np: Mapping[str, np.ndarray], trace_df: pd.DataFrame, *, alpha: float, mechanism: MechanismSpec, seed: int, runtime_sec: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "alpha": float(alpha),
        "seed": int(seed),
        "runtime_sec": float(runtime_sec),
        "mechanism": mechanism.mechanism_name,
        "mechanism_tag": mechanism.tag(),
        "c_upd": float(mechanism.c_upd),
        "z_cut": float(mechanism.z_cut),
        "depr_lambda": float(mechanism.depr_lambda),
        "depr_nu": float(mechanism.depr_nu),
        "depr_psi": float(mechanism.depr_psi),
    }
    for out_key, aliases in EVAL_ALIASES.items():
        raw_key = first_present(eval_np, aliases)
        if raw_key is None:
            continue
        scalar = _scalar_from_array(eval_np[raw_key])
        if scalar is not None:
            row[out_key] = scalar
    for c in [
        "energy_E_post",
        "update_counter",
        "funding_g",
        "funding_g_raw",
        "funding_alpha_eff",
        "depr_F_post",
        "episode_return_end_mean",
        "vartd_VarTD",
    ]:
        if c in trace_df.columns:
            row[f"final_{c}"] = float(pd.to_numeric(trace_df[c], errors="coerce").iloc[-1])
    return row


def aggregate_trace_df(trace_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in trace_df.columns if c not in {"mechanism_tag", "mechanism", "alpha", "seed", "update"}]
    agg = trace_df.groupby("update")[metric_cols].agg(["mean", q25, q75, "std"]).reset_index()
    out = flatten_multiindex_columns(agg)
    if "env_steps_mean" not in out.columns and "env_steps" in trace_df.columns:
        out["env_steps_mean"] = trace_df.groupby("update")["env_steps"].mean().to_numpy(dtype=float)
    return out


def save_band_plot(agg_df: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path, x_col: str = "env_steps_mean") -> None:
    mean_col = f"{metric}_mean"
    q25_col = f"{metric}_q25"
    q75_col = f"{metric}_q75"
    if mean_col not in agg_df.columns:
        return
    x_name = x_col if x_col in agg_df.columns else "update"
    x = agg_df[x_name].to_numpy(dtype=float)
    y = agg_df[mean_col].to_numpy(dtype=float)
    finite = np.isfinite(y)
    if finite.sum() == 0:
        return
    plt.figure(figsize=(8.2, 4.8))
    plt.plot(x[finite], y[finite], linewidth=2.0, label="mean")
    if q25_col in agg_df.columns and q75_col in agg_df.columns:
        lo = agg_df[q25_col].to_numpy(dtype=float)
        hi = agg_df[q75_col].to_numpy(dtype=float)
        ok = np.isfinite(lo) & np.isfinite(hi) & finite
        if ok.sum() > 0:
            plt.fill_between(x[ok], lo[ok], hi[ok], alpha=0.25, label="q25-q75")
    plt.xlabel("cumulative environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_one_alpha(mod: Any, *, alpha: float, mechanism: MechanismSpec, seeds: Sequence[int], out_dir: Path, total_timesteps: int, force_rerun: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import jax

    ensure_dir(out_dir)
    cfg = build_config(alpha=float(alpha), mechanism=mechanism, total_timesteps=int(total_timesteps))
    train_fn = mod.make_train(cfg)
    compiled_train = jax.jit(train_fn)

    endpoint_rows: List[Dict[str, Any]] = []
    all_traces: List[pd.DataFrame] = []

    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed:02d}"
        ensure_dir(seed_dir)
        trace_path = seed_dir / "trace_per_update.csv"
        eval_path = seed_dir / "eval_summary.json"
        if trace_path.exists() and eval_path.exists() and not force_rerun:
            trace_df = coerce_numeric(pd.read_csv(trace_path))
            eval_row = json.loads(eval_path.read_text(encoding="utf-8"))
            all_traces.append(trace_df)
            endpoint_rows.append(eval_row)
            continue

        t0 = time.perf_counter()
        out = compiled_train(master_seed_key(int(seed)))
        metrics_device = out["metrics"]
        eval_device = out["eval"]
        block_until_ready(metrics_device)
        block_until_ready(eval_device)
        runtime_sec = time.perf_counter() - t0

        metrics_np = to_numpy_tree(metrics_device)
        eval_np = to_numpy_tree(eval_device)
        trace_df = extract_trace_df(metrics_np, alpha=float(alpha), mechanism=mechanism, seed=int(seed))
        eval_row = extract_eval_row(eval_np, trace_df, alpha=float(alpha), mechanism=mechanism, seed=int(seed), runtime_sec=float(runtime_sec))

        trace_df.to_csv(trace_path, index=False)
        eval_path.write_text(json.dumps(eval_row, indent=2, sort_keys=True), encoding="utf-8")

        all_traces.append(trace_df)
        endpoint_rows.append(eval_row)
        del out, metrics_device, eval_device, metrics_np, eval_np, trace_df
        gc.collect()

    endpoint_df = coerce_numeric(pd.DataFrame(endpoint_rows))
    endpoint_df.to_csv(out_dir / "seed_endpoint_summary.csv", index=False)

    trace_long = pd.concat(all_traces, ignore_index=True)
    trace_long = coerce_numeric(trace_long)
    trace_long.to_csv(out_dir / "trace_per_seed_long.csv", index=False)

    trace_agg = aggregate_trace_df(trace_long)
    trace_agg.to_csv(out_dir / "trace_aggregate.csv", index=False)

    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    for metric, title, ylabel in PLOT_METRICS:
        save_band_plot(trace_agg, metric, f"alpha={alpha:.3f}: {title}", ylabel, plots_dir / f"{metric}.png")

    return endpoint_df, trace_agg


# -----------------------------------------------------------------------------
# Late-window observables and candidate scoring.
# -----------------------------------------------------------------------------
def compute_seed_late_metrics(trace_long: pd.DataFrame, late_frac: float) -> pd.DataFrame:
    required = {"seed", "update", "update_u"}
    if not required.issubset(trace_long.columns):
        raise ValueError(f"trace_per_seed_long.csv missing required columns {required}")

    rows: List[Dict[str, Any]] = []
    for seed, sdf in trace_long.groupby("seed"):
        sdf = sdf.sort_values("update").copy()
        tail_n = max(1, int(math.ceil(len(sdf) * float(late_frac))))
        late = sdf.iloc[-tail_n:].copy()
        u = pd.to_numeric(late["update_u"], errors="coerce").to_numpy(dtype=float)
        u = u[np.isfinite(u)]
        late_count = float(np.sum(u)) if u.size > 0 else np.nan
        block_edges = np.linspace(0, tail_n, LATE_BLOCKS + 1, dtype=int)
        active_blocks = 0
        if u.size > 0:
            for b in range(LATE_BLOCKS):
                lo, hi = int(block_edges[b]), int(block_edges[b + 1])
                if hi > lo and float(np.sum(u[lo:hi])) > 0.0:
                    active_blocks += 1
        sustained = int(
            np.isfinite(late_count)
            and late_count >= float(MIN_LATE_UPDATES_FOR_ACTIVE)
            and active_blocks >= int(MIN_ACTIVE_BLOCKS)
        )
        any_active = int(np.isfinite(late_count) and late_count > 0.0)
        bursty_only = int(any_active == 1 and sustained == 0)
        row: Dict[str, Any] = {
            "seed": int(seed),
            "late_window_updates": int(tail_n),
            "rho_seed": float(np.mean(u)) if u.size > 0 else np.nan,
            "active_seed": any_active if u.size > 0 else np.nan,
            "late_update_count_seed": late_count,
            "active_blocks_seed": int(active_blocks),
            "sustained_active_seed": sustained if u.size > 0 else np.nan,
            "bursty_only_seed": bursty_only if u.size > 0 else np.nan,
        }
        for col in ["energy_E_post", "depr_F_post", "funding_g", "episode_return_end_mean", "update_counter"]:
            if col in late.columns:
                arr = pd.to_numeric(late[col], errors="coerce").to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                row[f"late_mean_{col}"] = float(np.mean(arr)) if arr.size > 0 else np.nan
                row[f"final_{col}"] = float(arr[-1]) if arr.size > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_alpha_metrics(seed_late_df: pd.DataFrame, endpoint_df: pd.DataFrame) -> Dict[str, Any]:
    rho = pd.to_numeric(seed_late_df["rho_seed"], errors="coerce").to_numpy(dtype=float)
    rho = rho[np.isfinite(rho)]
    active = pd.to_numeric(seed_late_df["active_seed"], errors="coerce").to_numpy(dtype=float)
    active = active[np.isfinite(active)]
    sustained = pd.to_numeric(seed_late_df["sustained_active_seed"], errors="coerce").to_numpy(dtype=float)
    sustained = sustained[np.isfinite(sustained)]
    bursty = pd.to_numeric(seed_late_df["bursty_only_seed"], errors="coerce").to_numpy(dtype=float)
    bursty = bursty[np.isfinite(bursty)]
    late_updates = pd.to_numeric(seed_late_df["late_update_count_seed"], errors="coerce").to_numpy(dtype=float)
    late_updates = late_updates[np.isfinite(late_updates)]
    active_blocks = pd.to_numeric(seed_late_df.get("active_blocks_seed", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    active_blocks = active_blocks[np.isfinite(active_blocks)]
    out: Dict[str, Any] = {
        "n_seeds": int(len(seed_late_df)),
        "late_window_updates": int(seed_late_df["late_window_updates"].iloc[0]) if not seed_late_df.empty else 0,
        "rho_mean": float(np.mean(rho)) if rho.size > 0 else np.nan,
        "rho_sd": float(np.std(rho, ddof=1)) if rho.size > 1 else 0.0,
        "p_active_mean": float(np.mean(active)) if active.size > 0 else np.nan,
        "p_active_sd": float(np.std(active, ddof=1)) if active.size > 1 else 0.0,
        "p_sustained_mean": float(np.mean(sustained)) if sustained.size > 0 else np.nan,
        "p_sustained_sd": float(np.std(sustained, ddof=1)) if sustained.size > 1 else 0.0,
        "p_bursty_mean": float(np.mean(bursty)) if bursty.size > 0 else np.nan,
        "p_bursty_sd": float(np.std(bursty, ddof=1)) if bursty.size > 1 else 0.0,
        "late_update_count_mean": float(np.mean(late_updates)) if late_updates.size > 0 else np.nan,
        "late_update_count_sd": float(np.std(late_updates, ddof=1)) if late_updates.size > 1 else 0.0,
        "active_blocks_mean": float(np.mean(active_blocks)) if active_blocks.size > 0 else np.nan,
        "active_blocks_sd": float(np.std(active_blocks, ddof=1)) if active_blocks.size > 1 else 0.0,
        # N=3 only: use a seed-fluctuation proxy rather than finite-size susceptibility language.
        "chi_rho": float(np.var(rho, ddof=1)) if rho.size > 1 else 0.0,
    }
    for col in [
        "eval_return_end_mean",
        "eval_CSR_post",
        "eval_dbar_post_mean",
        "final_update_counter",
        "final_energy_E_post",
        "final_depr_F_post",
        "final_vartd_VarTD",
    ]:
        if col in endpoint_df.columns:
            arr = pd.to_numeric(endpoint_df[col], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            out[f"{col}_mean"] = float(np.mean(arr)) if arr.size > 0 else np.nan
            out[f"{col}_sd"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return out


def monotonicity_violations(values: Sequence[float], tolerance: float) -> int:
    vals = np.asarray(values, dtype=float)
    violations = 0
    for i in range(len(vals) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i + 1] < vals[i] - float(tolerance):
            violations += 1
    return int(violations)


def estimate_transition_window(summary_df: pd.DataFrame) -> Tuple[float, float, float]:
    d = summary_df.sort_values("alpha").reset_index(drop=True)
    alphas = d["alpha"].to_numpy(dtype=float)
    p = pd.to_numeric(d["p_sustained_mean"], errors="coerce").to_numpy(dtype=float)
    if len(alphas) == 0:
        return float("nan"), float("nan"), float("nan")
    crossing_idx = None
    for i in range(1, len(alphas)):
        if np.isfinite(p[i - 1]) and np.isfinite(p[i]) and p[i - 1] < 0.5 <= p[i]:
            crossing_idx = i
            break
    if crossing_idx is not None:
        return float(alphas[crossing_idx - 1]), float(alphas[crossing_idx]), float((alphas[crossing_idx - 1] + alphas[crossing_idx]) / 2.0)
    diffs = np.abs(p - 0.5)
    if np.isfinite(diffs).any():
        j = int(np.nanargmin(diffs))
    else:
        j = len(alphas) // 2
    lo_idx = max(j - 1, 0)
    hi_idx = min(j + 1, len(alphas) - 1)
    return float(alphas[lo_idx]), float(alphas[hi_idx]), float(alphas[j])


def chi_peak_metrics(summary_df: pd.DataFrame) -> Dict[str, Any]:
    d = summary_df.sort_values("alpha").reset_index(drop=True)
    chi_values = pd.to_numeric(d.get("chi_rho", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    alphas = pd.to_numeric(d.get("alpha", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    out: Dict[str, Any] = {
        "chi_peak_interior": 0,
        "chi_peak_alpha": float("nan"),
        "chi_peak_value": float("nan"),
        "chi_peak_ratio": float("nan"),
    }
    finite_mask = np.isfinite(chi_values)
    if finite_mask.sum() == 0:
        return out
    valid_indices = np.flatnonzero(finite_mask)
    peak_valid_pos = int(np.nanargmax(chi_values[finite_mask]))
    peak_idx = int(valid_indices[peak_valid_pos])
    out["chi_peak_alpha"] = float(alphas[peak_idx]) if peak_idx < len(alphas) else float("nan")
    out["chi_peak_value"] = float(chi_values[peak_idx])
    out["chi_peak_interior"] = int(0 < peak_idx < len(chi_values) - 1)
    finite_chi = chi_values[finite_mask]
    if finite_chi.size >= 3:
        med = float(np.nanmedian(finite_chi))
        peak = float(np.nanmax(finite_chi))
        if med > 0.0:
            out["chi_peak_ratio"] = peak / med
        elif peak > 0.0:
            out["chi_peak_ratio"] = float("inf")
        else:
            out["chi_peak_ratio"] = 0.0
    return out


def candidate_passes_criteria(row: Mapping[str, Any], criteria: Mapping[str, Any]) -> bool:
    chi_ratio = float(row.get("chi_peak_ratio", np.nan))
    if not np.isfinite(chi_ratio):
        chi_ratio = -np.inf
    require_chi = bool(criteria.get("require_chi_peak", False))
    if require_chi:
        chi_ok = int(row.get("chi_peak_interior", 0)) == 1 and chi_ratio >= float(criteria.get("chi_peak_ratio_min", 0.0))
    else:
        chi_ok = True if not np.isfinite(chi_ratio) else chi_ratio >= float(criteria.get("chi_peak_ratio_min", 0.0))
    return bool(
        float(row.get("left_edge_p_sustained", np.nan)) <= float(criteria.get("left_p_max", np.inf))
        and float(row.get("right_edge_p_sustained", np.nan)) >= float(criteria.get("right_p_min", -np.inf))
        and float(row.get("left_edge_late_update_count_mean", np.nan)) <= float(criteria.get("left_late_updates_max", np.inf))
        and float(row.get("right_edge_late_update_count_mean", np.nan)) >= float(criteria.get("right_late_updates_min", -np.inf))
        and int(row.get("p_sustained_violations", 999)) <= int(criteria.get("max_p_viol", 999))
        and int(row.get("rho_violations", 999)) <= int(criteria.get("max_rho_viol", 999))
        and chi_ok
    )


def apply_selection_ladder(ranked_df: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    if ranked_df.empty:
        return None, pd.DataFrame(columns=["tier_index", "tier_name", "n_pass", "selected_condition"])
    ordered_all = pd.DataFrame(sorted(ranked_df.to_dict("records"), key=candidate_sort_key)).reset_index(drop=True)
    for criteria in SELECTION_LADDER:
        col = f"pass_{criteria['name']}"
        if col not in ordered_all.columns:
            ordered_all[col] = [int(candidate_passes_criteria(row, criteria)) for row in ordered_all.to_dict("records")]
    audit_rows: List[Dict[str, Any]] = []
    for tier_index, criteria in enumerate(SELECTION_LADDER):
        col = f"pass_{criteria['name']}"
        passed = ordered_all.loc[ordered_all[col].astype(int) == 1].copy()
        audit_rows.append({
            "tier_index": int(tier_index),
            "tier_name": str(criteria["name"]),
            "n_pass": int(len(passed)),
            "selected_condition": str(passed.iloc[0]["mechanism_tag"]) if not passed.empty else "",
        })
        if not passed.empty:
            selected = passed.iloc[0].to_dict()
            selected["selection_mode"] = "criteria_pass"
            selected["selection_tier_name"] = str(criteria["name"])
            selected["selection_tier_index"] = int(tier_index)
            selected["strict_candidate_pass"] = int(candidate_passes_criteria(selected, SELECTION_LADDER[0]))
            selected["selected_candidate_pass"] = 1
            return selected, pd.DataFrame(audit_rows)
    selected = ordered_all.iloc[0].to_dict()
    selected["selection_mode"] = "best_negative_result"
    selected["selection_tier_name"] = "fallback_best_negative_result"
    selected["selection_tier_index"] = int(len(SELECTION_LADDER))
    selected["strict_candidate_pass"] = int(candidate_passes_criteria(selected, SELECTION_LADDER[0]))
    selected["selected_candidate_pass"] = 0
    audit_rows.append({
        "tier_index": int(len(SELECTION_LADDER)),
        "tier_name": "fallback_best_negative_result",
        "n_pass": 1,
        "selected_condition": str(selected.get("mechanism_tag", "")),
    })
    return selected, pd.DataFrame(audit_rows)


def evaluate_candidate(summary_df: pd.DataFrame, mechanism: MechanismSpec) -> Dict[str, Any]:
    d = summary_df.sort_values("alpha").reset_index(drop=True)
    if d.empty:
        raise ValueError("Empty summary_df")
    n_seeds = int(d["n_seeds"].iloc[0]) if "n_seeds" in d.columns else 0
    W = int(d["late_window_updates"].iloc[0]) if "late_window_updates" in d.columns else 1
    p_tol = 1.0 / max(n_seeds, 1)
    rho_tol = 1.0 / max(W, 1)
    p_sustained_values = pd.to_numeric(d["p_sustained_mean"], errors="coerce").to_numpy(dtype=float)
    p_active_values = pd.to_numeric(d["p_active_mean"], errors="coerce").to_numpy(dtype=float)
    rho_values = pd.to_numeric(d["rho_mean"], errors="coerce").to_numpy(dtype=float)

    left_edge = d.iloc[0].to_dict()
    right_edge = d.iloc[-1].to_dict()
    alpha_lo, alpha_hi, alpha_guess = estimate_transition_window(d)
    p_viol = monotonicity_violations(p_sustained_values, tolerance=p_tol)
    rho_viol = monotonicity_violations(rho_values, tolerance=rho_tol)
    total_viol = p_viol + rho_viol
    p_sustained_span = float(np.nanmax(p_sustained_values) - np.nanmin(p_sustained_values)) if len(p_sustained_values) else np.nan
    p_active_span = float(np.nanmax(p_active_values) - np.nanmin(p_active_values)) if len(p_active_values) else np.nan
    rho_span = float(np.nanmax(rho_values) - np.nanmin(rho_values)) if len(rho_values) else np.nan
    chi_metrics = chi_peak_metrics(d)

    metrics = {
        "mechanism": mechanism.mechanism_name,
        "mechanism_tag": mechanism.tag(),
        "c_upd": float(mechanism.c_upd),
        "z_cut": float(mechanism.z_cut),
        "depr_lambda": float(mechanism.depr_lambda),
        "depr_nu": float(mechanism.depr_nu),
        "depr_psi": float(mechanism.depr_psi),
        "left_edge_alpha": float(left_edge["alpha"]),
        "right_edge_alpha": float(right_edge["alpha"]),
        "left_edge_p_active": float(left_edge.get("p_active_mean", np.nan)),
        "right_edge_p_active": float(right_edge.get("p_active_mean", np.nan)),
        "left_edge_p_sustained": float(left_edge.get("p_sustained_mean", np.nan)),
        "right_edge_p_sustained": float(right_edge.get("p_sustained_mean", np.nan)),
        "left_edge_p_bursty": float(left_edge.get("p_bursty_mean", np.nan)),
        "right_edge_p_bursty": float(right_edge.get("p_bursty_mean", np.nan)),
        "left_edge_rho": float(left_edge.get("rho_mean", np.nan)),
        "right_edge_rho": float(right_edge.get("rho_mean", np.nan)),
        "left_edge_active_blocks_mean": float(left_edge.get("active_blocks_mean", np.nan)),
        "right_edge_active_blocks_mean": float(right_edge.get("active_blocks_mean", np.nan)),
        "left_edge_late_update_count_mean": float(left_edge.get("late_update_count_mean", np.nan)),
        "right_edge_late_update_count_mean": float(right_edge.get("late_update_count_mean", np.nan)),
        "p_sustained_violations": int(p_viol),
        "rho_violations": int(rho_viol),
        "total_violations": int(total_viol),
        "p_sustained_span": p_sustained_span,
        "p_active_span": p_active_span,
        "rho_span": rho_span,
        "alpha_transition_lo": alpha_lo,
        "alpha_transition_hi": alpha_hi,
        "alpha_transition_guess": alpha_guess,
        "n_seeds": int(n_seeds),
        "late_window_updates": int(W),
        **chi_metrics,
    }
    metrics["candidate_pass"] = int(candidate_passes_criteria(metrics, SELECTION_LADDER[0]))
    metrics["candidate_pass_profile"] = STRICT_PROFILE_NAME
    metrics["strict_candidate_pass"] = int(metrics["candidate_pass"])
    return metrics


def candidate_sort_key(row: Mapping[str, Any]) -> Tuple:
    def benefit(key: str) -> float:
        val = float(row.get(key, np.nan))
        return val if np.isfinite(val) else -np.inf

    def penalty(key: str) -> float:
        val = float(row.get(key, np.nan))
        return val if np.isfinite(val) else np.inf

    chi_ratio = benefit("chi_peak_ratio")
    return (
        -benefit("p_sustained_span"),
        int(row.get("total_violations", 999)),
        -int(row.get("chi_peak_interior", 0)),
        -chi_ratio,
        -benefit("right_edge_late_update_count_mean"),
        -benefit("rho_span"),
        penalty("right_edge_p_bursty"),
        0 if str(row.get("mechanism", "")) == "hardzero_only" else 1,
        penalty("c_upd"),
        penalty("z_cut"),
        penalty("depr_lambda"),
        penalty("depr_psi"),
    )


def build_refined_alpha_grid(row: Mapping[str, Any]) -> Tuple:
    return (
        -int(row.get("candidate_pass", 0)),
        0 if str(row.get("mechanism", "")) == "hardzero_only" else 1,
        int(row.get("total_violations", 999)),
        -float(row.get("p_active_span", -np.inf)),
        -float(row.get("right_edge_late_update_count_mean", -np.inf)),
        -float(row.get("rho_span", -np.inf)),
        float(row.get("c_upd", np.inf)),
        float(row.get("z_cut", np.inf)),
        float(row.get("depr_lambda", np.inf)),
        float(row.get("depr_psi", np.inf)),
    )


def build_refined_alpha_grid(candidate_row: Mapping[str, Any], *, alpha_max: float = DISCOVER_ALPHA_MAX, step: float = REFINE_ALPHA_STEP, pad_left: float = REFINE_PAD_LEFT, pad_right: float = REFINE_PAD_RIGHT) -> List[float]:
    a_lo = float(candidate_row.get("alpha_transition_lo", 0.0))
    a_hi = float(candidate_row.get("alpha_transition_hi", a_lo))
    start = max(0.0, a_lo - float(pad_left))
    stop = min(float(alpha_max), a_hi + float(pad_right))
    if stop < start:
        stop = start
    n_steps = int(round((stop - start) / float(step)))
    grid = [round(start + i * float(step), 3) for i in range(n_steps + 1)]
    if round(stop, 3) not in grid:
        grid.append(round(stop, 3))
    grid = sorted(set(grid))
    return grid


# -----------------------------------------------------------------------------
# Sweep runners.
# -----------------------------------------------------------------------------
def run_alpha_sweep(repo_root: Path, *, mechanism: MechanismSpec, alpha_values: Sequence[float], total_timesteps: int, seed_start: int, seed_stop: int, late_frac: float, out_dir: Path, force_repatch: bool, force_rerun: bool) -> pd.DataFrame:
    repo_root = infer_repo_root(repo_root)
    derived_master = write_engineered_master(repo_root, force=bool(force_repatch))
    mod = load_module_from_path(f"n3_engineered_phase_{mechanism.tag()}", derived_master, repo_root)
    seeds = list(range(int(seed_start), int(seed_stop)))
    if not seeds:
        raise ValueError("Empty seed range.")
    ensure_dir(out_dir)

    summary_rows: List[Dict[str, Any]] = []
    for alpha in [float(a) for a in alpha_values]:
        alpha_dir = out_dir / f"alpha_{alpha:g}"
        endpoint_df, _ = run_one_alpha(
            mod,
            alpha=float(alpha),
            mechanism=mechanism,
            seeds=seeds,
            out_dir=alpha_dir,
            total_timesteps=int(total_timesteps),
            force_rerun=bool(force_rerun),
        )
        trace_long = coerce_numeric(pd.read_csv(alpha_dir / "trace_per_seed_long.csv"))
        seed_late_df = compute_seed_late_metrics(trace_long, late_frac=float(late_frac))
        seed_late_df["alpha"] = float(alpha)
        seed_late_df.to_csv(alpha_dir / "seed_late_activity_metrics.csv", index=False)
        phase_summary = aggregate_alpha_metrics(seed_late_df, endpoint_df)
        phase_summary.update(
            {
                "alpha": float(alpha),
                "mechanism": mechanism.mechanism_name,
                "mechanism_tag": mechanism.tag(),
                "c_upd": float(mechanism.c_upd),
                "z_cut": float(mechanism.z_cut),
                "depr_lambda": float(mechanism.depr_lambda),
                "depr_nu": float(mechanism.depr_nu),
                "depr_psi": float(mechanism.depr_psi),
                "total_timesteps": int(total_timesteps),
            }
        )
        summary_rows.append(phase_summary)

    summary_df = coerce_numeric(pd.DataFrame(summary_rows).sort_values("alpha"))
    summary_df.to_csv(out_dir / "alpha_sweep_phase_summary.csv", index=False)
    candidate = evaluate_candidate(summary_df, mechanism)
    Path(out_dir / "candidate_metrics.json").write_text(json.dumps(candidate, indent=2, sort_keys=True), encoding="utf-8")
    return summary_df


def run_discovery_pass(repo_root: Path, *, mechanisms: Sequence[MechanismSpec], alpha_values: Sequence[float], total_timesteps: int, seed_start: int, seed_stop: int, late_frac: float, out_dir: Path, force_repatch: bool, force_rerun: bool) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for mechanism in mechanisms:
        mech_dir = out_dir / mechanism.tag()
        run_alpha_sweep(
            repo_root,
            mechanism=mechanism,
            alpha_values=alpha_values,
            total_timesteps=total_timesteps,
            seed_start=seed_start,
            seed_stop=seed_stop,
            late_frac=late_frac,
            out_dir=mech_dir,
            force_repatch=force_repatch,
            force_rerun=force_rerun,
        )
        candidate = json.loads((mech_dir / "candidate_metrics.json").read_text(encoding="utf-8"))
        rows.append(candidate)
    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        return rank_df
    ordered_records = sorted(rank_df.to_dict("records"), key=candidate_sort_key)
    rank_df = pd.DataFrame(ordered_records)
    rank_df.to_csv(out_dir / "discovery_ranked.csv", index=False)
    return rank_df


# -----------------------------------------------------------------------------
# Reporting and graph package.
# -----------------------------------------------------------------------------
def write_discovery_report(out_dir: Path, ranked_df: pd.DataFrame, pass_name: str, selected: Optional[Mapping[str, Any]], selection_audit: Optional[pd.DataFrame] = None) -> None:
    lines = [
        "N=3 engineered phase-transition discovery report",
        f"script_version={SCRIPT_VERSION}",
        f"pass_name={pass_name}",
        "",
        "This is an explicit engineered extension, not the accepted clipped thesis core.",
        "Primary observables: rho (late-window mean of update/u) and sustained late activity.",
        "Any-late-update survival is retained only as a diagnostic.",
        "Return is secondary.",
        "",
    ]
    if ranked_df.empty:
        lines.append("No candidates were evaluated.")
    else:
        lines.append(f"n_candidates={len(ranked_df)}")
        strict_n = int(ranked_df["pass_strict"].astype(int).sum()) if "pass_strict" in ranked_df.columns else int(ranked_df.get("candidate_pass", pd.Series(dtype=int)).astype(int).sum())
        lines.append(f"n_strict_pass={strict_n}")
        if selected is None:
            lines.append("No candidate could be selected.")
        else:
            lines.append("Selected candidate")
            for k, v in selected.items():
                lines.append(f"{k}={v}")
            if int(selected.get("selected_candidate_pass", 0)) != 1:
                lines.append("")
                lines.append("Interpretation")
                lines.append("The selected candidate did not satisfy the strict pre-locked criteria.")
                lines.append("It was chosen through the progressive relaxation ladder as the strongest available negative-result candidate for locked_candidate and graph_locked.")
                lines.append("If the locked rerun still fails the strict criteria, report the final graph package as a negative result after a predeclared best-effort search, not as a confirmed phase transition.")
    if selection_audit is not None and not selection_audit.empty:
        lines.append("")
        lines.append("Selection ladder")
        for _, row in selection_audit.iterrows():
            lines.append(
                f"tier_index={int(row['tier_index'])} tier_name={row['tier_name']} n_pass={int(row['n_pass'])} selected_condition={row['selected_condition']}"
            )
    (out_dir / "DISCOVERY_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_locked_graphs(summary_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    d = summary_df.sort_values("alpha")
    xs = d["alpha"].to_numpy(dtype=float)
    plots = [
        ("rho_mean", r"Late-window mean activity $\rho$", "rho_vs_alpha.png"),
        ("p_sustained_mean", r"Sustained active-seed fraction $P_{sustained}$", "p_sustained_vs_alpha.png"),
        ("p_active_mean", r"Any-late-update seed fraction $P_{any}$ (diagnostic)", "p_active_any_vs_alpha.png"),
        ("p_bursty_mean", r"Bursty-only seed fraction $P_{bursty}$", "p_bursty_vs_alpha.png"),
        ("active_blocks_mean", r"Mean active late-window blocks", "active_blocks_vs_alpha.png"),
        ("chi_rho", r"Seed fluctuation proxy $\chi_\rho = Var_{seed}(\rho_{seed})$", "chi_rho_vs_alpha.png"),
        ("eval_return_end_mean_mean", "Mean final deterministic evaluation return", "mean_final_eval_return_vs_alpha.png"),
    ]
    for metric, ylabel, filename in plots:
        if metric not in d.columns:
            continue
        ys = pd.to_numeric(d[metric], errors="coerce").to_numpy(dtype=float)
        plt.figure(figsize=(8.2, 4.8))
        plt.plot(xs, ys, marker="o", linewidth=2.0)
        plt.xlabel("alpha")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs alpha")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=160)
        plt.close()


def save_late_update_boxplot(locked_dir: Path, summary_df: pd.DataFrame, out_dir: Path) -> None:
    labels: List[str] = []
    data: List[np.ndarray] = []
    for alpha in summary_df.sort_values("alpha")["alpha"].to_numpy(dtype=float):
        seed_path = locked_dir / f"alpha_{alpha:g}" / "seed_late_activity_metrics.csv"
        if not seed_path.exists():
            continue
        seed_df = coerce_numeric(pd.read_csv(seed_path))
        arr = pd.to_numeric(seed_df["late_update_count_seed"], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        labels.append(f"{alpha:.3f}")
        data.append(arr)
    if not data:
        return
    plt.figure(figsize=(max(8.0, 0.55 * len(data)), 5.0))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xlabel("alpha")
    plt.ylabel("Late-window executed updates per seed")
    plt.title("Per-seed late-update-count distributions")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "late_update_count_distributions.png", dpi=160)
    plt.close()


def choose_representative_alphas(summary_df: pd.DataFrame) -> List[Tuple[str, float]]:
    d = summary_df.sort_values("alpha").reset_index(drop=True)
    if d.empty:
        return []
    left_alpha = float(d.iloc[0]["alpha"])
    right_alpha = float(d.iloc[-1]["alpha"])
    p_vals = pd.to_numeric(d.get("p_sustained_mean", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(p_vals).any():
        mid_idx = int(np.nanargmin(np.abs(p_vals - 0.5)))
    else:
        mid_idx = len(d) // 2
    mid_alpha = float(d.iloc[mid_idx]["alpha"])
    reps = [("inactive_side", left_alpha), ("crossover", mid_alpha), ("active_side", right_alpha)]
    seen = set()
    out: List[Tuple[str, float]] = []
    for tag, alpha in reps:
        if round(alpha, 6) not in seen:
            seen.add(round(alpha, 6))
            out.append((tag, alpha))
    return out


def select_representative_seed(seed_late_df: pd.DataFrame) -> int:
    vals = pd.to_numeric(seed_late_df["late_update_count_seed"], errors="coerce")
    target = float(vals.median())
    idx = int(np.nanargmin(np.abs(vals.to_numpy(dtype=float) - target)))
    return int(seed_late_df.iloc[idx]["seed"])


def save_representative_trace_plots(locked_dir: Path, summary_df: pd.DataFrame, out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tag, alpha in choose_representative_alphas(summary_df):
        alpha_dir = locked_dir / f"alpha_{alpha:g}"
        seed_late_path = alpha_dir / "seed_late_activity_metrics.csv"
        if not seed_late_path.exists():
            continue
        seed_late_df = coerce_numeric(pd.read_csv(seed_late_path))
        rep_seed = select_representative_seed(seed_late_df)
        trace_path = alpha_dir / f"seed_{rep_seed:02d}" / "trace_per_update.csv"
        if not trace_path.exists():
            continue
        trace_df = coerce_numeric(pd.read_csv(trace_path))
        x = pd.to_numeric(trace_df["update"], errors="coerce").to_numpy(dtype=float)

        fig, axes = plt.subplots(5, 1, figsize=(9.0, 11.0), sharex=True)
        metrics = [
            ("update_u", "u"),
            ("energy_E_post", "energy"),
            ("funding_g", "g*"),
            ("depr_F_post", "F"),
            ("episode_return_end_mean", "train return"),
        ]
        for ax, (col, ylabel) in zip(axes, metrics):
            if col in trace_df.columns:
                y = pd.to_numeric(trace_df[col], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() > 0:
                    ax.plot(x[ok], y[ok], linewidth=1.2)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("update index")
        fig.suptitle(f"Representative trace: {tag}, alpha={alpha:.3f}, seed={rep_seed:02d}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        out_path = out_dir / f"representative_trace_{tag}_alpha_{alpha:.3f}_seed_{rep_seed:02d}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        rows.append({"tag": tag, "alpha": float(alpha), "seed": int(rep_seed), "path": str(out_path.name)})
    return rows


def write_graph_report(out_dir: Path, candidate: Mapping[str, Any], reps: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "N=3 engineered phase-transition graph package",
        f"script_version={SCRIPT_VERSION}",
        "",
        "This package documents an engineered absorbing-state-like crossover search in late update activity for N=3.",
        "It must be described as an explicit engineered extension, not as latent evidence that the accepted clipped thesis core contained a true thermodynamic phase transition.",
        "",
        "Locked candidate",
    ]
    for k, v in candidate.items():
        lines.append(f"{k}={v}")
    lines.append("")
    lines.append("Graph interpretation")
    lines.append("Primary activity graph: p_sustained_vs_alpha.png")
    lines.append("Diagnostic any-burst graph: p_active_any_vs_alpha.png")
    if int(candidate.get("selected_candidate_pass", candidate.get("candidate_pass", 0))) != 1:
        lines.append("This locked candidate should be reported as a best-available negative-result candidate chosen by the predeclared relaxation ladder.")
        lines.append("If the locked rerun still fails the strict criteria, the graph package supports an honest negative result: no convincing engineered phase transition was found despite the final best-effort search.")
    else:
        lines.append("This locked candidate satisfied the strict discovery criteria and should be described as the strongest engineered crossover candidate found in the final search.")
    lines.append("")
    lines.append("Representative traces")
    for row in reps:
        lines.append(f"tag={row['tag']} alpha={row['alpha']} seed={row['seed']} file={row['path']}")
    (out_dir / "GRAPH_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI modes.
# -----------------------------------------------------------------------------
def mode_discover(args: argparse.Namespace) -> None:
    repo_root = infer_repo_root(args.repo_root)
    base_out = repo_root / "runs" / "n3_engineered_phase_transition" / "discover"
    ensure_dir(base_out)

    pass1_mechs = [MechanismSpec(c_upd=c, z_cut=z, depr_lambda=0.0, depr_nu=DEPR_NU, depr_psi=0.0) for z in args.z_cut_values for c in args.c_upd_values]
    pass1_dir = base_out / "pass1_hardzero_only"
    pass1_ranked = run_discovery_pass(
        repo_root,
        mechanisms=pass1_mechs,
        alpha_values=args.alpha_values,
        total_timesteps=int(args.total_timesteps),
        seed_start=int(args.seed_start),
        seed_stop=int(args.seed_stop),
        late_frac=float(args.late_frac),
        out_dir=pass1_dir,
        force_repatch=bool(args.force_repatch),
        force_rerun=bool(args.force_rerun),
    )

    final_ranked = pass1_ranked.copy()
    if not final_ranked.empty:
        for criteria in SELECTION_LADDER:
            col = f"pass_{criteria['name']}"
            final_ranked[col] = [int(candidate_passes_criteria(row, criteria)) for row in final_ranked.to_dict("records")]

    if not bool(args.skip_depr_pass):
        pass2_mechs = [
            MechanismSpec(c_upd=c, z_cut=z, depr_lambda=lam, depr_nu=float(args.depr_nu), depr_psi=psi)
            for z in args.z_cut_values
            for c in args.c_upd_values
            for lam in args.depr_lambdas
            for psi in args.depr_psis
        ]
        pass2_dir = base_out / "pass2_hardzero_plus_depr"
        pass2_ranked = run_discovery_pass(
            repo_root,
            mechanisms=pass2_mechs,
            alpha_values=args.alpha_values,
            total_timesteps=int(args.total_timesteps),
            seed_start=int(args.seed_start),
            seed_stop=int(args.seed_stop),
            late_frac=float(args.late_frac),
            out_dir=pass2_dir,
            force_repatch=bool(args.force_repatch),
            force_rerun=bool(args.force_rerun),
        )
        if not pass2_ranked.empty:
            for criteria in SELECTION_LADDER:
                col = f"pass_{criteria['name']}"
                pass2_ranked[col] = [int(candidate_passes_criteria(row, criteria)) for row in pass2_ranked.to_dict("records")]
            final_ranked = pd.concat([final_ranked, pass2_ranked], ignore_index=True) if not final_ranked.empty else pass2_ranked.copy()

    selected: Optional[Dict[str, Any]] = None
    selection_audit = pd.DataFrame(columns=["tier_index", "tier_name", "n_pass", "selected_condition"])
    if not final_ranked.empty:
        final_ranked = pd.DataFrame(sorted(final_ranked.to_dict("records"), key=candidate_sort_key)).reset_index(drop=True)
        selected, selection_audit = apply_selection_ladder(final_ranked)
        final_ranked.to_csv(base_out / "discovery_ranked.csv", index=False)
        selection_audit.to_csv(base_out / "selection_ladder_audit.csv", index=False)

    if selected is not None:
        Path(base_out / "selected_candidate.json").write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    write_discovery_report(base_out, final_ranked, pass_name="auto", selected=selected, selection_audit=selection_audit)


def mode_lock_candidate(args: argparse.Namespace) -> None:
    repo_root = infer_repo_root(args.repo_root)
    discover_dir = repo_root / "runs" / "n3_engineered_phase_transition" / "discover"
    cand_path = Path(args.candidate_json) if args.candidate_json is not None else discover_dir / "selected_candidate.json"
    if not cand_path.exists():
        raise FileNotFoundError(f"No selected candidate JSON found at {cand_path}. Run discover first.")
    candidate = json.loads(cand_path.read_text(encoding="utf-8"))
    mechanism = MechanismSpec(
        c_upd=float(candidate["c_upd"]),
        z_cut=float(candidate["z_cut"]),
        depr_lambda=float(candidate.get("depr_lambda", 0.0)),
        depr_nu=float(candidate.get("depr_nu", DEPR_NU)),
        depr_psi=float(candidate.get("depr_psi", 0.0)),
    )
    refined_alphas = build_refined_alpha_grid(candidate, alpha_max=float(args.alpha_max), step=float(args.refine_alpha_step), pad_left=float(args.refine_pad_left), pad_right=float(args.refine_pad_right))
    locked_dir = repo_root / "runs" / "n3_engineered_phase_transition" / "locked_candidate" / mechanism.tag()
    summary_df = run_alpha_sweep(
        repo_root,
        mechanism=mechanism,
        alpha_values=refined_alphas,
        total_timesteps=int(args.total_timesteps),
        seed_start=int(args.seed_start),
        seed_stop=int(args.seed_stop),
        late_frac=float(args.late_frac),
        out_dir=locked_dir,
        force_repatch=bool(args.force_repatch),
        force_rerun=bool(args.force_rerun),
    )
    locked_candidate = evaluate_candidate(summary_df, mechanism)
    for criteria in SELECTION_LADDER:
        locked_candidate[f"pass_{criteria['name']}"] = int(candidate_passes_criteria(locked_candidate, criteria))
    locked_candidate["refined_alpha_values"] = refined_alphas
    if int(locked_candidate.get("candidate_pass", 0)) == 1:
        locked_candidate["selection_mode"] = "criteria_pass"
        locked_candidate["selection_tier_name"] = STRICT_PROFILE_NAME
        locked_candidate["selection_tier_index"] = 0
        locked_candidate["selected_candidate_pass"] = 1
    else:
        locked_candidate["selection_mode"] = str(candidate.get("selection_mode", "best_negative_result")) if int(candidate.get("selected_candidate_pass", 0)) == 0 else "best_negative_result"
        locked_candidate["selection_tier_name"] = str(candidate.get("selection_tier_name", "fallback_best_negative_result")) if int(candidate.get("selected_candidate_pass", 0)) == 0 else "fallback_best_negative_result"
        locked_candidate["selection_tier_index"] = int(candidate.get("selection_tier_index", len(SELECTION_LADDER))) if int(candidate.get("selected_candidate_pass", 0)) == 0 else int(len(SELECTION_LADDER))
        locked_candidate["selected_candidate_pass"] = 0
    Path(locked_dir / "locked_candidate.json").write_text(json.dumps(locked_candidate, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "N=3 engineered phase-transition locked candidate",
        f"script_version={SCRIPT_VERSION}",
        "",
        "This is an explicit engineered extension.",
        "Do not use it to rewrite the accepted clipped thesis-core boundary claims.",
        "",
        "Locked candidate metrics",
    ]
    for k, v in locked_candidate.items():
        lines.append(f"{k}={v}")
    (locked_dir / "LOCKED_CANDIDATE_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mode_graph_locked(args: argparse.Namespace) -> None:
    repo_root = infer_repo_root(args.repo_root)
    if args.locked_dir is not None:
        locked_dir = Path(args.locked_dir).resolve()
    else:
        candidate_json = Path(args.candidate_json) if args.candidate_json is not None else repo_root / "runs" / "n3_engineered_phase_transition" / "discover" / "selected_candidate.json"
        if not candidate_json.exists():
            raise FileNotFoundError("Need a selected candidate JSON or --locked-dir.")
        candidate = json.loads(candidate_json.read_text(encoding="utf-8"))
        mechanism = MechanismSpec(
            c_upd=float(candidate["c_upd"]),
            z_cut=float(candidate["z_cut"]),
            depr_lambda=float(candidate.get("depr_lambda", 0.0)),
            depr_nu=float(candidate.get("depr_nu", DEPR_NU)),
            depr_psi=float(candidate.get("depr_psi", 0.0)),
        )
        locked_dir = repo_root / "runs" / "n3_engineered_phase_transition" / "locked_candidate" / mechanism.tag()
    summary_path = locked_dir / "alpha_sweep_phase_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing locked alpha sweep summary: {summary_path}")
    candidate_path = locked_dir / "locked_candidate.json"
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing locked candidate JSON: {candidate_path}")
    summary_df = coerce_numeric(pd.read_csv(summary_path))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    out_dir = locked_dir / "graph_package"
    ensure_dir(out_dir)
    save_locked_graphs(summary_df, out_dir)
    save_late_update_boxplot(locked_dir, summary_df, out_dir)
    reps = save_representative_trace_plots(locked_dir, summary_df, out_dir)
    write_graph_report(out_dir, candidate, reps)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rigorous N=3 engineered phase-transition script (hard-zero funding plus optional inactivity-driven depreciation).")
    sub = p.add_subparsers(dest="mode", required=True)

    p_dis = sub.add_parser("discover", help="Run disciplined discovery for an engineered N=3 absorbing-state-like crossover, with automatic relaxed-candidate fallback for negative-result reporting if no strict pass is found.")
    p_dis.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    p_dis.add_argument("--alpha-values", type=float, nargs="+", default=DISCOVER_ALPHA_VALUES)
    p_dis.add_argument("--c-upd-values", type=float, nargs="+", default=DISCOVER_C_UPD_VALUES)
    p_dis.add_argument("--z-cut-values", type=float, nargs="+", default=DISCOVER_Z_CUT_VALUES)
    p_dis.add_argument("--depr-lambdas", type=float, nargs="+", default=DEPR_LAMBDAS)
    p_dis.add_argument("--depr-psis", type=float, nargs="+", default=DEPR_PSIS)
    p_dis.add_argument("--depr-nu", type=float, default=DEPR_NU)
    p_dis.add_argument("--total-timesteps", type=int, default=DISCOVER_TIMESTEPS)
    p_dis.add_argument("--seed-start", type=int, default=DISCOVER_SEED_START)
    p_dis.add_argument("--seed-stop", type=int, default=DISCOVER_SEED_STOP)
    p_dis.add_argument("--late-frac", type=float, default=LATE_FRAC)
    p_dis.add_argument("--skip-depr-pass", action="store_true")
    p_dis.add_argument("--force-repatch", action="store_true")
    p_dis.add_argument("--force-rerun", action="store_true")

    p_lock = sub.add_parser("lock_candidate", help="Rerun the selected candidate with a refined alpha grid, longer horizon, and more seeds. If discovery selected a relaxed-best negative-result candidate, this mode preserves that provenance.")
    p_lock.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    p_lock.add_argument("--candidate-json", type=Path, default=None)
    p_lock.add_argument("--total-timesteps", type=int, default=LOCKED_TIMESTEPS)
    p_lock.add_argument("--seed-start", type=int, default=LOCK_SEED_START)
    p_lock.add_argument("--seed-stop", type=int, default=LOCK_SEED_STOP)
    p_lock.add_argument("--late-frac", type=float, default=LATE_FRAC)
    p_lock.add_argument("--alpha-max", type=float, default=DISCOVER_ALPHA_MAX)
    p_lock.add_argument("--refine-alpha-step", type=float, default=REFINE_ALPHA_STEP)
    p_lock.add_argument("--refine-pad-left", type=float, default=REFINE_PAD_LEFT)
    p_lock.add_argument("--refine-pad-right", type=float, default=REFINE_PAD_RIGHT)
    p_lock.add_argument("--force-repatch", action="store_true")
    p_lock.add_argument("--force-rerun", action="store_true")

    p_graph = sub.add_parser("graph_locked", help="Produce the final N=3 engineered graph package from the locked candidate run, including honest negative-result framing if strict criteria were not met.")
    p_graph.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    p_graph.add_argument("--candidate-json", type=Path, default=None)
    p_graph.add_argument("--locked-dir", type=Path, default=None)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "discover":
        mode_discover(args)
    elif args.mode == "lock_candidate":
        mode_lock_candidate(args)
    elif args.mode == "graph_locked":
        mode_graph_locked(args)
    else:
        raise SystemExit(f"Unsupported mode {args.mode!r}")


if __name__ == "__main__":
    main()
