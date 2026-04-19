#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
import types
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Locked project constants
# -----------------------------------------------------------------------------
R0 = -27.256548
D0 = 0.733348
R_REF_SHORT = -22.315781
R_THRESH_SHORT = -24.786164
R_REF_FULL = -24.779596
R_THRESH_FULL = -26.018072
EPS_OCC = 0.2
M_EVAL = 100
SHORT_TIMESTEPS = 400_000
FULL_TIMESTEPS = 1_000_000
LONG_TIMESTEPS = 4_000_000
NUM_ENVS = 25
NUM_STEPS = 128
SEED_START = 30
SEED_STOP_EXCLUSIVE = 62
BETA = float(np.log(10.0))
Z_MAX = 10.0
EPS_NORM = 1e-8
E0 = 1.0
E_MAX = 1.0
ENV_NAME = "MPE_simple_spread_v3"
ENV_KWARGS = {
    "num_agents": 3,
    "num_landmarks": 3,
    "local_ratio": 0.5,
    "max_steps": 25,
    "action_type": "Discrete",
}

DEFAULT_SUPPORT_MODE = "frontload"
DEFAULT_SUPPORT_TOTAL = 0.60
DEFAULT_SUPPORT_START_UPDATE = 0
DEFAULT_SUPPORT_WINDOW_FRAC = 0.10
DEFAULT_BORROW_LIMIT = 0.60
DEFAULT_BORROW_INTEREST = 0.02
DEFAULT_CHECKPOINT_EVAL_EVERY = 20

TRACE_ALIASES: Dict[str, List[str]] = {
    "update": ["update"],
    "update_u": ["update/u"],
    "update_counter": ["update/update_counter"],
    "opt_step": ["update/opt_step"],
    "energy_E_pre": ["energy/E_pre"],
    "energy_E_raw": ["energy/E_raw"],
    "energy_E_post": ["energy/E_post"],
    "funding_z": ["funding/z"],
    "funding_g": ["funding/g"],
    "funding_income": ["funding/income"],
    "support_s": ["support/s"],
    "debt_charge": ["debt/charge"],
    "debt_level_pre": ["debt/level_pre"],
    "debt_level_post": ["debt/level_post"],
    "episode_return_end_mean": ["episode/return_end_mean"],
    "episode_len_end_mean": ["episode/len_end_mean"],
    "terminal_min_dists_end_mean": ["terminal/min_dists_end_mean"],
    "terminal_dbar_end": ["terminal/dbar_end"],
    "terminal_occupied_end_mean": ["terminal/occupied_end_mean"],
    "terminal_CSR_train": ["terminal/CSR_train"],
    "terminal_done_count_end": ["terminal/done_count_end"],
    "vartd_VarTD": ["vartd/VarTD"],
    "actor_loss": ["actor_loss"],
    "critic_loss": ["critic_loss", "value_loss"],
    "entropy": ["entropy"],
    "total_loss": ["total_loss"],
    "ratio0": ["ratio0"],
    "eval_return_end_mean_cp": ["eval/return_end_mean_cp"],
    "eval_csr_cp": ["eval/csr_cp"],
    "eval_dbar_end_cp": ["eval/dbar_end_cp"],
    "eval_checkpoint_flag": ["eval/checkpoint_flag"],
}

EVAL_ALIASES: Dict[str, List[str]] = {
    "eval_done_count": ["done_count"],
    "eval_return_end_mean": ["return_end_mean"],
    "eval_len_end_mean": ["len_end_mean"],
    "eval_CSR_post": ["csr_post", "CSR_post", "csr"],
    "eval_CSR_pre": ["csr_pre"],
    "eval_dbar_post_mean": ["dbar_post", "dbar_post_mean", "dbar_end"],
    "eval_dbar_pre_mean": ["dbar_pre", "dbar_pre_mean"],
    "eval_occupied_post_mean": ["occupied_post_mean", "occupied_end_mean"],
    "eval_occupied_pre_mean": ["occupied_pre_mean"],
    "eval_max_tilde_post_mean": ["max_tilde_post_mean", "max_tilde_end_mean"],
    "eval_max_tilde_pre_mean": ["max_tilde_pre_mean"],
    "eval_episodes_per_env": ["episodes_per_env"],
    "eval_t_eval": ["t_eval"],
    "eval_done_per_env_min": ["done_per_env_min"],
    "eval_done_per_env_max": ["done_per_env_max"],
}

PLOT_SPECS = [
    ("energy_E_post", "Post-update energy", "energy"),
    ("debt_level_post", "Debt level", "debt"),
    ("support_s", "Support injection s_k", "support"),
    ("update_counter", "Cumulative executed updates", "executed updates"),
    ("funding_g", "Funding signal g_k", "g"),
    ("episode_return_end_mean", "End-of-episode training return", "return"),
    ("vartd_VarTD", "VarTD", "VarTD"),
    ("eval_return_end_mean_cp", "Checkpoint eval return", "eval return"),
    ("update_u", "Executed-update indicator u_k", "u"),
]

PHASE_TITLES = {
    "audit_existing": "Phase E0A archival audit",
    "reproduce_baseline": "Phase E0B baseline reproduction",
    "baseline_4m": "Phase E0C 4M baseline robustness",
    "rescue_short": "Phase E1 short-horizon rescue discovery",
    "extension_full": "Phase E2 full-horizon and 4M intervention tests",
}


STAGEE_SCRIPT_VERSION = "stageE_clipped_robustness_extension_rigorous_v5_reproguard"
SEED_KEY_SCHEME = "hydra_single_seed_split"
BASELINE_RETURN_TOL = 0.35
BASELINE_LEARN_PROB_TOL = 0.10
EXACT_MATCH_ATOL = 1e-12
CHECKPOINT_FLAT_TOL = 0.25
CHECKPOINT_DECLINE_TOL = 0.50
PROVENANCE_PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "matplotlib",
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "distrax",
    "chex",
]


# -----------------------------------------------------------------------------
# Stage-E master patching
# -----------------------------------------------------------------------------
HELPER_INSERT = '''


def support_from_schedule(
    k: jnp.ndarray,
    num_updates: int,
    total_support: jnp.ndarray,
    mode: str,
    start_update: int,
    window_updates: int,
) -> jnp.ndarray:
    """Deterministic exogenous support-injection schedule."""
    total_support = jnp.asarray(total_support, dtype=jnp.float32)
    if mode == "none" or float(total_support) == 0.0:
        return jnp.asarray(0.0, dtype=jnp.float32)

    k_i = k.astype(jnp.int32)
    start_i = jnp.asarray(start_update, dtype=jnp.int32)
    window_i = jnp.asarray(max(int(window_updates), 1), dtype=jnp.int32)

    if mode == "pulse":
        return jnp.where(
            jnp.equal(k_i, start_i),
            total_support,
            jnp.asarray(0.0, dtype=jnp.float32),
        ).astype(jnp.float32)

    if mode == "frontload":
        per = total_support / jnp.asarray(max(int(window_updates), 1), dtype=jnp.float32)
        active = jnp.logical_and(k_i >= start_i, k_i < start_i + window_i)
        return jnp.where(active, per, jnp.asarray(0.0, dtype=jnp.float32)).astype(jnp.float32)

    if mode == "drip":
        remaining = max(int(num_updates) - int(start_update), 1)
        per = total_support / jnp.asarray(remaining, dtype=jnp.float32)
        active = k_i >= start_i
        return jnp.where(active, per, jnp.asarray(0.0, dtype=jnp.float32)).astype(jnp.float32)

    raise ValueError(f"Unsupported SUPPORT_MODE={mode!r}")
'''


def patch_master_source(src: str) -> str:
    marker = "return z_k.astype(jnp.float32), g_k.astype(jnp.float32)\n\n\ndef vartd_path_a("
    if marker not in src:
        raise RuntimeError("Could not find funding_from_return marker in accepted master script.")
    src = src.replace(
        marker,
        "return z_k.astype(jnp.float32), g_k.astype(jnp.float32)\n" + HELPER_INSERT + "\n\ndef vartd_path_a(",
        1,
    )

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
    config.setdefault("BORROW_LIMIT", 0.0)
    config.setdefault("BORROW_INTEREST", 0.0)
    config.setdefault("SUPPORT_MODE", "none")
    config.setdefault("SUPPORT_TOTAL", 0.0)
    config.setdefault("SUPPORT_START_UPDATE", 0)
    config.setdefault("SUPPORT_WINDOW_UPDATES", 1)
    config.setdefault("CHECKPOINT_EVAL_EVERY", 0)
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
        raise RuntimeError("Could not find default-config block in accepted master script.")
    src = src.replace(old_defaults, new_defaults, 1)

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
    borrow_limit = jnp.asarray(config["BORROW_LIMIT"], dtype=jnp.float32)
    borrow_interest = jnp.asarray(config["BORROW_INTEREST"], dtype=jnp.float32)
    support_mode = str(config["SUPPORT_MODE"])
    support_total = jnp.asarray(config["SUPPORT_TOTAL"], dtype=jnp.float32)
    support_start_update = int(config["SUPPORT_START_UPDATE"])
    support_window_updates = int(config["SUPPORT_WINDOW_UPDATES"])
    checkpoint_eval_every = int(config.get("CHECKPOINT_EVAL_EVERY", 0))
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
        raise RuntimeError("Could not find derived-config block in accepted master script.")
    src = src.replace(old_derived, new_derived, 1)

    old_eval_setup = '''    if do_eval:
        if m_eval % int(config["NUM_ENVS"]) != 0:
            raise ValueError("LOCKED: M_EVAL must be divisible by NUM_ENVS for fixed-shape evaluation.")
        episodes_per_env = m_eval // int(config["NUM_ENVS"])
        t_eval = episodes_per_env * episode_len
    else:
        episodes_per_env = 0
        t_eval = 0
'''
    new_eval_setup = '''    if do_eval:
        if m_eval % int(config["NUM_ENVS"]) != 0:
            raise ValueError("LOCKED: M_EVAL must be divisible by NUM_ENVS for fixed-shape evaluation.")
        episodes_per_env = m_eval // int(config["NUM_ENVS"])
        t_eval = episodes_per_env * episode_len
    else:
        episodes_per_env = 0
        t_eval = 0

    do_checkpoint_eval = bool(do_eval and checkpoint_eval_every > 0)
'''
    if old_eval_setup not in src:
        raise RuntimeError("Could not find eval setup block in accepted master script.")
    src = src.replace(old_eval_setup, new_eval_setup, 1)

    old_gate = '''            income = alpha * g_k
            e_pre = energy
            u_bool = e_pre >= c_upd
            u_f = u_bool.astype(jnp.float32)
'''
    new_gate = '''            support_k = support_from_schedule(
                k=jnp.asarray(k, dtype=jnp.int32),
                num_updates=int(config["NUM_UPDATES"]),
                total_support=support_total,
                mode=support_mode,
                start_update=support_start_update,
                window_updates=support_window_updates,
            )
            income = alpha * g_k + support_k
            e_pre = energy
            debt_level_pre = jnp.maximum(-e_pre, 0.0)
            debt_charge = borrow_interest * debt_level_pre
            u_bool = (e_pre + borrow_limit) >= c_upd
            u_f = u_bool.astype(jnp.float32)
'''
    if old_gate not in src:
        raise RuntimeError("Could not find gate block in accepted master script.")
    src = src.replace(old_gate, new_gate, 1)

    old_energy_update = '            e_post = jnp.clip(e_pre - c_upd * u_f + income, 0.0, e_max).astype(jnp.float32)\n'
    new_energy_update = '''            e_raw = e_pre - c_upd * u_f + income - debt_charge
            e_post = jnp.clip(e_raw, -borrow_limit, e_max).astype(jnp.float32)
            debt_level_post = jnp.maximum(-e_post, 0.0).astype(jnp.float32)
'''
    if old_energy_update not in src:
        raise RuntimeError("Could not find energy update line in accepted master script.")
    src = src.replace(old_energy_update, new_energy_update, 1)

    old_metric_line = '                "funding/income": income.astype(jnp.float32),\n'
    new_metric_line = '''                "funding/income": income.astype(jnp.float32),
                "support/s": support_k.astype(jnp.float32),
                "debt/charge": debt_charge.astype(jnp.float32),
                "debt/level_pre": debt_level_pre.astype(jnp.float32),
                "debt/level_post": debt_level_post.astype(jnp.float32),
                "energy/E_raw": e_raw.astype(jnp.float32),
'''
    if old_metric_line not in src:
        raise RuntimeError("Could not find funding/income metric line in accepted master script.")
    src = src.replace(old_metric_line, new_metric_line, 1)

    old_before_runner = '''            runner_state_next = (
'''
    new_before_runner = '''            if do_checkpoint_eval:
                is_cp = jnp.equal(
                    (k + jnp.asarray(1, dtype=jnp.int32)) % jnp.asarray(checkpoint_eval_every, dtype=jnp.int32),
                    0,
                )

                def _eval_branch(_):
                    ev = _run_eval(train_state_next)
                    return {
                        "eval/return_end_mean_cp": ev["return_end_mean"].astype(jnp.float32),
                        "eval/csr_cp": ev["csr"].astype(jnp.float32),
                        "eval/dbar_end_cp": ev["dbar_end"].astype(jnp.float32),
                        "eval/checkpoint_flag": jnp.asarray(1, dtype=jnp.int32),
                    }

                def _skip_eval_branch(_):
                    nan32 = jnp.asarray(jnp.nan, dtype=jnp.float32)
                    return {
                        "eval/return_end_mean_cp": nan32,
                        "eval/csr_cp": nan32,
                        "eval/dbar_end_cp": nan32,
                        "eval/checkpoint_flag": jnp.asarray(0, dtype=jnp.int32),
                    }

                cp_metrics = jax.lax.cond(is_cp, _eval_branch, _skip_eval_branch, operand=None)
                metric = {**metric, **cp_metrics}
            else:
                metric = {
                    **metric,
                    "eval/return_end_mean_cp": jnp.asarray(jnp.nan, dtype=jnp.float32),
                    "eval/csr_cp": jnp.asarray(jnp.nan, dtype=jnp.float32),
                    "eval/dbar_end_cp": jnp.asarray(jnp.nan, dtype=jnp.float32),
                    "eval/checkpoint_flag": jnp.asarray(0, dtype=jnp.int32),
                }

            runner_state_next = (
'''
    if old_before_runner not in src:
        raise RuntimeError("Could not find runner_state_next marker in accepted master script.")
    src = src.replace(old_before_runner, new_before_runner, 1)

    return src


def write_stagee_master(repo_root: Path, force: bool = False) -> Path:
    accepted = find_repo_file(
        repo_root,
        preferred_relative_paths=[Path("baselines") / "IPPO" / "ippo_ff_mpe_gate5_energy_gated.py"],
        filename="ippo_ff_mpe_gate5_energy_gated.py",
        descriptor="accepted Stage-E baseline master",
        preferred_path_parts=["baselines", "IPPO"],
    )
    derived = accepted.parent / "ippo_ff_mpe_stagee_clipped_extension.py"
    src_text = accepted.read_text(encoding="utf-8")
    patched = patch_master_source(src_text)
    if derived.exists() and not force:
        existing = derived.read_text(encoding="utf-8")
        if existing == patched:
            return derived
    derived.write_text(patched, encoding="utf-8")
    return derived


# -----------------------------------------------------------------------------
# General helpers
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


def parse_kv_report(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    out.columns = ["_".join(str(x) for x in col if str(x) != "") for col in df.columns]
    return out


def maybe_to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    try:
        converted = pd.to_numeric(series)
    except Exception:
        return series
    return converted


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = maybe_to_numeric(out[c])
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


def _scalar_from_array(x: Any) -> Optional[float]:
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def canonical_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, default=_json_default)


def sha256_jsonable(obj: Any) -> str:
    return sha256_text(canonical_json_dumps(obj))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(canonical_json_dumps(obj) + "\n", encoding="utf-8")


def safe_rel(path: Optional[Path], root: Path) -> str:
    if path is None:
        return "none"
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def package_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in PROVENANCE_PACKAGE_NAMES:
        try:
            out[name] = importlib_metadata.version(name)
        except Exception:
            out[name] = "not-installed"
    return out


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
        self._closed = False
        self.encoding = next((getattr(s, "encoding", None) for s in streams if getattr(s, "encoding", None) is not None), None)
        self.errors = next((getattr(s, "errors", None) for s in streams if getattr(s, "errors", None) is not None), None)

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                continue

    def close(self) -> None:
        # Logging frameworks may call close() on the current stderr/stdout stream
        # at interpreter shutdown. We deliberately do not close the wrapped base
        # streams here, because that could close the real console streams.
        self.flush()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def tee_stdout_stderr(log_path: Path):
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as log_f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeStream(old_stdout, log_f)
        sys.stderr = TeeStream(old_stderr, log_f)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def collect_session_provenance(repo_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    prov = {
        "stagee_script_version": STAGEE_SCRIPT_VERSION,
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd().resolve()),
        "repo_root": str(repo_root.resolve()),
        "script_path": str(current_script_path()),
        "script_sha256": current_script_sha256(),
        "argv": list(sys.argv),
        "args": jsonable_args(args),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "seed_key_scheme": SEED_KEY_SCHEME,
    }
    return prov


def build_runtime_identity(
    repo_root: Path,
    accepted_master: Path,
    generated_master: Path,
    spec: "RunSpec",
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    identity = {
        "stagee_script_version": STAGEE_SCRIPT_VERSION,
        "script_path": str(current_script_path()),
        "script_sha256": current_script_sha256(),
        "repo_root": str(repo_root.resolve()),
        "accepted_master_path": str(accepted_master.resolve()),
        "accepted_master_sha256": sha256_file(accepted_master),
        "generated_master_path": str(generated_master.resolve()),
        "generated_master_sha256": sha256_file(generated_master),
        "spec": asdict(spec),
        "config": dict(config),
        "seed_key_scheme": SEED_KEY_SCHEME,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
    }
    identity["runtime_identity_hash"] = sha256_jsonable(identity)
    return identity


def try_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def finite_or_nan(value: Any) -> float:
    x = try_float(value)
    return x if np.isfinite(x) else float("nan")


def measured_summary_value(summary_row: Mapping[str, Any], key: str) -> float:
    return finite_or_nan(summary_row.get(f"{key}_mean", float("nan")))


def maybe_add_pip_freeze(out_dir: Path) -> None:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.stdout:
            (out_dir / "pip_freeze.txt").write_text(proc.stdout, encoding="utf-8")
    except Exception:
        pass


def current_script_path() -> Path:
    return Path(__file__).resolve()


def current_script_sha256() -> str:
    return sha256_file(current_script_path())


def print_startup_banner() -> None:
    print(f"[Stage E] stagee_script_version={STAGEE_SCRIPT_VERSION}")
    print(f"[Stage E] script_path={current_script_path()}")
    print(f"[Stage E] script_sha256={current_script_sha256()}")
    print(f"[Stage E] seed_key_scheme={SEED_KEY_SCHEME}")


def build_reproduction_gate_signature(
    repo_root: Path,
    args: argparse.Namespace,
    seeds: Sequence[int],
    accepted_master: Path,
    generated_master: Path,
) -> Dict[str, Any]:
    return {
        "stagee_script_version": STAGEE_SCRIPT_VERSION,
        "script_path": str(current_script_path()),
        "script_sha256": current_script_sha256(),
        "seed_key_scheme": SEED_KEY_SCHEME,
        "accepted_master_sha256": sha256_file(accepted_master),
        "generated_master_sha256": sha256_file(generated_master),
        "seed_start": int(min(seeds)),
        "seed_stop_exclusive": int(max(seeds) + 1),
        "seed_count": int(len(seeds)),
        "checkpoint_eval_every": int(args.checkpoint_eval_every),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
    }


def reproduction_gate_is_current(
    summary_path: Path,
    current_signature: Mapping[str, Any],
) -> Tuple[bool, str]:
    if not summary_path.exists():
        return False, "missing_summary"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False, "unreadable_summary"
    saved_signature = summary.get("reproduction_gate_signature")
    if not isinstance(saved_signature, dict):
        return False, "missing_gate_signature"
    if sha256_jsonable(saved_signature) != sha256_jsonable(dict(current_signature)):
        return False, "gate_signature_mismatch"
    return True, "current"


# -----------------------------------------------------------------------------
# Repo / evidence loading
# -----------------------------------------------------------------------------
EXCLUDED_SEARCH_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}
REPO_ROOT_MARKERS = [
    Path("baselines") / "IPPO" / "ippo_ff_mpe_gate5_energy_gated.py",
    Path("stageB_completion_bridge.py"),
    Path("stageC_subset_refine.py"),
    Path("stageD_representative_mechanistic_traces.py"),
]


def need(path: Optional[Path], descriptor: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"Required file not found: {descriptor}")
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {descriptor} (expected {path})")
    return path.resolve()


def _normalized_dir(path: Path) -> Path:
    path = path.resolve()
    return path if path.is_dir() else path.parent


def _candidate_roots(*starts: Optional[Path]) -> List[Path]:
    seen: set[Path] = set()
    out: List[Path] = []
    for start in starts:
        if start is None:
            continue
        try:
            base = _normalized_dir(start)
        except FileNotFoundError:
            continue
        for cand in [base, *base.parents]:
            if cand not in seen:
                out.append(cand)
                seen.add(cand)
    return out


def looks_like_repo_root(path: Path) -> bool:
    try:
        path = path.resolve()
    except FileNotFoundError:
        return False
    return all((path / marker).exists() for marker in REPO_ROOT_MARKERS)


def infer_repo_root(explicit: Optional[Path] = None) -> Path:
    starts = _candidate_roots(explicit, Path.cwd(), Path(__file__).resolve().parent)
    for cand in starts:
        if looks_like_repo_root(cand):
            return cand
    search_roots = starts or [Path.cwd().resolve()]
    for base in search_roots:
        for candidate in [base, *base.parents]:
            try:
                for hit in candidate.rglob("ippo_ff_mpe_gate5_energy_gated.py"):
                    if any(part in EXCLUDED_SEARCH_PARTS for part in hit.parts):
                        continue
                    root = hit.parent.parent.parent.resolve()
                    if looks_like_repo_root(root):
                        return root
            except Exception:
                continue
    searched = ", ".join(str(p) for p in search_roots)
    raise FileNotFoundError(
        "Could not infer the repository root. Pass --repo-root to the directory that contains "
        "baselines/IPPO/ippo_ff_mpe_gate5_energy_gated.py and the Stage B/C/D scripts. "
        f"Search starts were: {searched}"
    )


def _path_preference_score(path: Path, preferred_path_parts: Optional[Sequence[str]] = None) -> Tuple[int, int, str]:
    parts = list(path.parts)
    matched = 0
    if preferred_path_parts:
        for part in preferred_path_parts:
            if part in parts:
                matched += 1
    return (-matched, len(parts), str(path))


def find_repo_file(
    repo_root: Path,
    *,
    preferred_relative_paths: Optional[Sequence[Path]] = None,
    filename: Optional[str] = None,
    descriptor: str,
    preferred_path_parts: Optional[Sequence[str]] = None,
    required: bool = True,
) -> Optional[Path]:
    repo_root = repo_root.resolve()
    preferred_relative_paths = list(preferred_relative_paths or [])
    for rel in preferred_relative_paths:
        candidate = (repo_root / rel).resolve()
        if candidate.exists():
            return candidate
    if filename is not None:
        matches: List[Path] = []
        for hit in repo_root.rglob(filename):
            if not hit.is_file():
                continue
            if any(part in EXCLUDED_SEARCH_PARTS for part in hit.parts):
                continue
            matches.append(hit.resolve())
        if matches:
            matches.sort(key=lambda p: _path_preference_score(p, preferred_path_parts))
            return matches[0]
    if required:
        tried = ", ".join(str(repo_root / rel) for rel in preferred_relative_paths) if preferred_relative_paths else "recursive search only"
        raise FileNotFoundError(f"Required file not found: {descriptor}. Tried {tried} under {repo_root}")
    return None


def load_repo_inputs(repo_root: Path) -> Dict[str, Optional[Path]]:
    repo_root = infer_repo_root(repo_root)
    return {
        "project_plan": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("project_v2_1.txt")],
            filename="project_v2_1.txt",
            descriptor="accepted clipped project plan",
            preferred_path_parts=[],
            required=False,
        ),
        "accepted_master": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("baselines") / "IPPO" / "ippo_ff_mpe_gate5_energy_gated.py"],
            filename="ippo_ff_mpe_gate5_energy_gated.py",
            descriptor="accepted Stage-E baseline master",
            preferred_path_parts=["baselines", "IPPO"],
        ),
        "stageb_script": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("stageB_completion_bridge.py")],
            filename="stageB_completion_bridge.py",
            descriptor="Stage B completion bridge script",
            preferred_path_parts=[],
        ),
        "stagec_script": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("stageC_subset_refine.py")],
            filename="stageC_subset_refine.py",
            descriptor="Stage C subset refine script",
            preferred_path_parts=[],
        ),
        "staged_script": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("stageD_representative_mechanistic_traces.py")],
            filename="stageD_representative_mechanistic_traces.py",
            descriptor="Stage D representative mechanistic traces script",
            preferred_path_parts=[],
        ),
        "stageb_report": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("runs") / "stageB_completion_bridge_reports" / "STAGE_B_COMPLETION_REPORT.txt"],
            filename="STAGE_B_COMPLETION_REPORT.txt",
            descriptor="Stage B completion report",
            preferred_path_parts=["runs", "stageB_completion_bridge_reports"],
        ),
        "stageb_by_point": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("runs") / "stageB_completion_bridge_reports" / "stageB_bridge_by_point.csv"],
            filename="stageB_bridge_by_point.csv",
            descriptor="Stage B by-point summary",
            preferred_path_parts=["runs", "stageB_completion_bridge_reports"],
        ),
        "stagec_report": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("runs") / "stageC_subset_refine_reports" / "STAGE_C_SUBSET_FULL_HORIZON_REPORT.txt"],
            filename="STAGE_C_SUBSET_FULL_HORIZON_REPORT.txt",
            descriptor="Stage C full-horizon report",
            preferred_path_parts=["runs", "stageC_subset_refine_reports"],
        ),
        "stagec_by_point": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("runs") / "stageC_subset_refine_reports" / "stageC_subset_refine_by_point.csv"],
            filename="stageC_subset_refine_by_point.csv",
            descriptor="Stage C by-point summary",
            preferred_path_parts=["runs", "stageC_subset_refine_reports"],
        ),
        "stagec_alpha_boundary": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("runs") / "stageC_subset_refine_reports" / "stageC_subset_refine_alpha_boundary.csv"],
            filename="stageC_subset_refine_alpha_boundary.csv",
            descriptor="Stage C alpha boundary summary",
            preferred_path_parts=["runs", "stageC_subset_refine_reports"],
        ),
        "staged_report": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("runs") / "stageD_representative_mechanistic_traces" / "STAGE_D_REPRESENTATIVE_MECHANISTIC_TRACES_REPORT.txt"],
            filename="STAGE_D_REPRESENTATIVE_MECHANISTIC_TRACES_REPORT.txt",
            descriptor="Stage D mechanistic trace report",
            preferred_path_parts=["runs", "stageD_representative_mechanistic_traces"],
        ),
        "staged_endpoints": find_repo_file(
            repo_root,
            preferred_relative_paths=[Path("runs") / "stageD_representative_mechanistic_traces" / "stageD_endpoint_summary_all_conditions.csv"],
            filename="stageD_endpoint_summary_all_conditions.csv",
            descriptor="Stage D endpoint summary",
            preferred_path_parts=["runs", "stageD_representative_mechanistic_traces"],
        ),
    }


# -----------------------------------------------------------------------------
# Authoritative reference tables
# -----------------------------------------------------------------------------
def load_authoritative_reference_tables(repo_root: Path) -> Dict[str, pd.DataFrame]:
    inputs = load_repo_inputs(repo_root)
    stageb = coerce_numeric(pd.read_csv(need(inputs["stageb_by_point"], "Stage B by-point summary")))
    stagec = coerce_numeric(pd.read_csv(need(inputs["stagec_by_point"], "Stage C by-point summary")))
    staged = coerce_numeric(pd.read_csv(need(inputs["staged_endpoints"], "Stage D endpoint summary")))
    return {"stageb": stageb, "stagec": stagec, "staged": staged}


def load_stage_d_seed_endpoint_tables(repo_root: Path) -> List[Dict[str, Any]]:
    inputs = load_repo_inputs(repo_root)
    stage_d_root = need(inputs["staged_report"], "Stage D mechanistic trace report").parent
    tables: List[Dict[str, Any]] = []
    for path in sorted(stage_d_root.rglob("seed_endpoint_summary.csv")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_SEARCH_PARTS for part in path.parts):
            continue
        try:
            df = coerce_numeric(pd.read_csv(path))
        except Exception:
            continue
        if df.empty:
            continue
        tables.append(
            {
                "path": path.resolve(),
                "condition_name": path.parent.name,
                "df": df,
            }
        )
    return tables


def authoritative_row_for_spec(spec: "RunSpec", refs: Mapping[str, pd.DataFrame]) -> Optional[pd.Series]:
    if spec.total_timesteps == SHORT_TIMESTEPS:
        df = refs["stageb"]
        mask = np.isclose(pd.to_numeric(df["alpha"], errors="coerce"), spec.alpha) & np.isclose(pd.to_numeric(df["c_upd"], errors="coerce"), spec.c_upd)
        rows = df.loc[mask]
        if not rows.empty:
            return rows.iloc[0]
    if spec.total_timesteps == FULL_TIMESTEPS:
        df = refs["stagec"]
        mask = np.isclose(pd.to_numeric(df["alpha"], errors="coerce"), spec.alpha) & np.isclose(pd.to_numeric(df["c_upd"], errors="coerce"), spec.c_upd)
        rows = df.loc[mask]
        if not rows.empty:
            return rows.iloc[0]
    return None


# -----------------------------------------------------------------------------
# Audit helpers (Phase E0A)
# -----------------------------------------------------------------------------
def compute_var_td_alignment_note() -> Dict[str, Any]:
    episode_len = 26
    lcm_steps = math.lcm(NUM_STEPS, episode_len)
    beat_updates = lcm_steps // NUM_STEPS
    return {
        "num_steps": NUM_STEPS,
        "episode_len": episode_len,
        "lcm": lcm_steps,
        "beat_updates": beat_updates,
    }


def write_audit_outputs(repo_root: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    inputs = load_repo_inputs(repo_root)
    stageb_kv = parse_kv_report(need(inputs["stageb_report"], "Stage B completion report"))
    stagec_kv = parse_kv_report(need(inputs["stagec_report"], "Stage C full-horizon report"))
    refs = load_authoritative_reference_tables(repo_root)
    stagec_df = refs["stagec"]
    staged_df = refs["staged"]

    stagec_fragile = stagec_df.loc[
        np.isclose(pd.to_numeric(stagec_df["alpha"], errors="coerce"), 0.0)
        & np.isclose(pd.to_numeric(stagec_df["c_upd"], errors="coerce"), 0.58)
    ].iloc[0]
    staged_fragile = staged_df.loc[staged_df["condition"] == "fragile_full"].iloc[0]

    mismatch = {
        "stagec_mean_eval_return_end_mean": float(stagec_fragile["mean_eval_return_end_mean"]),
        "staged_eval_return_end_mean": float(staged_fragile["eval_return_end_mean"]),
        "return_gap": float(staged_fragile["eval_return_end_mean"] - stagec_fragile["mean_eval_return_end_mean"]),
        "stagec_mean_update_fraction": float(stagec_fragile["mean_update_fraction"]),
        "staged_mean_update_fraction": float(staged_fragile["mean_update_fraction"]),
    }
    beat = compute_var_td_alignment_note()

    claims_rows = [
        {
            "claim_id": "C1",
            "claim": "Accepted project scope is the revised clipped-energy plan, not the broader old project.",
            "authority_tier": "core_definition",
            "evidence_file": str(inputs["project_plan"].relative_to(repo_root)) if inputs["project_plan"] is not None else "project_v2_1.txt not required at runtime",
            "evidence_locator": "accepted project scope",
            "note": "Strategic state is locked to the clipped path; runtime does not require external project-knowledge bundles.",
        },
        {
            "claim_id": "C2",
            "claim": "Stage B found one short-horizon in-domain bracket for alpha=0.0 between c_upd=0.50 and 0.60.",
            "authority_tier": "authoritative_boundary",
            "evidence_file": "runs/stageB_completion_bridge_reports/STAGE_B_COMPLETION_REPORT.txt",
            "evidence_locator": "alpha=0.0 classification=in_domain_bracket_ready",
            "note": "Bridge result only; not the final full-horizon map.",
        },
        {
            "claim_id": "C3",
            "claim": "Stage C completed 1088/1088 runs and found no interior in-domain full-horizon boundary.",
            "authority_tier": "authoritative_boundary",
            "evidence_file": "runs/stageC_subset_refine_reports/STAGE_C_SUBSET_FULL_HORIZON_REPORT.txt",
            "evidence_locator": "completed_runs=1088, in_domain_count=0, right_censored_count=11",
            "note": "Stage C is the source of truth for final boundary claims.",
        },
        {
            "claim_id": "C4",
            "claim": "Alpha=0.0 stayed above threshold through c_upd=0.58 at full horizon.",
            "authority_tier": "authoritative_boundary",
            "evidence_file": "runs/stageC_subset_refine_reports/STAGE_C_SUBSET_FULL_HORIZON_REPORT.txt",
            "evidence_locator": "alpha=0.0 boundary evidence",
            "note": "Right-censored above sampled refine band.",
        },
        {
            "claim_id": "C5",
            "claim": "Alpha=0.1 to 1.0 remained right-censored above c_upd=1.0 at full horizon.",
            "authority_tier": "authoritative_boundary",
            "evidence_file": "runs/stageC_subset_refine_reports/STAGE_C_SUBSET_FULL_HORIZON_REPORT.txt",
            "evidence_locator": "alpha=0.1..1.0 entries",
            "note": "Full-horizon map is right-censored in the physical domain.",
        },
        {
            "claim_id": "C6",
            "claim": "Stage D is mechanism-only evidence with enabled_full, fragile_full, and disabled_short representatives.",
            "authority_tier": "supportive_mechanism",
            "evidence_file": "runs/stageD_representative_mechanistic_traces/STAGE_D_REPRESENTATIVE_MECHANISTIC_TRACES_REPORT.txt",
            "evidence_locator": "Selected conditions",
            "note": "Useful for mechanism, not for boundary truth.",
        },
        {
            "claim_id": "C7",
            "claim": "Known audit caveat: the uploaded Stage D fragile_full endpoint summary does not align perfectly with Stage C.",
            "authority_tier": "audit_caveat",
            "evidence_file": "Stage C by-point CSV vs Stage D endpoint summary CSV",
            "evidence_locator": "alpha=0.0, c_upd=0.58 comparison",
            "note": f"Return gap={mismatch['return_gap']:.6f}; Stage C remains authoritative.",
        },
        {
            "claim_id": "C8",
            "claim": "Likely periodic VarTD beat explanation: NUM_STEPS=128 and episode_len=26 imply a 13-update alignment cycle.",
            "authority_tier": "reasoned_consistency_check",
            "evidence_file": "locked constants + arithmetic",
            "evidence_locator": f"lcm={beat['lcm']}, beat_updates={beat['beat_updates']}",
            "note": "Reasoned explanation consistent with setup; not directly logged theorem.",
        },
    ]
    claims_df = pd.DataFrame(claims_rows)
    claims_df.to_csv(out_dir / "claims_to_evidence_map.csv", index=False)

    authority_df = pd.DataFrame([
        {
            "evidence_class": "authoritative_boundary",
            "uses": "Stage B bridge + Stage C full-horizon map",
            "allowed_claims": "boundary location, right-censoring, final accepted empirical thesis core",
        },
        {
            "evidence_class": "supportive_mechanism",
            "uses": "Stage D representative traces",
            "allowed_claims": "executed-update patterns, energy depletion/recharge, VarTD behavior, qualitative mechanism",
        },
        {
            "evidence_class": "audit_caveat",
            "uses": "Stage D fragile_full mismatch vs Stage C",
            "allowed_claims": "must be acknowledged explicitly; cannot override Stage C boundary truth",
        },
    ])
    authority_df.to_csv(out_dir / "authoritative_vs_supportive_evidence.csv", index=False)

    lines: List[str] = []
    lines.append("Stage E archival audit report")
    lines.append("")
    lines.append("Verified prerequisites")
    for k, v in inputs.items():
        if v is None:
            lines.append(f"{k}=not_required_or_not_found")
        else:
            try:
                rel = v.relative_to(repo_root)
            except ValueError:
                rel = v
            lines.append(f"{k}={rel}")
    lines.append("")
    lines.append("Locked thesis-core facts")
    lines.append(f"stage_b_complete={stageb_kv.get('stage_b_complete', 'unknown')}")
    lines.append(f"stage_c_subset_complete={stagec_kv.get('stage_c_subset_complete', 'unknown')}")
    lines.append(f"stage_c_completed_runs={stagec_kv.get('completed_runs', 'unknown')}")
    lines.append(f"stage_c_expected_runs={stagec_kv.get('expected_runs', 'unknown')}")
    lines.append(f"stage_c_right_censored_count={stagec_kv.get('right_censored_count', 'unknown')}")
    lines.append("")
    lines.append("Claims-to-evidence outputs")
    lines.append("- claims_to_evidence_map.csv")
    lines.append("- authoritative_vs_supportive_evidence.csv")
    lines.append("")
    lines.append("Stage D fragile_full audit caveat")
    lines.append(
        f"stagec_mean_eval_return_end_mean={fmt(mismatch['stagec_mean_eval_return_end_mean'])} "
        f"staged_eval_return_end_mean={fmt(mismatch['staged_eval_return_end_mean'])} "
        f"return_gap={fmt(mismatch['return_gap'])}"
    )
    lines.append(
        f"stagec_mean_update_fraction={fmt(mismatch['stagec_mean_update_fraction'])} "
        f"staged_mean_update_fraction={fmt(mismatch['staged_mean_update_fraction'])}"
    )
    lines.append(
        "Interpretation: treat Stage D as mechanism-only evidence; do not let the uploaded fragile_full endpoint table override Stage C boundary truth."
    )
    lines.append("")
    lines.append("Likely VarTD beat explanation (consistency check, not theorem)")
    lines.append(f"NUM_STEPS={beat['num_steps']}")
    lines.append(f"episode_len={beat['episode_len']}")
    lines.append(f"LCM={beat['lcm']}")
    lines.append(f"beat_updates={beat['beat_updates']}")
    lines.append(
        "Reasoned interpretation: rollout/episode alignment recurs every 13 updates, which is consistent with the periodic sawtooth-like VarTD dips in enabled_full."
    )
    lines.append("")
    lines.append("Authority rule")
    lines.append("Stage B and Stage C are authoritative for boundary claims.")
    lines.append("Stage D is supportive mechanism evidence only.")
    lines.append("")
    lines.append("Outputs")
    lines.append("- stageE_archival_audit.txt")
    lines.append("- claims_to_evidence_map.csv")
    lines.append("- authoritative_vs_supportive_evidence.csv")
    (out_dir / "stageE_archival_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Runtime structures
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSpec:
    phase: str
    label: str
    family: str
    metric_horizon: str
    total_timesteps: int
    alpha: float
    c_upd: float
    support_mode: str = "none"
    support_total: float = 0.0
    support_start_update: int = 0
    support_window_updates: int = 1
    borrow_limit: float = 0.0
    borrow_interest: float = 0.0
    checkpoint_eval_every: int = DEFAULT_CHECKPOINT_EVAL_EVERY
    note: str = ""


# -----------------------------------------------------------------------------
# Loading and running the patched master
# -----------------------------------------------------------------------------
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


def load_master(repo_root: Path, path: Path):
    install_wandb_stub()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    module_name = f"_stagee_master_{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:
        sys.modules.pop(module_name, None)
        missing = getattr(exc, "name", None) or str(exc)
        raise ModuleNotFoundError(
            f"Missing dependency while importing the Stage-E master script: {missing}. "
            "Run Stage E inside the same environment that successfully ran Stage C/Stage D."
        ) from exc
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not hasattr(mod, "make_train"):
        raise AttributeError(f"Stage-E master script at {path} has no make_train(config) function.")
    return mod


def build_config(spec: RunSpec) -> Dict[str, Any]:
    return {
        "ENV_NAME": ENV_NAME,
        "ENV_KWARGS": dict(ENV_KWARGS),
        "LR": 2.5e-4,
        "NUM_ENVS": NUM_ENVS,
        "NUM_STEPS": NUM_STEPS,
        "TOTAL_TIMESTEPS": int(spec.total_timesteps),
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "NUM_SEEDS": 1,
        "SEED": 0,
        "WANDB_MODE": "disabled",
        "ENTITY": "",
        "PROJECT": "JaxMARL_Gate5_StageE",
        "PRINT_EVERY": 1_000_000_000,
        "EPS_OCC": EPS_OCC,
        "D0": D0,
        "R0": R0,
        "ALPHA": float(spec.alpha),
        "C_UPD": float(spec.c_upd),
        "BETA": BETA,
        "Z_MAX": Z_MAX,
        "EPS_NORM": EPS_NORM,
        "USE_GEOM_FUNDING": True,
        "E0": E0,
        "E_MAX": E_MAX,
        "BORROW_LIMIT": float(spec.borrow_limit),
        "BORROW_INTEREST": float(spec.borrow_interest),
        "SUPPORT_MODE": str(spec.support_mode),
        "SUPPORT_TOTAL": float(spec.support_total),
        "SUPPORT_START_UPDATE": int(spec.support_start_update),
        "SUPPORT_WINDOW_UPDATES": int(spec.support_window_updates),
        "CHECKPOINT_EVAL_EVERY": int(spec.checkpoint_eval_every),
        "DO_EVAL": True,
        "M_EVAL": M_EVAL,
    }


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


def master_seed_key(seed: int):
    """Match the accepted master's CLI seed semantics exactly.

    The accepted master does not pass PRNGKey(SEED) directly into train_fn.
    In main(), it creates rng = PRNGKey(SEED), then rngs = split(rng, NUM_SEEDS),
    and with NUM_SEEDS=1 the single run receives split(PRNGKey(SEED), 1)[0].

    Stage C used that CLI path, so Stage E must use the same per-seed key mapping
    during baseline reproduction if it is to reproduce the authoritative N=3 results.
    """
    import jax

    return jax.random.split(jax.random.PRNGKey(int(seed)), 1)[0]


def support_window_updates(total_timesteps: int, frac: float) -> int:
    num_updates = int(total_timesteps) // (NUM_ENVS * NUM_STEPS)
    return max(1, int(round(num_updates * float(frac))))


def extract_trace_df(metrics_np: Mapping[str, np.ndarray], spec: RunSpec, seed: int) -> pd.DataFrame:
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
            raise ValueError(f"Mismatched length for {raw_key}: {arr.shape[0]} vs {length}")
        data[out_key] = arr
    if length is None:
        raise RuntimeError("No metric traces extracted from the Stage-E master output.")
    if "update" not in data:
        data["update"] = np.arange(length, dtype=int)
    df = pd.DataFrame(data)
    df.insert(0, "seed", int(seed))
    df.insert(0, "c_upd", float(spec.c_upd))
    df.insert(0, "alpha", float(spec.alpha))
    df.insert(0, "metric_horizon", spec.metric_horizon)
    df.insert(0, "family", spec.family)
    df.insert(0, "phase", spec.phase)
    df.insert(0, "condition", spec.label)
    return df


def extract_eval_row(
    eval_np: Mapping[str, np.ndarray],
    trace_df: pd.DataFrame,
    spec: RunSpec,
    seed: int,
    runtime_sec: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "condition": spec.label,
        "phase": spec.phase,
        "family": spec.family,
        "metric_horizon": spec.metric_horizon,
        "alpha": float(spec.alpha),
        "c_upd": float(spec.c_upd),
        "seed": int(seed),
        "runtime_sec": float(runtime_sec),
        "support_mode": spec.support_mode,
        "support_total": float(spec.support_total),
        "borrow_limit": float(spec.borrow_limit),
        "borrow_interest": float(spec.borrow_interest),
        "total_timesteps": int(spec.total_timesteps),
    }
    for out_key, aliases in EVAL_ALIASES.items():
        raw_key = first_present(eval_np, aliases)
        if raw_key is None:
            continue
        scalar = _scalar_from_array(eval_np[raw_key])
        if scalar is not None:
            row[out_key] = scalar

    if "update_u" in trace_df.columns:
        row["mean_update_fraction"] = float(pd.to_numeric(trace_df["update_u"], errors="coerce").mean())
    if "update_counter" in trace_df.columns:
        row["final_executed_updates"] = float(trace_df["update_counter"].iloc[-1])
    if "energy_E_post" in trace_df.columns:
        row["final_energy"] = float(trace_df["energy_E_post"].iloc[-1])
    if "support_s" in trace_df.columns:
        row["total_support_injected"] = float(pd.to_numeric(trace_df["support_s"], errors="coerce").fillna(0.0).sum())
    if "debt_charge" in trace_df.columns:
        row["total_debt_charge"] = float(pd.to_numeric(trace_df["debt_charge"], errors="coerce").fillna(0.0).sum())
    if "debt_level_post" in trace_df.columns:
        row["max_debt_level"] = float(pd.to_numeric(trace_df["debt_level_post"], errors="coerce").max())
    if "vartd_VarTD" in trace_df.columns:
        row["final_vartd"] = float(trace_df["vartd_VarTD"].iloc[-1])
    if "episode_return_end_mean" in trace_df.columns:
        row["train_final_return_end_mean"] = float(trace_df["episode_return_end_mean"].iloc[-1])

    ret = row.get("eval_return_end_mean")
    if ret is not None:
        if spec.metric_horizon == "short":
            row["I_R_short"] = (float(ret) - R0) / (R_REF_SHORT - R0)
            row["learned_R_short_0p5"] = float(float(ret) >= R_THRESH_SHORT)
        elif spec.metric_horizon == "full":
            row["I_R_full"] = (float(ret) - R0) / (R_REF_FULL - R0)
            row["learned_R_full_0p5"] = float(float(ret) >= R_THRESH_FULL)
        elif spec.metric_horizon == "long4m":
            row["I_R_full_anchor_compat"] = (float(ret) - R0) / (R_REF_FULL - R0)
            row["above_R_THRESH_FULL_anchor"] = float(float(ret) >= R_THRESH_FULL)

    return row


def aggregate_trace_df(trace_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in trace_df.columns
        if c not in {"condition", "phase", "family", "metric_horizon", "alpha", "c_upd", "seed", "update"}
    ]
    agg = trace_df.groupby("update")[metric_cols].agg(["mean", "median", q25, q75, "min", "max"])
    agg = agg.reset_index()
    return flatten_multiindex_columns(agg)


def save_band_plot(agg_df: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    q25_col = f"{metric}_q25"
    q75_col = f"{metric}_q75"
    if mean_col not in agg_df.columns:
        return
    x = agg_df["update"].to_numpy(dtype=float)
    y = agg_df[mean_col].to_numpy(dtype=float)
    finite = np.isfinite(y)
    if finite.sum() == 0:
        return
    lo = agg_df[q25_col].to_numpy(dtype=float) if q25_col in agg_df.columns else None
    hi = agg_df[q75_col].to_numpy(dtype=float) if q75_col in agg_df.columns else None
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(x[finite], y[finite], linewidth=2.0, label="mean")
    if lo is not None and hi is not None:
        qmask = np.isfinite(lo) & np.isfinite(hi)
        if qmask.sum() > 0:
            plt.fill_between(x[qmask], lo[qmask], hi[qmask], alpha=0.25, label="q25-q75")
    plt.xlabel("update")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_overlay_plot(agg_map: Mapping[str, pd.DataFrame], metric: str, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(8.0, 4.8))
    plotted = False
    for label, agg_df in agg_map.items():
        col = f"{metric}_mean"
        if col not in agg_df.columns:
            continue
        x = agg_df["update"].to_numpy(dtype=float)
        y = agg_df[col].to_numpy(dtype=float)
        finite = np.isfinite(y)
        if finite.sum() == 0:
            continue
        plt.plot(x[finite], y[finite], linewidth=2.0, label=label)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("update")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def maybe_load_cached_seed(
    seed_dir: Path,
    spec: RunSpec,
    seed: int,
    expected_identity_hash: str,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    trace_path = seed_dir / "trace_per_update.csv"
    eval_path = seed_dir / "eval_summary.json"
    meta_path = seed_dir / "seed_metadata.json"
    if not trace_path.exists() or not eval_path.exists() or not meta_path.exists():
        return None, None
    try:
        trace_df = pd.read_csv(trace_path)
        with eval_path.open("r", encoding="utf-8") as f:
            eval_row = json.load(f)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None, None
    if trace_df.empty:
        return None, None
    if str(meta.get("runtime_identity_hash", "")) != str(expected_identity_hash):
        return None, None
    if int(meta.get("seed", -1)) != int(seed):
        return None, None
    if str(meta.get("condition", "")) != spec.label:
        return None, None
    trace_df = coerce_numeric(trace_df)
    trace_df["condition"] = spec.label
    trace_df["phase"] = spec.phase
    trace_df["family"] = spec.family
    trace_df["metric_horizon"] = spec.metric_horizon
    trace_df["alpha"] = float(spec.alpha)
    trace_df["c_upd"] = float(spec.c_upd)
    trace_df["seed"] = int(seed)
    eval_row = dict(eval_row)
    eval_row["condition"] = spec.label
    eval_row["phase"] = spec.phase
    eval_row["family"] = spec.family
    eval_row["metric_horizon"] = spec.metric_horizon
    eval_row["alpha"] = float(spec.alpha)
    eval_row["c_upd"] = float(spec.c_upd)
    eval_row["seed"] = int(seed)
    return trace_df, eval_row


def run_condition(
    mod: Any,
    spec: RunSpec,
    seeds: Sequence[int],
    out_dir: Path,
    runtime_identity: Mapping[str, Any],
    force_rerun: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import jax

    ensure_dir(out_dir)
    write_json(out_dir / "condition_provenance.json", dict(runtime_identity))
    config = build_config(spec)
    train_fn = mod.make_train(config)
    compiled_train = jax.jit(train_fn)
    endpoint_rows: List[Dict[str, Any]] = []
    all_traces: List[pd.DataFrame] = []
    expected_identity_hash = str(runtime_identity["runtime_identity_hash"])

    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed:02d}"
        ensure_dir(seed_dir)

        if not force_rerun:
            cached_trace_df, cached_eval_row = maybe_load_cached_seed(
                seed_dir,
                spec,
                seed,
                expected_identity_hash=expected_identity_hash,
            )
            if cached_trace_df is not None and cached_eval_row is not None:
                print(f"[Stage E] cache hit {spec.label} seed={seed:02d}")
                all_traces.append(cached_trace_df)
                endpoint_rows.append(cached_eval_row)
                continue

        print(f"[Stage E] run {spec.label} seed={seed:02d} seed_key_scheme={SEED_KEY_SCHEME}")
        t0 = time.perf_counter()
        seed_key = master_seed_key(int(seed))
        out = compiled_train(seed_key)
        metrics_device = out["metrics"]
        eval_device = out["eval"]
        block_until_ready(metrics_device)
        block_until_ready(eval_device)
        runtime_sec = time.perf_counter() - t0

        metrics_np = to_numpy_tree(metrics_device)
        eval_np = to_numpy_tree(eval_device)
        del out, metrics_device, eval_device

        trace_df = extract_trace_df(metrics_np, spec, seed)
        eval_row = extract_eval_row(eval_np, trace_df, spec, seed, runtime_sec)

        trace_df.to_csv(seed_dir / "trace_per_update.csv", index=False)
        with (seed_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
            json.dump(eval_row, f, indent=2, sort_keys=True)
        seed_meta = {
            **dict(runtime_identity),
            "condition": spec.label,
            "seed": int(seed),
            "runtime_sec": float(runtime_sec),
        }
        write_json(seed_dir / "seed_metadata.json", seed_meta)

        all_traces.append(trace_df)
        endpoint_rows.append(eval_row)

        del metrics_np, eval_np, trace_df
        gc.collect()

    if not endpoint_rows or not all_traces:
        raise RuntimeError(f"No Stage E outputs were produced for condition {spec.label}.")

    endpoint_df = coerce_numeric(pd.DataFrame(endpoint_rows))
    endpoint_df.to_csv(out_dir / "seed_endpoint_summary.csv", index=False)

    trace_long = pd.concat(all_traces, ignore_index=True)
    trace_long = coerce_numeric(trace_long)
    trace_long.to_csv(out_dir / "trace_per_seed_long.csv", index=False)

    trace_agg = aggregate_trace_df(trace_long)
    trace_agg.to_csv(out_dir / "trace_aggregate.csv", index=False)

    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    for metric, title, ylabel in PLOT_SPECS:
        save_band_plot(trace_agg, metric, f"{spec.label}: {title}", ylabel, plots_dir / f"{metric}.png")

    return endpoint_df, trace_agg


# -----------------------------------------------------------------------------
# Phase construction
# -----------------------------------------------------------------------------
def build_run_label(prefix: str, family: str, horizon_tag: str, alpha: float, c_upd: float) -> str:
    return f"{prefix}__{family}__{horizon_tag}__a{alpha:.2f}__c{c_upd:.2f}".replace(".", "p")


def make_base_spec(phase: str, prefix: str, horizon_tag: str, metric_horizon: str, total_timesteps: int, alpha: float, c_upd: float) -> RunSpec:
    return RunSpec(
        phase=phase,
        label=build_run_label(prefix, "base", horizon_tag, alpha, c_upd),
        family="base",
        metric_horizon=metric_horizon,
        total_timesteps=total_timesteps,
        alpha=alpha,
        c_upd=c_upd,
        note="Support off and borrowing off.",
    )


def make_support_spec(base: RunSpec, args: argparse.Namespace) -> RunSpec:
    return replace(
        base,
        label=base.label.replace("__base__", "__support__"),
        family="support",
        support_mode=str(args.support_mode),
        support_total=float(args.support_total),
        support_start_update=int(args.support_start_update),
        support_window_updates=support_window_updates(base.total_timesteps, float(args.support_window_frac)),
        borrow_limit=0.0,
        borrow_interest=0.0,
        note="Pure support-injection intervention.",
    )


def make_borrow_spec(base: RunSpec, args: argparse.Namespace) -> RunSpec:
    return replace(
        base,
        label=base.label.replace("__base__", "__borrow__"),
        family="borrow",
        support_mode="none",
        support_total=0.0,
        support_start_update=0,
        support_window_updates=1,
        borrow_limit=float(args.borrow_limit),
        borrow_interest=float(args.borrow_interest),
        note="Pure borrowing intervention.",
    )


def specs_for_reproduce_baseline(args: argparse.Namespace) -> List[RunSpec]:
    return [
        make_base_spec("reproduce_baseline", "anchor_short_fail", "t400k", "short", SHORT_TIMESTEPS, 0.0, 0.60),
        make_base_spec("reproduce_baseline", "anchor_fragile_full", "t1m", "full", FULL_TIMESTEPS, 0.0, 0.58),
        make_base_spec("reproduce_baseline", "anchor_edge_full", "t1m", "full", FULL_TIMESTEPS, 0.1, 1.00),
        make_base_spec("reproduce_baseline", "anchor_enabled_full", "t1m", "full", FULL_TIMESTEPS, 1.0, 0.90),
    ]


def specs_for_baseline_4m(args: argparse.Namespace) -> List[RunSpec]:
    return [
        make_base_spec("baseline_4m", "anchor_short_fail_probe", "t4m", "long4m", LONG_TIMESTEPS, 0.0, 0.60),
        make_base_spec("baseline_4m", "anchor_fragile_full_probe", "t4m", "long4m", LONG_TIMESTEPS, 0.0, 0.58),
        make_base_spec("baseline_4m", "anchor_edge_full_probe", "t4m", "long4m", LONG_TIMESTEPS, 0.1, 1.00),
        make_base_spec("baseline_4m", "anchor_enabled_full_probe", "t4m", "long4m", LONG_TIMESTEPS, 1.0, 0.90),
    ]


def specs_for_rescue_short(args: argparse.Namespace, include_sweep: bool = False) -> List[RunSpec]:
    base = make_base_spec("rescue_short", "rescue_short_anchor", "t400k", "short", SHORT_TIMESTEPS, 0.0, 0.60)
    specs = [base, make_support_spec(base, args), make_borrow_spec(base, args)]
    if include_sweep:
        for c in [0.54, 0.56, 0.58, 0.60, 0.62]:
            sweep_base = make_base_spec("rescue_short", f"rescue_short_sweep_c{c:.2f}", "t400k", "short", SHORT_TIMESTEPS, 0.0, c)
            specs.extend([sweep_base, make_support_spec(sweep_base, args), make_borrow_spec(sweep_base, args)])
    return specs


def specs_for_extension_full(args: argparse.Namespace) -> List[RunSpec]:
    anchors: List[Tuple[str, float, float]] = [
        ("fragile_full", 0.0, 0.58),
        ("edge_full", 0.1, 1.00),
    ]
    if not args.skip_enabled_control:
        anchors.append(("enabled_full", 1.0, 0.90))

    specs: List[RunSpec] = []
    for tag, alpha, c_upd in anchors:
        base_1m = make_base_spec("extension_full", f"{tag}", "t1m", "full", FULL_TIMESTEPS, alpha, c_upd)
        specs.extend([base_1m, make_support_spec(base_1m, args), make_borrow_spec(base_1m, args)])
        if not args.skip_4m_in_extension:
            base_4m = make_base_spec("extension_full", f"{tag}", "t4m", "long4m", LONG_TIMESTEPS, alpha, c_upd)
            specs.extend([base_4m, make_support_spec(base_4m, args), make_borrow_spec(base_4m, args)])
    return specs


# -----------------------------------------------------------------------------
# Summaries and reports
# -----------------------------------------------------------------------------
def summarize_endpoints(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_seeds": int(len(df))}
    for c in df.columns:
        if c in {"condition", "phase", "family", "metric_horizon", "alpha", "c_upd", "seed", "support_mode"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out[f"{c}_mean"] = float(df[c].mean())
            out[f"{c}_sd"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def summary_get(summary: Mapping[str, Any], base_key: str) -> Any:
    return summary.get(f"{base_key}_mean", float("nan"))


def summary_sd(summary: Mapping[str, Any], base_key: str) -> Any:
    return summary.get(f"{base_key}_sd", float("nan"))


def build_condition_summary_row(spec: RunSpec, summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "condition": spec.label,
        "phase": spec.phase,
        "family": spec.family,
        "metric_horizon": spec.metric_horizon,
        "alpha": spec.alpha,
        "c_upd": spec.c_upd,
        "total_timesteps": spec.total_timesteps,
        **dict(summary),
    }


def _common_stage_d_metric_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    preferred = [
        "eval_return_end_mean",
        "eval_dbar_post_mean",
        "eval_CSR_post",
        "mean_update_fraction",
        "final_executed_updates",
        "final_energy",
        "final_vartd",
        "learned_R_short_0p5",
        "learned_R_full_0p5",
    ]
    return [c for c in preferred if c in df_a.columns and c in df_b.columns]


def compare_to_stage_d_seed_exact(
    spec: RunSpec,
    endpoint_df: pd.DataFrame,
    stage_d_seed_tables: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "stage_d_seed_match_available": 0,
        "stage_d_seed_exact_match": 0,
        "stage_d_seed_match_condition": "",
    }
    if spec.family != "base":
        return out
    candidates: List[Mapping[str, Any]] = []
    for item in stage_d_seed_tables:
        df = item["df"]
        if "alpha" not in df.columns or "c_upd" not in df.columns:
            continue
        alpha0 = float(pd.to_numeric(df["alpha"], errors="coerce").iloc[0])
        c0 = float(pd.to_numeric(df["c_upd"], errors="coerce").iloc[0])
        if not np.isclose(alpha0, spec.alpha):
            continue
        if not np.isclose(c0, spec.c_upd):
            continue
        if "total_timesteps" in df.columns:
            t0 = int(pd.to_numeric(df["total_timesteps"], errors="coerce").iloc[0])
            if t0 != int(spec.total_timesteps):
                continue
        candidates.append(item)
    if not candidates:
        return out

    candidates = sorted(candidates, key=lambda item: len(str(item["path"])))
    target = candidates[0]
    df_stage_d = coerce_numeric(target["df"]).copy()
    df_stage_e = coerce_numeric(endpoint_df).copy()
    if "seed" not in df_stage_d.columns or "seed" not in df_stage_e.columns:
        return out

    common_metrics = _common_stage_d_metric_columns(df_stage_e, df_stage_d)
    if not common_metrics:
        out["stage_d_seed_match_available"] = 1
        out["stage_d_seed_match_condition"] = str(target["condition_name"])
        return out

    merged = pd.merge(
        df_stage_e[["seed", *common_metrics]],
        df_stage_d[["seed", *common_metrics]],
        on="seed",
        suffixes=("_stagee", "_staged"),
        how="inner",
    )
    if merged.empty:
        return out

    out["stage_d_seed_match_available"] = 1
    out["stage_d_seed_match_condition"] = str(target["condition_name"])
    metric_exact_flags: List[bool] = []
    for metric in common_metrics:
        a = pd.to_numeric(merged[f"{metric}_stagee"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(merged[f"{metric}_staged"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() == 0:
            continue
        max_abs_diff = float(np.max(np.abs(a[mask] - b[mask])))
        out[f"stage_d_{metric}_max_abs_diff"] = max_abs_diff
        metric_exact_flags.append(bool(max_abs_diff <= EXACT_MATCH_ATOL))
    out["stage_d_seed_exact_match"] = int(bool(metric_exact_flags) and all(metric_exact_flags))
    return out


def compare_to_authoritative(
    spec: RunSpec,
    summary_row: Mapping[str, Any],
    refs: Mapping[str, pd.DataFrame],
    endpoint_df: pd.DataFrame,
    stage_d_seed_tables: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    row = authoritative_row_for_spec(spec, refs)
    if spec.metric_horizon == "long4m":
        comparison_kind = "new_long_horizon_fact"
    elif spec.family == "base" and spec.total_timesteps in {SHORT_TIMESTEPS, FULL_TIMESTEPS}:
        comparison_kind = "baseline_reproduction"
    elif spec.total_timesteps in {SHORT_TIMESTEPS, FULL_TIMESTEPS}:
        comparison_kind = "intervention_at_authoritative_anchor"
    else:
        comparison_kind = "new_fact"

    out: Dict[str, Any] = {
        "condition": spec.label,
        "phase": spec.phase,
        "family": spec.family,
        "metric_horizon": spec.metric_horizon,
        "authoritative_source": "none" if row is None else ("stageB_bridge_by_point" if spec.total_timesteps == SHORT_TIMESTEPS else "stageC_subset_refine_by_point"),
        "comparison_kind": comparison_kind,
    }
    out.update(compare_to_stage_d_seed_exact(spec, endpoint_df, stage_d_seed_tables))

    if row is None:
        out["note"] = "No matched authoritative short/full baseline row exists for this condition."
        return out

    measured_reference_pairs = [
        ("eval_return_end_mean", "mean_eval_return_end_mean"),
        ("mean_update_fraction", "mean_update_fraction"),
        ("final_executed_updates", "mean_final_executed_updates"),
        ("final_energy", "mean_final_energy"),
        ("final_vartd", "mean_final_VarTD"),
    ]
    if spec.total_timesteps == SHORT_TIMESTEPS:
        measured_reference_pairs.append(("learned_R_short_0p5", "P_learn_R_short_0p5"))
    elif spec.total_timesteps == FULL_TIMESTEPS:
        measured_reference_pairs.append(("learned_R_full_0p5", "P_learn_R_full_0p5"))

    for measured_key, ref_key in measured_reference_pairs:
        measured = measured_summary_value(summary_row, measured_key)
        if not np.isfinite(measured):
            continue
        if ref_key not in row.index:
            continue
        try:
            reference = float(row[ref_key])
        except Exception:
            continue
        out[f"{measured_key}_measured"] = measured
        out[f"{measured_key}_reference"] = reference
        out[f"{measured_key}_diff"] = measured - reference

    if spec.family == "base" and spec.total_timesteps in {SHORT_TIMESTEPS, FULL_TIMESTEPS}:
        return_diff = finite_or_nan(out.get("eval_return_end_mean_diff", float("nan")))
        learn_key = "learned_R_short_0p5_diff" if spec.total_timesteps == SHORT_TIMESTEPS else "learned_R_full_0p5_diff"
        learn_diff = finite_or_nan(out.get(learn_key, float("nan")))
        out["baseline_return_tol"] = BASELINE_RETURN_TOL
        out["baseline_learn_prob_tol"] = BASELINE_LEARN_PROB_TOL
        out["baseline_return_pass"] = int(np.isfinite(return_diff) and abs(return_diff) <= BASELINE_RETURN_TOL)
        out["baseline_learn_prob_pass"] = int(np.isfinite(learn_diff) and abs(learn_diff) <= BASELINE_LEARN_PROB_TOL)
        out["baseline_reproduction_pass"] = int(bool(out["baseline_return_pass"] and out["baseline_learn_prob_pass"]))
    return out


def build_checkpoint_behavior_row(spec: RunSpec, agg_df: pd.DataFrame) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "condition": spec.label,
        "phase": spec.phase,
        "family": spec.family,
        "metric_horizon": spec.metric_horizon,
        "alpha": spec.alpha,
        "c_upd": spec.c_upd,
        "total_timesteps": spec.total_timesteps,
        "checkpoint_behavior_class": "no_checkpoint_eval",
    }
    col = "eval_return_end_mean_cp_mean"
    if col not in agg_df.columns:
        return row
    cp_df = agg_df.loc[np.isfinite(pd.to_numeric(agg_df[col], errors="coerce"))].copy()
    if cp_df.empty:
        return row
    vals = pd.to_numeric(cp_df[col], errors="coerce").to_numpy(dtype=float)
    updates = pd.to_numeric(cp_df["update"], errors="coerce").to_numpy(dtype=float)
    first = float(vals[0])
    peak_idx = int(np.nanargmax(vals))
    peak = float(vals[peak_idx])
    peak_update = float(updates[peak_idx])
    final = float(vals[-1])
    row["checkpoint_count"] = int(len(vals))
    row["checkpoint_first_return"] = first
    row["checkpoint_peak_return"] = peak
    row["checkpoint_peak_update"] = peak_update
    row["checkpoint_final_return"] = final
    row["checkpoint_delta_first_to_final"] = final - first
    row["checkpoint_delta_peak_to_final"] = final - peak
    if abs(final - first) <= CHECKPOINT_FLAT_TOL:
        row["checkpoint_behavior_class"] = "flat"
    elif (peak - final) > CHECKPOINT_DECLINE_TOL and peak > first + CHECKPOINT_FLAT_TOL:
        row["checkpoint_behavior_class"] = "improve_then_decline"
    elif final > first + CHECKPOINT_FLAT_TOL:
        row["checkpoint_behavior_class"] = "net_improvement"
    else:
        row["checkpoint_behavior_class"] = "mixed"
    return row


def validate_baseline_reproduction(comparisons: Sequence[Mapping[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for comp in comparisons:
        if comp.get("comparison_kind") != "baseline_reproduction":
            continue
        row = dict(comp)
        row["baseline_reproduction_pass"] = int(row.get("baseline_reproduction_pass", 0))
        rows.append(row)
    df = coerce_numeric(pd.DataFrame(rows))
    summary = {
        "stagee_script_version": STAGEE_SCRIPT_VERSION,
        "timestamp_utc": utc_now_iso(),
        "baseline_return_tol": BASELINE_RETURN_TOL,
        "baseline_learn_prob_tol": BASELINE_LEARN_PROB_TOL,
        "n_reproduction_conditions": int(len(df)),
        "overall_reproduction_pass": bool(len(df) > 0 and df["baseline_reproduction_pass"].astype(int).eq(1).all()),
    }
    if not df.empty:
        summary["failing_conditions"] = [str(x) for x in df.loc[df["baseline_reproduction_pass"].astype(int) != 1, "condition"].tolist()]
    else:
        summary["failing_conditions"] = []
    return df, summary



def save_phase_overlays(agg_map: Mapping[str, pd.DataFrame], out_dir: Path, phase: str) -> None:
    ensure_dir(out_dir)
    save_overlay_plot(agg_map, "eval_return_end_mean_cp", f"{PHASE_TITLES.get(phase, phase)}: checkpoint eval return", "eval return", out_dir / "checkpoint_eval_return_overlay.png")
    save_overlay_plot(agg_map, "energy_E_post", f"{PHASE_TITLES.get(phase, phase)}: post-update energy", "energy", out_dir / "energy_overlay.png")
    save_overlay_plot(agg_map, "debt_level_post", f"{PHASE_TITLES.get(phase, phase)}: debt level", "debt", out_dir / "debt_overlay.png")
    save_overlay_plot(agg_map, "support_s", f"{PHASE_TITLES.get(phase, phase)}: support schedule", "support", out_dir / "support_overlay.png")
    save_overlay_plot(agg_map, "update_counter", f"{PHASE_TITLES.get(phase, phase)}: executed updates", "executed updates", out_dir / "update_counter_overlay.png")


def write_phase_report(
    out_dir: Path,
    repo_root: Path,
    phase: str,
    specs: Sequence[RunSpec],
    summaries: Mapping[str, Dict[str, Any]],
    refs: Mapping[str, pd.DataFrame],
    comparisons: Sequence[Dict[str, Any]],
    generated_master: Path,
    accepted_master: Path,
    seeds: Sequence[int],
    phase_provenance: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    baseline_validation_summary: Optional[Mapping[str, Any]] = None,
) -> None:
    lines: List[str] = []
    lines.append(PHASE_TITLES.get(phase, phase))
    lines.append("")
    lines.append(f"stagee_script_version={STAGEE_SCRIPT_VERSION}")
    lines.append(f"script_path={phase_provenance.get('script_path')}")
    lines.append(f"script_sha256={phase_provenance.get('script_sha256')}")
    lines.append(f"repo_root={repo_root}")
    lines.append(f"accepted_master={safe_rel(accepted_master, repo_root)}")
    lines.append(f"accepted_master_sha256={sha256_file(accepted_master)}")
    lines.append(f"generated_master={safe_rel(generated_master, repo_root)}")
    lines.append(f"generated_master_sha256={sha256_file(generated_master)}")
    lines.append(f"python={sys.version.split()[0]}")
    lines.append(f"platform={platform.platform()}")
    lines.append(f"seeds={min(seeds)}..{max(seeds)} ({len(seeds)} total)")
    lines.append(f"seed_key_scheme={SEED_KEY_SCHEME}")
    lines.append(f"phase_args={phase_provenance.get('args')}")
    lines.append("")
    lines.append("Locked constants")
    lines.append(f"R0={R0}")
    lines.append(f"D0={D0}")
    lines.append(f"R_REF_SHORT={R_REF_SHORT}")
    lines.append(f"R_THRESH_SHORT={R_THRESH_SHORT}")
    lines.append(f"R_REF_FULL={R_REF_FULL}")
    lines.append(f"R_THRESH_FULL={R_THRESH_FULL}")
    lines.append(f"EPS_OCC={EPS_OCC}")
    lines.append(f"M_EVAL={M_EVAL}")
    lines.append(f"NUM_ENVS={NUM_ENVS}")
    lines.append(f"NUM_STEPS={NUM_STEPS}")
    lines.append(f"E0={E0}")
    lines.append(f"E_MAX={E_MAX}")
    lines.append("")
    if baseline_validation_summary is not None:
        lines.append("Baseline reproduction gate")
        lines.append(f"overall_reproduction_pass={baseline_validation_summary.get('overall_reproduction_pass')}")
        lines.append(f"failing_conditions={baseline_validation_summary.get('failing_conditions')}")
        lines.append("")
    lines.append("Conditions")
    for spec in specs:
        lines.append(
            f"condition={spec.label} family={spec.family} metric_horizon={spec.metric_horizon} "
            f"timesteps={spec.total_timesteps} alpha={fmt(spec.alpha, 3)} c_upd={fmt(spec.c_upd, 3)} "
            f"support_mode={spec.support_mode} support_total={fmt(spec.support_total, 3)} "
            f"borrow_limit={fmt(spec.borrow_limit, 3)} borrow_interest={fmt(spec.borrow_interest, 3)}"
        )
        if spec.note:
            lines.append(f"note={spec.note}")
    lines.append("")
    lines.append("Per-condition endpoint summaries")
    for spec in specs:
        s = summaries[spec.label]
        pieces = [
            f"condition={spec.label}",
            f"eval_return_end_mean={fmt(summary_get(s, 'eval_return_end_mean'))}±{fmt(summary_sd(s, 'eval_return_end_mean'))}",
            f"eval_dbar_post_mean={fmt(summary_get(s, 'eval_dbar_post_mean'))}±{fmt(summary_sd(s, 'eval_dbar_post_mean'))}",
            f"eval_CSR_post={fmt(summary_get(s, 'eval_CSR_post'))}±{fmt(summary_sd(s, 'eval_CSR_post'))}",
            f"mean_update_fraction={fmt(summary_get(s, 'mean_update_fraction'))}±{fmt(summary_sd(s, 'mean_update_fraction'))}",
            f"final_executed_updates={fmt(summary_get(s, 'final_executed_updates'))}±{fmt(summary_sd(s, 'final_executed_updates'))}",
            f"final_energy={fmt(summary_get(s, 'final_energy'))}±{fmt(summary_sd(s, 'final_energy'))}",
            f"total_support_injected={fmt(summary_get(s, 'total_support_injected'))}",
            f"total_debt_charge={fmt(summary_get(s, 'total_debt_charge'))}",
            f"max_debt_level={fmt(summary_get(s, 'max_debt_level'))}",
            f"final_vartd={fmt(summary_get(s, 'final_vartd'))}±{fmt(summary_sd(s, 'final_vartd'))}",
        ]
        if spec.metric_horizon == "short":
            pieces.append(f"P_learn_R_short_0p5={fmt(summary_get(s, 'learned_R_short_0p5'))}")
        elif spec.metric_horizon == "full":
            pieces.append(f"P_learn_R_full_0p5={fmt(summary_get(s, 'learned_R_full_0p5'))}")
        else:
            pieces.append(f"I_R_full_anchor_compat={fmt(summary_get(s, 'I_R_full_anchor_compat'))}")
        lines.append(" ".join(pieces))
    lines.append("")
    lines.append("Authoritative comparison rows")
    for row in comparisons:
        pieces = [
            f"condition={row['condition']}",
            f"source={row['authoritative_source']}",
            f"comparison_kind={row['comparison_kind']}",
        ]
        if row.get("note"):
            pieces.append(f"note={row['note']}")
        for k in sorted(row):
            if k in {"condition", "authoritative_source", "comparison_kind", "note"}:
                continue
            pieces.append(f"{k}={fmt(row[k])}")
        lines.append(" ".join(pieces))
    lines.append("")
    lines.append("Checkpoint-behavior diagnostics")
    for row in checkpoint_rows:
        pieces = [f"{k}={v}" if not isinstance(v, float) else f"{k}={fmt(v)}" for k, v in row.items()]
        lines.append(" ".join(pieces))
    lines.append("")
    lines.append("Outputs")
    lines.append("- phase_provenance.json")
    lines.append("- phase_manifest.csv")
    lines.append("- phase_endpoint_summary_all_conditions.csv")
    lines.append("- phase_endpoint_summary_by_condition.csv")
    lines.append("- authoritative_comparison.csv")
    lines.append("- checkpoint_behavior_by_condition.csv")
    if baseline_validation_summary is not None:
        lines.append("- baseline_validation_per_condition.csv")
        lines.append("- baseline_validation_summary.json")
    lines.append("- PHASE_REPORT.txt")
    lines.append("- cross_condition_plots/*.png")
    lines.append("- conditions/<condition>/condition_provenance.json")
    lines.append("- conditions/<condition>/seed_XX/trace_per_update.csv")
    lines.append("- conditions/<condition>/seed_XX/eval_summary.json")
    lines.append("- conditions/<condition>/seed_XX/seed_metadata.json")
    lines.append("- conditions/<condition>/seed_endpoint_summary.csv")
    lines.append("- conditions/<condition>/trace_per_seed_long.csv")
    lines.append("- conditions/<condition>/trace_aggregate.csv")
    lines.append("- conditions/<condition>/plots/*.png")
    (out_dir / "PHASE_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Rescue logic and phase execution
# -----------------------------------------------------------------------------
def rescue_is_clear(by_condition_df: pd.DataFrame) -> bool:
    if by_condition_df.empty or "condition" not in by_condition_df.columns:
        return False
    base_rows = by_condition_df.loc[by_condition_df["family"] == "base"]
    if base_rows.empty:
        return False
    base = base_rows.iloc[0]
    base_p = float(base.get("learned_R_short_0p5_mean", np.nan))
    for fam in ["support", "borrow"]:
        rows = by_condition_df.loc[by_condition_df["family"] == fam]
        if rows.empty:
            continue
        p = float(rows.iloc[0].get("learned_R_short_0p5_mean", np.nan))
        if np.isfinite(base_p) and np.isfinite(p) and p >= 0.75 and (p - base_p) >= 0.25:
            return True
    return False


def run_phase(
    repo_root: Path,
    phase: str,
    specs: Sequence[RunSpec],
    seeds: Sequence[int],
    force_repatch: bool,
    force_rerun: bool,
    args: Optional[argparse.Namespace] = None,
) -> Path:
    out_dir = repo_root / "runs" / "stageE_clipped_robustness_extension" / phase
    ensure_dir(out_dir)
    inputs = load_repo_inputs(repo_root)
    accepted_master = need(inputs["accepted_master"], "accepted Stage-E baseline master")
    generated_master = write_stagee_master(repo_root, force=force_repatch)
    mod = load_master(repo_root, generated_master)
    refs = load_authoritative_reference_tables(repo_root)
    stage_d_seed_tables = load_stage_d_seed_endpoint_tables(repo_root)

    phase_manifest = pd.DataFrame([asdict(s) for s in specs])
    phase_manifest.to_csv(out_dir / "phase_manifest.csv", index=False)

    phase_args = args if args is not None else argparse.Namespace(mode=phase, repo_root=repo_root, seed_start=min(seeds), seed_stop=max(seeds) + 1)
    phase_provenance = collect_session_provenance(repo_root, phase_args)
    phase_provenance.update(
        {
            "phase": phase,
            "accepted_master_path": str(accepted_master.resolve()),
            "accepted_master_sha256": sha256_file(accepted_master),
            "generated_master_path": str(generated_master.resolve()),
            "generated_master_sha256": sha256_file(generated_master),
            "phase_manifest_sha256": sha256_file(out_dir / "phase_manifest.csv"),
        }
    )
    write_json(out_dir / "phase_provenance.json", phase_provenance)
    maybe_add_pip_freeze(out_dir)

    endpoint_frames: List[pd.DataFrame] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    summary_rows: Dict[str, Dict[str, Any]] = {}
    agg_map: Dict[str, pd.DataFrame] = {}
    comparisons: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []

    conditions_root = out_dir / "conditions"
    ensure_dir(conditions_root)
    for spec in specs:
        cond_dir = conditions_root / spec.label
        config = build_config(spec)
        runtime_identity = build_runtime_identity(
            repo_root=repo_root,
            accepted_master=accepted_master,
            generated_master=generated_master,
            spec=spec,
            config=config,
        )
        endpoint_df, trace_agg = run_condition(
            mod,
            spec,
            seeds=seeds,
            out_dir=cond_dir,
            runtime_identity=runtime_identity,
            force_rerun=force_rerun,
        )
        endpoint_frames.append(endpoint_df)
        agg_map[spec.label] = trace_agg
        summaries[spec.label] = summarize_endpoints(endpoint_df)
        summary_rows[spec.label] = build_condition_summary_row(spec, summaries[spec.label])
        comp = compare_to_authoritative(spec, summary_rows[spec.label], refs, endpoint_df, stage_d_seed_tables)
        comparisons.append(comp)
        checkpoint_rows.append(build_checkpoint_behavior_row(spec, trace_agg))

    endpoint_all = coerce_numeric(pd.concat(endpoint_frames, ignore_index=True))
    endpoint_all.to_csv(out_dir / "phase_endpoint_summary_all_conditions.csv", index=False)

    by_condition_df = coerce_numeric(pd.DataFrame([summary_rows[s.label] for s in specs]))
    by_condition_df.to_csv(out_dir / "phase_endpoint_summary_by_condition.csv", index=False)

    comparison_df = coerce_numeric(pd.DataFrame(comparisons))
    comparison_df.to_csv(out_dir / "authoritative_comparison.csv", index=False)

    checkpoint_df = coerce_numeric(pd.DataFrame(checkpoint_rows))
    checkpoint_df.to_csv(out_dir / "checkpoint_behavior_by_condition.csv", index=False)

    baseline_validation_summary: Optional[Dict[str, Any]] = None
    if phase == "reproduce_baseline":
        validation_df, baseline_validation_summary = validate_baseline_reproduction(comparisons)
        gate_signature = build_reproduction_gate_signature(
            repo_root=repo_root,
            args=(args if args is not None else argparse.Namespace(checkpoint_eval_every=DEFAULT_CHECKPOINT_EVAL_EVERY)),
            seeds=seeds,
            accepted_master=accepted_master,
            generated_master=generated_master,
        )
        baseline_validation_summary["reproduction_gate_signature"] = gate_signature
        baseline_validation_summary["script_path"] = str(current_script_path())
        baseline_validation_summary["script_sha256"] = current_script_sha256()
        baseline_validation_summary["seed_key_scheme"] = SEED_KEY_SCHEME
        validation_df.to_csv(out_dir / "baseline_validation_per_condition.csv", index=False)
        write_json(out_dir / "baseline_validation_summary.json", baseline_validation_summary)

    cross_dir = out_dir / "cross_condition_plots"
    save_phase_overlays(agg_map, cross_dir, phase)
    write_phase_report(
        out_dir,
        repo_root,
        phase,
        specs,
        summaries,
        refs,
        comparisons,
        generated_master,
        accepted_master,
        seeds,
        phase_provenance,
        checkpoint_rows,
        baseline_validation_summary=baseline_validation_summary,
    )
    return out_dir


def ensure_validated_reproduction(
    repo_root: Path,
    args: argparse.Namespace,
    seeds: Sequence[int],
) -> bool:
    reproduce_dir = repo_root / "runs" / "stageE_clipped_robustness_extension" / "reproduce_baseline"
    summary_path = reproduce_dir / "baseline_validation_summary.json"

    inputs = load_repo_inputs(repo_root)
    accepted_master = need(inputs["accepted_master"], "accepted Stage-E baseline master")
    generated_master = write_stagee_master(repo_root, force=args.force_repatch)
    current_signature = build_reproduction_gate_signature(
        repo_root=repo_root,
        args=args,
        seeds=seeds,
        accepted_master=accepted_master,
        generated_master=generated_master,
    )
    signature_ok, signature_reason = reproduction_gate_is_current(summary_path, current_signature)

    need_rerun = bool(args.force_rerun or not summary_path.exists() or not signature_ok)
    if need_rerun:
        print(f"[Stage E] running baseline reproduction gate before later phases (reason={signature_reason})")
        reproduce_specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_reproduce_baseline(args)]
        reproduce_dir = run_phase(repo_root, "reproduce_baseline", reproduce_specs, seeds, args.force_repatch, args.force_rerun, args=args)
        summary_path = reproduce_dir / "baseline_validation_summary.json"

    if not summary_path.exists():
        raise SystemExit("Baseline reproduction summary is missing; cannot validate later phases.")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if bool(summary.get("overall_reproduction_pass", False)):
        print("[Stage E] baseline reproduction gate PASSED")
        return True

    msg = (
        "Baseline reproduction gate FAILED. "
        "The extension patch did not reproduce all authoritative short/full anchors within the configured "
        "return and learning-probability tolerances. Refusing later-phase claims until this is resolved."
    )
    if args.allow_unvalidated_baseline:
        print(f"[Stage E] WARNING: {msg}")
        return False
    raise SystemExit(msg)


# -----------------------------------------------------------------------------
# CLI orchestration
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage E clipped robustness extension orchestrator.")
    parser.add_argument(
        "mode",
        choices=["audit_existing", "reproduce_baseline", "baseline_4m", "rescue_short", "extension_full", "all"],
        help="Stage E mode to run.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--seed-stop", type=int, default=SEED_STOP_EXCLUSIVE)
    parser.add_argument("--checkpoint-eval-every", type=int, default=DEFAULT_CHECKPOINT_EVAL_EVERY)
    parser.add_argument("--support-mode", choices=["none", "frontload", "drip", "pulse"], default=DEFAULT_SUPPORT_MODE)
    parser.add_argument("--support-total", type=float, default=DEFAULT_SUPPORT_TOTAL)
    parser.add_argument("--support-start-update", type=int, default=DEFAULT_SUPPORT_START_UPDATE)
    parser.add_argument("--support-window-frac", type=float, default=DEFAULT_SUPPORT_WINDOW_FRAC)
    parser.add_argument("--borrow-limit", type=float, default=DEFAULT_BORROW_LIMIT)
    parser.add_argument("--borrow-interest", type=float, default=DEFAULT_BORROW_INTEREST)
    parser.add_argument("--force-repatch", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--allow-unvalidated-baseline", action="store_true", help="Allow E0C/E1/E2 to proceed even when the E0B reproduction gate fails. Not recommended for defense-safe claims.")
    parser.add_argument("--skip-short-sweep", action="store_true", help="Do not run the optional E1 local sweep even if rescue is clear.")
    parser.add_argument("--skip-enabled-control", action="store_true", help="Skip the optional enabled full-horizon control in E2.")
    parser.add_argument("--skip-4m-in-extension", action="store_true", help="Run E2 only at the accepted 1M horizon.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = infer_repo_root(args.repo_root)
    args.repo_root = repo_root

    log_root = repo_root / "runs" / "stageE_clipped_robustness_extension" / "logs"
    ensure_dir(log_root)
    stamp = utc_now_compact()
    log_path = log_root / f"{stamp}__{args.mode}.log"

    with tee_stdout_stderr(log_path):
        print(f"[Stage E] mode={args.mode}")
        print(f"[Stage E] repo_root={repo_root}")
        print(f"[Stage E] log_path={log_path}")
        print_startup_banner()
        load_repo_inputs(repo_root)  # fail fast if the repository itself is incomplete
        session_prov = collect_session_provenance(repo_root, args)
        write_json(log_root / f"{stamp}__{args.mode}__provenance.json", session_prov)

        seeds = list(range(int(args.seed_start), int(args.seed_stop)))
        if not seeds:
            raise SystemExit("Seed range is empty.")

        if args.mode == "audit_existing":
            out_dir = repo_root / "runs" / "stageE_clipped_robustness_extension" / "audit_existing"
            write_audit_outputs(repo_root, out_dir)
            print(f"[Stage E] wrote audit outputs to {out_dir}")
            return

        if args.mode == "reproduce_baseline":
            specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_reproduce_baseline(args)]
            out = run_phase(repo_root, "reproduce_baseline", specs, seeds, args.force_repatch, args.force_rerun, args=args)
            summary_path = out / "baseline_validation_summary.json"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
                print(f"[Stage E] reproduce_baseline overall_reproduction_pass={summary.get('overall_reproduction_pass')}")
                print(f"[Stage E] reproduce_baseline failing_conditions={summary.get('failing_conditions')}")
            print(f"[Stage E] reproduce_baseline complete: {out}")
            return

        if args.mode == "baseline_4m":
            ensure_validated_reproduction(repo_root, args, seeds)
            specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_baseline_4m(args)]
            out = run_phase(repo_root, "baseline_4m", specs, seeds, args.force_repatch, args.force_rerun, args=args)
            print(f"[Stage E] baseline_4m complete: {out}")
            return

        if args.mode == "rescue_short":
            ensure_validated_reproduction(repo_root, args, seeds)
            initial_specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_rescue_short(args, include_sweep=False)]
            out = run_phase(repo_root, "rescue_short", initial_specs, seeds, args.force_repatch, args.force_rerun, args=args)
            by_condition = pd.read_csv(out / "phase_endpoint_summary_by_condition.csv")
            if not args.skip_short_sweep and rescue_is_clear(by_condition):
                print("[Stage E] rescue is clear; running optional short local sweep")
                sweep_specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_rescue_short(args, include_sweep=True)]
                out = run_phase(repo_root, "rescue_short", sweep_specs, seeds, args.force_repatch, args.force_rerun, args=args)
            print(f"[Stage E] rescue_short complete: {out}")
            return

        if args.mode == "extension_full":
            ensure_validated_reproduction(repo_root, args, seeds)
            specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_extension_full(args)]
            out = run_phase(repo_root, "extension_full", specs, seeds, args.force_repatch, args.force_rerun, args=args)
            print(f"[Stage E] extension_full complete: {out}")
            return

        if args.mode == "all":
            audit_dir = repo_root / "runs" / "stageE_clipped_robustness_extension" / "audit_existing"
            write_audit_outputs(repo_root, audit_dir)
            print(f"[Stage E] audit_existing complete: {audit_dir}")

            reproduce_specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_reproduce_baseline(args)]
            reproduce_dir = run_phase(repo_root, "reproduce_baseline", reproduce_specs, seeds, args.force_repatch, args.force_rerun, args=args)
            print(f"[Stage E] reproduce_baseline complete: {reproduce_dir}")

            summary_path = reproduce_dir / "baseline_validation_summary.json"
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            print(f"[Stage E] reproduce_baseline overall_reproduction_pass={summary.get('overall_reproduction_pass')}")
            print(f"[Stage E] reproduce_baseline failing_conditions={summary.get('failing_conditions')}")
            if not bool(summary.get("overall_reproduction_pass", False)) and not args.allow_unvalidated_baseline:
                raise SystemExit(
                    "Baseline reproduction gate failed during all-mode. "
                    "Stopping before E0C/E1/E2 to preserve academic validity."
                )

            baseline4m_specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_baseline_4m(args)]
            baseline4m_dir = run_phase(repo_root, "baseline_4m", baseline4m_specs, seeds, args.force_repatch, args.force_rerun, args=args)
            print(f"[Stage E] baseline_4m complete: {baseline4m_dir}")

            rescue_specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_rescue_short(args, include_sweep=False)]
            rescue_dir = run_phase(repo_root, "rescue_short", rescue_specs, seeds, args.force_repatch, args.force_rerun, args=args)
            by_condition = pd.read_csv(rescue_dir / "phase_endpoint_summary_by_condition.csv")
            if not args.skip_short_sweep and rescue_is_clear(by_condition):
                print("[Stage E] rescue is clear; running optional short local sweep")
                rescue_specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_rescue_short(args, include_sweep=True)]
                rescue_dir = run_phase(repo_root, "rescue_short", rescue_specs, seeds, args.force_repatch, args.force_rerun, args=args)
            print(f"[Stage E] rescue_short complete: {rescue_dir}")

            extension_specs = [replace(s, checkpoint_eval_every=int(args.checkpoint_eval_every)) for s in specs_for_extension_full(args)]
            extension_dir = run_phase(repo_root, "extension_full", extension_specs, seeds, args.force_repatch, args.force_rerun, args=args)
            print(f"[Stage E] extension_full complete: {extension_dir}")
            return

        raise SystemExit(f"Unsupported mode {args.mode!r}")


if __name__ == "__main__":
    main()