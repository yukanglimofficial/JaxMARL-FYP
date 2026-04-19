#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import types
from dataclasses import dataclass, asdict
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "n3_stageE_candidate_interventions_v3_groupbyfix"

# Fixed N=3 / Stage-E defaults
NUM_ENVS = 25
NUM_STEPS = 128
M_EVAL = 100
E0 = 1.0
E_MAX = 1.0
BETA = float(np.log(10.0))
Z_MAX = 10.0
EPS_NORM = 1e-8
D0 = 0.733348
R0 = -27.256548
ENV_NAME = "MPE_simple_spread_v3"
ENV_KWARGS = {
    "num_agents": 3,
    "num_landmarks": 3,
    "local_ratio": 0.5,
    "max_steps": 25,
    "action_type": "Discrete",
}
DEFAULT_TIMESTEPS = 4_000_000
DEFAULT_SEED_START = 30
DEFAULT_SEED_STOP = 62
DEFAULT_LATE_FRAC = 0.25
DEFAULT_ALPHA_STEP = 0.003
DEFAULT_ALPHA_MIN = 0.0
DEFAULT_ALPHA_MAX = 0.108
DEFAULT_SUPPORT_MODE = "frontload"
DEFAULT_SUPPORT_TOTAL = 0.60
DEFAULT_SUPPORT_START_UPDATE = 0
DEFAULT_SUPPORT_WINDOW_FRAC = 0.10
DEFAULT_BORROW_LIMIT = 0.60
DEFAULT_BORROW_INTEREST = 0.02

EXCLUDED_SEARCH_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}
REPO_ROOT_MARKERS = [
    Path("baselines") / "IPPO" / "ippo_ff_mpe_gate5_energy_gated.py",
    Path("n3_engineered_phase_transition.py"),
    Path("stageE_clipped_robustness_extension.py"),
]

TRACE_ALIASES = {
    "update": ["update"],
    "update_u": ["update/u"],
    "update_counter": ["update/update_counter"],
    "energy_E_pre": ["energy/E_pre"],
    "energy_E_post": ["energy/E_post"],
    "energy_E_raw": ["energy/E_raw"],
    "funding_z": ["funding/z"],
    "funding_g": ["funding/g"],
    "funding_g_raw": ["funding/g_raw"],
    "funding_cut_active": ["funding/cut_active"],
    "funding_income": ["funding/income"],
    "funding_alpha_eff": ["funding/alpha_eff"],
    "support_s": ["support/s"],
    "debt_charge": ["debt/charge"],
    "debt_level_pre": ["debt/level_pre"],
    "debt_level_post": ["debt/level_post"],
    "episode_return_end_mean": ["episode/return_end_mean"],
    "terminal_min_dists_end_mean": ["terminal/min_dists_end_mean"],
    "terminal_dbar_end": ["terminal/dbar_end"],
    "terminal_occupied_end_mean": ["terminal/occupied_end_mean"],
    "terminal_CSR_train": ["terminal/CSR_train"],
    "vartd_VarTD": ["vartd/VarTD"],
    "depr_F": ["depr/F"],
    "depr_F_post": ["depr/F_post"],
}

EVAL_ALIASES = {
    "eval_done_count": ["done_count"],
    "eval_return_end_mean": ["return_end_mean"],
    "eval_len_end_mean": ["len_end_mean"],
    "eval_CSR_post": ["CSR_post", "csr_post", "csr"],
    "eval_dbar_post_mean": ["dbar_post_mean", "dbar_post", "dbar_end"],
    "eval_occupied_post_mean": ["occupied_post_mean", "occupied_end_mean"],
    "eval_episodes_per_env": ["episodes_per_env"],
    "eval_t_eval": ["t_eval"],
}

HELPER_INSERT = '''

def hard_zero_filter(g_k, z_k, hard_zero_funding: bool, z_cut: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if hard_zero_funding:
        cut_active = z_k <= z_cut
        g_eff = jnp.where(cut_active, g_k, 0.0).astype(jnp.float32)
    else:
        cut_active = jnp.asarray(True)
        g_eff = g_k.astype(jnp.float32)
    return cut_active.astype(jnp.float32), g_eff.astype(jnp.float32)


def support_from_schedule(
    k: jnp.ndarray,
    num_updates: int,
    total_support: jnp.ndarray,
    mode: str,
    start_update: int,
    window_updates: int,
) -> jnp.ndarray:
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


@dataclass(frozen=True)
class InterventionFamilySpec:
    family: str
    support_mode: str = "none"
    support_total: float = 0.0
    support_start_update: int = 0
    support_window_frac: float = DEFAULT_SUPPORT_WINDOW_FRAC
    borrow_limit: float = 0.0
    borrow_interest: float = 0.0

    def tag(self) -> str:
        if self.family == "base":
            return "base"
        if self.family == "support":
            return f"support__{self.support_mode}__tot{self.support_total:.2f}".replace(".", "p")
        if self.family == "borrow":
            return f"borrow__B{self.borrow_limit:.2f}__r{self.borrow_interest:.2f}".replace(".", "p")
        return self.family


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


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (Path, PathLike)):
        return os.fspath(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NA:
        return None
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, 'item'):
        try:
            return _jsonable(obj.item())
        except Exception:
            pass
    if hasattr(obj, 'isoformat'):
        try:
            return obj.isoformat()
        except Exception:
            pass
    return str(obj)


def dumps_json(obj: Any, *, indent: int = 2, sort_keys: bool = True) -> str:
    return json.dumps(_jsonable(obj), indent=indent, sort_keys=sort_keys)


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


INTERVENTION_TRACE_METADATA_COLS = {"mechanism_tag", "mechanism", "alpha", "family", "seed", "update"}


def aggregate_trace_df_local(trace_df: pd.DataFrame) -> pd.DataFrame:
    if "update" not in trace_df.columns:
        raise ValueError("trace_df missing required column 'update'")
    metric_cols: List[str] = []
    for c in trace_df.columns:
        if c in INTERVENTION_TRACE_METADATA_COLS:
            continue
        if pd.api.types.is_numeric_dtype(trace_df[c]):
            metric_cols.append(c)
            continue
        converted = pd.to_numeric(trace_df[c], errors="coerce")
        if converted.notna().any():
            metric_cols.append(c)
    if not metric_cols:
        raise ValueError("No numeric intervention trace columns available for aggregation.")
    work = trace_df.copy()
    for c in metric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    agg = work.groupby("update")[metric_cols].agg(["mean", q25, q75, "std"]).reset_index()
    out = flatten_multiindex_columns(agg)
    if "env_steps_mean" not in out.columns and "env_steps" in work.columns:
        out["env_steps_mean"] = work.groupby("update")["env_steps"].mean().to_numpy(dtype=float)
    return out


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
    return all((path / marker).exists() for marker in REPO_ROOT_MARKERS[:1])


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
    raise FileNotFoundError(f"Could not infer repo root. Search starts were: {searched}")


def load_module_from_path(name: str, path: Path, repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def support_window_updates(total_timesteps: int, frac: float) -> int:
    num_updates = int(total_timesteps) // (NUM_ENVS * NUM_STEPS)
    return max(1, int(round(num_updates * float(frac))))


def find_script(repo_root: Path, explicit: Optional[str], default_name: str) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if p.exists():
            return p
    candidate = repo_root / default_name
    if candidate.exists():
        return candidate.resolve()
    hits = [p for p in repo_root.rglob(default_name) if p.is_file() and not any(part in EXCLUDED_SEARCH_PARTS for part in p.parts)]
    if hits:
        return hits[0].resolve()
    raise FileNotFoundError(f"Could not find {default_name} under {repo_root}")


def locate_finalize_artifacts(repo_root: Path, selected_attempt_json: Optional[str]) -> Path:
    if selected_attempt_json is not None:
        p = Path(selected_attempt_json)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if p.exists():
            return p
    default = repo_root / "runs" / "n3_locked_candidate_transition_finalize" / "selected_attempt.json"
    if default.exists():
        return default.resolve()
    raise FileNotFoundError("Could not locate finalized selected_attempt.json. Pass --selected-attempt-json.")


def parse_alpha_values(value: Any) -> List[float]:
    if isinstance(value, list):
        vals = value
    elif isinstance(value, str):
        vals = json.loads(value)
    else:
        raise ValueError(f"Unsupported alpha-values payload: {type(value)}")
    return sorted({round(float(x), 3) for x in vals})


def build_intervention_alpha_grid(selected_payload: Mapping[str, Any], source_summary: pd.DataFrame, alpha_min: float, alpha_max: float, alpha_step: float) -> List[float]:
    vals: set[float] = set()
    if "alpha_values" in selected_payload:
        vals.update(parse_alpha_values(selected_payload["alpha_values"]))
    if "certify_alpha_values" in selected_payload:
        vals.update(parse_alpha_values(selected_payload["certify_alpha_values"]))
    if "alpha" in source_summary.columns:
        vals.update(round(float(x), 3) for x in pd.to_numeric(source_summary["alpha"], errors="coerce").dropna().to_numpy(dtype=float))
    lo = float(alpha_min)
    hi = float(alpha_max)
    x = lo
    while x <= hi + 1e-12:
        vals.add(round(x, 3))
        x += float(alpha_step)
    return sorted(vals)


def patch_master_source(src: str) -> str:
    marker = "return z_k.astype(jnp.float32), g_k.astype(jnp.float32)\n\n\ndef vartd_path_a("
    if marker not in src:
        raise RuntimeError("Could not find funding_from_return marker in accepted master script.")
    src = src.replace(marker, "return z_k.astype(jnp.float32), g_k.astype(jnp.float32)\n" + HELPER_INSERT + "\n\ndef vartd_path_a(", 1)

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
    config.setdefault("BORROW_LIMIT", 0.0)
    config.setdefault("BORROW_INTEREST", 0.0)
    config.setdefault("SUPPORT_MODE", "none")
    config.setdefault("SUPPORT_TOTAL", 0.0)
    config.setdefault("SUPPORT_START_UPDATE", 0)
    config.setdefault("SUPPORT_WINDOW_UPDATES", 1)
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
    borrow_limit = jnp.asarray(config.get("BORROW_LIMIT", 0.0), dtype=jnp.float32)
    borrow_interest = jnp.asarray(config.get("BORROW_INTEREST", 0.0), dtype=jnp.float32)
    support_mode = str(config.get("SUPPORT_MODE", "none"))
    support_total = jnp.asarray(config.get("SUPPORT_TOTAL", 0.0), dtype=jnp.float32)
    support_start_update = int(config.get("SUPPORT_START_UPDATE", 0))
    support_window_updates = int(config.get("SUPPORT_WINDOW_UPDATES", 1))
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

    old_gate = '''            income = alpha * g_k
            e_pre = energy
            u_bool = e_pre >= c_upd
            u_f = u_bool.astype(jnp.float32)
'''
    new_gate = '''            cut_active, g_eff = hard_zero_filter(g_k, z_k, hard_zero_funding, z_cut)
            alpha_eff = alpha * jnp.exp(-depr_psi * depr_F)
            support_k = support_from_schedule(
                k=jnp.asarray(k, dtype=jnp.int32),
                num_updates=int(config["NUM_UPDATES"]),
                total_support=support_total,
                mode=support_mode,
                start_update=support_start_update,
                window_updates=support_window_updates,
            )
            income = alpha_eff * g_eff + support_k
            e_pre = energy
            debt_level_pre = jnp.maximum(-e_pre, 0.0)
            debt_charge = borrow_interest * debt_level_pre
            u_bool = (e_pre + borrow_limit) >= c_upd
            u_f = u_bool.astype(jnp.float32)
'''
    if old_gate not in src:
        raise RuntimeError("Could not find gate block.")
    src = src.replace(old_gate, new_gate, 1)

    old_energy = '            e_post = jnp.clip(e_pre - c_upd * u_f + income, 0.0, e_max).astype(jnp.float32)\n'
    new_energy = '''            e_raw = e_pre - c_upd * u_f + income - debt_charge
            e_post = jnp.clip(e_raw, -borrow_limit, e_max).astype(jnp.float32)
            debt_level_post = jnp.maximum(-e_post, 0.0).astype(jnp.float32)
            depr_F_post = jnp.maximum((1.0 - depr_nu * u_f) * depr_F + depr_lambda * (1.0 - u_f), 0.0).astype(jnp.float32)
'''
    if old_energy not in src:
        raise RuntimeError("Could not find energy update line.")
    src = src.replace(old_energy, new_energy, 1)

    old_metric_line = '                "funding/g": g_k.astype(jnp.float32),\n'
    new_metric_line = '''                "funding/g": g_eff.astype(jnp.float32),
                "funding/g_raw": g_k.astype(jnp.float32),
                "funding/cut_active": cut_active.astype(jnp.float32),
                "funding/alpha_eff": alpha_eff.astype(jnp.float32),
                "support/s": support_k.astype(jnp.float32),
                "debt/charge": debt_charge.astype(jnp.float32),
                "debt/level_pre": debt_level_pre.astype(jnp.float32),
                "debt/level_post": debt_level_post.astype(jnp.float32),
                "energy/E_raw": e_raw.astype(jnp.float32),
                "depr/F": depr_F.astype(jnp.float32),
                "depr/F_post": depr_F_post.astype(jnp.float32),
'''
    if old_metric_line not in src:
        raise RuntimeError("Could not find funding/g metric line.")
    src = src.replace(old_metric_line, new_metric_line, 1)

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


def write_intervention_master(repo_root: Path, force: bool = False) -> Path:
    accepted = repo_root / "baselines" / "IPPO" / "ippo_ff_mpe_gate5_energy_gated.py"
    if not accepted.exists():
        raise FileNotFoundError(f"Accepted master script not found: {accepted}")
    derived = repo_root / "baselines" / "IPPO" / "ippo_ff_mpe_n3_transition_with_stagee_interventions.py"
    patched = patch_master_source(accepted.read_text(encoding="utf-8"))
    if derived.exists() and not force and derived.read_text(encoding="utf-8") == patched:
        return derived
    derived.write_text(patched, encoding="utf-8")
    return derived


def build_config(base_module: Any, *, alpha: float, mechanism: Any, family: InterventionFamilySpec, total_timesteps: int) -> Dict[str, Any]:
    cfg = base_module.build_config(alpha=float(alpha), mechanism=mechanism, total_timesteps=int(total_timesteps))
    cfg.update(
        {
            "BORROW_LIMIT": float(family.borrow_limit),
            "BORROW_INTEREST": float(family.borrow_interest),
            "SUPPORT_MODE": str(family.support_mode),
            "SUPPORT_TOTAL": float(family.support_total),
            "SUPPORT_START_UPDATE": int(family.support_start_update),
            "SUPPORT_WINDOW_UPDATES": support_window_updates(int(total_timesteps), float(family.support_window_frac)),
        }
    )
    cfg["PROJECT"] = "JaxMARL_N3_PhaseTransition_StageEInterventions"
    return cfg


def extract_trace_df(metrics_np: Mapping[str, np.ndarray], *, alpha: float, mechanism: Any, family: InterventionFamilySpec, seed: int) -> pd.DataFrame:
    data: Dict[str, np.ndarray] = {}
    length: Optional[int] = None
    for out_key, aliases in TRACE_ALIASES.items():
        raw_key = None
        for a in aliases:
            if a in metrics_np:
                raw_key = a
                break
        if raw_key is None:
            continue
        arr = np.asarray(metrics_np[raw_key])
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D trace for {raw_key}, got {arr.shape}")
        if length is None:
            length = int(arr.shape[0])
        if arr.shape[0] != length:
            raise ValueError(f"Mismatched trace length for {raw_key}: {arr.shape[0]} vs {length}")
        data[out_key] = arr
    if length is None:
        raise RuntimeError("No metric traces extracted from intervention master output.")
    if "update" not in data:
        data["update"] = np.arange(length, dtype=int)
    df = pd.DataFrame(data)
    df.insert(0, "seed", int(seed))
    df.insert(0, "family", family.family)
    df.insert(0, "alpha", float(alpha))
    df.insert(0, "mechanism_tag", mechanism.tag())
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


def extract_eval_row(eval_np: Mapping[str, np.ndarray], trace_df: pd.DataFrame, *, alpha: float, mechanism: Any, family: InterventionFamilySpec, seed: int, runtime_sec: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "alpha": float(alpha),
        "family": family.family,
        "mechanism": mechanism.mechanism_name,
        "mechanism_tag": mechanism.tag(),
        "c_upd": float(mechanism.c_upd),
        "z_cut": float(mechanism.z_cut),
        "depr_lambda": float(mechanism.depr_lambda),
        "depr_nu": float(mechanism.depr_nu),
        "depr_psi": float(mechanism.depr_psi),
        "seed": int(seed),
        "runtime_sec": float(runtime_sec),
        "support_mode": family.support_mode,
        "support_total": float(family.support_total),
        "borrow_limit": float(family.borrow_limit),
        "borrow_interest": float(family.borrow_interest),
    }
    for out_key, aliases in EVAL_ALIASES.items():
        raw_key = None
        for a in aliases:
            if a in eval_np:
                raw_key = a
                break
        if raw_key is None:
            continue
        scalar = _scalar_from_array(eval_np[raw_key])
        if scalar is not None:
            row[out_key] = scalar
    if "update_u" in trace_df.columns:
        row["mean_update_fraction"] = float(pd.to_numeric(trace_df["update_u"], errors="coerce").mean())
    if "update_counter" in trace_df.columns:
        row["final_update_counter"] = float(pd.to_numeric(trace_df["update_counter"], errors="coerce").iloc[-1])
    if "energy_E_post" in trace_df.columns:
        row["final_energy_E_post"] = float(pd.to_numeric(trace_df["energy_E_post"], errors="coerce").iloc[-1])
    if "depr_F_post" in trace_df.columns:
        row["final_depr_F_post"] = float(pd.to_numeric(trace_df["depr_F_post"], errors="coerce").iloc[-1])
    if "vartd_VarTD" in trace_df.columns:
        row["final_vartd_VarTD"] = float(pd.to_numeric(trace_df["vartd_VarTD"], errors="coerce").iloc[-1])
    if "support_s" in trace_df.columns:
        row["total_support_s"] = float(pd.to_numeric(trace_df["support_s"], errors="coerce").fillna(0.0).sum())
    if "debt_charge" in trace_df.columns:
        row["total_debt_charge"] = float(pd.to_numeric(trace_df["debt_charge"], errors="coerce").fillna(0.0).sum())
    if "debt_level_post" in trace_df.columns:
        row["max_debt_level_post"] = float(pd.to_numeric(trace_df["debt_level_post"], errors="coerce").max())
    return row


def run_one_alpha(mod: Any, base_module: Any, *, alpha: float, mechanism: Any, family: InterventionFamilySpec, seeds: Sequence[int], out_dir: Path, total_timesteps: int, force_rerun: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import jax

    ensure_dir(out_dir)
    cfg = build_config(base_module, alpha=float(alpha), mechanism=mechanism, family=family, total_timesteps=int(total_timesteps))
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
        trace_df = extract_trace_df(metrics_np, alpha=float(alpha), mechanism=mechanism, family=family, seed=int(seed))
        eval_row = extract_eval_row(eval_np, trace_df, alpha=float(alpha), mechanism=mechanism, family=family, seed=int(seed), runtime_sec=float(runtime_sec))

        trace_df.to_csv(trace_path, index=False)
        eval_path.write_text(dumps_json(eval_row, indent=2, sort_keys=True), encoding="utf-8")

        all_traces.append(trace_df)
        endpoint_rows.append(eval_row)
        del out, metrics_device, eval_device, metrics_np, eval_np, trace_df
        gc.collect()

    endpoint_df = coerce_numeric(pd.DataFrame(endpoint_rows))
    endpoint_df.to_csv(out_dir / "seed_endpoint_summary.csv", index=False)

    trace_long = pd.concat(all_traces, ignore_index=True)
    trace_long = coerce_numeric(trace_long)
    trace_long.to_csv(out_dir / "trace_per_seed_long.csv", index=False)

    trace_agg = aggregate_trace_df_local(trace_long)
    trace_agg.to_csv(out_dir / "trace_aggregate.csv", index=False)

    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    for metric, title, ylabel in getattr(base_module, "PLOT_METRICS"):
        base_module.save_band_plot(trace_agg, metric, f"{family.family} alpha={alpha:.3f}: {title}", ylabel, plots_dir / f"{metric}.png")

    return endpoint_df, trace_agg


def run_family_alpha_sweep(repo_root: Path, base_module: Any, mechanism: Any, family: InterventionFamilySpec, alpha_values: Sequence[float], total_timesteps: int, seed_start: int, seed_stop: int, late_frac: float, out_dir: Path, force_repatch: bool, force_rerun: bool) -> pd.DataFrame:
    repo_root = infer_repo_root(repo_root)
    install_wandb_stub()
    derived_master = write_intervention_master(repo_root, force=bool(force_repatch))
    mod = load_module_from_path(f"n3_candidate_intervention_{mechanism.tag()}_{family.tag()}", derived_master, repo_root)
    seeds = list(range(int(seed_start), int(seed_stop)))
    if not seeds:
        raise ValueError("Empty seed range.")
    ensure_dir(out_dir)

    summary_rows: List[Dict[str, Any]] = []
    for alpha in [float(a) for a in alpha_values]:
        alpha_dir = out_dir / f"alpha_{alpha:g}"
        endpoint_df, _ = run_one_alpha(mod, base_module, alpha=float(alpha), mechanism=mechanism, family=family, seeds=seeds, out_dir=alpha_dir, total_timesteps=int(total_timesteps), force_rerun=bool(force_rerun))
        trace_long = coerce_numeric(pd.read_csv(alpha_dir / "trace_per_seed_long.csv"))
        seed_late_df = base_module.compute_seed_late_metrics(trace_long, late_frac=float(late_frac))
        seed_late_df["alpha"] = float(alpha)
        seed_late_df["family"] = family.family
        seed_late_df.to_csv(alpha_dir / "seed_late_activity_metrics.csv", index=False)
        phase_summary = base_module.aggregate_alpha_metrics(seed_late_df, endpoint_df)
        phase_summary.update(
            {
                "alpha": float(alpha),
                "family": family.family,
                "mechanism": mechanism.mechanism_name,
                "mechanism_tag": mechanism.tag(),
                "c_upd": float(mechanism.c_upd),
                "z_cut": float(mechanism.z_cut),
                "depr_lambda": float(mechanism.depr_lambda),
                "depr_nu": float(mechanism.depr_nu),
                "depr_psi": float(mechanism.depr_psi),
                "support_mode": family.support_mode,
                "support_total": float(family.support_total),
                "support_start_update": int(family.support_start_update),
                "support_window_frac": float(family.support_window_frac),
                "borrow_limit": float(family.borrow_limit),
                "borrow_interest": float(family.borrow_interest),
                "total_timesteps": int(total_timesteps),
            }
        )
        summary_rows.append(phase_summary)

    summary_df = coerce_numeric(pd.DataFrame(summary_rows).sort_values("alpha"))
    summary_df.to_csv(out_dir / "alpha_sweep_phase_summary.csv", index=False)
    candidate = base_module.evaluate_candidate(summary_df, mechanism)
    candidate.update(
        {
            "family": family.family,
            "support_mode": family.support_mode,
            "support_total": float(family.support_total),
            "borrow_limit": float(family.borrow_limit),
            "borrow_interest": float(family.borrow_interest),
        }
    )
    # candidate-local interpretive label
    left_ps = float(candidate.get("left_edge_p_sustained", np.nan))
    right_ps = float(candidate.get("right_edge_p_sustained", np.nan))
    if int(candidate.get("candidate_pass", 0)) == 1:
        candidate["transition_status"] = "strict_resolved_on_grid"
    elif np.isfinite(left_ps) and np.isfinite(right_ps) and left_ps > 0.10 and right_ps >= 0.90:
        candidate["transition_status"] = "left_shifted_below_grid_or_overactivated"
    elif np.isfinite(left_ps) and np.isfinite(right_ps) and left_ps <= 0.10 and right_ps < 0.90:
        candidate["transition_status"] = "right_shifted_above_grid_or_suppressed"
    else:
        candidate["transition_status"] = "unresolved_or_blunted"
    (out_dir / "candidate_metrics.json").write_text(dumps_json(candidate, indent=2, sort_keys=True), encoding="utf-8")
    return summary_df


def family_overlay_plot(summary_by_family: Mapping[str, pd.DataFrame], col: str, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(8.4, 5.0))
    plotted = False
    for family, sdf in summary_by_family.items():
        if col not in sdf.columns:
            continue
        x = pd.to_numeric(sdf["alpha"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sdf[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() == 0:
            continue
        plt.plot(x[ok], y[ok], linewidth=2.0, label=family)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel(r"$\alpha$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return centre - half, centre + half


def p_sustained_wilson_panels(summary_by_family: Mapping[str, pd.DataFrame], out_path: Path) -> None:
    fams = list(summary_by_family.keys())
    if not fams:
        return
    fig, axes = plt.subplots(len(fams), 1, figsize=(8.4, 3.2 * len(fams)), sharex=True)
    if len(fams) == 1:
        axes = [axes]
    for ax, family in zip(axes, fams):
        sdf = summary_by_family[family].sort_values("alpha")
        x = pd.to_numeric(sdf["alpha"], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(sdf["p_sustained_mean"], errors="coerce").to_numpy(dtype=float)
        n = int(pd.to_numeric(sdf["n_seeds"], errors="coerce").dropna().iloc[0]) if "n_seeds" in sdf.columns and not sdf.empty else 0
        lo: List[float] = []
        hi: List[float] = []
        for val in p:
            if np.isfinite(val) and n > 0:
                l, h = wilson_interval(int(round(val * n)), n)
            else:
                l, h = (np.nan, np.nan)
            lo.append(l)
            hi.append(h)
        ok = np.isfinite(x) & np.isfinite(p)
        if ok.sum() > 0:
            ax.plot(x[ok], p[ok], linewidth=2.0, label=family)
            lo_arr = np.asarray(lo, dtype=float)
            hi_arr = np.asarray(hi, dtype=float)
            q = ok & np.isfinite(lo_arr) & np.isfinite(hi_arr)
            if q.sum() > 0:
                ax.fill_between(x[q], lo_arr[q], hi_arr[q], alpha=0.25)
        ax.set_ylabel("P_sustained")
        ax.set_title(f"{family}: sustained-active fraction with 95% Wilson interval")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(r"$\alpha$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def late_update_boxplot_by_family(summary_by_family: Mapping[str, pd.DataFrame], family_dirs: Mapping[str, Path], out_path: Path) -> None:
    labels: List[str] = []
    data: List[np.ndarray] = []
    for family, sdf in summary_by_family.items():
        # choose nearest alpha to the family transition guess if available, else mid alpha
        alpha_guess = float(sdf.get("alpha", pd.Series(dtype=float)).iloc[len(sdf)//2])
        cand_path = family_dirs[family] / "candidate_metrics.json"
        if cand_path.exists():
            cand = json.loads(cand_path.read_text(encoding="utf-8"))
            if np.isfinite(float(cand.get("alpha_transition_guess", np.nan))):
                alpha_guess = float(cand["alpha_transition_guess"])
        alphas = pd.to_numeric(sdf["alpha"], errors="coerce").to_numpy(dtype=float)
        if alphas.size == 0:
            continue
        alpha_pick = float(alphas[np.nanargmin(np.abs(alphas - alpha_guess))])
        path = family_dirs[family] / f"alpha_{alpha_pick:g}" / "seed_late_activity_metrics.csv"
        if not path.exists():
            continue
        seed_df = coerce_numeric(pd.read_csv(path))
        arr = pd.to_numeric(seed_df["late_update_count_seed"], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        labels.append(f"{family}\nα={alpha_pick:.3f}")
        data.append(arr)
    if not data:
        return
    plt.figure(figsize=(max(8.0, 1.8 * len(data)), 5.0))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.ylabel("Late-window executed updates per seed")
    plt.title("Per-family late-update-count distributions near each family's transition region")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def choose_family_trace_alpha(candidate: Mapping[str, Any], summary_df: pd.DataFrame) -> float:
    guess = float(candidate.get("alpha_transition_guess", np.nan))
    alphas = pd.to_numeric(summary_df["alpha"], errors="coerce").dropna().to_numpy(dtype=float)
    if alphas.size == 0:
        return float("nan")
    if np.isfinite(guess):
        return float(alphas[np.nanargmin(np.abs(alphas - guess))])
    p = pd.to_numeric(summary_df.get("p_sustained_mean", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(p).any():
        idx = int(np.nanargmin(np.abs(p - 0.5)))
        return float(alphas[idx])
    return float(alphas[len(alphas)//2])


def save_family_trace_plots(summary_by_family: Mapping[str, pd.DataFrame], family_dirs: Mapping[str, Path], out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ensure_dir(out_dir)
    for family, sdf in summary_by_family.items():
        cand_path = family_dirs[family] / "candidate_metrics.json"
        if not cand_path.exists():
            continue
        cand = json.loads(cand_path.read_text(encoding="utf-8"))
        alpha = choose_family_trace_alpha(cand, sdf)
        if not np.isfinite(alpha):
            continue
        alpha_dir = family_dirs[family] / f"alpha_{alpha:g}"
        seed_late_path = alpha_dir / "seed_late_activity_metrics.csv"
        if not seed_late_path.exists():
            continue
        seed_late_df = coerce_numeric(pd.read_csv(seed_late_path))
        if seed_late_df.empty:
            continue
        # representative = seed closest to family median late activity
        vals = pd.to_numeric(seed_late_df["late_update_count_seed"], errors="coerce")
        vals = vals.astype(float)
        target = float(np.nanmedian(vals.to_numpy(dtype=float))) if np.isfinite(vals).any() else 0.0
        idx = int(np.nanargmin(np.abs(vals.to_numpy(dtype=float) - target)))
        rep_seed = int(seed_late_df.iloc[idx]["seed"])
        trace_path = alpha_dir / f"seed_{rep_seed:02d}" / "trace_per_update.csv"
        if not trace_path.exists():
            continue
        trace_df = coerce_numeric(pd.read_csv(trace_path))
        x = pd.to_numeric(trace_df["update"], errors="coerce").to_numpy(dtype=float)
        fig, axes = plt.subplots(6, 1, figsize=(9.0, 12.5), sharex=True)
        metrics = [
            ("update_u", "u"),
            ("energy_E_post", "energy"),
            ("funding_g", "g*"),
            ("support_s", "support"),
            ("debt_level_post", "debt"),
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
        fig.suptitle(f"Representative intervention trace: {family}, alpha={alpha:.3f}, seed={rep_seed:02d}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        out_path = out_dir / f"representative_trace_{family}_alpha_{alpha:.3f}_seed_{rep_seed:02d}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        rows.append({"family": family, "alpha": float(alpha), "seed": int(rep_seed), "path": str(out_path.name)})
    return rows


def write_report(out_root: Path, selected_attempt: Mapping[str, Any], family_rows: pd.DataFrame, alpha_values: Sequence[float], graph_dir: Path, reps: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "N=3 finalized-candidate Stage-E intervention bridge report",
        f"script_version={SCRIPT_VERSION}",
        "",
        "This is a post-selection perturbation analysis of the finalized engineered N=3 crossover candidate.",
        "It is intended to connect the accepted Stage-E intervention semantics to the finalized hard-zero candidate.",
        "It should be labeled exploratory/post hoc rather than confirmatory discovery.",
        "",
        "Locked finalized candidate",
    ]
    for key in ["attempt_name", "c_upd", "z_cut", "depr_lambda", "depr_psi", "raw_alpha_transition_lo", "raw_alpha_transition_hi", "raw_alpha_transition_guess", "certification_label"]:
        if key in selected_attempt:
            lines.append(f"{key}={selected_attempt[key]}")
    lines.append(f"alpha_grid={list(alpha_values)}")
    lines.append("")
    lines.append("Family summary")
    for _, row in family_rows.iterrows():
        lines.append(
            " ".join(
                [
                    f"family={row['family']}",
                    f"candidate_pass={int(row.get('candidate_pass', 0))}",
                    f"transition_status={row.get('transition_status')}",
                    f"alpha_transition_lo={fmt(row.get('alpha_transition_lo'))}",
                    f"alpha_transition_hi={fmt(row.get('alpha_transition_hi'))}",
                    f"alpha_transition_guess={fmt(row.get('alpha_transition_guess'))}",
                    f"left_p_sustained={fmt(row.get('left_edge_p_sustained'))}",
                    f"right_p_sustained={fmt(row.get('right_edge_p_sustained'))}",
                    f"left_late_updates={fmt(row.get('left_edge_late_update_count_mean'))}",
                    f"right_late_updates={fmt(row.get('right_edge_late_update_count_mean'))}",
                    f"chi_peak_ratio={fmt(row.get('chi_peak_ratio'))}",
                    f"p_sustained_violations={int(row.get('p_sustained_violations', 0))}",
                ]
            )
        )
    lines.append("")
    lines.append("Graph package")
    lines.append(f"graph_dir={graph_dir}")
    lines.append("Headline graph: p_sustained_vs_alpha_family_overlay.png")
    lines.append("This graph shows whether support or borrowing shift, sharpen, blunt, or saturate the engineered crossover.")
    lines.append("")
    lines.append("Representative traces")
    for row in reps:
        lines.append(f"family={row['family']} alpha={row['alpha']} seed={row['seed']} file={row['path']}")
    (out_root / "INTERVENTION_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_provenance(repo_root: Path, base_script: Path, stagee_script: Path, selected_attempt_json: Path, derived_master: Path, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "repo_root": str(repo_root),
        "base_script_path": str(base_script.resolve()),
        "base_script_sha256": sha256_file(base_script),
        "stagee_script_path": str(stagee_script.resolve()),
        "stagee_script_sha256": sha256_file(stagee_script),
        "selected_attempt_json_path": str(selected_attempt_json.resolve()),
        "selected_attempt_json_sha256": sha256_file(selected_attempt_json),
        "derived_master_path": str(derived_master.resolve()),
        "derived_master_sha256": sha256_file(derived_master),
        "python_version": sys.version,
        "platform": platform.platform(),
        "command_argv": list(sys.argv),
        "args": vars(args),
    }


def family_dir_tag_from_row(row: Mapping[str, Any]) -> str:
    family = str(row.get("family", ""))
    if family == "base":
        return "base"
    if family == "support":
        spec = InterventionFamilySpec(
            family="support",
            support_mode=str(row.get("support_mode", DEFAULT_SUPPORT_MODE)),
            support_total=float(row.get("support_total", DEFAULT_SUPPORT_TOTAL)),
            support_start_update=int(row.get("support_start_update", DEFAULT_SUPPORT_START_UPDATE)),
            support_window_frac=float(row.get("support_window_frac", DEFAULT_SUPPORT_WINDOW_FRAC)),
        )
        return spec.tag()
    if family == "borrow":
        spec = InterventionFamilySpec(
            family="borrow",
            borrow_limit=float(row.get("borrow_limit", DEFAULT_BORROW_LIMIT)),
            borrow_interest=float(row.get("borrow_interest", DEFAULT_BORROW_INTEREST)),
        )
        return spec.tag()
    return family


def locate_family_dir(out_root: Path, row: Mapping[str, Any]) -> Path:
    exact = out_root / family_dir_tag_from_row(row)
    if exact.exists():
        return exact
    if not out_root.exists():
        return exact
    family = str(row.get("family", ""))
    preferred_prefix = family if family == "base" else f"{family}__"
    matches = sorted(
        [p for p in out_root.iterdir() if p.is_dir() and p.name.startswith(preferred_prefix)],
        key=lambda p: p.name,
    )
    if matches:
        return matches[0]
    fallback = sorted(
        [p for p in out_root.iterdir() if p.is_dir() and p.name.startswith(family)],
        key=lambda p: p.name,
    )
    if fallback:
        return fallback[0]
    return exact


def resolve_selected_summary_csv(repo_root: Path, selected_json: Path, selected_attempt: Mapping[str, Any]) -> Path:
    raw_value = str(selected_attempt.get("source_summary_csv", "")).strip()
    if not raw_value:
        raise FileNotFoundError("selected_attempt.json does not contain source_summary_csv")

    p = Path(raw_value)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
        parts = list(p.parts)
        if "runs" in parts:
            idx = parts.index("runs")
            candidates.append((repo_root / Path(*parts[idx:])).resolve())
    else:
        candidates.append((selected_json.parent / p).resolve())
        candidates.append((repo_root / p).resolve())

    seen: set[Path] = set()
    ordered: List[Path] = []
    for cand in candidates:
        try:
            rc = cand.resolve()
        except Exception:
            rc = cand
        if rc not in seen:
            ordered.append(rc)
            seen.add(rc)

    for cand in ordered:
        if cand.exists() and cand.is_file():
            return cand

    hits = sorted(
        [h.resolve() for h in repo_root.rglob(p.name) if h.is_file() and not any(part in EXCLUDED_SEARCH_PARTS for part in h.parts)],
        key=lambda q: (len(q.parts), str(q)),
    )
    if hits:
        return hits[0]

    tried = ", ".join(str(x) for x in ordered) if ordered else raw_value
    raise FileNotFoundError(f"Could not resolve source_summary_csv={raw_value!r}. Tried: {tried}")


def mode_run(args: argparse.Namespace) -> None:
    repo_root = infer_repo_root(args.repo_root)
    base_script = find_script(repo_root, args.base_script, "n3_engineered_phase_transition.py")
    stagee_script = find_script(repo_root, args.stagee_script, "stageE_clipped_robustness_extension.py")
    install_wandb_stub()
    base = load_module_from_path("_n3_intervention_base", base_script, repo_root)

    selected_json = locate_finalize_artifacts(repo_root, args.selected_attempt_json)
    selected_attempt = json.loads(selected_json.read_text(encoding="utf-8"))
    summary_csv = resolve_selected_summary_csv(repo_root, selected_json, selected_attempt)
    source_summary = coerce_numeric(pd.read_csv(summary_csv))

    # lock mechanism from finalized candidate
    c_upd = float(selected_attempt.get("raw_c_upd", selected_attempt.get("c_upd", 0.58)))
    z_cut = float(selected_attempt.get("raw_z_cut", selected_attempt.get("z_cut", 1.0)))
    depr_lambda = float(selected_attempt.get("raw_depr_lambda", selected_attempt.get("depr_lambda", 0.0)))
    depr_nu = float(selected_attempt.get("raw_depr_nu", selected_attempt.get("depr_nu", 0.5)))
    depr_psi = float(selected_attempt.get("raw_depr_psi", selected_attempt.get("depr_psi", 0.0)))
    mechanism = base.MechanismSpec(c_upd=c_upd, z_cut=z_cut, depr_lambda=depr_lambda, depr_nu=depr_nu, depr_psi=depr_psi)

    alpha_values = build_intervention_alpha_grid(selected_attempt, source_summary, float(args.alpha_min), float(args.alpha_max), float(args.alpha_step))

    out_root = repo_root / "runs" / "n3_stageE_candidate_interventions"
    ensure_dir(out_root)
    derived_master = write_intervention_master(repo_root, force=bool(args.force_repatch))
    prov = collect_provenance(repo_root, base_script, stagee_script, selected_json, derived_master, args)
    (out_root / "provenance.json").write_text(dumps_json(prov, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "alpha_grid.json").write_text(dumps_json(alpha_values, indent=2, sort_keys=False), encoding="utf-8")

    families = [
        InterventionFamilySpec(family="base"),
        InterventionFamilySpec(
            family="support",
            support_mode=str(args.support_mode),
            support_total=float(args.support_total),
            support_start_update=int(args.support_start_update),
            support_window_frac=float(args.support_window_frac),
        ),
        InterventionFamilySpec(
            family="borrow",
            borrow_limit=float(args.borrow_limit),
            borrow_interest=float(args.borrow_interest),
        ),
    ]

    family_rows: List[Dict[str, Any]] = []
    summary_by_family: Dict[str, pd.DataFrame] = {}
    family_dirs: Dict[str, Path] = {}
    for fam in families:
        fam_dir = out_root / fam.tag()
        family_dirs[fam.family] = fam_dir
        summary_df = run_family_alpha_sweep(
            repo_root,
            base_module=base,
            mechanism=mechanism,
            family=fam,
            alpha_values=alpha_values,
            total_timesteps=int(args.total_timesteps),
            seed_start=int(args.seed_start),
            seed_stop=int(args.seed_stop),
            late_frac=float(args.late_frac),
            out_dir=fam_dir,
            force_repatch=bool(args.force_repatch),
            force_rerun=bool(args.force_rerun),
        )
        summary_by_family[fam.family] = summary_df
        cand = json.loads((fam_dir / "candidate_metrics.json").read_text(encoding="utf-8"))
        cand["family"] = fam.family
        cand["support_mode"] = fam.support_mode
        cand["support_total"] = fam.support_total
        cand["borrow_limit"] = fam.borrow_limit
        cand["borrow_interest"] = fam.borrow_interest
        family_rows.append(cand)

    family_df = coerce_numeric(pd.DataFrame(family_rows))
    if not family_df.empty:
        base_guess = float(family_df.loc[family_df["family"] == "base", "alpha_transition_guess"].iloc[0]) if (family_df["family"] == "base").any() else np.nan
        if np.isfinite(base_guess):
            family_df["delta_alpha_transition_guess_vs_base"] = family_df["alpha_transition_guess"].astype(float) - base_guess
    family_df.to_csv(out_root / "family_transition_summary.csv", index=False)

    graph_dir = out_root / "graph_package"
    ensure_dir(graph_dir)
    family_overlay_plot(summary_by_family, "p_sustained_mean", "Finalized candidate: sustained-active fraction by intervention family", "P_sustained", graph_dir / "p_sustained_vs_alpha_family_overlay.png")
    family_overlay_plot(summary_by_family, "p_bursty_mean", "Finalized candidate: bursty-only fraction by intervention family", "P_bursty", graph_dir / "p_bursty_vs_alpha_family_overlay.png")
    family_overlay_plot(summary_by_family, "rho_mean", "Finalized candidate: late-window mean activity by intervention family", "rho", graph_dir / "rho_vs_alpha_family_overlay.png")
    family_overlay_plot(summary_by_family, "active_blocks_mean", "Finalized candidate: active late-window blocks by intervention family", "active blocks", graph_dir / "active_blocks_vs_alpha_family_overlay.png")
    family_overlay_plot(summary_by_family, "chi_rho", "Finalized candidate: seed fluctuation proxy by intervention family", "chi_rho", graph_dir / "chi_rho_vs_alpha_family_overlay.png")
    family_overlay_plot(summary_by_family, "eval_return_end_mean_mean", "Finalized candidate: deterministic eval return by intervention family", "eval return", graph_dir / "mean_final_eval_return_vs_alpha_family_overlay.png")
    p_sustained_wilson_panels(summary_by_family, graph_dir / "p_sustained_with_wilson_ci_by_family.png")
    late_update_boxplot_by_family(summary_by_family, family_dirs, graph_dir / "late_update_count_distributions_by_family.png")
    reps = save_family_trace_plots(summary_by_family, family_dirs, graph_dir)

    write_report(out_root, selected_attempt, family_df, alpha_values, graph_dir, reps)
    print(f"[n3-stageE-candidate] wrote outputs to {out_root}")


def mode_graph(args: argparse.Namespace) -> None:
    repo_root = infer_repo_root(args.repo_root)
    out_root = repo_root / "runs" / "n3_stageE_candidate_interventions"
    summary_path = out_root / "family_transition_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing family_transition_summary.csv under {out_root}; run mode first.")
    graph_dir = out_root / "graph_package"
    ensure_dir(graph_dir)
    family_df = coerce_numeric(pd.read_csv(summary_path))
    summary_by_family: Dict[str, pd.DataFrame] = {}
    family_dirs: Dict[str, Path] = {}
    for row in family_df.to_dict(orient="records"):
        family = str(row.get("family", ""))
        fam_dir = locate_family_dir(out_root, row)
        family_dirs[family] = fam_dir
        sdf = coerce_numeric(pd.read_csv(fam_dir / "alpha_sweep_phase_summary.csv"))
        summary_by_family[family] = sdf
    family_overlay_plot(summary_by_family, "p_sustained_mean", "Finalized candidate: sustained-active fraction by intervention family", "P_sustained", graph_dir / "p_sustained_vs_alpha_family_overlay.png")
    family_overlay_plot(summary_by_family, "p_bursty_mean", "Finalized candidate: bursty-only fraction by intervention family", "P_bursty", graph_dir / "p_bursty_vs_alpha_family_overlay.png")
    family_overlay_plot(summary_by_family, "rho_mean", "Finalized candidate: late-window mean activity by intervention family", "rho", graph_dir / "rho_vs_alpha_family_overlay.png")
    family_overlay_plot(summary_by_family, "active_blocks_mean", "Finalized candidate: active late-window blocks by intervention family", "active blocks", graph_dir / "active_blocks_vs_alpha_family_overlay.png")
    family_overlay_plot(summary_by_family, "chi_rho", "Finalized candidate: seed fluctuation proxy by intervention family", "chi_rho", graph_dir / "chi_rho_vs_alpha_family_overlay.png")
    family_overlay_plot(summary_by_family, "eval_return_end_mean_mean", "Finalized candidate: deterministic eval return by intervention family", "eval return", graph_dir / "mean_final_eval_return_vs_alpha_family_overlay.png")
    p_sustained_wilson_panels(summary_by_family, graph_dir / "p_sustained_with_wilson_ci_by_family.png")
    late_update_boxplot_by_family(summary_by_family, family_dirs, graph_dir / "late_update_count_distributions_by_family.png")
    reps = save_family_trace_plots(summary_by_family, family_dirs, graph_dir)
    selected_json = locate_finalize_artifacts(repo_root, args.selected_attempt_json)
    selected_attempt = json.loads(selected_json.read_text(encoding="utf-8"))
    alpha_values = json.loads((out_root / "alpha_grid.json").read_text()) if (out_root / "alpha_grid.json").exists() else []
    write_report(out_root, selected_attempt, family_df, alpha_values, graph_dir, reps)
    print(f"[n3-stageE-candidate] refreshed graph package in {graph_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-selection Stage-E intervention test on the finalized N=3 engineered crossover candidate.")
    parser.add_argument("mode", choices=["run", "graph"])
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--base-script", type=str, default=None, help="Path to n3_engineered_phase_transition.py")
    parser.add_argument("--stagee-script", type=str, default=None, help="Path to stageE_clipped_robustness_extension.py")
    parser.add_argument("--selected-attempt-json", type=str, default=None, help="Path to finalized selected_attempt.json")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-stop", type=int, default=DEFAULT_SEED_STOP)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--late-frac", type=float, default=DEFAULT_LATE_FRAC)
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX)
    parser.add_argument("--alpha-step", type=float, default=DEFAULT_ALPHA_STEP)
    parser.add_argument("--support-mode", choices=["none", "frontload", "drip", "pulse"], default=DEFAULT_SUPPORT_MODE)
    parser.add_argument("--support-total", type=float, default=DEFAULT_SUPPORT_TOTAL)
    parser.add_argument("--support-start-update", type=int, default=DEFAULT_SUPPORT_START_UPDATE)
    parser.add_argument("--support-window-frac", type=float, default=DEFAULT_SUPPORT_WINDOW_FRAC)
    parser.add_argument("--borrow-limit", type=float, default=DEFAULT_BORROW_LIMIT)
    parser.add_argument("--borrow-interest", type=float, default=DEFAULT_BORROW_INTEREST)
    parser.add_argument("--force-repatch", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "run":
        mode_run(args)
    elif args.mode == "graph":
        mode_graph(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
