#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R0 = -27.256548
D0 = 0.733348
R_REF_SHORT = -22.315781
R_THRESH_SHORT = -24.786164
R_REF_FULL = -24.779596
R_THRESH_FULL = -26.018072
EPS_OCC = 0.2
M_EVAL = 100
FULL_TIMESTEPS = 1_000_000
SHORT_TIMESTEPS = 400_000
NUM_ENVS = 25
NUM_STEPS = 128
SEED_START = 30
SEED_STOP_EXCLUSIVE = 62
BETA = float(np.log(10.0))
Z_MAX = 10.0
EPS_NORM = 1e-8
E0 = 1.0
E_MAX = 1.0
PHYSICAL_C_MAX = 1.0
ENV_NAME = "MPE_simple_spread_v3"
ENV_KWARGS = {
    "num_agents": 3,
    "num_landmarks": 3,
    "local_ratio": 0.5,
    "max_steps": 25,
    "action_type": "Discrete",
}

TRACE_ALIASES: Dict[str, List[str]] = {
    "update": ["update"],
    "update_u": ["update/u"],
    "update_counter": ["update/update_counter"],
    "opt_step": ["update/opt_step"],
    "energy_E_pre": ["energy/E_pre"],
    "energy_E_post": ["energy/E_post"],
    "funding_z": ["funding/z"],
    "funding_g": ["funding/g"],
    "funding_income": ["funding/income"],
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
    ("update_counter", "Cumulative executed updates", "executed updates"),
    ("funding_g", "Funding signal g_k", "g"),
    ("episode_return_end_mean", "End-of-episode training return", "return"),
    ("vartd_VarTD", "VarTD", "VarTD"),
    ("update_u", "Executed-update indicator u_k", "u"),
]


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    stage_source: str
    horizon_label: str
    total_timesteps: int
    alpha: float
    c_upd: float
    expected_status: str
    rationale: str
    note: str = ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt(x: Any, digits: int = 6) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    if np.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def parse_kv_report(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _finite_numeric_values(x: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def q25(x: pd.Series) -> float:
    vals = _finite_numeric_values(x)
    if vals.size == 0:
        return float("nan")
    return float(np.nanquantile(vals, 0.25))


def q75(x: pd.Series) -> float:
    vals = _finite_numeric_values(x)
    if vals.size == 0:
        return float("nan")
    return float(np.nanquantile(vals, 0.75))


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


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    out.columns = ["_".join(str(x) for x in col if str(x) != "") for col in out.columns]
    return out


def need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_inputs(repo_root: Path) -> Dict[str, Path]:
    return {
        "stagec_report": need(repo_root / "runs" / "stageC_subset_refine_reports" / "STAGE_C_SUBSET_FULL_HORIZON_REPORT.txt"),
        "stagec_alpha_boundary": need(repo_root / "runs" / "stageC_subset_refine_reports" / "stageC_subset_refine_alpha_boundary.csv"),
        "stagec_by_point": need(repo_root / "runs" / "stageC_subset_refine_reports" / "stageC_subset_refine_by_point.csv"),
        "stageb_bridge_by_point": need(repo_root / "runs" / "stageB_completion_bridge_reports" / "stageB_bridge_by_point.csv"),
        "gate5_master": need(repo_root / "baselines" / "IPPO" / "ippo_ff_mpe_gate5_energy_gated.py"),
    }


def verify_stage_c(stagec_report_path: Path) -> Dict[str, str]:
    kv = parse_kv_report(stagec_report_path)
    ok = kv.get("stage_c_subset_complete", "").lower() == "yes"
    completed = int(pd.to_numeric(kv.get("completed_runs", -1), errors="coerce"))
    expected = int(pd.to_numeric(kv.get("expected_runs", -1), errors="coerce"))
    missing = int(pd.to_numeric(kv.get("missing_runs", -1), errors="coerce"))
    if not ok or completed != expected or missing != 0:
        raise RuntimeError(
            f"Stage C is not complete: stage_c_subset_complete={kv.get('stage_c_subset_complete')}, "
            f"completed_runs={completed}, expected_runs={expected}, missing_runs={missing}"
        )
    return kv


def choose_conditions(
    stagec_alpha: pd.DataFrame,
    stagec_by_point: pd.DataFrame,
    stageb_bridge: pd.DataFrame,
    include_gateoff: bool,
) -> List[ConditionSpec]:
    enabled_pool = stagec_by_point.copy()
    if "P_learn_R_full_0p5" in enabled_pool.columns:
        enabled_pool = enabled_pool.loc[pd.to_numeric(enabled_pool["P_learn_R_full_0p5"], errors="coerce") >= 0.99]
    if enabled_pool.empty:
        raise RuntimeError("No clearly enabled full-horizon Stage C point found.")
    enabled_sort_cols = [c for c in ["mean_I_R_full", "mean_eval_return_end_mean"] if c in enabled_pool.columns]
    if not enabled_sort_cols:
        raise RuntimeError("Stage C by-point table has no usable ranking columns for enabled exemplar selection.")
    enabled_row = enabled_pool.sort_values(enabled_sort_cols, ascending=[False] * len(enabled_sort_cols)).iloc[0]

    alpha0_stagec = stagec_by_point.loc[np.isclose(pd.to_numeric(stagec_by_point["alpha"], errors="coerce"), 0.0)]
    if alpha0_stagec.empty:
        raise RuntimeError("No alpha=0.0 rows in Stage C by-point table.")
    fragile_row = alpha0_stagec.sort_values("c_upd").iloc[-1]

    alpha0_boundary = stagec_alpha.loc[np.isclose(pd.to_numeric(stagec_alpha["alpha"], errors="coerce"), 0.0)]
    bracket_lo = 0.58
    if not alpha0_boundary.empty and "bracket_lo" in alpha0_boundary.columns:
        try:
            bracket_lo = float(alpha0_boundary.iloc[0]["bracket_lo"])
        except Exception:
            bracket_lo = 0.58

    alpha0_stageb = stageb_bridge.loc[np.isclose(pd.to_numeric(stageb_bridge["alpha"], errors="coerce"), 0.0)].copy()
    if alpha0_stageb.empty:
        raise RuntimeError("No alpha=0.0 rows in Stage B bridge table.")
    p_col = "P_learn_R_short_0p5" if "P_learn_R_short_0p5" in alpha0_stageb.columns else "P_learn"
    physical = pd.to_numeric(alpha0_stageb["c_upd"], errors="coerce") <= PHYSICAL_C_MAX + 1e-12
    disabled_pool = alpha0_stageb.loc[
        physical
        & (pd.to_numeric(alpha0_stageb["c_upd"], errors="coerce") > bracket_lo + 1e-12)
        & (pd.to_numeric(alpha0_stageb[p_col], errors="coerce") < 0.5)
    ].copy()
    if disabled_pool.empty:
        disabled_pool = alpha0_stageb.loc[physical & (pd.to_numeric(alpha0_stageb[p_col], errors="coerce") < 0.5)].copy()
    if disabled_pool.empty:
        raise RuntimeError("No accepted short-horizon disabled Stage B point found.")
    disabled_row = disabled_pool.sort_values(["c_upd", p_col], ascending=[True, False]).iloc[0]

    conditions = [
        ConditionSpec(
            label="enabled_full",
            stage_source="Stage C",
            horizon_label="full",
            total_timesteps=FULL_TIMESTEPS,
            alpha=float(enabled_row["alpha"]),
            c_upd=float(enabled_row["c_upd"]),
            expected_status="learning-enabled full-horizon exemplar",
            rationale="Strongest learning-enabled full-horizon point in the completed Stage C by-point table.",
        ),
        ConditionSpec(
            label="fragile_full",
            stage_source="Stage C",
            horizon_label="full",
            total_timesteps=FULL_TIMESTEPS,
            alpha=float(fragile_row["alpha"]),
            c_upd=float(fragile_row["c_upd"]),
            expected_status="boundary-near / fragile full-horizon exemplar",
            rationale="Largest sampled in-domain alpha=0.0 Stage C refine point; Stage C says the true full-horizon boundary lies above this sampled band.",
        ),
        ConditionSpec(
            label="disabled_short",
            stage_source="Stage B",
            horizon_label="short",
            total_timesteps=SHORT_TIMESTEPS,
            alpha=float(disabled_row["alpha"]),
            c_upd=float(disabled_row["c_upd"]),
            expected_status="disabled within accepted short horizon",
            rationale="First accepted in-domain short-horizon alpha=0.0 point beyond the full-horizon fragile band with P_learn < 0.5.",
            note="This is a short-horizon disabled exemplar only. It must not be described as a full-horizon in-domain crossing.",
        ),
    ]

    if include_gateoff:
        gateoff_pool = alpha0_stageb.loc[pd.to_numeric(alpha0_stageb["c_upd"], errors="coerce") > PHYSICAL_C_MAX].copy()
        if not gateoff_pool.empty:
            gateoff_row = gateoff_pool.sort_values("c_upd").iloc[0]
            conditions.append(
                ConditionSpec(
                    label="gateoff_control_short",
                    stage_source="Stage B",
                    horizon_label="short",
                    total_timesteps=SHORT_TIMESTEPS,
                    alpha=float(gateoff_row["alpha"]),
                    c_upd=float(gateoff_row["c_upd"]),
                    expected_status="out-of-domain gate-off diagnostic",
                    rationale="Optional diagnostic control beyond the physical c_upd domain.",
                    note="Outside the physical c_upd domain; diagnostic only.",
                )
            )
    return conditions


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
    module_name = f"_stageD_gate5_master_{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:12]}"
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
            f"Missing dependency while importing the accepted master script: {missing}. "
            "Run Stage D inside the same environment that successfully ran Stage C."
        ) from exc
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not hasattr(mod, "make_train"):
        raise AttributeError(f"Accepted master script at {path} has no make_train(config) function.")
    return mod


def build_config(cond: ConditionSpec) -> Dict[str, Any]:
    return {
        "ENV_NAME": ENV_NAME,
        "ENV_KWARGS": dict(ENV_KWARGS),
        "LR": 2.5e-4,
        "NUM_ENVS": NUM_ENVS,
        "NUM_STEPS": NUM_STEPS,
        "TOTAL_TIMESTEPS": int(cond.total_timesteps),
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
        "PROJECT": "JaxMARL_Gate5_StageD",
        "PRINT_EVERY": 1_000_000_000,
        "EPS_OCC": EPS_OCC,
        "D0": D0,
        "R0": R0,
        "ALPHA": float(cond.alpha),
        "C_UPD": float(cond.c_upd),
        "BETA": BETA,
        "Z_MAX": Z_MAX,
        "EPS_NORM": EPS_NORM,
        "USE_GEOM_FUNDING": True,
        "E0": E0,
        "E_MAX": E_MAX,
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


def first_present(mapping: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in mapping:
            return k
    return None


def extract_trace_df(metrics_np: Mapping[str, np.ndarray], cond: ConditionSpec, seed: int) -> pd.DataFrame:
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
        raise RuntimeError("No metric traces extracted from the master script output.")
    if "update" not in data:
        data["update"] = np.arange(length, dtype=int)
    df = pd.DataFrame(data)
    df.insert(0, "seed", int(seed))
    df.insert(0, "c_upd", float(cond.c_upd))
    df.insert(0, "alpha", float(cond.alpha))
    df.insert(0, "horizon_label", cond.horizon_label)
    df.insert(0, "condition", cond.label)
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


def extract_eval_row(
    eval_np: Mapping[str, np.ndarray],
    trace_df: pd.DataFrame,
    cond: ConditionSpec,
    seed: int,
    runtime_sec: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "condition": cond.label,
        "horizon_label": cond.horizon_label,
        "alpha": float(cond.alpha),
        "c_upd": float(cond.c_upd),
        "seed": int(seed),
        "runtime_sec": float(runtime_sec),
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
    if "funding_z" in trace_df.columns:
        row["final_funding_z"] = float(trace_df["funding_z"].iloc[-1])
    if "funding_g" in trace_df.columns:
        row["final_funding_g"] = float(trace_df["funding_g"].iloc[-1])
    if "vartd_VarTD" in trace_df.columns:
        row["final_vartd"] = float(trace_df["vartd_VarTD"].iloc[-1])
    if "episode_return_end_mean" in trace_df.columns:
        row["train_final_return_end_mean"] = float(trace_df["episode_return_end_mean"].iloc[-1])
    if "terminal_min_dists_end_mean" in trace_df.columns:
        row["train_final_min_dists_end_mean"] = float(trace_df["terminal_min_dists_end_mean"].iloc[-1])
    if "terminal_occupied_end_mean" in trace_df.columns:
        row["train_final_occupied_end_mean"] = float(trace_df["terminal_occupied_end_mean"].iloc[-1])
    if "terminal_CSR_train" in trace_df.columns:
        row["train_final_CSR_train"] = float(trace_df["terminal_CSR_train"].iloc[-1])
    if "terminal_dbar_end" in trace_df.columns:
        row["train_final_dbar_end"] = float(trace_df["terminal_dbar_end"].iloc[-1])
    return row


def aggregate_trace_df(trace_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in trace_df.columns if c not in {"condition", "horizon_label", "alpha", "c_upd", "seed", "update"}]
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
    if np.isfinite(y).sum() == 0:
        return
    lo = agg_df[q25_col].to_numpy(dtype=float) if q25_col in agg_df.columns else None
    hi = agg_df[q75_col].to_numpy(dtype=float) if q75_col in agg_df.columns else None
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(x, y, linewidth=2.0, label="mean")
    if lo is not None and hi is not None:
        plt.fill_between(x, lo, hi, alpha=0.25, label="q25-q75")
    plt.xlabel("update")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def maybe_load_cached_seed(seed_dir: Path, cond: ConditionSpec, seed: int) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    trace_path = seed_dir / "trace_per_update.csv"
    eval_path = seed_dir / "eval_summary.json"
    if not trace_path.exists() or not eval_path.exists():
        return None, None
    try:
        trace_df = pd.read_csv(trace_path)
        with eval_path.open("r", encoding="utf-8") as f:
            eval_row = json.load(f)
    except Exception:
        return None, None
    if trace_df.empty:
        return None, None
    trace_df = coerce_numeric(trace_df)
    trace_df["condition"] = cond.label
    trace_df["horizon_label"] = cond.horizon_label
    trace_df["alpha"] = float(cond.alpha)
    trace_df["c_upd"] = float(cond.c_upd)
    trace_df["seed"] = int(seed)
    eval_row = dict(eval_row)
    eval_row["condition"] = cond.label
    eval_row["horizon_label"] = cond.horizon_label
    eval_row["alpha"] = float(cond.alpha)
    eval_row["c_upd"] = float(cond.c_upd)
    eval_row["seed"] = int(seed)
    return trace_df, eval_row


def run_condition(
    mod: Any,
    cond: ConditionSpec,
    seeds: Sequence[int],
    out_dir: Path,
    force_rerun: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import jax

    ensure_dir(out_dir)
    train_fn = mod.make_train(build_config(cond))
    compiled_train = jax.jit(train_fn)
    endpoint_rows: List[Dict[str, Any]] = []
    all_traces: List[pd.DataFrame] = []

    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed:02d}"
        ensure_dir(seed_dir)

        if not force_rerun:
            cached_trace_df, cached_eval_row = maybe_load_cached_seed(seed_dir, cond, seed)
            if cached_trace_df is not None and cached_eval_row is not None:
                print(f"[Stage D] cache hit {cond.label} seed={seed:02d}")
                all_traces.append(cached_trace_df)
                endpoint_rows.append(cached_eval_row)
                continue

        print(f"[Stage D] run {cond.label} seed={seed:02d}")
        t0 = time.perf_counter()
        out = compiled_train(jax.random.PRNGKey(int(seed)))
        metrics_device = out["metrics"]
        eval_device = out["eval"]
        block_until_ready(metrics_device)
        block_until_ready(eval_device)
        runtime_sec = time.perf_counter() - t0

        metrics_np = to_numpy_tree(metrics_device)
        eval_np = to_numpy_tree(eval_device)
        del out, metrics_device, eval_device

        trace_df = extract_trace_df(metrics_np, cond, seed)
        eval_row = extract_eval_row(eval_np, trace_df, cond, seed, runtime_sec)

        trace_df.to_csv(seed_dir / "trace_per_update.csv", index=False)
        with (seed_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
            json.dump(eval_row, f, indent=2, sort_keys=True)

        all_traces.append(trace_df)
        endpoint_rows.append(eval_row)

        del metrics_np, eval_np, trace_df
        gc.collect()

    if not endpoint_rows or not all_traces:
        raise RuntimeError(f"No Stage D outputs were produced for condition {cond.label}.")

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
        save_band_plot(trace_agg, metric, f"{cond.label}: {title}", ylabel, plots_dir / f"{metric}.png")

    return endpoint_df, trace_agg


def summarize_endpoints(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_seeds": int(len(df))}
    for c in df.columns:
        if c in {"condition", "horizon_label", "alpha", "c_upd", "seed"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out[f"{c}_mean"] = float(df[c].mean())
            out[f"{c}_sd"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def summary_get(summary: Mapping[str, Any], base_key: str) -> Any:
    return summary.get(f"{base_key}_mean", float("nan"))


def summary_sd(summary: Mapping[str, Any], base_key: str) -> Any:
    return summary.get(f"{base_key}_sd", float("nan"))


def write_report(
    out_dir: Path,
    repo_root: Path,
    inputs: Mapping[str, Path],
    stagec_kv: Mapping[str, str],
    conditions: Sequence[ConditionSpec],
    summaries: Mapping[str, Dict[str, Any]],
    seeds: Sequence[int],
) -> None:
    lines: List[str] = []
    lines.append("Stage D representative mechanistic traces report")
    lines.append("")
    lines.append("Frozen prerequisites verified")
    lines.append(f"repo_root={repo_root}")
    for k, v in inputs.items():
        lines.append(f"{k}={v.relative_to(repo_root)}")
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
    lines.append(f"SEEDS={min(seeds)}..{max(seeds)} ({len(seeds)} total)")
    lines.append("")
    lines.append("Stage C completion evidence")
    for k in ["stage_c_subset_complete", "completed_runs", "expected_runs", "missing_runs", "right_censored_count"]:
        lines.append(f"{k}={stagec_kv.get(k, 'unknown')}")
    lines.append("")
    lines.append("Interpretation constraint")
    lines.append(
        "Stage C completed successfully, but it did not produce an interior full-horizon in-domain crossing. "
        "Alpha=0.0 stayed above threshold through c_upd=0.58, and alpha=0.1..1.0 remained right-censored above c_upd=1.0. "
        "Therefore this Stage D package uses one full-horizon enabled exemplar, one full-horizon fragile exemplar, and one accepted short-horizon disabled exemplar."
    )
    lines.append("")
    lines.append("Selected conditions")
    for cond in conditions:
        lines.append(
            f"condition={cond.label} stage_source={cond.stage_source} horizon={cond.horizon_label} "
            f"alpha={fmt(cond.alpha, 3)} c_upd={fmt(cond.c_upd, 3)} expected_status={cond.expected_status}"
        )
        lines.append(f"rationale={cond.rationale}")
        if cond.note:
            lines.append(f"note={cond.note}")
    lines.append("")
    lines.append("Per-condition endpoint summaries")
    for cond in conditions:
        s = summaries[cond.label]
        lines.append(
            f"condition={cond.label} "
            f"eval_return_end_mean={fmt(summary_get(s, 'eval_return_end_mean'))}±{fmt(summary_sd(s, 'eval_return_end_mean'))} "
            f"eval_dbar_post_mean={fmt(summary_get(s, 'eval_dbar_post_mean'))}±{fmt(summary_sd(s, 'eval_dbar_post_mean'))} "
            f"eval_occupied_post_mean={fmt(summary_get(s, 'eval_occupied_post_mean'))}±{fmt(summary_sd(s, 'eval_occupied_post_mean'))} "
            f"eval_CSR_post={fmt(summary_get(s, 'eval_CSR_post'))}±{fmt(summary_sd(s, 'eval_CSR_post'))} "
            f"mean_update_fraction={fmt(summary_get(s, 'mean_update_fraction'))}±{fmt(summary_sd(s, 'mean_update_fraction'))} "
            f"final_executed_updates={fmt(summary_get(s, 'final_executed_updates'))}±{fmt(summary_sd(s, 'final_executed_updates'))} "
            f"final_energy={fmt(summary_get(s, 'final_energy'))}±{fmt(summary_sd(s, 'final_energy'))} "
            f"final_vartd={fmt(summary_get(s, 'final_vartd'))}±{fmt(summary_sd(s, 'final_vartd'))}"
        )
    lines.append("")
    lines.append("Outputs")
    lines.append("- stageD_condition_manifest.csv")
    lines.append("- stageD_endpoint_summary_all_conditions.csv")
    lines.append("- STAGE_D_REPRESENTATIVE_MECHANISTIC_TRACES_REPORT.txt")
    lines.append("- stageD_manifest.json")
    lines.append("- cross_condition_plots/*.png")
    lines.append("- conditions/<condition>/seed_XX/trace_per_update.csv")
    lines.append("- conditions/<condition>/seed_XX/eval_summary.json")
    lines.append("- conditions/<condition>/seed_endpoint_summary.csv")
    lines.append("- conditions/<condition>/trace_per_seed_long.csv")
    lines.append("- conditions/<condition>/trace_aggregate.csv")
    lines.append("- conditions/<condition>/plots/*.png")
    (out_dir / "STAGE_D_REPRESENTATIVE_MECHANISTIC_TRACES_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage D representative mechanistic traces for the frozen JaxMARL Gate 5 repair repo.")
    p.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Frozen repair repo root. Default: current working directory.")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: <repo-root>/runs/stageD_representative_mechanistic_traces")
    p.add_argument("--seed-start", type=int, default=SEED_START)
    p.add_argument("--seed-stop", type=int, default=SEED_STOP_EXCLUSIVE, help="Exclusive upper bound.")
    p.add_argument("--include-gateoff-control", action="store_true", help="Also run the accepted out-of-domain Stage B gate-off control.")
    p.add_argument("--force-rerun", action="store_true", help="Ignore cached per-seed outputs and rerun every seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = (args.output_dir or (repo_root / "runs" / "stageD_representative_mechanistic_traces")).resolve()
    ensure_dir(out_dir)
    ensure_dir(out_dir / "conditions")

    seeds = list(range(int(args.seed_start), int(args.seed_stop)))
    if not seeds:
        raise ValueError("Empty seed range.")

    inputs = load_inputs(repo_root)
    stagec_kv = verify_stage_c(inputs["stagec_report"])
    stagec_alpha = coerce_numeric(pd.read_csv(inputs["stagec_alpha_boundary"]))
    stagec_by_point = coerce_numeric(pd.read_csv(inputs["stagec_by_point"]))
    stageb_bridge = coerce_numeric(pd.read_csv(inputs["stageb_bridge_by_point"]))
    conditions = choose_conditions(stagec_alpha, stagec_by_point, stageb_bridge, bool(args.include_gateoff_control))

    mod = load_master(repo_root, inputs["gate5_master"])

    endpoint_frames: List[pd.DataFrame] = []
    aggregate_by_condition: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    condition_rows: List[Dict[str, Any]] = []

    for cond in conditions:
        cond_dir = out_dir / "conditions" / cond.label
        print(
            f"[Stage D] {cond.label}: horizon={cond.horizon_label}, "
            f"alpha={cond.alpha:.3f}, c_upd={cond.c_upd:.3f}, seeds={seeds[0]}..{seeds[-1]}"
        )
        endpoint_df, agg_df = run_condition(mod, cond, seeds, cond_dir, force_rerun=bool(args.force_rerun))
        endpoint_frames.append(endpoint_df)
        aggregate_by_condition[cond.label] = agg_df
        summaries[cond.label] = summarize_endpoints(endpoint_df)
        condition_rows.append({**asdict(cond), **summaries[cond.label]})
        gc.collect()

    endpoint_all = pd.concat(endpoint_frames, ignore_index=True)
    endpoint_all = coerce_numeric(endpoint_all)
    endpoint_all.to_csv(out_dir / "stageD_endpoint_summary_all_conditions.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(out_dir / "stageD_condition_manifest.csv", index=False)

    cross_dir = out_dir / "cross_condition_plots"
    ensure_dir(cross_dir)
    for metric, title, ylabel in PLOT_SPECS:
        plt.figure(figsize=(8.4, 5.0))
        drew = False
        for cond in conditions:
            agg_df = aggregate_by_condition[cond.label]
            mean_col = f"{metric}_mean"
            if mean_col not in agg_df.columns:
                continue
            x = agg_df["update"].to_numpy(dtype=float)
            y = agg_df[mean_col].to_numpy(dtype=float)
            if np.isfinite(y).sum() == 0:
                continue
            plt.plot(x, y, linewidth=2.0, label=cond.label)
            drew = True
        if drew:
            plt.xlabel("update")
            plt.ylabel(ylabel)
            plt.title(f"Cross-condition comparison: {title}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(cross_dir / f"{metric}.png", dpi=160)
        plt.close()

    write_report(out_dir, repo_root, inputs, stagec_kv, conditions, summaries, seeds)

    manifest = {
        "generated_at_unix": float(time.time()),
        "repo_root": str(repo_root),
        "output_dir": str(out_dir),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "cpu_count": int(os.cpu_count() or 1),
        },
        "locked_constants": {
            "R0": R0,
            "D0": D0,
            "R_REF_SHORT": R_REF_SHORT,
            "R_THRESH_SHORT": R_THRESH_SHORT,
            "R_REF_FULL": R_REF_FULL,
            "R_THRESH_FULL": R_THRESH_FULL,
            "EPS_OCC": EPS_OCC,
            "M_EVAL": M_EVAL,
            "NUM_ENVS": NUM_ENVS,
            "NUM_STEPS": NUM_STEPS,
            "SEED_START": seeds[0],
            "SEED_STOP_INCLUSIVE": seeds[-1],
        },
        "input_files": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in inputs.items()},
        "selected_conditions": [asdict(c) for c in conditions],
    }
    with (out_dir / "stageD_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("[Stage D] Done.")
    print(f"[Stage D] Output directory: {out_dir}")
    print(f"[Stage D] Conditions: {', '.join(c.label for c in conditions)}")


if __name__ == "__main__":
    main()