#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "n3_stageE_borrow_flatline_disambiguation_v1"

DEFAULT_TOTAL_TIMESTEPS = 4_000_000
DEFAULT_LATE_FRAC = 0.25
DEFAULT_CANONICAL_SEED_START = 30
DEFAULT_CANONICAL_SEED_STOP = 62  # 32 seeds
DEFAULT_PROBE_SEED_START = 30
DEFAULT_PROBE_SEED_STOP = 42      # 12 seeds
DEFAULT_CANONICAL_ALPHA_MAX = 0.300
DEFAULT_CANONICAL_EXT_STEP_FINE = 0.006
DEFAULT_CANONICAL_EXT_STEP_COARSE = 0.012
DEFAULT_CANONICAL_EXT_MID = 0.180
DEFAULT_SUPPORT_MODE = "frontload"
DEFAULT_SUPPORT_TOTAL = 0.60
DEFAULT_SUPPORT_START_UPDATE = 0
DEFAULT_SUPPORT_WINDOW_FRAC = 0.10
DEFAULT_BORROW_LIMIT = 0.60
DEFAULT_BORROW_INTEREST = 0.02
DEFAULT_SMALL_LIMIT = 0.30
DEFAULT_HALF_INTEREST = 0.01
DEFAULT_TRACE_FLOOR_TOL = 1e-6


def load_module_from_path(name: str, path: Path, repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def first_finite(values: Sequence[float], xs: Sequence[float], thresh: float) -> float:
    arr_v = np.asarray(values, dtype=float)
    arr_x = np.asarray(xs, dtype=float)
    mask = np.isfinite(arr_v) & np.isfinite(arr_x) & (arr_v >= thresh)
    if mask.any():
        return float(arr_x[np.argmax(mask)])
    return float("nan")


@dataclass(frozen=True)
class BorrowVariant:
    name: str
    borrow_limit: float
    borrow_interest: float
    alpha_values: Tuple[float, ...]
    seed_start: int
    seed_stop: int
    total_timesteps: int
    description: str
    certify_priority: int = 0
    label_type: str = "diagnostic"  # canonical or diagnostic

    def family_spec(self, bridge_mod: Any):
        return bridge_mod.InterventionFamilySpec(
            family="borrow",
            borrow_limit=float(self.borrow_limit),
            borrow_interest=float(self.borrow_interest),
        )


# -----------------------------------------------------------------------------
# Alpha grids
# -----------------------------------------------------------------------------
def unique_sorted(xs: Iterable[float], ndigits: int = 12) -> List[float]:
    out = sorted({round(float(x), ndigits) for x in xs})
    return [float(x) for x in out]


def frange(start: float, stop: float, step: float) -> List[float]:
    vals: List[float] = []
    x = float(start)
    step = float(step)
    if step <= 0:
        raise ValueError("step must be positive")
    limit = float(stop) + 1e-12
    while x <= limit:
        vals.append(round(x, 12))
        x += step
    return vals


def build_extended_canonical_alpha_grid(shared_alpha_values: Sequence[float], max_alpha: float, mid_alpha: float, fine_step: float, coarse_step: float) -> List[float]:
    shared = unique_sorted(shared_alpha_values)
    if not shared:
        raise ValueError("shared alpha grid is empty")
    current_max = float(max(shared))
    ext1 = frange(current_max + fine_step, min(max_alpha, mid_alpha), fine_step) if current_max < min(max_alpha, mid_alpha) else []
    ext2_start = max(mid_alpha + coarse_step, current_max + fine_step)
    ext2 = frange(ext2_start, max_alpha, coarse_step) if ext2_start <= max_alpha else []
    return unique_sorted(list(shared) + ext1 + ext2)


def build_probe_alpha_grid(shared_alpha_values: Sequence[float], max_alpha: float, mid_alpha: float, fine_step: float, coarse_step: float) -> List[float]:
    # Focus probes on the ambiguous / active-side range to save compute while retaining continuity.
    shared = unique_sorted(shared_alpha_values)
    left_probe = [x for x in shared if x >= 0.060]
    if 0.108 not in left_probe:
        left_probe.append(0.108)
    ext1 = frange(0.114, min(max_alpha, mid_alpha), fine_step) if 0.114 <= min(max_alpha, mid_alpha) else []
    ext2 = frange(mid_alpha + coarse_step, max_alpha, coarse_step) if (mid_alpha + coarse_step) <= max_alpha else []
    return unique_sorted(left_probe + ext1 + ext2)


# -----------------------------------------------------------------------------
# Existing run reuse
# -----------------------------------------------------------------------------
def try_load_existing_family_summary(existing_root: Path, family: str) -> Optional[pd.DataFrame]:
    if not existing_root.exists():
        return None
    candidates: List[Path] = []
    for p in existing_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if family == "base" and name == "base":
            candidates.append(p / "alpha_sweep_phase_summary.csv")
        elif family == "support" and name.startswith("support"):
            candidates.append(p / "alpha_sweep_phase_summary.csv")
        elif family == "borrow" and name.startswith("borrow"):
            candidates.append(p / "alpha_sweep_phase_summary.csv")
    candidates = [c for c in candidates if c.exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(str(p)))
    return coerce_numeric(pd.read_csv(candidates[0]))


# -----------------------------------------------------------------------------
# Trace-derived borrow diagnostics
# -----------------------------------------------------------------------------
def compute_trace_late_metrics(trace_long: pd.DataFrame, alpha: float, late_frac: float, borrow_limit: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if trace_long.empty:
        return pd.DataFrame()
    for seed, sdf in trace_long.groupby("seed"):
        sdf = coerce_numeric(sdf.sort_values("update"))
        n = len(sdf)
        tail_n = max(1, int(round(n * float(late_frac))))
        tail = sdf.iloc[-tail_n:].copy()
        funding_g = pd.to_numeric(tail.get("funding_g", pd.Series(np.nan, index=tail.index)), errors="coerce")
        funding_g_raw = pd.to_numeric(tail.get("funding_g_raw", funding_g), errors="coerce")
        debt_charge = pd.to_numeric(tail.get("debt_charge", pd.Series(0.0, index=tail.index)), errors="coerce").fillna(0.0)
        energy_post = pd.to_numeric(tail.get("energy_E_post", pd.Series(np.nan, index=tail.index)), errors="coerce")
        debt_level_post = pd.to_numeric(tail.get("debt_level_post", pd.Series(np.nan, index=tail.index)), errors="coerce")
        update_u = pd.to_numeric(tail.get("update_u", pd.Series(np.nan, index=tail.index)), errors="coerce")
        cut_active = pd.to_numeric(tail.get("funding_cut_active", pd.Series(np.nan, index=tail.index)), errors="coerce")

        g_eff_mean = float(np.nanmean(funding_g.to_numpy(dtype=float)))
        g_raw_mean = float(np.nanmean(funding_g_raw.to_numpy(dtype=float)))
        debt_mean = float(np.nanmean(debt_charge.to_numpy(dtype=float)))
        alpha_g_mean = float(float(alpha) * g_eff_mean) if np.isfinite(g_eff_mean) else float("nan")
        margin_mean = float(alpha_g_mean - debt_mean) if np.isfinite(alpha_g_mean) and np.isfinite(debt_mean) else float("nan")
        if np.isfinite(g_eff_mean) and g_eff_mean > 1e-12 and np.isfinite(debt_mean):
            break_even_alpha = float(debt_mean / g_eff_mean)
        else:
            break_even_alpha = float("inf")
        floor_frac = float(np.mean(np.isfinite(energy_post.to_numpy(dtype=float)) & (energy_post.to_numpy(dtype=float) <= (-float(borrow_limit) + DEFAULT_TRACE_FLOOR_TOL))))
        neg_frac = float(np.mean(np.isfinite(energy_post.to_numpy(dtype=float)) & (energy_post.to_numpy(dtype=float) < -DEFAULT_TRACE_FLOOR_TOL)))
        rows.append(
            {
                "seed": int(seed),
                "late_g_eff_mean": g_eff_mean,
                "late_g_raw_mean": g_raw_mean,
                "late_alpha_g_eff_mean": alpha_g_mean,
                "late_debt_charge_mean": debt_mean,
                "late_margin_mean": margin_mean,
                "late_break_even_alpha_est": break_even_alpha,
                "late_debt_floor_frac": floor_frac,
                "late_negative_energy_frac": neg_frac,
                "late_debt_level_mean": float(np.nanmean(debt_level_post.to_numpy(dtype=float))),
                "late_update_u_mean": float(np.nanmean(update_u.to_numpy(dtype=float))),
                "late_cut_active_frac": float(np.nanmean(cut_active.to_numpy(dtype=float))),
                "late_window_updates": int(tail_n),
            }
        )
    return coerce_numeric(pd.DataFrame(rows))


def aggregate_trace_late_metrics(seed_diag: pd.DataFrame) -> Dict[str, Any]:
    if seed_diag.empty:
        return {}
    out: Dict[str, Any] = {"diag_n_seeds": int(len(seed_diag))}
    for col in [
        "late_g_eff_mean",
        "late_g_raw_mean",
        "late_alpha_g_eff_mean",
        "late_debt_charge_mean",
        "late_margin_mean",
        "late_break_even_alpha_est",
        "late_debt_floor_frac",
        "late_negative_energy_frac",
        "late_debt_level_mean",
        "late_update_u_mean",
        "late_cut_active_frac",
    ]:
        vals = pd.to_numeric(seed_diag[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[f"{col}_mean"] = float("nan")
            out[f"{col}_sd"] = float("nan")
        else:
            out[f"{col}_mean"] = float(np.mean(vals))
            out[f"{col}_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return out


# -----------------------------------------------------------------------------
# Ranking / classification
# -----------------------------------------------------------------------------
def onset_alpha(summary: pd.DataFrame, thresh: float) -> float:
    if summary.empty:
        return float("nan")
    x = pd.to_numeric(summary.get("alpha", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(summary.get("p_sustained_mean", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    return first_finite(p, x, thresh)


def variant_score(summary: pd.DataFrame, cand: Mapping[str, Any]) -> float:
    cand_pass = int(cand.get("candidate_pass", 0))
    right_ps = float(cand.get("right_edge_p_sustained", np.nan))
    chi = float(cand.get("chi_peak_ratio", 0.0) or 0.0)
    viol = float(cand.get("p_sustained_violations", 999.0) or 999.0)
    onset50 = onset_alpha(summary, 0.5)
    onset_penalty = 0.0 if np.isnan(onset50) else float(onset50)
    return 1000.0 * cand_pass + 100.0 * np.nan_to_num(right_ps, nan=0.0) + 10.0 * chi - 2.0 * viol - 10.0 * onset_penalty


def classify_flatline(canonical_summary: pd.DataFrame, canonical_candidate: Mapping[str, Any], probe_summaries: Mapping[str, pd.DataFrame], probe_candidates: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["canonical_candidate_pass"] = int(canonical_candidate.get("candidate_pass", 0))
    out["canonical_transition_status"] = str(canonical_candidate.get("transition_status", "unknown"))
    out["canonical_onset_p10"] = onset_alpha(canonical_summary, 0.10)
    out["canonical_onset_p50"] = onset_alpha(canonical_summary, 0.50)
    out["canonical_onset_p90"] = onset_alpha(canonical_summary, 0.90)
    out["canonical_right_p_sustained"] = float(canonical_candidate.get("right_edge_p_sustained", np.nan))
    out["canonical_right_late_updates"] = float(canonical_candidate.get("right_edge_late_update_count_mean", np.nan))

    tail = canonical_summary.sort_values("alpha").tail(max(3, len(canonical_summary) // 5)) if not canonical_summary.empty else canonical_summary
    tail_floor = float(pd.to_numeric(tail.get("late_debt_floor_frac_mean", pd.Series(dtype=float)), errors="coerce").mean()) if not tail.empty and "late_debt_floor_frac_mean" in tail.columns else float("nan")
    tail_margin = float(pd.to_numeric(tail.get("late_margin_mean_mean", pd.Series(dtype=float)), errors="coerce").mean()) if not tail.empty and "late_margin_mean_mean" in tail.columns else float("nan")
    out["canonical_tail_debt_floor_frac"] = tail_floor
    out["canonical_tail_margin_mean"] = tail_margin

    def _probe(name: str) -> Tuple[float, float, float]:
        sdf = probe_summaries.get(name, pd.DataFrame())
        cand = probe_candidates.get(name, {})
        return onset_alpha(sdf, 0.50), float(cand.get("right_edge_p_sustained", np.nan)), float(cand.get("candidate_pass", 0))

    noint_on50, noint_right, noint_pass = _probe("borrow_no_interest_probe")
    half_on50, half_right, half_pass = _probe("borrow_half_interest_probe")
    small_on50, small_right, small_pass = _probe("borrow_small_limit_probe")
    slni_on50, slni_right, slni_pass = _probe("borrow_small_limit_no_interest_probe")
    out.update(
        {
            "borrow_no_interest_onset_p50": noint_on50,
            "borrow_half_interest_onset_p50": half_on50,
            "borrow_small_limit_onset_p50": small_on50,
            "borrow_small_limit_no_interest_onset_p50": slni_on50,
        }
    )

    interest_effect = (
        (np.isfinite(noint_on50) and not np.isfinite(out["canonical_onset_p50"]))
        or (np.isfinite(out["canonical_onset_p50"]) and np.isfinite(noint_on50) and noint_on50 + 1e-12 < out["canonical_onset_p50"])
        or (noint_pass > 0 and out["canonical_candidate_pass"] == 0)
    )
    limit_effect = (
        (np.isfinite(small_on50) and not np.isfinite(out["canonical_onset_p50"]))
        or (np.isfinite(out["canonical_onset_p50"]) and np.isfinite(small_on50) and small_on50 + 1e-12 < out["canonical_onset_p50"])
        or (small_pass > 0 and out["canonical_candidate_pass"] == 0)
    )

    if out["canonical_candidate_pass"] == 1:
        out["canonical_flatline_resolution"] = "canonical_borrow_resolved_on_extended_grid"
    elif np.isfinite(out["canonical_onset_p10"]) or out["canonical_right_p_sustained"] > 0.0:
        out["canonical_flatline_resolution"] = "canonical_borrow_right_shift_partially_resolved"
    elif np.isfinite(tail_floor) and np.isfinite(tail_margin) and tail_floor >= 0.8 and tail_margin <= 0.0:
        if interest_effect and limit_effect:
            out["canonical_flatline_resolution"] = "canonical_flatline_due_to_joint_interest_and_debt_floor_overshoot"
        elif interest_effect:
            out["canonical_flatline_resolution"] = "canonical_flatline_interest_limited"
        elif limit_effect:
            out["canonical_flatline_resolution"] = "canonical_flatline_debt_floor_overshoot_limited"
        else:
            out["canonical_flatline_resolution"] = "canonical_flatline_suppressed_within_extended_grid"
    else:
        out["canonical_flatline_resolution"] = "canonical_flatline_unresolved_after_extension"

    out["interest_effect_detected"] = int(bool(interest_effect))
    out["limit_effect_detected"] = int(bool(limit_effect))
    return out


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def family_line_plot(summary_map: Mapping[str, pd.DataFrame], col: str, title: str, ylabel: str, out_path: Path, *, x_max: Optional[float] = None, vline: Optional[float] = None, band: Optional[Tuple[float, float]] = None) -> None:
    plt.figure(figsize=(8.6, 5.2))
    plotted = False
    for label, sdf in summary_map.items():
        if col not in sdf.columns:
            continue
        x = pd.to_numeric(sdf["alpha"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sdf[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() == 0:
            continue
        plt.plot(x[ok], y[ok], linewidth=2.0, label=label)
        plotted = True
    if not plotted:
        plt.close()
        return
    if band is not None:
        plt.axvspan(float(band[0]), float(band[1]), alpha=0.12)
    if vline is not None and np.isfinite(vline):
        plt.axvline(float(vline), linestyle="--", linewidth=1.4)
    if x_max is not None and np.isfinite(x_max):
        plt.xlim(left=0.0, right=float(x_max))
    plt.xlabel(r"$\alpha$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_recharge_vs_charge(summary_map: Mapping[str, pd.DataFrame], out_path: Path) -> None:
    plt.figure(figsize=(8.6, 5.2))
    plotted = False
    for label, sdf in summary_map.items():
        if "late_alpha_g_eff_mean_mean" not in sdf.columns or "late_debt_charge_mean_mean" not in sdf.columns:
            continue
        x = pd.to_numeric(sdf["alpha"], errors="coerce").to_numpy(dtype=float)
        ag = pd.to_numeric(sdf["late_alpha_g_eff_mean_mean"], errors="coerce").to_numpy(dtype=float)
        dc = pd.to_numeric(sdf["late_debt_charge_mean_mean"], errors="coerce").to_numpy(dtype=float)
        ok1 = np.isfinite(x) & np.isfinite(ag)
        ok2 = np.isfinite(x) & np.isfinite(dc)
        if ok1.sum() > 0:
            plt.plot(x[ok1], ag[ok1], linewidth=2.0, label=f"{label}: α·g_eff")
            plotted = True
        if ok2.sum() > 0:
            plt.plot(x[ok2], dc[ok2], linewidth=2.0, linestyle="--", label=f"{label}: debt charge")
            plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel(r"$\alpha$")
    plt.ylabel("late-window mean")
    plt.title("Borrow disambiguation: recharge term vs debt charge")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_break_even(summary_map: Mapping[str, pd.DataFrame], out_path: Path) -> None:
    plt.figure(figsize=(8.6, 5.2))
    plotted = False
    xs_max = 0.0
    for label, sdf in summary_map.items():
        if "late_break_even_alpha_est_mean" not in sdf.columns:
            continue
        x = pd.to_numeric(sdf["alpha"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sdf["late_break_even_alpha_est_mean"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() == 0:
            continue
        xs_max = max(xs_max, float(np.nanmax(x[ok])))
        plt.plot(x[ok], y[ok], linewidth=2.0, label=f"{label}: α_break-even")
        plotted = True
    if not plotted:
        plt.close()
        return
    if xs_max > 0:
        diag = np.linspace(0.0, xs_max, 200)
        plt.plot(diag, diag, linewidth=1.5, linestyle="--", label=r"$\alpha_{break-even}=\alpha$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"estimated $\alpha_{break-even}$")
    plt.title("Borrow disambiguation: estimated break-even alpha")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def pick_trace(alpha_dir: Path, strategy: str, borrow_limit: float) -> Optional[Tuple[int, Path]]:
    trace_path = alpha_dir / "trace_per_seed_long.csv"
    if not trace_path.exists():
        return None
    df = coerce_numeric(pd.read_csv(trace_path))
    if df.empty:
        return None
    candidates = []
    for seed, sdf in df.groupby("seed"):
        sdf = sdf.sort_values("update")
        final_e = float(pd.to_numeric(sdf.get("energy_E_post", pd.Series([np.nan])), errors="coerce").iloc[-1]) if "energy_E_post" in sdf.columns else float("nan")
        max_debt = float(pd.to_numeric(sdf.get("debt_level_post", pd.Series([np.nan])), errors="coerce").max()) if "debt_level_post" in sdf.columns else float("nan")
        final_updates = float(pd.to_numeric(sdf.get("update_counter", pd.Series([np.nan])), errors="coerce").iloc[-1]) if "update_counter" in sdf.columns else float("nan")
        late_floor = float(np.mean(pd.to_numeric(sdf.get("energy_E_post", pd.Series(np.nan, index=sdf.index)), errors="coerce").to_numpy(dtype=float) <= (-float(borrow_limit) + DEFAULT_TRACE_FLOOR_TOL))) if "energy_E_post" in sdf.columns else float("nan")
        candidates.append((int(seed), final_updates, late_floor, final_e, max_debt))
    if not candidates:
        return None
    if strategy == "most_dead":
        candidates.sort(key=lambda x: (x[1], -x[2], x[3]))
    elif strategy == "most_active":
        candidates.sort(key=lambda x: (-x[1], x[2], -x[3]))
    else:
        candidates.sort(key=lambda x: abs(x[1] - np.median([c[1] for c in candidates])))
    seed = candidates[0][0]
    seed_path = alpha_dir / f"seed_{seed:02d}" / "trace_per_update.csv"
    if seed_path.exists():
        return (seed, seed_path)
    return None


def save_trace_plot(base_mod: Any, trace_csv: Path, title: str, out_path: Path) -> None:
    trace_df = coerce_numeric(pd.read_csv(trace_csv))
    metrics = [
        ("update_u", "u"),
        ("energy_E_post", "energy"),
        ("debt_level_post", "debt"),
        ("funding_g", "g_eff"),
        ("debt_charge", "debt charge"),
        ("episode_return_end_mean", "train return"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8.6, 2.2 * len(metrics)), sharex=True)
    x = pd.to_numeric(trace_df["update"], errors="coerce").to_numpy(dtype=float)
    for ax, (col, ylabel) in zip(axes, metrics):
        if col not in trace_df.columns:
            continue
        y = pd.to_numeric(trace_df[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 0:
            ax.plot(x[ok], y[ok], linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel("update")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def write_report(out_root: Path, selected_attempt: Mapping[str, Any], canonical_variant: BorrowVariant, canonical_candidate: Mapping[str, Any], diagnostic_rows: pd.DataFrame, classification: Mapping[str, Any], graph_dir: Path, promoted_variant: Optional[BorrowVariant]) -> None:
    lines: List[str] = []
    lines.append("N=3 finalized-candidate borrow-flatline disambiguation report")
    lines.append(f"script_version={SCRIPT_VERSION}")
    lines.append("")
    lines.append("Purpose")
    lines.append("This run preserves the finalized engineered candidate and the canonical Stage-E borrowing semantics.")
    lines.append("Its goal is to determine whether the previously observed borrow flatline is a true suppression on the tested range or a right shift beyond the shared alpha window.")
    lines.append("Diagnostic borrow ablations are reported separately and must not be conflated with the canonical Stage-E borrow family.")
    lines.append("")
    lines.append("Locked finalized candidate")
    for k in ["attempt_name", "raw_c_upd", "raw_z_cut", "raw_alpha_transition_lo", "raw_alpha_transition_hi", "raw_alpha_transition_guess", "certification_label"]:
        if k in selected_attempt:
            lines.append(f"{k}={selected_attempt[k]}")
    lines.append("")
    lines.append("Canonical borrow sweep")
    lines.append(f"borrow_limit={canonical_variant.borrow_limit}")
    lines.append(f"borrow_interest={canonical_variant.borrow_interest}")
    lines.append(f"seed_range={canonical_variant.seed_start}:{canonical_variant.seed_stop}")
    lines.append(f"total_timesteps={canonical_variant.total_timesteps}")
    lines.append(f"alpha_min={min(canonical_variant.alpha_values):.3f}")
    lines.append(f"alpha_max={max(canonical_variant.alpha_values):.3f}")
    lines.append(f"n_alpha={len(canonical_variant.alpha_values)}")
    lines.append("")
    lines.append("Canonical borrow summary")
    for k in [
        "candidate_pass",
        "transition_status",
        "alpha_transition_lo",
        "alpha_transition_hi",
        "alpha_transition_guess",
        "left_edge_p_sustained",
        "right_edge_p_sustained",
        "left_edge_late_update_count_mean",
        "right_edge_late_update_count_mean",
        "p_sustained_violations",
        "chi_peak_ratio",
    ]:
        if k in canonical_candidate:
            lines.append(f"{k}={canonical_candidate[k]}")
    lines.append("")
    lines.append("Flatline resolution classification")
    for k, v in classification.items():
        lines.append(f"{k}={v}")
    lines.append("")
    if diagnostic_rows is not None and not diagnostic_rows.empty:
        lines.append("Diagnostic borrow variants")
        for _, row in diagnostic_rows.sort_values(["promoted_to_certify", "score"], ascending=[False, False]).iterrows():
            lines.append(
                f"variant={row['variant_name']} label_type={row['label_type']} candidate_pass={int(row['candidate_pass'])} "
                f"transition_status={row['transition_status']} score={row['score']:.6f} "
                f"onset_p50={row['onset_p50']} right_edge_p_sustained={row['right_edge_p_sustained']}"
            )
    lines.append("")
    if promoted_variant is not None:
        lines.append(f"promoted_variant_for_full_seed_certification={promoted_variant.name}")
    else:
        lines.append("promoted_variant_for_full_seed_certification=none")
    lines.append("")
    lines.append("Graph package")
    lines.append(f"graph_dir={graph_dir}")
    lines.append("Headline graph: p_sustained_canonical_shared_vs_extended.png")
    lines.append("Borrow-only disambiguation graph: p_sustained_borrow_variants_extended.png")
    lines.append("Mechanism graphs: borrow_recharge_vs_charge.png, borrow_break_even_alpha.png, borrow_debt_floor_frac_extended.png")
    lines.append("")
    lines.append("Interpretation rule")
    lines.append("Canonical Stage-E borrow claims must be based on the canonical borrow family only.")
    lines.append("Diagnostic borrow variants are mechanism ablations used to explain canonical flatlines; they are not replacements for the canonical Stage-E comparison.")
    (out_root / "BORROW_DISAMBIGUATION_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Borrow-flatline disambiguation for the finalized N=3 Stage-E candidate.")
    p.add_argument("mode", choices=["run", "graph", "all"], help="run computations, rebuild graphs, or do both")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--intervention-script", type=str, default="n3_stageE_candidate_interventions.py")
    p.add_argument("--base-script", type=str, default="n3_engineered_phase_transition.py")
    p.add_argument("--stagee-script", type=str, default="stageE_clipped_robustness_extension.py")
    p.add_argument("--selected-attempt-json", type=str, default=None)
    p.add_argument("--existing-intervention-root", type=Path, default=None)
    p.add_argument("--canonical-alpha-max", type=float, default=DEFAULT_CANONICAL_ALPHA_MAX)
    p.add_argument("--canonical-ext-mid", type=float, default=DEFAULT_CANONICAL_EXT_MID)
    p.add_argument("--canonical-fine-step", type=float, default=DEFAULT_CANONICAL_EXT_STEP_FINE)
    p.add_argument("--canonical-coarse-step", type=float, default=DEFAULT_CANONICAL_EXT_STEP_COARSE)
    p.add_argument("--late-frac", type=float, default=DEFAULT_LATE_FRAC)
    p.add_argument("--canonical-seed-start", type=int, default=DEFAULT_CANONICAL_SEED_START)
    p.add_argument("--canonical-seed-stop", type=int, default=DEFAULT_CANONICAL_SEED_STOP)
    p.add_argument("--probe-seed-start", type=int, default=DEFAULT_PROBE_SEED_START)
    p.add_argument("--probe-seed-stop", type=int, default=DEFAULT_PROBE_SEED_STOP)
    p.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    p.add_argument("--borrow-limit", type=float, default=DEFAULT_BORROW_LIMIT)
    p.add_argument("--borrow-interest", type=float, default=DEFAULT_BORROW_INTEREST)
    p.add_argument("--small-limit", type=float, default=DEFAULT_SMALL_LIMIT)
    p.add_argument("--half-interest", type=float, default=DEFAULT_HALF_INTEREST)
    p.add_argument("--support-mode", type=str, default=DEFAULT_SUPPORT_MODE)
    p.add_argument("--support-total", type=float, default=DEFAULT_SUPPORT_TOTAL)
    p.add_argument("--support-start-update", type=int, default=DEFAULT_SUPPORT_START_UPDATE)
    p.add_argument("--support-window-frac", type=float, default=DEFAULT_SUPPORT_WINDOW_FRAC)
    p.add_argument("--skip-diagnostic-probes", action="store_true")
    p.add_argument("--skip-promotion", action="store_true")
    p.add_argument("--force-repatch", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    return p.parse_args()


def run_family_with_diagnostics(bridge_mod: Any, phase_mod: Any, repo_root: Path, mechanism: Any, family_spec: Any, alpha_values: Sequence[float], seed_start: int, seed_stop: int, total_timesteps: int, late_frac: float, out_dir: Path, force_repatch: bool, force_rerun: bool) -> pd.DataFrame:
    summary_df = bridge_mod.run_family_alpha_sweep(
        repo_root=repo_root,
        base_module=phase_mod,
        mechanism=mechanism,
        family=family_spec,
        alpha_values=alpha_values,
        total_timesteps=int(total_timesteps),
        seed_start=int(seed_start),
        seed_stop=int(seed_stop),
        late_frac=float(late_frac),
        out_dir=out_dir,
        force_repatch=bool(force_repatch),
        force_rerun=bool(force_rerun),
    )
    diag_rows: List[Dict[str, Any]] = []
    for alpha in [float(a) for a in alpha_values]:
        alpha_dir = out_dir / f"alpha_{alpha:g}"
        trace_path = alpha_dir / "trace_per_seed_long.csv"
        if not trace_path.exists():
            continue
        trace_long = coerce_numeric(pd.read_csv(trace_path))
        seed_diag = compute_trace_late_metrics(trace_long, alpha=float(alpha), late_frac=float(late_frac), borrow_limit=float(family_spec.borrow_limit))
        seed_diag.to_csv(alpha_dir / "borrow_seed_late_diagnostics.csv", index=False)
        agg = aggregate_trace_late_metrics(seed_diag)
        agg["alpha"] = float(alpha)
        diag_rows.append(agg)
    if diag_rows:
        diag_df = coerce_numeric(pd.DataFrame(diag_rows).sort_values("alpha"))
        diag_df.to_csv(out_dir / "borrow_alpha_diagnostics.csv", index=False)
        merged = summary_df.merge(diag_df, on="alpha", how="left")
    else:
        merged = summary_df.copy()
    merged = coerce_numeric(merged)
    merged.to_csv(out_dir / "alpha_sweep_phase_summary_with_borrow_diagnostics.csv", index=False)
    return merged


def select_promoted_probe(rows: pd.DataFrame) -> Optional[str]:
    if rows.empty:
        return None
    cands = rows.copy()
    cands = cands.loc[cands["label_type"] == "diagnostic"]
    if cands.empty:
        return None
    cands = cands.sort_values(["candidate_pass", "right_edge_p_sustained", "score"], ascending=[False, False, False])
    top = cands.iloc[0]
    # Promote only if it is actually informative.
    if int(top.get("candidate_pass", 0)) == 1 or float(top.get("right_edge_p_sustained", 0.0)) >= 0.5:
        return str(top["variant_name"])
    return None


def load_selected_attempt(bridge_mod: Any, repo_root: Path, args: argparse.Namespace) -> Tuple[Path, Dict[str, Any], pd.DataFrame]:
    selected_json = bridge_mod.locate_finalize_artifacts(repo_root, args.selected_attempt_json)
    payload = json.loads(selected_json.read_text(encoding="utf-8"))
    summary_csv = bridge_mod.resolve_selected_summary_csv(repo_root, selected_json, payload)
    source_summary = coerce_numeric(pd.read_csv(summary_csv))
    return selected_json, payload, source_summary


def build_context_plots(graph_dir: Path, existing_base: Optional[pd.DataFrame], existing_support: Optional[pd.DataFrame], existing_borrow_shared: Optional[pd.DataFrame], canonical_extended: pd.DataFrame, selected_attempt: Mapping[str, Any]) -> None:
    context_map: Dict[str, pd.DataFrame] = {}
    if existing_base is not None:
        context_map["base(shared)"] = existing_base
    if existing_support is not None:
        context_map["support(shared)"] = existing_support
    if existing_borrow_shared is not None:
        context_map["borrow(shared)"] = existing_borrow_shared
    context_map["borrow(canonical extended)"] = canonical_extended

    band = None
    lo = selected_attempt.get("raw_alpha_transition_lo")
    hi = selected_attempt.get("raw_alpha_transition_hi")
    try:
        band = (float(lo), float(hi)) if lo is not None and hi is not None else None
    except Exception:
        band = None
    family_line_plot(context_map, "p_sustained_mean", "Finalized candidate: canonical borrow disambiguation in context", "P_sustained", graph_dir / "p_sustained_canonical_shared_vs_extended.png", x_max=None, vline=0.108, band=band)
    family_line_plot(context_map, "rho_mean", "Finalized candidate: late-window mean activity in context", r"$\rho$", graph_dir / "rho_canonical_shared_vs_extended.png", x_max=None, vline=0.108, band=band)
    family_line_plot(context_map, "eval_return_end_mean_mean", "Finalized candidate: deterministic eval return in context", "eval return", graph_dir / "eval_return_canonical_shared_vs_extended.png", x_max=None, vline=0.108, band=band)


def build_variant_graphs(graph_dir: Path, variant_summaries: Mapping[str, pd.DataFrame], selected_attempt: Mapping[str, Any]) -> None:
    lo = float(selected_attempt.get("raw_alpha_transition_lo", np.nan))
    hi = float(selected_attempt.get("raw_alpha_transition_hi", np.nan))
    band = (lo, hi) if np.isfinite(lo) and np.isfinite(hi) else None
    family_line_plot(variant_summaries, "p_sustained_mean", "Borrow flatline disambiguation: sustained-active fraction", "P_sustained", graph_dir / "p_sustained_borrow_variants_extended.png", band=band)
    family_line_plot(variant_summaries, "p_bursty_mean", "Borrow flatline disambiguation: bursty-only fraction", "P_bursty", graph_dir / "p_bursty_borrow_variants_extended.png", band=band)
    family_line_plot(variant_summaries, "rho_mean", "Borrow flatline disambiguation: late-window mean activity", r"$\rho$", graph_dir / "rho_borrow_variants_extended.png", band=band)
    family_line_plot(variant_summaries, "late_update_count_mean", "Borrow flatline disambiguation: mean late updates", "late updates", graph_dir / "late_updates_borrow_variants_extended.png", band=band)
    family_line_plot(variant_summaries, "late_debt_floor_frac_mean", "Borrow flatline disambiguation: debt-floor occupancy", "late debt-floor fraction", graph_dir / "borrow_debt_floor_frac_extended.png", band=band)
    family_line_plot(variant_summaries, "late_margin_mean_mean", "Borrow flatline disambiguation: recharge minus debt-charge margin", r"$\alpha g_{eff} - r\,\max(-E,0)$", graph_dir / "borrow_margin_extended.png", band=band)
    plot_recharge_vs_charge(variant_summaries, graph_dir / "borrow_recharge_vs_charge.png")
    plot_break_even(variant_summaries, graph_dir / "borrow_break_even_alpha.png")


def mode_run(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    # Load the validated intervention bridge module and the base phase-transition module.
    intervention_script = (repo_root / args.intervention_script).resolve() if not Path(args.intervention_script).is_absolute() else Path(args.intervention_script).resolve()
    bridge_mod = load_module_from_path("_borrow_flatline_bridge", intervention_script, repo_root)
    repo_root = bridge_mod.infer_repo_root(repo_root)
    base_script = bridge_mod.find_script(repo_root, args.base_script, "n3_engineered_phase_transition.py")
    stagee_script = bridge_mod.find_script(repo_root, args.stagee_script, "stageE_clipped_robustness_extension.py")
    phase_mod = load_module_from_path("_borrow_flatline_phase", base_script, repo_root)

    selected_json, selected_attempt, source_summary = load_selected_attempt(bridge_mod, repo_root, args)

    # Lock mechanism from the finalized candidate.
    mechanism = phase_mod.MechanismSpec(
        c_upd=float(selected_attempt.get("raw_c_upd", selected_attempt.get("c_upd", 0.58))),
        z_cut=float(selected_attempt.get("raw_z_cut", selected_attempt.get("z_cut", 1.0))),
        depr_lambda=float(selected_attempt.get("raw_depr_lambda", selected_attempt.get("depr_lambda", 0.0))),
        depr_nu=float(selected_attempt.get("raw_depr_nu", selected_attempt.get("depr_nu", 0.5))),
        depr_psi=float(selected_attempt.get("raw_depr_psi", selected_attempt.get("depr_psi", 0.0))),
    )

    shared_alpha_values = bridge_mod.build_intervention_alpha_grid(selected_attempt, source_summary, 0.0, 0.108, 0.003)
    canonical_alpha_values = build_extended_canonical_alpha_grid(shared_alpha_values, float(args.canonical_alpha_max), float(args.canonical_ext_mid), float(args.canonical_fine_step), float(args.canonical_coarse_step))
    probe_alpha_values = build_probe_alpha_grid(shared_alpha_values, float(args.canonical_alpha_max), float(args.canonical_ext_mid), float(args.canonical_fine_step), float(args.canonical_coarse_step))

    out_root = repo_root / "runs" / "n3_stageE_borrow_flatline_disambiguation"
    ensure_dir(out_root)
    graph_dir = out_root / "graph_package"
    ensure_dir(graph_dir)

    prov = {
        "script_version": SCRIPT_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "repo_root": str(repo_root),
        "intervention_script_path": str(intervention_script),
        "base_script_path": str(base_script),
        "stagee_script_path": str(stagee_script),
        "selected_attempt_json_path": str(selected_json),
        "selected_attempt_json_sha256": bridge_mod.sha256_file(selected_json),
        "python_version": sys.version,
        "platform": platform.platform(),
        "shared_alpha_values": shared_alpha_values,
        "canonical_extended_alpha_values": canonical_alpha_values,
        "probe_alpha_values": probe_alpha_values,
        "args": vars(args),
    }
    (out_root / "provenance.json").write_text(bridge_mod.dumps_json(prov, indent=2, sort_keys=True), encoding="utf-8")

    existing_root = Path(args.existing_intervention_root).resolve() if args.existing_intervention_root is not None else (repo_root / "runs" / "n3_stageE_candidate_interventions")
    existing_base = try_load_existing_family_summary(existing_root, "base")
    existing_support = try_load_existing_family_summary(existing_root, "support")
    existing_borrow_shared = try_load_existing_family_summary(existing_root, "borrow")

    # Canonical borrow extension at full seeds.
    canonical_variant = BorrowVariant(
        name="borrow_canonical_extended",
        borrow_limit=float(args.borrow_limit),
        borrow_interest=float(args.borrow_interest),
        alpha_values=tuple(canonical_alpha_values),
        seed_start=int(args.canonical_seed_start),
        seed_stop=int(args.canonical_seed_stop),
        total_timesteps=int(args.total_timesteps),
        description="Canonical Stage-E borrow semantics on a right-extended alpha grid.",
        certify_priority=100,
        label_type="canonical",
    )
    canonical_dir = out_root / canonical_variant.name
    canonical_summary = run_family_with_diagnostics(
        bridge_mod,
        phase_mod,
        repo_root,
        mechanism,
        canonical_variant.family_spec(bridge_mod),
        canonical_variant.alpha_values,
        canonical_variant.seed_start,
        canonical_variant.seed_stop,
        canonical_variant.total_timesteps,
        float(args.late_frac),
        canonical_dir,
        bool(args.force_repatch),
        bool(args.force_rerun),
    )
    canonical_candidate = json.loads((canonical_dir / "candidate_metrics.json").read_text(encoding="utf-8"))

    variant_rows: List[Dict[str, Any]] = []
    variant_rows.append(
        {
            "variant_name": canonical_variant.name,
            "label_type": canonical_variant.label_type,
            "borrow_limit": canonical_variant.borrow_limit,
            "borrow_interest": canonical_variant.borrow_interest,
            "candidate_pass": int(canonical_candidate.get("candidate_pass", 0)),
            "transition_status": str(canonical_candidate.get("transition_status", "unknown")),
            "alpha_transition_guess": float(canonical_candidate.get("alpha_transition_guess", np.nan)),
            "right_edge_p_sustained": float(canonical_candidate.get("right_edge_p_sustained", np.nan)),
            "chi_peak_ratio": float(canonical_candidate.get("chi_peak_ratio", np.nan)),
            "p_sustained_violations": float(canonical_candidate.get("p_sustained_violations", np.nan)),
            "onset_p50": onset_alpha(canonical_summary, 0.50),
            "score": variant_score(canonical_summary, canonical_candidate),
            "promoted_to_certify": 1,
        }
    )

    probe_summaries: Dict[str, pd.DataFrame] = {}
    probe_candidates: Dict[str, Dict[str, Any]] = {}
    if not args.skip_diagnostic_probes and int(canonical_candidate.get("candidate_pass", 0)) == 0:
        probe_variants = [
            BorrowVariant(
                name="borrow_no_interest_probe",
                borrow_limit=float(args.borrow_limit),
                borrow_interest=0.0,
                alpha_values=tuple(probe_alpha_values),
                seed_start=int(args.probe_seed_start),
                seed_stop=int(args.probe_seed_stop),
                total_timesteps=int(args.total_timesteps),
                description="Diagnostic probe: same borrow limit, zero interest.",
                certify_priority=50,
                label_type="diagnostic",
            ),
            BorrowVariant(
                name="borrow_half_interest_probe",
                borrow_limit=float(args.borrow_limit),
                borrow_interest=float(args.half_interest),
                alpha_values=tuple(probe_alpha_values),
                seed_start=int(args.probe_seed_start),
                seed_stop=int(args.probe_seed_stop),
                total_timesteps=int(args.total_timesteps),
                description="Diagnostic probe: same borrow limit, half interest.",
                certify_priority=40,
                label_type="diagnostic",
            ),
            BorrowVariant(
                name="borrow_small_limit_probe",
                borrow_limit=float(args.small_limit),
                borrow_interest=float(args.borrow_interest),
                alpha_values=tuple(probe_alpha_values),
                seed_start=int(args.probe_seed_start),
                seed_stop=int(args.probe_seed_stop),
                total_timesteps=int(args.total_timesteps),
                description="Diagnostic probe: reduced borrow limit, canonical interest.",
                certify_priority=30,
                label_type="diagnostic",
            ),
            BorrowVariant(
                name="borrow_small_limit_no_interest_probe",
                borrow_limit=float(args.small_limit),
                borrow_interest=0.0,
                alpha_values=tuple(probe_alpha_values),
                seed_start=int(args.probe_seed_start),
                seed_stop=int(args.probe_seed_stop),
                total_timesteps=int(args.total_timesteps),
                description="Diagnostic probe: reduced borrow limit, zero interest.",
                certify_priority=20,
                label_type="diagnostic",
            ),
        ]
        for variant in probe_variants:
            out_dir = out_root / variant.name
            sdf = run_family_with_diagnostics(
                bridge_mod,
                phase_mod,
                repo_root,
                mechanism,
                variant.family_spec(bridge_mod),
                variant.alpha_values,
                variant.seed_start,
                variant.seed_stop,
                variant.total_timesteps,
                float(args.late_frac),
                out_dir,
                bool(args.force_repatch),
                bool(args.force_rerun),
            )
            cand = json.loads((out_dir / "candidate_metrics.json").read_text(encoding="utf-8"))
            probe_summaries[variant.name] = sdf
            probe_candidates[variant.name] = cand
            variant_rows.append(
                {
                    "variant_name": variant.name,
                    "label_type": variant.label_type,
                    "borrow_limit": variant.borrow_limit,
                    "borrow_interest": variant.borrow_interest,
                    "candidate_pass": int(cand.get("candidate_pass", 0)),
                    "transition_status": str(cand.get("transition_status", "unknown")),
                    "alpha_transition_guess": float(cand.get("alpha_transition_guess", np.nan)),
                    "right_edge_p_sustained": float(cand.get("right_edge_p_sustained", np.nan)),
                    "chi_peak_ratio": float(cand.get("chi_peak_ratio", np.nan)),
                    "p_sustained_violations": float(cand.get("p_sustained_violations", np.nan)),
                    "onset_p50": onset_alpha(sdf, 0.50),
                    "score": variant_score(sdf, cand),
                    "promoted_to_certify": 0,
                }
            )

    variant_df = coerce_numeric(pd.DataFrame(variant_rows))
    promoted_name: Optional[str] = None
    promoted_variant: Optional[BorrowVariant] = None
    promoted_summary: Optional[pd.DataFrame] = None
    promoted_candidate: Optional[Dict[str, Any]] = None
    if not args.skip_promotion and not variant_df.empty and int(canonical_candidate.get("candidate_pass", 0)) == 0:
        promoted_name = select_promoted_probe(variant_df)
        if promoted_name is not None:
            if promoted_name == canonical_variant.name:
                promoted_variant = canonical_variant
                promoted_summary = canonical_summary
                promoted_candidate = canonical_candidate
            else:
                # reconstruct and certify the chosen diagnostic variant at full seeds.
                row = variant_df.loc[variant_df["variant_name"] == promoted_name].iloc[0]
                promoted_variant = BorrowVariant(
                    name=promoted_name.replace("_probe", "_certified"),
                    borrow_limit=float(row["borrow_limit"]),
                    borrow_interest=float(row["borrow_interest"]),
                    alpha_values=tuple(probe_summaries[promoted_name]["alpha"].astype(float).tolist()),
                    seed_start=int(args.canonical_seed_start),
                    seed_stop=int(args.canonical_seed_stop),
                    total_timesteps=int(args.total_timesteps),
                    description=f"Certified rerun of diagnostic variant {promoted_name}.",
                    certify_priority=10,
                    label_type="diagnostic_certified",
                )
                promoted_dir = out_root / promoted_variant.name
                promoted_summary = run_family_with_diagnostics(
                    bridge_mod,
                    phase_mod,
                    repo_root,
                    mechanism,
                    promoted_variant.family_spec(bridge_mod),
                    promoted_variant.alpha_values,
                    promoted_variant.seed_start,
                    promoted_variant.seed_stop,
                    promoted_variant.total_timesteps,
                    float(args.late_frac),
                    promoted_dir,
                    bool(args.force_repatch),
                    bool(args.force_rerun),
                )
                promoted_candidate = json.loads((promoted_dir / "candidate_metrics.json").read_text(encoding="utf-8"))
                variant_df.loc[variant_df["variant_name"] == promoted_name, "promoted_to_certify"] = 1
                variant_rows.append(
                    {
                        "variant_name": promoted_variant.name,
                        "label_type": promoted_variant.label_type,
                        "borrow_limit": promoted_variant.borrow_limit,
                        "borrow_interest": promoted_variant.borrow_interest,
                        "candidate_pass": int(promoted_candidate.get("candidate_pass", 0)),
                        "transition_status": str(promoted_candidate.get("transition_status", "unknown")),
                        "alpha_transition_guess": float(promoted_candidate.get("alpha_transition_guess", np.nan)),
                        "right_edge_p_sustained": float(promoted_candidate.get("right_edge_p_sustained", np.nan)),
                        "chi_peak_ratio": float(promoted_candidate.get("chi_peak_ratio", np.nan)),
                        "p_sustained_violations": float(promoted_candidate.get("p_sustained_violations", np.nan)),
                        "onset_p50": onset_alpha(promoted_summary, 0.50),
                        "score": variant_score(promoted_summary, promoted_candidate),
                        "promoted_to_certify": 1,
                    }
                )
                variant_df = coerce_numeric(pd.DataFrame(variant_rows))

    classification = classify_flatline(canonical_summary, canonical_candidate, probe_summaries, probe_candidates)
    classification_path = out_root / "borrow_flatline_questions.json"
    classification_path.write_text(bridge_mod.dumps_json(classification, indent=2, sort_keys=True), encoding="utf-8")

    variant_df.to_csv(out_root / "borrow_variant_summary.csv", index=False)

    # Graphs.
    build_context_plots(graph_dir, existing_base, existing_support, existing_borrow_shared, canonical_summary, selected_attempt)
    variant_summary_map: Dict[str, pd.DataFrame] = {canonical_variant.name: canonical_summary}
    variant_summary_map.update(probe_summaries)
    if promoted_variant is not None and promoted_summary is not None:
        variant_summary_map[promoted_variant.name] = promoted_summary
    build_variant_graphs(graph_dir, variant_summary_map, selected_attempt)

    # Representative traces.
    trace_specs: List[Tuple[str, Path, str, float]] = []
    # canonical lowest alpha and highest alpha always
    for alpha_val, strat, label in [
        (float(min(canonical_variant.alpha_values)), "most_dead", "canonical_low"),
        (float(max(canonical_variant.alpha_values)), "most_active", "canonical_high"),
    ]:
        alpha_dir = canonical_dir / f"alpha_{alpha_val:g}"
        pick = pick_trace(alpha_dir, strat, canonical_variant.borrow_limit)
        if pick is not None:
            seed, trace_csv = pick
            out_path = graph_dir / f"representative_trace_{label}_alpha_{alpha_val:.3f}_seed_{seed:02d}.png"
            save_trace_plot(phase_mod, trace_csv, f"{label} (canonical borrow, alpha={alpha_val:.3f}, seed={seed:02d})", out_path)
            trace_specs.append((label, out_path, str(seed), alpha_val))

    if promoted_variant is not None and promoted_summary is not None:
        alpha_mid = onset_alpha(promoted_summary, 0.50)
        if not np.isfinite(alpha_mid):
            alpha_mid = float(promoted_summary["alpha"].iloc[len(promoted_summary)//2])
        promoted_dir = out_root / promoted_variant.name
        alpha_dir = promoted_dir / f"alpha_{alpha_mid:g}"
        if not alpha_dir.exists():
            # match nearest alpha dir
            alpha_dirs = sorted([p for p in promoted_dir.iterdir() if p.is_dir() and p.name.startswith("alpha_")], key=lambda p: abs(float(p.name.split("alpha_")[1]) - alpha_mid))
            alpha_dir = alpha_dirs[0] if alpha_dirs else alpha_dir
        if alpha_dir.exists():
            pick = pick_trace(alpha_dir, "most_active", promoted_variant.borrow_limit)
            if pick is not None:
                seed, trace_csv = pick
                out_path = graph_dir / f"representative_trace_{promoted_variant.name}_seed_{seed:02d}.png"
                save_trace_plot(phase_mod, trace_csv, f"{promoted_variant.name} (alpha≈{alpha_mid:.3f}, seed={seed:02d})", out_path)
                trace_specs.append((promoted_variant.name, out_path, str(seed), alpha_mid))

    write_report(out_root, selected_attempt, canonical_variant, canonical_candidate, variant_df, classification, graph_dir, promoted_variant)
    print(f"[borrow-disambiguation] wrote outputs to {out_root}")


def mode_graph(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    out_root = repo_root / "runs" / "n3_stageE_borrow_flatline_disambiguation"
    graph_dir = out_root / "graph_package"
    ensure_dir(graph_dir)

    selected_json = None
    try:
        selected_json = next(out_root.glob("../n3_locked_candidate_transition_finalize/selected_attempt.json"))
    except StopIteration:
        pass

    selected_attempt = {}
    if selected_json is not None and selected_json.exists():
        selected_attempt = json.loads(selected_json.read_text(encoding="utf-8"))

    existing_root = Path(args.existing_intervention_root).resolve() if args.existing_intervention_root is not None else (repo_root / "runs" / "n3_stageE_candidate_interventions")
    existing_base = try_load_existing_family_summary(existing_root, "base")
    existing_support = try_load_existing_family_summary(existing_root, "support")
    existing_borrow_shared = try_load_existing_family_summary(existing_root, "borrow")

    canonical_summary = None
    canonical_dir = out_root / "borrow_canonical_extended"
    if (canonical_dir / "alpha_sweep_phase_summary_with_borrow_diagnostics.csv").exists():
        canonical_summary = coerce_numeric(pd.read_csv(canonical_dir / "alpha_sweep_phase_summary_with_borrow_diagnostics.csv"))
    elif (canonical_dir / "alpha_sweep_phase_summary.csv").exists():
        canonical_summary = coerce_numeric(pd.read_csv(canonical_dir / "alpha_sweep_phase_summary.csv"))
    if canonical_summary is None:
        raise FileNotFoundError("Canonical borrow summary not found. Run mode first.")

    build_context_plots(graph_dir, existing_base, existing_support, existing_borrow_shared, canonical_summary, selected_attempt)

    variant_summary_map: Dict[str, pd.DataFrame] = {"borrow_canonical_extended": canonical_summary}
    for p in out_root.iterdir():
        if not p.is_dir() or p.name == "borrow_canonical_extended" or p.name == "graph_package":
            continue
        sp = p / "alpha_sweep_phase_summary_with_borrow_diagnostics.csv"
        if sp.exists():
            variant_summary_map[p.name] = coerce_numeric(pd.read_csv(sp))
    build_variant_graphs(graph_dir, variant_summary_map, selected_attempt)
    print(f"[borrow-disambiguation] rebuilt graphs in {graph_dir}")


def main() -> None:
    args = parse_args()
    if args.mode in {"run", "all"}:
        mode_run(args)
    if args.mode in {"graph", "all"}:
        mode_graph(args)


if __name__ == "__main__":
    main()
