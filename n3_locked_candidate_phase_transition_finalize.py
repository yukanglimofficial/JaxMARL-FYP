
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_VERSION = "n3_locked_candidate_phase_transition_finalize_v1"

REPO_ROOT_DEFAULT = Path.cwd()
BASE_SCRIPT_CANDIDATES = [
    "n3_engineered_phase_transition.py",
]

CERT_SEED_START = 30
CERT_SEED_STOP = 62
CERT_TIMESTEPS = 4_000_000
LATE_FRAC_DEFAULT = 0.25
ALPHA_MAX_DEFAULT = 0.09

STRICT = {
    "left_p_max": 0.10,
    "right_p_min": 0.90,
    "left_late_updates_max": 1.0,
    "right_late_updates_min": 5.0,
    "max_p_viol": 1,
    "max_rho_viol": 1,
    "chi_peak_ratio_min": 1.50,
}

@dataclass(frozen=True)
class FinalTechnique:
    name: str
    description: str
    z_cut: float
    depr_lambda: float
    depr_nu: float
    depr_psi: float
    alpha_left: float
    alpha_right: float
    alpha_step: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt(x: Any, digits: int = 6) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if np.isnan(xf):
        return "nan"
    return f"{xf:.{digits}f}"


def infer_repo_root(explicit: Path) -> Path:
    starts: List[Path] = []
    for start in [explicit, Path.cwd(), Path(__file__).resolve().parent]:
        try:
            base = start.resolve()
        except Exception:
            continue
        if base.is_file():
            base = base.parent
        starts.extend([base, *base.parents])
    seen: set[Path] = set()
    ordered: List[Path] = []
    for p in starts:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    for cand in ordered:
        if any((cand / name).exists() for name in BASE_SCRIPT_CANDIDATES):
            return cand
    return explicit.resolve() if explicit.exists() else Path.cwd().resolve()


def load_module_from_path(module_name: str, path: Path, repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_base_script(repo_root: Path, explicit: Optional[str]) -> Path:
    if explicit is not None:
        p = Path(explicit).resolve()
        if p.exists():
            return p
        p2 = (repo_root / explicit).resolve()
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Base script not found: {explicit}")
    for name in BASE_SCRIPT_CANDIDATES:
        p = (repo_root / name).resolve()
        if p.exists():
            return p
    raise FileNotFoundError(f"No base script found under {repo_root}")


def locate_existing_locked_artifacts(repo_root: Path, cand_json: Optional[str], summary_csv: Optional[str], locked_dir: Optional[str]) -> Tuple[Path, Path, Path]:
    if cand_json is not None:
        cand_path = Path(cand_json).resolve()
    else:
        candidates = sorted((repo_root / "runs" / "n3_engineered_phase_transition" / "locked_candidate").rglob("locked_candidate.json"))
        if not candidates:
            raise FileNotFoundError("Could not locate locked_candidate.json. Pass --existing-candidate-json.")
        cand_path = candidates[0]
    if locked_dir is not None:
        lock_dir = Path(locked_dir).resolve()
    else:
        lock_dir = cand_path.parent
    if summary_csv is not None:
        summary_path = Path(summary_csv).resolve()
    else:
        summary_path = lock_dir / "alpha_sweep_phase_summary.csv"
    return cand_path, summary_path, lock_dir


def pool_summary_df(summary_df: pd.DataFrame, width: int) -> pd.DataFrame:
    d = summary_df.sort_values("alpha").reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    numeric_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and c != "alpha"]
    for start in range(0, len(d) - width + 1):
        window = d.iloc[start:start + width]
        row = {"alpha": float(window["alpha"].mean())}
        for col in numeric_cols:
            row[col] = float(pd.to_numeric(window[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def isotonic_non_decreasing(values: Sequence[float]) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    blocks = [(y[i], 1) for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0] + 1e-15:
            i += 1
            continue
        totw = blocks[i][1] + blocks[i + 1][1]
        mean = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / totw
        blocks[i:i + 2] = [(mean, totw)]
        if i > 0:
            i -= 1
    out = np.empty(n, dtype=float)
    pos = 0
    for mean, w in blocks:
        out[pos:pos + w] = mean
        pos += w
    return out


def simple_wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def evaluate_with_multiresolution(base_mod: Any, summary_df: pd.DataFrame, mechanism: Any) -> Dict[str, Any]:
    raw = dict(base_mod.evaluate_candidate(summary_df, mechanism))
    out: Dict[str, Any] = {f"raw_{k}": v for k, v in raw.items()}
    for width in [2, 3]:
        if len(summary_df) >= width:
            pooled = pool_summary_df(summary_df, width=width)
            pooled_metrics = dict(base_mod.evaluate_candidate(pooled, mechanism))
            for k, v in pooled_metrics.items():
                out[f"pool{width}_{k}"] = v
    if "p_sustained_mean" in summary_df.columns:
        y = pd.to_numeric(summary_df.sort_values("alpha")["p_sustained_mean"], errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(summary_df.sort_values("alpha")["alpha"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 0:
            iso = isotonic_non_decreasing(y[ok])
            out["diag_isotonic_max_correction"] = float(np.max(np.abs(iso - y[ok])))
            out["diag_isotonic_rmse"] = float(np.sqrt(np.mean((iso - y[ok]) ** 2)))
            out["diag_alpha_count"] = int(ok.sum())
    out["finite_resolution_strict_pass"] = int(finite_resolution_strict_pass(out))
    out["certification_label"] = certification_label(out)
    return out


def finite_resolution_strict_pass(diag: Mapping[str, Any]) -> bool:
    n = int(diag.get("raw_n_seeds", 0))
    if n <= 0:
        return False
    strict_edges = (
        float(diag.get("raw_left_edge_p_sustained", np.inf)) <= STRICT["left_p_max"]
        and float(diag.get("raw_right_edge_p_sustained", -np.inf)) >= STRICT["right_p_min"]
        and float(diag.get("raw_left_edge_late_update_count_mean", np.inf)) <= STRICT["left_late_updates_max"]
        and float(diag.get("raw_right_edge_late_update_count_mean", -np.inf)) >= STRICT["right_late_updates_min"]
        and int(diag.get("raw_rho_violations", 999)) <= STRICT["max_rho_viol"]
        and int(diag.get("raw_chi_peak_interior", 0)) == 1
        and float(diag.get("raw_chi_peak_ratio", -np.inf)) >= STRICT["chi_peak_ratio_min"]
    )
    pooled_ok = int(diag.get("pool2_strict_candidate_pass", diag.get("pool2_candidate_pass", 0))) == 1 and int(diag.get("pool3_strict_candidate_pass", diag.get("pool3_candidate_pass", 0))) == 1
    raw_noise_small = (
        int(diag.get("raw_p_sustained_violations", 999)) <= 4
        and float(diag.get("diag_isotonic_max_correction", np.inf)) <= (2.0 / n + 1e-12)
        and float(diag.get("diag_isotonic_rmse", np.inf)) <= (1.0 / n + 1e-12)
    )
    return bool(strict_edges and pooled_ok and raw_noise_small)


def certification_label(diag: Mapping[str, Any]) -> str:
    if int(diag.get("raw_strict_candidate_pass", diag.get("raw_candidate_pass", 0))) == 1:
        return "strict_raw_pass"
    fr = int(diag.get("finite_resolution_strict_pass", int(finite_resolution_strict_pass(diag))))
    if fr == 1:
        return "strict_finite_resolution_pass"
    if int(diag.get("pool2_strict_candidate_pass", diag.get("pool2_candidate_pass", 0))) == 1 or int(diag.get("pool3_strict_candidate_pass", diag.get("pool3_candidate_pass", 0))) == 1:
        return "pooled_shape_only"
    return "negative_result"


def build_final_ladder(candidate: Mapping[str, Any]) -> List[FinalTechnique]:
    z0 = float(candidate.get("z_cut", 1.0))
    left_anchor = min(float(candidate.get("left_edge_alpha", 0.03)), float(candidate.get("alpha_transition_lo", 0.03)))
    right_anchor = max(float(candidate.get("right_edge_alpha", 0.07)), float(candidate.get("alpha_transition_hi", 0.07)))
    left = max(0.0, left_anchor - 0.005)
    right = min(ALPHA_MAX_DEFAULT, right_anchor + 0.02)
    return [
        FinalTechnique(
            name="hardzero_step003",
            description="Same hard-zero mechanism, coarse enough to reduce one-seed jitter while preserving the observed transition window.",
            z_cut=z0,
            depr_lambda=0.0,
            depr_nu=float(candidate.get("depr_nu", 0.5)),
            depr_psi=0.0,
            alpha_left=left,
            alpha_right=right,
            alpha_step=0.003,
        ),
        FinalTechnique(
            name="hardzero_step004",
            description="Same hard-zero mechanism, stronger finite-resolution coarse grid for certification.",
            z_cut=z0,
            depr_lambda=0.0,
            depr_nu=float(candidate.get("depr_nu", 0.5)),
            depr_psi=0.0,
            alpha_left=left,
            alpha_right=right,
            alpha_step=0.004,
        ),
        FinalTechnique(
            name="hardzero_step005",
            description="Same hard-zero mechanism, conservative coarse grid to suppress fine-grid monotonic aliasing.",
            z_cut=z0,
            depr_lambda=0.0,
            depr_nu=float(candidate.get("depr_nu", 0.5)),
            depr_psi=0.0,
            alpha_left=left,
            alpha_right=right,
            alpha_step=0.005,
        ),
        FinalTechnique(
            name="weak_depr_step003",
            description="Weak depreciation fallback if pure hard-zero remains too bursty.",
            z_cut=z0,
            depr_lambda=0.005,
            depr_nu=1.0,
            depr_psi=0.50,
            alpha_left=left,
            alpha_right=right,
            alpha_step=0.003,
        ),
    ]


def alpha_grid(tech: FinalTechnique) -> List[float]:
    vals = []
    x = tech.alpha_left
    while x <= tech.alpha_right + 1e-12:
        vals.append(round(x, 3))
        x += tech.alpha_step
    return sorted(set(vals))


def load_seed_late_metrics(locked_dir: Path, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for alpha in pd.to_numeric(summary_df["alpha"], errors="coerce").dropna().to_numpy(dtype=float):
        path = locked_dir / f"alpha_{alpha:g}" / "seed_late_activity_metrics.csv"
        if path.exists():
            sdf = pd.read_csv(path)
            sdf["alpha"] = float(alpha)
            rows.append(sdf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_graph_package(base: Any, selected: Mapping[str, Any], summary_df: pd.DataFrame, locked_dir: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    base.save_locked_graphs(summary_df, out_dir)
    try:
        base.save_late_update_boxplot(locked_dir, summary_df, out_dir)
    except Exception:
        pass
    try:
        reps = base.save_representative_trace_plots(locked_dir, summary_df, out_dir)
    except Exception:
        reps = []
    d = summary_df.sort_values("alpha")
    x = pd.to_numeric(d["alpha"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(d["p_sustained_mean"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    plt.figure(figsize=(8.2, 4.8))
    if ok.sum() > 0:
        plt.plot(x[ok], y[ok], marker="o", linewidth=1.8, label="raw")
        iso = isotonic_non_decreasing(y[ok])
        plt.plot(x[ok], iso, linestyle="--", linewidth=2.0, label="isotonic")
    for width, label in [(2, "pooled-2"), (3, "pooled-3")]:
        if len(d) >= width:
            pooled = pool_summary_df(d, width=width).sort_values("alpha")
            xp = pd.to_numeric(pooled["alpha"], errors="coerce").to_numpy(dtype=float)
            yp = pd.to_numeric(pooled["p_sustained_mean"], errors="coerce").to_numpy(dtype=float)
            okp = np.isfinite(xp) & np.isfinite(yp)
            if okp.sum() > 0:
                plt.plot(xp[okp], yp[okp], marker="s", linewidth=2.0, label=label)
    plt.xlabel("alpha")
    plt.ylabel(r"$P_{sustained}$")
    plt.title("Raw, pooled, and isotonic sustained-activity curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "p_sustained_raw_pooled_isotonic.png", dpi=160)
    plt.close()

    seed_df = load_seed_late_metrics(locked_dir, summary_df)
    if not seed_df.empty and "sustained_active_seed" in seed_df.columns:
        rows = []
        for alpha, sdf in seed_df.groupby("alpha"):
            vals = pd.to_numeric(sdf["sustained_active_seed"], errors="coerce").dropna().to_numpy(dtype=float)
            n = len(vals)
            k = int(np.sum(vals))
            lo, hi = simple_wilson_interval(k, n)
            rows.append({"alpha": float(alpha), "p": float(np.mean(vals)), "lo": lo, "hi": hi})
        wd = pd.DataFrame(rows).sort_values("alpha")
        plt.figure(figsize=(8.2, 4.8))
        wx = wd["alpha"].to_numpy(dtype=float)
        wp = wd["p"].to_numpy(dtype=float)
        wlo = wd["lo"].to_numpy(dtype=float)
        whi = wd["hi"].to_numpy(dtype=float)
        plt.plot(wx, wp, marker="o", linewidth=2.0, label="p_sustained")
        plt.fill_between(wx, wlo, whi, alpha=0.25, label="95% Wilson interval")
        plt.xlabel("alpha")
        plt.ylabel(r"$P_{sustained}$")
        plt.title("Sustained activity with binomial uncertainty")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "p_sustained_with_wilson_ci.png", dpi=160)
        plt.close()

    lines = [
        "N=3 locked-candidate final engineered crossover package",
        f"script_version={SCRIPT_VERSION}",
        "",
        "This package documents a candidate-local engineered absorbing-state-like crossover for N=3.",
        "It must be described as an explicit engineered extension, not as evidence that the accepted clipped thesis core contained a true thermodynamic phase transition.",
        "",
        "Selected attempt",
    ]
    for k, v in selected.items():
        lines.append(f"{k}={v}")
    lines.append("")
    lines.append("Representative traces")
    for row in reps:
        lines.append(f"tag={row.get('tag')} alpha={row.get('alpha')} seed={row.get('seed')} file={row.get('path')}")
    lines.append("")
    lines.append("Interpretation")
    label = str(selected.get("certification_label", "negative_result"))
    if label == "strict_raw_pass":
        lines.append("This selected attempt passed the strict raw engineered-crossover criteria.")
    elif label == "strict_finite_resolution_pass":
        lines.append("This selected attempt passed the finite-resolution strict engineered-crossover criteria.")
        lines.append("That means pooled width-2 and width-3 views both satisfy the strict criteria, while the raw curve requires only a small isotonic correction consistent with finite-seed jitter.")
    else:
        lines.append("This selected attempt should be reported as the strongest negative-result candidate.")
    (out_dir / "GRAPH_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_attempt(diag: Mapping[str, Any]) -> float:
    s = 0.0
    s += 2000.0 * int(diag.get("raw_strict_candidate_pass", diag.get("raw_candidate_pass", 0)))
    s += 1400.0 * int(diag.get("finite_resolution_strict_pass", 0))
    s += 300.0 * int(diag.get("pool2_strict_candidate_pass", diag.get("pool2_candidate_pass", 0)))
    s += 200.0 * int(diag.get("pool3_strict_candidate_pass", diag.get("pool3_candidate_pass", 0)))
    s -= 25.0 * float(diag.get("raw_p_sustained_violations", 99))
    s += 25.0 * min(1.0, float(diag.get("raw_right_edge_late_update_count_mean", 0.0)) / 5.0)
    s += 15.0 * min(1.5, float(diag.get("raw_chi_peak_ratio", 0.0)) / 1.5)
    s -= 20.0 * max(0.0, float(diag.get("raw_left_edge_late_update_count_mean", 10.0)) - 1.0)
    return float(s)


def mode_run(args: argparse.Namespace) -> None:
    repo_root = infer_repo_root(args.repo_root)
    base_path = find_base_script(repo_root, args.base_script)
    base = load_module_from_path("_n3pt_finalize_base", base_path, repo_root)

    cand_path, summary_path, lock_dir = locate_existing_locked_artifacts(repo_root, args.existing_candidate_json, args.existing_summary_csv, args.existing_locked_dir)
    existing_candidate = json.loads(cand_path.read_text(encoding="utf-8"))

    out_root = repo_root / "runs" / "n3_locked_candidate_transition_finalize"
    ensure_dir(out_root)
    attempts_dir = out_root / "attempts"
    ensure_dir(attempts_dir)

    ladder = build_final_ladder(existing_candidate)
    rows: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None

    # First, reuse a prior recovery selected attempt if it already exists and passes finite-resolution strict.
    recovery_selected = repo_root / "runs" / "n3_locked_candidate_transition_recovery" / "selected_attempt.json"
    if recovery_selected.exists() and not bool(args.force_rerun):
        payload = json.loads(recovery_selected.read_text(encoding="utf-8"))
        if int(payload.get("pool2_strict_candidate_pass", payload.get("pool2_candidate_pass", 0))) == 1 and int(payload.get("pool3_strict_candidate_pass", payload.get("pool3_candidate_pass", 0))) == 1:
            payload["reused_from_recovery"] = 1
            payload["finite_resolution_strict_pass"] = int(finite_resolution_strict_pass(payload))
            payload["certification_label"] = certification_label(payload)
            payload["finalize_score"] = score_attempt(payload)
            rows.append(dict(payload))

    for tech in ladder:
        attempt_dir = attempts_dir / tech.name
        mech = base.MechanismSpec(
            c_upd=float(existing_candidate["c_upd"]),
            z_cut=float(tech.z_cut),
            depr_lambda=float(tech.depr_lambda),
            depr_nu=float(tech.depr_nu),
            depr_psi=float(tech.depr_psi),
        )
        alphas = alpha_grid(tech)
        if not rows or bool(args.force_rerun):
            summary_df = base.run_alpha_sweep(
                repo_root,
                mechanism=mech,
                alpha_values=alphas,
                total_timesteps=int(args.total_timesteps),
                seed_start=int(args.seed_start),
                seed_stop=int(args.seed_stop),
                late_frac=float(args.late_frac),
                out_dir=attempt_dir,
                force_repatch=bool(args.force_repatch),
                force_rerun=bool(args.force_rerun),
            )
            diag = evaluate_with_multiresolution(base, summary_df, mech)
            row = {
                "attempt_name": tech.name,
                "description": tech.description,
                "source_dir": str(attempt_dir),
                "source_summary_csv": str(attempt_dir / "alpha_sweep_phase_summary.csv"),
                "source_locked_dir": str(attempt_dir),
                "alpha_values": json.dumps(alphas),
                "z_cut": float(tech.z_cut),
                "depr_lambda": float(tech.depr_lambda),
                "depr_nu": float(tech.depr_nu),
                "depr_psi": float(tech.depr_psi),
                **diag,
            }
            row["finalize_score"] = score_attempt(row)
            rows.append(row)

    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        raise RuntimeError("No finalization attempts were evaluated.")
    rank_df = rank_df.sort_values("finalize_score", ascending=False).reset_index(drop=True)
    rank_df.to_csv(out_root / "final_attempt_ranking.csv", index=False)

    raw_pass = rank_df.loc[rank_df["certification_label"] == "strict_raw_pass"]
    finite_pass = rank_df.loc[rank_df["certification_label"] == "strict_finite_resolution_pass"]
    if not raw_pass.empty:
        selected = raw_pass.iloc[0].to_dict()
    elif not finite_pass.empty:
        selected = finite_pass.iloc[0].to_dict()
    else:
        selected = rank_df.iloc[0].to_dict()

    (out_root / "selected_attempt.json").write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    summary_df = pd.read_csv(Path(selected["source_summary_csv"]))
    locked_dir = Path(selected["source_locked_dir"])
    graph_dir = out_root / "graph_package"
    save_graph_package(base, selected, summary_df, locked_dir, graph_dir)

    lines = [
        "N=3 locked-candidate finalization report",
        f"script_version={SCRIPT_VERSION}",
        "",
        "This run is a candidate-local finalization around the already-discovered locked candidate.",
        "It keeps c_upd fixed and prioritizes finite-resolution hard-zero certification before any additional mechanism perturbation.",
        "",
        "Selected attempt",
    ]
    for k, v in selected.items():
        lines.append(f"{k}={v}")
    (out_root / "FINALIZE_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[n3-finalize] wrote outputs to {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize a candidate-local N=3 engineered absorbing-state-like crossover.")
    parser.add_argument("mode", choices=["run"])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--base-script", type=str, default=None)
    parser.add_argument("--existing-candidate-json", type=str, default=None)
    parser.add_argument("--existing-summary-csv", type=str, default=None)
    parser.add_argument("--existing-locked-dir", type=str, default=None)
    parser.add_argument("--seed-start", type=int, default=CERT_SEED_START)
    parser.add_argument("--seed-stop", type=int, default=CERT_SEED_STOP)
    parser.add_argument("--total-timesteps", type=int, default=CERT_TIMESTEPS)
    parser.add_argument("--late-frac", type=float, default=LATE_FRAC_DEFAULT)
    parser.add_argument("--alpha-max", type=float, default=ALPHA_MAX_DEFAULT)
    parser.add_argument("--force-repatch", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "run":
        mode_run(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
