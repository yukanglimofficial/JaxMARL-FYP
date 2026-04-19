#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np


# =========================
# Locked Stage C constants
# =========================
R0 = -27.256548
D0 = 0.733348
R_REF_FULL = -24.779596
R_THRESH_FULL = R0 + 0.5 * (R_REF_FULL - R0)
EPS_OCC = 0.2
M_EVAL = 100
FULL_TIMESTEPS = 1_000_000
NUM_ENVS = 25
NUM_STEPS = 128
BETA = 2.302585092994046  # ln(10)
Z_MAX = 10.0
EPS_NORM = 1e-8
E0 = 1.0
E_MAX = 1.0
PHYSICAL_C_MAX = 1.0
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260318
LOGISTIC_RIDGE = 1e-6
LOGISTIC_MAX_ITERS = 100
LOGISTIC_TOL = 1e-10

SEEDS = list(range(30, 62))  # 32 seeds, locked from the frozen checkpoint

SCRIPT_PATH = Path("baselines/IPPO/ippo_ff_mpe_gate5_energy_gated.py")
CONFIG_PATH = Path("baselines/IPPO/config/ippo_ff_mpe.yaml")
STAGE_B_REPORT = Path("runs/stageB_completion_bridge_reports/STAGE_B_COMPLETION_REPORT.txt")
STAGE_C_READY_REPORT = Path("runs/stageB_completion_bridge_reports/STAGE_C_SUBSET_READINESS_REPORT.txt")
FULLREF_REPORT = Path("runs/stageB_completion_bridge_reports/fullref_32seed_report.txt")
FULLREF_PER_RUN_CSV = Path("runs/stageB_completion_bridge_reports/fullref_32seed_per_run.csv")
STAGEB_ALPHA_CSV = Path("runs/stageB_completion_bridge_reports/stageB_alpha_classification.csv")
STAGEC_CANDIDATE_CSV = Path("runs/stageB_completion_bridge_reports/stageC_candidate_alphas_from_bridge.csv")

TARGET_BASE = Path("runs/stageC_subset_refine_full_32seed")
REPORT_BASE = Path("runs/stageC_subset_refine_reports")

PER_RUN_CSV = REPORT_BASE / "stageC_subset_refine_per_run.csv"
BY_POINT_CSV = REPORT_BASE / "stageC_subset_refine_by_point.csv"
ALPHA_BOUNDARY_CSV = REPORT_BASE / "stageC_subset_refine_alpha_boundary.csv"
MANIFEST_CSV = REPORT_BASE / "stageC_subset_refine_manifest.csv"
MISSING_TXT = REPORT_BASE / "stageC_subset_refine_missing_runs.txt"
REPORT_TXT = REPORT_BASE / "STAGE_C_SUBSET_FULL_HORIZON_REPORT.txt"


# =========================
# Generic helpers
# =========================
def need_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fmt_tag(x: str) -> str:
    return x.replace(".", "p")


def canonical_cli_float(value: float, decimals: int = 3) -> str:
    s = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def canonical_csv_float(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def canonical_alpha_str(value: float) -> str:
    return f"{value:.1f}"


def parse_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("Gate 5 auto-summary"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def safe_float(d: dict[str, str], *keys: str, default: float = float("nan")) -> float:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except ValueError:
                pass
    return default


def read_wall_seconds(run_dir: Path) -> float:
    time_path = run_dir / "time.txt"
    if not time_path.exists():
        return float("nan")
    txt = time_path.read_text(encoding="utf-8").strip()
    if "=" in txt:
        txt = txt.split("=", 1)[1]
    try:
        return float(txt)
    except ValueError:
        return float("nan")


def bootstrap_mean_ci(values: list[float], n_boot: int = BOOTSTRAP_N, seed: int = 0) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for __ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def bootstrap_prob_ci(bits: list[int], n_boot: int = BOOTSTRAP_N, seed: int = 0) -> tuple[float, float]:
    if not bits:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(bits)
    vals = []
    for _ in range(n_boot):
        sample = [bits[rng.randrange(n)] for __ in range(n)]
        vals.append(sum(sample) / n)
    vals.sort()
    lo = vals[int(0.025 * n_boot)]
    hi = vals[int(0.975 * n_boot)]
    return lo, hi


def nanmean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return mean(vals) if vals else float("nan")


def nanstdev(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return stdev(vals) if len(vals) >= 2 else float("nan")


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


# =========================
# Frozen-state loading
# =========================
def load_stageb_alpha_map() -> dict[str, dict[str, str]]:
    need_file(STAGEB_ALPHA_CSV)
    out: dict[str, dict[str, str]] = {}
    with STAGEB_ALPHA_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alpha = canonical_alpha_str(float(row["alpha"]))
            out[alpha] = row
    if len(out) != 11:
        raise RuntimeError(f"Expected 11 alpha classifications in {STAGEB_ALPHA_CSV}, found {len(out)}")
    return out


def load_stagec_manifest() -> list[dict[str, str]]:
    need_file(STAGEC_CANDIDATE_CSV)
    alpha_map = load_stageb_alpha_map()
    rows: list[dict[str, str]] = []
    with STAGEC_CANDIDATE_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alpha = canonical_alpha_str(float(row["alpha"]))
            stageb = alpha_map[alpha]
            candidate_type = row["candidate_type"].strip()
            refine_values = [s.strip() for s in row["proposed_full_horizon_refine_c_values"].split(",") if s.strip()]
            if not refine_values:
                raise RuntimeError(f"No refine values listed for alpha={alpha} in {STAGEC_CANDIDATE_CSV}")
            for c_raw in refine_values:
                c_float = float(c_raw)
                rows.append(
                    {
                        "alpha": alpha,
                        "c_upd": canonical_csv_float(c_float),
                        "alpha_cli": canonical_cli_float(float(alpha), decimals=1),
                        "c_upd_cli": canonical_cli_float(c_float),
                        "candidate_type": candidate_type,
                        "stageB_classification": stageb["classification"],
                        "stageB_enabled_anchor_c": stageb.get("enabled_anchor_c", ""),
                        "stageB_boundary_proxy_c": stageb.get("boundary_proxy_c", ""),
                        "stageB_boundary_proxy_P": stageb.get("boundary_proxy_P", ""),
                        "stageB_below_proxy_c": stageb.get("below_proxy_c", ""),
                        "stageB_below_proxy_P": stageb.get("below_proxy_P", ""),
                        "stageB_in_domain_bracket_lo": stageb.get("in_domain_bracket_lo", ""),
                        "stageB_in_domain_bracket_hi_or_gateoff": stageb.get("in_domain_bracket_hi_or_gateoff", ""),
                        "stageB_note": stageb.get("note", ""),
                    }
                )

    # Frozen-check assertions from the handoff
    alphas = sorted({row["alpha"] for row in rows}, key=float)
    if alphas != [canonical_alpha_str(x / 10.0) for x in range(11)]:
        raise RuntimeError(f"Unexpected Stage C alpha set: {alphas}")

    alpha0_vals = [row["c_upd"] for row in rows if row["alpha"] == "0.0"]
    if alpha0_vals != ["0.520", "0.540", "0.560", "0.580"]:
        raise RuntimeError(f"Frozen alpha=0.0 refine band mismatch: {alpha0_vals}")

    edge_alphas = [a for a in alphas if a != "0.0"]
    for alpha in edge_alphas:
        vals = [row["c_upd"] for row in rows if row["alpha"] == alpha]
        if vals != ["0.900", "0.950", "1.000"]:
            raise RuntimeError(f"Frozen edge refine set mismatch for alpha={alpha}: {vals}")

    if len(rows) != 34:
        raise RuntimeError(f"Expected 34 Stage C point definitions, found {len(rows)}")

    return rows


def verify_fullref_lock() -> dict[str, float]:
    need_file(FULLREF_PER_RUN_CSV)
    rows = []
    with FULLREF_PER_RUN_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 32:
        raise RuntimeError(f"Expected 32 full-reference runs in {FULLREF_PER_RUN_CSV}, found {len(rows)}")
    returns = [float(r["eval_return_end_mean"]) for r in rows]
    mean_return = mean(returns)
    sd_return = stdev(returns) if len(returns) >= 2 else 0.0
    if abs(mean_return - R_REF_FULL) > 1e-6:
        raise RuntimeError(
            f"Locked full-horizon reference mismatch: csv mean={mean_return:.6f} vs locked R_REF_FULL={R_REF_FULL:.6f}"
        )
    lo, hi = bootstrap_mean_ci(returns, seed=4242)
    frac_below_random = sum(r < R0 for r in returns) / len(returns)
    return {
        "mean_return": mean_return,
        "sd_return": sd_return,
        "bootstrap_lo": lo,
        "bootstrap_hi": hi,
        "frac_below_random": frac_below_random,
        "n_runs": float(len(rows)),
    }


# =========================
# Launch helpers
# =========================
def build_run_tag(alpha_cli: str, c_cli: str, seed: int) -> str:
    return f"a{fmt_tag(alpha_cli)}_c{fmt_tag(c_cli)}_seed{seed}"


def build_command(alpha_cli: str, c_cli: str, seed: int, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(SCRIPT_PATH),
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=True",
        "WANDB_MODE=disabled",
        "NUM_SEEDS=1",
        f"SEED={seed}",
        f"NUM_ENVS={NUM_ENVS}",
        f"TOTAL_TIMESTEPS={FULL_TIMESTEPS}",
        f"NUM_STEPS={NUM_STEPS}",
        "ENV_NAME=MPE_simple_spread_v3",
        "++ENV_KWARGS.num_agents=3",
        "++ENV_KWARGS.num_landmarks=3",
        "++ENV_KWARGS.local_ratio=0.5",
        "++ENV_KWARGS.max_steps=25",
        "++ENV_KWARGS.action_type=Discrete",
        f"++EPS_OCC={EPS_OCC}",
        f"++D0={D0}",
        f"++R0={R0}",
        f"++BETA={BETA}",
        f"++Z_MAX={Z_MAX}",
        f"++EPS_NORM={EPS_NORM}",
        "++USE_GEOM_FUNDING=True",
        "++DO_EVAL=True",
        f"++M_EVAL={M_EVAL}",
        f"++C_UPD={c_cli}",
        f"++ALPHA={alpha_cli}",
        f"++E0={E0}",
        f"++E_MAX={E_MAX}",
    ]


def run_command(cmd: list[str], stdout_path: Path, time_path: Path) -> None:
    ensure_dir(stdout_path.parent)
    with stdout_path.open("w", encoding="utf-8") as out:
        proc = subprocess.run(
            ["/usr/bin/time", "-f", "wall_seconds=%e", "-o", str(time_path)] + cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}. See {stdout_path}")


def run_point_seed(manifest_row: dict[str, str], seed: int) -> None:
    alpha_cli = manifest_row["alpha_cli"]
    c_cli = manifest_row["c_upd_cli"]
    tag = build_run_tag(alpha_cli, c_cli, seed)
    run_dir = TARGET_BASE / tag
    summary_path = run_dir / "gate5_summary_AUTOGEN.txt"
    if summary_path.is_file():
        print(f"[stageC skip] {tag} already present")
        return

    print(
        f"[stageC run] {tag} "
        f"(alpha={manifest_row['alpha']} c_upd={manifest_row['c_upd']} candidate_type={manifest_row['candidate_type']})"
    )
    cmd = build_command(alpha_cli=alpha_cli, c_cli=c_cli, seed=seed, run_dir=run_dir)
    run_command(cmd, run_dir / "stdout.log", run_dir / "time.txt")


# =========================
# Boundary fitting helpers
# =========================
def pava_nondecreasing(y: list[float], w: list[float]) -> list[float]:
    blocks: list[dict[str, float | int]] = []
    for idx, (yy, ww) in enumerate(zip(y, w)):
        blocks.append({"start": idx, "end": idx, "sum_w": float(ww), "sum_yw": float(yy) * float(ww)})
        while len(blocks) >= 2:
            m1 = float(blocks[-2]["sum_yw"]) / float(blocks[-2]["sum_w"])
            m2 = float(blocks[-1]["sum_yw"]) / float(blocks[-1]["sum_w"])
            if m1 <= m2 + 1e-15:
                break
            b2 = blocks.pop()
            b1 = blocks.pop()
            blocks.append(
                {
                    "start": int(b1["start"]),
                    "end": int(b2["end"]),
                    "sum_w": float(b1["sum_w"]) + float(b2["sum_w"]),
                    "sum_yw": float(b1["sum_yw"]) + float(b2["sum_yw"]),
                }
            )
    out = [0.0] * len(y)
    for block in blocks:
        level = float(block["sum_yw"]) / float(block["sum_w"])
        for idx in range(int(block["start"]), int(block["end"]) + 1):
            out[idx] = level
    return out


def pava_nonincreasing(y: list[float], w: list[float]) -> list[float]:
    return [-v for v in pava_nondecreasing([-yy for yy in y], w)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic_boundary(x: list[float], y: list[int]) -> dict[str, float | bool | str]:
    if not x or not y or len(x) != len(y):
        return {"valid": False, "reason": "empty_or_mismatched"}
    y_arr = np.asarray(y, dtype=np.float64)
    if np.all(y_arr == y_arr[0]):
        return {"valid": False, "reason": "all_same_class"}
    x_arr = np.asarray(x, dtype=np.float64)
    X = np.column_stack([np.ones_like(x_arr), x_arr])
    p0 = float(np.clip(y_arr.mean(), 1e-4, 1.0 - 1e-4))
    beta = np.array([math.log(p0 / (1.0 - p0)), -1.0], dtype=np.float64)

    for _ in range(LOGISTIC_MAX_ITERS):
        eta = X @ beta
        p = sigmoid(eta)
        w = p * (1.0 - p) + 1e-12
        H = X.T @ (w[:, None] * X) + LOGISTIC_RIDGE * np.eye(2)
        g = X.T @ (y_arr - p) - LOGISTIC_RIDGE * beta
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return {"valid": False, "reason": "singular_hessian"}
        beta_new = beta + delta
        if np.max(np.abs(delta)) < LOGISTIC_TOL:
            beta = beta_new
            break
        beta = beta_new

    intercept = float(beta[0])
    slope = float(beta[1])
    if not math.isfinite(intercept) or not math.isfinite(slope):
        return {"valid": False, "reason": "nonfinite_beta"}
    if slope >= -1e-10:
        return {"valid": False, "reason": "nonnegative_slope", "intercept": intercept, "slope": slope}
    c_star = -intercept / slope
    if not math.isfinite(c_star):
        return {"valid": False, "reason": "nonfinite_c_star", "intercept": intercept, "slope": slope}
    return {
        "valid": True,
        "reason": "ok",
        "intercept": intercept,
        "slope": slope,
        "c_star": float(c_star),
    }


def classify_monotone_boundary(c_values: list[float], probs: list[float]) -> dict[str, float | str | None]:
    tol = 1e-12
    if len(c_values) != len(probs):
        raise ValueError("c_values/probs length mismatch")
    if not c_values:
        return {"classification": "unresolved", "bracket_lo": None, "bracket_hi": None, "c_star_isotonic": None}

    exact_idx = [i for i, p in enumerate(probs) if abs(p - 0.5) <= tol]
    if exact_idx:
        start = exact_idx[0]
        end = exact_idx[-1]
        return {
            "classification": "in_domain_estimate",
            "bracket_lo": float(c_values[start]),
            "bracket_hi": float(c_values[end]),
            "c_star_isotonic": 0.5 * float(c_values[start] + c_values[end]),
        }

    if all(p > 0.5 or abs(p - 0.5) <= tol for p in probs):
        max_c = float(c_values[-1])
        classification = "right_censored_gt_1p0" if max_c >= PHYSICAL_C_MAX - 1e-12 else "right_censored_gt_refine_band"
        return {"classification": classification, "bracket_lo": max_c, "bracket_hi": None, "c_star_isotonic": None}

    if all(p < 0.5 and abs(p - 0.5) > tol for p in probs):
        min_c = float(c_values[0])
        return {"classification": "left_censored_lt_refine_band", "bracket_lo": None, "bracket_hi": min_c, "c_star_isotonic": None}

    for i in range(len(c_values) - 1):
        p_lo = probs[i]
        p_hi = probs[i + 1]
        c_lo = float(c_values[i])
        c_hi = float(c_values[i + 1])
        if p_lo >= 0.5 and p_hi < 0.5:
            if abs(p_hi - p_lo) <= tol:
                c_star = 0.5 * (c_lo + c_hi)
            else:
                frac = (0.5 - p_lo) / (p_hi - p_lo)
                c_star = c_lo + frac * (c_hi - c_lo)
            return {
                "classification": "in_domain_estimate",
                "bracket_lo": c_lo,
                "bracket_hi": c_hi,
                "c_star_isotonic": float(c_star),
            }

    # Safety fallback: monotone sequence with a crossing pattern should have returned above.
    return {"classification": "unresolved", "bracket_lo": None, "bracket_hi": None, "c_star_isotonic": None}


def choose_boundary_estimate(
    c_values: list[float],
    point_bits: dict[float, list[int]],
    monotone_result: dict[str, float | str | None],
) -> tuple[float, str]:
    if monotone_result["classification"] != "in_domain_estimate":
        return float("nan"), "none"

    x: list[float] = []
    y: list[int] = []
    for c in c_values:
        bits = point_bits[float(c)]
        x.extend([float(c)] * len(bits))
        y.extend(int(b) for b in bits)

    logistic = fit_logistic_boundary(x, y)
    iso_c = monotone_result["c_star_isotonic"]
    c_lo = monotone_result["bracket_lo"]
    c_hi = monotone_result["bracket_hi"]
    if logistic.get("valid"):
        c_star = float(logistic["c_star"])
        if c_lo is not None and c_hi is not None and (float(c_lo) - 1e-9) <= c_star <= (float(c_hi) + 1e-9):
            return c_star, "logistic"
    if iso_c is None:
        return float("nan"), "none"
    return float(iso_c), "isotonic"


def bootstrap_boundary(
    c_values: list[float],
    point_bits: dict[float, list[int]],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | str]:
    rng = random.Random(seed)
    cstars: list[float] = []
    right_censored = 0
    left_censored = 0
    in_domain = 0
    unresolved = 0
    method_counter: Counter[str] = Counter()

    weights = [len(point_bits[c]) for c in c_values]
    for _ in range(n_boot):
        sampled_bits: dict[float, list[int]] = {}
        probs: list[float] = []
        for c in c_values:
            bits = point_bits[c]
            n = len(bits)
            sample = [bits[rng.randrange(n)] for __ in range(n)]
            sampled_bits[c] = sample
            probs.append(sum(sample) / n)
        probs_iso = pava_nonincreasing(probs, weights)
        monotone = classify_monotone_boundary(c_values, probs_iso)
        klass = str(monotone["classification"])
        if klass == "in_domain_estimate":
            c_star, method = choose_boundary_estimate(c_values, sampled_bits, monotone)
            if math.isfinite(c_star):
                cstars.append(c_star)
                in_domain += 1
                method_counter[method] += 1
            else:
                unresolved += 1
        elif klass.startswith("right_censored"):
            right_censored += 1
        elif klass.startswith("left_censored"):
            left_censored += 1
        else:
            unresolved += 1

    cstars_sorted = sorted(cstars)
    return {
        "bootstrap_in_domain_fraction": in_domain / n_boot,
        "bootstrap_right_censored_fraction": right_censored / n_boot,
        "bootstrap_left_censored_fraction": left_censored / n_boot,
        "bootstrap_unresolved_fraction": unresolved / n_boot,
        "bootstrap95_c_star_lo": percentile(cstars_sorted, 0.025),
        "bootstrap95_c_star_hi": percentile(cstars_sorted, 0.975),
        "bootstrap_in_domain_n": in_domain,
        "bootstrap_method_mode": method_counter.most_common(1)[0][0] if method_counter else "none",
    }


# =========================
# Summarization
# =========================
def expected_run_dir(manifest_row: dict[str, str], seed: int) -> Path:
    tag = build_run_tag(manifest_row["alpha_cli"], manifest_row["c_upd_cli"], seed)
    return TARGET_BASE / tag


def summarize_stagec(manifest_rows: list[dict[str, str]], fullref_meta: dict[str, float]) -> dict[str, object]:
    per_run_fields = [
        "run_dir",
        "alpha",
        "c_upd",
        "candidate_type",
        "stageB_classification",
        "stageB_in_domain_bracket_lo",
        "stageB_in_domain_bracket_hi_or_gateoff",
        "seed",
        "eval_return_end_mean",
        "I_R_full",
        "learned_R_full_0p5",
        "eval_dbar_post",
        "eval_occupied_post_mean",
        "eval_CSR_mean",
        "mean_update_fraction",
        "final_executed_updates_mean",
        "final_energy",
        "final_VarTD",
        "wall_seconds",
    ]
    by_point_fields = [
        "alpha",
        "c_upd",
        "candidate_type",
        "stageB_classification",
        "n_seeds",
        "successes_R_full_0p5",
        "P_learn_R_full_0p5",
        "bootstrap95_P_learn_lo",
        "bootstrap95_P_learn_hi",
        "mean_I_R_full",
        "std_I_R_full",
        "mean_eval_return_end_mean",
        "std_eval_return_end_mean",
        "bootstrap95_mean_return_lo",
        "bootstrap95_mean_return_hi",
        "mean_eval_dbar_post",
        "mean_eval_occupied_post_mean",
        "mean_eval_CSR_mean",
        "mean_update_fraction",
        "std_mean_update_fraction",
        "mean_final_executed_updates",
        "mean_final_energy",
        "mean_final_VarTD",
        "mean_wall_seconds",
    ]
    alpha_boundary_fields = [
        "alpha",
        "candidate_type",
        "stageB_classification",
        "full_horizon_classification",
        "fit_method",
        "bracket_lo",
        "bracket_hi",
        "c_star_hat",
        "bootstrap95_c_star_lo",
        "bootstrap95_c_star_hi",
        "bootstrap_in_domain_fraction",
        "bootstrap_right_censored_fraction",
        "bootstrap_left_censored_fraction",
        "bootstrap_unresolved_fraction",
        "bootstrap_method_mode",
        "c_values",
        "observed_P_values",
        "observed_iso_P_values",
        "observed_mean_returns",
        "observed_mean_I_R_full",
        "note",
    ]

    per_run_rows: list[dict[str, object]] = []
    missing: list[str] = []

    for manifest_row in manifest_rows:
        for seed in SEEDS:
            run_dir = expected_run_dir(manifest_row, seed)
            summary_path = run_dir / "gate5_summary_AUTOGEN.txt"
            if not summary_path.is_file():
                missing.append(str(run_dir))
                continue
            d = parse_summary(summary_path)
            eval_return = safe_float(d, "eval_return_end_mean")
            per_run_rows.append(
                {
                    "run_dir": str(run_dir),
                    "alpha": manifest_row["alpha"],
                    "c_upd": manifest_row["c_upd"],
                    "candidate_type": manifest_row["candidate_type"],
                    "stageB_classification": manifest_row["stageB_classification"],
                    "stageB_in_domain_bracket_lo": manifest_row["stageB_in_domain_bracket_lo"],
                    "stageB_in_domain_bracket_hi_or_gateoff": manifest_row["stageB_in_domain_bracket_hi_or_gateoff"],
                    "seed": str(seed),
                    "eval_return_end_mean": eval_return,
                    "I_R_full": (eval_return - R0) / (R_REF_FULL - R0),
                    "learned_R_full_0p5": int(eval_return >= R_THRESH_FULL),
                    "eval_dbar_post": safe_float(d, "eval_dbar_post", "eval_dbar_end"),
                    "eval_occupied_post_mean": safe_float(d, "eval_occupied_post_mean", "eval_occupied_end_mean"),
                    "eval_CSR_mean": safe_float(d, "eval_CSR_mean"),
                    "mean_update_fraction": safe_float(d, "mean_update_fraction"),
                    "final_executed_updates_mean": safe_float(d, "final_executed_updates_mean"),
                    "final_energy": safe_float(d, "final_energy"),
                    "final_VarTD": safe_float(d, "final_VarTD"),
                    "wall_seconds": read_wall_seconds(run_dir),
                }
            )

    write_csv(PER_RUN_CSV, per_run_rows, per_run_fields)

    by_point: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    point_manifest: dict[tuple[str, str], dict[str, str]] = {}
    for row in manifest_rows:
        point_manifest[(row["alpha"], row["c_upd"])] = row
    for row in per_run_rows:
        by_point[(str(row["alpha"]), str(row["c_upd"]))].append(row)

    agg_rows: list[dict[str, object]] = []
    point_stats: dict[tuple[str, str], dict[str, object]] = {}
    point_bits: dict[tuple[str, str], list[int]] = {}

    for key in sorted(point_manifest.keys(), key=lambda x: (float(x[0]), float(x[1]))):
        alpha, c_upd = key
        rs = by_point.get(key, [])
        manifest_row = point_manifest[key]
        learned = [int(r["learned_R_full_0p5"]) for r in rs]
        i_r = [float(r["I_R_full"]) for r in rs]
        rets = [float(r["eval_return_end_mean"]) for r in rs]
        dbars = [float(r["eval_dbar_post"]) for r in rs]
        occs = [float(r["eval_occupied_post_mean"]) for r in rs]
        csrs = [float(r["eval_CSR_mean"]) for r in rs]
        upds = [float(r["mean_update_fraction"]) for r in rs]
        exec_upd = [float(r["final_executed_updates_mean"]) for r in rs]
        energies = [float(r["final_energy"]) for r in rs]
        vartd = [float(r["final_VarTD"]) for r in rs]
        walls = [float(r["wall_seconds"]) for r in rs]

        p_learn = mean(learned) if learned else float("nan")
        p_lo, p_hi = bootstrap_prob_ci(learned, seed=int(round(float(alpha) * 1000 + float(c_upd) * 1000))) if learned else (float("nan"), float("nan"))
        ret_lo, ret_hi = bootstrap_mean_ci(rets, seed=int(round(float(alpha) * 2000 + float(c_upd) * 2000))) if rets else (float("nan"), float("nan"))

        agg = {
            "alpha": alpha,
            "c_upd": c_upd,
            "candidate_type": manifest_row["candidate_type"],
            "stageB_classification": manifest_row["stageB_classification"],
            "n_seeds": len(rs),
            "successes_R_full_0p5": sum(learned),
            "P_learn_R_full_0p5": p_learn,
            "bootstrap95_P_learn_lo": p_lo,
            "bootstrap95_P_learn_hi": p_hi,
            "mean_I_R_full": mean(i_r) if i_r else float("nan"),
            "std_I_R_full": stdev(i_r) if len(i_r) >= 2 else float("nan"),
            "mean_eval_return_end_mean": mean(rets) if rets else float("nan"),
            "std_eval_return_end_mean": stdev(rets) if len(rets) >= 2 else float("nan"),
            "bootstrap95_mean_return_lo": ret_lo,
            "bootstrap95_mean_return_hi": ret_hi,
            "mean_eval_dbar_post": nanmean(dbars),
            "mean_eval_occupied_post_mean": nanmean(occs),
            "mean_eval_CSR_mean": nanmean(csrs),
            "mean_update_fraction": nanmean(upds),
            "std_mean_update_fraction": nanstdev(upds),
            "mean_final_executed_updates": nanmean(exec_upd),
            "mean_final_energy": nanmean(energies),
            "mean_final_VarTD": nanmean(vartd),
            "mean_wall_seconds": nanmean(walls),
        }
        agg_rows.append(agg)
        point_stats[key] = agg
        point_bits[key] = learned

    write_csv(BY_POINT_CSV, agg_rows, by_point_fields)

    alpha_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        alpha_groups[row["alpha"]].append(row)

    boundary_rows: list[dict[str, object]] = []
    for alpha in sorted(alpha_groups.keys(), key=float):
        group = sorted(alpha_groups[alpha], key=lambda r: float(r["c_upd"]))
        c_values = [float(r["c_upd"]) for r in group]
        probs = [
            float(point_stats[(alpha, canonical_csv_float(c))]["P_learn_R_full_0p5"])
            if (alpha, canonical_csv_float(c)) in point_stats
            else float("nan")
            for c in c_values
        ]
        mean_returns = [
            float(point_stats[(alpha, canonical_csv_float(c))]["mean_eval_return_end_mean"])
            if (alpha, canonical_csv_float(c)) in point_stats
            else float("nan")
            for c in c_values
        ]
        mean_i_r = [
            float(point_stats[(alpha, canonical_csv_float(c))]["mean_I_R_full"])
            if (alpha, canonical_csv_float(c)) in point_stats
            else float("nan")
            for c in c_values
        ]
        bits_by_c = {
            float(r["c_upd"]): point_bits.get((alpha, r["c_upd"]), [])
            for r in group
        }
        weights = [len(bits_by_c[c]) for c in c_values]
        if all(w > 0 for w in weights):
            probs_iso = pava_nonincreasing(probs, weights)
            monotone = classify_monotone_boundary(c_values, probs_iso)
            c_star_hat, fit_method = choose_boundary_estimate(c_values, bits_by_c, monotone)
            boot = bootstrap_boundary(
                c_values=c_values,
                point_bits=bits_by_c,
                seed=BOOTSTRAP_SEED + int(round(float(alpha) * 1000.0)),
            )
        else:
            probs_iso = [float("nan")] * len(c_values)
            monotone = {"classification": "incomplete", "bracket_lo": None, "bracket_hi": None, "c_star_isotonic": None}
            c_star_hat, fit_method = float("nan"), "none"
            boot = {
                "bootstrap_in_domain_fraction": float("nan"),
                "bootstrap_right_censored_fraction": float("nan"),
                "bootstrap_left_censored_fraction": float("nan"),
                "bootstrap_unresolved_fraction": float("nan"),
                "bootstrap95_c_star_lo": float("nan"),
                "bootstrap95_c_star_hi": float("nan"),
                "bootstrap_in_domain_n": 0,
                "bootstrap_method_mode": "none",
            }

        note = ""
        candidate_type = group[0]["candidate_type"]
        stageb_classification = group[0]["stageB_classification"]
        full_class = str(monotone["classification"])
        if full_class == "in_domain_estimate" and stageb_classification == "in_domain_bracket_ready":
            note = "Full horizon preserves an in-domain boundary for alpha=0.0."
        elif full_class == "in_domain_estimate" and stageb_classification == "edge_censored_gt_1p0":
            note = "Full horizon pulls the short-horizon edge-censored case into the physical domain."
        elif full_class.startswith("right_censored") and stageb_classification == "edge_censored_gt_1p0":
            note = "Full horizon preserves right-censoring above the physical c_upd domain."
        elif full_class.startswith("right_censored"):
            note = "All refined full-horizon points remain above the learning threshold; the boundary lies above the sampled refine band."
        elif full_class.startswith("left_censored"):
            note = "All refined full-horizon points fall below the learning threshold; the boundary lies below the sampled refine band."
        elif full_class == "incomplete":
            note = "Stage C run set is incomplete for this alpha."
        else:
            note = "Boundary evidence could not be classified cleanly from the completed Stage C subset."

        boundary_rows.append(
            {
                "alpha": alpha,
                "candidate_type": candidate_type,
                "stageB_classification": stageb_classification,
                "full_horizon_classification": full_class,
                "fit_method": fit_method,
                "bracket_lo": "" if monotone["bracket_lo"] is None else canonical_csv_float(float(monotone["bracket_lo"])),
                "bracket_hi": "" if monotone["bracket_hi"] is None else canonical_csv_float(float(monotone["bracket_hi"])),
                "c_star_hat": "" if not math.isfinite(c_star_hat) else f"{float(c_star_hat):.6f}",
                "bootstrap95_c_star_lo": "" if math.isnan(float(boot["bootstrap95_c_star_lo"])) else f"{float(boot['bootstrap95_c_star_lo']):.6f}",
                "bootstrap95_c_star_hi": "" if math.isnan(float(boot["bootstrap95_c_star_hi"])) else f"{float(boot['bootstrap95_c_star_hi']):.6f}",
                "bootstrap_in_domain_fraction": "" if math.isnan(float(boot["bootstrap_in_domain_fraction"])) else f"{float(boot['bootstrap_in_domain_fraction']):.6f}",
                "bootstrap_right_censored_fraction": "" if math.isnan(float(boot["bootstrap_right_censored_fraction"])) else f"{float(boot['bootstrap_right_censored_fraction']):.6f}",
                "bootstrap_left_censored_fraction": "" if math.isnan(float(boot["bootstrap_left_censored_fraction"])) else f"{float(boot['bootstrap_left_censored_fraction']):.6f}",
                "bootstrap_unresolved_fraction": "" if math.isnan(float(boot["bootstrap_unresolved_fraction"])) else f"{float(boot['bootstrap_unresolved_fraction']):.6f}",
                "bootstrap_method_mode": str(boot["bootstrap_method_mode"]),
                "c_values": ",".join(canonical_csv_float(c) for c in c_values),
                "observed_P_values": ",".join("nan" if math.isnan(p) else f"{p:.6f}" for p in probs),
                "observed_iso_P_values": ",".join("nan" if math.isnan(p) else f"{p:.6f}" for p in probs_iso),
                "observed_mean_returns": ",".join("nan" if math.isnan(v) else f"{v:.6f}" for v in mean_returns),
                "observed_mean_I_R_full": ",".join("nan" if math.isnan(v) else f"{v:.6f}" for v in mean_i_r),
                "note": note,
            }
        )

    write_csv(ALPHA_BOUNDARY_CSV, boundary_rows, alpha_boundary_fields)

    write_csv(MANIFEST_CSV, manifest_rows, list(manifest_rows[0].keys()) if manifest_rows else ["alpha", "c_upd"])
    write_text(MISSING_TXT, "\n".join(missing) + ("\n" if missing else ""))

    completed_runs = len(per_run_rows)
    expected_runs = len(manifest_rows) * len(SEEDS)
    boundary_in_domain = sum(1 for row in boundary_rows if row["full_horizon_classification"] == "in_domain_estimate")
    boundary_right_censored = sum(1 for row in boundary_rows if str(row["full_horizon_classification"]).startswith("right_censored"))
    boundary_left_censored = sum(1 for row in boundary_rows if str(row["full_horizon_classification"]).startswith("left_censored"))
    boundary_incomplete = sum(1 for row in boundary_rows if row["full_horizon_classification"] == "incomplete")
    boundary_other = len(boundary_rows) - boundary_in_domain - boundary_right_censored - boundary_left_censored - boundary_incomplete

    lines = [
        "Stage C subset full-horizon refinement report",
        "",
        "Frozen inputs verified",
        f"stage_b_report={STAGE_B_REPORT}",
        f"stage_c_subset_readiness_report={STAGE_C_READY_REPORT}",
        f"fullref_report={FULLREF_REPORT}",
        f"fullref_per_run_csv={FULLREF_PER_RUN_CSV}",
        f"stageb_alpha_csv={STAGEB_ALPHA_CSV}",
        f"stagec_candidate_csv={STAGEC_CANDIDATE_CSV}",
        "",
        "Locked constants",
        f"R0={R0:.6f}",
        f"D0={D0:.6f}",
        f"R_REF_FULL={R_REF_FULL:.6f}",
        f"R_THRESH_FULL={R_THRESH_FULL:.6f}",
        f"EPS_OCC={EPS_OCC:.6f}",
        f"M_EVAL={M_EVAL}",
        f"FULL_TIMESTEPS={FULL_TIMESTEPS}",
        f"NUM_ENVS={NUM_ENVS}",
        f"NUM_STEPS={NUM_STEPS}",
        f"SEEDS={SEEDS[0]}..{SEEDS[-1]} ({len(SEEDS)} total)",
        "",
        "Locked full-reference verification",
        f"fullref_completed_runs={int(fullref_meta['n_runs'])}",
        f"fullref_mean_eval_return_end_mean={fullref_meta['mean_return']:.6f}",
        f"fullref_sd_eval_return_end_mean={fullref_meta['sd_return']:.6f}",
        f"fullref_bootstrap95_mean_return=[{fullref_meta['bootstrap_lo']:.6f}, {fullref_meta['bootstrap_hi']:.6f}]",
        f"fullref_fraction_below_R0={fullref_meta['frac_below_random']:.6f}",
        "",
        "Stage C run completion",
        f"completed_runs={completed_runs}",
        f"expected_runs={expected_runs}",
        f"missing_runs={len(missing)}",
        f"stage_c_subset_complete={'yes' if completed_runs == expected_runs else 'no'}",
        f"selected_full_horizon_points={len(manifest_rows)}",
        f"selected_alphas={len({row['alpha'] for row in manifest_rows})}",
        "",
        "Boundary classification summary",
        f"in_domain_count={boundary_in_domain}",
        f"right_censored_count={boundary_right_censored}",
        f"left_censored_count={boundary_left_censored}",
        f"incomplete_count={boundary_incomplete}",
        f"other_count={boundary_other}",
        "",
        "Per-alpha boundary evidence",
    ]

    for row in boundary_rows:
        lines.append(
            f"alpha={row['alpha']} candidate_type={row['candidate_type']} "
            f"stageB_classification={row['stageB_classification']} "
            f"full_horizon_classification={row['full_horizon_classification']} "
            f"fit_method={row['fit_method']} "
            f"bracket_lo={row['bracket_lo'] or 'none'} "
            f"bracket_hi={row['bracket_hi'] or 'none'} "
            f"c_star_hat={row['c_star_hat'] or 'none'} "
            f"bootstrap95_c_star={row['bootstrap95_c_star_lo'] or 'none'}:{row['bootstrap95_c_star_hi'] or 'none'} "
            f"P={row['observed_P_values']} "
            f"P_iso={row['observed_iso_P_values']} "
            f"mean_returns={row['observed_mean_returns']} "
            f"mean_I_R_full={row['observed_mean_I_R_full']} "
            f"note={row['note']}"
        )

    lines.extend(
        [
            "",
            "Output files",
            str(MANIFEST_CSV),
            str(PER_RUN_CSV),
            str(BY_POINT_CSV),
            str(ALPHA_BOUNDARY_CSV),
            str(MISSING_TXT),
            str(REPORT_TXT),
            "",
            "Next files for dissertation use",
            str(BY_POINT_CSV),
            str(ALPHA_BOUNDARY_CSV),
            str(REPORT_TXT),
        ]
    )

    if missing:
        lines.extend(["", "First 100 missing Stage C runs", *missing[:100]])

    write_text(REPORT_TXT, "\n".join(lines) + "\n")

    return {
        "completed_runs": completed_runs,
        "expected_runs": expected_runs,
        "missing_runs": len(missing),
        "boundary_rows": boundary_rows,
    }


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage C subset full-horizon refinement runner + summarizer")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not launch training runs; only summarize whatever Stage C outputs already exist.",
    )
    args = parser.parse_args()

    need_file(STAGE_B_REPORT)
    need_file(STAGE_C_READY_REPORT)
    need_file(FULLREF_REPORT)
    need_file(STAGEB_ALPHA_CSV)
    need_file(STAGEC_CANDIDATE_CSV)
    fullref_meta = verify_fullref_lock()
    manifest_rows = load_stagec_manifest()

    ensure_dir(TARGET_BASE)
    ensure_dir(REPORT_BASE)

    print(f"[stageC-subset] repo_root={Path.cwd()}")
    print(f"[stageC-subset] python={sys.executable}")
    print(f"[stageC-subset] target_dir={TARGET_BASE}")
    print(f"[stageC-subset] report_dir={REPORT_BASE}")
    print(f"[stageC-subset] summary_only={args.summary_only}")
    print(f"[stageC-subset] locked R_THRESH_FULL={R_THRESH_FULL:.6f}")
    print(f"[stageC-subset] locked fullref mean verified at {fullref_meta['mean_return']:.6f}")
    print(f"[stageC-subset] selected points={len(manifest_rows)}")
    print(f"[stageC-subset] seed slots={len(manifest_rows) * len(SEEDS)}")

    if not args.summary_only:
        need_file(SCRIPT_PATH)
        need_file(CONFIG_PATH)
        print("[stageC-subset] === launching missing full-horizon refinement runs ===")
        for manifest_row in manifest_rows:
            for seed in SEEDS:
                run_point_seed(manifest_row, seed)

    print("[stageC-subset] === summarizing ===")
    summary = summarize_stagec(manifest_rows, fullref_meta)
    print("[stageC-subset] done")
    print(f"[stageC-subset] completed_runs={summary['completed_runs']} / {summary['expected_runs']}")
    print(f"[stageC-subset] missing_runs={summary['missing_runs']}")
    print(f"[stageC-subset] read: {REPORT_TXT}")
    print(f"[stageC-subset] read: {ALPHA_BOUNDARY_CSV}")
    print(f"[stageC-subset] read: {BY_POINT_CSV}")


if __name__ == "__main__":
    main()
