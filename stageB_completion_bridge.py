#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

# =========================
# Locked project constants
# =========================
R0 = -27.256548
R_REF_SHORT = -22.315781
R_THRESH_SHORT = R0 + 0.5 * (R_REF_SHORT - R0)

SHORT_SEEDS = list(range(30, 62))   # 32 seeds total
FULLREF_SEEDS = list(range(30, 62)) # 32 seeds total

SCRIPT_PATH = Path("baselines/IPPO/ippo_ff_mpe_gate5_energy_gated.py")
CONFIG_PATH = Path("baselines/IPPO/config/ippo_ff_mpe.yaml")

SHORT_TARGET_BASE = Path("runs/stageB_completion_bridge_32seed")
FULLREF_TARGET_BASE = Path("runs/ref_full_ungated_32seed")
REPORT_BASE = Path("runs/stageB_completion_bridge_reports")

SHORT_SOURCE_DIRS = [
    Path("runs/stageB_short_coarse"),
    Path("runs/stageB_short_cupd_extension"),
    Path("runs/stageB_short_dense_11x11_seed8"),
    Path("runs/stageB_completion_bridge_32seed"),
]
FULLREF_SOURCE_DIRS = [
    Path("runs/ref_full_ungated_seed30"),
    Path("runs/ref_full_ungated_seed31"),
    Path("runs/ref_full_ungated_seed32"),
    Path("runs/ref_full_ungated_seed33"),
    Path("runs/ref_full_ungated_24seed"),
    Path("runs/ref_full_ungated_32seed"),
]

# Each tuple is: (alpha, c_upd, roles, notes)
# c_upd=1.05 is a diagnostic gate-off control, NOT a physical map point.
MANIFEST = [
    ("0.0", "0.1",  "enabled_anchor",            "highest 8-seed P=1 point"),
    ("0.0", "0.5",  "boundary_proxy",            "closest current point above the 0.5 crossing"),
    ("0.0", "0.6",  "below_proxy",               "first current in-domain point below 0.5"),
    ("0.0", "1.0",  "edge_max",                  "top of physical c_upd domain"),
    ("0.0", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),

    ("0.1", "0.1",  "enabled_anchor",            "highest 8-seed P=1 point"),
    ("0.1", "0.9",  "boundary_proxy",            "closest current point to 0.5"),
    ("0.1", "1.0",  "edge_max",                  "top of physical c_upd domain"),
    ("0.1", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),

    ("0.2", "0.8",  "enabled_anchor",            "highest 8-seed P=1 point"),
    ("0.2", "0.9",  "boundary_proxy",            "closest current point to 0.5"),
    ("0.2", "1.0",  "edge_max",                  "top of physical c_upd domain"),
    ("0.2", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),

    ("0.3", "0.9",  "enabled_anchor",            "highest 8-seed P=1 point"),
    ("0.3", "0.6",  "boundary_proxy",            "closest current point to 0.5"),
    ("0.3", "1.0",  "edge_max",                  "top of physical c_upd domain"),
    ("0.3", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),

    ("0.4", "0.9",  "boundary_proxy",            "closest current point to 0.5"),
    ("0.4", "1.0",  "enabled_anchor;edge_max",   "highest 8-seed P=1 point and top of physical domain"),
    ("0.4", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),

    ("0.5", "0.4",  "boundary_proxy",            "closest current point to 0.5"),
    ("0.5", "0.7",  "enabled_anchor",            "highest 8-seed P=1 point"),
    ("0.5", "1.0",  "edge_max",                  "top of physical c_upd domain"),
    ("0.5", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),

    ("0.6", "0.8",  "boundary_proxy",            "closest current point to 0.5"),
    ("0.6", "1.0",  "enabled_anchor;edge_max",   "highest 8-seed P=1 point and top of physical domain"),
    ("0.6", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),

    ("0.7", "0.9",  "boundary_proxy",            "closest current point to 0.5"),
    ("0.7", "1.0",  "enabled_anchor;edge_max",   "highest 8-seed P=1 point and top of physical domain"),
    ("0.7", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),

    ("0.8", "0.8",  "boundary_proxy",            "closest current point to 0.5"),
    ("0.8", "1.0",  "enabled_anchor;edge_max",   "highest 8-seed P=1 point and top of physical domain"),
    ("0.8", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),

    ("0.9", "0.8",  "boundary_proxy",            "closest current point to 0.5"),
    ("0.9", "1.0",  "enabled_anchor;edge_max",   "highest 8-seed P=1 point and top of physical domain"),
    ("0.9", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),

    ("1.0", "0.9",  "enabled_anchor",            "highest 8-seed P=1 point"),
    ("1.0", "1.0",  "boundary_proxy;edge_max",   "closest current point to 0.5 and top of physical domain"),
    ("1.0", "1.05", "gateoff_control",           "diagnostic gate-off control beyond physical domain"),
]


# =========================
# Generic helpers
# =========================
def need_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def fmt_tag(x: str) -> str:
    return x.replace(".", "p")


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


def realpath(path: Path) -> str:
    return str(path.resolve())


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


def bootstrap_mean_ci(values: list[float], n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
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


def bootstrap_prob_ci(bits: list[int], n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
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


# =========================
# Reuse helpers
# =========================
def link_or_copy(src_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for name in ["gate5_summary_AUTOGEN.txt", "time.txt", "stdout.log"]:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(realpath(src), dst)
        except OSError:
            shutil.copy2(src, dst)
    write_text(dst_dir / "reused_from.txt", f"{src_dir}\n")


def find_existing_short_run(alpha: str, c_upd: str, seed: int) -> Path | None:
    tag = f"a{fmt_tag(alpha)}_c{fmt_tag(c_upd)}_seed{seed}"
    for src in SHORT_SOURCE_DIRS:
        candidate = src / tag
        if (candidate / "gate5_summary_AUTOGEN.txt").is_file():
            return candidate
    return None


def find_existing_fullref_run(seed: int) -> Path | None:
    # Check modern /seedNN layout first.
    for src in FULLREF_SOURCE_DIRS:
        candidate = src / f"seed{seed}"
        if (candidate / "gate5_summary_AUTOGEN.txt").is_file():
            return candidate
    # Check old flat runs/ref_full_ungated_seedNN layout.
    candidate = Path(f"runs/ref_full_ungated_seed{seed}")
    if (candidate / "gate5_summary_AUTOGEN.txt").is_file():
        return candidate
    return None


# =========================
# Launch helpers
# =========================
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


def run_short_point_seed(alpha: str, c_upd: str, seed: int) -> None:
    tag = f"a{fmt_tag(alpha)}_c{fmt_tag(c_upd)}_seed{seed}"
    run_dir = SHORT_TARGET_BASE / tag
    summary_path = run_dir / "gate5_summary_AUTOGEN.txt"
    if summary_path.is_file():
        print(f"[short skip] {tag} already present")
        return

    existing = find_existing_short_run(alpha, c_upd, seed)
    if existing is not None:
        print(f"[short reuse] {tag} <= {existing}")
        link_or_copy(existing, run_dir)
        return

    print(f"[short run] {tag}")
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_PATH),
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=True",
        "WANDB_MODE=disabled",
        "NUM_SEEDS=1",
        f"SEED={seed}",
        "NUM_ENVS=25",
        "TOTAL_TIMESTEPS=400000",
        "NUM_STEPS=128",
        "ENV_NAME=MPE_simple_spread_v3",
        "++ENV_KWARGS.num_agents=3",
        "++ENV_KWARGS.num_landmarks=3",
        "++ENV_KWARGS.local_ratio=0.5",
        "++ENV_KWARGS.max_steps=25",
        "++ENV_KWARGS.action_type=Discrete",
        "++EPS_OCC=0.2",
        "++D0=0.733348",
        "++R0=-27.256548",
        "++BETA=2.302585092994046",
        "++Z_MAX=10.0",
        "++EPS_NORM=1e-8",
        "++USE_GEOM_FUNDING=True",
        "++DO_EVAL=True",
        "++M_EVAL=100",
        f"++C_UPD={c_upd}",
        f"++ALPHA={alpha}",
        "++E0=1.0",
        "++E_MAX=1.0",
    ]
    run_command(cmd, run_dir / "stdout.log", run_dir / "time.txt")


def run_fullref_seed(seed: int) -> None:
    run_dir = FULLREF_TARGET_BASE / f"seed{seed}"
    summary_path = run_dir / "gate5_summary_AUTOGEN.txt"
    if summary_path.is_file():
        print(f"[fullref skip] seed{seed} already present")
        return

    existing = find_existing_fullref_run(seed)
    if existing is not None:
        print(f"[fullref reuse] seed{seed} <= {existing}")
        link_or_copy(existing, run_dir)
        return

    print(f"[fullref run] seed{seed}")
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_PATH),
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=True",
        "WANDB_MODE=disabled",
        "NUM_SEEDS=1",
        f"SEED={seed}",
        "NUM_ENVS=25",
        "TOTAL_TIMESTEPS=1000000",
        "NUM_STEPS=128",
        "ENV_NAME=MPE_simple_spread_v3",
        "++ENV_KWARGS.num_agents=3",
        "++ENV_KWARGS.num_landmarks=3",
        "++ENV_KWARGS.local_ratio=0.5",
        "++ENV_KWARGS.max_steps=25",
        "++ENV_KWARGS.action_type=Discrete",
        "++EPS_OCC=0.2",
        "++D0=0.733348",
        "++R0=-27.256548",
        "++BETA=2.302585092994046",
        "++Z_MAX=10.0",
        "++EPS_NORM=1e-8",
        "++USE_GEOM_FUNDING=True",
        "++DO_EVAL=True",
        "++M_EVAL=100",
        "++C_UPD=0.0",
        "++ALPHA=0.0",
        "++E0=1.0",
        "++E_MAX=1.0",
    ]
    run_command(cmd, run_dir / "stdout.log", run_dir / "time.txt")


# =========================
# Summarization
# =========================
def summarize_short_bridge() -> dict:
    point_meta = {(a, c): {"roles": roles, "notes": notes} for a, c, roles, notes in MANIFEST}
    per_run_rows: list[dict] = []
    missing: list[str] = []

    for alpha, c_upd, roles, notes in MANIFEST:
        tag_alpha = fmt_tag(alpha)
        tag_c = fmt_tag(c_upd)
        for seed in SHORT_SEEDS:
            tag = f"a{tag_alpha}_c{tag_c}_seed{seed}"
            run_dir = SHORT_TARGET_BASE / tag
            summary_path = run_dir / "gate5_summary_AUTOGEN.txt"
            if not summary_path.is_file():
                missing.append(tag)
                continue
            d = parse_summary(summary_path)
            ret = safe_float(d, "eval_return_end_mean")
            per_run_rows.append(
                {
                    "run_dir": str(run_dir),
                    "alpha": alpha,
                    "c_upd": c_upd,
                    "roles": roles,
                    "notes": notes,
                    "seed": str(seed),
                    "eval_return_end_mean": ret,
                    "I_R_short": (ret - R0) / (R_REF_SHORT - R0),
                    "learned_0p5": int(ret >= R_THRESH_SHORT),
                    "eval_dbar_post": safe_float(d, "eval_dbar_post"),
                    "eval_occupied_post_mean": safe_float(d, "eval_occupied_post_mean"),
                    "eval_CSR_mean": safe_float(d, "eval_CSR_mean"),
                    "mean_update_fraction": safe_float(d, "mean_update_fraction"),
                    "final_VarTD": safe_float(d, "final_VarTD"),
                    "wall_seconds": read_wall_seconds(run_dir),
                }
            )

    by_point: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in per_run_rows:
        by_point[(r["alpha"], r["c_upd"])].append(r)

    agg_rows: list[dict] = []
    point_stats: dict[tuple[str, str], dict] = {}
    for (alpha, c_upd), rs in sorted(by_point.items(), key=lambda x: (float(x[0][0]), float(x[0][1]))):
        learned = [int(r["learned_0p5"]) for r in rs]
        i_r = [float(r["I_R_short"]) for r in rs]
        rets = [float(r["eval_return_end_mean"]) for r in rs]
        dbars = [float(r["eval_dbar_post"]) for r in rs]
        occs = [float(r["eval_occupied_post_mean"]) for r in rs]
        csrs = [float(r["eval_CSR_mean"]) for r in rs]
        upds = [float(r["mean_update_fraction"]) for r in rs]
        vartd = [float(r["final_VarTD"]) for r in rs]
        walls = [float(r["wall_seconds"]) for r in rs if not math.isnan(float(r["wall_seconds"]))]
        meta = point_meta[(alpha, c_upd)]
        p_learn = mean(learned)
        p_lo, p_hi = bootstrap_prob_ci(learned, seed=int(round(float(alpha) * 1000 + float(c_upd) * 1000)))
        ret_lo, ret_hi = bootstrap_mean_ci(rets, seed=int(round(float(alpha) * 2000 + float(c_upd) * 2000)))
        agg = {
            "alpha": alpha,
            "c_upd": c_upd,
            "roles": meta["roles"],
            "notes": meta["notes"],
            "n_seeds": len(rs),
            "successes_R_short_0p5": sum(learned),
            "P_learn_R_short_0p5": p_learn,
            "bootstrap95_P_learn_lo": p_lo,
            "bootstrap95_P_learn_hi": p_hi,
            "mean_I_R_short": mean(i_r),
            "std_I_R_short": stdev(i_r) if len(i_r) >= 2 else 0.0,
            "mean_eval_return_end_mean": mean(rets),
            "std_eval_return_end_mean": stdev(rets) if len(rets) >= 2 else 0.0,
            "bootstrap95_mean_return_lo": ret_lo,
            "bootstrap95_mean_return_hi": ret_hi,
            "mean_eval_dbar_post": mean(dbars),
            "mean_eval_occupied_post_mean": mean(occs),
            "mean_eval_CSR_mean": mean(csrs),
            "mean_update_fraction": mean(upds),
            "std_mean_update_fraction": stdev(upds) if len(upds) >= 2 else 0.0,
            "mean_final_VarTD": mean(vartd),
            "mean_wall_seconds": mean(walls) if walls else float("nan"),
        }
        agg_rows.append(agg)
        point_stats[(alpha, c_upd)] = agg

    alpha_rows: list[dict] = []
    stage_c_candidates: list[dict] = []
    unresolved_alphas: list[str] = []
    in_domain_count = 0
    edge_censored_count = 0
    no_boundary_gateoff_count = 0

    unique_alphas = sorted({alpha for alpha, _, _, _ in MANIFEST}, key=float)
    for alpha in unique_alphas:
        points = {float(c): stats for (a, c), stats in point_stats.items() if a == alpha}
        in_domain_points = [(c, s) for c, s in points.items() if c <= 1.0 + 1e-12]
        in_domain_points.sort(key=lambda x: x[0])

        above = [(c, s) for c, s in in_domain_points if float(s["P_learn_R_short_0p5"]) >= 0.5]
        below = [(c, s) for c, s in in_domain_points if float(s["P_learn_R_short_0p5"]) < 0.5]
        highest_above = max(above, key=lambda x: x[0]) if above else None
        lowest_below_after = None
        if highest_above is not None:
            later = [(c, s) for c, s in below if c > highest_above[0]]
            if later:
                lowest_below_after = min(later, key=lambda x: x[0])

        edge_stats = points.get(1.0)
        gateoff_stats = points.get(1.05)
        enabled_anchor = None
        boundary_proxy = None
        below_proxy = None
        for c, s in points.items():
            roles = set(str(s["roles"]).split(";"))
            if "enabled_anchor" in roles:
                enabled_anchor = (c, s)
            if "boundary_proxy" in roles:
                boundary_proxy = (c, s)
            if "below_proxy" in roles:
                below_proxy = (c, s)

        classification = "unresolved"
        bracket_lo = float("nan")
        bracket_hi = float("nan")
        proposed_refine = ""
        note = ""

        if highest_above is not None and lowest_below_after is not None:
            classification = "in_domain_bracket_ready"
            in_domain_count += 1
            bracket_lo = highest_above[0]
            bracket_hi = lowest_below_after[0]
            width = bracket_hi - bracket_lo
            n_interior = 5 if width >= 0.2 else 4
            step = width / (n_interior + 1)
            proposed = [round(bracket_lo + step * i, 3) for i in range(1, n_interior + 1)]
            proposed_refine = ",".join(f"{x:.3f}" for x in proposed)
            note = "32-seed bridge now contains an in-domain short-horizon bracket."
            stage_c_candidates.append(
                {
                    "alpha": alpha,
                    "candidate_type": "in_domain",
                    "c_lo": f"{bracket_lo:.3f}",
                    "c_hi": f"{bracket_hi:.3f}",
                    "proposed_full_horizon_refine_c_values": proposed_refine,
                }
            )
        elif edge_stats is not None and gateoff_stats is not None and float(edge_stats["P_learn_R_short_0p5"]) >= 0.5 and float(gateoff_stats["P_learn_R_short_0p5"]) < 0.5:
            classification = "edge_censored_gt_1p0"
            edge_censored_count += 1
            bracket_lo = 1.0
            bracket_hi = 1.05
            proposed_refine = "0.900,0.950,1.000"
            note = "In-domain edge stays above threshold, but the gate-off control falls below threshold. Treat as right-censored above c_upd=1.0 in the physical map."
            stage_c_candidates.append(
                {
                    "alpha": alpha,
                    "candidate_type": "edge_censored",
                    "c_lo": "1.000",
                    "c_hi": "1.050",
                    "proposed_full_horizon_refine_c_values": proposed_refine,
                }
            )
        elif gateoff_stats is not None and float(gateoff_stats["P_learn_R_short_0p5"]) >= 0.5:
            classification = "no_boundary_even_gateoff"
            no_boundary_gateoff_count += 1
            note = "Even the gate-off control stays at or above the short-horizon learning threshold; no short-horizon boundary is visible under the locked setup."
        else:
            unresolved_alphas.append(alpha)
            note = "Selected 32-seed bridge points do not yet give a clean classification."
            if gateoff_stats is None:
                note = "Missing gate-off control statistics."

        alpha_rows.append(
            {
                "alpha": alpha,
                "classification": classification,
                "enabled_anchor_c": "" if enabled_anchor is None else f"{enabled_anchor[0]:.3f}",
                "enabled_anchor_P": "" if enabled_anchor is None else f"{float(enabled_anchor[1]['P_learn_R_short_0p5']):.6f}",
                "boundary_proxy_c": "" if boundary_proxy is None else f"{boundary_proxy[0]:.3f}",
                "boundary_proxy_P": "" if boundary_proxy is None else f"{float(boundary_proxy[1]['P_learn_R_short_0p5']):.6f}",
                "below_proxy_c": "" if below_proxy is None else f"{below_proxy[0]:.3f}",
                "below_proxy_P": "" if below_proxy is None else f"{float(below_proxy[1]['P_learn_R_short_0p5']):.6f}",
                "edge_max_P": "" if edge_stats is None else f"{float(edge_stats['P_learn_R_short_0p5']):.6f}",
                "gateoff_P": "" if gateoff_stats is None else f"{float(gateoff_stats['P_learn_R_short_0p5']):.6f}",
                "gateoff_mean_update_fraction": "" if gateoff_stats is None else f"{float(gateoff_stats['mean_update_fraction']):.6f}",
                "in_domain_bracket_lo": "" if math.isnan(bracket_lo) else f"{bracket_lo:.3f}",
                "in_domain_bracket_hi_or_gateoff": "" if math.isnan(bracket_hi) else f"{bracket_hi:.3f}",
                "proposed_refine_c_values": proposed_refine,
                "note": note,
            }
        )

    stage_b_complete = (len(missing) == 0) and (len(unresolved_alphas) == 0)

    write_csv(REPORT_BASE / "stageB_bridge_per_run.csv", per_run_rows, list(per_run_rows[0].keys()) if per_run_rows else ["run_dir"])
    write_csv(REPORT_BASE / "stageB_bridge_by_point.csv", agg_rows, list(agg_rows[0].keys()) if agg_rows else ["alpha", "c_upd"])
    write_csv(REPORT_BASE / "stageB_alpha_classification.csv", alpha_rows, list(alpha_rows[0].keys()) if alpha_rows else ["alpha"])
    if stage_c_candidates:
        write_csv(REPORT_BASE / "stageC_candidate_alphas_from_bridge.csv", stage_c_candidates, list(stage_c_candidates[0].keys()))
    else:
        write_csv(
            REPORT_BASE / "stageC_candidate_alphas_from_bridge.csv",
            [],
            ["alpha", "candidate_type", "c_lo", "c_hi", "proposed_full_horizon_refine_c_values"],
        )

    lines = [
        "Stage B completion bridge report",
        "",
        "Locked facts",
        f"R0={R0:.6f}",
        f"R_REF_SHORT={R_REF_SHORT:.6f}",
        f"R_THRESH_SHORT={R_THRESH_SHORT:.6f}",
        "gateoff_control_note=c_upd=1.05 is a diagnostic gate-off control point beyond the physical map domain [0,1]; it is not part of the final physical phase diagram.",
        "",
        "Bridge summary",
        f"completed_runs={len(per_run_rows)}",
        f"expected_runs={len(MANIFEST) * len(SHORT_SEEDS)}",
        f"missing_runs={len(missing)}",
        f"classified_in_domain_boundary_count={in_domain_count}",
        f"classified_edge_censored_count={edge_censored_count}",
        f"classified_no_boundary_even_gateoff_count={no_boundary_gateoff_count}",
        f"unresolved_alpha_count={len(unresolved_alphas)}",
        f"stage_b_complete={'yes' if stage_b_complete else 'no'}",
        "",
        "Per-alpha classification",
    ]
    for row in alpha_rows:
        lines.append(
            f"alpha={row['alpha']} classification={row['classification']} "
            f"enabled_anchor_c={row['enabled_anchor_c'] or 'none'} enabled_anchor_P={row['enabled_anchor_P'] or 'none'} "
            f"boundary_proxy_c={row['boundary_proxy_c'] or 'none'} boundary_proxy_P={row['boundary_proxy_P'] or 'none'} "
            f"below_proxy_c={row['below_proxy_c'] or 'none'} below_proxy_P={row['below_proxy_P'] or 'none'} "
            f"edge_max_P={row['edge_max_P'] or 'none'} gateoff_P={row['gateoff_P'] or 'none'} gateoff_mean_update_fraction={row['gateoff_mean_update_fraction'] or 'none'} "
            f"bracket_lo={row['in_domain_bracket_lo'] or 'none'} bracket_hi_or_gateoff={row['in_domain_bracket_hi_or_gateoff'] or 'none'} "
            f"proposed_refine={row['proposed_refine_c_values'] or 'none'} note={row['note']}"
        )
    if missing:
        lines.extend(["", "First 50 missing short bridge runs", *missing[:50]])
    write_text(REPORT_BASE / "STAGE_B_COMPLETION_REPORT.txt", "\n".join(lines) + "\n")

    return {
        "stage_b_complete": stage_b_complete,
        "in_domain_count": in_domain_count,
        "edge_censored_count": edge_censored_count,
        "no_boundary_gateoff_count": no_boundary_gateoff_count,
        "unresolved_alpha_count": len(unresolved_alphas),
        "stage_c_candidates": stage_c_candidates,
    }


def summarize_fullref() -> dict:
    rows: list[dict] = []
    missing: list[str] = []
    for seed in FULLREF_SEEDS:
        run_dir = FULLREF_TARGET_BASE / f"seed{seed}"
        summary_path = run_dir / "gate5_summary_AUTOGEN.txt"
        if not summary_path.is_file():
            missing.append(str(seed))
            continue
        d = parse_summary(summary_path)
        rows.append(
            {
                "run_dir": str(run_dir),
                "seed": str(seed),
                "eval_return_end_mean": safe_float(d, "eval_return_end_mean"),
                "eval_dbar_post": safe_float(d, "eval_dbar_post"),
                "eval_occupied_post_mean": safe_float(d, "eval_occupied_post_mean"),
                "eval_CSR_mean": safe_float(d, "eval_CSR_mean"),
                "mean_update_fraction": safe_float(d, "mean_update_fraction"),
                "final_VarTD": safe_float(d, "final_VarTD"),
                "wall_seconds": read_wall_seconds(run_dir),
            }
        )

    write_csv(REPORT_BASE / "fullref_32seed_per_run.csv", rows, list(rows[0].keys()) if rows else ["run_dir"])

    rets = [float(r["eval_return_end_mean"]) for r in rows]
    dbars = [float(r["eval_dbar_post"]) for r in rows]
    occs = [float(r["eval_occupied_post_mean"]) for r in rows]
    csrs = [float(r["eval_CSR_mean"]) for r in rows]
    lo, hi = bootstrap_mean_ci(rets, seed=4242) if rets else (float("nan"), float("nan"))
    frac_below_random = (sum(r < R0 for r in rets) / len(rets)) if rets else float("nan")

    lines = [
        "Full-horizon ungated reference 32-seed report",
        "",
        f"completed_runs={len(rows)}",
        f"expected_runs={len(FULLREF_SEEDS)}",
        f"missing_runs={len(missing)}",
    ]
    if rets:
        lines.extend(
            [
                f"mean_eval_return_end_mean={mean(rets):.6f}",
                f"sd_eval_return_end_mean={stdev(rets):.6f}" if len(rets) >= 2 else "sd_eval_return_end_mean=0.000000",
                f"median_eval_return_end_mean={median(rets):.6f}",
                f"min_eval_return_end_mean={min(rets):.6f}",
                f"max_eval_return_end_mean={max(rets):.6f}",
                f"bootstrap95_mean_return_lo={lo:.6f}",
                f"bootstrap95_mean_return_hi={hi:.6f}",
                f"fraction_below_R0={frac_below_random:.6f}",
                f"mean_eval_dbar_post={mean(dbars):.6f}",
                f"mean_eval_occupied_post_mean={mean(occs):.6f}",
                f"mean_eval_CSR_mean={mean(csrs):.6f}",
            ]
        )
    if missing:
        lines.extend(["", "Missing full-reference seeds", ",".join(missing)])
    write_text(REPORT_BASE / "fullref_32seed_report.txt", "\n".join(lines) + "\n")

    fullref_ready = len(rows) == len(FULLREF_SEEDS) and not math.isnan(lo) and lo > R0
    return {
        "completed_runs": len(rows),
        "expected_runs": len(FULLREF_SEEDS),
        "missing_runs": len(missing),
        "mean_return": mean(rets) if rets else float("nan"),
        "sd_return": stdev(rets) if len(rets) >= 2 else float("nan"),
        "bootstrap_lo": lo,
        "bootstrap_hi": hi,
        "frac_below_random": frac_below_random,
        "fullref_ready": fullref_ready,
    }


def write_stage_c_subset_readiness(short_meta: dict, full_meta: dict) -> None:
    stage_c_subset_ready = bool(short_meta["stage_b_complete"] and full_meta["fullref_ready"] and short_meta["stage_c_candidates"])
    focus_alphas = ",".join(row["alpha"] for row in short_meta["stage_c_candidates"]) if short_meta["stage_c_candidates"] else "none"

    lines = [
        "Stage C subset readiness report",
        "",
        "Locked facts",
        f"R0={R0:.6f}",
        f"R_REF_SHORT={R_REF_SHORT:.6f}",
        f"R_THRESH_SHORT={R_THRESH_SHORT:.6f}",
        "",
        "Prerequisite status",
        f"stage_b_complete={'yes' if short_meta['stage_b_complete'] else 'no'}",
        f"short_bridge_in_domain_boundary_count={short_meta['in_domain_count']}",
        f"short_bridge_edge_censored_count={short_meta['edge_censored_count']}",
        f"short_bridge_no_boundary_even_gateoff_count={short_meta['no_boundary_gateoff_count']}",
        f"short_bridge_unresolved_alpha_count={short_meta['unresolved_alpha_count']}",
        f"fullref_32seed_completed={'yes' if full_meta['missing_runs'] == 0 else 'no'}",
        f"fullref_32seed_mean_return={full_meta['mean_return']:.6f}" if not math.isnan(full_meta['mean_return']) else "fullref_32seed_mean_return=nan",
        f"fullref_32seed_sd_return={full_meta['sd_return']:.6f}" if not math.isnan(full_meta['sd_return']) else "fullref_32seed_sd_return=nan",
        f"fullref_32seed_bootstrap95_mean_return=[{full_meta['bootstrap_lo']:.6f}, {full_meta['bootstrap_hi']:.6f}]" if not math.isnan(full_meta['bootstrap_lo']) else "fullref_32seed_bootstrap95_mean_return=[nan, nan]",
        f"fullref_32seed_fraction_below_R0={full_meta['frac_below_random']:.6f}" if not math.isnan(full_meta['frac_below_random']) else "fullref_32seed_fraction_below_R0=nan",
        f"fullref_32seed_mean_confidently_above_R0={'yes' if full_meta['fullref_ready'] else 'no'}",
        "",
        "Decision",
        f"stage_c_subset_ready={'yes' if stage_c_subset_ready else 'no'}",
        f"recommended_stage_c_focus_alphas={focus_alphas}",
        "note=This script never auto-launches Stage C. It only tells you whether the locked Stage-B bridge now supports a defensible Stage-C subset while allowing no-boundary and right-censored alphas to be reported honestly.",
        "",
        "Next files to inspect",
        "STAGE_B_COMPLETION_REPORT.txt",
        "stageB_alpha_classification.csv",
        "fullref_32seed_report.txt",
        "stageC_candidate_alphas_from_bridge.csv",
    ]
    write_text(REPORT_BASE / "STAGE_C_SUBSET_READINESS_REPORT.txt", "\n".join(lines) + "\n")


# =========================
# Manifest export
# =========================
def write_manifest_csv() -> None:
    rows = [
        {"alpha": a, "c_upd": c, "roles": roles, "notes": notes}
        for a, c, roles, notes in MANIFEST
    ]
    write_csv(REPORT_BASE / "stageB_completion_bridge_manifest.csv", rows, ["alpha", "c_upd", "roles", "notes"])


# =========================
# Main
# =========================
def main() -> None:
    need_file(SCRIPT_PATH)
    need_file(CONFIG_PATH)
    ensure_dir(SHORT_TARGET_BASE)
    ensure_dir(FULLREF_TARGET_BASE)
    ensure_dir(REPORT_BASE)

    selected_short_points = len(MANIFEST)
    existing_short_points = sum(1 for _, c, _, _ in MANIFEST if float(c) <= 1.0)
    gateoff_points = sum(1 for _, c, _, _ in MANIFEST if float(c) > 1.0)
    short_slot_count = selected_short_points * len(SHORT_SEEDS)
    short_reuse_if_prior = existing_short_points * 8
    short_new_if_prior = short_slot_count - short_reuse_if_prior
    fullref_slot_count = len(FULLREF_SEEDS)
    fullref_new_if_prior = fullref_slot_count - 24

    print(f"[stageB-completion-bridge] repo_root={Path.cwd()}")
    print(f"[stageB-completion-bridge] python={sys.executable}")
    print(f"[stageB-completion-bridge] short_target_dir={SHORT_TARGET_BASE}")
    print(f"[stageB-completion-bridge] fullref_target_dir={FULLREF_TARGET_BASE}")
    print(f"[stageB-completion-bridge] report_dir={REPORT_BASE}")
    print(f"[stageB-completion-bridge] selected_short_points={selected_short_points}")
    print(f"[stageB-completion-bridge] existing_short_points_in_domain={existing_short_points}")
    print(f"[stageB-completion-bridge] gateoff_control_points={gateoff_points}")
    print(f"[stageB-completion-bridge] short_seed_slots={short_slot_count}")
    print(f"[stageB-completion-bridge] short_expected_new_runs_if_previous_dense_outputs_present={short_new_if_prior}")
    print(f"[stageB-completion-bridge] fullref_seed_slots={fullref_slot_count}")
    print(f"[stageB-completion-bridge] fullref_expected_new_runs_if_24seed_outputs_present={fullref_new_if_prior}")

    write_manifest_csv()

    print("[stageB-completion-bridge] === short-horizon bridge runs ===")
    for alpha, c_upd, _, _ in MANIFEST:
        for seed in SHORT_SEEDS:
            run_short_point_seed(alpha, c_upd, seed)

    print("[stageB-completion-bridge] === full-horizon ungated reference 32-seed top-up ===")
    for seed in FULLREF_SEEDS:
        run_fullref_seed(seed)

    print("[stageB-completion-bridge] === summarizing ===")
    short_meta = summarize_short_bridge()
    full_meta = summarize_fullref()
    write_stage_c_subset_readiness(short_meta, full_meta)

    print("[stageB-completion-bridge] done")
    print(f"[stageB-completion-bridge] read: {REPORT_BASE / 'STAGE_B_COMPLETION_REPORT.txt'}")
    print(f"[stageB-completion-bridge] read: {REPORT_BASE / 'STAGE_C_SUBSET_READINESS_REPORT.txt'}")
    print(f"[stageB-completion-bridge] read: {REPORT_BASE / 'fullref_32seed_report.txt'}")


if __name__ == "__main__":
    main()