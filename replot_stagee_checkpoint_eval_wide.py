#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE_TITLES = {
    "reproduce_baseline": "Phase E0B baseline reproduction",
    "baseline_4m": "Phase E0C 4M baseline robustness",
    "rescue_short": "Phase E1 short-horizon rescue discovery",
    "extension_full": "Phase E2 full-horizon and 4M intervention tests",
}


@dataclass(frozen=True)
class OverlaySpec:
    output_name: str
    metric: str
    title_suffix: str
    ylabel: str

    @property
    def y_col(self) -> str:
        return f"{self.metric}_mean"

    @property
    def wide_output_name(self) -> str:
        stem = Path(self.output_name).stem
        suffix = Path(self.output_name).suffix
        return f"{stem}_wide{suffix}"


# These five files are exactly the cross-condition overlays that the Stage E
# script writes via save_phase_overlays(...) for each phase.
OVERLAY_SPECS = [
    OverlaySpec(
        output_name="checkpoint_eval_return_overlay.png",
        metric="eval_return_end_mean_cp",
        title_suffix="checkpoint eval return",
        ylabel="eval return",
    ),
    OverlaySpec(
        output_name="energy_overlay.png",
        metric="energy_E_post",
        title_suffix="post-update energy",
        ylabel="energy",
    ),
    OverlaySpec(
        output_name="debt_overlay.png",
        metric="debt_level_post",
        title_suffix="debt level",
        ylabel="debt",
    ),
    OverlaySpec(
        output_name="support_overlay.png",
        metric="support_s",
        title_suffix="support schedule",
        ylabel="support",
    ),
    OverlaySpec(
        output_name="update_counter_overlay.png",
        metric="update_counter",
        title_suffix="executed updates",
        ylabel="executed updates",
    ),
]


def infer_repo_root(start: Path) -> Path:
    start = start.resolve()

    # Normal full-repository layout.
    for candidate in [start, *start.parents]:
        if (candidate / "runs" / "stageE_clipped_robustness_extension").exists():
            return candidate

    # Convenience fallback: direct Stage-E archive root.
    for candidate in [start, *start.parents]:
        if (candidate / "extension_full").exists() and (candidate / "baseline_4m").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find either a repository root containing "
        "runs/stageE_clipped_robustness_extension or a direct Stage-E archive root. "
        "Run this script from the full repository, or pass --repo-root explicitly."
    )



def resolve_phase_dir(repo_root: Path, phase: str) -> Path:
    phase_dir = repo_root / "runs" / "stageE_clipped_robustness_extension" / phase
    if phase_dir.exists():
        return phase_dir

    alt_phase_dir = repo_root / phase
    if alt_phase_dir.exists():
        return alt_phase_dir

    raise FileNotFoundError(f"Missing phase directory: {phase_dir}")



def ordered_condition_labels(phase_dir: Path) -> List[str]:
    manifest_path = phase_dir / "phase_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if "label" in manifest.columns:
            labels = [str(x) for x in manifest["label"].tolist()]
            if labels:
                return labels

    conditions_dir = phase_dir / "conditions"
    if not conditions_dir.exists():
        raise FileNotFoundError(
            f"{conditions_dir} does not exist. The full repository should contain "
            "conditions/<condition>/trace_aggregate.csv files."
        )

    return sorted(p.name for p in conditions_dir.iterdir() if p.is_dir())



def load_trace_aggregate(trace_path: Path, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(trace_path)

    if x_col not in df.columns:
        raise KeyError(f"{trace_path} is missing x column {x_col!r}")
    if y_col not in df.columns:
        raise KeyError(f"{trace_path} is missing y column {y_col!r}")

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]



def plot_overlay(
    phase_dir: Path,
    labels: Iterable[str],
    *,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    width: float,
    height: float,
    dpi: int,
    legend_fontsize: float,
    right_margin: float,
    legend_anchor_x: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height))
    plotted = False

    for label in labels:
        trace_path = phase_dir / "conditions" / label / "trace_aggregate.csv"
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing expected aggregate trace file: {trace_path}")

        x, y = load_trace_aggregate(trace_path, x_col=x_col, y_col=y_col)
        if y.size == 0:
            continue

        ax.plot(x, y, linewidth=2.0, label=label)
        plotted = True

    if not plotted:
        raise RuntimeError(f"No finite {y_col!r} values were found in {phase_dir / 'conditions'}")

    ax.set_xlabel("update")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Reserve right-side figure space and place the legend fully outside the axes.
    fig.subplots_adjust(right=right_margin)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(legend_anchor_x, 0.5),
        fontsize=legend_fontsize,
        frameon=True,
        borderaxespad=0.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def select_overlay_specs(requested_names: Optional[List[str]]) -> List[OverlaySpec]:
    if not requested_names:
        return list(OVERLAY_SPECS)

    by_name = {spec.output_name: spec for spec in OVERLAY_SPECS}
    selected: List[OverlaySpec] = []
    for name in requested_names:
        if name not in by_name:
            valid = ", ".join(spec.output_name for spec in OVERLAY_SPECS)
            raise ValueError(f"Unknown overlay {name!r}. Valid values are: {valid}")
        selected.append(by_name[name])
    return selected



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-render the Stage E cross-condition overlays with a wider figure and an external legend, "
            "without rerunning training. By default this rewrites all overlays for extension_full."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Any directory inside the full repository. Default: current working directory.",
    )
    parser.add_argument(
        "--phase",
        default="extension_full",
        choices=["reproduce_baseline", "baseline_4m", "rescue_short", "extension_full"],
        help="Which Stage E phase to replot. Default: extension_full.",
    )
    parser.add_argument(
        "--overlay",
        action="append",
        default=None,
        help=(
            "Optional original overlay filename to replot. May be passed multiple times. "
            "Default: replot all files written by save_phase_overlays(...)."
        ),
    )
    parser.add_argument(
        "--x-col",
        default="update",
        help="X-axis column in each trace_aggregate.csv. Default: update.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=20.0,
        help="Figure width in inches. Default: 20.0.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=6.8,
        help="Figure height in inches. Default: 6.8.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Saved image DPI. Default: 160.",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=9.0,
        help="Legend font size. Default: 9.0.",
    )
    parser.add_argument(
        "--right-margin",
        type=float,
        default=0.63,
        help=(
            "Figure right margin passed to fig.subplots_adjust(right=...). "
            "Smaller values leave more space for the legend. Default: 0.63."
        ),
    )
    parser.add_argument(
        "--legend-anchor-x",
        type=float,
        default=1.01,
        help="Legend bbox_to_anchor x coordinate. Default: 1.01.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original PNGs instead of writing *_wide.png files.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    repo_root = infer_repo_root(args.repo_root)
    phase_dir = resolve_phase_dir(repo_root, args.phase)
    labels = ordered_condition_labels(phase_dir)
    specs = select_overlay_specs(args.overlay)

    cross_dir = phase_dir / "cross_condition_plots"
    cross_dir.mkdir(parents=True, exist_ok=True)

    print(f"repo_root={repo_root}")
    print(f"phase={args.phase}")
    print(f"n_conditions={len(labels)}")
    print(f"cross_condition_plots={cross_dir}")

    for spec in specs:
        title = f"{PHASE_TITLES.get(args.phase, args.phase)}: {spec.title_suffix}"
        output_path = cross_dir / (spec.output_name if args.overwrite else spec.wide_output_name)

        plot_overlay(
            phase_dir,
            labels,
            x_col=args.x_col,
            y_col=spec.y_col,
            title=title,
            ylabel=spec.ylabel,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
            legend_fontsize=args.legend_fontsize,
            right_margin=args.right_margin,
            legend_anchor_x=args.legend_anchor_x,
            output_path=output_path,
        )

        print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
