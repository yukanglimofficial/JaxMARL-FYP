# Energy-Gated Learning in Cooperative Multi-Agent Reinforcement Learning

This repository is the public code and run-summary repository for the dissertation *Energy-Gated Learning in Cooperative Multi-Agent Reinforcement Learning: Clipped-Energy Regimes and an Engineered Hard-Zero Crossover in JaxMARL Simple Spread*. It is organized around two empirical components:

- the principal clipped-energy study of reward-funded, energy-gated IPPO, including the clipped Stage E robustness extension;
- the separate engineered hard-zero extension implemented in the `n3_*` scripts.

The locked Simple Spread setting for the main study is:

- environment: `MPE_simple_spread_v3`
- agents / landmarks: `N = 3`
- reward mixing: `local_ratio = 0.5`
- action space: discrete
- horizon parameter: `max_cycles = 25` in Gate 4 and `max_steps = 25` in Gate 5 and later
- vectorized baseline: `NUM_ENVS = 25`, `NUM_STEPS = 128`

## Principal clipped-core result

The principal clipped-core result is the following.

- Stage B identified one short-horizon in-domain bracket at `alpha = 0.0` between `c_upd = 0.50` and `0.60`.
- Stage C completed all `1088` planned full-horizon refinement runs and is the authoritative source for clipped-core boundary claims.
- At the principal full horizon, `alpha = 0.0` stayed above threshold through `c_upd = 0.58`.
- At the principal full horizon, `alpha = 0.1` through `1.0` remained right-censored above `c_upd = 1.0`.
- Stage D is retained for mechanism interpretation only; it does not override Stage C on clipped-core boundaries.

The clipped Stage E workflow broadens the descriptive picture but does not redraw the Stage C map. In the checked-in reproduction-guarded v5 workflow, support partially rescues the short-horizon failing anchor, both support and borrowing raise learning probability at the fragile full-horizon anchor, and the `4,000,000`-step baseline probe establishes additional persistence and late-decline facts. The engineered hard-zero extension is a separate explicit extension with its own observables, figures, and interpretations.

For the principal clipped-core boundary claim, use Stage B and Stage C. Stage D is mechanism-only evidence. The clipped Stage E workflow extends the descriptive picture beyond the main map, and the engineered `n3_*` scripts address a separate hard-zero extension question. Consistent with the dissertation, this repository is best understood as containing the principal scripts, staged run summaries, and archived figure sources used in the manuscript.

## Environment setup

Run all commands from the repository root.

```bash
cd <repo-root>
```

The recorded runs were executed on Ubuntu 24.04 under WSL2 with Python `3.12.3`. The repository includes both a conda capture (`environment.yml`) and an exact pip freeze (`requirements-exact.txt`) for that software stack.

```bash
conda env create -f environment.yml
conda activate final-venv-capture
```

A Linux / WSL `venv` path also works.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-exact.txt
```

The repository vendors the local `jaxmarl/` tree, so run all commands from the repository root.

## Reproduction notes

The run counts below are clean-rerun counts. Several scripts are resumable and will skip, cache, or reuse existing outputs when compatible run trees are already present.

A few details matter for exact path matching.

First, **Gate 2** writes its report only because stdout is redirected. Running the script without redirection recomputes the check, but it does not recreate the same report path.

Second, **Gate 3** writes beside `__file__`. To recreate the checked-in Gate 3 artifact directory exactly, copy the script into the target directory, run it there with `PYTHONPATH=../..`, then remove the copied script.

Third, **Gate 4** and **Gate 5+** use different environment keyword names by design.

- Gate 4 uses `ENV_KWARGS.N` and `ENV_KWARGS.max_cycles`.
- Gate 5 and later scripts use `++ENV_KWARGS.num_agents`, `++ENV_KWARGS.num_landmarks`, and `++ENV_KWARGS.max_steps`.

Fourth, the direct Gate 5 suite was launched under `baseline_artifacts/gate5_reacceptance_20260316_204642/` and then renamed to `baseline_artifacts/gate5_mechanics_acceptance_20260316_204642/`. If exact outer-folder matching matters, launch under the original name first and rename the parent directory after the four runs finish.

Fifth, when `TOTAL_TIMESTEPS = 1000000` with `NUM_ENVS = 25` and `NUM_STEPS = 128`, the realized environment-step count is `998,400` because the training scripts use an integer update count. The commands below therefore report the configured `TOTAL_TIMESTEPS` value where relevant, not an exact realized step count.

## Reproduction order

For the main study, run:

1. Gate 2
2. Gate 3
3. Gate 4
4. the four direct Gate 5 mechanics checks
5. Stage B
6. Stage C
7. Stage D

Then run the clipped Stage E robustness extension.

Then run the separate engineered hard-zero extension (`n3_*` scripts).

## Main study

### Gate 2 — `verify_step_env_semantics.py`

What it does:

- one semantic verification pass
- no training
- no seeds

Run count:

- `1` script execution
- `0` training jobs

Command:

```bash
mkdir -p baseline_artifacts/gate2_step_env_semantics

python verify_step_env_semantics.py \
  > baseline_artifacts/gate2_step_env_semantics/step_env_semantics_report.txt 2>&1
```

Main output:

- `baseline_artifacts/gate2_step_env_semantics/step_env_semantics_report.txt`

### Gate 3 — `verify_terminal_geometry_and_calibration.py`

What it does:

- one calibration and terminal-geometry contract check
- no training
- no seeds

Run count:

- `1` script execution
- `0` training jobs

Command:

```bash
mkdir -p baseline_artifacts/gate3_terminal_geometry_and_calibration
cp verify_terminal_geometry_and_calibration.py baseline_artifacts/gate3_terminal_geometry_and_calibration/

(
  cd baseline_artifacts/gate3_terminal_geometry_and_calibration
  PYTHONPATH=../.. python verify_terminal_geometry_and_calibration.py \
    > terminal_geometry_and_calibration.log 2>&1
)

rm baseline_artifacts/gate3_terminal_geometry_and_calibration/verify_terminal_geometry_and_calibration.py
```

Main outputs include:

- `baseline_artifacts/gate3_terminal_geometry_and_calibration/obs_layout_contract.json`
- `baseline_artifacts/gate3_terminal_geometry_and_calibration/eps_occ_sanity_histogram.png`
- `baseline_artifacts/gate3_terminal_geometry_and_calibration/calibration_random_baselines.json`
- `baseline_artifacts/gate3_terminal_geometry_and_calibration/terminal_geometry_and_calibration.log`

### Gate 4 — `baselines/IPPO/ippo_ff_mpe_gate4_stepenv.py`

What it does:

- one direct run with `TOTAL_TIMESTEPS = 1000000` (`998,400` realized environment steps)
- one seed
- terminal-correct `step_env` plus masked reset integrated into the collector
- training-time terminal geometry metrics recorded directly from that collector

Run count:

- `1` Hydra run
- `1` seed
- `1` training job

Command:

```bash
mkdir -p baseline_artifacts/gate4_training_collector_geometry/full_1e6_eps0p2

python baselines/IPPO/ippo_ff_mpe_gate4_stepenv.py \
  ENV_NAME=MPE_simple_spread_v3 \
  ENV_KWARGS.N=3 \
  ENV_KWARGS.local_ratio=0.5 \
  ENV_KWARGS.max_cycles=25 \
  ENV_KWARGS.action_type=Discrete \
  NUM_ENVS=25 \
  NUM_STEPS=128 \
  TOTAL_TIMESTEPS=1000000 \
  NUM_SEEDS=1 \
  SEED=30 \
  WANDB_MODE=disabled \
  +EPS_OCC=0.2 \
  hydra.run.dir=baseline_artifacts/gate4_training_collector_geometry/full_1e6_eps0p2 \
  hydra.job.chdir=True \
  > baseline_artifacts/gate4_training_collector_geometry/full_1e6_eps0p2/stdout.log 2>&1
```

Main outputs include:

- `baseline_artifacts/gate4_training_collector_geometry/full_1e6_eps0p2/.hydra/*`
- `baseline_artifacts/gate4_training_collector_geometry/full_1e6_eps0p2/stdout.log`
- `baseline_artifacts/gate4_training_collector_geometry/full_1e6_eps0p2/ippo_ff_mpe_gate4_stepenv.log`
- `baseline_artifacts/gate4_training_collector_geometry/full_1e6_eps0p2/ippo_ff_gate4_return_MPE_simple_spread_v3.png`
- `baseline_artifacts/gate4_training_collector_geometry/full_1e6_eps0p2/ippo_ff_gate4_episode_len_MPE_simple_spread_v3.png`
- `baseline_artifacts/gate4_training_collector_geometry/full_1e6_eps0p2/ippo_ff_gate4_min_dists_end_MPE_simple_spread_v3.png`
- `baseline_artifacts/gate4_training_collector_geometry/full_1e6_eps0p2/ippo_ff_gate4_CSR_train_MPE_simple_spread_v3.png`

### Direct Gate 5 mechanics suite — `baselines/IPPO/ippo_ff_mpe_gate5_energy_gated.py`

What it does:

- four direct one-seed mechanics checks
- two quick checks at `32,000` timesteps
- two full checks with `TOTAL_TIMESTEPS = 1000000` (`998,400` realized environment steps each)

Run count:

- `4` Hydra runs
- `1` seed each
- `4` training jobs

Use the original Hydra root first.

```bash
G5_ROOT=baseline_artifacts/gate5_reacceptance_20260316_204642
```

#### `quick_gating_off`

```bash
mkdir -p "$G5_ROOT"/runs/quick_gating_off

python baselines/IPPO/ippo_ff_mpe_gate5_energy_gated.py \
  ENV_NAME=MPE_simple_spread_v3 \
  NUM_ENVS=25 \
  NUM_STEPS=128 \
  TOTAL_TIMESTEPS=32000 \
  NUM_SEEDS=1 \
  SEED=30 \
  WANDB_MODE=disabled \
  ++ENV_KWARGS.num_agents=3 \
  ++ENV_KWARGS.num_landmarks=3 \
  ++ENV_KWARGS.local_ratio=0.5 \
  ++ENV_KWARGS.max_steps=25 \
  ++ENV_KWARGS.action_type=Discrete \
  ++EPS_OCC=0.2 \
  ++D0=0.733348 \
  ++R0=-27.256548 \
  ++BETA=2.302585092994046 \
  ++Z_MAX=10.0 \
  ++EPS_NORM=1e-8 \
  ++USE_GEOM_FUNDING=True \
  ++DO_EVAL=True \
  ++M_EVAL=100 \
  ++C_UPD=0.0 \
  ++ALPHA=0.0 \
  ++E0=1.0 \
  ++E_MAX=1.0 \
  hydra.run.dir="$G5_ROOT"/runs/quick_gating_off \
  hydra.job.chdir=True \
  > "$G5_ROOT"/runs/quick_gating_off/stdout.log 2>&1
```

#### `quick_gating_on`

```bash
mkdir -p "$G5_ROOT"/runs/quick_gating_on

python baselines/IPPO/ippo_ff_mpe_gate5_energy_gated.py \
  ENV_NAME=MPE_simple_spread_v3 \
  NUM_ENVS=25 \
  NUM_STEPS=128 \
  TOTAL_TIMESTEPS=32000 \
  NUM_SEEDS=1 \
  SEED=30 \
  WANDB_MODE=disabled \
  ++ENV_KWARGS.num_agents=3 \
  ++ENV_KWARGS.num_landmarks=3 \
  ++ENV_KWARGS.local_ratio=0.5 \
  ++ENV_KWARGS.max_steps=25 \
  ++ENV_KWARGS.action_type=Discrete \
  ++EPS_OCC=0.2 \
  ++D0=0.733348 \
  ++R0=-27.256548 \
  ++BETA=2.302585092994046 \
  ++Z_MAX=10.0 \
  ++EPS_NORM=1e-8 \
  ++USE_GEOM_FUNDING=True \
  ++DO_EVAL=True \
  ++M_EVAL=100 \
  ++C_UPD=0.02 \
  ++ALPHA=0.10 \
  ++E0=0.02 \
  ++E_MAX=1.0 \
  hydra.run.dir="$G5_ROOT"/runs/quick_gating_on \
  hydra.job.chdir=True \
  > "$G5_ROOT"/runs/quick_gating_on/stdout.log 2>&1
```

#### `full_1e6_gating_off`

```bash
mkdir -p "$G5_ROOT"/runs/full_1e6_gating_off

python baselines/IPPO/ippo_ff_mpe_gate5_energy_gated.py \
  ENV_NAME=MPE_simple_spread_v3 \
  NUM_ENVS=25 \
  NUM_STEPS=128 \
  TOTAL_TIMESTEPS=1000000 \
  NUM_SEEDS=1 \
  SEED=30 \
  WANDB_MODE=disabled \
  ++ENV_KWARGS.num_agents=3 \
  ++ENV_KWARGS.num_landmarks=3 \
  ++ENV_KWARGS.local_ratio=0.5 \
  ++ENV_KWARGS.max_steps=25 \
  ++ENV_KWARGS.action_type=Discrete \
  ++EPS_OCC=0.2 \
  ++D0=0.733348 \
  ++R0=-27.256548 \
  ++BETA=2.302585092994046 \
  ++Z_MAX=10.0 \
  ++EPS_NORM=1e-8 \
  ++USE_GEOM_FUNDING=True \
  ++DO_EVAL=True \
  ++M_EVAL=100 \
  ++C_UPD=0.0 \
  ++ALPHA=0.0 \
  ++E0=1.0 \
  ++E_MAX=1.0 \
  hydra.run.dir="$G5_ROOT"/runs/full_1e6_gating_off \
  hydra.job.chdir=True \
  > "$G5_ROOT"/runs/full_1e6_gating_off/stdout.log 2>&1
```

#### `full_1e6_gating_mild`

```bash
mkdir -p "$G5_ROOT"/runs/full_1e6_gating_mild

python baselines/IPPO/ippo_ff_mpe_gate5_energy_gated.py \
  ENV_NAME=MPE_simple_spread_v3 \
  NUM_ENVS=25 \
  NUM_STEPS=128 \
  TOTAL_TIMESTEPS=1000000 \
  NUM_SEEDS=1 \
  SEED=30 \
  WANDB_MODE=disabled \
  ++ENV_KWARGS.num_agents=3 \
  ++ENV_KWARGS.num_landmarks=3 \
  ++ENV_KWARGS.local_ratio=0.5 \
  ++ENV_KWARGS.max_steps=25 \
  ++ENV_KWARGS.action_type=Discrete \
  ++EPS_OCC=0.2 \
  ++D0=0.733348 \
  ++R0=-27.256548 \
  ++BETA=2.302585092994046 \
  ++Z_MAX=10.0 \
  ++EPS_NORM=1e-8 \
  ++USE_GEOM_FUNDING=True \
  ++DO_EVAL=True \
  ++M_EVAL=100 \
  ++C_UPD=0.02 \
  ++ALPHA=0.10 \
  ++E0=0.02 \
  ++E_MAX=1.0 \
  hydra.run.dir="$G5_ROOT"/runs/full_1e6_gating_mild \
  hydra.job.chdir=True \
  > "$G5_ROOT"/runs/full_1e6_gating_mild/stdout.log 2>&1
```

Rename the parent directory after the suite finishes.

```bash
mv "$G5_ROOT" baseline_artifacts/gate5_mechanics_acceptance_20260316_204642
```

Main outputs from the four raw commands:

- `baseline_artifacts/gate5_mechanics_acceptance_20260316_204642/runs/<four run dirs>/`

The checked-in artifact directory also contains `baseline_artifacts/gate5_mechanics_acceptance_20260316_204642/gate5_acceptance_report.md` and `baseline_artifacts/gate5_mechanics_acceptance_20260316_204642/gate5_acceptance_report.json`. Those two prebuilt summary files are included in the repository and are not produced by the four raw training commands alone.

### Stage B — `stageB_completion_bridge.py`

What it does:

- launches a locked 32-seed short-horizon bridge on selected `(alpha, c_upd)` points
- extends that bridge with one 32-seed full-horizon ungated reference ensemble
- writes the per-run and per-point summaries used to define the Stage C subset

The short-horizon bridge manifest is:

```text
alpha=0.0 -> c_upd = 0.10, 0.50, 0.60, 1.00, 1.05
alpha=0.1 -> c_upd = 0.10, 0.90, 1.00, 1.05
alpha=0.2 -> c_upd = 0.80, 0.90, 1.00, 1.05
alpha=0.3 -> c_upd = 0.90, 0.60, 1.00, 1.05
alpha=0.4 -> c_upd = 0.90, 1.00, 1.05
alpha=0.5 -> c_upd = 0.40, 0.70, 1.00, 1.05
alpha=0.6 -> c_upd = 0.80, 1.00, 1.05
alpha=0.7 -> c_upd = 0.90, 1.00, 1.05
alpha=0.8 -> c_upd = 0.80, 1.00, 1.05
alpha=0.9 -> c_upd = 0.80, 1.00, 1.05
alpha=1.0 -> c_upd = 0.90, 1.00, 1.05
```

Run count:

- short-horizon bridge: `39` points × `32` seeds = `1248` seed-runs
- full-horizon ungated reference: `32` seed-runs
- total: `1280` seed-runs

Command:

```bash
python stageB_completion_bridge.py
```

Main outputs include:

- `runs/stageB_completion_bridge_32seed/`
- `runs/ref_full_ungated_32seed/`
- `runs/stageB_completion_bridge_reports/STAGE_B_COMPLETION_REPORT.txt`
- `runs/stageB_completion_bridge_reports/STAGE_C_SUBSET_READINESS_REPORT.txt`
- `runs/stageB_completion_bridge_reports/fullref_32seed_report.txt`
- `runs/stageB_completion_bridge_reports/fullref_32seed_per_run.csv`
- `runs/stageB_completion_bridge_reports/stageB_bridge_per_run.csv`
- `runs/stageB_completion_bridge_reports/stageB_bridge_by_point.csv`
- `runs/stageB_completion_bridge_reports/stageB_alpha_classification.csv`
- `runs/stageB_completion_bridge_reports/stageC_candidate_alphas_from_bridge.csv`

### Stage C — `stageC_subset_refine.py`

What it does:

- reads the Stage C candidate set emitted by Stage B
- verifies the locked 32-seed full-horizon ungated reference mean
- launches only the selected full-horizon refinement points
- writes the per-run table, per-point table, boundary table, missing-run list, and full report

Stage C candidate set:

- `alpha = 0.0`: `c_upd = 0.52, 0.54, 0.56, 0.58`
- `alpha = 0.1` through `1.0`: `c_upd = 0.90, 0.95, 1.00`

Run count:

- `34` points × `32` seeds = `1088` seed-runs

Commands:

```bash
python stageC_subset_refine.py
```

```bash
python stageC_subset_refine.py --summary-only
```

Main outputs include:

- `runs/stageC_subset_refine_full_32seed/`
- `runs/stageC_subset_refine_reports/stageC_subset_refine_manifest.csv`
- `runs/stageC_subset_refine_reports/stageC_subset_refine_per_run.csv`
- `runs/stageC_subset_refine_reports/stageC_subset_refine_by_point.csv`
- `runs/stageC_subset_refine_reports/stageC_subset_refine_alpha_boundary.csv`
- `runs/stageC_subset_refine_reports/stageC_subset_refine_missing_runs.txt`
- `runs/stageC_subset_refine_reports/STAGE_C_SUBSET_FULL_HORIZON_REPORT.txt`

### Stage D — `stageD_representative_mechanistic_traces.py`

What it does:

- verifies that Stage C is complete before doing anything else
- chooses representative conditions from the Stage B and Stage C summary tables
- reruns only those representative conditions across the locked seed range
- writes per-update traces, per-seed endpoint summaries, aggregate traces, cross-condition plots, and a report

Default condition set:

- `enabled_full`: `alpha = 1.0`, `c_upd = 0.90`, `TOTAL_TIMESTEPS = 1000000`
- `fragile_full`: `alpha = 0.0`, `c_upd = 0.58`, `TOTAL_TIMESTEPS = 1000000`
- `disabled_short`: `alpha = 0.0`, `c_upd = 0.60`, `TOTAL_TIMESTEPS = 400000`

Run count:

- default: `3` conditions × `32` seeds = `96` seed-runs
- with `--include-gateoff-control`: `4` conditions × `32` seeds = `128` seed-runs

Audit caveat:

The checked-in `fragile_full` endpoint summary is materially inconsistent with the accepted Stage C by-point summary for `(alpha = 0.0, c_upd = 0.58)`. Use Stage C as the source of truth for boundary claims and Stage D for mechanism-only evidence.

Commands:

```bash
python stageD_representative_mechanistic_traces.py
```

```bash
python stageD_representative_mechanistic_traces.py --include-gateoff-control
```

```bash
python stageD_representative_mechanistic_traces.py --force-rerun
```

Main outputs include:

- `runs/stageD_representative_mechanistic_traces/conditions/<condition>/seed_XX/trace_per_update.csv`
- `runs/stageD_representative_mechanistic_traces/conditions/<condition>/seed_XX/eval_summary.json`
- `runs/stageD_representative_mechanistic_traces/conditions/<condition>/seed_endpoint_summary.csv`
- `runs/stageD_representative_mechanistic_traces/conditions/<condition>/trace_per_seed_long.csv`
- `runs/stageD_representative_mechanistic_traces/conditions/<condition>/trace_aggregate.csv`
- `runs/stageD_representative_mechanistic_traces/conditions/<condition>/plots/*.png`
- `runs/stageD_representative_mechanistic_traces/cross_condition_plots/*.png`
- `runs/stageD_representative_mechanistic_traces/stageD_endpoint_summary_all_conditions.csv`
- `runs/stageD_representative_mechanistic_traces/stageD_condition_manifest.csv`
- `runs/stageD_representative_mechanistic_traces/stageD_manifest.json`
- `runs/stageD_representative_mechanistic_traces/STAGE_D_REPRESENTATIVE_MECHANISTIC_TRACES_REPORT.txt`

## Stage E — `stageE_clipped_robustness_extension.py`

What it does:

- writes a patched Stage E master at runtime under `baselines/IPPO/ippo_ff_mpe_stagee_clipped_extension.py`; this generated file need not already exist in a fresh clone
- preserves the clipped-core semantics that matter for the dissertation: pre-income gating, the upper clip `E_MAX = 1.0`, and the deterministic evaluation contract
- adds support schedules and a scientifically operative borrowing rule on top of the reference Gate 5 master
- organizes the clipped robustness extension into named phases with a reproduction gate before later intervention claims
- caches per-seed outputs and writes provenance files for each phase

Supported modes:

- `audit_existing`
- `reproduce_baseline`
- `baseline_4m`
- `rescue_short`
- `extension_full`
- `all`

The script enforces the baseline-reproduction gate before `baseline_4m`, `rescue_short`, `extension_full`, and the later phases of `all`. `--allow-unvalidated-baseline` disables that guard and is for debugging rather than reported runs. In the checked-in v5 workflow, the borrowing rule is made operative through `u_bool = (e_pre + borrow_limit) >= c_upd`, matching the Stage E description in the dissertation.

Important: for `baseline_4m`, `rescue_short`, and `extension_full`, the per-mode counts below are phase-local inventories. The exact standalone commands shown with `--force-rerun` first rerun `reproduce_baseline` (`128` seed-runs) unless a current validated baseline summary already exists and `--force-rerun` is omitted.

### `audit_existing`

What it does:

- archival audit only
- no training

Run count:

- `0` training runs

Command:

```bash
python stageE_clipped_robustness_extension.py audit_existing
```

Main outputs include:

- `runs/stageE_clipped_robustness_extension/audit_existing/stageE_archival_audit.txt`
- `runs/stageE_clipped_robustness_extension/audit_existing/claims_to_evidence_map.csv`
- `runs/stageE_clipped_robustness_extension/audit_existing/authoritative_vs_supportive_evidence.csv`

### `reproduce_baseline`

What it does:

- reruns four baseline anchor conditions with support and borrowing both off
- verifies that the extension patch preserves the clipped baseline semantics

Condition set:

- short horizon: `alpha = 0.0`, `c_upd = 0.60`, `TOTAL_TIMESTEPS = 400000`
- full horizon: `alpha = 0.0`, `c_upd = 0.58`, `TOTAL_TIMESTEPS = 1000000`
- full horizon: `alpha = 0.1`, `c_upd = 1.00`, `TOTAL_TIMESTEPS = 1000000`
- full horizon: `alpha = 1.0`, `c_upd = 0.90`, `TOTAL_TIMESTEPS = 1000000`

Run count:

- `4` conditions × `32` seeds = `128` seed-runs

Command:

```bash
python stageE_clipped_robustness_extension.py reproduce_baseline \
  --checkpoint-eval-every 20 \
  --force-repatch \
  --force-rerun
```

Main outputs include:

- `runs/stageE_clipped_robustness_extension/reproduce_baseline/phase_manifest.csv`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/phase_endpoint_summary_all_conditions.csv`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/phase_endpoint_summary_by_condition.csv`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/authoritative_comparison.csv`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/baseline_validation_per_condition.csv`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/baseline_validation_summary.json`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/checkpoint_behavior_by_condition.csv`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/conditions/<condition>/...`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/cross_condition_plots/*.png`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/phase_provenance.json`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/pip_freeze.txt`
- `runs/stageE_clipped_robustness_extension/reproduce_baseline/PHASE_REPORT.txt`

### `baseline_4m`

What it does:

- reruns the same four base-only anchor conditions at `TOTAL_TIMESTEPS = 4000000`
- establishes the long-horizon baseline before intervention claims

Run count:

- phase-local inventory: `4` conditions × `32` seeds = `128` seed-runs
- clean standalone command shown here with `--force-rerun`: `128 + 128 = 256` seed-runs

Command:

```bash
python stageE_clipped_robustness_extension.py baseline_4m \
  --checkpoint-eval-every 20 \
  --force-repatch \
  --force-rerun
```

Main outputs include:

- `runs/stageE_clipped_robustness_extension/baseline_4m/phase_manifest.csv`
- `runs/stageE_clipped_robustness_extension/baseline_4m/phase_endpoint_summary_all_conditions.csv`
- `runs/stageE_clipped_robustness_extension/baseline_4m/phase_endpoint_summary_by_condition.csv`
- `runs/stageE_clipped_robustness_extension/baseline_4m/authoritative_comparison.csv`
- `runs/stageE_clipped_robustness_extension/baseline_4m/checkpoint_behavior_by_condition.csv`
- `runs/stageE_clipped_robustness_extension/baseline_4m/conditions/<condition>/...`
- `runs/stageE_clipped_robustness_extension/baseline_4m/cross_condition_plots/*.png`
- `runs/stageE_clipped_robustness_extension/baseline_4m/phase_provenance.json`
- `runs/stageE_clipped_robustness_extension/baseline_4m/pip_freeze.txt`
- `runs/stageE_clipped_robustness_extension/baseline_4m/PHASE_REPORT.txt`

### `rescue_short`

What it does:

- runs the short-horizon failing anchor at `alpha = 0.0`, `c_upd = 0.60`, `TOTAL_TIMESTEPS = 400000`
- compares three families: `base`, `support`, and `borrow`
- expands to a local sweep at `c_upd = 0.54, 0.56, 0.58, 0.60, 0.62` when the built-in rescue criterion is met

Run count:

- phase-local anchor comparison: `3` conditions × `32` seeds = `96` seed-runs
- phase-local expanded sweep inventory: `18` conditions × `32` seeds = `576` seed-runs
- phase-local executed inventory when the expansion triggers: `672` seed-runs
- clean standalone command shown here with `--force-rerun`: `128 + 96 = 224` seed-runs without expansion, `128 + 672 = 800` seed-runs with expansion

Command:

```bash
python stageE_clipped_robustness_extension.py rescue_short \
  --checkpoint-eval-every 20 \
  --support-mode frontload \
  --support-total 0.60 \
  --support-start-update 0 \
  --support-window-frac 0.10 \
  --borrow-limit 0.60 \
  --borrow-interest 0.02 \
  --force-repatch \
  --force-rerun
```

Main outputs include:

- `runs/stageE_clipped_robustness_extension/rescue_short/phase_manifest.csv`
- `runs/stageE_clipped_robustness_extension/rescue_short/phase_endpoint_summary_all_conditions.csv`
- `runs/stageE_clipped_robustness_extension/rescue_short/phase_endpoint_summary_by_condition.csv`
- `runs/stageE_clipped_robustness_extension/rescue_short/authoritative_comparison.csv`
- `runs/stageE_clipped_robustness_extension/rescue_short/checkpoint_behavior_by_condition.csv`
- `runs/stageE_clipped_robustness_extension/rescue_short/conditions/<condition>/...`
- `runs/stageE_clipped_robustness_extension/rescue_short/cross_condition_plots/*.png`
- `runs/stageE_clipped_robustness_extension/rescue_short/phase_provenance.json`
- `runs/stageE_clipped_robustness_extension/rescue_short/pip_freeze.txt`
- `runs/stageE_clipped_robustness_extension/rescue_short/PHASE_REPORT.txt`

### `extension_full`

What it does:

- runs base, support, and borrow at the main fragile, edge, and enabled anchors
- covers both the accepted `TOTAL_TIMESTEPS = 1000000` horizon (`998,400` realized environment steps) and the `TOTAL_TIMESTEPS = 4000000` horizon

Default anchor set:

- `fragile_full = (alpha = 0.0, c_upd = 0.58)`
- `edge_full = (alpha = 0.1, c_upd = 1.00)`
- `enabled_full = (alpha = 1.0, c_upd = 0.90)`

Run count:

- phase-local default inventory: `3` anchors × `2` horizons × `3` families × `32` seeds = `576` seed-runs
- phase-local `--skip-enabled-control`: `384` seed-runs
- phase-local `--skip-4m-in-extension`: `288` seed-runs
- clean standalone command shown here with `--force-rerun`: `128 + 576 = 704` seed-runs by default, `128 + 384 = 512` with `--skip-enabled-control`, and `128 + 288 = 416` with `--skip-4m-in-extension`

Command:

```bash
python stageE_clipped_robustness_extension.py extension_full \
  --checkpoint-eval-every 20 \
  --support-mode frontload \
  --support-total 0.60 \
  --support-start-update 0 \
  --support-window-frac 0.10 \
  --borrow-limit 0.60 \
  --borrow-interest 0.02 \
  --force-repatch \
  --force-rerun
```

Main outputs include:

- `runs/stageE_clipped_robustness_extension/extension_full/phase_manifest.csv`
- `runs/stageE_clipped_robustness_extension/extension_full/phase_endpoint_summary_all_conditions.csv`
- `runs/stageE_clipped_robustness_extension/extension_full/phase_endpoint_summary_by_condition.csv`
- `runs/stageE_clipped_robustness_extension/extension_full/authoritative_comparison.csv`
- `runs/stageE_clipped_robustness_extension/extension_full/checkpoint_behavior_by_condition.csv`
- `runs/stageE_clipped_robustness_extension/extension_full/conditions/<condition>/...`
- `runs/stageE_clipped_robustness_extension/extension_full/cross_condition_plots/*.png`
- `runs/stageE_clipped_robustness_extension/extension_full/phase_provenance.json`
- `runs/stageE_clipped_robustness_extension/extension_full/pip_freeze.txt`
- `runs/stageE_clipped_robustness_extension/extension_full/PHASE_REPORT.txt`

The current checked-in run tree also contains additional widened cross-condition overlay PNGs under `runs/stageE_clipped_robustness_extension/extension_full/cross_condition_plots/`, produced by:

```bash
python replot_stagee_checkpoint_eval_wide.py
```

### `all`

What it does:

1. `audit_existing`
2. `reproduce_baseline`
3. `baseline_4m`
4. `rescue_short`
5. `extension_full`

Command:

```bash
python stageE_clipped_robustness_extension.py all \
  --checkpoint-eval-every 20 \
  --support-mode frontload \
  --support-total 0.60 \
  --support-start-update 0 \
  --support-window-frac 0.10 \
  --borrow-limit 0.60 \
  --borrow-interest 0.02 \
  --force-repatch \
  --force-rerun
```

Run count:

- checked-in all-mode training inventory without the optional `rescue_short` expansion: `(4 + 4 + 3 + 18) × 32 = 928` seed-runs
- conditional script-path total if the optional `rescue_short` expansion triggers under `--force-rerun`: `1504` seed-runs

## Engineered hard-zero extension (`n3_*` scripts)

This line of work is separate from the principal clipped-core Stage B / Stage C boundary result. In the dissertation it is presented as an explicit engineered hard-zero extension built on the same benchmark and reference master, with activity-centered observables and separate evidentiary claims. It does not revise the clipped-core map, and it is not used to support a claim of a true thermodynamic phase transition.

### `n3_engineered_phase_transition.py`

Modes:

- `discover`
- `lock_candidate`
- `graph_locked`

#### `discover`

What it does:

- discovery scan over engineered N=3 candidates
- pass 1: hard-zero only
- pass 2: hard-zero plus depreciation variants
- default alpha grid length: `10`
- default `c_upd` grid length: `6`
- default `z_cut` grid length: `3`
- default seed range: `30..41`, which is `12` seeds
- default horizon: `4,000,000` timesteps

Run count:

- pass 1: `6 × 3 = 18` candidates
- pass 2: `6 × 3 × 3 lambdas × 3 psis = 162` candidates
- total: `180` candidates
- clean rerun count: `180 × 10 × 12 = 21600` seed-runs

Command:

```bash
python n3_engineered_phase_transition.py discover \
  --force-repatch \
  --force-rerun
```

Main outputs include:

- `runs/n3_engineered_phase_transition/discover/DISCOVERY_REPORT.txt`
- `runs/n3_engineered_phase_transition/discover/discovery_ranked.csv`
- `runs/n3_engineered_phase_transition/discover/selected_candidate.json`
- `runs/n3_engineered_phase_transition/discover/selection_ladder_audit.csv`

#### `lock_candidate`

What it does:

- reruns the selected discovery candidate on the refined alpha grid
- default seed range: `30..61`, which is `32` seeds
- checked-in locked grid length: `41` alpha values
- default horizon: `4,000,000` timesteps

Run count:

- `41 × 32 = 1312` seed-runs

Command:

```bash
python n3_engineered_phase_transition.py lock_candidate \
  --force-repatch \
  --force-rerun
```

Main outputs include:

- `runs/n3_engineered_phase_transition/locked_candidate/<mechanism_tag>/...`

#### `graph_locked`

What it does:

- rebuilds the locked-candidate graph package from existing locked outputs
- no new training

Run count:

- `0` training runs

Command:

```bash
python n3_engineered_phase_transition.py graph_locked
```

### `n3_locked_candidate_phase_transition_finalize.py`

Mode:

- `run`

What it does:

- performs the final local-refinement study around the selected engineered candidate
- evaluates four predefined attempt families and selects the final reported configuration

Attempt families and alpha counts:

- `hardzero_step003`: `22`
- `hardzero_step004`: `17`
- `hardzero_step005`: `14`
- `weak_depr_step003`: `22`

Run count:

- `(22 + 17 + 14 + 22) × 32 = 2400` seed-runs

Command:

```bash
python n3_locked_candidate_phase_transition_finalize.py run \
  --base-script n3_engineered_phase_transition.py \
  --force-repatch \
  --force-rerun
```

Main outputs include:

- `runs/n3_locked_candidate_transition_finalize/FINALIZE_REPORT.txt`
- `runs/n3_locked_candidate_transition_finalize/final_attempt_ranking.csv`
- `runs/n3_locked_candidate_transition_finalize/selected_attempt.json`
- `runs/n3_locked_candidate_transition_finalize/graph_package/*.png`

### `n3_stageE_candidate_interventions.py`

Modes:

- `run`
- `graph`

#### `run`

What it does:

- runs the follow-up intervention analysis on the selected engineered candidate
- compares the `base`, `support`, and `borrow` families on a shared alpha range
- alpha grid length: `59`
- default seed range: `30..61`, which is `32` seeds
- default horizon: `4,000,000` timesteps

Run count:

- `59 × 3 × 32 = 5664` seed-runs

Command:

```bash
python n3_stageE_candidate_interventions.py run \
  --base-script n3_engineered_phase_transition.py \
  --stagee-script stageE_clipped_robustness_extension.py \
  --force-repatch \
  --force-rerun
```

Main outputs include:

- `runs/n3_stageE_candidate_interventions/INTERVENTION_REPORT.txt`
- `runs/n3_stageE_candidate_interventions/alpha_grid.json`
- `runs/n3_stageE_candidate_interventions/family_transition_summary.csv`
- `runs/n3_stageE_candidate_interventions/graph_package/*.png`

#### `graph`

What it does:

- rebuilds the graph package from existing outputs
- no new training

Run count:

- `0` training runs

Command:

```bash
python n3_stageE_candidate_interventions.py graph \
  --base-script n3_engineered_phase_transition.py \
  --stagee-script stageE_clipped_robustness_extension.py
```

### `n3_stageE_borrow_flatline_disambiguation.py`

Modes:

- `run`
- `graph`
- `all`

#### `run`

What it does:

- runs the borrow-family disambiguation stage for the engineered extension
- includes the canonical extended borrow sweep, the promoted certified borrow variant, and four diagnostic probe families
- tests whether the flat initial-window borrow response is a right-shifted onset rather than a plotting failure

Run count:

- canonical extended family: `81` alpha values × `32` seeds = `2592` seed-runs
- promoted certified family: `49` alpha values × `32` seeds = `1568` seed-runs
- four probe families: each `49` alpha values × `12` seeds = `588` seed-runs
- total: `6512` seed-runs

Command:

```bash
python n3_stageE_borrow_flatline_disambiguation.py run \
  --intervention-script n3_stageE_candidate_interventions.py \
  --base-script n3_engineered_phase_transition.py \
  --stagee-script stageE_clipped_robustness_extension.py \
  --force-repatch \
  --force-rerun
```

Main outputs include:

- `runs/n3_stageE_borrow_flatline_disambiguation/BORROW_DISAMBIGUATION_REPORT.txt`
- `runs/n3_stageE_borrow_flatline_disambiguation/borrow_variant_summary.csv`
- `runs/n3_stageE_borrow_flatline_disambiguation/borrow_flatline_questions.json`
- `runs/n3_stageE_borrow_flatline_disambiguation/graph_package/*.png`

#### `graph`

What it does:

- rebuilds the graph package from existing outputs
- no new training

Run count:

- `0` training runs

Command:

```bash
python n3_stageE_borrow_flatline_disambiguation.py graph \
  --intervention-script n3_stageE_candidate_interventions.py \
  --base-script n3_engineered_phase_transition.py \
  --stagee-script stageE_clipped_robustness_extension.py
```

#### `all`

What it does:

- runs the full training inventory from `run`
- then rebuilds the graph package in the same invocation

Run count:

- the same `6512` training seed-runs as `run`
- graph rebuild adds `0` training runs

Command:

```bash
python n3_stageE_borrow_flatline_disambiguation.py all \
  --intervention-script n3_stageE_candidate_interventions.py \
  --base-script n3_engineered_phase_transition.py \
  --stagee-script stageE_clipped_robustness_extension.py \
  --force-repatch \
  --force-rerun
```
