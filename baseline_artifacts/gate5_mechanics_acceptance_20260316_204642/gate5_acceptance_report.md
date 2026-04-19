# Gate 5 re-acceptance report

Root: `/home/legend/jaxmarl_phase_project/JaxMARL_gate5_repair/baseline_artifacts/gate5_reacceptance_20260316_204642`

## quick_gating_off

- stdout.log present: True
- check lines present: True
- mean_update_fraction: 1.0
- eval_done_count_mean: 100.0
- eval_len_end_mean: 26.0
- eval_csr_mean: 0.0
- final_return_end_mean: -26.766146
- final_min_dists_end_mean: 2.13391
- final_CSR_train: 0.008
- final_energy: 1.0
- final_VarTD: 0.169127
- eval_return_end_mean: -23.42099
- eval_dbar_end: 0.474523
- plots: ippo_ff_gate5_CSR_train_MPE_simple_spread_v3.png, ippo_ff_gate5_VarTD_MPE_simple_spread_v3.png, ippo_ff_gate5_energy_MPE_simple_spread_v3.png, ippo_ff_gate5_min_dists_end_MPE_simple_spread_v3.png, ippo_ff_gate5_return_MPE_simple_spread_v3.png, ippo_ff_gate5_update_fraction_MPE_simple_spread_v3.png

## quick_gating_on

- stdout.log present: True
- check lines present: True
- mean_update_fraction: 0.5
- eval_done_count_mean: 100.0
- eval_len_end_mean: 26.0
- eval_csr_mean: 0.0
- final_return_end_mean: -27.22473
- final_min_dists_end_mean: 2.236593
- final_CSR_train: 0.0
- final_energy: 0.015681
- final_VarTD: 0.16883
- eval_return_end_mean: -23.241333
- eval_dbar_end: 0.53238
- plots: ippo_ff_gate5_CSR_train_MPE_simple_spread_v3.png, ippo_ff_gate5_VarTD_MPE_simple_spread_v3.png, ippo_ff_gate5_energy_MPE_simple_spread_v3.png, ippo_ff_gate5_min_dists_end_MPE_simple_spread_v3.png, ippo_ff_gate5_return_MPE_simple_spread_v3.png, ippo_ff_gate5_update_fraction_MPE_simple_spread_v3.png

## full_1e6_gating_off

- stdout.log present: True
- check lines present: True
- mean_update_fraction: 1.0
- eval_done_count_mean: 100.0
- eval_len_end_mean: 26.0
- eval_csr_mean: 0.0
- final_return_end_mean: -23.966433
- final_min_dists_end_mean: 1.782282
- final_CSR_train: 0.0
- final_energy: 1.0
- final_VarTD: 3.551679
- eval_return_end_mean: -23.323811
- eval_dbar_end: 0.573175
- plots: ippo_ff_gate5_CSR_train_MPE_simple_spread_v3.png, ippo_ff_gate5_VarTD_MPE_simple_spread_v3.png, ippo_ff_gate5_energy_MPE_simple_spread_v3.png, ippo_ff_gate5_min_dists_end_MPE_simple_spread_v3.png, ippo_ff_gate5_return_MPE_simple_spread_v3.png, ippo_ff_gate5_update_fraction_MPE_simple_spread_v3.png

## full_1e6_gating_mild

- stdout.log present: True
- check lines present: True
- mean_update_fraction: 0.8013
- eval_done_count_mean: 100.0
- eval_len_end_mean: 26.0
- eval_csr_mean: 0.0
- final_return_end_mean: -23.754669
- final_min_dists_end_mean: 1.777608
- final_CSR_train: 0.0
- final_energy: 0.019996
- final_VarTD: 3.219997
- eval_return_end_mean: -23.59086
- eval_dbar_end: 0.720929
- plots: ippo_ff_gate5_CSR_train_MPE_simple_spread_v3.png, ippo_ff_gate5_VarTD_MPE_simple_spread_v3.png, ippo_ff_gate5_energy_MPE_simple_spread_v3.png, ippo_ff_gate5_min_dists_end_MPE_simple_spread_v3.png, ippo_ff_gate5_return_MPE_simple_spread_v3.png, ippo_ff_gate5_update_fraction_MPE_simple_spread_v3.png

## Verdict

- quick_gating_off_pass: True
- quick_gating_on_pass: True
- full_1e6_gating_off_pass: True
- full_1e6_gating_mild_pass: True
- quick_suite_pass: True
- full_suite_mechanics_pass: True

## Next step

Gate 5 mechanics passed, but full ungated eval CSR is still ~0. Per the locked plan, do NOT proceed to sweeps. Next stage is criterion/eval/training mismatch debugging.
