# Cleanup deletion plan

No file has been deleted. Every action below waits for human confirmation.

Protected records stay in place: deployment criterion registries, Phase-4C/4D/4E reports, data registries, scenario and holdout manifests, integrity audits, B2 fit records, tau / Gate-1 records, deployment verification outputs, and failure attribution outputs. `backup/snapshots` stays untouched because those snapshots are the round history.

| file | reason | risk | action |
| --- | --- | --- | --- |
| `src/**/__pycache__/*.pyc` (906 files) | Python bytecode cache | low | delete after confirmation |
| `round_36/**/__pycache__/*.pyc` (735 files) | Python bytecode cache | low | delete after confirmation |
| `tests/**/__pycache__/*.pyc` (122 files) | Python bytecode cache | low | delete after confirmation |
| `experiments/**/__pycache__/*.pyc` (14 files) | Python bytecode cache | low | delete after confirmation |
| `backup/snapshots/**` | immutable pre-round snapshot, including its bytecode | high | keep |
| `src/wt/lowlevel_v2/**/outputs/HTW_Phase4*.md` (19 files) | duplicate of the root audit report | medium | archive only after confirmation; do not delete |
| `src/wt/lowlevel_v2/outputs/phase0/LLV2_DATA_REGISTRY.json` | data registry | high | keep |
| `src/wt/lowlevel_v2/phase4d_deployment_criterion_registry/deployment_criterion_registry.json` | official criterion | high | keep |
| `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/deployment_holdout_manifest.json` | consumed holdout manifest | high | keep |
| `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/holdout_labels.csv` | holdout ground truth | high | keep |
| `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/holdout_integrity_audit.csv` | holdout integrity audit | high | keep |
| `src/wt/lowlevel_v2/phase4e_deployment_verification/outputs/deployment_verification_results.csv` | deployment verification | high | keep |
| `src/wt/lowlevel_v2/phase4e_deployment_verification/outputs/deployment_failure_attribution.json` | failure attribution | high | keep |
| `src/wt/lowlevel_v2/htw_phase4a_hindsight_credit/outputs/phase4a_hindsight_credit.csv` | B2 label source | high | keep |
| `src/wt/lowlevel_v2/htw_phase4b_b2_shadow/outputs/b2_shadow_decisions.csv` | frozen B2 / tau evidence | high | keep |
| `.test-tmp/**` empty directories | empty test temp dirs | low | delete after confirmation |
| `round_02/t*` empty directories | empty tempfile directories | low | delete after confirmation |
| empty directories inside `backup/snapshots` | snapshot structure | high | keep |

The 1,777 delete candidates are all `.pyc` files outside `backup/`. Their paths are the `DELETE_CANDIDATE` rows in `src/wt/lowlevel_v2/project_cleanup_audit/outputs/cleanup_file_inventory.csv`.

The 19 archive rows are duplicate Phase-4 markdown reports. The root copy of each report stays. Do not delete the module copy until the root copy has been checked.
