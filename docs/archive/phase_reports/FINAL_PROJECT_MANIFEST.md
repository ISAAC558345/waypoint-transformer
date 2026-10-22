# 最终工程目录清单

## A. 最终模型

- `src/wt/final/pipeline.py`
- `src/wt/final/run_final_hier.py`
- 复用：`src/wt/htw_phase100/high_level.py`、`src/wt/htw_phase943/order_free_decoder.py`、`src/wt/htw_phase103/gate.py`、`src/wt/l3b/`

## B. 最终配置

- `final_artifacts/configs/environment.json`
- `final_artifacts/configs/model.json`
- `final_artifacts/configs/decoder.json`
- `final_artifacts/configs/executor.json`
- `final_artifacts/configs/evaluation.json`

## C. 最终结果

- `final_artifacts/results/final_results_table.csv`
- `final_artifacts/results/final_pairwise_statistics.csv`
- 源：`src/wt/htw_phase106/outputs/untouched/`（只读）

## D. 论文图

- `scripts/plot_final_results.py` → `final_artifacts/figures/`
- Phase-10.6 原始图仍在 `src/wt/htw_phase106/outputs/untouched/`

## E. 历史实验

- `src/wt/htw_phase*`、`src/wt/round*`、`src/wt/htw_phase61`–`64`、HGNN/PPO/TW 代码与报告
- 根目录 `HTW_Phase*.md`

## F. Checkpoint

- `final_artifacts/checkpoints/deepsets_util_seed0.pt`
- 源归档：`src/wt/htw_phase100/outputs/handoff/frozen_deepsets_util/deepsets_util_seed0.pt`

## G. 测试

- `tests/test_htw_phase11.py`（工程测试，禁止 untouched）
- 历史：`tests/test_htw_phase10*.py` 等

## H. 复现入口

- `python -m src.wt.final.run_final_hier`
- `python -m wt.final.pack_artifacts`
- `python scripts/plot_final_results.py`
- `FINAL_README_CN.md`
