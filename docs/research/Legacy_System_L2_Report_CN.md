# 旧系统 L2 报告

父提交是 `c4b0752fb7796e2770357a29376bda16e9d55e26`。本报告只记录本包实际完成的旧系统准备。环境步、训练更新和模型前向都是 0。没有合并正式分支或 `research/ocean-observation-isolated`。

## 论文身份

- 当前正文另存为 `docs/paper/Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261104.md`。SHA-256 是 `d7987d35be30c6e619b9e977603e147ffb35eeb42fc40c102619f12232a27792`。声明仍是 2026-11-04、证据覆盖完成的 E19，并且这次没有编译。
- `Revised_20261102.md` 的字节与提交 `b30200cf8fd9fda0c5dece7bceb52edaa918f9bc` 的同名文件一致，Git blob 是 `7708c858911169d9d8cfbc5cb9875df4a81b9f91`。`.gitattributes` 对该文件关闭换行转换，避免提交时改掉这个 blob。
- 旧 `docs/paper/latex/source_manifest.json` 归档到 `docs/paper/latex/archive/source_manifest_20261102.json`。新清单重新计算了当前 Markdown 和实际引用的 TeX。括号与环境检查通过。上一次结构检查没有转入新稿。
- 两份旧 PDF 保持原字节。它们是先前独立排版，不是本 LaTeX 工程已核验的编译结果。
- 一次可用性检查没有找到 pdflatex、xelatex、lualatex、latexmk、tectonic，也没有找到 TeX Live 或 MiKTeX 的常见目录。编译、分页和引用解析仍未完成。
- ranking-loss 的 `summary.json` 仍然缺失。E19 仍是 T 与 M 选择相同且没有采用。正式重放 1488 步，另有 30 步作废计数，合计 1518。

## 数据准备

准备命令：

```text
py -3 -m wt.legacy_arrival_aware_ab.prepare --inventory docs/research/Legacy_ArrivalAware_AB_Source_Inventory_20261104.json --labels results/phase6i_urgency_arrival_aware_expansion/labels.csv --candidates results/phase6i_urgency_arrival_aware_expansion/selected_candidates.csv --source-root results/legacy_arrival_aware_ab_l2/source_states --output results/legacy_arrival_aware_ab_l2
```

- 从提交 `8a34efc718665a7cfe4375b662e8f5249263f03d` 恢复 32 个原件，共 100230 字节。每个文件的 Git blob、字节数和 SHA-256 记在 `results/legacy_arrival_aware_ab_l2/source_files.json`。
- 16 个修订 NPZ 的文件 SHA-256 等于各自 manifest 的 `checksum_state`。D6 的 `state_hash` 等于 manifest 的 `source_state_hash`。这两个哈希没有互相替代。`service_progress` 和 `uav_step_displacement_m` 仍是原件里的有限数组。
- 配对表有 2048 行。每行同时有 `zero_payload_bits` 和 `real_payload_bits`，所以标签值是 4096，不是 4096 个独立样本。连接键是场景、时刻、候选编号和 assignment。
- train 是 322016–322021，12 个状态、1536 个候选。explore-eval 是 322022–322023，4 个状态、512 个候选。同一场景的 t=15 和 t=30 在同一组。这组场景属于既有训练池，不是新的独立验证或正式 holdout。
- 图转换通过。16 个图都是既有七类张量；队列边追加后的宽度是 4。payload、名次和 scenario_id 不在图文件里。两臂共用同一图文件和同一行顺序。
- 图转换调用既有 `hgnn_tensors`。该函数读取 `arrivals_at`，没有调用环境 `step`，没有加载 checkpoint，也没有做网络前向。
- 结束后又按输出重读了 32 个身份、16 个状态、2048 个候选、标签与输入分离、分组和两臂对应。该项检查通过。

## 协议

`docs/research/Legacy_ArrivalAware_HighLevel_AB_Protocol_CN.md` 已把已知的标签重排和模型能否学到这一改进分开。本对照只换纯 payload 标签。`lambda_H` 与 `delta_payload_max` 继续空着。损失仍是 `top_regret_loss`。epoch 6 文件存在，SHA-256 是 `7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde`。解码器文件存在，SHA-256 是 `de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5`。既有编码器基础文件 `results/phase5l_joint_encoder/joint_utility_encoder.pt` 不在当前工作区。训练草稿没有执行。

## 遗留

- 当前 LaTeX 工程尚未编译，页面和引用解析没有核对。
- `results/phase6if_ranking_loss/summary.json` 仍然缺失。
- 编码器基础权重文件缺失，既有训练命令因此不能启动。
- 10 个 epoch、一个种子 240 次更新、三个种子 720 次更新的训练没有开始。
- `lambda_H` 与 `delta_payload_max` 仍未设定。
- 本包没有新的科学结论，也没有重跑 R1、R1A、原 192 步、D5、D6、E19、旧验证、holdout 或历史全量测试。
