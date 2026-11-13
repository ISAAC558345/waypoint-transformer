# 方案二论文稿与测试入口说明

本目录对应研究方向“面向通信任务收益的无人机主动模型辨识与长期调度”。论文沿用仓库当前英文双栏稿的组织形式。作者名沿用原稿 Tiancheng Zheng。

本文是待测试的方法设计稿。正文包含系统模型、未知功耗参数、贝叶斯更新、校准任务选择与停止、解析算例、对照实验和统计方案。没有将既有 payload 增益作为新方向结果，也没有运行新的训练或仿真。

## 文件

- 主论文位于上一级 `Task_Directed_Active_Model_Identification_MultiUAV.pdf`。
- `manuscript.json` 是本次交付的规范正文，包含段落、公式、表格及算法步骤。
- `manuscript.tex` 是由同一正文导出的可编辑 LaTeX 稿。
- `references_verified.bib` 保存参考文献，ASID 使用 ICLR 正式版本，海洋预测波束论文标为预印本。
- `build_paper.py` 从规范正文生成双栏 PDF 与 LaTeX 稿，并检查公式编号、引文和解析算例。
- `verification.json` 保存文稿生成检查，不是环境性能评测。
- `experiment_spec.json` 是首个功耗辨识试验的接口与对照清单，不是已实现的配置入口。

## 与现有分支的关系

本次复用 `research/ocean-observation-isolated`。其已有 `Ocean_Observation_Solution_CN.md` 包含少量参数主动辨识与跨航次阶段。本次新增论文放在独立子目录，原有 M0、M0A、M0B 材料和实验记录作为各自已有证据保存。

论文样式参照默认分支提交 `e58e5d190109e5a0e860cb5c6a412321c0d74e31` 的 `docs/paper/Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV.pdf`，该 PDF 的 Git blob 为 `92ba3fe88ad8a6e99dbf45b655268aa300f802f9`。正文只复用层次决策、实际交付记账与场景配对评价的设计经验。

## 建议先测试的机制

1. 使用 2 架无人机和 4 个静态浮标，再扩展至 6 架和 12 个浮标。
2. 先只开放一个正值功耗比例参数，明确电量单位、风观测和辅助功耗核算。
3. 正常采集航段和专门校准航段使用同一个估计器，均提供同样的遥测。
4. 设置参数不确定性相同、任务决策敏感程度不同的两类场景。主指标为论文公式 (7) 的净任务效用；分别报告最终接收端按时交付、能量成本及二者的权衡，另计验证次数、时间和机会成本。
5. 对比固定模型、被动更新、周期校准、最大不确定性、成本匹配随机验证和相同信息下的滚动优化。
6. 固定算法与统计口径后再运行保留测试。模型失配、跨航次变化和知识共享分别进入后续试验。

当前分支的 M0B 已显示一组收益可由固定接收点解释。新试验需要保留部署匹配的固定接收对照，并核算业务量和上传回传能力。本论文不把 M0B 的派生记录当成主动辨识的独立实验。

## 重建与编辑

推荐用 Python 3.10 或更新版本。排版需要 Windows 的 Times New Roman 和 Cambria Math，或 Linux 的 Liberation Serif 和 DejaVu Serif 字体；脚本将字体嵌入 PDF，确保希腊字母和积分符号正常显示。安装 `reportlab` 和 `pypdf` 后运行：

```powershell
python -m pip install reportlab pypdf
python build_paper.py
```

生成 PDF 位于本目录。如需保持本次仓库布局，可将生成文件复制到上一级。PDF 直接由 ReportLab 排版，LaTeX 源码由同一规范正文导出。本次未使用 TeX 编译器编译导出稿。

若使用 LaTeX，可在安装所需包的 TeX 环境中运行：

```text
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

直接编辑 LaTeX 也可以，但再次运行 `build_paper.py` 会根据 JSON 重新导出。需要同时更新 PDF 与 LaTeX 时，应修改 `manuscript.json`。论文主内容为英文，中文测试说明在本文件。

## 当前交付范围

本次仅新增文稿、参考文献、排版脚本、测试设计和文稿核查记录。环境中的真实能量、传感器和校准动作需要后续单独实现。本稿中的解析数值只对应第五节明确给出的人工两状态问题。
