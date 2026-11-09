# L12：源码工件与历史路径收尾、论文索引对齐

日期：2026-11-09。结果基线：fa845dfda498b8ba18134505a9cf3ab03f61ba6d。维护分支：feature/legacy-system-maintenance。
先读本次Legacy_System_L11_Push_Review_CN.md。

## 1. 范围与零研究消耗

用户已选C：完成旧系统交付，暂不新增模型研究。L11正式链路迁移和一次48步一致性通过，不重复执行。当前问题是有限的目录、资源解析和论文交付缺口。

本轮新增环境步、低层推理、高层模型前向、训练更新、反向全部为0。不打开322040–322047，不生成标签，不执行L8/L9或保留集，不改变正式模型、参数、种子、候选池、奖励或served定义。海洋隔离分支保持不变。

只处理本项目主检出；保留其他用户文件、未提交修改、独立工作树、backup快照、环境及Git元数据。临时项留在项目backup，不再整体重排已经正确的功能源码。

## 2. 把剩余旧权重与成套输出真正移出src

当前Git树src下还有199个.pt，总316,947,845字节。将它们按用途迁入checkpoints/legacy/<语义主题>/，保留原字节，不能加载再保存。源名冲突按主题区分，不覆盖、不以一串新的轮次目录代替。源码中同批outputs里的结果表、日志、统计等非代码工件也按用途归入results或data；实际代码与必要轻量代码资源不要混着搬。

五份planning/task_option/outputs/handoff/deepsets_util_seed0.pt至seed4.pt（每份1,546,165字节）属于旧高层DeepSets-util，不是现行FormalPlanner的SchemeCAdapter/load_primary资源。正式四份checkpoint及其资源ID/路径保持不变；本轮不改其加载链。

只补充这次实际移动对象的源/目标/字节SHA/旧blob到迁移索引。Git追踪工件用可逆移动并继续追踪；本地忽略工件保持原有本地属性。不要改checkpoint内容、原始结果、锁和日志，不强制上传缓存。

同步这些资源在旧逻辑路径和当前源码路径下的解析。根文档中“权重离开src”的完成声明，只有实际扫描确认无实验checkpoint留在src之后才能保留。

## 3. 修复旧逻辑路径到最终物理位置

现resolve_logical只替换一次，而resources里包含“原路径→中间路径”和“中间路径→最终路径”，因此部分历史记录失效。

基于实际迁移索引生成/修正可闭合的映射。可以优先直接登记old_path→final_path；若组合前缀改名，必须得到最终路径，处理循环/冲突并保持重复解析结果稳定。不能只对一个示例写补丁、留下同类嵌套改名未处理。也不能改写历史JSON的旧路径来掩盖resolver缺口。

至少修正：
- round_13/amendments/round13_completion_amendment_20260825_0010 → archive/experiments/service_action_controllability/amendments/completion_amendment_20260825_0010；
- round_11/AGENTS.md → archive/experiments/task_waypoint_predictor/historical_instructions.md，及同类历史指令文件改名；
- 本轮199权重及其原模块/中间模块路径；
- 下列两份摘要的逻辑定位。

原summary逻辑名：
~~~text
results/phase6iab_planner_stability/summary.json
results/phase6iac_independent_holdout/summary.json
~~~
实际已存在的位置：
~~~text
results/reproducibility/formal_scheme_c/evidence_archive/results/phase6iab_planner_stability/summary.json
results/reproducibility/formal_scheme_c/evidence_archive/results/phase6iac_independent_holdout/summary.json
~~~

AB原blob=73e01163aafb5edc771cd7f4aee6139fdd775b11，AC=e23290d36258acc7d417a79d21c07a5502d95702。它们不是缺失证据，不重新生成、不复制一套替代结果。benchmark入口继续读取这些原件。

## 4. 只做必要的无模型验证

- 一次检查本轮移动工件的目标存在且字节哈希/原Git对象不变；src不再留实验.pt/.pth等checkpoint。不要反复扫描无关历史库。
- 在已知实际存在的迁移对象上验证旧逻辑路径可直接到最终文件，特别是嵌套改名、historical_instructions、两份摘要和本轮权重。不能把源本就缺失的历史文件当作本轮失败后强行恢复或重新生成。
- 运行一次scripts/benchmark.py，只读两个归档summary。应打印真实paired均值/CI及正平负数，而非absent。将实际输出和原件路径/blob记录在results/reproducibility/delivery_closeout_20261109/。
- AB：16场景，mean 2644990.8087698035 bit，CI [180100.0770928479,5016616.160644779]，11/0/5；
  AC：16场景，mean 3502191.5265653464 bit，CI [1270476.686997788,5395098.500613804]，14/0/2。
  只读取已算好的统计，不重新bootstrap、不重跑episodes。
- 核对正式4份资源绑定、路径和哈希未变；两个登记场景预览正确且不加载torch。不要为本轮不涉及的正式模型再跑一次48步。

不运行历史全量测试、安全/压力/故障注入矩阵，也不逐个导入可能执行旧实验的历史模块。

## 5. 更正来源与缺失说明

README.md、docs/USAGE.md、docs/DELIVERY_STATUS_CN.md、当前修订说明和evidence_20261109.json要明确AB/AC摘要已经保存在归档，旧脚本误报来自路径未解析。记录真实原件路径和blob/字节SHA，不写为新恢复/新统计收益。

ranking-loss顶层summary、基线侧分状态等真正缺失的证据继续如实标注，不把它们与AB/AC混淆。原L11报告可保留原文并附更正说明；不要把其错误“缺失”描述继续当当前状态。

保留L11那次48步原始manifest/episode/steps。该manifest把instruction_commit记为204dd3309115dfd369236c408f7e59306a0871d4，实际迁移指令是b55981aad49e58366870ed3391217af427786c90。新增派生provenance_correction.json，指向原manifest及其哈希，说明错误字段、正确来源及“只改来源解释，结果未重跑”。本轮没有新episode，不能假造新的运行manifest。交付索引引用更正说明。

## 6. 论文小幅补齐定义并更新最新入口

保持2026-11-09为当前可编辑稿和同名PDF，不再制造一套v2文件夹。2026-11-08及更早正文/PDF保持原字节。

同时修改11月9日Markdown与当前TeX：
- Sec.VII开头“两场景四窗口”明确只限定L3–L5，不概括后来L8/L9。
- 定义L8的R为R12：从E6初始化、冻结matching decoder、在322016–322021×t15/t30的12状态上用真实到达标签训练120更新，使用温度1的期望归一化regret。
- 简要给出L=sum softmax(score)_i * ((max y-y_i)/max(max y-min y,1e-6))，标签脱离梯度；这是软选择代理，不等于直接优化argmax。
- 定义L9的R28：同E6/损失/120更新，训练数据为原12状态加L8的16比较状态，共28；固定4完整循环+前8状态。L8比较322024–322031，L9比较322032–322039，各t15/t30；后者仍历史TRAIN，不是独立泛化。L8数据转训练后不能再当L9评估证据。
- ListNet参考仍是L3B10，不是正式Scheme C；L8/L9均NOT_PROMOTED，数值和门槛不改。把结论仍写“证据仅推进至E18”的旧范围句改成与当前内容一致。
- 只补必要段落，不新增实验、文献搜索或新模型方向。

docs/paper/README.md及latex/README_CN.md改为真实最新11月9日稿、PDF、证据和本次可执行构建命令，保留旧版链接并明确历史身份。

先恢复11月8日TeX原Git字节归档（源提交8fc6fda3d50d51b414cb609b7a8953951788eb18）：
~~~text
docs/paper/latex/archive/main_20261108.tex -> blob 215135796397a26bf5c29bf55f3271897b66a531
docs/paper/latex/archive/body_20261108.tex -> blob 65f4ff71c93d1085a2438c9cf0e2edeef6bb4937
~~~
用原对象/二进制写出，避免PowerShell文本重编码或Git过滤再次归一化；必要时只对这些档案路径设置准确的-text规则。不要全仓normalize。旧档案身份恢复不需要编译，当前正文变化才需要编译。

复用已有Tectonic与缓存，进行一次必要构建；只在实际编译或排版问题发生时修正重编译。临时产物放backup/local_build/delivery_closeout_20261109，旧构建记录不覆盖。记录真实退出码、页数，逐页查看最终PDF；不强制7页。提交一份适量的本次构建日志，使GitHub可核对，不只引用被忽略的backup日志。

source_manifest_20261109.json必须指向最终当前正文/TeX/PDF/证据的真实字节和实际Git对象。当前main/body旧git_blob是LF规范化身份，必要时保留为normalized_lf_git_blob，不能再与实际提交对象混称。git_blob应从将提交的实际tree/index取得，并在推送后读取Git树核对；workspace_sha256单独记录编译时字节。不要只用会受过滤影响的git hash-object结果冒充远端对象。

## 7. 用户追加：删除无用文件，和L12一起完成

2026-11-09用户明确追加“顺便将没用的文件都删掉”。这授权清理本项目主检出中的无用缓存、已失效维护文件和可无损合并的冗余副本，不要求再为已确认的常规清理逐项请示。不能据此把独有历史证据、未采用模型或旧文件全部当垃圾删除。

### 已在GitHub删除的6个明确无用文件

本次协调提交已删除Matplotlib fontlist缓存及5个失效的一次性旧清理/拆分脚本，具体路径、原blob和理由在：
results/reproducibility/delivery_closeout_20261109/cleanup_manifest.json。

这份清单只记录已发生的Git追踪文件删除，没有声称本地缓存或权重去重已经完成。执行L12时同步这些删除；若本地自动重建字体缓存，可清除这份已确认缓存的变化，但不能reset其他未提交文件。不要再运行已移除的旧全仓清理脚本。

### 本地可再生文件

在最终PDF、来源清单、该留的构建日志均已保存后，直接删除当前主检出的__pycache__、pyc/pyo、pytest/mypy/ruff缓存、Matplotlib字体缓存、明确的临时探针与下载残片，以及本轮已无用途的TeX aux/fls/fdb_latexmk/xdv等构建中间文件。只处理已确认由程序生成的缓存/临时文件，不按全仓*.log、*.json、*.out或“空文件”规则删除。

保留需要下次修改论文使用的Tectonic及TeX缓存、现有环境，保留原始/被引用的构建日志、实际标签/效用数据缓存。ull_cache.json、utility_cache.json等可能是科学数据，不能按cache字样删。归档的5份空日志也是历史记录，不删。

每次递归删除前确认解析后的绝对目标在本项目主检出范围内，跳过.git、backup快照、baseline_worktrees、虚拟环境、重解析点和其他项目。用同一PowerShell或Python文件操作流程，不拼接跨shell删除命令。清理已确认项无需新测试矩阵，也不要把删除改成堆一个新的垃圾备份目录。

### 11份完全相同的权重副本无损去重

当前Git树中已确认10组.pt内容完全相同，合计11份冗余副本，重复字节14,419,249。下面列原Git内容对象、组内副本数、单份字节数：

| 原Git blob | 副本数 | 单份字节 |
| --- | ---: | ---: |
| 52f49635bf35449793b223208edda58581389dbf | 2 | 3141751 |
| f25d7a8d8495eb3e3e89fc7de78be0e8091c55c7 | 3 | 1546165 |
| 2e27ffa235c4bcf16052a50a172f317f91cbd326 | 2 | 1546165 |
| 9caf4bd6c4bacbad44e8ca04a4d7dd7ca2b90911 | 2 | 1546165 |
| 6ebf90a9413872eed2dea95740a250292caec996 | 2 | 1546165 |
| 3af2e138465bcba875ebd77565acb6352433685d | 2 | 1546165 |
| e80b9bc40076a350b46f1a73ed06a560b4d16432 | 2 | 600409 |
| 0a65a9bc984fefdd93fa3f33174c461741598ede | 2 | 600409 |
| a142b7fa3cf835442ed258fdee264e255b6255b5 | 2 | 599709 |
| b353c13b65afc018c5d4254b86100d180b2d8023 | 2 | 199981 |

结合本轮199份源码权重迁移，每组在checkpoints下保留一个语义位置的原字节文件。先建立所有旧逻辑路径到保留原件的精确映射，并核对现有代码/配置访问仍能正确定位，再真正删除多余物理副本。不留旧目录链接、不为每个旧路径复制一份回来。

同一份权重的多种历史用途和来源都保存在迁移/资源索引中，不能因去重丢掉角色记录。199是迁移前逻辑文件数；去重后物理文件数可以减少，报告区分原逻辑覆盖数、保留唯一内容数和删除副本数。不能把这个清理解释为删除所有未采用研究权重。

其他重复导出仅在原字节相同、有明确保留原件且引用已更新时删除。独有数据/权重、正式4份资源、来源索引、锁、历史论文/原始实验结果保留；来源不明的文件先保持原位或已有语义归档，继续完成明确清理项，不为猜测用途停下整轮工作。

### 实际清理记录

在现有cleanup_manifest.json中追加实际删除的本地缓存/中间文件、去重删除路径、保留原件、理由和字节数；保留已提交的6项删除记录。L12报告列实际删除数、去重后保留数和当前工作树减少的字节。不要把候选数当成已删除数，也不声称普通文件删除会清除Git历史对象；不重写Git历史。

本项训练、环境步、模型前向均为0，不引入新研究。

## 8. 最终交付与停止

提交实际权重/结果移动、闭合映射与只读摘要修复、来源更正、当前文档和论文/PDF/构建证据，以及docs/research/Legacy_System_L12_Report_CN.md。

报告给出：
- 移出src的工件数量/字节与哈希核对结果；
- 嵌套路径/旧指令文件/两份summary的实际解析位置；
- benchmark真实只读输出；
- 旧48步来源更正记录，及本轮0环境步、0低层推理、0模型前向、0更新、0反向；
- 新PDF真实页数/构建日志及源产物对应；
- 实际缓存/失效文件删除和权重副本去重清单、保留原件及释放的工作树字节；
- 当前剩余的真实限制，不再误报已存在摘要缺失。

推送维护分支的实际代码、该追踪的移动工件、结果与论文，核对远端路径和manifest对象，不force push，不推送本地缓存。正式分支e58e5d190109e5a0e860cb5c6a412321c0d74e31和海洋隔离分支4d8622ef183e734f2d9a890d60d735ed716f5d74保持不变。

完成这些已知交付项后报告并停止。不能为了“下一步”自动增加一轮检查、训练、候选协议或模型研究。
