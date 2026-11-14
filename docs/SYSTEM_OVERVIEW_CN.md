# 当前系统说明

## 正式处理链

当前系统运行在位队列多无人机通信环境中。高层通过G1_RATE为每架无人机保留k=6候选，枚举合法部分分配，再由Scheme C Epoch 1 HGNN与matching decoder评分。选定任务保持15步；冻结MAPPO-V2产生动作，SafetyProjector执行投影。48步回合在0/15/30/45进行高层决策。

高层结构和matching decoder的实际实现位于src/wt/models；候选与FormalPlanner位于planning；低层执行和投影位于control；环境位于envs；回合与参考比较位于evaluation。学习标签不是运行时输入。浮标间不提供物理通信或队列转移，图边用于计算。

## 数据与服务口径

payload只累计实际served bits。生成、入队、丢弃、排队和数值修正不能算成服务。FIFO及环境动力学保持原实现。正式模型、参数和资源字节不因目录整理改变。

## 当前配置与权重

configs/formal_scheme_c_policy.json：scheme_c、G1_RATE、k6、hold15。checkpoints/formal存放Scheme C、MAPPO seed0和matching decoder；checkpoints/initialization保存E6。configs/resources.json提供统一资源ID，common.paths从项目标识定位根，不依赖工作目录或固定parents层数。

## 已有证据

历史验证16场景的配对均值为2644990.8087698035 bit，区间[180100.0770928479,5016616.160644779]；历史保留集16场景为3502191.5265653464 bit，区间[1270476.686997788,5395098.500613804]。原summary在results/reproducibility/formal_scheme_c/evidence_archive中，scripts/benchmark.py只读这些记录，不重新运行或重估统计。

L8与L9均未满足预注册采用门槛，维持NOT_PROMOTED。这里的研究ListNet对照是L3B10，不能与正式Scheme C Epoch 1混淆。历史训练池内的128候选、15步离线比较，不是新的正式48步收益或独立泛化证据。

## 运行与交付

当前公共入口为python -m wt.cli.evaluate，仅登记322000/322001；默认预览，--run才实际执行。已有322001的48步回归与R1参考一致。命令详见USAGE.md。

根目录按配置、源码、权重、结果、脚本、测试和文档划分。旧轮次中的配置、锁、说明和结果已按功能分开；历史源记录仍保留旧路径字符串，通过docs/development中的迁移索引和resolver定位。旧原始材料不重写。

按用户选择，当前仅完成旧系统整理，不新增模型研究、不生成标签、不打开新比较场景，也不合并海洋观测方案。不存在自动启动的新训练路线。
