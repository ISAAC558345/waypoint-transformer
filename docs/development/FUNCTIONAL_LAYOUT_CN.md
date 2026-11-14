# 功能目录与历史文件定位

2026-11-14从维护提交7e4428f整理并发布至默认分支。原round目录中的文件不再整包放在另一处轮次目录：配置进configs/archive，旧权重进checkpoints/legacy，结果进results/archive，历史说明进docs/archive，锁与来源进docs/provenance。上游脚本进scripts/legacy/upstream。

```text
configs/                    当前配置与历史配置
src/wt/
  envs/                     通信与位队列环境
  models/                   HGNN、matching decoder及低层网络
  planning/                 候选生成、正式规划器、任务接口
  control/                  冻结MAPPO执行与安全投影
  evaluation/               回合运行、统计与参考比较
  algorithms/               排序损失及已有算法公共函数
  data/                     构图、快照和数据加载代码
  common/                   项目根与资源路径解析
  cli/                      当前命令入口
  research/                 已完成的研究实现
  legacy/                   历史实现
data/                       输入数据与数据包说明
checkpoints/                正式、初始化及历史权重
results/                    已有基准、复现记录和研究输出
scripts/                    当前命令及历史脚本
tests/                      功能测试与历史测试
docs/                       使用、系统说明、论文和历史来源
```

functional_layout_20261114.json逐文件记录移动、原Git blob与原位置。legacy_paths_20261114.json给出旧逻辑文件到最终位置的映射，resources.py支持组合前缀改名。历史原JSON/日志保留旧路径文字，不做全局改写。拆分后的原目录可能对应多种功能目录，应按文件索引定位。

权重去重只合并原字节完全相同的文件，保存全部旧角色映射；正式4份checkpoint及正式策略保持原位置与内容。数据和历史结果不作为“未采用”而删除。
