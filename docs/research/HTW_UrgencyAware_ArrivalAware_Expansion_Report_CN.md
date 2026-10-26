# Stage D6 报告：扩展 Arrival-aware 高层候选标签验证

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`327eb05376b2e1f62581e09526e5a9b0bd782fe2`
- 前置 Gate：`ARRIVAL_AWARE_URGENCY_LABEL_COMPLETE`，本阶段没有修改
- 本阶段 Gate：`ARRIVAL_AWARE_LABEL_EXPANSION_INCONCLUSIVE`
- 性质：训练池内的离线双标签验证。`learned_policy = false`，没有训练 HGNN 或 MAPPO

## 1. 直接结论

32 个状态都跑完，完整性通过，预注册复核也通过。到达对排序的影响随决策时刻分开：

- t=0 的 15 步窗口里，8 个场景的零到达标签与真实到达标签数值相同。36,000,000 bit 新到达没有进入服务量，全部留在终态队列或被丢弃。
- t=15 与 t=30 的 15 步窗口里，8 个场景都达到预注册的 material 标准。真实到达改变 payload 排序，也改变紧急服务的冠军或成对顺序。
- t=45 只剩 3 步，正式 `arrivals_at` 在这个窗口的生成量是 0。两臂标签相同。这组结果单独列出，不与 15 步标签平均。

因此本阶段不把四个时刻收成一个高层训练目标，也不训练。t=15 与 t=30 的排序变化可以写入下一份决策文件分开讨论。`lambda_H` 与 `delta_payload_max` 仍未定，payload 损失预算仍要研究负责人明确。8 个场景只作扩展验证，不宣称显著性或泛化。

## 2. 规模与耗时

场景来自 `phase6ib_highlevel_dataset` 的 TRAIN_POOL registry，按 ID 排序，去掉 D0/D1/D5 已用的 322000–322015 后取前 8 个，并逐个通过 `assert_allowed(split="train")`：322016–322023。registry 与预期一致。没有读取 validation 或 reserved test。

状态沿既有 `MAX_RATE_INDEPENDENT` 路径，由冻结 MAPPO-V2 seed 0、hold=15 重放。t=0、15、30 与冻结 shard 的状态哈希一致。t=45 没有 15 步 shard，协议身份就是这次重放。

每个状态 128 个候选，种子 605，`SeedSequence([605, scenario_id, time_step])`。t=0/15/30 用冻结 J15 的稳定前 32 名，再按 `utility_rank` 抽 96 个。t=45 用当前状态合法 G1_RATE k=6 池的加性速率做同样的 32+96 分层；该分数只用于抽样。候选列表在对应 rollout 之前写入 `selected_candidates.csv`。两臂使用同一批候选 ID 和 assignment。全池每个状态 13,327 个合法候选。

第一个状态是 322016 的 t=0：重放 1.080 秒，128 个候选的双标签 21.451 秒，计入重放后 22.531 秒。按“每个状态都按这次 15 步打分计费，每个场景一次重放，再加一次复核”估计总时间 717.6 秒，低于 20 分钟，也低于单状态 120 秒。随后 32 个状态都完成。日志里最长的单状态打分是 322020 的 t=15，27.427 秒。

研究 rollout 为 8 × 4 × 128 × 2 = 8,192。预注册复核是 322016 的 t=30，同一 128 个候选再打分一次，256 次 rollout。合计执行 8,448 次，没有增加场景、候选或 horizon。研究阶段墙钟 623.4 秒，复核 25.86 秒，合计 649.3 秒。

## 3. 完整性

4,096 行候选与 4,096 行标签一一对应。

- t=0/15/30 的零到达 payload 与冻结 `j15_bits` 的最大绝对误差是 0。t=45 的 3 步零到达结果没有当作旧 J15。
- 同一状态内，真实外生到达量在 128 个候选之间的差不超过 1 bit，并等于该窗口上 `arrivals_at(t)` 的总和：t=0 为 36,000,000 bit，t=15 为 42,000,000 bit，t=30 为 24,000,000 bit，t=45 为 0。
- 到达差额拆成服务、丢弃、终态队列和 float snap 之后，残差在每浮标容差乘以浮标数的范围内。实测最大残差约 1.49e-8 bit。
- ledger 与 `q_bits`、payload 与 served、队列守恒的逐步误差都在 1 bit 内。
- 候选来自合法 G1_RATE k=6 池。t=0/15/30 的池顺序与冻结 shard 一致。
- 第一次复核在打分前停止：抽样行当时没有写入 horizon，异常发生在第一步 rollout 之前，研究标签没有重算。补上该字段后，同一状态、同一子集哈希 `573242a298546ca190149fb465a67b948000c4ca7fbfa52f11bf993002124a25` 再打分，最大标签差 3.73e-9 bit。
- D0–D5 结果、正式 `planner.py` 和这 8 个场景的数据集文件在运行前后哈希一致。冻结 checkpoint 与既有登记值一致。没有修改 Planner、奖励、动作、checkpoint 或旧数据集。

## 4. 分时刻排序

Top-1 取标签最大者，1 bit 内并列时取较小候选 ID。反转只统计两臂差距都超过 1 bit 且顺序相反的候选对。Kendall 使用同一批可比候选对：`(一致对数 - 相反对数) / 可比对数`。下面的均值是 8 个场景的描述统计，不是显著性检验。15 步与 3 步没有合并。

### t=0，horizon 15，效果 small

8/8 场景的 payload Spearman、紧急服务 Spearman 和两套 Kendall 都是 1，反转比例是 0，payload 与紧急服务的 top-1 候选 ID 都不变。真实到达相对零到达的 payload 增量均值和紧急服务增量均值都是 0。

322016 的例子可以代表这一组。Payload top-1 在两臂都是候选 4644，assignment `0|1|2|3|5|4`，payload 15,000,000.00 bit。紧急服务 top-1 在两臂都是候选 4894，assignment `0|2|3|4|5|1`，紧急服务 1,843,198.42 bit，payload 同样是 15,000,000 bit。冠军都落在开局浮标 {0,…,5} 的匹配上。紧急服务冠军位于 payload 的 1 bit 并列里。

这 128 个候选的服务量没有吃到新到达。36,000,000 bit 里，24,000,000 bit 进入终态队列，12,000,000 bit 被丢弃。8 个场景的这四个数相同。这个子集不能代表全池；它只说明被抽到的 128 个候选在开局 15 步里没有把新到达变成交付。

### t=15，horizon 15，效果 material_consistent

8/8 为 material。Payload Spearman 均值 0.346，反转比例均值 0.353，Kendall 约 0.295。紧急服务 Spearman 均值 0.892，反转比例均值 0.141，Kendall 约 0.718。真实到达标签上，payload 与紧急服务的 Spearman 均值 0.400。

Payload top-1 在 8/8 场景改变。真实 payload 冠军相对零到达冠军的真实 payload 增益均值 1,863,104 bit，范围 332,350 bit（322022）到 2,890,764 bit（322016）。紧急服务 top-1 在 7/8 场景改变。322019 的紧急服务冠军仍是候选 11153，增益 0；该场景的紧急服务反转比例仍有 0.134，所以按预注册规则仍是 material。其余场景的紧急服务增益最高 429,847 bit（322018）。均值 119,294 bit。

候选平均多交付 4,054,933 bit，紧急服务平均多 255,142 bit。322023 的紧急服务平均增量是 -131,718 bit：真实到达可以提高冠军，同时降低子集的平均紧急服务。丢弃平均多 23,772,411 bit，终态队列平均多 14,172,656 bit。三者与 42,000,000 bit 的生成量对齐。

### t=30，horizon 15，效果 material_consistent

8/8 为 material，方向与 D5 在 322000–322003 上看到的一致，场景集合没有重叠。Payload Spearman 均值 0.375，反转比例均值 0.360，Kendall 约 0.281。紧急服务 Spearman 均值 0.906，反转比例均值 0.121，Kendall 约 0.759。两目标 Spearman 均值 0.528。

Payload top-1 与紧急服务 top-1 都在 8/8 场景改变。Payload 增益均值 4,178,634 bit，范围 3,019,790 bit（322021）到 5,361,436 bit（322022）。紧急服务增益均值 1,023,747 bit，范围 503,318 bit（322023）到 2,069,271 bit（322022）。

候选平均多交付 7,542,190 bit，紧急服务平均多 1,260,543 bit，丢弃平均多 11,098,596 bit，终态队列平均多 5,359,215 bit。四项对应 24,000,000 bit 新到达。

322016 的零到达 payload 冠军 4919（`6|8|10|7|9|11`）在真实到达下交付 25,277,676 bit；真实 payload 冠军 10012（`9|10|11|7|-1|-1`）交付 30,377,321 bit。零到达紧急服务冠军 8166 的真实紧急服务是 8,026,702 bit，低于它自己的零到达紧急服务 8,674,189 bit。真实紧急服务冠军 9328 达到 9,093,957 bit。

### t=45，horizon 3，两臂一致

8/8 的生成量是 0，payload 增量、紧急服务增量、丢弃增量和队列增量都是 0。Payload 与紧急服务的 Spearman 为 1，反转比例为 0，top-1 候选不变。3 步零到达标签只作为诊断，不称为旧 J15。窗口内的 payload–紧急服务相关均值 0.941，描述的是同一套标签内部的两个目标，不是到达造成的重排。

## 5. Payload 与紧急服务的权衡

真实到达标签上的 Pareto 前沿用 1 bit 容差。一个候选被保留，表示子集里没有另一个候选在两个坐标上都不更差、且至少一个坐标严格更好。

- t=0：8 个场景的前沿大小都是 1。开局 payload 在冠军处顶到 15,000,000 bit，紧急服务冠军就在这个 1 bit 并列里。到达没有新制造 payload 损失。
- t=15：前沿大小 1 到 7。322022 的真实 payload 冠军和真实紧急服务冠军是同一个候选 9914。322016 则分开：payload 冠军 1762 交付 29,390,870 bit、紧急服务 6,080,071 bit；紧急服务冠军 11071 交付 28,412,592 bit、紧急服务 8,033,053 bit。大约 978,278 bit 的 payload 与 1,952,982 bit 的紧急服务分列两端。
- t=30：前沿大小 2 到 5。322016 的 payload 冠军与紧急服务冠军相差约 4,004,317 bit payload 和 1,803,454 bit 紧急服务。
- t=45：前沿大小 1 或 2，来自同一套无到达标签。

两目标的排序相关在 t=15 只有约 0.40，在 t=30 约 0.53。子集里存在需要 payload 损失才能提高紧急服务的候选。本阶段不把这个交换折成一个 `lambda_H`，也不设置 `delta_payload_max`。

分层只作描述。t=30 上，idle=0 的紧急服务反转比例均值约 0.585，idle 升高后反转比例下降，payload 增量在 idle=3 附近达到约 10.9 Mbit。t=15 的 idle 切片没有同一形状。这些切片的候选数不同，idle=5 只出现在少数场景，不能当成原因。

## 6. 覆盖限制

- 场景是训练池中尚未进入 D0/D1/D5 的前 8 个，不是全部 322000–322047，也不是 dev 或 reserved。
- 每个状态只评了 128/13,327 个候选。t=0 的“没有服务到新到达”限于这个子集。
- 状态来自数据集重放，不是 Scheme C 的闭环轨迹。
- t=45 的 3 步标签与 15 步标签尺度不同，汇总时保持分开。
- 统计单位是场景。候选对只描述排序。
- 低层策略、账本、环境和 checkpoint 都是冻结的。标签是离线诊断，没有写回观测。

## 7. 哈希

冻结文件：

- MAPPO-V2 seed 0：`92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`
- Scheme C：`f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864`
- epoch 6：`7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde`
- adapter config：`3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63`
- 冻结 shard 提交：`aadbb3c5c5954669dd335dce9afb3462d7de114c`

D0–D5 产物和 `planner.py` 的运行前后哈希在 `summary.json` 的 `watched_sha256`。

| 场景 | t | horizon | 效果 | state_hash | pool_hash | subset_hash |
| --- | --- | --- | --- | --- | --- | --- |
| 322016 | 0 | 15 | small | `1f66feeaead91bd533ed3ca96ee8510b162b51f2fba632be51ff282ba7f6fa41` | `74b698f3aa54c3e21f168a793fe9ebe81788b0f35c1a8c6e39f7650339f22b48` | `efefa4acbe765c9ef0e2087c27a38cdec78cdb4ca183eeb3df1c446299a5ef72` |
| 322016 | 15 | 15 | material | `05a6fd071f0a9bdfa277d7141460b5d455fe1dfccede30baceb3b9f1f0788f01` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `7e1b8945755204f2dae438b768104285b0d9b960648c45a08767359150379574` |
| 322016 | 30 | 15 | material | `63999185f9df6e67871aeabb8e4e0a46645514c89fa5a9cb959f2cd2d45d2e37` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `573242a298546ca190149fb465a67b948000c4ca7fbfa52f11bf993002124a25` |
| 322016 | 45 | 3 | arms_match | `dfbd9aaa47feb4e4da565d479140bc64cfec46667f2f38417a9e9b585da82c05` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `480e6ca56eeb3f6735253d1b9472ddd1186c2c0f3687a8f5d608afe37db40bfd` |
| 322017 | 0 | 15 | small | `be4fc36e04ae6d1c9a6b5e1d5fe018ee3c9ca1b8fd4a08baac796556eda7994b` | `74b698f3aa54c3e21f168a793fe9ebe81788b0f35c1a8c6e39f7650339f22b48` | `ea36b4190dfc6627162561c9176defde4447257aca64191c424b02cbed2e6ab3` |
| 322017 | 15 | 15 | material | `4921637504c47e340fb60f21df44e171fbef62e15d9cb4a1232b7b6b2b0df8db` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `5ca1b198e2c15ebc8d1db9cc9e61e5e96d9918ff8c809f23948dc4c499535c45` |
| 322017 | 30 | 15 | material | `1a521c801a5fdc5ecfce953874f8c1ee645b321fad4a6dbfa2225f04e569e2b3` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `fdf93bb3c88f03f830223385c4f5d6936f13c122b27ca060b0cd0367f36eac0f` |
| 322017 | 45 | 3 | arms_match | `e27efc83e1a176a647fc2492322772959235912cc8a241e1b136c29574e1a9f2` | `3843a83b9fb3dc6f733a585047e31e892bd35055b75ac510783f7381b9c8f16f` | `83fd5ebfa5f4655d8bcf54aac6ef86f0d3375f7e9cc6f10e82853d3de8ab1c71` |
| 322018 | 0 | 15 | small | `e632048040f5fffe65ac10aa7d6bf4cefd03d7a942b3beaa2229ac3fce23cf56` | `74b698f3aa54c3e21f168a793fe9ebe81788b0f35c1a8c6e39f7650339f22b48` | `f3f5ea61e3620bb16c5c5acb1cfbaed65d35d1242735ec25c3af481fc9861139` |
| 322018 | 15 | 15 | material | `9d63b853b31d820af61835bfc2208cec1f3a4a8eca5b43199bc85821de0b6e63` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `d2f27414d76c6f53d6da498ffb6e9fc8870629518f2c5def23446c953c79c9f6` |
| 322018 | 30 | 15 | material | `d7462d051a2055f9b741450c7a8ceba7d284c253406b81f816651c93cbc4fb13` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `7df9f88a7cb203127c0eefa6908464dadffc860a9bc0627f61af95bbfa5d3ae2` |
| 322018 | 45 | 3 | arms_match | `889bd6584d7f63662de730bd7df95d2a9ac6c7c17880c7085a8d9bd3b459caf5` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `182ed7b68a6e3c19909e3e1b734aa5b6df57f7b535369de5b20e830185d7e356` |
| 322019 | 0 | 15 | small | `a9a5076d4ef603444883747074f038ccb7b8fbf8170db8b668b418697475edca` | `74b698f3aa54c3e21f168a793fe9ebe81788b0f35c1a8c6e39f7650339f22b48` | `731e66f84b8887ce4eb39e053e90cecbdd07b85c544a5c8bdf83c430ea00e952` |
| 322019 | 15 | 15 | material | `c2cf6ae591b0e0537e71d97dc417c91edece2f07346a334b738966e1516a52e6` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `00305d38202c6d58ed64530fa7db4547729ab636a5ee6d5d0c9b734fb88df320` |
| 322019 | 30 | 15 | material | `4733cecc59baa898498f4b36f3a5d422b0b6c59d493803f345408276d316972c` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `8333bce1830f7ad9dac11339bed2646de2c68bb2101096c6ea6a077245d8ed9d` |
| 322019 | 45 | 3 | arms_match | `eba069189d81528f24f681e9218a063bfe5736cb153339bc49cd32465e393168` | `3843a83b9fb3dc6f733a585047e31e892bd35055b75ac510783f7381b9c8f16f` | `12e88acda92f339f7b84c6b64ba22b01b0591020fa5be188794bb325f1e4a1e6` |
| 322020 | 0 | 15 | small | `927d4a2a05daaf5cd0fecf74af1c1bd7d66433a2fb4ef7c00361ad739ec79e1d` | `74b698f3aa54c3e21f168a793fe9ebe81788b0f35c1a8c6e39f7650339f22b48` | `8a6fc5f5c2f491b46b708e5ccee7bcdc43c6495a3d78a69c950d5430ae1abc13` |
| 322020 | 15 | 15 | material | `66eee365ce26eb59060243bee31565146f899b9cfc608f0a8bd524193e4dc1ac` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `8fa0d5291e4b90ab77e9cef37b005bec212f72c96bf022285787cf217a69b1c2` |
| 322020 | 30 | 15 | material | `e9063e2d1574ea7eb18db204ed4566dff1a98b888601193622c4606e1da34ff1` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `f808818a040b14abd8931ad925d3c2f35e41c35c1269f83c3d6fa719db5037ec` |
| 322020 | 45 | 3 | arms_match | `9328493aea415ab3a34e735207b4c13e2c4b998d19aeee002724d3b18d8df6dc` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `741287effb59e4bd320af944226980a7f3211b67468f89d90914fdc73bfeaf8f` |
| 322021 | 0 | 15 | small | `a4df1caf416d1d7e8e812c83d0abbf5f9a301cda6dd749276b2366af46a34281` | `74b698f3aa54c3e21f168a793fe9ebe81788b0f35c1a8c6e39f7650339f22b48` | `9023954c5abf6ae0caba220f918afc3ade0f99895d8867a8bccf4a4e3227a02a` |
| 322021 | 15 | 15 | material | `c267dcbdfda36dc74cc9c51f7a4a7115a6d5abe789d8558cc6dde8566baa7f3b` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `64215aaf2f84d9103fa7de189cd733b4a05558d30af0a5bd8222869f16d52b46` |
| 322021 | 30 | 15 | material | `0d4b7a8637f49a4130517b7423e36dc37c520e113175acf0745e066cb8926a83` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `606a97f2853b9d8dbde408df76617619c59441a00260b02b6c458b6b1135452d` |
| 322021 | 45 | 3 | arms_match | `3c28331fd56c9fe093bb690c757c89c4f2c75cbf7ce142983dec7fdbbd6e1b32` | `3843a83b9fb3dc6f733a585047e31e892bd35055b75ac510783f7381b9c8f16f` | `167aabebc74f2bf208d657925f18c577c5a03dc9601a7a2468bdd17bcd7e8227` |
| 322022 | 0 | 15 | small | `88628557ecafe3272c4561aff823477b7fffc99b8b57fb6b9abc2a8e14091bf3` | `74b698f3aa54c3e21f168a793fe9ebe81788b0f35c1a8c6e39f7650339f22b48` | `613a03eecf7fa8361d9cf7d1290761dd4f5a00d7ce25ac801fc4fdb80ecbcf75` |
| 322022 | 15 | 15 | material | `08070f1f43b8bc2b3f642522b5db2a89430122cc459abc3f1462122279cb048e` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `97d89f8acb84d371c4078217404b7209d6a0829a2579c93a18cb025f2a485f3c` |
| 322022 | 30 | 15 | material | `aaa2cbfef78eb9670603d2544b6561bc11fe24b3a00d2ca432631caff963f061` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `af82afb738677c4f4380ad4e4f1846ce12d36217b835ab1dc434b35ce7d40716` |
| 322022 | 45 | 3 | arms_match | `453780d421f6976a9de3d47754db55177b7b3b879a4ad2ea86453a66c128764a` | `14d70d2bff04b6fc081c1e20a43ded3b43aa216c831c5fae70edbc0e09dec8bc` | `90ddf60c0d4a88734e7f11406593167b35320392932ed033be268ed5ea8433bb` |
| 322023 | 0 | 15 | small | `5e5f1c617aac2e3f999bc66ddcfd2d2851de57dc6ddfbea7c5149b6bd9e71382` | `74b698f3aa54c3e21f168a793fe9ebe81788b0f35c1a8c6e39f7650339f22b48` | `99245f5bab072deebbbaa3534718a2b8899c220955f915422ae9cb0581427fd6` |
| 322023 | 15 | 15 | material | `d6f47b72c2a66fd89b13597fac388d2246465b4c24a049ccfd5317ae23df8af7` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `91c12b40f59c2557a408e9d350ae0a8e6dd4a3fe28f863713bc90e9e4f7aa101` |
| 322023 | 30 | 15 | material | `e25380f3f0aa0f69aa8e4732727b59e0e00e2eca4fe79e76ae5439d3db0286cd` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `42c96146b3d5f0213d82c325bf8507b76d0829c670602c497bbc60bad397641a` |
| 322023 | 45 | 3 | arms_match | `f41899516da38645723982acb1341be85b3e4549687f5144ff9edcb61abda912` | `45a04bf411c28644de7f66465886f88fa94774021a0b3041954b137d2687b4fb` | `a9010d124e727e44384c78ef5ec9e9558aa08b07a408c8504d9fb534a5193b08` |

## 8. 停止

Gate 为 `ARRIVAL_AWARE_LABEL_EXPANSION_INCONCLUSIVE`。下一步只准备一份按决策时刻分开的训练目标决策文件；在研究负责人给出 payload 损失预算之前，不选择 `lambda_H`，不构建全量数据集，不训练 HGNN。本阶段到此停止。
