"""Chinese audit report. The recommendation rule is fixed before the probe numbers."""
from __future__ import annotations


def _num(value, digits: int = 6) -> str:
    if value is None:
        return "缺失"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{float(value):.{digits}g}"


def _batch_lines(batches: list[dict]) -> list[str]:
    lines = []
    for batch in batches:
        credit = batch.get("credit") or {}
        if not credit.get("published"):
            lines.append(f"- {batch['variant']} seed {batch['model_seed']}（{batch['source']}）：梯度未发布。")
            continue
        lines.append(
            f"- {batch['variant']} seed {batch['model_seed']}（{batch['source']}）："
            f"标签反事实梯度差范数 {_num(credit.get('tag_difference_separate_norm'))}，"
            f"共享尺度差范数 {_num(credit.get('tag_difference_shared_norm'))}，"
            f"正奖励子集优势增量平均绝对值 {_num((credit.get('tag_advantage_positive_bonus') or {}).get('mean_abs'))}，"
            f"联合裁剪显示比例 {_num(credit.get('joint_clip_scale'))}。"
        )
    return lines


def render_report(summary: dict) -> str:
    history = summary["history"]
    probes = summary["probes"]
    exposure = history["exposure_totals"]
    means = history["gate"]["means"]
    mean_lines = [
        f"- seed {item['model_seed']}：相对同种子 L4_C 的标记服务均值 {_num(item['versus_l4_c_tagged_mean'])}，"
        f"区间 [{_num(item['versus_l4_c_ci_low'])}, {_num(item['versus_l4_c_ci_high'])}]；"
        f"相对 L0 的标记服务均值 {_num(item['versus_l0_tagged_mean'])}。"
        for item in means
    ]
    initial = [batch for batch in probes["batches"] if batch["role"] == "initial_reconstruction"]
    terminal = [batch for batch in probes["batches"] if batch["role"] == "terminal_frozen_batch"]
    matched = sum(bool((batch.get("sample_match") or {}).get("ok")) for batch in initial)
    published = sum(bool(batch.get("historical_gradient_published")) for batch in initial)
    lines = [
        "# 训练信号与团队信用审计",
        "",
        f"科学 Gate：`{summary['scientific_gate']}`。E12 的 `QUEUE_LOOKAHEAD_BENCHMARK_FAIL` 保持不变，passing seeds 为 0。",
        "本轮没有训练，没有调用参数更新，没有合并，也没有创建 L5。审计完成不是性能验收，也不构成合并许可。",
        "",
        "## 1. 历史日志能够证明什么",
        "",
        "320 行开发池结果按冻结的 `score_l4` 与 `choose_gate` 重算后仍是 FAIL。三个同种子对照的标记服务均值置信区间都跨过 0；相对 L0 有两个种子的标记服务均值为负，这是预注册 FAIL，不是显著劣化的证明。",
        *mean_lines,
        f"六组训练日志合计 {exposure['updates']} 次更新。正标签奖励步数 {_num(exposure['positive_tag_bonus_steps'], 0)}，"
        f"float32 可分辨步数 {_num(exposure['float32_distinguishable_tag_steps'], 0)}。"
        "这些计数说明标签奖励进入过训练记录，也说明不能把 float32 说成大面积抹掉了奖励。",
        "保存的样本只有每个种子的更新 1、150、300 且只有环境 0，每组 144 行。样本里没有完整观测、价值、终止标志、旧对数概率、minibatch 顺序和 Adam 矩。因此更新 150 和 300 的真实梯度与参数更新不能从现有文件精确复原，也没有用最终 checkpoint 回填。",
        "同一场景、同一次更新、同一条 rollout 里的 UAV 步不是独立实验。日志分布只作描述。",
        "",
        "## 2. 前瞻暴露是否在真实训练中持续",
        "",
        f"L4_C 的非零前瞻输入 UAV 步为 {_num(exposure['l4_c_nonzero_input_uav_steps'], 0)}。"
        f"L4_Q 的非零前瞻输入 UAV 步为 {_num(exposure['l4_q_nonzero_input_uav_steps'], 0)}，"
        f"带标记积压的有效关联 UAV 步为 {_num(exposure['l4_q_tagged_backlog_valid_uav_steps'], 0)}。",
        "控制组输入保持为零，处理组只在真实标记积压出现时接收非零前瞻列。这个对比说明前瞻暴露在 L4_Q 的训练日志中持续存在；它不证明额外观测改善了开发池服务。",
        "",
        "## 3. 标签增量在奖励、优势和策略梯度中的尺度",
        "",
        f"初始重建与保存的环境 0 切片吻合 {matched}/6。历史梯度发布 {published}/6。环境 0 吻合只覆盖保存下来的切片，不能称为完整复现的四环境历史批次。",
        "优势分解同时报告两种尺度：共享 full-advantage 的均值和总体标准差时，基线、年龄增量和标签增量相加；各自按原训练规则重新归一化的差异不是可加分解。float64 递推只作数值诊断，历史训练使用的是 float32 GAE。",
        "策略梯度用冻结批次上的 `autograd.grad` 计算。`g_policy_full - g_policy_no_tag` 是固定 critic 和固定批次上的标签反事实梯度差，不是长期因果贡献。旧对数概率与当前对数概率在采样点对齐，ratio 接近 1，因此这次静态探针不重现多轮参数更新后的 PPO 裁剪。历史裁剪频率只来自训练日志。",
        *_batch_lines(initial + terminal),
        "",
        "## 4. 团队广播和联合裁剪：证据与推测",
        "",
        "源码和批次都显示一个团队标量优势被广播到全部 UAV，并且 `policy_active_mask` 全为 1。同一步上各 UAV 的策略项权重相同；服务标签的 UAV 与其他 UAV 的差别来自它们落在哪些时间步，以及各自动作的对数概率梯度。各 UAV 的策略梯度片可以相加还原总梯度。",
        "当前没有服务、但优势中仍有未来标签增量的步，不能解释成没有未来影响。共享团队优势是原 MAPPO 设计，本身不是错误。由此说无关 UAV 稀释了信用，只是诊断假设，不是由广播或相关单独得到的因果结论。",
        "实现把 actor 与 critic 参数连在一起，使用一个 `max_grad_norm` 和 Adam。报告写出了合成梯度范数、critic 加权范数、联合范数、平方范数占比，以及原裁剪规则下的显示比例。分开裁剪的比例只展示，没有应用，也没有接入训练。梯度被缩放和 Adam 更新被同一系数缩放不是一回事。E12 checkpoint 没有历史优化器矩，所以不能由裁剪系数推断历史更新缩水。联合裁剪是不是主要瓶颈：未证实。",
        "",
        "## 5. 是否有证据只改一个组件",
        "",
        "没有。日志证明奖励和前瞻输入进过训练，静态探针给出了冻结批次上的尺度，但它们没有把失败隔离到奖励系数、观测列、团队广播或联合裁剪中的单独一个。下一步不改奖励系数，不新增学习器，不开始训练。正式系统与 L0–L4 保持冻结，E10 与 E12 的 FAIL 和原合并标准保持不变。",
        "",
        "## 边界",
        "",
        f"注册批次 {probes['registered_batches']}。同配置第二次完整 rollout 没有列入预算；RNG 恢复检查结果为 {_num(probes['rng_check'].get('restore_returns_captured_state'))}。",
        f"新增失败数 {summary['tests'].get('new_failure_count')}。`release_eligible` 为 false，`merged` 为 false。",
        "",
    ]
    return "\n".join(lines)
