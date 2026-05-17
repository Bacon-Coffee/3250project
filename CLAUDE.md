# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

这是一份课程论文（3250）的实验代码仓库，主题为 **"经典调度算法 → Linux CFS → EEVDF 的工业演化实证对比"**。仓库当前为空，按 `/Users/xavier-macbookair/.claude/plans/linux-cfs-crispy-blossom.md`（已被用户批准）逐步构建。每次开工前先读这份计划文件。

最终交付：一个 Python 调度模拟器（6 种算法 + 双核 + 简化 load balancing）+ 实验图谱 + IEEE 格式论文。

## 关键架构不变量（违反任何一条都会拖垮论文论证）

1. **手写红黑树是核心卖点，禁止用 `sortedcontainers` 替代**。`src/core/rbtree.py` 必须从零实现（左旋/右旋/双红/双黑修复全部齐全），同时被 `cfs.py` 和 `eevdf.py` 复用，差异仅在比较 key（CFS 用 `vruntime`，EEVDF 用 `virtual_deadline`）。
2. **统一调度器接口**。`src/core/scheduler_base.py` 中定义的 `on_arrival / on_tick / on_block / on_unblock / pick_next / peek_steal_candidate` 是所有 6 个算法的契约 — 不要为 CFS/EEVDF 开特例分支，新算法必须实现全部接口。
3. **Load balancer 算法无关**。`src/core/cpu.py::LoadBalancer` 只通过 `peek_steal_candidate(cpu_id)` 接口拿候选任务，算法本身负责定义"steal 谁最便宜"（CFS/EEVDF 选 rightmost，FCFS/RR 选队尾等）。
4. **内核常量必须带出处注释**。`prio_to_weight[]`、`NICE_0_LOAD`、`sched_latency`、`min_granularity` 等数字直接从 Linux 内核源码复制，并在 docstring 标注 commit hash 或文件路径。论文里要引用，代码里就得能溯源。
5. **EEVDF 的 lag / eligibility / virtual_deadline 公式必须在 docstring 注明来自 Peter Zijlstra 2023 系列 commit**。这是论文 Methodology 章的引用基础。
6. **事件驱动模拟器是单线程**。模拟双核 CPU 不引入真实并发；论文里诚实声明"未建模 SMP 内存一致性"。

## 项目结构（按计划文件构建）

```
src/
├── core/           # Process、事件引擎、抽象基类、指标采集、RB 树、多 CPU
├── algorithms/     # 6 个调度器实现（FCFS / SJF+SRTF / RR / MLFQ / CFS / EEVDF）
├── workloads/      # 合成负载生成器 + Bitbrains GWA-T-12 trace 加载器
├── visualization/  # Gantt、RB 树插入动画（论文图源）、对比图
└── experiments/    # run_all.py + microbench_rbtree.py
data/bitbrains/     # GWA-T-12 真实 trace（需手动下载）
results/{figures,csv}/
paper/{main.tex,references.bib}
tests/
```

实现顺序参照计划文件的 Phase 0–10，**不要跳跃**：Phase 2（手写 RB 树）必须早于 Phase 4（CFS），Phase 4 必须早于 Phase 5（EEVDF 复用 CFS 红黑树）。

## 常用命令

仓库尚未初始化，以下命令会在 Phase 0 完成后可用：

| 操作 | 命令 |
|---|---|
| 安装依赖 | `pip install -e ".[dev]"` |
| 跑全部测试 | `pytest` |
| 跑单个测试文件 | `pytest tests/test_rbtree.py -v` |
| 跑单个测试函数 | `pytest tests/test_cfs.py::test_fairness_with_nice -v` |
| 测试覆盖率 | `pytest --cov=src --cov-report=term-missing` |
| Lint | `ruff check src/ tests/` |
| 自动修复 lint | `ruff check --fix src/ tests/` |
| 一键跑全部实验 | `python -m src.experiments.run_all` |
| RB 树 microbench | `python -m src.experiments.microbench_rbtree` |
| 编译论文 | `cd paper && latexmk -pdf main.tex` |

## 验证手段（实现后必须自查）

代码完成不等于实验成功。**必须跑通**这些 sanity check（详见计划文件验证小节）：

1. **手写 RB 树**：10k 随机插入/删除后 inorder 序列与 `sortedcontainers.SortedList` 严格相同。
2. **CFS 公平性**：`nice=0 / nice=0 / nice=-5` 三个长跑任务，CPU 占比近似 `1024 : 1024 : 3121`。
3. **EEVDF 复现**：跑 Zijlstra commit 中描述的"短任务被长任务挤压"case，EEVDF 的 P99 响应时间应显著低于 CFS。
4. **双核负载均衡**：所有任务初始到达同一 CPU 的极端场景下，最终两 CPU 利用率均 ≥80%。

## 论文输出约束

- 全程只画 **5 张主图**（计划文件 Phase 8 列出），其他图归附录。"图多但故事散"是论文头号风险。
- References 必须包含 **真实 Linux commit hash**（去 `git log kernel/sched/fair.c` 找 Ingo Molnár 2007 引入 CFS 的提交、Peter Zijlstra 2023 EEVDF 系列）。
- 文档类文件（README、CLAUDE.md、论文正文外的笔记）用中文；代码注释用英文。

## 不要做的事

- 不要重新加回原计划中删除的 Priority Scheduling（与 MLFQ 重复度高，已剪枝）。
- 不要扩到 4+ 核 NUMA — 计划范围是 2 核 + 简化 idle balance，再大就超时。
- 不要把 EEVDF 退化成"CFS 加几行"。它的 key 是 `virtual_deadline`，pick_next 必须先做 eligibility 过滤再选 deadline 最小者。
- 不要进程数开到 1000+（Python 模拟会爆）。规模上限 500。
