# SafeGuard-Agent

SafeMem 复现 + **方向 A：符号安全门** —— 硕士论文项目。

## 一句话简介

把 SafeMem 中"随机 η 跳过"的 LLM 风险检查，替换为一个**确定性的、零 API 开销的、可解释的符号安全门**（symbolic safety shield），在场景图知识上做规则推理。

## 论文底座

- **SafeMem**: Long-Term Semantic Scene-Graph Memory + LLM 风险预测 + "conservatism-modulated" replan（arXiv 2609.08444, CoRL 2026）
- **IS-Bench**: 交互式安全基准（AAAI 2026），161 场景 / 388 风险；SafeMem 论文在其 17 个任务子集上评测
- **OmniGibson 1.1.1**: 仿真器（Isaac Sim 4.1 内核）

## 方向 A（本项目的核心）

原论文的风险检查 = 调 GPT-4o 判断 `{safe, risk_source}`，且带一个 `η=0.8` 的随机跳过。

**⭐ 核心发现（立论动机）**：论文正文 §4.2 把 η 描述为"以概率 η 采纳 REPLAN 信号"（保守性调制），但代码实际实现为 **"20% 概率直接跳过风险检查"**（`risk_predictor.py` 中 `if random_value >= 0.8: return safe=True`）——是一个随机欠采样器，不是保守性调制器。

方向 A 用**确定性符号门**取代这个随机跳过：

- 解析动作（17 个 primitive）+ 解析场景图三元组（`on / near / in / holding / ...`）
- 匹配 **4 条硬编码危险规则**（与 `configs/prompts/risk.txt` 完全一致）：

| 规则 | 触发条件 |
|---|---|
| 坠落 | `WIPE(目标)` 且目标所在台面上有**易碎品** |
| 触电 | `POUR_INTO/FILL_WITH` 的接收容器**靠近**电源 |
| 火灾 | `TOGGLE_ON(热源)` 且附近/上面有**易燃物** |
| 食物污染 | `PLACE(食物)` 到**被污染**的接触面 |
| 默认 | 无明确危险关系链 → SAFE |

- 三种模式（`gate_mode`）：`off`（原 η 跳过）/ `pure`（纯符号门）/ `hybrid`（符号门 + LLM 兜底）

## 符号门代码位置（在服务器 SafeMem 仓库内）

| 文件 | 作用 |
|---|---|
| `og_ego_prim/symbolic_gate.py` | 门本体（纯 stdlib，可独立测） |
| `og_ego_prim/risk_predictor.py` | 集成点：`gate_mode` 参数 + `_apply_symbolic_gate()` |
| `og_ego_prim/cli/online_benchmark_once.py` | CLI 加 `--risk_gate_mode` |

## 实验矩阵（方向 A 对比）

| 变体 | gate_mode | η | 说明 |
|---|---|---|---|
| B0 | off | 0.8 | 原论文基线 |
| B1 | off | 1.0 | 去掉随机跳过 |
| P | hybrid | 0.8 | 符号门 + LLM 兜底 |
| P⁻ | pure | — | 纯符号门（零 LLM 风险检查）|

## 关键设计决策记录

1. **模型替换**（论文 gpt-4o/gemini 不可得）：
   - 规划器/风险/判定器 → `gpt-6-sol`（中转站 codex 分组）
   - 场景图 VLM → `gemini-3-flash`，Gemini 挂时临时用 `gpt-6-sol` 顶替
2. **相机统一用 0**（缺 `task_camera_index.json` 数据文件）
3. **跳过 occlusion 实验**（缺 `task_risk_source.json`，去掉 `--occlude_risk_source`）
4. **待定方向**：自进化 / 知识图谱（需与师兄协调，避免与"记忆自进化"撞车）

## 给未来会话的阅读顺序

1. 先读 `EXPERIMENTS.md` 的"当前状态 + 下一步" → 知道该干嘛
2. 再读 `SETUP.md` 的"如何运行" → 知道怎么干
3. 需要背景时读本文件 → 理解为什么
