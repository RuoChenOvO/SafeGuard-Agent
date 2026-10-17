# safe-mem-patches

对 SafeMem 仓库（`/home/sdb/ruochen/SafeMem-workspace/repos/SafeMem/`）的改动文件。**按下面路径放回对应位置**：

| 本目录文件 | 放回 SafeMem 的目标路径 | 改动 |
|---|---|---|
| `risk_predictor.py` | `og_ego_prim/risk_predictor.py` | 加 `gate_mode`（off/pure/hybrid）+ `_apply_symbolic_gate()`；max_completion_tokens 512→4096 |
| `online_benchmark_once.py` | `og_ego_prim/cli/online_benchmark_once.py` | 加 `--risk_gate_mode` CLI 参数 |
| `plan_agent.py` | `og_ego_prim/models/plan_agent.py` | `parse_output` 健壮化（兼容裸 JSON）|
| `server_inference.py` | `og_ego_prim/models/server_inference.py` | max_completion_tokens 512→4096 |
| `scene_graph_client.py` | `og_ego_prim/scene_graph_client.py` | SAM2 绑 `SCENE_GRAPH_GPU`（默认 GPU0=A40）；timeout 600→3600 |
| `evaluator.py` | `og_ego_prim/benchmark/evaluator/evaluator.py` | 判定器模型 gpt-4o→gpt-6-sol |
| `gemini.py` | `scene_graph/vlms/gemini.py` | Gemini 调用加 5 次重试 |

**新增文件**（不在 SafeMem 原仓库，需单独添加）：
- `og_ego_prim/symbolic_gate.py` ← 本仓库根目录的 `symbolic_gate.py`（符号门本体）

> 符号门本体在仓库根目录 `../symbolic_gate.py`（纯 stdlib，可独立运行测试），也是方向 A 的核心交付物。
