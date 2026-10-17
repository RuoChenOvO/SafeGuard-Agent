# SETUP —— 环境、运行、踩过的坑

> 本文件是"操作手册"：每次要跑实验时看这个。首次接手先读 `EXPERIMENTS.md` 的"下一步"。

## 1. 服务器

| 项 | 值 |
|---|---|
| IP | `10.21.22.46`（校园网内网）|
| 用户 | `ruochen` |
| 密码 | 向用户索要（**不写入仓库**）|
| 登录方式 | 从本机 WSL 用 `sshpass -p '<密码>' ssh ruochen@10.21.22.46` |
| ⚠️ 公钥免密 | 失败原因：家目录 `/home/sdb/ruochen` 属主是 `ys:ys` 且组可写，OpenSSH StrictModes 拒绝 `authorized_keys`。要用免密需 `chown ruochen:ruochen` + `chmod 700`（需 root）|

## 2. 路径清单

| 路径 | 内容 |
|---|---|
| `/home/sdb/ruochen/SafeMem-workspace/` | 项目工作区根 |
| `repos/SafeMem` | SafeMem 代码（符号门改动都在这里；`data` 是软链 → `repos/IS-Bench/data`）|
| `repos/IS-Bench` | IS-Bench 数据（161 任务/161 bddl/29 场景/metas/cameras）|
| `envs/isbench` | conda 环境（Python 3.10 / torch 2.5.1 / CUDA 12.1 / omnigibson 1.1.1）|
| `data/omnigibson` + `assets/omnigibson` | OmniGibson 数据集 + 资产 |
| `checkpoints/sam2.1_hiera_large.pt` | SAM2 权重 |
| `checkpoints/Qwen2.5-VL-7B-Instruct/` | 本地 Qwen（备用）|
| `third_party/sam2` | SAM2 源码（已 pip install -e）|
| `run_gate.sh` | 运行脚本（工作区根）|

## 3. conda 环境

- **直接用 `envs/isbench/bin/python`**，不要 `conda activate`（非交互 ssh 会失败）
- 完整路径：`/home/sdb/ruochen/SafeMem-workspace/envs/isbench/bin/python`

## 4. GPU（会被其他用户动态抢占！）

| index | 卡 | 显存 |
|---|---|---|
| 0 | A40 | 46G |
| 1 | RTX 4090 | 24G |
| 2 | RTX 4090 | 24G |

- **渲染器**：自动选（通常 GPU1）。**不要**给主进程设 `CUDA_VISIBLE_DEVICES`（会和 Isaac Sim 的枚举冲突导致渲染崩溃）
- **SAM2**：通过 `SCENE_GRAPH_GPU` 绑定（`scene_graph_client.py`，默认 `"0"` = A40，显存余量最大最抗抢占）
- 跑之前先 `nvidia-smi` 看哪张卡空闲

## 5. API 中转站

| 项 | 值 |
|---|---|
| base_url | `https://api.wpironman.top/v1`（**直连，不要设代理**）|
| 规划/风险/判定模型 | `gpt-6-sol` |
| 场景图模型 | `gemini-3-flash`（挂时临时用 `gpt-6-sol`）|
| codex key（gpt-6-sol）| 服务器本地 `~/.codex/auth.json` 的 `OPENAI_API_KEY`（**不进仓库**）|
| gemini key | 在 `run_gate.sh`（⚠️ **提交 git 前必须移除硬编码 key**，改成读服务器本地文件）|

## 6. 如何运行

`run_gate.sh` 参数：`<task> <gate_mode> <try_id> <prob> <gpu> <camera>`

```bash
bash /home/sdb/ruochen/SafeMem-workspace/run_gate.sh \
  cook_tofu__with_bottle__of__oil off smoke_off_01 0.8 '' 0
```

- `gate_mode`: `off`(原η跳过) / `pure`(纯符号门) / `hybrid`(门+LLM兜底)
- `prob`: η 值（off 模式用，pure 忽略）
- `gpu`: 留空（渲染器自动）；`camera`: 统一 `0`

日志会写到 `run_<try_id>.log`。结果在 `repos/SafeMem/results/smoke/benchmark/<task>___<scene>/<try_id>_<model>/report.json`。

## 7. ⭐ 7 条关键修复（本次调试的结晶，别再踩）

| # | 问题 | 根因 | 修复 | 文件 |
|---|---|---|---|---|
| 1 | 渲染崩溃 | `CUDA_VISIBLE_DEVICES` 与 Isaac Sim 设备枚举冲突 | 主进程不设该变量 | run_gate.sh |
| 2 | SAM2 OOM | 和渲染器/他人任务挤同一张 4090 | 绑到 A40 | scene_graph_client.py (`SCENE_GRAPH_GPU`) |
| 3 | 相机"多观测"报错 | 缺 `task_camera_index.json` | 用 `--scene_graph_camera_index 0` | run_gate.sh |
| 4 | API 连接失败 | 误设代理指向已关闭的 17897 端口 | 去掉代理，直连 wpironman | run_gate.sh |
| 5 | 判定器/规划器 404 | 硬编码 `gpt-4o` 但分组没有 | 全改 `gpt-6-sol` | evaluator.py 等 |
| 6 | 缺数据文件崩溃 | 缺 `task_risk_source.json` | 去掉 `--occlude_risk_source` | run_gate.sh |
| 7 | 规划器输出解析失败 | gpt-6-sol 输出裸 JSON（无 ```json 围栏），且推理吃满 512 token | `parse_output` 健壮化 + `max_completion_tokens=4096` | plan_agent.py, server_inference.py, risk_predictor.py |

## 8. 已知坑

- **Gemini 会间歇性 503**（中转站上游故障）：重试 5 次（已加在 gemini.py）；仍失败就把场景图切到 `gpt-6-sol`
- **GPU 抢占**：其他用户任务会中途抢占空闲卡 → SAM2 绑 A40 最稳
- **段错误**：多为 GPU 抢占/OOM 的连带（native 崩溃），不是代码 bug
- **缺两个数据文件**：`task_camera_index.json`、`task_risk_source.json`（SafeMem 发布数据未给全，见 #3/#6 的绕过方式）
