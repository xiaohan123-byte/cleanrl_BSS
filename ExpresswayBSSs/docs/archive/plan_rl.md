> **存档说明（2026-09-02）**：本文件是 RL（PPO）部分的旧版实现计划，所基于的 MPC 接口已被 `../../code_revision_plan_v3.md` 的连续事件架构取代；RL 功能尚未实现，后续实施前需按 v3 接口修订本计划。

## Approved Plan:
# RL 部分（PPO）实现计划：环境 + 训练 + MPC 接口

## 一、按论文实现 RL 的合理性评估

**结论：整体合理，可以按论文第 3 节实现；三处需做有文档记录的工程简化。**

合理且可直接实现的部分：

1. **分层结构合理**：MPC 处理离散组合路径（强约束、可解释），RL 处理逐电池连续充电功率（跨期累积影响）。动作维度小（3 站 × 5 槽 = 15 维连续），PPO 标准适用。
2. **计算可行**：环境每步需求解一次 MILP，实测默认实例单轮求解 0.01–0.21 s（`data_generation_test/output/mpc_run_result.json`）。一个 episode = 8 步（num_periods=8），每轮 PPO 迭代（num_envs=4 × num_steps=128 = 512 步）约 512 次求解 ≈ 1 分钟量级，CPU 可承受。
3. **critic 终端价值接口已就绪**：MPC 侧 `RLSignals`（`terminal_soc_value` λ^S、`outside_swap_value(i, rho)` ΔV^out）与 `RLProvider` 协议已定义好，真实 RL 只需实现同一协议即可替换 `MockRLProvider`，MPC 代码零改动。λ^S = ∂V/∂S 用 autograd 直接提取；ΔV^out 用论文式 (outside_swap_value) 的成对反事实 critic 评估（继承 `RLSignals` 并重写 `outside_swap_value` 方法即可支持非线性价值）。
4. **接口预留完好**：`run_rolling_mpc(params, network, mock, plan, rl_provider=...)` 与 `generate_dayahead_plan(..., rl_provider=...)` 都已支持注入真实 provider；`run_mpc.py` 的 `_execute_first_stage` / `_path_swap_events` 即论文 Exec_q 执行算子，环境可直接复用，避免重写已验证的执行逻辑。

需要简化/注意的部分（均在代码 docstring 中明确标注）：

1. **参考 rollout（论文式 eq:rl_power_rollout）的"名义可行路径组合"**（先最小化路径变更数再字典序选择）实现成本高。简化为确定性名义路径：决策预约用户沿**日前基准路径**（按有效入口 SOC 检查逐弧可行性，不可行则该用户不参与 rollout 库存递推），加固定承诺事件与预测随机请求 FCFS。这是论文名义路径的零调整特例。
2. **日前计划的循环依赖**：论文规定日前充电功率由"训练完成的 actor"生成，但训练本身需要日前计划。训练期日前计划固定用 `MockRLProvider` 生成（与当前 MPC 测试一致）；评估/部署时可用训练后的 provider 重新生成日前计划（提供开关）。与论文"部署时由训练完成的 actor 生成"一致。
3. **站级能量上限的可微归一化**：论文建议 actor 内部做可微归一化且 PPO 比率针对归一化后动作计算，其雅可比复杂。工程折中：策略分布为逐槽独立 squashed Gaussian（sigmoid 映射到 [0, 槽位上限-容差]，log_prob 含雅可比修正）；站级能量上限的确定性归一化投影放在环境侧执行（类似 DDPG 的 action clipping 惯例）。偏差写入 docstring。

环境数据仍用 `data_generation_test` 的 mock 生成器（每 episode 换 seed），与现有 MPC 管线一致；`data_generation_rl` 的真实 NIO 数据接入不在本次范围。

## 二、实现内容

### 1. 新建 `src/env_bss.py` —— Gymnasium 环境 `BSSChargingEnv`

- **构造参数**：`params`、`candidate_network`、`rl_provider`（缺省 `MockRLProvider`，此时动作用于替代其 h=0 请求）、`reward_scale`、`episode_seed_sampler`。
- **reset(seed)**：`generate_mock_data(params, seed)` 生成新 episode 数据 → `generate_dayahead_plan(..., rl_provider=MockRLProvider)` → 初始化 `soc_obs`、commitments、ell=0 → 返回编码观测。
- **step(action)**（复用 `run_mpc.py` 已验证逻辑，通过 `from run_mpc import _execute_first_stage, _path_swap_events, _index_reservations`）：
  1. 动作投影：逐槽 clip 到 `[0, slot_limit - p_tol]`，站级 ΣP 归一化到 `station_limit - n_slot*p_tol`（式 eq:rl_action_projection）；
  2. 预约四态划分（pending/arr/fix/done，含逾期未到处理，同 run_mpc.py）；
  3. 构造 `RLSignals`：h=0 用投影后动作，h=1..H-1 用 provider 的参考 rollout 序列，终端价值取 provider 的 λ^S/ΔV^out；
  4. `MPCController.solve_step` → 发布 arr 用户路径 → `_execute_first_stage` 用**实际**随机到达执行（Exec_q）→ 新承诺入库；
  5. reward =（式 eq:rl_reward）已实现预约+随机收益 − 实际充电成本 − 本轮发布用户调整成本，乘 reward_scale；info 携带各分项；
  6. terminated = (ell == num_periods-1)；返回下一观测。
- **观测编码 `_encode_state()`**（固定维度，对应式 eq:rl_state 的工程版）：
  - 全局：ell 归一化、时段 sin/cos；
  - 未来 H 步电价窗口（补尾）、当前服务价；
  - 逐站逐槽 SOC（原始顺序，actor 逐槽用；critic 内部池化）；
  - 预约聚合 f_A：各 O-D 的 arr/pending 计数、平均剩余进入时间、平均入口 SOC、基准路径站访问计数；固定承诺的逐站未来窗口事件数与平均退回 SOC；
  - 随机：上一时段实际到达（每站计数/服务数/平均 SOC），预测窗口（每站每步计数与平均 SOC）。
- `observation_space = Box(dim)`，`action_space = Box(n_sta*n_slot)`（未投影的原始动作，环境内投影）。
- 每步自检沿用 run_mpc.py：SOC∈[0,1]、电池守恒、预约优先、状态衔接；MPC 非最优抛清晰异常。

### 2. 重写 `RL/ppo.py` —— 连续动作 PPO + RLProvider 实现

- **Agent（nn.Module）**：
  - critic：逐槽共享 MLP 编码（SOC + 站嵌入 + 全局上下文）→ 求和池化（置换不变，DeepSets 式）→ MLP → V(s)；提供 `value_with_soc_grad(obs)`: autograd 计算 λ^S_{i,b}=∂V/∂S_{i,b}，以及 `outside_swap_value(state, i, rho)`: 将规范化槽位 SOC 置 ρ 与置 1 的两次前向差值（式 eq:outside_swap_value）；
  - actor：逐槽共享 MLP → 每槽高斯均值，全局可学 log_std；sigmoid squash 到 (0,1)，log_prob 含雅可比修正（squashed Gaussian）；`mean_action()` 供参考 rollout 与部署。
- **`PPOProvider`（实现 `RLProvider` 协议）**：持有 agent 引用 + 环境语义知识：
  - `get_signals(params, period_ell, horizon, soc_obs, ...)`：从当前观测出发做 H 步参考 rollout——均值动作 → 满电补足/容量校正 → 名义换电（固定承诺 + 基准路径事件 + 预测随机 FCFS，最小可用槽位规则）→ 得到请求功率序列与参考终端状态 s_ref；λ^S 由 critic 在 s_ref 的梯度给出；返回 `CriticRLSignals(RLSignals)`（重写 `outside_swap_value` 为 critic 成对反事实）。
  - 注：provider 需要预约/预测信息做 rollout，而 `RLProvider.get_signals` 协议签名只有 soc_obs。环境每步调用 provider 的扩展方法（携带窗口上下文）构造信号；同时保留协议兼容的回退实现（无上下文时只做充电递推，等价 Mock 语义）。
- **训练主循环**（保留 CleanRL 骨架与现有注释风格）：
  - `gym.vector.SyncVectorEnv` 包 `BSSChargingEnv`；连续 Box 动作断言替换原 Discrete 断言；
  - 存储 obs/actions/logprobs/rewards/dones/values；GAE（γ=0.99 默认，episode 仅 8 步）；clip surrogate、可选 value clip、熵正则、梯度裁剪、可选 target_kl、学习率退火；
  - 每轮迭代前 `provider.update_agent(agent)`（SyncVectorEnv 同进程共享对象）；
  - tensorboard 记录 episodic_return、业务分项（收入/充电成本/调整成本/随机服务率）、losses、SPS；日志写 `logs/`；
  - 定期保存 checkpoint（last + 按验证 episode return 的 best）到 `RL/checkpoints/`。
- **CLI（tyro）**：`--seed --total-timesteps --num-envs --num-steps --gamma --reward-scale ...`；`--eval --checkpoint PATH` 模式：加载 agent，用 `PPOProvider` 跑 `run_rolling_mpc(..., rl_provider=provider)` 并与 `MockRLProvider` 结果对比输出。
- **运行环境**：conda 环境 `py310`（torch 2.4.1、 gymnasium 0.29.1、tyro、gurobipy 13.0.2，Gurobi 许可证已验证可用）。

### 3. 验证

- `python -m py_compile` 两个文件；
- 环境冒烟：MockRLProvider + 随机动作跑 1 个完整 episode，核对 reward 分项与 run_mpc.py 同 seed 结果口径一致、SOC/守恒自检通过；
- 短训练（如 2k–5k timesteps）确认 loss/返回正常、无 NaN、checkpoint 可加载；
- eval 模式用训练后 provider 跑 `run_rolling_mpc` 成功输出汇总（与 Mock 基线对比）。

### 4. 不改动的文件

`run_mpc.py`、`src/mpc_model.py`、`src/dayahead_plan.py`、`data_generation_test/*`、论文 tex 均不修改（provider 协议已预留）。`todo.md` 的 P2"接入并评估训练后的 PPO-MPC"条目在实现完成后勾选对应子项。

## 三、风险与对策

- **MPC 求解时间随预约数增长**：默认 4 预约很小；若增大 mock 规模，先测单步时间再定 total_timesteps。
- **episode 短（8 步）**：PPO 的 num_steps 跨多个 episode，done 截断由标准 bootstrap 逻辑处理；gamma 默认 0.99，可加 `--gamma 1.0` 对照。
- **不可行/非最优**：沿用 run_mpc.py 的显式异常；训练循环捕获后计为该 episode 失败并终止（截断），避免污染 batch。