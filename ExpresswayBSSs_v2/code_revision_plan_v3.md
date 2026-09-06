# MPC 连续事件代码修订执行计划（RL 输出暂用 Mock）

> 需求基线：以 `paper/manuscript_revision_plan_v3.md` 为主，以 `paper/manuscript_revision_plan_v2.md` 补充实现细节  
> 适用范围：MPC、连续事件执行、Mock 信号、测试、配置和可再生成的六站模拟数据；不实现 RL 训练，不修改论文正文

## 当前状态（2026-09-03 集中更新，替代原文末的历次实施更新）

**阶段完成度**：P0-1、P0-2、P0-3、P1-4、P1-5、P1-6、P1-7、P2-8 均已完成；
P0-0 仅完成环境固定，未创建历史 golden fixture。RL 训练仍明确不在本
计划范围内。

**当前实现链路**：`run_mpc.py` → `src.continuous_runner.run_continuous_rolling_mpc`
→ `src.paper_mpc.solve_paper_mpc`（运行模式 `PAPER_GUROBI_OPTIMAL`）：每轮
由本机 Gurobi 联合优化路径流 y、访问量 x、调整指示 d、预约
激活/服务/失败/未决 a/s/f/omega、随机请求结果、边界存活 A 与未决交付 w；
附录 B--C 的排队、连续充电和槽位分配用站级事件模式扩展式表示，解后由
统一 ContinuousEventEngine 回放核验。

**2026-08-31**：路径—事件联合 MILP（`src/paper_mpc.py`）接入，运行模式
从 EVENT_PATH_ENUM_REPLAY 升级为 PAPER_GUROBI_OPTIMAL；默认六站 12 轮
全部 OPTIMAL（Gurobi 12.0.1，许可证 2791221）。压力验收：联合共享库存
产生一次优化改路成本；60 kW 场景产生 0.175438596491 h 等待；0 kW 场景
产生一次 1000 的预约失败成本且下游失活。

**2026-09-03**：P2-8 清理完成。`src/mpc_model.py` 的事件接口与精简版
`MPCController` 并入 `src/paper_mpc.py`，旧离散 MILP 及旧类型
（`ReservationObservation`、`FixedCommitment`、`MPCWindowInput` 等）全部
删除；`run_mpc.py` 由 956 行瘦至 236 行，删除无调用方的旧路径函数；
`src/dayahead_plan.py` 删除旧离散日前实现（约 950 → 278 行），公开接口
委托 `src.continuous_dayahead`；`data_generation_test/parameter.py` 删除
`station_power_limit_kw`、`delta_hours`、`full_soc_tolerance`、
`full_power_tolerance_kw` 迁移兼容接口；`src/reference_rollout.py` 删除
`entry_*` 旧字段回退。

**验证基线**：`python -m unittest discover -s tests` 64 项全绿（1 项
跳过）；`python run_mpc.py --seed 42` 12 轮全部 PAPER_GUROBI_OPTIMAL、
首区间回放全部通过，总收益 505.387、调整成本 13、违约 0。RL
actor/critic 仍未训练，P_hat、终端 SOC 系数和域外交付系数继续使用 Mock
参数（输出标注 `signal_source="mock"`，不得作为 RL 性能结论）。

## 0. 目标、依据与完成定义

本计划把论文修订规范中的最终模型口径落实为可编码、可测试、可分阶段验收的工程任务。现有代码不是空白实现：离线候选网络、日前路径三步规则、MPC 目标骨架、预约有效到达链、最小槽位分配和终端信号接口可以复用；时间、用户状态、等待、连续充电、剩余路径发布和实际记账必须重构。

本轮不实现神经网络 actor/critic、Gymnasium 环境或 PPO 训练。论文接口需要的逐槽请求功率、终端 SOC 边际价值和域外交付价值全部由确定性的 `MockRLProvider` 生成；实时位置、车辆 SOC、ETA、预约、随机需求、电价和能量限额也全部由带 seed 的模拟输入提供。Mock 只用于打通模型和验证逻辑，不得用于论文性能结论。

需求优先级固定为：

1. `manuscript_revision_plan_v3.md` 第 2、3.5--3.8、4、8.4 节定义最终业务和数学口径。
2. `manuscript_revision_plan_v2.md` 只补充 v3 未展开的实现细节，例如迟到但未进入用户、携带请求最早服务时刻、同刻逐事件写回、参考需求构造和共享物理内核。
3. 现有代码行为只作为迁移基线，不能覆盖前两项。
4. v2 与 v3 冲突时一律采用 v3：外层长度使用论文符号 `\sigma`、预测域采用半开区间、请求功率是固定参数、日前采用严格到站可用接纳、终端未决并入现有域外价值近似。

v3 中“只修改 `paper/main.tex`”是该论文修订任务的文件边界，不继承到本代码计划。本计划单独授权下文列出的代码和测试文件，但仍不修改 `paper/main.tex`、图片、真实业务数据或参考文献。

全部完成必须同时满足：

- 真实执行、预测重放、reference rollout 和日前库存递推共用同一个连续事件内核。
- 预约和随机请求均可跨外层边界等待，原请求 ID、到站时刻和截止时刻不变。
- 在途用户从实时虚拟起点重优化未执行路径，路径发布和调整成本只在真实发布时发生一次。
- 槽位 SOC 按连续时长积分；满电停充，换入低 SOC 电池后恢复当前区间请求功率。
- 服务就绪的物理阈值是 SOC 精确达到 1；求解器容差不改变电量、收入或续航。
- 同一槽位在一个外层区间内可以多次服务，但两次交付之间必须真实重新充满。
- 真实服务、超时、充电分段和发布均有全日唯一 ID，跨轮不漏记、不重记；终端 SOC/域外价值只是每次求解重算的目标近似，绝不进入真实 ledger。
- v3 第 8.4 节的 14 个场景全部成为自动化测试。
- 小实例的 MPC incumbent 可由连续事件执行器重放，重放后的服务、SOC、队列和目标分项一致。
- 默认端到端实例包含 6 个换电站，所有外部字段和终端信号均可由同一 seed 完整再生成。

## 1. 固定技术口径

以下决定不是实施时的可选分支。

| 主题 | 固定实现 |
| --- | --- |
| 物理时间 | 所有连续时刻统一用从运营日起点开始的小时数 `float`；外层区间索引只用于动作、电价和能量限额。 |
| 代码命名 | `interval_hours` 对应论文 `\sigma`，`max_wait_hours` 对应 `\Delta t`；删除业务代码中的 `delta_hours` 和“等待时长也叫 delta”的混用。 |
| 时间边界 | `TimeGrid.interval(q)=[q*sigma,(q+1)*sigma)`；预测域为 `[t_ell,t_{ell+H})`；终点事件交给下一轮，终端状态取 `t_end^-`。 |
| 数值比较 | `time_epsilon` 只用于把已知由边界运算产生的浮点误差规范到精确边界；它不能吞并真实的近边界事件。时刻严格小于右端的事件仍属于当前轮，恰等于右端的事件才归下一轮。SOC 不使用“补足”容差。 |
| 服务时长 | 换电为瞬时事件，不增加工位或作业时长。 |
| 请求截止 | `deadline = arrival_time + max_wait_hours`，生成后不可重写。 |
| 同刻顺序 | 先推进并登记充电完成，再登记到站，再按队列分配，最后对仍未服务者判超时；截止时刻充满可以先服务。 |
| 队列优先级 | 预约优先于随机；两类内部均按 `(arrival_time, stable_id)`；未来尚未到站的预约不占用电池。 |
| 槽位规则 | 每次服务选择编号最小的已满槽位；服务后该槽位立即写入退回 SOC。 |
| 请求功率 | 每个外层区间、每站、每槽一个固定请求功率；未满时实际功率等于请求值，满电立即变为 0，换入低 SOC 电池后恢复该请求值。 |
| 动作投影 | 先逐槽裁剪到 `[0, slot_power_limit_kw]`；若 `interval_hours * sum(P_hat) > station_energy_limit_kwh[i][q]`，按同一比例缩放该站全部已裁剪功率。 |
| 站级限制 | 使用逐区间站级能量上限；删除当前代码额外采用的站级瞬时功率上限，不新增充电器数量约束。 |
| 路径差异 | 当前网络是单向下游 DAG，站点位置严格递增，因此站点访问向量唯一确定剩余站序；输入校验不满足这一性质时直接拒绝实例。 |
| 未到用户基准 | 始终使用日前初始发布路径；内部预测改路不发布、不覆盖基准、不进入实际 reward。 |
| 在途用户基准 | 使用最近一次实际发布路径的未执行站序；虚拟起点移动但站序不变时 `d=0`。 |
| 发布 | 只要在途用户的未执行站序变化，就在本轮边界发布并结算一次；未变化不发布。 |
| 最小换电距离 | 相对入口或最近一次真实换电位置计算，不以每轮移动的虚拟起点为基准。 |
| 调用顺序 | 参考路径构造/修复 → Mock 请求功率 rollout → Mock 终端参数 → MPC；未来替换为真实 RL 时仍保持该顺序。本轮最优 `y/alpha/z` 不得回流到 Mock 输入。 |
| 求解异常 | 保留显式的输入错误、不可行和无 incumbent 异常；本计划不发明启发式回退策略。 |
| JSON 迁移 | 可再生成的 mock、日前计划和运行结果直接提升 schema；旧 schema 由 loader 给出“请使用 `--regenerate`”错误，不保留双语义兼容层。 |

## 2. 当前实现审计与目标映射

现有 8 个 `unittest` 已分别在默认 Python 3.9.19/gurobipy 12.0.1 和统一 `py310` 环境下通过。后续开发固定使用已验证的核心环境：Python 3.10.19、NumPy 2.2.6、gurobipy 13.0.2。旧测试是迁移基线，不是新口径的验收结论。

| 当前文件/符号 | 可复用部分 | 与 v3 的冲突或缺口 | 目标处理 |
| --- | --- | --- | --- |
| `data_generation_test/parameter.py::BusinessParameters` | 站点、槽位、价格、SOC、路径成本、求解器参数 | `delta_hours`、`full_soc_tolerance`、`station_power_limit_kw`；缺少等待上限和逐区间能量限额 | 重命名时间字段，新增等待和能量矩阵，删除物理容差补足接口 |
| `data_generation_test/candidate_network.py::generate_candidate_network` | 离线 SOC 分档网络和四步生成法 | `get_feasible_arcs` 只支持入口起点；没有已执行前缀、实时位置/SOC、真实换电历史 | 保留离线生成，新增在线剩余网络构造 |
| `data_generation_test/rl_data.py` | 连续到站时间戳、预测/实际请求分离、`RLSignals` 骨架 | mock 强制旧 FCFS 当期拒绝；provider 只有 SOC，无法形成完整 reference rollout | 升级数据 schema 和 provider 上下文 |
| `src/dayahead_plan.py::_simulate_inventory` | 提交顺序、选站排序、完整路径拒绝 | 整时段先充后换、退回后下期充、容差补足、每槽每期一次 | 改为调用共享连续事件引擎，仍执行严格到站接纳 |
| `src/mpc_model.py::ReservationObservation` | 复合用户键、日前基准路径 | `is_new_arrival` 只区分新进入/未来；没有实时位置、实时 SOC、最近发布剩余路径或等待请求 | 由统一领域状态替换 |
| `src/mpc_model.py::FixedCommitment` | 携带尚未执行事件的想法 | 把在途路径锁死，正是 v3 删除的 `fix` 路径类型 | 删除，所有在途用户继续进入路径决策 |
| `src/mpc_model.py::_build_events` | 按入弧固定枚举预约事件、上游成功链 | 事件按到达区间批量化；`dec/fix` 分类；没有 deadline/pending/服务时刻 | 改为连续候选请求和边界状态 |
| `src/mpc_model.py::build_model` 的 Constraint 3--5 | Gurobi、目标骨架、indicator 使用方式 | 容差补足；整段充电后统一服务；每槽每时段最多一次；按时段库存计优先级 | 重写为事件位置驱动 MILP |
| `run_mpc.py::_execute_first_stage` | 真实请求与预测请求分离、预约优先、最小槽位 | 先充完整时段再批量服务；未服务随机立即拒绝；没有等待队列、deadline 或区间内再次充满 | 删除并由 `ContinuousEventEngine` 替换 |
| `run_mpc.py::run_rolling_mpc` | 滚动、只执行首期、实际 reward 分项 | `pending/arr/fixed/completed` 状态机，只发布新进入用户，固定在途路径 | 改为 `future/enroute` 活动状态与发布剩余路径 |
| `RL/ppo.py` | PPO、clipped objective 和 GAE 骨架 | 仍是 CartPole 离散动作 | 本轮不修改；列入后续独立 RL 计划 |
| `tests/test_full_battery_fcfs.py` | 复合键、预约链、最小槽位等回归思路 | 断言旧的整时段容量和立即拒绝行为 | 先保留为迁移基线，随后由新场景测试替换 |

`main.py` 当前已有用户未提交修改，本计划不修改 `main.py` 或 `RL/ppo.py`，避免覆盖无关工作。

## 3. 目标模块和数据契约

### 3.1 新增 `src/domain.py`

集中定义跨 MPC、执行器、日前和 Mock 信号接口共享的数据结构，避免在 `run_mpc.py` 与 `mpc_model.py` 之间复制状态逻辑：

- `UserKey = tuple[int, int]`。
- `RequestKind`：`RESERVATION`、`RANDOM`。
- `PhysicalRequestStatus`：`WAITING`、`SERVED`、`TIMED_OUT`、`CANCELLED`，只描述已发生的真实状态。
- `PredictedOutcome`：`SERVED_IN_HORIZON`、`FAILED_IN_HORIZON`、`PENDING_AT_HORIZON`，只存在于单次 reference/MPC 结果，不能写回 `RollingState` 或 `RealizedLedger`。
- `WaitingRequest`：`request_id`、`kind`、`station`、`user_key`、`source_arc`、`arrival_time`、`deadline`、`return_soc`、`path_order`。`arrival_time` 和 `deadline` 构造后只读。
- `EnrouteReservation`：实时位置、实时车辆 SOC、已执行站点前缀、日前初始路径、最近发布剩余路径、最近真实换电位置、已知连续 ETA、当前等待请求 ID。
- `SlotState`：`station`、`slot`、`soc`、`ready`、`completion_due_at`、`last_update_time`；槽号代表库存位置，不代表永久电池。初始满电槽 `ready=true`；服务写回后为 false；若恰在预测右端达到 SOC 1，则保存 `completion_due_at=t_end`，下一轮按同刻顺序先把 `ready` 置 true，再允许分配。
- `RollingState`：`now`、逐槽状态、future/enroute 用户、逐站两类真实等待队列、已记账事件 ID 集、最近实际随机历史。只有已经真实到站且尚未结束的请求进入队列；未到预约仍留在 future/enroute，不得用 `PENDING_AT_HORIZON` 代替物理状态。
- `CandidateRequest`：构模前固定的预约入弧事件或随机请求；预约使用 `(p,k,j,i)` 形成稳定 `event_id`。
- `LedgerEntry`：`event_id`、`event_type`、`occurred_at`、`interval`、金额/电量和关联请求；事件 ID 全日唯一。

所有结构提供 `to_dict/from_dict`；反序列化后必须保持稳定 ID、时间和队列顺序。

### 3.2 新增 `src/time_grid.py`

提供唯一时间边界实现：

- `TimeGrid.start(q)`、`end(q)`、`interval(q)`、`interval_of(t)`。
- `prediction_bounds(ell, H) -> (t_start, t_end)`。
- `snap_boundary(t, provenance)` 仅在 `provenance` 表示该时刻由网格端点算术产生时规范浮点误差；不得按“距边界多近”批量改写外生到站或 deadline。
- `contains_execution_time(t, ell)` 使用半开区间。
- `is_terminal_event(t, ell, H)` 保证恰在 `t_end` 的事件不进入当前轮。
- `compare_to_boundary(t,boundary)` 返回 `BEFORE/EQUAL/AFTER`，并只把可证明来自相同边界表达式的运算误差判为 `EQUAL`；模型提取后和执行器重放均使用此唯一判定。

禁止其他文件直接用 `floor(t / interval_hours)` 或手写端点比较。

### 3.3 新增 `src/event_engine.py`

`ContinuousEventEngine` 是唯一物理真相，至少暴露：

`simulate_interval(state, interval_index, requested_power, arrivals) -> ExecutionResult`

`simulate_horizon(state, start_interval, horizon, requested_power, arrivals, stop_before_end=True) -> ExecutionResult`

每次推进采用以下确定算法：

1. 从当前时刻积分到下一个“区间边界、充满、到站、deadline”中的最早时刻。
2. 对所有槽位按 `eta * P * duration / E_B` 更新 SOC；达到 1 后将实际功率置 0。
3. 若时刻是外层边界，关闭上一功率段并启用新区间请求功率。
4. 登记该时刻的全部充满和到站事件。
5. 在每个站反复执行队列分配，直到队列为空或没有满电槽；每次都重新选择最小编号满电槽并立刻写入退回 SOC。
6. 对仍在队列且 `deadline == now` 的请求判超时。
7. 服务写入低 SOC 后，按当前区间同一请求功率重新计算下一充满时刻。

`ExecutionResult` 必须返回：终态、真实携带队列、充电分段、电量、服务、超时、实际功率轨迹和 ledger。预测运行可额外返回派生的 `horizon_pending_ids`，但该字段不是事件、不入 ledger、也不直接写回真实队列。预测终点使用事件前快照，不执行终点事件。

### 3.4 新增 `src/accounting.py`

`RealizedLedger` 只消费真实执行器事件：

- 服务收入按 `TimeGrid.interval_of(service_time)` 的服务价格计算。
- 充电成本对执行器的每个功率段按 `price * power * duration` 积分。
- 路径调整成本在 `PATH_PUBLISHED` 事件时计算。
- 预约失败成本在 `RESERVATION_TIMEOUT` 事件时计算。
- 随机超时记录数量但金额为 0。
- `event_id` 已存在时拒绝重复入账。
- `RealizedLedger` 拒绝 `PENDING_AT_HORIZON`、`terminal_soc_value`、`outside_delivery_value`；这些量没有真实发生时刻，只在当前 MPC 目标分解中出现。

`reward_for_interval(q)` 固定返回 `I_actual - C_ch - C_adj - C_fail`，不加入等待成本或随机流失成本。

### 3.5 新增 `src/reference_rollout.py`

- `ReferencePathBuilder`：在 Mock 功率生成前，从日前路径、最近发布剩余路径、模拟实时位置/SOC/ETA 和候选网络构造或修复参考路径。
- `ReferenceRolloutContext`：只包含求解前已知状态、参考路径、预测请求、价格和时间特征。
- `ReferenceRollout`：用 `MockRLProvider` 的确定性请求功率和 `ContinuousEventEngine` 推进 H 个区间，允许预约软失败。
- 未到用户可复用上一轮内部参考；失效时回到日前路径并按当前已知信息修复。
- 在途用户优先使用最近发布剩余站序。
- 参考需求特征从固定参考事件和参考有效到达链汇总，不从本轮待求 `y` 生成。

### 3.6 六站模拟输入与 Mock 输出契约

`get_default_parameters()` 和 `generate_mock_data()` 构成当前唯一外部输入源，并明确标记 `data_source="synthetic"`：

- 6 个站点，ID 为 `0..5`，位置固定为 `[80,180,280,380,480,580]` km，每站 5 个槽位。
- O-D 0：入口 0 km、出口 430 km、沿线站点 0--3；O-D 1：入口 0 km、出口 680 km、沿线站点 0--5。这样端到端实例实际覆盖全部 6 站。
- 默认时间轴固定为 `interval_hours=1.0`、`num_periods=12`、`horizon=4`，即 12 h 模拟日；默认预约入口时刻限制在 `[0,2]` h，使最远 O-D 的出口 ETA 不晚于模拟日终点。上述值全部落入输入快照，不从环境变量读取。
- 初始逐槽 SOC 使用 `[1.0,0.9,0.6,0.4,0.2]` 并复制到 6 站；价格、服务价和逐区间能量上限生成 6 行模拟序列。
- 默认生成 6 个模拟预约；其中一个 coverage anchor 是 O-D 1、入口时刻 0 h、入口 SOC 1.0，用于验证最远路线。预测和实际随机请求分别使用由 `SeedSequence(seed)` 派生的独立 `PCG64` 流，以 `p=0.20` 的 Bernoulli 分布为每站每区间生成 0--1 个请求，并以确定性补点保证全天每个站至少各有一个预测和实际请求。实际流不得由预测流的结果派生。
- 其余默认模拟参数不留“实现时决定”：车速 75 km/h、续航 300 km、电池 100 kWh、效率 0.95、逐槽上限 60 kW、逐站区间能量上限 240 kWh、`max_wait_hours=0.25`、`min_swap_spacing_km=100`、`path_adjustment_penalty=1`、`reservation_failure_penalty=1000`、`terminal_value_weight=1`、`time_epsilon=1e-9 h`、`FeasibilityTol=1e-8`。12 段电价为 `[0.35,0.35,0.65,1.10,1.10,0.65,0.40,0.35,0.35,0.65,1.10,0.65]` 元/kWh，各站复制；换电服务价固定 1.2 元/kWh。所有数值只用于打通代码并标记 synthetic。
- 预约的日前/实际入口时刻与 SOC、每轮实时位置、车辆 SOC、节点 ETA 均随 seed 一次性生成并写入输入快照。ETA 是外生模拟字段，不在运行时加入随机交通模型。
- `MockRLProvider` 以“低 SOC 槽优先”的确定性规则生成 H 步 `P_hat`，随后调用统一动作投影。
- Mock 终端 SOC 边际价值沿用当前线性形式：窗口平均模拟电价乘电池容量；域外交付价值沿用 `lambda_i * (rho-1)`。
- 每份输入和信号保存 `schema_version`、`seed`、`generator_version`、`data_source="synthetic"` 和 `signal_source="mock"`。
- 完整 `SyntheticScenario` 可保存全天 ground truth，但只由真实执行器持有；`observation_at(now)` 仅暴露 `arrival_time<=now` 的实际到达、截至 now 的车辆快照/历史和独立预测。Mock provider、reference builder 和 MPC 的类型签名只接收该受限视图，禁止接收 scenario/oracle。
- 任何端到端结果都必须明确标为模拟结果，不得与真实运营数据混用。

## 4. 分阶段实施

每一阶段应独立提交；阶段内先写失败测试，再改实现。不得等全部代码完成后一次性补测试。

### P0-0：冻结基线和依赖

修改/新增：

- 新增 `environment.yml`，只固化本轮需要的 Python 3.10.19、NumPy 2.2.6 和 gurobipy 13.0.2；不加入 Gymnasium、PyTorch、Tyro 或 TensorBoard。
- 新增 `tests/fixtures/legacy_seed42_summary.json`，只保存现有 seed 42 的业务汇总和 schema，不保存求解时间。

执行：

1. 运行当前 8 个测试并记录通过结果。
2. 记录 Python、Gurobi 和依赖版本。
3. 对当前 seed 42 结果做 JSON 往返和确定性检查。
4. 不把旧的整时段行为写成新模型的永久 golden test。

验收：

`conda run -n py310 python -m unittest discover -s tests -v` 在持有 Gurobi 许可证的系统用户下通过。

### P0-1：统一时间、参数和领域状态

修改：

- `data_generation_test/parameter.py`
- 新增 `src/time_grid.py`、`src/domain.py`
- 新增 `tests/test_time_grid.py`、`tests/test_domain_state.py`

具体任务：

1. `BusinessParameters.delta_hours` 改为 `interval_hours`。
2. 新增 `max_wait_hours` 和 `station_energy_limit_kwh[station][interval]`。
3. `max_wait_hours` 是模拟参数快照的必填字段，不提供静默默认值；默认六站样例由生成器显式写入 0.25 h，并在输出元数据标记为 synthetic。
4. 默认站点改为 6 站 × 5 槽；站点 ID、位置和两个 O-D 按第 3.6 节固定，所有按站二维数组同步扩为 6 行；默认时间轴同时改为 12 个 1 h 区间、预测域 H=4。
5. 默认测试实例令 `station_energy_limit_kwh[i][q] = 240 * interval_hours`，只保持现有测试容量量级，不再把 240 解释为站级瞬时功率上限。
6. 保留 `slot_power_limit_kw`；删除 `station_power_limit_kw`、`full_soc_tolerance`、`power_needed_to_full_kw`、`full_power_tolerance_kw` 和容差补足描述。
7. 通用 `validate()` 按传入的 `num_stations` 校验连续 ID、位置、能量矩阵形状、正值、价格长度、站点严格下游排序和单向 DAG 前提，不把 6 硬编码为所有测试实例的下限；只由默认生成器和 S25 断言 6 站。
8. 所有请求由连续到站时刻派生区间，不持久化重复的 `arrival_period` 真相字段。
9. 建立 future/enroute/queue 状态和全日唯一事件 ID。
10. mock、日前计划、运行结果的 loader 遇到旧 schema 时明确要求重新生成。

验收：

- `H` 个区间准确覆盖 `[t_ell,t_{ell+H})`。
- 恰在区间边界的到站归入新区间。
- 状态 JSON 往返后 deadline 和队列顺序不变。
- 默认参数的 `num_stations==6`，所有站级数组第一维均为 6，候选网络验证确认 O-D 1 覆盖站点 0--5。
- `rg -n "delta_hours|full_soc_tolerance|full_power_tolerance_kw" data_generation_test src run_mpc.py` 只允许在尚未迁移阶段的显式待办注释中出现；P2 清理后必须为零。

### P0-2：实现动作投影、连续充电和队列执行器

修改：

- 新增 `src/event_engine.py`、`src/accounting.py`
- `data_generation_test/rl_data.py` 中的 Mock 请求功率生成改用统一投影
- 新增 `tests/test_power_projection.py`、`tests/test_event_engine.py`、`tests/test_accounting.py`

具体任务：

1. 实现逐槽裁剪和站级区间能量同比缩放；同一输入必须确定性输出。
2. 实现充满时刻解析计算；`P=0` 时不产生伪充满事件。
3. 实现满电停充、服务后恢复和同槽多次服务。
4. 实现两类队列、跨边界携带、deadline 和固定同刻顺序。
5. 实现最小编号槽位分配以及同刻多服务逐次写回。
6. 实现上游预约失败后下游请求失活的通知接口。
7. ledger 以事件发生时刻记账并拒绝重复 ID；服务等待时长由 `service_time-arrival_time` 派生，超时请求记录的等待时长必须等于 `max_wait_hours`，不能写成 0。
8. 实际到站未结束请求以 `WAITING` 跨轮；未到预约仍在 future/enroute；`PENDING_AT_HORIZON` 只从预测结果派生，禁止塞入真实等待队列或 ledger。
9. 恰在右端的充满/到站/服务/超时保存必要的 `completion_due_at` 或原请求状态，并由下一轮开始按“充满→到站→分配→超时”执行一次；严格早于右端（包括非常接近右端）的事件必须保留在当前轮。

阶段验收测试：

- 到站前已满时立即服务。
- deadline 恰好充满时先服务。
- deadline 后充满时超时。
- 随机已到而预约未到时不扣留电池。
- 两类同时等待时预约优先，类内稳定排序。
- 同槽同区间二次服务之间真实达到 SOC 1。
- 跨边界等待不重置身份和 deadline。
- 服务价格跨区间时取实际服务区间。
- 上期到站、本期服务只记一次收入。
- 同一超时事件多次提交 ledger 时第二次被拒绝。
- 终端 SOC/域外价值提交 ledger 时立即报类型错误。
- 预测右端的充满和 deadline 本轮不产生 realized 事件，下一轮恰好产生一个服务或超时结果且只记账一次。

### P0-3：构造在线剩余网络和发布基准

修改：

- `data_generation_test/candidate_network.py`
- 新增 `src/path_state.py`
- 新增 `tests/test_remaining_network.py`、`tests/test_path_publication.py`

保留 `generate_candidate_network`、`get_feasible_arcs` 和离线 JSON 的四步结构；新增：

`build_remaining_network(network, params, reservation_state, now) -> RemainingNetwork`

具体任务：

1. future 用户从入口和日前有效 SOC 构造完整网络。
2. enroute 用户删除已执行节点、已执行弧和当前位置上游节点。
3. 从 `virtual_origin` 到下游站/出口补首段弧，使用实时位置、实时车辆 SOC 和求解前已知 ETA 筛选。
4. 后续站间弧继续复用离线骨架和满电出发 SOC。
5. `min_swap_spacing_km` 以入口或 `last_actual_swap_km` 为基准检查。
6. 当前站正在等待的预约请求强制保留，路径选择不能取消。
7. `reference_visits` 在当前候选实体站点与参考未执行站点并集上补零。
8. `publish_if_changed` 比较剩余站序而非虚拟起点；只有 enroute 变化时生成发布事件和更新参考。

验收：

- 虚拟起点移动但剩余站序不变，`d=0` 且不发布。
- enroute 剩余站序变化，恰发布一次，下一轮以新路径为参考。
- future 多轮内部改路仍相对日前初始路径。
- 最近换电位置不随虚拟起点移动。
- 等待中的当前站不被剩余网络删除。

### P1-4：升级 mock 数据和日前严格接纳

修改：

- `data_generation_test/rl_data.py`
- `src/dayahead_plan.py`
- 新增 `tests/test_mock_schema.py`、`tests/test_dayahead_continuous.py`

数据 schema：

- mock schema 升为 2。
- 随机预测与实际请求保留 `request_id/arrival_time/arrival_soc`，deadline 由参数派生。
- `SyntheticScenario` 保存供真实执行器使用的完整模拟轨迹；新增 `observation_at(now) -> ObservationView`，只返回当前时点已揭示的位置、车辆 SOC、连续 ETA、实际历史和独立预测，不得向 MPC/Mock/reference 泄漏未来真实随机请求或未来真实车辆轨迹。
- 删除用于规避“首站行驶时间必须大于一个区间”的 `timing_workaround`。
- 生成器固定读取六站默认参数，生成 6 个预约和每站 0--1 个随机请求，并在 schema 中断言全部站级数组第一维为 6。
- 电价、服务价、能量上限、位置、SOC、ETA、预测需求和实际需求全部写入带 seed 的完整模拟输入快照；`run_mpc.py` 将 ground truth 封装在只供真实执行首区间查询的 oracle 中，优化链只能获得 `ObservationView`，运行时不得再从环境或真实数据文件补字段。

日前计划：

1. 保留提交顺序、库存余量排序、靠近出口的并列规则、下游可达性和完整路径拒绝。
2. 每次试排调用 `ContinuousEventEngine`，不再维护独立的整时段 `_simulate_inventory` 物理规则。
3. 退回电池同一时刻写入，并立即按当前区间请求功率充电。
4. 预约接纳仍严格要求预计到站时已经有满电池；不得用等待窗扩大接纳。
5. 日前路径写为 `initial_published_path`，成为 future 用户固定调整基准。
6. 日前计划 schema 升为 2，库存轨迹保存连续事件日志和逐区间左/右边界快照。

验收：

- 日前与在线事件引擎对同一输入产生相同 SOC 和服务日志。
- 预计到站无满电但 deadline 前能充满的预约仍被拒绝。
- 同区间退回电池可以继续充电，且只有真实充满后才能再次交付。
- 被拒预约不残留路径、请求或库存写入。
- 在 `now` 前后各放一个实际随机到达和车辆轨迹点，断言 `ObservationView` 只暴露前者；函数签名层面不能把 `SyntheticScenario` 传给 Mock/reference/MPC。

### P1-5：实现 reference rollout 和 Mock 终端信号接口

修改：

- `data_generation_test/rl_data.py`
- 新增 `src/reference_rollout.py`
- 新增 `tests/test_reference_rollout.py`

接口替换：

`RLProvider.get_signals(params, period_ell, horizon, soc_obs)`

替换为：

`RLProvider.build_signals(context: ReferenceRolloutContext) -> RLSignals`

`RLSignals` 保留逐槽请求功率、终端 SOC 边际价值和域外交付价值；新增稳定 `pending_delivery_id`，仅供单次 MPC 目标内部防重复估值。该 ID 不加入 `RollingState.accounted_event_ids`，下轮必须基于新的 reference 终态重新计算。

具体任务：

1. ReferencePathBuilder 只使用求解前已知位置、SOC、ETA、日前路径、最近发布路径和候选网络。
2. `MockRLProvider` 按低 SOC 槽优先规则产生 H 个区间请求功率，并统一投影；本轮不存在神经网络 actor。
3. reference 执行用共享事件引擎，允许预约软失败。
4. Mock 终端参数在 reference 终态上按第 3.6 节线性公式计算；`lambda_soc` 只模拟逐槽 SOC 一阶边际价值，本轮不存在神经网络 critic。
5. 尚未服务且退回 SOC 未进入终端库存的预约交付进入现有 `outside_delivery_value` 近似。
6. 对每个路径激活且上游仍存活的未来交付定义 `eligible_delivery=1`；必须满足 `incorporated_in_terminal_state + pending_outside = eligible_delivery`。域内已服务并写入预测终端物理状态时前者为 1，尚未服务时后者为 1，上游失败或路径未激活时两者均为 0；两者都是单次求解变量，不是 realized ledger 事件，既不允许双计，也不允许漏计。
7. `MockRLProvider` 是本轮唯一实现，输出必须携带 `signal_source="mock"`，不得伪装为训练模型结果。
8. reference 和 Mock provider 只接受 `ObservationView` 与当前 `RollingState`；用静态类型和运行时字段白名单双重阻止 ground truth 泄漏。

验收：

- 对 Mock provider 输入做快照，确认不含本轮 `MPCResult`、`y`、`alpha` 或 `z`。
- 参考、日前和真实执行对相同事件使用相同优先/等待/充电规则。
- 每个 eligible 未来交付恰好落入 incorporated 或 pending_outside，一项且仅一项。
- 参考软失败不会造成 rollout 异常退出。
- terminal/pending 目标项不改变真实 ledger，也不把未到请求转成 `WAITING`。

### P1-6：重写事件位置驱动的 MPC（待实施）

修改：

- `src/mpc_model.py`
- 新增 `tests/test_mpc_event_model.py`、`tests/test_mpc_replay.py`

保留：

- `MPCError`、`MPCInputError`、`MPCInfeasibleError`、`MPCNoSolutionError`。
- Gurobi 参数、路径流平衡、`y/x_A/d` 目标骨架、入弧预约候选事件、有效到达链和目标分项提取。

删除：

- `_EVENT_DEC/_EVENT_FIX` 和 `FixedCommitment`。
- `is_new_arrival` 对路径可调性的控制。
- `P` 作为 MPC 自由变量、`g/F` 的整时段库存计数、`pow_defer/pow_fill` 容差补足。
- `match_ready` 的“每槽每时段最多一次”约束。
- `S_pre` 作为整个时段统一服务前快照。
- 预测终态必须等于实际终态的交叉检查。

新的构模顺序：

1. 在建模前固定枚举所有 `CandidateRequest`。预约事件由 `(p,k,j,i)` 和可行弧生成，激活量为 `y[j,i]`；随机请求存在量固定为 1。
2. 预约有效到达使用上游成功链；已在站等待请求的有效性固定为 1，不能被改路取消。
3. 每个请求建立 `served/failed/pending`：
   - 预约：`served + failed + pending = effective_arrival`；
   - 随机：`served + lost + pending = 1`，其中 `lost = 1-served-pending`，不进入目标成本；
   - `deadline >= t_end` 时，该 deadline 事件尚未进入半开预测域：预约固定 `failed=0`，随机固定 `lost=0`，未服务请求必须 `pending=1`；
   - `deadline < t_end` 时 deadline 已完整进入预测域，固定 `pending=0`；
   - 特别地，`deadline == t_end` 由下一轮处理，不得在当前轮提前失败。
4. 服务请求建立连续 `service_time`，物理服务窗为 `[max(arrival_time,t_start), deadline]`。当前轮语义严格为 `service_time<t_end`，但 MILP 不用人为缩小时间窗：先以闭上界建模并在解提取后由 `TimeGrid.compare_to_boundary` 判定；若某个当前服务恰等于 `t_end`，将其视为 pending，加入该服务位置/匹配组合的 no-good cut 后重解，直至首区间可被执行器重放。严格早于 `t_end` 的服务不得被 cut 或改写。
5. 区间归属用 `interval_member` 编码统一的半开规则：内部边界事件归入后一外层区间，预测右端没有当前区间成员。收入、能量和事件重放都读取同一成员变量/`TimeGrid` 结果。
6. 每站预分配服务事件位置和充满事件位置。服务位置上限为该站候选请求数；每个槽的初始充电 episode 以及每次被服务后产生的 episode 至多对应一个充满事件，因此充满位置总上限为 `num_slots + candidate_requests`。未激活位置只能在稳定序列尾部。
7. 用 `event_time/event_kind/event_active`、`interval_member`、`slot_match`、`full_slot_match` 和事件前后 `soc/ready` 建立包含“固定区间边界、充满、到站/服务、超时”的站内序列。同刻优先级固定为充满→到站→队列分配→超时。
8. 每两个相邻断点建立 `segment_duration`、`charge_on[i,b,m]` 和 `charged_duration[i,b,m]`。用四条标准 binary×bounded-continuous 线性化使 `charged_duration=segment_duration` 当且仅当该槽未满且 episode 激活，否则为 0；`P_hat` 是参数，因此 SOC 增量与成本 `P_hat*charged_duration` 均保持线性。每个 `M_duration` 取当前区间长度，不用全日常数。
9. 充满事件要求对应 episode 的事件前 SOC 精确达到 1，并把 `ready` 置 true；事件后所有后续充电段关闭，直到服务把低 SOC 电池换入。若 episode 在预测右端达到 1，只写终端 `soc=1, ready=false, completion_due_at=t_end`，不激活当前轮充满/服务事件。
10. 服务事件要求被匹配槽位 `ready=true`，随后写入 `return_soc`、`ready=false` 并激活该槽下一充电 episode；这样同槽可重复服务，但每两次服务之间必须存在匹配的充满事件。
11. 每站每区间的 `sum(P_hat*charged_duration)` 同时用于实际电量和充电成本，并校验不超过 `station_energy_limit_kwh[i][q]`；电价或请求功率跨内部区间边界时必须切段，不能用单一平均价覆盖。
12. 预约优先、类内 FCFS、最小槽位和“有资源立即服务”作为硬约束，不用目标函数打破平局；同刻规则与执行器完全一致。
13. 终端状态取 `t_end^-`；当前轮结果区分真实携带 `WAITING`、未到 future/enroute 和求解态 `PENDING_AT_HORIZON`，三者不得互相覆盖。

Big-M 规则：

- 每条时间 indicator/Big-M 约束按该约束左右两侧变量及常数的实际上下界计算最小合法 `M_time`；不得把 `t_end-t_start` 当作全局常数。预测域外预约交付完全剥离出域内事件序列，只保留激活/存活与终端近似项，不参与服务时间比较。
- SOC 约束只用 `M_soc = 1`。
- 计数约束只用该站固定候选请求数。
- 禁止任意 `1e6`；每个 indicator/Big-M 约束在代码旁记录变量上下界来源。

目标：

- `income_reservation`、`income_random`：按预测服务时刻所在价格区间。
- `charging_cost`：按事件段持续时间积分。
- `adjustment_cost`：future 相对日前路径、enroute 相对最近发布剩余路径。
- `reservation_failure_cost`：只对预测域内真正到 deadline 的预约失败。
- `terminal_value`：Mock 终端 SOC 边际参数加未写入域内物理状态的 pending 域外交付近似，并满足 P1-5 的 eligible 恰一关系。
- 不增加等待、随机流失或隐含等待最小化项。

模型验收：

- 每个小实例求解后，把首区间决策交给 `ContinuousEventEngine` 重放。
- 重放的服务 ID、服务时刻、槽位、终态 SOC、`ready/completion_due_at` 和真实 WAITING 队列与模型解一致；求解态 pending 单独比较。
- 构造 `service_time=deadline=t_end` 的用例，断言当前轮不服务，下一轮按“充满→分配→超时”恰好得到一个服务或超时结果，realized ledger 也恰好一条。
- 分别重放“区间内提前充满后停充”“服务换入后恢复充电”“内部价格边界切段”“预测右端恰好充满”四个最小 MPC 实例，SOC、ready、能量和成本均与执行器一致。
- 目标各分项由独立 Python 记账器复算，误差在求解器数值容差内。
- 模型统计记录变量数、约束数、候选事件数、状态、runtime、MIP gap 和 best bound。
- 默认可复现配置固定 `Seed=input.seed`、`Threads=1`、`MIPGap=0`、`FeasibilityTol=1e-8` 且不设 `TimeLimit`；候选请求、事件位置、弧和变量统一按稳定 ID 排序构造。诊断运行可覆盖这些参数，但不能冒充确定性验收结果。

### P1-7：重写滚动编排、发布和真实记账（Mock 候选路径搜索闭环已实施；联合事件位置 MILP 待 P1-6）

修改：

- `run_mpc.py`
- 新增 `tests/test_rolling_state.py`、`tests/test_rolling_accounting.py`

删除 `_execute_first_stage`、`_path_swap_events` 和 commitments 字典。每轮固定执行：

1. 从上一轮终态恢复逐槽 `soc/ready/completion_due_at`、真实 WAITING 队列、future/enroute 状态和最近发布剩余路径；不得从上轮求解态 pending 恢复物理队列。
2. 只调用 `SyntheticScenario.observation_at(now)` 读取当前已知的实时位置、车辆 SOC、ETA、随机预测、电价和实际历史；完整 ground truth 留在 oracle 内部。
3. 构造/修复参考路径。
4. 调用 `MockRLProvider` 生成请求功率和 Mock 终端参数，并验证 `signal_source="mock"`。
5. 构造剩余网络和 `MPCWindowInput`，求解完整 H 区间。
6. 对 enroute 用户调用 `publish_if_changed`；future 内部路径不发布。
7. 只把首区间请求功率和已发布路径交给真实 `ContinuousEventEngine`；由 oracle 仅揭示首区间内实际发生的到站/轨迹，优化对象从不接触后续 ground truth。
8. 将服务、超时、充电和发布事件提交 `RealizedLedger`。
9. 更新实时车辆状态、已执行前缀、最近真实换电位置、队列和逐槽 SOC。
10. 保存下一轮真实状态；terminal/pending 目标项不写 ledger 或物理状态。正常预测偏差只体现在真实状态，不触发停止或终态相等断言。

迟到但尚未真实进入高速的预约必须继续留在 future；不能仅因日前入口时刻早于 `now` 而删除。完成、真实超时或明确退出的用户才离开活动集合。

运行结果 schema 升为 3，逐轮至少保存：

- `time_start/time_end`、求解状态与 MIP 诊断；
- reference 路径和 Mock 请求功率；
- 预测服务/失败/pending；
- 实际服务时刻、超时、携带队列和发布；
- 充电分段、电量、逐槽边界 SOC；
- realized ledger 和 reward 分项；
- future/enroute/completed/failed 用户状态。

验收：

- 等待跨轮不重新计时。
- 上游预约超时后，下游事件取消且失败只记一次。
- 实际随机到达与预测不同时继续执行。
- 路径发布、服务和失败在多轮输出中各有唯一 ID。
- 同一 seed 在默认 `Threads=1/MIPGap=0` 配置下重跑两次，将列表按稳定 ID 排序并以 `json.dumps(sort_keys=True)` 规范化；剔除 runtime、bound 等诊断字段后 SHA-256 必须一致。
- 在 now 之后各放置一个实际到达和轨迹变化，断言它们既不出现在 `ObservationView`，也不出现在 Mock/reference/MPC 输入快照中。

### P2-8：清理旧逻辑、文档和模拟接口（待实施）

修改：

- 更新模块 docstring、`todo.md` 和命令示例。
- `plan_mpc.md`、`plan_rl.md` 保留为历史记录，并在顶部指向本计划；不再作为实现依据。
- 仅在核心验收完成后删除已无调用的旧 dataclass、函数和测试。
- 明确标注 `RL/ppo.py` 和 `main.py` 未接入本轮系统；不得在运行结果中写“trained actor”或“critic checkpoint”。

静态清理目标：

- 不再出现 `FixedCommitment`、`_EVENT_FIX`、`is_new_arrival`、`full_soc_tolerance`、`full_power_tolerance_kw`、`pow_defer`、`pow_fill`。
- 不再出现“先充完整时段再服务”“退回电池下一时段充电”“随机请求当期立即拒绝”的代码分支。
- `station_power_limit_kw` 被逐区间 `station_energy_limit_kwh` 替代。
- 实际执行路径中不读取预测 `MPCResult.assignments` 作为真实发生事件。
- 默认端到端输入和结果均声明 6 个站、`data_source="synthetic"`、`signal_source="mock"`。
- 增加六站端到端回归：至少一个真实或预测事件触达每个站点，且最远 O-D 能在 12 h 模拟日内走完。

本阶段不生成论文性能结论，不修改 `paper/main.tex`，不接入 `data_generation_rl` 的真实 NIO 数据。

## 5. 自动化验收矩阵

| ID | 场景 | 测试文件和测试名 | 必须断言 |
| --- | --- | --- | --- |
| S01 | 到站前电池已满 | `test_event_engine.py::test_ready_before_arrival_serves_immediately` | 服务时刻等于到站时刻，按队列分配 |
| S02 | deadline 恰好充满 | `test_event_engine.py::test_charge_complete_at_deadline_precedes_timeout` | 先服务，无超时 |
| S03 | deadline 后充满 | `test_event_engine.py::test_charge_after_deadline_times_out` | deadline 超时，不借区间净库存 |
| S04 | 随机已到、预约未到 | `test_event_engine.py::test_future_reservation_does_not_hold_inventory` | 随机立即服务 |
| S05 | 两类同时等待 | `test_event_engine.py::test_reservation_priority_and_stable_fcfs` | 预约优先，类内按时间和 ID |
| S06 | 同槽区间内二次服务 | `test_event_engine.py::test_same_slot_reused_only_after_recharge` | 中间存在完整充电段且 SOC 到 1 |
| S07 | 等待跨滚动边界 | `test_rolling_state.py::test_waiting_request_identity_survives_boundary` | ID、arrival、deadline 不变 |
| S08 | 等待窗跨预测终点 | `test_mpc_event_model.py::test_deadline_outside_horizon_is_pending` | pending=1，failed=0 |
| S09 | future 多轮内部改路 | `test_path_publication.py::test_future_reference_stays_dayahead` | 基准不被内部路径覆盖 |
| S10 | 虚拟起点移动、站序不变 | `test_path_publication.py::test_origin_motion_without_sequence_change_is_free` | d=0、无发布事件 |
| S11 | 上游预约超时 | `test_rolling_accounting.py::test_upstream_timeout_cancels_downstream_once` | 下游失活、一次失败成本 |
| S12 | 实际随机偏离预测 | `test_rolling_state.py::test_actual_prediction_mismatch_keeps_running` | 按真实事件更新，无强制终态相等 |
| S13 | 上期到站、本期服务 | `test_accounting.py::test_carried_service_income_booked_at_service` | 只在服务区间计一次收入 |
| S14 | 未来交付恰一估值 | `test_reference_rollout.py::test_eligible_delivery_is_accounted_exactly_once` | eligible 时 terminal-incorporated/pending 恰有一项为 1，且均不进入 realized ledger |
| S15 | 预测终点事件 | `test_time_grid.py::test_event_at_prediction_end_moves_to_next_round` | 本轮不执行、不记账；下一轮恰好服务或超时一次 |
| S16 | 动作投影 | `test_power_projection.py::test_projection_invariants` | 逐槽上限和站级能量均满足 |
| S17 | SOC 守恒 | `test_event_engine.py::test_piecewise_energy_conservation` | 每段 SOC 变化等于积分 |
| S18 | 最小槽位 | `test_event_engine.py::test_smallest_ready_slot_is_deterministic` | 不跳过较小可用槽号 |
| S19 | Mock 信号无解回流 | `test_reference_rollout.py::test_mock_context_has_no_mpc_solution` | 输入无 `y/alpha/z` 或上轮求解态 pending |
| S20 | 日前/在线一致 | `test_dayahead_continuous.py::test_shared_engine_matches_online` | 同输入产生同物理轨迹 |
| S21 | incumbent 可重放 | `test_mpc_replay.py::test_first_interval_replay_matches_solution` | 服务、队列、SOC、成本一致 |
| S22 | 多轮预测失败不重复 | `test_rolling_accounting.py::test_predicted_failure_is_not_realized_twice` | 只有真实 deadline 事件入账 |
| S23 | 迟到但尚未进入 | `test_rolling_state.py::test_late_not_entered_reservation_remains_future` | 用户仍在 future 且无虚假服务/失败 |
| S24 | 超时等待统计 | `test_accounting.py::test_timeout_wait_equals_patience` | 等待时长等于 `max_wait_hours`，不记为 0 |
| S25 | 六站默认实例 | `test_mock_schema.py::test_default_instance_has_six_stations` | ID 0--5、位置、SOC、价格和能量数组均为六站 |
| S26 | 外部参数全模拟 | `test_mock_schema.py::test_all_external_inputs_are_seeded_and_labeled` | 输入可由 seed 重建，元数据为 synthetic/mock，无真实数据依赖 |
| S27 | 六站端到端模拟 | `test_rolling_state.py::test_six_station_synthetic_run_uses_mock_signals` | 12 h 内六站均被事件覆盖，最远 O-D 完成，输出只含 synthetic/mock 来源 |
| S28 | MPC 提前充满停充 | `test_mpc_replay.py::test_charge_stops_after_internal_full_event` | 满电后功率/成本为 0，SOC 与执行器一致 |
| S29 | MPC 服务后恢复充电 | `test_mpc_replay.py::test_swap_starts_new_charge_episode` | 换入 SOC 写回后同区间按 P_hat 恢复，episode 唯一 |
| S30 | MPC 内部价格边界 | `test_mpc_replay.py::test_charge_cost_splits_at_price_boundary` | 两侧 duration×power×price 分段复算一致 |
| S31 | 右端恰好充满 | `test_mpc_replay.py::test_full_at_horizon_end_is_carried_once` | 本轮 ready=false/completion_due；下一轮先充满事件后恰一结果 |
| S32 | 未来 ground truth 屏蔽 | `test_mock_schema.py::test_observation_view_hides_future_actuals` | now 后实际到达/轨迹不进入 Mock、reference 或 MPC 输入 |
| S33 | 同 seed 确定性 | `test_rolling_state.py::test_seeded_business_json_is_reproducible` | 两次规范化业务 JSON 的 SHA-256 相同 |

## 6. 执行命令与最终门禁

从仓库根目录执行。Gurobi 求解测试必须在许可证对应的系统用户下运行。

```powershell
conda env update -n py310 -f environment.yml
```

### 6.1 语法和纯逻辑测试

```powershell
conda run -n py310 python -m py_compile `
  data_generation_test/parameter.py `
  data_generation_test/candidate_network.py `
  data_generation_test/rl_data.py `
  src/domain.py src/time_grid.py src/event_engine.py src/accounting.py `
  src/path_state.py src/reference_rollout.py src/dayahead_plan.py `
  src/mpc_model.py run_mpc.py

conda run -n py310 python -m unittest discover -s tests -v
```

### 6.2 数据生成和端到端

```powershell
New-Item -ItemType Directory -Path .\tmp -Force | Out-Null
$verifyDir = Join-Path (Resolve-Path .\tmp) ('code-v3-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $verifyDir | Out-Null

conda run --no-capture-output -n py310 python data_generation_test/candidate_network.py `
  --output (Join-Path $verifyDir 'candidate_network.json')
conda run --no-capture-output -n py310 python data_generation_test/rl_data.py --seed 42 `
  --output (Join-Path $verifyDir 'mock_rl_data.json')
conda run --no-capture-output -n py310 python src/dayahead_plan.py --seed 42 `
  --network (Join-Path $verifyDir 'candidate_network.json') `
  --mock-data (Join-Path $verifyDir 'mock_rl_data.json') `
  --output (Join-Path $verifyDir 'dayahead_plan.json')
conda run --no-capture-output -n py310 python run_mpc.py --seed 42 `
  --network (Join-Path $verifyDir 'candidate_network.json') `
  --mock-data (Join-Path $verifyDir 'mock_rl_data.json') `
  --plan (Join-Path $verifyDir 'dayahead_plan.json') `
  --output (Join-Path $verifyDir 'mpc_run_result.json')
```

要求：

- 当前 Mock 路径搜索链路的所有轮次应为 `EVENT_PATH_ENUM_REPLAY` 且首区间可重放；只有未来完成 P1-6 联合事件位置 MILP 并获得求解器证明后，正式优化验收才允许要求 `OPTIMAL`。
- 无 SOC 越界、负时长、重复事件 ID、队列丢失或路径断裂。
- 运行结果 JSON 保存/加载往返一致。
- 重新运行 seed 42 后业务字段一致；`solve_time_sec/MIPGap/best_bound` 可不同。

### 6.3 静态残留检查

```powershell
rg -n -g "*.py" "FixedCommitment|_EVENT_FIX|is_new_arrival|full_soc_tolerance|full_power_tolerance_kw|pow_defer|pow_fill|station_power_limit_kw" `
  data_generation_test src run_mpc.py RL tests
```

最终应无命中；若测试名称需要描述被删除的旧行为，改用不包含旧 API 名称的表述。

### 6.4 Mock 信号与六站冒烟

在按 `environment.yml` 创建或更新 `py310` 后执行：

```powershell
conda run -n py310 python -m unittest data_generation_test.test_mock_schema -v
conda run -n py310 python -m unittest discover -s tests -p "test_reference_rollout.py" -v
conda run -n py310 python -m unittest discover -s tests -p "test_continuous_integration.py" -v
```

要求：Mock 输出维度为 `[6][5][H]`，动作投影后满足逐槽和逐站区间能量约束；六站端到端运行完整结束，输入元数据分别标为 `data_source="synthetic"` 和 `signal_source="mock"`，realized ledger 可由事件日志逐项重算。

## 7. 实施顺序和提交门

严格按以下依赖推进：

```text
P0-0 基线
  └─ P0-1 时间/领域状态
       ├─ P0-2 连续事件与记账
       └─ P0-3 剩余网络与发布
P0-2 + P0-3
  └─ P1-4 日前接纳
       └─ P1-5 reference rollout
            └─ P1-6 事件位置 MPC
                 └─ P1-7 滚动真实执行
                      └─ P2-8 清理与总验收
```

每个阶段的提交门：

1. 新增测试在旧实现上按预期失败。
2. 完成该阶段最小实现。
3. 新测试和仍适用的旧测试全部通过。
4. 更新 schema/docstring/示例，禁止代码与文档两套口径。
5. 只提交本阶段文件，不覆盖工作区中已有的无关修改。

若 P1-6 的事件位置模型在最小实例上无法由执行器重放，则停止进入 P1-7；不得用放宽优先级、重新加入 SOC 容差补足或恢复每槽每区间一次限制来换取可解性。应先定位候选事件、时间界或 indicator 约束错误。

## 8. 明确不在本计划内

- 不修改 `paper/main.tex`、`paper/reference.bib`、样式、图片或论文 PDF。
- 不生成或声称任何数值性能结论。
- 不新增随机交通模型；ETA 仍是外生已知信息。
- 不增加换电作业时长、服务工位、充电器数量、站级瞬时功率参数。
- 不增加等待成本、随机流失成本或隐藏的等待最小化目标。
- 不把模拟器尚未揭示的未来实际到达泄漏给预测、reference rollout 或 Mock 请求功率生成器。
- 不实现或修改 `RL/ppo.py`、`src/env_bss.py`、actor/critic、Gymnasium 适配、训练循环、checkpoint 或策略评估。
- 不接入 `data_generation_rl` 或任何真实数据抽取流程；本轮外部参数和观测一律来自可复现的模拟快照。
- 不实现完美信息全日上界、正式消融和绘图；这些在核心口径通过后另立实验计划。
