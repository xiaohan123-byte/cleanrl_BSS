# 五分钟滚动换电优化：无终端价值基线

本目录实现 paper_v2/sections/03_model.tex 的离散业务模型，以及第4节的实际收益与状态更新。目标只有换电收入减充电费用、路径调整惩罚和预约失败惩罚；不含终端价值、库存残值、终端最低电量或强化学习动作。

## 运行

环境文件为 environment.yml；需要可用的 COPT 许可证。

求解器使用 COPT 8.0.6（Python 包 coptpy）。environment.yml 通过 pip 安装对应版本，也可在现有环境中执行 `python -m pip install coptpy==8.0.6`。参数 threads、time_limit_sec、mip_gap、feasibility_tol 分别映射到 COPT 的 Threads、TimeLimit、RelGap、FeasTol；整数容差取 feasibility_tol 与 1e-8 中较小值，绝对 gap 停止阈值设为零，避免替代相对 gap 要求。记录的 gap 使用 COPT 报告的相对 gap；实际日志由 --solver-log 控制。

```powershell
conda activate py310
cd ExpresswayBSSs_v2
python run_mpc.py
python -m unittest discover -s tests -v
```

默认读取 configs/baseline.json：6站、每站5槽，5分钟一段，运营144段（12小时），预测48段（4小时），每3段允许调整预约路径。6个预约默认在前2小时进入，每站随机需求为每小时0.2；实际随机请求采用独立种子的泊松过程，预测采用独立的逐轮随机流。它们用于模型联调，运行结果不是论文性能结论。

```powershell
python run_mpc.py --seed 42 --time-limit 30
python run_mpc.py --periods 24 --horizon 12 --output outputs/smoke.json
python run_mpc.py --scenario outputs/mpc_run_result_scenario.json --horizon 24 --output outputs/replay.json
```

--periods 改变生成场景的运营时长；--horizon 改变预测长度，末尾自动截短。加载场景时使用其中的物理参数和价格快照，可以覆盖预测长度与求解器设置，不可同时更换随机种子或运营长度。时间、车辆与需求配置在运行前校验。

COPT 达到时限但已有可行解时执行该解，逐轮记录状态、gap及求解时间；没有可行解时明确报错并停止，输入快照已保存，程序不会悄悄切换为启发式策略。

## 模型和执行约定

- 时间单位小时，功率kW，能量kWh；服务收入按电池容量×(1−退回SOC)×当时服务单价计算，充电成本按电网输入电量计算。
- 在每个决策时刻联合选择路径、请求、电池槽和功率。只能用当时已观测且未过期的请求执行首段换电，随后在整段实施选定恒定功率。同槽每个边界最多交付一块满电电池。
- 区间内到站的请求从下一边界才参与服务；截止时间始终从原始实际到站计算。截止恰在下一边界时先保留服务机会，严格早于下一边界且未服务才计超时。同类请求按严格到站先后排序；同时到站不固定槽位或先后，预约优先于随机。
- 后站到达时间来自前站换电时刻加行驶时间。基线行驶预测为给定恒速；当前等待请求独立保留，前站尚未换电时不预先固定后续实际到站。
- 日前路径只考虑道路和SOC可达性，先最少换电，再按站点位置从出口方向择优。候选网络剪枝不强制SOC档下界可达，也不因可直达出口而删除其它路径。
- 未进入高速的用户保留优化计划，优化惩罚始终相对日前路径计算，不收费、不发布。在途用户相对最近发布的剩余路径计变更；已完成和当前等待站不参与比较。首次进入时发布保留计划，不作为一次改道收费。两类用户在非路径更新轮均冻结各自计划。
- 预约超时只罚一次并取消该用户后续服务；随机超时不罚款。预测目标及未来服务不会进入实际账本。
- 下一轮从实际执行一个时段后的状态开始。仿真结束时不强制结清未到期需求，不将尚未到站或尚未出高速的车辆虚构为失败；最终状态保留未完成记录。

## 文件和接口

| 模块 | 职责 |
|---|---|
| src/parameters.py、configs/baseline.json | 业务、场景和求解参数 |
| src/candidate_network.py、src/dayahead_plan.py | SOC候选网络及日前参考路径 |
| src/scenario.py、src/forecast.py | 合成真值与可见观测隔离、独立预测 |
| src/domain.py、src/time_grid.py、src/path_state.py | 稳定真实身份、跨轮状态、时间与路径发布 |
| src/request_builder.py | 弧请求、当前等待请求和前驱服务依赖 |
| src/mpc_model.py | 唯一的路径/服务/功率联合MILP |
| src/execution.py、src/rolling_runner.py | 首段动作校验与执行、状态推进 |
| src/accounting.py、src/result_statistics.py | 实际账本、独立核算与报告 |
| tests/ | 微型约束案例、边界、预测隔离及集成回归 |

build_window 只接收观测状态和预测；solve_mpc 输出路径、逐时段服务槽位与功率；execute_step 执行首段并返回新真实状态。场景的未来随机真值只供执行环境使用。真实预约请求ID在到站时分配，跨轮不改；预测请求按弧标识，不能替换真实队列中的ID。

## 结果

默认 outputs/mpc_run_result.json 使用 schema_version=4，包含参数快照、日前路径、完整逐轮物理状态、预测、优化结果、实际事件、总账本和最终状态。逐轮状态用 ledger_event_count 引用顶层完整账本，避免重复存储全部历史。每轮都记录预测的终点SOC，但不会将其作为下一轮真实状态。

同目录保存场景、候选网络、日前路径，以及统计JSON、Markdown、分站和分时段CSV。统计器从真实事件重新计算价格、收入和费用，核对功率/SOC、重复事件与汇总，预测收益不参与实际净收益。

旧代码、旧测试及研究数据位于工作区的 archives/ExpresswayBSSs_v2_pre_baseline_20260912_164355/。新版无需旧数据目录、PPO或连续事件模块。本次实现未修改 paper_v2/ 和编辑器配置；实施期间检测到的并行论文更新也予以保留。

接口和参数依据：[COPT Python API](https://guide.coap.online/copt/en-doc/pyapiref.html)、[COPT 参数说明](https://guide.coap.online/copt/en-doc/parameter.html)。切换前的实现与运行结果保存在工作区 archives/ExpresswayBSSs_v2_pre_copt_20260912_172254/，新输出标记 solver_backend=copt。
COPT 切换验证：96 项测试全部通过。使用 `python run_mpc.py --time-limit 2 --output outputs/copt_validation_run.json` 完成六站、144 段、48 段预测域的运行；63 轮证明最优，81 轮使用时限内可行解，最大 COPT 相对 gap 为 19.35%。6 个预约全部完成，实际净收益为 1851.336878，账本、能量、价格、SOC 和站点功率核对通过。该运行用于功能验收，默认每轮时限仍为 30 秒；详细记录见 outputs/verification.json，统计报告见 outputs/copt_validation_run_statistics.md。
