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

默认读取 configs/baseline.json：6站、每站5槽，5分钟一段，运营144段（12小时），预测48段（4小时），每3段允许调整预约路径。6个预约默认在前2小时进入，每站随机需求为每小时0.2；入口SOC按 reservation_entry_soc_range（默认[0.3,1.0]）采样；该范围只控制合成数据，外部预约SOC可在[0,1]内取值，并须有完整可达路径。实际随机请求采用独立种子的泊松过程，预测采用独立的逐轮随机流。它们用于模型联调，运行结果不是论文性能结论。

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
- 日前路径按预约信息中的入口SOC构造，先最少换电，再按各换电站的位置依次优先选择下游站点。候选网络共用满电站间弧，首段按用户本轮已知的位置和SOC连接；等待用户的后续行程按当前站换满电后计算。短弧按换电间距依次剪枝，至少保留一条完整可达路径；既有剩余计划仍可行时保护其全部弧。首段间距从入口或上次实际换电位置计算。到出口的末段不参与间距剪枝，可直达出口不自动删除其它路径。
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

同目录保存场景、共享站间网络、日前路径，以及统计JSON、Markdown、分站和分时段CSV。场景和网络协议分别为 schema_version=2、3；旧分档版本的场景及网络需重新生成。统计器从真实事件重新计算价格、收入和费用，核对功率/SOC、重复事件与汇总，预测收益不参与实际净收益。

旧代码、旧测试及研究数据位于工作区的 archives/ExpresswayBSSs_v2_pre_baseline_20260912_164355/。新版无需旧数据目录、PPO或连续事件模块。当前候选网络与论文第3节使用同一套按用户SOC构造的规则。

接口和参数依据：[COPT Python API](https://guide.coap.online/copt/en-doc/pyapiref.html)、[COPT 参数说明](https://guide.coap.online/copt/en-doc/parameter.html)。切换前的实现与运行结果保存在工作区 archives/ExpresswayBSSs_v2_pre_copt_20260912_172254/，新输出标记 solver_backend=copt。
当前方案B验证：115项测试通过，论文编译结果为 outputs/paper_actual_soc_check/main.pdf。使用 `python run_mpc.py --time-limit 2 --output outputs/actual_soc_validation_run.json` 完成六站144轮、48段预测域验证；77轮证明最优，67轮执行时限内可行解，最大相对gap为29.02%。6个预约全部完成、无预约失败，实际净收益1809.801457。逐轮核对609条选择路径和405条冻结路径，账本、能量、库存和站点功率检查通过。该结果用于功能验证，不能据此判断方案B的收益或求解性能优于分档方案；默认每轮时限仍为30秒。详见 outputs/actual_soc_verification.json。

历史分档版本的COPT切换与下限剪枝修复记录分别见 outputs/verification.json 和 outputs/lower_soc_pruning_verification.json；其中结果与输入协议不代表当前版本。切换到方案B之前的相关文件保存在工作区 archives/ExpresswayBSSs_v2_pre_actual_soc_20260913/。

## 2026-09-14 终端价值及第 5 节实验

本轮按用户授权实现已确认的实验方案。实验入口为 `run_experiments.py`，单场景子进程为 `experiment_worker.py`；原 `run_mpc.py` 与 `configs/baseline.json` 继续用于旧基线回归。

正式采用 `configs/terminal_experiment_feasible_od.json`：既有六站道路、26 个固定 O-D、每站 21 块电池与充电槽、站级功率 960 kW、24 h 运营、6 h 预测。原 30 个 O-D 中，e2(94 km) 向前行驶的四个组合首站位于 255 km，需要至少 53.67% 的入口 SOC，与用户要求的所有入口 SOC >50% 均可达不一致。实际预实验触发后，固定排除这四个 O-D 并统一重算权重和随机需求份额；SOC、预测误差及其独立生成规则不变。原输入与失败证据保存在 `outputs/experiment_20260914/`，修正版使用 `outputs/experiment_20260914_od26/`，未筛选更容易的随机种子。

```powershell
python run_experiments.py --config configs/terminal_experiment_feasible_od.json --output-root outputs/experiment_20260914_od26 --stage all --pilot-workers 3 --deadline "2026-09-14T07:00:00+08:00"
```

`all` 先完成三个独立 24 h 零终端预实验，账本、能量与执行校验成功后，根据完整日耗时冻结正式预算；然后执行零终端/同特征线性/完整 ReLU 核心比较及第 5 节的联合消融、库存价值消融和单因素敏感性分析。`pilot` 只运行预实验；`formal` 运行核心正式比较；`paper` 继续第 5 节范围。正式阶段要求三个预实验均完成。每轮只用当前冻结策略新产生的完整日回报，按验证集实际净收益选择模型，测试集不参与选模。

每个求解器单线程、60 s 限时、相对间隙目标 0.1%；预实验进程并发数另行记录。有限时可行解按实际间隙报告，无可行解停止并保留模型及状态。模型输入使用固定业务尺度，配置指纹允许改变预测长度，拒绝误用不同需求或物理配置下的网络。

完成的五分钟时段先写入 `journal/rounds.jsonl`。中断后可按同样输入恢复，已完成收益不会重复记账，部分轨迹不能生成完整日 MC 标签。独立 `deadline_guard.py` 同时保护求解子进程和控制器训练；截止时终止本任务的进程树，已有记录保留。恢复须明确提供新的含时区截止时间。失败场景保留原种子；修复后可用 `--retry-failed` 重试，不能更换场景顶替。

第 5 节原稿中延长运营、训练集统计归一化及 Gurobi 等旧描述与本轮不同：实际采用 24 h 截断、不追加期末违约结算、固定业务尺度和 COPT。到期失败率与期末未完成预约分别报告。论文结果只能使用完成且核对通过的实验，不能使用实现测试或部分轨迹替代。

本轮报告入口为 `outputs/experiment_20260914_od26/reports/experiment_report.md`，由 `src.experiment_reporting.export_experiment_report(output_root)` 从真实记录生成；同目录 `unfinished_jobs.csv` 列出未完成正式任务、原始状态、进程核验与截止证据，完整日结果才进入正式统计。

运行根目录的 `monitor_samples.jsonl` 保存每分钟的监测快照，用于回溯运行过程；独立监测在本轮 07:00 截止时退出。
