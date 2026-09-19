"""Render the audited zero-terminal metrics into a report and shareable plots."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'outputs/zero_terminal_real_data_seed1_v1/reports'


def read(name):
    return json.loads((REPORTS/name).read_text(encoding='utf8'))


def main():
    audit = read('audit.json')
    if not audit['audit']['passed'] or audit['job_count'] != 35:
        raise ValueError('A clean audit of all 35 jobs is required before reporting.')
    days, summary = read('daily_metrics.json'), read('summary_by_horizon.json')
    lines = ['# 零终端价值实验：最终结果与一致性核验', '',
        '35/35 个日场景完成，完成时间为 2026-09-19 05:20:39（北京时间）。仅 β=0，H=4–8，'
        '每个 H 使用相同七天各运行一次，基础 seed=1；未开展多种子重复、RL 训练或完美信息实验。', '',
        '## 数据与核验', '',
        '- 数据版本：observed_means_11s6e_poisson70_seed1_v1。70 天均为基于真实站点/小时/星期均值的泊松合成场景，'
        '并非 70 天实测订单。源 CSV 的 10 份哈希及归档副本均一致；70 天场景和日前路径重新生成后与冻结版本完全相同。',
        '- 周一至周日测试日编号为 1、37、3、4、19、13、14，其余 63 天保留为训练/验证候选池。'
        '各 H 的同日输入、误差、初始库存和日前路径一致。每个日场景均为 200 名预约、200 个随机请求。',
        '- 随机请求由当日泊松样本的站点/小时联合权重进行固定总量多项分配；小时内均匀到达。'
        '随机预测仅采用原始对应星期均值（期望总量 200），按累计强度生成确定性请求；逐轮预测核对一致。',
        f'- 全部 {audit["total_rounds"]} 个执行轮次的持久化日志与结果逐条相同，无重复或漏段。'
        '35 份 statistics 和全部派生 metrics 重建后与原记录精确一致。逐日预约完成+失败=200、'
        '随机服务+超时=200；全部最终状态均无 active 用户或 waiting 请求。',
        '- 满电交付、250 个实际电池槽、SOC 与充电功率、收入与电费、预约优先和同类先到先服务、'
        '按实际前站服务时刻及分段行驶误差计算的后站到达、30 分钟等待、路径更新时钟和实际发布计费均通过核验。',
        '- 每名预约最多失败一次，失败费=1000×失败人数；实际调整费=10×发布变化次数。'
        '净收益=服务收入−充电费用−实际调整费用−预约失败费用。午夜后的所有收入和费用均记回原需求日。', '',
        '核验通过不代表所有 MILP 均达到目标 gap。逐轮状态、实际 gap、变量数及耗时见 solve_rounds.csv。', '',
        '## 七日均值', '',
        '等待时间先对每天全部已服务换电请求取平均，再对七天等权平均。主表的求解时间在每天 0–24 h '
        '需求期（96 轮）内计算均值/P95，再对七天取均值。', '',
        '| H | 窗口/min | 净收益/元 | 预约失败率/% | 随机服务率/% | 等待/min | 调整次数 | 需求期求解均值/P95，s |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["horizon"]} | {r["window_minutes"]} | {r["net_profit_yuan"]:.2f} | '
                     f'{100*r["reservation_failure_rate"]:.2f} | {100*r["random_service_rate"]:.2f} | '
                     f'{r["mean_wait_minutes_served"]:.2f} | {r["path_adjustments"]:.2f} | '
                     f'{r["demand_solver_seconds_mean"]:.2f}/{r["demand_solver_seconds_p95"]:.2f} |')
    lines += ['', '## 求解、跨日与库存诊断', '',
        '下表的比例、gap 和求解时间以每日含跨日处理的全部轮次为分母，再对七天等权平均；'
        '最大 gap 列为该 H 的全部轮次最大值。日内 P95 的平均值不等于合并全部轮次后的 P95。', '',
        '| H | 限时比例/% | gap达标/% | 平均gap/% | 最大gap/% | 全程求解均值/P95，s | 跨日均值/最大，h | 终库存/kWh | 满电数 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["horizon"]} | {100*r["time_limit_fraction"]:.2f} | {100*r["mip_gap_target_met_fraction"]:.2f} | '
                     f'{100*r["mip_gap_mean"]:.6f} | {100*r["worst_mip_gap"]:.4f} | '
                     f'{r["solver_seconds_mean"]:.2f}/{r["solver_seconds_p95"]:.2f} | '
                     f'{r["cleanup_hours"]:.2f}/{r["max_cleanup_hours"]:.2f} | {r["final_inventory_kwh"]:.2f} | '
                     f'{r["final_full_batteries"]:.2f} |')
    lines += ['', '主表只用需求期 96 轮；上表含跨日轮次，因此其求解时间均值通常更低。'
              '两套统计及仅需求期的限时比例/gap达标比例均保存在 daily_metrics.csv 和 summary_by_horizon.csv。', '',
              '## 收入与成本分解（七日均值，元）', '',
              '| H | 服务收入 | 充电成本 | 调整成本 | 预约失败成本 | 净收益 |', '|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| '+str(r['horizon'])+' | '+' | '.join(f'{r[k]:.2f}' for k in
            ['service_income_yuan','charging_cost_yuan','adjustment_cost_yuan','reservation_failure_cost_yuan','net_profit_yuan'])+' |')
    lines += ['', '## 结果解读与论文对照', '',
        'H=4–6 的高失败率和低充电量符合短窗口不易看到完整充电收益的机制：以退回 SOC=0.2 为例，'
        '按 60 kW、95% 效率、100 kWh 充至满电约需 84.21 分钟（至少六个 15 分钟充电时段），'
        '服务只能在随后边界发生，H=7 才能同时容纳该示例的充电过程和后续服务。'
        '这是解释结果的机制推断，不是对所有 SOC/请求的阈值定理。H=7 与 H=8 的净收益分别为 '
        '37180.19、37639.56 元，增幅仅 1.24%；随机服务率提高 2.64 个百分点，预约失败人数却从七天合计 3 人变为 6 人。'
        '有限求解时限也影响所执行的决策，不能声称所有指标随 H 单调改善或已经证明 RL 有效。', '',
        '长预测域的求解难度不是单由变量数量线性决定的。需求期二元变量数的七日均值为 '
        '2489、2760、3069、3487、3933；复杂度增加同时反映在限时比例和实际 gap。'
        '此前聊天中将 H=4–8 写成 16–32 个时间段是错误的，实际为 4–8 段。', '',
        '| 项目 | 对照结论及实际口径 |', '|---|---|',
        '| 时间网格 | 第3节仍给出默认 5 min、K=3；第5节已明确本实验覆盖为 15 min、K=2，H=4–8 段。 |',
        '| 100 km 间距 | 第3节候选网络采用有可行性和既有路径保护的剪枝，不是硬约束。已核对每轮所选路径属于按该规则重建的网络。 |',
        '| 入口 SOC | 代码在日前和未入场网络可达性判断中使用 max(0.5,报告SOC−0.05)；预测退回SOC仍用报告值。原第3节未显式写该缓冲，第5节已补充输入处理口径。 |',
        '| 完成定义 | 第4节不再保留全部换电完成用户；执行端也在最后换电/无剩余换电且出口SOC可行时完成服务，未继续模拟无服务的最终离场路段。已在第5节说明。 |',
        '| 30 min 等待 | 所有实际服务均在到站与截止之间；截止恰在边界时保留到该轮服务机会。 |',
        '| 路径费用 | 优化目标包含未入场相对日前计划的惩罚；实际账务只统计在途发布变化。与第3、4节相符。 |',
        '| 充电与收入 | 满电交付；功率为电网输入，SOC增量乘效率；补充电量×(电价+服务费)计收入，电网输入×电价另行扣费。 |',
        '| 跨日 | 24:00后无新增需求/预测，继续处理已有用户；价格循环、同日归账、不设残值或强制补满。最长14.5 h，均未触及186段安全上限。 |',
        '| 解的精度 | RelGap=0.0001是目标。H=7/H=8最坏实际gap为6.02%/16.97%，必须保留这一限制。 |', '',
        '这些说明将原论文未明确的实验输入与服务结算口径写入第5节；本次没有修改模型、求解设置或补跑日场景。', '',
        '## 数值修复与恢复留痕', '',
        'H=7 曾遇 Windows 原子 JSON 写入冲突，加入有界写入重试；H=8 曾遇充电功率边界数值残差。'
        '恢复从完整轮次日志继续，用户暂停和恢复也按边界处理。各 worker 源码快照与原计划比较，'
        '差异仅为 atomic_io.py 和 execution.py；MILP、数据生成、预测、账务等代码哈希相同。'
        '不同阶段执行端容差修复保留各自源码快照，早期已完成结果未重算。',
        f'全35日已执行轨迹中，求解器首段功率与执行功率的最大差为 {max(r["max_power_correction_kw"] for r in days):.12g} kW；'
        f'站级实际功率最大超限残差为 {max(r["max_station_excess_kw"] for r in days):.12g} kW，'
        f'SOC能量递推最大残差为 {max(r["max_soc_balance_residual"] for r in days):.12g}。'
        '独立账务核验仍使用原物理容差（单槽/站总功率1e-7 kW），均通过；没有为核验放宽物理边界。', '',
        '启动前记录的138项检查均通过；本次逐轨迹核验结果以 audit.json 为准。'
        '没有重新执行与本次任务无关、使用过期全局截止时间的旧诊断套件。', '',
        '## 复现与文件', '',
        '在 ExpresswayBSSs_v2 目录运行：', '', '```powershell',
        r'D:\App\miniconda3\envs\py310\python.exe audit_final_zero_terminal.py',
        r'D:\App\miniconda3\envs\py310\python.exe report_final_zero_terminal.py', '```', '',
        '- [机器可读核验及原结果哈希](audit.json)',
        '- [35日指标](daily_metrics.csv) / [按H汇总](summary_by_horizon.csv)',
        '- [逐轮求解状态、gap、耗时、规模](solve_rounds.csv)',
        '- [固定测试日](test_days.csv)',
        '- [更新前论文备份](05_numerical_experiments_before_update.tex)',
        '- 原始服务、电池、账务轨迹仍保留于 ../runs/h*_day*/result.json.gz 和 journal/rounds.jsonl。', '',
        '统计适用于每日初始全满的独立需求日，不代表连续运营稳态；未对终态库存估值。'
        '35 个场景是5个H乘7个测试日，不能将其当作35次独立随机重复。', '']
    (REPORTS/'verification_report.md').write_text('\n'.join(lines),encoding='utf8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    hs = [r['horizon'] for r in summary]
    axes[0].plot(hs,[r['net_profit_yuan']/1000 for r in summary],'o-',color='#2463a6')
    axes[0].axhline(0,color='gray',linewidth=.8)
    axes[0].set(ylabel='Net profit (thousand CNY)', title='Operating result')
    axes[1].plot(hs,[100*r['reservation_failure_rate'] for r in summary],'o-',label='Reservation failure')
    axes[1].plot(hs,[100*r['random_service_rate'] for r in summary],'s-',label='Random service')
    axes[1].set(ylabel='Rate (%)',ylim=(-3,103),title='Service outcomes')
    axes[1].legend(fontsize=8)
    axes[2].semilogy(hs,[r['demand_solver_seconds_mean'] for r in summary],'o-',label='Mean')
    axes[2].semilogy(hs,[r['demand_solver_seconds_p95'] for r in summary],'s-',label='P95')
    axes[2].set(ylabel='Solve time (s, log scale)',title='Demand-period computation')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(hs)
        ax.set_xlabel('H (15-min periods)')
        ax.grid(alpha=.2)
    fig.savefig(REPORTS/'zero_terminal_summary.pdf',bbox_inches='tight')
    fig.savefig(REPORTS/'zero_terminal_summary.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    manifest = {}
    for path in [Path(__file__),ROOT/'audit_final_zero_terminal.py',ROOT/'paper_v2/sections/05_numerical_experiments.tex']:
        manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (REPORTS/'report_source_hashes.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('Report, PDF/PNG plots and source hashes exported.')


if __name__=='__main__':
    main()
