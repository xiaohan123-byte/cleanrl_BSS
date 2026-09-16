'''
data_demand.py 生成了70天的随机需求数据，文件名为3s6e_station_demand_poisson_70d.json。
在这个文件，只需要提取一天的数据，作为MPC模型的输入。
需要提取的数据格式为：站点i在时段t初始SOC为n的随机需求量 f^R_{n,i,t}。
保存为json文件，命名为3s6e_station_demand_poisson_kd.json。
key: (n, i, t)，value: 需求量

首先需要读取3s6e_station_demand_poisson_70d.json文件，提取第k天的数据。
然后将第k天的第i个站点在时段t的需求量随机分配到不同的初始SOC状态n上，满足总需求量不变。
初始状态n的范围为0到N_min，其中N_min=2是一个合理的选择，表示只考虑初始SOC较低的车辆需求。
分配方法可以使用numpy的random.multinomial函数，根据需求量和初始SOC状态的数量进行随机分配。最后将提取的数据保存为新的json文件，供MPC模型使用。
'''

import json
import numpy as np

# ============ 参数配置 ============
k = 0                 # 提取第k天的数据（0-indexed）
N_min = 2             # 初始SOC状态上界，n ∈ {0, 1, ..., N_min}
TimePeriods = 24      # 输出的时段数，24表示逐小时，8表示每3小时聚合
seed = 42             # 随机种子

input_file = 'data_generation_rl/output/3s6e_station_demand_poisson_70d.json'
output_file = f'data_generation_rl/output/3s6e_station_demand_poisson_{k}d.json'

# ============ 读取数据 ============
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

stations = list(data.keys())  # e.g. ['s1', 's2', 's3']

# ============ 提取第k天需求并按SOC分配 ============
np.random.seed(seed)

fR = {}  # {(n, i, t): demand}

for i in stations:
    day_demand = data[i]['demand'][k]  # 长度24的列表，每小时的需求量

    # 如果TimePeriods < 24，将24小时聚合为TimePeriods个时段
    if TimePeriods < 24:
        hours_per_period = 24 // TimePeriods
        aggregated = []
        for t in range(TimePeriods):
            period_sum = sum(day_demand[t * hours_per_period:(t + 1) * hours_per_period])
            aggregated.append(period_sum)
        day_demand = aggregated

    for t, total in enumerate(day_demand):
        # 将时段t的总需求随机分配到 (N_min+1) 个SOC状态上
        # multinomial: 从total次试验中，分配到N_min+1个类别的次数
        if total > 0:
            probs = np.ones(N_min + 1) / (N_min + 1)  # 均匀概率
            counts = np.random.multinomial(total, probs)
        else:
            counts = np.zeros(N_min + 1, dtype=int)

        for n in range(N_min + 1):
            fR[(n, i, t)] = int(counts[n])

# ============ 保存为JSON ============
# key使用字符串形式的元组，与MPC模型读取方式一致（eval解析）
fR_serializable = {str(key): value for key, value in fR.items()}

with open(output_file, 'w') as f:
    json.dump(fR_serializable, f, indent=2)

print(f'已生成: {output_file}')
print(f'站点数: {len(stations)}, 时段数: {TimePeriods}, SOC状态: 0~{N_min}')
print(f'总条目数: {len(fR)}')