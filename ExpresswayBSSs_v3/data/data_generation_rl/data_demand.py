#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 3s6e 站点 70 天 24h 泊松模拟需求数据集
输入: 3s6e_station_demand_info.xlsx（含各站点周一~周日每时段λ）
输出: 3s6e_station_demand_poisson_70d.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ==================== 配置 ====================
np.random.seed(42)  # 固定随机种子，确保结果可复现

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "3s6e_station_demand_info.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "3s6e_station_demand_poisson_70d.json"

SIMULATION_DAYS = 70   # 模拟天数
HOURS_PER_DAY = 24     # 每天小时数

# ==================== 读取原始 λ 数据 ====================
df = pd.read_excel(INPUT_FILE)

# 去掉可能的表头重复行（如果Excel第一行是列名，pandas已自动处理）
# 列 1~7 分别对应周一~周日的 24h λ，格式为逗号分隔字符串

# ==================== 生成模拟需求 ====================
result = {}

for _, row in df.iterrows():
    node_idx = row["node_idx"]
    node_name = row["node_name"]
    lat = float(row["Lat"])
    lng = float(row["Lng"])
    distance = int(row["distance"])

    # 解析周一~周日（列号 1~7）的 λ 列表
    weekly_lambdas = {}
    for day_col in range(1, 8):  # 1=周一, 2=周二, ..., 7=周日
        lambda_str = str(row[day_col])
        lambda_list = [float(x) for x in lambda_str.split(",")]
        if len(lambda_list) != HOURS_PER_DAY:
            raise ValueError(
                f"站点 {node_idx} 星期 {day_col} 的 λ 长度应为 {HOURS_PER_DAY}，"
                f"实际得到 {len(lambda_list)}"
            )
        weekly_lambdas[day_col] = lambda_list

    # 70 天循环：day 0 -> 周一(day_col=1), day 1 -> 周二(day_col=2) ...
    demand_70d = []
    for day in range(SIMULATION_DAYS):
        weekday_col = (day % 7) + 1          # 映射到 1~7
        day_lambdas = weekly_lambdas[weekday_col]

        # 逐小时泊松采样
        day_demand = [int(np.random.poisson(lam=lam)) for lam in day_lambdas]
        demand_70d.append(day_demand)

    result[node_idx] = {
        "node_idx": node_idx,
        "node_name": node_name,
        "Lat": lat,
        "Lng": lng,
        "distance": distance,
        "demand": demand_70d,
    }

# ==================== 保存 JSON ====================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"✅ 生成完成！")
print(f"   输出路径: {OUTPUT_FILE}")
print(f"   站点数量: {len(result)}")
print(f"   每个站点天数: {SIMULATION_DAYS}")
print(f"   每天小时数: {HOURS_PER_DAY}")
print(f"\n示例 —— s1 第1天(周一): {result['s1']['demand'][0]}")
