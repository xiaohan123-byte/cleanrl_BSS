#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 3s6e 站点价格信息合并到 70 天泊松需求数据中。

输入:
	output/3s6e_station_demand_poisson_70d.json
	output/3s6e_station_price_info.json

输出:
	默认生成 output/3s6e_station_demand_poisson_70d_merged.json
"""

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DEMAND_FILE = OUTPUT_DIR / "3s6e_station_demand_poisson_70d.json"
PRICE_FILE = OUTPUT_DIR / "3s6e_station_price_info.json"
MERGED_FILE = OUTPUT_DIR / "3s6e_station_demand_poisson_70d_merged.json"


def merge_price_into_demand(demand_file=DEMAND_FILE, price_file=PRICE_FILE, output_file=MERGED_FILE):
	with open(demand_file, "r", encoding="utf-8") as f:
		demand_data = json.load(f)

	with open(price_file, "r", encoding="utf-8") as f:
		price_data = json.load(f)

	demand_keys = set(demand_data.keys())
	price_keys = set(price_data.keys())

	missing_in_price = sorted(demand_keys - price_keys)
	missing_in_demand = sorted(price_keys - demand_keys)
	if missing_in_price or missing_in_demand:
		raise ValueError(
			"站点键不一致，无法安全合并。"
			f" demand 中缺失: {missing_in_price};"
			f" price 中缺失: {missing_in_demand}"
		)

	merged = {}
	for node_idx, demand_item in demand_data.items():
		price_item = price_data[node_idx]
		merged[node_idx] = {
			**demand_item,
			"service_fee": price_item["service_fee"],
			"hourly_price": price_item["hourly_price"],
		}

	target_file = output_file or MERGED_FILE
	with open(target_file, "w", encoding="utf-8") as f:
		json.dump(merged, f, ensure_ascii=False, indent=4)

	return target_file, len(merged)


if __name__ == "__main__":
	output_file, station_count = merge_price_into_demand()
	print(f"✅ 合并完成！输出文件: {output_file}")
	print(f"   站点数量: {station_count}")
