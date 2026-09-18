#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 order_info.csv 中两个方向的需求量相加，生成新的 CSV 文件。

按 node_idx、distance、time、weekday 分组，将 bj 与 sh 两个方向的 demand 相加。
输出列为 node_idx,distance,time,weekday,demand，不含 direction。

用法（默认在脚本所在目录读取 order_info.csv，并输出 mix_direction.csv）：
    python mix_direction.py

也可以指定输入和输出：
    python mix_direction.py order_info.csv -o demand.csv
"""

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_COLUMNS = ["node_idx", "distance", "time", "weekday", "direction", "demand"]
OUTPUT_COLUMNS = ["node_idx", "distance", "time", "weekday", "demand"]
GROUP_COLUMNS = INPUT_COLUMNS[:4]


def mix_direction(input_path):
    """返回按距离、时间、星期排序后的方向合计行。"""
    grouped = {}
    with Path(input_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(INPUT_COLUMNS) - set(reader.fieldnames or []):
            missing = set(INPUT_COLUMNS) - set(reader.fieldnames or [])
            raise ValueError(f"输入文件缺少列: {','.join(sorted(missing))}")
        for row in reader:
            key = tuple(row[column].strip() for column in GROUP_COLUMNS)
            grouped[key] = grouped.get(key, 0) + int(row["demand"])
    rows = []
    for key, demand in grouped.items():
        row = dict(zip(GROUP_COLUMNS, key))
        row["demand"] = demand
        rows.append(row)
    rows.sort(key=lambda row: (
        float(row["distance"]),
        int(row["time"]),
        int(row["weekday"]),
        row["node_idx"],
    ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=SCRIPT_DIR / "order_info.csv", help="输入CSV路径")
    parser.add_argument("-o", "--output", type=Path,
                        default=SCRIPT_DIR / "mix_direction.csv", help="输出CSV路径")
    args = parser.parse_args()
    try:
        rows = mix_direction(args.input)
        if not rows:
            raise ValueError(f"输入文件没有数据行: {args.input}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"错误: {exc}\n")
    print(f"已合并 {len(rows)} 组方向需求，输出: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
