#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将订单柱状图转换为 CSV：node_idx,distance,time,weekday,direction,demand。

目录名 images_s1_bj_10 表示 node_idx=s1、direction=bj、history_index=10。
图片名 1.jpg 至 7.jpg 表示周一至周日，每张图片生成 time=0..23 的24行。
distance 从 node_13e.csv 的 node,distance 列读取。

用法（在脚本所在目录执行）：
    python order_info.py
    python order_info.py ./0917 -o order_info.csv
    python order_info.py ./0917/images_s1_bj_10 --node-csv ../../node_13e.csv

默认递归扫描脚本所在目录，结果保存到该目录的 order_info.csv。
若任一图片提取失败或元数据无效，终止转换，不写入不完整的结果。
"""

import argparse
import csv
import sys
from pathlib import Path

if __package__:
    from .extract_orders import collect_images, extract_chart_data, parse_image_metadata
else:
    from extract_orders import collect_images, extract_chart_data, parse_image_metadata


SCRIPT_DIR = Path(__file__).resolve().parent
COLUMNS = ["node_idx", "distance", "time", "weekday", "direction", "demand"]


def load_distances(node_csv):
    with Path(node_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"node", "distance"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{node_csv}: 缺少 node 或 distance 列")
        return {row["node"].strip(): float(row["distance"]) for row in reader}


def extract_order_rows(image_paths, distances):
    """所有图片成功提取后返回CSV行，避免写入不完整结果。"""
    image_files = collect_images(image_paths, recursive=True)
    if not image_files:
        raise ValueError("未找到图片文件")
    rows = []
    for image_file in image_files:
        try:
            node_idx, direction, history_index = parse_image_metadata(image_file)
            if node_idx not in distances:
                raise ValueError(f"距离表中未找到站点 {node_idx}")
            stem = Path(image_file).stem
            if stem not in {str(day) for day in range(1, 8)}:
                raise ValueError("图片名必须为 1 至 7，分别表示周一至周日")
            weekday = int(stem)
            values, error = extract_chart_data(image_file, history_index)
            if error:
                raise ValueError(error)
            if values is None or len(values) != 24:
                raise ValueError("提取结果必须包含24个小时的需求量")
            rows.extend(
                {
                    "node_idx": node_idx,
                    "distance": distances[node_idx],
                    "time": hour,
                    "weekday": weekday,
                    "direction": direction,
                    "demand": demand,
                }
                for hour, demand in enumerate(values)
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"{image_file}: {exc}") from exc
    rows.sort(key=lambda row: (
        row["distance"], row["node_idx"], row["direction"], row["weekday"], row["time"]
    ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=[str(SCRIPT_DIR)],
                        help="图片、目录或通配符，默认递归扫描脚本目录")
    parser.add_argument("-o", "--output", type=Path,
                        default=SCRIPT_DIR / "order_info.csv", help="输出CSV路径")
    parser.add_argument("--node-csv", type=Path,
                        default=SCRIPT_DIR.parents[1] / "node_13e.csv",
                        help="站点距离表路径")
    args = parser.parse_args()
    try:
        rows = extract_order_rows(args.paths, load_distances(args.node_csv))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"错误: {exc}\n")
    print(f"已处理 {len(rows) // 24} 张图片，输出 {len(rows)} 行：{args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
