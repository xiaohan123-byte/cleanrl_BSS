#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量提取柱状图订单量数据工具

从24小时柱状图提取订单量数值，输出24个逗号分隔的整数（索引0-23对应0:00-23:00）

用法:
    python extract_orders.py <历史时刻索引> <图片路径/目录/通配符> [图片路径2 ...]

示例:
    python extract_orders.py 21 ./images/*.jpg
    python extract_orders.py 15 img1.jpg img2.jpg img3.jpg
    python extract_orders.py 21 ./images/
"""

import os
import sys
import numpy as np
from PIL import Image
from glob import glob
from itertools import groupby
from numpy.polynomial import polynomial as P


def measure_bar_height(x_pos, mask_img, base_y, search_w=8):
    """测量青色柱子高度"""
    x = int(round(x_pos))
    x_start = max(0, x - search_w)
    x_end = min(mask_img.shape[1], x + search_w + 1)
    max_h = 0
    for xi in range(x_start, x_end):
        col = mask_img[:, xi]
        nonzero = np.where(col > 0)[0]
        if len(nonzero) > 0:
            top_y = nonzero.min()
            h = base_y - top_y
            max_h = max(max_h, h)
    return max_h


def has_real_dark_bar(dx, dark_mask, base_y):
    """检查是否有真正的深色柱子（底部有>=15px的实心段）"""
    x = dx
    x_start = max(0, x - 10)
    x_end = min(dark_mask.shape[1], x + 11)
    for xi in range(x_start, x_end):
        col = dark_mask[:, xi]
        nonzero = np.where(col > 0)[0]
        if len(nonzero) > 0:
            segments = []
            start = nonzero[0]
            prev = nonzero[0]
            for y in nonzero[1:]:
                if y > prev + 3:
                    segments.append((start, prev, prev - start + 1))
                    start = y
                prev = y
            segments.append((start, prev, prev - start + 1))
            for s_start, s_end, length in segments:
                if s_end >= base_y - 5 and length >= 15:
                    return True
    return False


def detect_dashed_line(dx, dark_mask, base_y):
    """检测虚线（底部无长实心段，但有很多均匀分布的短段）"""
    x = dx
    x_start = max(0, x - 10)
    x_end = min(dark_mask.shape[1], x + 11)
    all_segments = []
    for xi in range(x_start, x_end):
        col = dark_mask[:, xi]
        nonzero = np.where(col > 0)[0]
        if len(nonzero) > 0:
            start = nonzero[0]
            prev = nonzero[0]
            for y in nonzero[1:]:
                if y > prev + 3:
                    length = prev - start + 1
                    all_segments.append((xi, start, prev, length))
                    start = y
                prev = y
            length = prev - start + 1
            all_segments.append((xi, start, prev, length))
    if not all_segments:
        return False, 0
    has_long_bottom = any(
        length >= 15 and s_end >= base_y - 10
        for _, _, s_end, length in all_segments
    )
    if has_long_bottom:
        return False, 0
    short_segments = [s for s in all_segments if s[3] <= 5]
    y_min = min(s[1] for s in all_segments)
    y_max = max(s[2] for s in all_segments)
    y_span = y_max - y_min
    is_dashed = len(short_segments) >= 8 and y_span > 150
    if is_dashed:
        x_counts = {}
        for xi, _, _, _ in short_segments:
            x_counts[xi] = x_counts.get(xi, 0) + 1
        if x_counts:
            return True, max(x_counts, key=x_counts.get)
    return False, 0


def find_px_per_unit(heights):
    nonzero = sorted([h for h in heights if h > 20])
    if not nonzero:
        return 75.0
    min_h = nonzero[0]
    if min_h <= 90:
        return float(min_h)
    unit2 = min_h / 2.0
    valid = sum(1 for h in nonzero if abs(h / unit2 - round(h / unit2)) < 0.15)
    if valid >= len(nonzero) * 0.8 and unit2 >= 35:
        return unit2
    return float(min_h)


def extract_chart_data(image_path, history_index):
    img = Image.open(image_path)
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # 1. 定位图表区域
    r = img_np[:, :, 0].astype(int)
    g = img_np[:, :, 1].astype(int)
    b = img_np[:, :, 2].astype(int)
    mask_cyan_wide = (
        (r < 200) & (g > 140) & (b > 140) &
        (np.abs(g - b) < 50) & (g > r + 20)
    )
    best_score = 0
    best_y = None
    for y in range(50, h - 450):
        window = mask_cyan_wide[y:y + 400, :].astype(np.uint8)
        col_proj = np.sum(window > 0, axis=0)
        peaks = col_proj > 50
        peak_count = sum(1 for k, _g in groupby(peaks) if k)
        if 8 <= peak_count <= 20:
            score = np.sum(window) + peak_count * 1000
            if score > best_score:
                best_score = score
                best_y = y
    if best_y is None:
        return None, "无法定位图表区域"

    chart = img_np[best_y:best_y + 400, :, :]
    rc = chart[:, :, 0].astype(int)
    gc = chart[:, :, 1].astype(int)
    bc = chart[:, :, 2].astype(int)

    mask_cyan = (rc < 100) & (gc > 160) & (bc > 160) & (np.abs(gc - bc) < 30)
    mask_dark = (rc < 120) & (gc < 120) & (bc < 120) & ~mask_cyan
    mask_all = (mask_cyan | mask_dark).astype(np.uint8) * 255
    mask_dark_u8 = mask_dark.astype(np.uint8) * 255
    mask_cyan_u8 = mask_cyan.astype(np.uint8) * 255

    # 2. 找基线
    row_sums = np.sum(mask_all > 0, axis=1)
    baseline_y = None
    for y in range(len(row_sums) - 1, -1, -1):
        if row_sums[y] > 50:
            baseline_y = y
            break
    if baseline_y is None:
        return None, "无法找到基线"

    # 3. 找青色柱子
    col_sums_cyan = np.sum(mask_cyan_u8 > 0, axis=0)
    cyan_bars = []
    for k, grp in groupby(enumerate(col_sums_cyan > 15), key=lambda x: x[1]):
        if k:
            group = list(grp)
            center = (group[0][0] + group[-1][0]) // 2
            cyan_bars.append(center)
    if len(cyan_bars) < 2:
        return None, f"青色柱子不足: {len(cyan_bars)}"

    # 4. 找深色区域
    dark_positions = []
    for k, grp in groupby(enumerate(np.sum(mask_dark_u8 > 0, axis=0) > 20), key=lambda x: x[1]):
        if k:
            group = list(grp)
            center = (group[0][0] + group[-1][0]) // 2
            dark_positions.append(center)

    # 5. 寻找锚点：深色柱子 > 虚线
    anchor_x = None
    anchor_type = None
    for dx in dark_positions:
        if has_real_dark_bar(dx, mask_dark_u8, baseline_y):
            anchor_x = dx
            anchor_type = "bar"
            break
    if anchor_x is None:
        for dx in dark_positions:
            is_dashed, dashed_x = detect_dashed_line(dx, mask_dark_u8, baseline_y)
            if is_dashed:
                anchor_x = dashed_x
                anchor_type = "dashed"
                break
    if anchor_x is None:
        return None, "未找到锚点（深色柱子或虚线）"

    # 6. 建立24小时位置映射
    spacings = [
        cyan_bars[i] - cyan_bars[i - 1]
        for i in range(1, len(cyan_bars))
        if cyan_bars[i] - cyan_bars[i - 1] < 100
    ]
    if not spacings:
        return None, "无法计算柱子间距"
    median_spacing = float(np.median(spacings))
    derived_indices = [
        round((cx - anchor_x) / median_spacing + history_index)
        for cx in cyan_bars
    ]
    all_indices = derived_indices + [history_index]
    all_x = cyan_bars + [anchor_x]
    filtered = [(i, x) for i, x in zip(all_indices, all_x) if 0 <= i <= 23]
    if len(filtered) < 2:
        return None, "拟合数据不足"
    coef, _ = P.polyfit([p[0] for p in filtered], [p[1] for p in filtered], 1, full=True)
    slope = coef[1]
    intercept = coef[0]
    positions_x = [slope * i + intercept for i in range(24)]

    # 7. 测量24个位置的高度
    heights = []
    for i, x in enumerate(positions_x):
        if i == history_index:
            if anchor_type == "dashed":
                h = 0
            else:
                h = measure_bar_height(x, mask_cyan_u8, baseline_y)
                if h < 20:
                    h_d = 0
                    x_i = int(round(x))
                    x_s = max(0, x_i - 10)
                    x_e = min(mask_dark_u8.shape[1], x_i + 11)
                    bottom_tops = []
                    for xi in range(x_s, x_e):
                        col = mask_dark_u8[:, xi]
                        nonzero = np.where(col > 0)[0]
                        if len(nonzero) > 0:
                            segments = []
                            start = nonzero[0]
                            prev = nonzero[0]
                            for y in nonzero[1:]:
                                if y > prev + 3:
                                    if prev - start >= 2:
                                        segments.append((start, prev))
                                    start = y
                                prev = y
                            if prev - start >= 2:
                                segments.append((start, prev))
                            for s_start, s_end in sorted(segments, key=lambda s: s[1], reverse=True):
                                if s_end >= baseline_y - 10:
                                    h = baseline_y - s_start
                                    bottom_tops.append(h)
                                    break
                    if bottom_tops:
                        h = int(np.median(bottom_tops))
        else:
            h = measure_bar_height(x, mask_cyan_u8, baseline_y)
        heights.append(h)

    # 8. 确定像素到值的转换
    px_unit = find_px_per_unit(heights)

    # 9. 计算24个值
    values = []
    for h in heights:
        if h > 15:
            values.append(round(h / px_unit))
        else:
            values.append(0)

    return values, None


def collect_images(paths):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in image_extensions:
                all_files.extend(glob(os.path.join(p, f'*{ext}')))
                all_files.extend(glob(os.path.join(p, f'*{ext.upper()}')))
        elif '*' in p or '?' in p:
            all_files.extend(glob(p))
        elif os.path.exists(p):
            all_files.append(p)
    result = []
    for f in sorted(set(all_files)):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            result.append(f)
    return sorted(result)


def main():
    if len(sys.argv) < 3:
        print("批量提取柱状图订单量数据")
        print()
        print("用法: python extract_orders.py <历史时刻索引> <图片...>")
        print()
        print("参数:")
        print("  历史时刻索引: 深色柱子或虚线（历史此刻）对应的0-23索引")
        print("  图片: 支持文件路径、目录、通配符，可多个")
        print()
        print("示例:")
        print("  python extract_orders.py 21 ./images/*.jpg")
        print("  python extract_orders.py 15 img1.jpg img2.jpg")
        sys.exit(1)

    try:
        history_index = int(sys.argv[1])
        if not (0 <= history_index <= 23):
            raise ValueError
    except ValueError:
        print("错误: 历史时刻索引必须是0-23的整数")
        sys.exit(1)

    image_files = collect_images(sys.argv[2:])
    if not image_files:
        print("错误: 未找到图片文件")
        sys.exit(1)

    print(f"历史时刻索引: {history_index}")
    print(f"图片数量: {len(image_files)}\n")
    print("=" * 60)

    for img_file in image_files:
        values, error = extract_chart_data(img_file, history_index)
        filename = os.path.basename(img_file)
        if error:
            print(f"【{filename}】 错误: {error}")
        else:
            result_str = ','.join(map(str, values))
            print(f"【{filename}】")
            print(f"{result_str}")
        print()


if __name__ == '__main__':
    main()
