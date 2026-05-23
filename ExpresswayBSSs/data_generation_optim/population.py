#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据Excel中的e节点经纬度，查询高德地图API获取城市，匹配人口数据，输出CSV
"""

import pandas as pd
import requests
import time
import csv

# 高德地图API配置
AMAP_KEY = "f9a8b42f50b3a8a6e75d21176160de1e"
REGEO_URL = "https://restapi.amap.com/v3/geocode/regeo"

snum = 6 # 需要处理的e节点数量（根据实际情况调整）

# 城市人口字典（根据你提供的数据）
CITY_POPULATION = {
    "北京": 2183,
    "廊坊": 547,
    "天津": 1364,
    "沧州": 723,
    "德州": 549,
    "济南": 944,
    "泰安": 529,
    "临沂": 1085,
    "徐州": 901,
    "淮安": 452,
    "扬州": 458,
    "泰州": 447,
    "无锡": 750,
    "苏州": 1299,
    "上海": 2487
}

# 备用：基于节点名称关键词的城市映射（当API失败时使用）
NODE_NAME_CITY_MAP = {
    "马驹桥": "北京",
    "北京": "北京",
    "泗村店": "廊坊",
    "冀津": "廊坊",
    "乐陵": "德州",
    "西花园": "沧州",
    "济阳": "济南",
    "济南": "济南",
    "莱芜": "济南",
    "蒙阴": "临沂",
    "郯城": "临沂",
    "刘老庄": "淮安",
    "宝应": "扬州",
    "八桥": "扬州",
    "靖江": "泰州",
    "上海": "上海"
}

def get_city_by_name(node_name):
    """根据节点名称关键词匹配城市（备用方案）"""
    for keyword, city in NODE_NAME_CITY_MAP.items():
        if keyword in node_name:
            return city
    return None

def get_city_by_location(lat, lng, max_retries=3):
    """
    调用高德地图逆地理编码API，根据经纬度获取城市名称
    参数:
        lat: 纬度
        lng: 经度  
        max_retries: 最大重试次数
    返回:
        城市名称（如"北京"、"济南"）
    """
    params = {
        "key": AMAP_KEY,
        "location": f"{lng},{lat}",  # 经度在前，纬度在后
        "extensions": "base",
        "output": "json"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(REGEO_URL, params=params, timeout=10)
            data = response.json()

            if data.get("status") == "1" and data.get("info") == "OK":
                address_component = data["regeocode"]["addressComponent"]
                city = address_component.get("city", [])
                province = address_component.get("province", [])

                # 处理直辖市情况：如果是空数组或空字符串，使用省份名称
                if not city or city == []:
                    city = province

                # 如果是列表，取第一个元素
                if isinstance(city, list):
                    city = city[0] if city else ""

                # 清理城市名称（去掉"市"后缀，如"济南市"转为"济南"）
                if city:
                    city = city.replace("市", "").strip()

                return city
            else:
                print(f"API返回错误: {data.get('info', '未知错误')}")
                return None

        except Exception as e:
            print(f"请求失败(尝试{attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None

    return None

def get_population_by_city(city_name):
    """
    根据城市名称匹配人口数据
    参数:
        city_name: 城市名称（如"济南"）
    返回:
        人口数值（万），如果未找到返回None
    """
    if not city_name:
        return None

    # 直接匹配
    if city_name in CITY_POPULATION:
        return CITY_POPULATION[city_name]

    # 尝试部分匹配（如输入"济南市"，匹配"济南"）
    for city, pop in CITY_POPULATION.items():
        if city in city_name or city_name in city:
            return pop

    return None

def process_excel(input_file, output_file):
    """
    主处理函数：读取Excel，查询API，生成CSV
    """
    print(f"正在读取Excel文件: {input_file}")

    # 读取Excel
    df = pd.read_excel(input_file)

    # 筛选e开头的节点
    e_nodes = df[df['node_idx'].str.startswith('e', na=False)].copy()

    print(f"找到 {len(e_nodes)} 个e节点")
    print("开始查询高德地图API...")

    # 存储结果
    results = []

    for idx, row in e_nodes.iterrows():
        node_id = row['node_idx']
        node_name = row['node_name']
        lat = row['Lat']
        lng = row['Lng']

        print(f"处理 {node_id} ({node_name}): 坐标({lat}, {lng})", end=" ")

        # 查询城市（优先使用API）
        city = get_city_by_location(lat, lng)

        # 如果API失败，使用备用方案
        if not city:
            print("[API失败，使用备用匹配]", end=" ")
            city = get_city_by_name(node_name)

        if city:
            # 匹配人口
            population = get_population_by_city(city)

            if population:
                results.append({
                    'node_idx': node_id,
                    'city': city,
                    'population': population,
                    'source': 'API' if city in str(get_city_by_location(lat, lng)) else '备用'
                })
                print(f"-> 城市: {city}, 人口: {population}万")
            else:
                print(f"-> 城市: {city}, 未找到人口数据")
                results.append({
                    'node_idx': node_id,
                    'city': city,
                    'population': '未匹配',
                    'source': 'API失败'
                })
        else:
            print(f"-> 无法获取城市信息")
            results.append({
                'node_idx': node_id,
                'city': '未知',
                'population': 'N/A',
                'source': '失败'
            })

        # 避免请求过快，添加延时
        time.sleep(0.2)

    # 生成CSV文件（两行格式：第一行节点序号，第二行人口）
    print(f"\n正在生成CSV文件: {output_file}")

    # 准备CSV数据（转置格式）
    node_ids = [r['node_idx'] for r in results]
    populations = [r['population'] for r in results]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(node_ids)
        writer.writerow(populations)

    print(f"CSV文件已生成！")
    print(f"节点数量: {len(node_ids)}")
    print("节点列表:", node_ids)
    print("人口数据:", populations)

    # 同时生成一个详细的CSV（包含城市名称，方便核对）
    detail_file = output_file.replace('.csv', '_detail.csv')
    with open(detail_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node_idx', 'city', 'population', 'source'])
        for r in results:
            writer.writerow([r['node_idx'], r['city'], r['population'], r.get('source', '')])

    print(f"详细对照表已生成: {detail_file}")

    return results

if __name__ == "__main__":
    # 配置文件路径
    INPUT_EXCEL = f"poi_node_{snum}e.xlsx"  # 输入的Excel文件名
    OUTPUT_CSV = f"population_{snum}e.csv"  # 输出的CSV文件名

    # 执行处理
    results = process_excel(INPUT_EXCEL, OUTPUT_CSV)

    print("\n处理完成！")