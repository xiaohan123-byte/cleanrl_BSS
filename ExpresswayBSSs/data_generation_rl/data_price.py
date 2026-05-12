'''
生成RL环境的价格数据
'''

import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "3s6e_station__price_info.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "3s6e_station_price_info.json"

def convert_excel_to_json():
    # 读取 Excel 文件
    df = pd.read_excel(INPUT_FILE)
    
    # 构建 JSON 数据结构
    result = {}
    
    for _, row in df.iterrows():
        node_idx = row['node_idx']
        
        # 提取 1-24 小时的价格
        hourly_price = [float(row[hour]) for hour in range(1, 25)]
        
        result[node_idx] = {
            "node_idx": node_idx,
            "node_name": row['node_name'],
            "Lat": float(row['Lat']),
            "Lng": float(row['Lng']),
            "distance": int(row['distance']),
            "service_fee": float(row['service_fee']),
            "hourly_price": hourly_price
        }
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 写入 JSON 文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 转换完成！文件已保存到: {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_excel_to_json()

