import pandas as pd
import numpy as np
import math
import os



def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两点间的Haversine距离（单位：km）
    """
    R = 6371.0  # 地球平均半径（公里）
    
    # 转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # 计算差值
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine公式
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = R * c
    return round(distance)

def main():
    # 配置换电站数目
    snum = 8  # 出入口数量

    # 配置路径
    input_file = f'poi_node_{snum}e.xlsx'  # 输入文件路径
    output_dir = './output'         # 输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取Excel文件
    print("正在读取Excel文件...")
    df = pd.read_excel(input_file)
    print(f"成功读取 {len(df)} 个节点")
    
    # 创建节点到坐标的映射
    node_coords = {}
    for _, row in df.iterrows():
        node_coords[row['node_idx']] = (row['Lat'], row['Lng'])
    
    # 定义指定序列
    if snum == 8:
        sequence = ["e1", "s1", "s2", "e2", "e3", "s3", "e4", "e5", "s4", "e6", "e7", "s5", "s6", "e8"]
    else:
        sequence = ["e1", "s1", "s2", "e2", "e3", "s3", "e4", "e5", "s4", "e6", "e7", "s5", "s6", "e8", "e9", "s7", "e10", "s8", "e11", "s9", "e12", "s10", "s11", "e13"]
    
    missing = [node for node in sequence if node not in node_coords]
    if missing:
        print(f"警告：以下节点在数据中未找到: {missing}")
        return
    
    # 2. 计算相邻两点之间的Haversine距离
    print("\n正在计算相邻节点距离...")
    adjacent_distances = {}
    
    for i in range(len(sequence) - 1):
        node1 = sequence[i]
        node2 = sequence[i+1]
        
        lat1, lon1 = node_coords[node1]
        lat2, lon2 = node_coords[node2]
        
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        key = f"{node1}-{node2}"
        adjacent_distances[key] = round(dist, 2)
    
    # 在命令行输出
    print("\n相邻节点距离（km）：")
    print(adjacent_distances)
    
    # 打印详细分段信息
    print("\n详细分段：")
    total_dist = 0
    for i in range(len(sequence) - 1):
        node1 = sequence[i]
        node2 = sequence[i+1]
        dist = adjacent_distances[f"{node1}-{node2}"]
        total_dist += dist
        print(f"{node1} -> {node2}: {dist:.2f} km (累计: {total_dist:.2f} km)")
    
    # 3. 生成累加距离CSV
    print("\n正在生成累加距离表...")
    cumulative_distances = {}
    current_dist = 0.0
    cumulative_distances['e1'] = 0.0
    
    for i in range(len(sequence) - 1):
        node1 = sequence[i]
        node2 = sequence[i+1]
        
        lat1, lon1 = node_coords[node1]
        lat2, lon2 = node_coords[node2]
        
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        current_dist += dist
        cumulative_distances[node2] = current_dist
    
    # 节点排序：先s后e，各自按数值顺序排列
    matrix_sequence = (
        sorted([n for n in sequence if n.startswith('s')], key=lambda x: int(x[1:])) +
        sorted([n for n in sequence if n.startswith('e')], key=lambda x: int(x[1:]))
    )
    
    # 创建DataFrame并按照序列顺序排列
    cumulative_df = pd.DataFrame({'node_idx': matrix_sequence})
    cumulative_df['distance'] = cumulative_df['node_idx'].map(cumulative_distances)
    
    # 保存为CSV
    output_path_1 = os.path.join(output_dir, f'cumulative_distances_{snum}e.csv')
    cumulative_df.to_csv(output_path_1, index=False)
    print(f"累加距离表已保存: {output_path_1}")
    
    # 4. 生成距离矩阵CSV
    print("\n正在生成距离矩阵...")
    n = len(matrix_sequence)
    distance_matrix = np.zeros((n, n))
    
    # 计算有向距离
    for i in range(n):
        for j in range(n):
            node_i = matrix_sequence[i]
            node_j = matrix_sequence[j]
            dist = cumulative_distances[node_j] - cumulative_distances[node_i]
            distance_matrix[i][j] = round(dist, 2)
    
    # 创建DataFrame
    matrix_df = pd.DataFrame(distance_matrix, index=matrix_sequence, columns=matrix_sequence)
    
    # 保存为CSV
    output_path_2 = os.path.join(output_dir, f'distance_matrix_{snum}e.csv')
    matrix_df.to_csv(output_path_2)
    print(f"距离矩阵已保存: {output_path_2}")
    
    # 打印矩阵预览
    print("\n距离矩阵预览（前10个节点）：")
    print(matrix_df.iloc[:10, :10].to_string())
    
    print(f"\n完成！总路径长度: {cumulative_distances[f'e{snum}']:.2f} km")

if __name__ == "__main__":
    main()