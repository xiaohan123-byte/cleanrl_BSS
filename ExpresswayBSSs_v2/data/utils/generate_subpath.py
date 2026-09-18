import pandas as pd
from pathlib import Path

current_file = Path(__file__).resolve()
parent_dir = current_file.parent

# 读取出入口和站点信息
df_entrances = pd.read_csv( parent_dir / 'distance_6e_entrance.csv')
df_stations = pd.read_csv(parent_dir / 'distance_stations.csv')

# 提取出入口和站点信息
entrances = df_entrances[['node', 'distance']]
stations = df_stations[['node', 'distance']]


# 定义一个函数来找到两个出入口之间经过的站点，并按方向排序
def find_stations_between(start, end):
    start_distance = entrances.loc[entrances['node'] == start, 'distance'].values[0]
    end_distance = entrances.loc[entrances['node'] == end, 'distance'].values[0]
    
    if start_distance < end_distance:
        passing_stations = stations[(stations['distance'] >= start_distance) & (stations['distance'] <= end_distance)]
    else:
        passing_stations = stations[(stations['distance'] <= start_distance) & (stations['distance'] >= end_distance)].sort_values(by='distance', ascending=False)
    
    return passing_stations['node'].tolist(), abs(end_distance - start_distance)

# 双向的O-D对
# 生成子路径列表
subpaths = []
path_id = 1
minL=200  # 只挑选路径长度大于等于200的O-D对

print(find_stations_between('e1', 'e5'))

for i, start in entrances.iterrows():
    for j, end in entrances.iterrows():
        if start['node'] != end['node']: # 确保不生成同一出入口的路径
            stations_between, path_length = find_stations_between(start['node'], end['node'])
            if path_length >= minL and stations_between: # 只保留路径长度大于等于200且有经过站点的路径
                subpaths.append([f'p{path_id}', start['node'], end['node'], ','.join(stations_between)])
                path_id += 1

# 转换为DataFrame并保存为CSV
subpaths_df = pd.DataFrame(subpaths, columns=['PATH', 'SOURCE', 'ROOT', 'STATIONS'])
subpaths_df.to_csv(parent_dir / 'subpaths_6e.csv', index=False)
print(subpaths_df)