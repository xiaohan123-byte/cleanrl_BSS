'''
生成并加载电池换电站需求实验实例
'''

import json
from parameter import parameter # 业务环境参数
from BSSChargingScheduling import BSSChargingScheduling
from pathlib import Path

class InstanceGenerator:

    def __init__(self):
        
        self.para=parameter()

    def load_inst(self,file_path):
        '''把生成好的实例 JSON 重新加载成一个可直接给环境或算法使用的对象。
            并且加上业务参数字段
        '''
        with open(file_path, 'r',encoding="utf-8") as f:
            inst_json = json.load(f)
            inst_json["parameter"]=self.para
            return BSSChargingScheduling(inst_json)
        

if __name__ == "__main__":

    # 项目根目录（也可以换成你自己的基准目录）
    BASE_DIR = Path(__file__).resolve().parents[1]

    # 输入目录 + 输入文件
    input_dir = BASE_DIR / "data_generation_rl" / "output"
    data = input_dir / "3s6e_station_demand_poisson_70d_merged.json"

    IG=InstanceGenerator()

    x=IG.load_inst(data)

    print(x.parameter.Power)

    



