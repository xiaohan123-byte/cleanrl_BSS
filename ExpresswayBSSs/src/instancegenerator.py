'''
生成并加载电池换电站需求实验实例:
1. 从原始资源文件里导出一个实验实例，并保存成 JSON。
它会读取 demand_data，提取各BSS的价格、预约和随机需求、地理位置、电池库存等信息。
然后写到 out_dir 下的实例文件里。

2. 把生成好的实例 JSON 重新加载成一个可直接给环境或算法使用的对象。
这个过程在同一个文件里的 load_inst 完成，
它会把 JSON 字段交给 BSSChargingScheduling.py 里的 BSSChargingScheduling 类。
这个类本身很轻，只是把传入的字段挂到对象上，方便后续代码直接访问。
'''

import sys
sys.path.append("C:/Users/myr/Desktop/科研/准备/Code/QMIX_BSS")
import json
import random
from src.parameter import parameter # 现实环境参数
import numpy as np
from src.BSSChargingScheduling import BSSChargingScheduling

class InstanceGenerator:

    def __init__(self,
                 bss_num: int,
                 Allocated_Peak_Shaving_Quantity: list,
                 Allow_Peak_Shaving: list,
                 time_periods=24,
                 exp_num=30):
        
        self.para=parameter()
        self._exp_num = exp_num
        self._time_periods = time_periods
        self.para.Power = self.para.Power/int(self._time_periods/24)
        self._bss_num=bss_num
        self.Allocated_Peak_Shaving_Quantity = Allocated_Peak_Shaving_Quantity
        self.Allow_Peak_Shaving = Allow_Peak_Shaving
        self.TOU=self.para.TOU

    def randomly_gen_instance(self, res_src: str, out_dir: str,addition_str):
        # read bss resource files
        random.seed(1)
        with open(res_src, 'r',encoding="utf-8") as f:
            json_data = json.load(f)

        num = len(json_data)
        if self._bss_num > num:
            raise ValueError(f'The required bss num should be smaller than {num}')

        for exp in range(self._exp_num):
            inst_name = f'B{self._bss_num}{addition_str}-{exp}'
            index_list = random.sample(range(num), self._bss_num)
            self.Name=[]
            self.Fix_Price=[]
            self.Use_Price=[]
            self.Demand=[]
            self.Charging_Slot_Number=[]
            self.Battery_Number=[]
            self.Base_Load=[]
            self.geo_pos=[]
            bss_list = []
            for i in range(self._bss_num):
                
                self.Fix_Price.append(json_data[f'{index_list[i]}']['fix_price'])
                self.Use_Price.append(json_data[f'{index_list[i]}']['use_price'])
                self.Name.append(json_data[f'{index_list[i]}']['name'])
                self.Charging_Slot_Number.append(json_data[f'{index_list[i]}']['charging_slot_num'])
                self.Battery_Number.append(json_data[f'{index_list[i]}']['charging_slot_num'])
                self.Demand.append(json_data[f'{index_list[i]}']['demand'])
                self.geo_pos.append(json_data[f'{index_list[i]}']['geo_pos'])

                #Calculate Base_Load
            self.Fix_Price=np.array(self.Fix_Price)
            self.Use_Price=np.array(self.Use_Price)
            self.Charging_Slot_Number=np.array(self.Charging_Slot_Number)
            self.Battery_Number=np.array(self.Battery_Number)
            self.Demand=np.array(self.Demand)
            self.Demand=self.Demand.transpose(1,0,2)
            self.Repeat()
            self.Base_Load=(self.para.SOCmax-(self.para.SOCinit_max+self.para.SOCinit_min)/2)*self.para.Battery_Volume*np.sum(np.mean(self.Demand,axis=1),axis=0)
            self.Base_Load=np.roll(self.Base_Load,1)




            inst_dict = {
                'inst_name':inst_name,
                'N':self._bss_num,
                'Name':self.Name,
                'geo_pos':self.geo_pos,
                'Battery_Number':self.Battery_Number.tolist(),
                'Charging_Slot_Number':self.Charging_Slot_Number.tolist(),
                'Demand':self.Demand.tolist(),
                'TOU':self.TOU.tolist(),
                'Fix_Price':self.Fix_Price.tolist(),
                'Use_Price':self.Use_Price.tolist(),
                'Time_Period_Num':self._time_periods,
                'Base_Load':self.Base_Load.tolist(),
                'Allocated_Peak_Shaving_Quantity':self.Allocated_Peak_Shaving_Quantity.tolist(),
                'Allow_Peak_Shaving':self.Allow_Peak_Shaving.tolist(),
            }
            with open(out_dir+ '/' + inst_name + '.json', 'w',  encoding="utf-8",newline='\n') as f2:
                f2.write(json.dumps(inst_dict, indent=4, ensure_ascii=False))
            

    def load_inst(self,file_path):
        with open(file_path, 'r',encoding="utf-8") as f:
            inst_json = json.load(f)
            inst_json["parameter"]=self.para
                  
            return BSSChargingScheduling(inst_json)
    def Repeat(self):
        
        if self._time_periods/24<1 or self._time_periods/24-int(self._time_periods/24)>1e-3:
            raise ValueError(f'Time Slot is not proper')
        
        self.Fix_Price=np.repeat(self.Fix_Price,int(self._time_periods/24),axis=1).reshape(self.Fix_Price.shape[0],-1)
        self.Use_Price=np.repeat(self.Use_Price,int(self._time_periods/24),axis=1).reshape(self.Use_Price.shape[0],-1)
        self.TOU=np.array(self.TOU)
        self.TOU=np.repeat(self.TOU,int(self._time_periods/24),axis=0).reshape(-1)
        self.Allocated_Peak_Shaving_Quantity=np.repeat(self.Allocated_Peak_Shaving_Quantity,int(self._time_periods/24),axis=0).reshape(-1)
        self.Allow_Peak_Shaving=np.repeat(self.Allow_Peak_Shaving,int(self._time_periods/24),axis=0).reshape(-1)
        times=np.array([1,1,int(self._time_periods/24)])
        Demand=np.zeros(self.Demand.shape*times)
        times=int(self._time_periods/24)
        for tp in range(24):
            temp=self.Demand[:,:,tp]

            for sepa in range(times):
                Demand[:,:,times*tp+sepa]=self.Demand[:,:,tp]//times
          
                if sepa==times-1:
                    Demand[:,:,times*tp+sepa]=temp
                temp-=self.Demand[:,:,tp]//times

        self.Demand = Demand
        
        

if __name__ == "__main__":
    bss_num=5
    time_slot=96
    Allocated=[0 for i in range(time_slot)]
    res="./res/demand&electricity-price&stochastic_ordered.json"
    out_dir="./res/test_inst"
    IG=InstanceGenerator(bss_num,Allocated,time_slot,exp_num=5)
    IG.randomly_gen_instance(res,out_dir,"")
    x=IG.load_inst("./res/test_inst/B5-0.json")
    print()
    



