"""
MPC模型Gurobi实现
参考手稿中的soc-wised expanded network建模方法
基于现有代码的Gurobi建模风格

目标函数：max I - C1 - C2
  = 服务收益 - 未满足惩罚 - 电站充电成本
"""

import json
import os
import pandas as pd
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from gurobipy import *
import time
import sys
import random
import tyro # 用来解析命令行参数

random.seed(2)


# =============================================================================
# 辅助函数，记录日志，处理文件路径等
# =============================================================================

class TeeStream:
    """Write output to both console and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_log_file(num_soc: int, num_stations: int) -> tuple[str, object]:
    """为当前运行设置日志文件。"""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(module_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_name = f"MPC_{num_soc}_{num_stations}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_name)
    log_file = open(log_path, 'a', encoding='utf-8')
    return log_path, log_file


def _resolve_data_path(path_str: str) -> str:
    """Resolve data file paths robustly regardless of current working directory."""
    if os.path.isabs(path_str):
        return path_str

    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)

    candidates = [
        os.path.join(project_root, path_str),
        os.path.join(module_dir, path_str),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class MPCConfig:
    """MPC模型配置参数"""
    
    # ---- 集合大小 ----
    num_stations: int = 3
    '''站点数量'''
    TimePeriods: int = 8
    '''预测时域T的时间段数量'''
    max_tau: int = 5
    '''
    最大路径行驶时间:用于记录还在行驶中的车辆（t<0）
    因为目前的数据中最长的行驶时间不超过5小时，所以暂时设置为5，后续可以根据数据调整
    '''
    N_soc: int = 5
    '''SOC状态数量，N = {0, 1, ..., N_soc}，其中N_soc为满电状态'''
    
    # ---- 成本参数 ----
    alpha: float = 300
    '''未满足预约需求的单位惩罚成本'''
    service_fee: float = 0.6
    '''
    换电服务费
    本来应该是按时间和站点变化的，但为了简化模型，暂时设置为一个固定值，后续可以根据数据调整
    '''
    
    # ---- 充电参数 ----
    beta: float = 25
    '''充电功率（单位：kW）'''
    # charge_slots: int = 10
    # '''每个站点的充电槽位数量M_i'''
    
    # ---- 电价参数 ----
    E: Tuple[float, ...] = (
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4, 1.4, 0.9, 0.9,
        0.9, 0.9, 0.9, 0.9, 0.9, 1.4, 1.4, 1.4, 0.9, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4, 1.4, 0.9, 0.9, 0.9,
    )
    '''分时电价，索引为t，一共38个时段，覆盖一天的24小时（0-23）和前一天的后5小时（16-24）'''
    
    # ---- 行驶参数 ----
    speed: float = 100.0
    '''假设时速100公里每小时'''

    R_range: float = 400.0
    '''续航里程（km）'''
    
    # ---- 数据文件路径 ----
    path_data_file: str = 'data_generation_optim/output/data_6e.json'
    '''路径数据文件'''
    # electricity_price_file: str = 'data_genaration_optim/output/electricity_price_6e.json'
    # '''电价数据文件'''
    dist_file: str = 'data_generation_optim/output/dist_6e.csv'
    '''距离数据文件'''
    demand_a_file: str = 'data_generation_optim/output/6e_flow_100_0.30.json'
    '''预约需求数据文件：O-D对p在时段t初始SOC为n的预约需求量 f^A_{n,p,t}'''
    demand_r_file: str = 'data_generation_rl/output/3s6e_station_demand_poisson_0d.json' # 只要第一天的就行
    '''随机需求数据文件：站点i在时段t初始SOC为n的随机需求量 f^R_{n,i,t}'''

    
    # ---- 求解参数 ----
    gurobi_time_limit: int = 3600
    '''Gurobi求解器的时间限制，单位秒'''
    
    # ---- 运行时生成的属性 ----
    stations: List[str] = field(init=False)
    '''站点列表S'''
    N_set: List[int] = field(init=False)
    '''SOC状态集合N = {0, 1, ..., N_soc}'''
    T_set: List[int] = field(init=False)
    '''时间集合T = {0, 1, ..., TimePeriods-1}'''
    T_ext: List[int] = field(init=False)
    '''扩展时间集合（含负时间，用于行驶中的车辆）'''
    
    def __post_init__(self):
        self.path_data_file = _resolve_data_path(self.path_data_file)
        self.dist_file = _resolve_data_path(self.dist_file)
        self.demand_a_file = _resolve_data_path(self.demand_a_file)
        self.demand_r_file = _resolve_data_path(self.demand_r_file)

        
        self.stations = [f's{i}' for i in range(1, self.num_stations + 1)]
        
        # SOC状态集合 N = {0, 1, ..., N_soc}
        self.N_set = list(range(self.N_soc + 1))
        # 时间集合
        self.T_set = list(range(self.TimePeriods))
        # 扩展时间集合（含行驶时间偏移）
        self.T_ext = list(range(self.TimePeriods))#暂时先不写，跑通了再说，后续需要考虑负时间段来记录还在行驶中的车辆


# =============================================================================
# 数据读取与预处理
# =============================================================================

def load_mpc_data(config: MPCConfig):
    """
    读取并预处理MPC模型所需的所有数据。
    
    返回:
        path_data_raw: 原始路径数据字典
        path_set: o-d pair 列表
        dist: 距离矩阵(DataFrame)
        tau: 行驶时间字典 {(p,i): tau_value}
        fA: 预约需求字典 {(n,p,t): demand_value}
        fR: 随机需求字典 {(n,i,t): demand_value}
        tau: 行驶时间字典 {(p,i): tau_value}
        A: 各路径的弧集合字典（按SOC状态n组织）{(n,p): [[i,j], ...]}
        Sp: 各路径经过的站点集合 {p: [stations]}
        source: 各路径起点 {p: origin_node}
        dest: 各路径终点 {p: dest_node}
        v_n_p: 预约用户待充电电量字典 {(n,p,i,j): v_value}
        v_n: 随机到达用户待充电电量字典 {(n,i): v_value}
        w_bar:初始电池库存字典 {(n,i,0): num}
        w_terminal: 末端满电电池库存下界{i: num}}
    """
    # ---- 读取路径数据 ----
    with open(config.path_data_file, 'r') as f:
        path_data_raw = json.load(f)

    A = {}

    for p in path_data_raw:
        A[p] = {}
        for n in config.N_set:  # 从SOC=2开始考虑预约用户的路径，因为SOC=0和1的用户不太可能上高速
            arcs = path_data_raw[p]['strategy'].get(str(n), [])
            A[p][n] = tuplelist(tuple(arc) for arc in arcs)
    
    Sp = {p: path_data_raw[p]['stations'] for p in path_data_raw}
    source = {p: path_data_raw[p]['source'] for p in path_data_raw}
    dest = {p: path_data_raw[p]['root'] for p in path_data_raw}

    # ---- 读取距离数据 ----
    dist = pd.read_csv(config.dist_file, header=0, index_col=0)

    # ---- 计算行驶时间 tau[p,i] ----
    tau = {}
    for p in path_data_raw:
        for i in path_data_raw[p]['stations']:
            tau[(p, i)] = math.ceil(dist.loc[path_data_raw[p]['source'], i] / config.speed)

    # ---- 读取需求数据 ----
    # 预约需求 fA[n,p,t]: O-D对p在时段t初始SOC为n的预约需求量
    fA = {}
    with open(config.demand_a_file, 'r') as file:
        json_data_fA = json.load(file)
    for key_str, value in json_data_fA.items():
        key = eval(key_str)  # 将字符串形式的元组转回元组
        fA[key] = value

    # 随机需求 fR[n,i,t]: 站点i在时段t初始SOC为n的随机需求量
    fR = {}
    with open(config.demand_r_file, 'r') as file:
        json_data_fR = json.load(file)
    for key_str, value in json_data_fR.items():
        key = eval(key_str)
        fR[key] = value

    
    # ---- 计算待重电量 ----
    # v_n_p[(n,p,i,j)]: 预约用户从站点i到j，需要补充的电量，用SOC表示
    # 比例关系ratio = 续航里程 / N_soc，即每个SOC单位对应的行驶距离
    # 分为两种情况:
    # 1. 当i是起点时，假设初始SOC为n，行驶距离为dist[i][j]，则需要补充的SOC单位数 = N_soc-n+ceil(dist[i][j] / ratio)
    # 2. 当i是站点时，假设用户到达时的SOC为n，行驶距离为dist[i][j]，则需要补充的SOC单位数 = ceil(dist[i][j] / ratio)
    path_set = list(path_data_raw.keys())
    ratio = config.R_range / config.N_soc
    v_n_p = {}
    for p in path_set:
        for n in config.N_set[2:]:  # 从SOC=2开始计算，因为里程焦虑，合理假设SOC=0和1的用户不会上高速
            for i, j in path_data_raw[p]['arcs']:
                if i in dist.index and j in dist.columns:
                    distance = dist.loc[i, j]
                    if i == source[p]:
                        # 1. 当i是起点时
                        v_n_p[(n, p, i, j)] = config.N_soc - n + math.ceil(distance / ratio)
                    else:
                        # 2. 当i是站点时
                        v_n_p[(n, p, i, j)] = math.ceil(distance / ratio)

    
    # v_n[n,i]: 到达站点i时电量为n的随机用户需要补充的电量
    # 假设随机用户只在电量快消耗完时换电，因此n的范围是config.N_set[0:3]
    v_n = {}
    for n in config.N_set[0:3]:
        for i in config.stations:
            v_n[(n, i)] = config.N_soc-n  # 需要补充的电量 = 满电状态 - 当前SOC状态
    
    
    # ---- 初始电池库存 w_bar[n,i,0] ----
    # 假设初始时所有电池都在满电状态
    w_bar = {}
    for n in config.N_set:
        for i in config.stations:
            if n == config.N_soc:
                w_bar[(n, i, 0)] = 21  # 满电电池初始数量
            else:
                w_bar[(n, i, 0)] = 0  # 非满电状态初始为0
    
    # ---- 终端库存约束下界 w_N_i^terminal ----
    w_terminal = {}
    for i in config.stations:
        w_terminal[i] = 2  # 至少保留2个满电电池
    
    return {
        'path_data_raw': path_data_raw,
        'path_set': path_set,
        'dist': dist,
        'tau': tau,
        'fA': fA,
        'fR': fR,
        'A': A,
        'Sp': Sp,
        'source': source,
        'dest': dest,
        'v_n_p': v_n_p,
        'v_n': v_n,
        'w_bar': w_bar,
        'w_terminal': w_terminal,
    }


# =============================================================================
# MPC模型建立
# =============================================================================

def BuildMPC(data: dict, cfg: MPCConfig):
    """
    建立手稿中的MPC数学模型。
    
    目标函数: max I - C1 - C2 - C3
      I   = 服务收益（公式2）
      C1  = 用户用电成本（公式3）
      C2  = 未满足预约需求惩罚（公式4）
      C3  = 电站充电成本（公式5）
    
    约束条件:
      (6a, 6b)  流平衡约束
      (7a, 7b)  电池状态转移约束
      (8a, 8b)  换电过程约束（预约优先）
      (9a-9i)   边界条件约束
    """
    
    # 解压数据
    path_set = data['path_set']
    tau = data['tau']
    fA = data['fA']
    fR = data['fR']
    A = data['A']
    Sp = data['Sp']
    source = data['source']
    dest = data['dest']
    v_n_p = data['v_n_p']
    v_n = data['v_n']
    s_fee = cfg.service_fee
    w_bar = data['w_bar']
    w_terminal = data['w_terminal']
    
    model = Model('MPC')
    
    # =========================================================================
    # 变量定义
    # =========================================================================
    
    # ---- yA[n,p,t,i,j]: 预约用户在弧(i,j)上的服务量 ----
    # 对应公式中的 y^A_{n,p,t,i,j}
    yA = {}
    for p in path_set:
        for n in cfg.N_set[2:]:
            for t in cfg.T_ext:
                for i, j in A[p][n]:
                    yA[(n, p, t, i, j)] = model.addVar(
                        lb=0, 
                        vtype=GRB.INTEGER,
                        name=f'yA_{n}_{p}_{t}_{i}_{j}'
                    )
    
    # ---- yR[n,i,t]: 随机用户在站点i的服务量 ----
    # 对应公式中的 y^R_{n,i,t}
    yR = {}
    for n in cfg.N_set[0:3]:  # 假设随机用户只在SOC=0,1,2时才会换电
        for i in cfg.stations:
            for t in cfg.T_set:
                yR[(n, i, t)] = model.addVar(
                    lb=0,
                    vtype=GRB.INTEGER,
                    name=f'yR_{n}_{i}_{t}'
                )
    
    # ---- w[n,i,t]: 站点i在时段t充电状态为n的电池数量 ----
    # 对应公式中的 w_{n,i,t}
    w = {}
    for n in cfg.N_set:
        for i in cfg.stations:
            for t in cfg.T_set:
                w[(n, i, t)] = model.addVar(
                    lb=0,
                    vtype=GRB.INTEGER,
                    name=f'w_{n}_{i}_{t}'
                )
    
    # =========================================================================
    # 约束条件
    # =========================================================================
    
    # ---- (6a) 起点出发的服务流量不超过预约需求 ----
    # Σ y^A_{n,p,t,op,j} ≤ f^A_{n,p,t}, ∀p∈P, t∈T, n∈N
    #                    {j|(op,j)∈A_{n,p}}
    for p in path_set:
        for t in cfg.T_set:
            for n in cfg.N_set[2:]:  # 从SOC=2开始考虑预约用户
                origin = source[p]
                outgoing_arcs = A[p][n].select(origin, '*')
                if len(outgoing_arcs) > 0:
                    model.addConstr(
                        quicksum(yA[(n, p, t, i, j)] for i, j in outgoing_arcs) <= fA.get((n, p, t), 0),
                        name=f'flow_limit_A_{n}_{p}_{t}'
                    )
    
    # ---- (6b) 路径上每个站点的流平衡约束 ----
    # Σ y^A_{n,p,t,i,j} = Σ y^A_{n,p,t,j,i}, ∀p∈P, t∈T, i∈S_p
    #  {j|(i,j)∈A_{n,p}}   {j|(j,i)∈A_{n,p}}
    for p in path_set:
        for t in cfg.T_set:
            for n in cfg.N_set[2:]:  # 从SOC=2开始考虑预约用户
                for i in Sp[p]:
                    in_arcs = A[p][n].select('*', i)
                    out_arcs = A[p][n].select(i, '*')
                    if len(in_arcs) > 0 or len(out_arcs) > 0:
                        model.addConstr(
                            quicksum(yA[(n, p, t, ii, jj)] for ii, jj in out_arcs)
                            == quicksum(yA[(n, p, t, ii, jj)] for ii, jj in in_arcs),
                            name=f'flow_balance_{n}_{p}_{t}_{i}'
                        )
    
    # ---- (7a) 电池状态转移约束（对SOC r < N） ----
    # w_{r,i,t} = Σ Σ Σ y^A_{n,p,t-τ_{p,i},j,i} · I(N - v^n_{j,i} = r)
    #            p∈P n∈N {j|(j,i)∈A_{n,p}}
    #           + Σ y^R_{n,i,t} · I(N - v^n_i = r)
    #            n∈N
    #           + Σ w_{m,i,t-1} · I(m + P_{m,i,t-1} = r)
    #            m∈N
    # ∀i∈S, r∈N\{N}, t∈T\{0}
    
    # 注：这里的充电转移 I(m + P = r) 简化为：当电池从状态m充电后变为状态r
    # 假设每时段充电可以使SOC增加1个单位（即 P = 1），则 I(m+1 = r)
    # 即：状态m的电池经过充电后变为状态m+1
    
    for i in cfg.stations:
        for t in cfg.T_set[1:]:
            for r in cfg.N_set:
                if r == cfg.N_soc:
                    continue  # 满电状态在(7b)中处理
                
                lhs = w[(r, i, t)]
                
                # 第一项：到达的预约用户退回的电池（SOC = r）
                rhs1 = LinExpr()
                for p in path_set:
                    for n in cfg.N_set[2:]:
                        # 计算用户从j到i后剩余的SOC
                        # 初始SOC为n，消耗v_n_p后剩余SOC
                        for j, ii in A[p][n].select('*', i):
                            if ii == i:
                                # 剩余SOC = N_soc_max - floor(v_n_p * N_soc_max)
                                soc_after = cfg.N_soc - math.floor(v_n_p.get((n, p, j, i), 0) * cfg.N_soc)
                                soc_after = max(0, min(soc_after, cfg.N_soc))
                                if soc_after == r:
                                    arr_time = t - tau.get((p, i), 0)
                                    if arr_time in cfg.T_ext:
                                        rhs1.add(yA[(n, p, arr_time, j, i)])
                
                # 第二项：随机用户退回的电池（SOC = r）
                rhs2 = LinExpr()
                for n in cfg.N_set[0:3]:
                    soc_after = cfg.N_soc - math.floor(v_n.get((n, i), 0) * cfg.N_soc)
                    soc_after = max(0, min(soc_after, cfg.N_soc))
                    if soc_after == r:
                        rhs2.add(yR[(n, i, t)])
                
                # 第三项：上一时刻状态m的电池充电后变为状态r
                rhs3 = LinExpr()
                for m in cfg.N_set:
                    # 假设充电功率使SOC每时段增加1
                    if m + 1 == r and m < cfg.N_soc:
                        rhs3.add(w[(m, i, t - 1)])
                
                model.addConstr(
                    lhs == rhs1 + rhs2 + rhs3,
                    name=f'battery_transfer_r{r}_i{i}_t{t}'
                )
    
    # ---- (7b) 满电电池状态转移约束 ----
    # w_{N,i,t} = w_{N,i,t-1} 
    #           + Σ w_{m,i,t-1} · I(m + P_{m,i,t-1} = N)   [充电到满电]
    #            m∈N
    #           - Σ Σ Σ y^A_{n,p,t-τ_{p,i},i,j}              [换走的满电电池]
    #            p∈P n∈N {j|(i,j)∈A_{n,p}}
    #           - Σ y^R_{n,i,t}                               [随机用户换走的]
    #            n∈N
    # ∀i∈S, t∈T\{0}
    
    for i in cfg.stations:
        for t in cfg.T_set[1:]:
            # 充电到满电的部分：上一时刻N-1状态的电池充电后变为满电
            charge_to_full = w[(cfg.N_soc - 1, i, t - 1)] if cfg.N_soc >= 1 else 0
            
            # 换走的满电电池（预约用户）
            swap_out_appointed = LinExpr()
            for p in path_set:
                for n in cfg.N_set[2:]:
                    for ii, j in A[p][n].select(i, '*'):
                        if ii == i:
                            arr_time = t - tau.get((p, i), 0)
                            if arr_time in cfg.T_ext:
                                swap_out_appointed.add(yA[(n, p, arr_time, i, j)])
            
            # 换走的满电电池（随机用户）
            swap_out_random = quicksum(yR[(n, i, t)] for n in cfg.N_set[0:3])
            
            model.addConstr(
                w[(cfg.N_soc, i, t)] == w[(cfg.N_soc, i, t - 1)] + charge_to_full
                - swap_out_appointed - swap_out_random,
                name=f'battery_transfer_full_i{i}_t{t}'
            )
    
    # ---- (8a) 换电过程约束（预约用户优先） ----
    # Σ y^R_{n,i,t} ≤ Σ w_{m,i,t-1} · I(m + P = N) - Σ Σ Σ y^A_{n,p,t-τ_{p,i},i,j}
    #  n∈N            m∈N                         p∈P n∈N {j|(i,j)∈A_{n,p}}
    # ∀i∈S, t∈T\{0}
    
    for i in cfg.stations:
        for t in cfg.T_set[1:]:
            # 可用的满电电池数量（上一时刻充电完成的）
            available_full = w[(cfg.N_soc, i, t - 1)]
            if cfg.N_soc >= 1:
                available_full += w[(cfg.N_soc - 1, i, t - 1)]
            
            # 预约用户需求（优先占用）
            appointed_demand = LinExpr()
            for p in path_set:
                for n in cfg.N_set[2:]:
                    for ii, j in A[p][n].select(i, '*'):
                        if ii == i:
                            arr_time = t - tau.get((p, i), 0)
                            if arr_time in cfg.T_ext:
                                appointed_demand.add(yA[(n, p, arr_time, i, j)])
            
            model.addConstr(
                quicksum(yR[(n, i, t)] for n in cfg.N_set[0:3])
                <= available_full - appointed_demand,
                name=f'swap_priority_i{i}_t{t}'
            )
    
    # ---- (8b) 随机用户需求约束 ----
    # 0 ≤ y^R_{n,i,t} ≤ f^R_{n,i,t}, ∀i∈S, n∈N, t∈T
    for n in cfg.N_set[0:3]:
        for i in cfg.stations:
            for t in cfg.T_set:
                model.addConstr(
                    yR[(n, i, t)] <= fR.get((n, i, t), 0),
                    name=f'random_demand_limit_{n}_{i}_{t}'
                )
    
    # ---- (9a) 初始电池库存约束 ----
    # w_{n,i,0} = w̄_{n,i,0}, ∀n∈N, i∈S
    for n in cfg.N_set:
        for i in cfg.stations:
            model.addConstr(
                w[(n, i, 0)] == w_bar.get((n, i, 0), 0),
                name=f'initial_battery_{n}_{i}'
            )
    
    # ---- (9b, 9c, 9d) 非负性约束 ----
    # 已在变量定义时通过lb=0保证
    
    # ---- (9e) 整数约束 ----
    # 已在变量定义时通过vtype=GRB.INTEGER保证
    
    # ---- (9h) 时间边界约束 ----
    # 若 k + τ_{p,i,k} ∉ T，则 y^A_{n,p,k,i,j} = 0
    for p in path_set:
        for n in cfg.N_set:
            for i, j in A[p][n]:
                if i in Sp[p]:  # 只考虑站点出发的弧
                    for t in cfg.T_ext:
                        if t < 0 or t >= cfg.TimePeriods:
                            model.addConstr(
                                yA[(n, p, t, i, j)] == 0,
                                name=f'time_boundary_{n}_{p}_{t}_{i}_{j}'
                            )
    
    # 额外：对t < 0的所有变量置零
    for p in path_set:
        for n in cfg.N_set:
            for t in range(-cfg.max_tau, 0):
                for i, j in A[p][n]:
                    if (n, p, t, i, j) in yA:
                        model.addConstr(
                            yA[(n, p, t, i, j)] == 0,
                            name=f'neg_time_zero_{n}_{p}_{t}_{i}_{j}'
                        )
    
    # ---- (9i) 终端库存约束 ----
    # w_{N,i,|T|-1} ≥ w^terminal_{N,i}, ∀i∈S
    for i in cfg.stations:
        model.addConstr(
            w[(cfg.N_soc, i, cfg.TimePeriods - 1)] >= w_terminal.get(i, 0),
            name=f'terminal_inventory_{i}'
        )
    
    # =========================================================================
    # 目标函数
    # =========================================================================
    
    # ---- I: 服务收益（公式2） ----
    # I = Σ Σ [ Σ Σ (y^A_{n,p,t-τ,i,j} · s_i,t) + Σ (y^R_{n,i,t} · s_i,t) ]
    #      t∈T n∈N  p∈P {(i,j)|(i,j)∈A_{n,p}}   i∈S
    
    income_swap = LinExpr()
    for t in cfg.T_set:
        for n in cfg.N_set[0:3]:
            # 预约用户服务费
            for p in path_set:
                for i, j in A[p][n]:
                    if i in Sp[p]:  # 只在站点处收取服务费
                        arr_time = t - tau.get((p, i), 0)
                        if arr_time in cfg.T_ext:
                            fee = s_fee
                            income_swap.add(yA[(n, p, arr_time, i, j)] * fee)
            # 随机用户服务费
            for i in cfg.stations:
                fee = s_fee
                income_swap.add(yR[(n, i, t)] * fee)
    
    # ---- C1: 用户用电成本（公式3） ----
    # C1 = Σ Σ [ Σ Σ (y^A_{n,p,t-τ,i,j} · v^n_{i,j} · e_i,t) + Σ (y^R_{n,i,t} · v^n_i · e_i,t) ]
    #       t∈T n∈N  p∈P {(i,j)|(i,j)∈A_{n,p}}   i∈S
    
    cost_user_electric = LinExpr()
    for t in cfg.T_set:
        for n in cfg.N_set[0:3]:
            for p in path_set:
                for i, j in A[p][n]:
                    if i in Sp[p]:
                        arr_time = t - tau.get((p, i), 0)
                        if arr_time in cfg.T_ext:
                            elec_cost = v_n_p.get((n, p, i, j), 0) * cfg.E[t]
                            cost_user_electric.add(yA[(n, p, arr_time, i, j)] * elec_cost)
            for i in cfg.stations:
                elec_cost = v_n.get((n, i), 0) * cfg.E[t]
                cost_user_electric.add(yR[(n, i, t)] * elec_cost)
    
    # ---- C2: 未满足预约需求惩罚（公式4） ----
    # C2 = α · Σ Σ Σ [ f^A_{n,p,t} - Σ y^A_{n,p,t,op,j} ]
    #           t∈T p∈P n∈N   {j|(op,j)∈A_{n,p}}
    
    penalty_unsatisfied = LinExpr()
    for t in cfg.T_set:
        for p in path_set:
            for n in cfg.N_set:
                demand = fA.get((n, p, t), 0)
                origin = source[p]
                served = LinExpr()
                for i, j in A[p][n].select(origin, '*'):
                    served.add(yA[(n, p, t, i, j)])
                penalty_unsatisfied.add(cfg.alpha * (demand - served))
    
    # ---- C3: 电站充电成本（公式5） ----
    # C3 = Σ Σ Σ [ w_{n,i,t} · P_{n,i,t} · e_i,t ]
    #       t∈T n∈N i∈S
    # 其中 P_{n,i,t} 是充电功率，假设只有n < N_soc_max的电池才需要充电
    
    cost_station_charge = LinExpr()
    for t in cfg.T_set:
        for n in cfg.N_set:
            if n < cfg.N_soc:  # 只有未满电的电池需要充电
                for i in cfg.stations:
                    cost_station_charge.add(w[(n, i, t)] * cfg.beta * cfg.E[t])
    
    # ---- 总目标函数 ----
    # max I - C1 - C2 - C3
    model.setObjective(
        income_swap - cost_user_electric - penalty_unsatisfied - cost_station_charge,
        GRB.MAXIMIZE
    )
    
    # 保存变量引用以便后续访问
    model._yA = yA
    model._yR = yR
    model._w = w
    
    model.update()
    return model


# =============================================================================
# 辅助功能函数
# =============================================================================

def extract_solution(model, data: dict, config: MPCConfig):
    """提取并格式化模型求解结果。"""
    if model.status != GRB.OPTIMAL and model.status != GRB.SUBOPTIMAL:
        print(f"模型未找到最优解，状态: {model.status}")
        return None
    
    path_set = data['path_set']
    yA = model._yA
    yR = model._yR
    w = model._w
    
    solution = {
        'objective_value': model.ObjVal,
        'runtime': model.Runtime,
        'gap': model.MIPGap if model.IsMIP else 0,
    }
    
    # 提取w变量（电池库存）
    w_sol = {}
    for n in cfg.N_set:
        for i in cfg.stations:
            w_sol[(n, i)] = [w[(n, i, t)].X for t in cfg.T_set]
    solution['w'] = w_sol
    
    # 提取yA变量（预约服务量）
    yA_sol = {}
    for p in path_set:
        for n in cfg.N_set[2:]:
            for t in cfg.T_set:
                for i, j in data['A'][p][n]:
                    val = yA[(n, p, t, i, j)].X
                    if val > 0.5:
                        yA_sol[(n, p, t, i, j)] = val
    solution['yA'] = yA_sol
    
    # 提取yR变量（随机服务量）
    yR_sol = {}
    for n in cfg.N_set[0:3]:
        for i in cfg.stations:
            for t in cfg.T_set:
                val = yR[(n, i, t)].X
                if val > 0.5:
                    yR_sol[(n, i, t)] = val
    solution['yR'] = yR_sol
    
    return solution


def save_solution(solution: dict, output_dir: str = None, filename: str = 'mpc_solution.json'):
    """保存求解结果到JSON文件。"""
    if solution is None:
        return
    
    module_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(module_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 将元组键转换为字符串以便JSON序列化
    solution_serializable = {
        'objective_value': solution['objective_value'],
        'runtime': solution['runtime'],
        'gap': solution['gap'],
    }
    
    if 'w' in solution:
        w_serial = {}
        for key, val in solution['w'].items():
            w_serial[str(key)] = val
        solution_serializable['w'] = w_serial
    
    if 'yA' in solution:
        yA_serial = {}
        for key, val in solution['yA'].items():
            yA_serial[str(key)] = val
        solution_serializable['yA'] = yA_serial
    
    if 'yR' in solution:
        yR_serial = {}
        for key, val in solution['yR'].items():
            yR_serial[str(key)] = val
        solution_serializable['yR'] = yR_serial
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(solution_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"Solution saved to: {filepath}")
    return filepath


def print_model_summary(model):
    """打印模型摘要信息。"""
    model.update()
    print("\n" + "="*60)
    print("MPC Model Summary")
    print("="*60)
    print(f"Model name: {model.ModelName}")
    print(f"Number of variables: {model.NumVars}")
    print(f"Number of integer variables: {sum(1 for v in model.getVars() if v.vType == GRB.INTEGER)}")
    print(f"Number of continuous variables: {sum(1 for v in model.getVars() if v.vType == GRB.CONTINUOUS)}")
    print(f"Number of constraints: {model.NumConstrs}")
    print(f"Objective sense: MAXIMIZE")
    print("="*60)


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    # 设置日志
    cfg = tyro.cli(MPCConfig)
    log_path, log_file = setup_run_log_file(cfg.N_soc, cfg.num_stations)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    
    try:
        print(f"Log file: {log_path}")
        print(f"MPC Model - N_soc={cfg.N_soc}, T={cfg.TimePeriods}, stations={cfg.num_stations}")
        print("-" * 60)
        
        # ---- 加载数据 ----
        print("Loading data...")
        stime = time.time()
        data = load_mpc_data(cfg)
        load_time = time.time() - stime
        print(f"Data loaded in {load_time:.2f} seconds")
        print(f"  - Path count: {len(data['path_set'])}")
        print(f"  - Stations: {cfg.stations}")
        print(f"  - SOC states: {cfg.N_set}")
        print(f"  - Time periods: {cfg.T_set}")
        
        # ---- 建立模型 ----
        print("\nBuilding MPC model...")
        stime = time.time()
        model = BuildMPC(data, cfg)
        build_time = time.time() - stime
        print(f"Model built in {build_time:.2f} seconds")
        
        # 打印模型摘要
        print_model_summary(model)
        
        # ---- 设置求解参数 ----
        model.setParam(GRB.Param.TimeLimit, cfg.gurobi_time_limit)
        model.setParam(GRB.Param.LogFile, log_path)
        model.setParam(GRB.Param.MIPGap, 0.01)  # 1% MIP gap
        
        # ---- 求解 ----
        print("\nSolving...")
        stime = time.time()
        model.optimize()
        solve_time = time.time() - stime
        
        # ---- 输出结果 ----
        print("\n" + "="*60)
        print("Optimization Results")
        print("="*60)
        
        if model.status == GRB.OPTIMAL:
            print(f"Status: OPTIMAL")
        elif model.status == GRB.SUBOPTIMAL:
            print(f"Status: SUBOPTIMAL")
        elif model.status == GRB.TIME_LIMIT:
            print(f"Status: TIME LIMIT REACHED")
        elif model.status == GRB.INFEASIBLE:
            print(f"Status: INFEASIBLE")
            # 计算IIS
            model.computeIIS()
            model.write("mpc_model.ilp")
            print("IIS written to mpc_model.ilp")
        else:
            print(f"Status: {model.status}")
        
        if model.SolCount > 0:
            print(f"Objective Value: {model.ObjVal:.2f}")
            print(f"Best Bound: {model.ObjBound:.2f}")
            print(f"MIP Gap: {model.MIPGap*100:.2f}%")
            print(f"Solve Time: {solve_time:.2f} seconds")
            
            # 提取并保存解
            solution = extract_solution(model, data, cfg)
            save_solution(solution)
        else:
            print("No feasible solution found.")
        
        print("="*60)
        
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
