import math
from typing import Optional
import numpy as np
import sys
import gym
from gym import spaces
# from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from src.instancegenerator import InstanceGenerator
from src.BSSChargingScheduling import BSSChargingScheduling
import matplotlib.pyplot as plt
# 对数组中的指定部分进行降序排列
from scipy.optimize import minimize_scalar


def optimize_func(Q,Saved_Power,evaluate=True):
    reward_DR=np.zeros(Saved_Power.shape[0])
    reward_DR[Saved_Power<0.7*Q]=-2*Q
            
    reward_DR[(Saved_Power>0.7*Q) & (Saved_Power<=1.2*Q)]=\
        Saved_Power[(Saved_Power>0.7*Q) & (Saved_Power<=1.2*Q)]
    reward_DR[Saved_Power>1.2*Q]=\
        1.2*Q
    if evaluate:
        return -np.sum(reward_DR)
    else:
        return reward_DR
    
class MultipleBSS(gym.Env): #
    """
    ### Description
    State:Every SOC in that BSS(arranged in decending order)
            some future average demand in future(in a order)
            some electricity price in future
    Action: Total power of a BSS
    reward: electricity cost + battery swap income + power bigger than SOCmax
    这里为了直觉把batchsize放在了所有索引的最前方，但是实际上调用时经常不在最前，可能导致性能的下降
    Args:
        gym (_type_): _description_
    """
    
    def __init__(self,args ,other_args,MILP_Power,seed,batch_size,env_time_type):
        self.parameter=args.parameter
        self.BSS_Number=args.N
        self.batch_size=batch_size
        self.MILP_Power_Get=np.expand_dims(MILP_Power,axis=0)
        self.MILP_Power_Get=np.repeat(self.MILP_Power_Get,self.batch_size,axis=0)
        
        self.seed=seed
        self.env_time_type=env_time_type
        np.random.seed(self.seed)
        self.Battery_Number=np.array(args.Battery_Number)
        self.Charging_Slot_Number=np.array(args.Charging_Slot_Number)
        self.Demand=np.array(args.Demand,dtype=np.float32)
        self.TOU=np.array(args.TOU)
        self.Name=args.Name
        self.Fix_Price=np.array(args.Fix_Price)
        self.Use_Price=np.array(args.Use_Price)
        self.TOU=np.array(args.TOU,dtype=np.float32)
        self.Time_Period_Num=args.Time_Period_Num
        self.Base_Load=args.Base_Load
        self.Allocated_Peak_Shaving_Quantity=args.Allocated_Peak_Shaving_Quantity
        self.Optimized_Peak_Shaving_Quantity=self.Allocated_Peak_Shaving_Quantity.copy()
        self.Allow_Peak_Shaving=args.Allow_Peak_Shaving
        self.discrete_number=args.discrete_number
        self.if_discrete=args.if_discrete
        self.demand_type=args.demand_type
        self.tou_number=args.tou_number
        self.demand_number=args.demand_number
        self.decide_DR=args.decide_DR
        self.if_only_AgentRewad=other_args.if_only_AgentReward
        # self.num_demand=args.state_num_demand
        # self.num_price=args.state_num_price
        Low=[0 for i in range(sum(self.Battery_Number))]
        Low.extend([0 for i in range(self.demand_number*self.BSS_Number)])
        # Low.extend([np.min(self.Demand) for i in range(self.BSS_Number*self.num_demand)])
        # Low.extend([np.min(self.TOU) for i in range(self.BSS_Number*self.num_price)])
        self.time_used_number=np.int32(np.ceil(np.log2(self.Time_Period_Num)))
        Low.extend([0 for i in range(self.tou_number)])
        if self.env_time_type=="absolute":
            Low.extend([0])
        else:
            Low.extend([0 for _ in range(self.time_used_number)])
        High=[]
        for i in range(self.BSS_Number):
            
            High.extend([self.parameter.SOCmax for i in range(self.Battery_Number[i])])
            High.extend([np.max(self.Demand) for i in range(self.demand_number)])
        
            
        # High=[self.parameter.SOCmax for i in range(sum(self.Battery_Number))]
        # High.extend([])
        High.extend([np.max(self.TOU) for i in range(self.tou_number)])
        self.time_used_number=np.int32(np.ceil(np.log2(self.Time_Period_Num)))
        # High.extend([np.max(self.Demand) for i in range(self.num_demand)])
        # High.extend([np.max(self.TOU) for i in range(self.num_price)])
        if self.env_time_type=="absolute":
            High.extend([self.Time_Period_Num])
        else:
            High.extend([1 for _ in range(self.time_used_number)])
        self.Demand_Ins=np.zeros([self.batch_size,self.BSS_Number,self.Time_Period_Num])
        self.Demand_fullfill=np.zeros([self.batch_size,self.BSS_Number,self.Time_Period_Num])
        self.Heu_Demand_fullfill=np.zeros([self.batch_size,self.BSS_Number,self.Time_Period_Num])
        self.MILP_Demand_fullfill=np.zeros([self.batch_size,self.BSS_Number,self.Time_Period_Num])
        self.Heu_Power=np.zeros([self.batch_size,self.BSS_Number,self.Time_Period_Num])
        self.MILP_Power_True=np.zeros([self.batch_size,self.BSS_Number,self.Time_Period_Num])
        self.Reward_Sepa=np.zeros([self.batch_size,self.BSS_Number])
        self.Reward_DR=np.zeros(self.batch_size)
        self.Heu_Reward_Sepa=np.zeros([self.batch_size,self.BSS_Number])
        self.MILP_Reward_DR=np.zeros(self.batch_size)
        self.MILP_Reward_Sepa=np.zeros([self.batch_size,self.BSS_Number])
        self.Heu_Reward_DR=np.zeros(self.batch_size)
        self.Power=np.zeros([self.batch_size,self.BSS_Number,self.Time_Period_Num])
        self.low=np.array(Low, dtype=np.float32)
        self.high=np.array(High, dtype=np.float32)
        
        if self.if_discrete:
            self.action_space=spaces.MultiDiscrete([self.discrete_number+1 for i in range(self.BSS_Number)]) # 功率被均分为折磨多份
        else:
            self.action_space= spaces.Box(np.array([0 for i in range(self.BSS_Number)], dtype=np.float32),np.array([self.parameter.Power*self.Charging_Slot_Number[i] for i in range(self.BSS_Number)], dtype=np.float32),dtype=np.float32)
        # self.action_space= spaces.Box(np.array([0,for ], dtype=np.float32),np.array([self.parameter.Power*self.parameter.Charging_Slot_Number], dtype=np.float32),dtype=np.float32)
        self.observation_space= spaces.Box(self.low,self.high,dtype=np.float32)
        self.state= np.repeat(np.array(self.low, dtype=np.float32)[np.newaxis],self.batch_size,axis=0)
        self.bss_begin_index=np.zeros(self.BSS_Number+1,dtype=np.int32)
        begin=0
        for i in range(self.BSS_Number):
            self.bss_begin_index[i]=begin
            begin+=(self.Battery_Number[i]+self.demand_number)
        self.bss_begin_index[self.BSS_Number]=begin
        self.tool_batch=np.arange(self.batch_size)
        # print(self.action_space)
        # print(self.parameter)
        # print(self.Demand)
        # print(self.TOU)
        
    def step(self, action):  #see if seed in setup can control here? action=(batch_size,action_dim)
        
        # assert self.action_space.contains(
        #     action
        # ), f"{action!r} ({type(action)}) invalid"
        # time_slot_now=self.state[-1]
        
        self.time_slot_now=np.int32(self.time_slot_now)
        
        Demand_Instance= self.Demand[self.random_day,:,self.time_slot_now] # 这里维度是Nxbatch
        # Demand_Instance=np.transpose(Demand_Instance)
        reward= self.calculate(self.time_slot_now,Demand_Instance,action)
        self.update_Demand_and_TOU(Demand_Instance)
        self.time_slot_now=(self.time_slot_now+1)%self.Time_Period_Num
        binary_time=np.array([int(char) for char in np.binary_repr(self.time_slot_now, width=self.time_used_number)])
        
        
        self.state[:,-self.time_used_number:]=binary_time
        self.virtual_state[:,-self.time_used_number:]=binary_time
        self.MILP_state[:,-self.time_used_number:]=binary_time
        terminated=False
        if self.time_slot_now==0:
            terminated=True
        self.demand_instance=Demand_Instance
        

        return np.array(self.state,dtype=np.float32), reward, terminated, {}
    def calculate(self,time_slot,Demand,action):#这种先充soc大的可能并不是最优（烤面包机的情况） #这个里面bss——begin用的是soc开始的索引，后续如果要加别的状态的话，并不影响，在init里面改一下就好，还有别的状态也要加上更新
        # print("Im In")
        
        # print(remain_power)
        reward=np.zeros(self.batch_size)
        reward_virtual=np.zeros(self.batch_size)
        reward_sepa_BSS=np.zeros([self.batch_size,self.BSS_Number])
        reward_virtual_sepa_BSS=np.zeros([self.batch_size,self.BSS_Number])
        reward_virtual_DR=np.zeros(self.batch_size)
        reward_DR=np.zeros(self.batch_size)
        reward_MILP_sepa_BSS=np.zeros([self.batch_size,self.BSS_Number])
        reward_MILP_DR=np.zeros(self.batch_size)
        before_soc_states=0
        time_slot=np.int32(time_slot)
        
        # for bss_index in range(self.BSS_Number):
            
            # truly_charged_count=0
        if self.if_discrete:
            transfered_power=action/(self.action_space.nvec[0]-1)*self.parameter.Power*self.Charging_Slot_Number #这里如何乘 
        else:
            action[action<0]=0
            transfered_power=action
        # remain_power=transfered_power
        truly_allocated_power=self.allocate_power(transfered_power,if_virtual=False,if_MILP=False)

        self.Power[:,:,time_slot]=truly_allocated_power
        self._update_state_according_to_power(time_slot,if_virtual=False,if_MILP=False)

        
        # Then for virtual Power allocation
        virtual_transfered_power=np.ones([self.batch_size,self.BSS_Number])*self.parameter.Power*self.Charging_Slot_Number #这里如何乘 
        virtual_truly_allocated_power=self.allocate_power(virtual_transfered_power,if_virtual=True,if_MILP=False)
        self.Heu_Power[:,:,time_slot]=virtual_truly_allocated_power
        self._update_state_according_to_power(time_slot,if_virtual=True,if_MILP=False)

        # Then for MILP Power allocation
        MILP_allocated_power=self.MILP_Power_Get[:,:,time_slot]
        MILP_truly_allocated_power=self.allocate_power(MILP_allocated_power,if_virtual=True,if_MILP=True)
        self.MILP_Power_True[:,:,time_slot]=MILP_truly_allocated_power
        self._update_state_according_to_power(time_slot,if_virtual=True,if_MILP=True)
        
        #Charge Price for BSSs
        reward_sepa_BSS-=self.Power[:,:,time_slot]*self.TOU[time_slot]
        reward_virtual_sepa_BSS-=self.Heu_Power[:,:,time_slot]*self.TOU[time_slot]
        reward_MILP_sepa_BSS-=self.MILP_Power_True[:,:,time_slot]*self.TOU[time_slot]
        #----------------- Then satisfy demand ------------------  

        for i in range(self.BSS_Number):
            full_battery_after_i=np.zeros(self.batch_size)
            virtual_full_battery_after_i=np.zeros(self.batch_size)
            MILP_full_battery_after_i=np.zeros(self.batch_size)
            for _batch in range(self.batch_size): #有多少满电电池
                full_battery_after_i[_batch]=self.Battery_Number[i]-np.searchsorted(self.state[_batch,self.bss_begin_index[i]:self.bss_begin_index[i]+self.Battery_Number[i]][::-1],self.parameter.SOCmax-5e-3)
                virtual_full_battery_after_i[_batch]=self.Battery_Number[i]-np.searchsorted(self.virtual_state[_batch,self.bss_begin_index[i]:self.bss_begin_index[i]+self.Battery_Number[i]][::-1],self.parameter.SOCmax-5e-3)
                MILP_full_battery_after_i[_batch]=self.Battery_Number[i]-np.searchsorted(self.MILP_state[_batch,self.bss_begin_index[i]:self.bss_begin_index[i]+self.Battery_Number[i]][::-1],self.parameter.SOCmax-5e-3)
            income_SOC=self.parameter.SOCinit_min+(self.parameter.SOCinit_max-self.parameter.SOCinit_min)*np.random.random([self.batch_size,np.int32(np.max(Demand[:,i]))+1])
            # income_SOC= (self.parameter.SOCinit_max+self.parameter.SOCinit_min)/2*np.ones([self.batch_size,np.int32(np.max(Demand[:,i]))+1])
            true_satisfied_demand=np.int32(np.minimum(Demand[:,i],full_battery_after_i))
            virtual_true_satisfied_demand=np.int32(np.minimum(Demand[:,i],virtual_full_battery_after_i))
            MILP_true_satified_demand=np.int32(np.minimum(Demand[:,i],MILP_full_battery_after_i))
            for _batch in range(self.batch_size):
                reward_1=true_satisfied_demand[_batch]*self.Fix_Price[i,time_slot]+\
                    np.sum((self.state[_batch,self.bss_begin_index[i]:self.bss_begin_index[i]+true_satisfied_demand[_batch]]-income_SOC[_batch,0:true_satisfied_demand[_batch]]))*self.parameter.Battery_Volume*self.Use_Price[i,time_slot]
                reward_v=virtual_true_satisfied_demand[_batch]*self.Fix_Price[i,time_slot]+\
                    np.sum((self.virtual_state[_batch,self.bss_begin_index[i]:self.bss_begin_index[i]+virtual_true_satisfied_demand[_batch]]-income_SOC[_batch,0:virtual_true_satisfied_demand[_batch]]))*self.parameter.Battery_Volume*self.Use_Price[i,time_slot]
                reward_milp=MILP_true_satified_demand[_batch]*self.Fix_Price[i,time_slot]+\
                    np.sum((self.MILP_state[_batch,self.bss_begin_index[i]:self.bss_begin_index[i]+MILP_true_satified_demand[_batch]]-income_SOC[_batch,0:MILP_true_satified_demand[_batch]]))*self.parameter.Battery_Volume*self.Use_Price[i,time_slot]
                if reward_1.size==0:
                    reward_1=0
                if reward_v.size==0:
                    reward_v=0
                if reward_milp.size==0:
                    reward_milp=0
                reward_sepa_BSS[_batch,i]+=reward_1
                reward_virtual_sepa_BSS[_batch,i]+=reward_v
                reward_MILP_sepa_BSS[_batch,i]+=reward_milp
                
                #  about not satisfied demand
                reward_sepa_BSS[_batch,i]-=self.parameter.unit_customer_loss_price*(Demand[_batch,i]-true_satisfied_demand[_batch])
                reward_virtual_sepa_BSS[_batch,i]-=self.parameter.unit_customer_loss_price*(Demand[_batch,i]-virtual_true_satisfied_demand[_batch])
                reward_MILP_sepa_BSS[_batch,i]-=self.parameter.unit_customer_loss_price*(Demand[_batch,i]-MILP_true_satified_demand[_batch])
                self.Demand_fullfill[_batch,i,time_slot]=true_satisfied_demand[_batch]
                self.Heu_Demand_fullfill[_batch,i,time_slot]=virtual_true_satisfied_demand[_batch]
                self.MILP_Demand_fullfill[_batch,i,time_slot]=MILP_true_satified_demand[_batch]
            
            temp_max=max(true_satisfied_demand)
            indices=np.arange(temp_max)
            mask=indices<true_satisfied_demand[:,np.newaxis] #对使用的电池SOC进行替换
            self.state[:,self.bss_begin_index[i]:self.bss_begin_index[i]+temp_max][mask]=income_SOC[:,:temp_max][mask]
            
            temp_max=max(virtual_true_satisfied_demand)
            indices=np.arange(temp_max)
            mask=indices<virtual_true_satisfied_demand[:,np.newaxis]
            self.virtual_state[:,self.bss_begin_index[i]:self.bss_begin_index[i]+temp_max][mask]=income_SOC[:,:temp_max][mask]
           
            temp_max=max(MILP_true_satified_demand)
            indices=np.arange(temp_max)
            mask=indices<MILP_true_satified_demand[:,np.newaxis]
            self.MILP_state[:,self.bss_begin_index[i]:self.bss_begin_index[i]+temp_max][mask]=income_SOC[:,:temp_max][mask]
        self.Demand_Ins[:,:,time_slot]=Demand

        
        #-------------DR Reward---------------
        
            
        PS=self.Base_Load[time_slot]-np.sum(self.Power[:,:,time_slot],axis=1) #(batchsize,1)
        if self.Allow_Peak_Shaving[time_slot]==1:
            if self.decide_DR:
                if self.evaluate:
                    
                    reward_DR[PS<0.7*self.Optimized_Peak_Shaving_Quantity[time_slot]]=-2*self.parameter.unit_peak_shaving_price*self.Optimized_Peak_Shaving_Quantity[time_slot]
                    
                    reward_DR[(PS>0.7*self.Optimized_Peak_Shaving_Quantity[time_slot]) & (PS<=1.2*self.Optimized_Peak_Shaving_Quantity[time_slot])]=\
                        PS[(PS>0.7*self.Optimized_Peak_Shaving_Quantity[time_slot]) & (PS<=1.2*self.Optimized_Peak_Shaving_Quantity[time_slot])]*self.parameter.unit_peak_shaving_price
                    reward_DR[PS>1.2*self.Optimized_Peak_Shaving_Quantity[time_slot]]=\
                        1.2*self.Optimized_Peak_Shaving_Quantity[time_slot]*self.parameter.unit_peak_shaving_price
                else:
                    
                    self.Optimized_Peak_Shaving_Quantity[time_slot]=minimize_scalar(optimize_func,args=(PS),bounds=(0,max(max(PS),0)),method='bounded').x
                    if self.if_only_AgentRewad:
                        reward_sepa_BSS-=self.Power[:,:,time_slot]*self.parameter.unit_peak_shaving_price
                        reward_virtual_sepa_BSS-=self.Heu_Power[:,:,time_slot]*self.parameter.unit_peak_shaving_price
                        reward_MILP_sepa_BSS-=self.MILP_Power_True[:,:,time_slot]*self.parameter.unit_peak_shaving_price
                        reward_DR=optimize_func(self.Optimized_Peak_Shaving_Quantity[time_slot],PS,evaluate=False)*self.parameter.unit_peak_shaving_price
                    else:
                        reward_DR=optimize_func(self.Optimized_Peak_Shaving_Quantity[time_slot],PS,evaluate=False)*self.parameter.unit_peak_shaving_price
            else:
                if not self.evaluate:
                    if self.if_only_AgentRewad:
                        reward_sepa_BSS-=self.Power[:,:,time_slot]*self.parameter.unit_peak_shaving_price
                        reward_virtual_sepa_BSS-=self.Heu_Power[:,:,time_slot]*self.parameter.unit_peak_shaving_price
                        reward_MILP_sepa_BSS-=self.MILP_Power_True[:,:,time_slot]*self.parameter.unit_peak_shaving_price
                
                
                reward_DR[PS<0.7*self.Allocated_Peak_Shaving_Quantity[time_slot]]=-2*self.parameter.unit_peak_shaving_price*self.Allocated_Peak_Shaving_Quantity[time_slot]
                reward_DR[(PS>0.7*self.Allocated_Peak_Shaving_Quantity[time_slot])& (PS<=1.2*self.Allocated_Peak_Shaving_Quantity[time_slot])]\
                    =PS[(PS>0.7*self.Allocated_Peak_Shaving_Quantity[time_slot])& (PS<=1.2*self.Allocated_Peak_Shaving_Quantity[time_slot])]*self.parameter.unit_peak_shaving_price
                reward_DR[PS>1.2*self.Allocated_Peak_Shaving_Quantity[time_slot]]=1.2*self.Allocated_Peak_Shaving_Quantity[time_slot]*self.parameter.unit_peak_shaving_price
            # Virtual_PS=self.Base_Load[time_slot]-np.sum(self.Heu_Power[:,:,time_slot],axis=1)
            
            # reward_virtual_DR[Virtual_PS<0.7*self.Allocated_Peak_Shaving_Quantity[time_slot]]=-2*self.parameter.unit_peak_shaving_price*self.Allocated_Peak_Shaving_Quantity[time_slot]
            # reward_virtual_DR[(Virtual_PS>0.7*self.Allocated_Peak_Shaving_Quantity[time_slot])& (Virtual_PS<=1.2*self.Allocated_Peak_Shaving_Quantity[time_slot])]\
            #     =Virtual_PS[(Virtual_PS>0.7*self.Allocated_Peak_Shaving_Quantity[time_slot])& (Virtual_PS<=1.2*self.Allocated_Peak_Shaving_Quantity[time_slot])]*self.parameter.unit_peak_shaving_price
            # reward_virtual_DR[Virtual_PS>1.2*self.Allocated_Peak_Shaving_Quantity[time_slot]]=1.2*self.Allocated_Peak_Shaving_Quantity[time_slot]*self.parameter.unit_peak_shaving_price
            reward_virtual_DR=0
            MILP_PS=self.Base_Load[time_slot]-np.sum(self.MILP_Power_True[:,:,time_slot],axis=1)
            reward_MILP_DR[MILP_PS<0.7*self.Allocated_Peak_Shaving_Quantity[time_slot]]=-2*self.parameter.unit_peak_shaving_price*self.Allocated_Peak_Shaving_Quantity[time_slot]
            reward_MILP_DR[(MILP_PS>0.7*self.Allocated_Peak_Shaving_Quantity[time_slot])& (MILP_PS<=1.2*self.Allocated_Peak_Shaving_Quantity[time_slot])]\
                =MILP_PS[(MILP_PS>0.7*self.Allocated_Peak_Shaving_Quantity[time_slot])& (MILP_PS<=1.2*self.Allocated_Peak_Shaving_Quantity[time_slot])]*self.parameter.unit_peak_shaving_price
            reward_MILP_DR[MILP_PS>1.2*self.Allocated_Peak_Shaving_Quantity[time_slot]]=1.2*self.Allocated_Peak_Shaving_Quantity[time_slot]*self.parameter.unit_peak_shaving_price
        else:
            pass #since the reward is already set before
        self.step_sepa_reward=reward_sepa_BSS #----------------
        self.step_DR_reward=reward_DR
        
        self.Reward_Sepa+=reward_sepa_BSS
        self.Heu_Reward_Sepa+=reward_virtual_sepa_BSS
        self.MILP_Reward_Sepa+=reward_MILP_sepa_BSS
        self.Reward_DR+=reward_DR
        self.Heu_Reward_DR+=reward_virtual_DR
        self.MILP_Reward_DR+=reward_MILP_DR
        
            
        if self.Allow_Peak_Shaving[time_slot]==1:
            if not self.evaluate:
                if self.if_only_AgentRewad:
                    reward_sepa_BSS+=self.Power[:,:,time_slot]*self.parameter.unit_peak_shaving_price
                    reward_virtual_sepa_BSS+=self.Heu_Power[:,:,time_slot]*self.parameter.unit_peak_shaving_price
                    reward_MILP_sepa_BSS+=self.MILP_Power_True[:,:,time_slot]*self.parameter.unit_peak_shaving_price
                    # reward=reward=np.sum(reward_sepa_BSS,axis=1)+reward_DR
            #     else:
            #         reward=np.sum(reward_sepa_BSS,axis=1)+reward_DR
            # else:
            #     reward=np.sum(reward_sepa_BSS,axis=1)+reward_DR
        reward=np.sum(reward_sepa_BSS,axis=1)+reward_DR
        self.sort_state_soc()
        
        self.calculate_avail_an()
        # self.state[:,-1]=(time_slot+1)%self.Time_Period_Num
        # self.virtual_state[:,-1]=(time_slot+1)%self.Time_Period_Num

        return reward
    def update_Demand_and_TOU(self,Demand):
        for i in range(self.BSS_Number):
            # for _batch in range(self.batch_size):
            self.state[:,self.bss_begin_index[i]+self.Battery_Number[i]:self.bss_begin_index[i+1]]=np.roll(self.state[:,self.bss_begin_index[i]+self.Battery_Number[i]:self.bss_begin_index[i+1]],-1,axis=1)
            self.virtual_state[:,self.bss_begin_index[i]+self.Battery_Number[i]:self.bss_begin_index[i+1]]=np.roll(self.virtual_state[:,self.bss_begin_index[i]+self.Battery_Number[i]:self.bss_begin_index[i+1]],-1,axis=1)
            self.MILP_state[:,self.bss_begin_index[i]+self.Battery_Number[i]:self.bss_begin_index[i+1]]=np.roll(self.MILP_state[:,self.bss_begin_index[i]+self.Battery_Number[i]:self.bss_begin_index[i+1]],-1,axis=1)
            if self.demand_number>0:
                if self.demand_type=="Past":
                    self.state[:,self.bss_begin_index[i+1]-1]=Demand[:,i] 
                    self.virtual_state[:,self.bss_begin_index[i+1]-1]=Demand[:,i]
                    self.MILP_state[:,self.bss_begin_index[i+1]-1]=Demand[:,i]
                else:
                    self.state[:,self.bss_begin_index[i+1]-1]=np.mean(self.Demand[:,i,(self.time_slot_now+self.demand_number)%self.Time_Period_Num])
                    self.virtual_state[:,self.bss_begin_index[i+1]-1]=np.mean(self.Demand[:,i,(self.time_slot_now+self.demand_number)%self.Time_Period_Num])
                    self.MILP_state[:,self.bss_begin_index[i+1]-1]=np.mean(self.Demand[:,i,(self.time_slot_now+self.demand_number)%self.Time_Period_Num])
            # self.state[:,self.bss_begin_index[i+1]]=Demand[:,i]
        if self.tou_number>0:
            self.state[:,self.bss_begin_index[i+1]:self.bss_begin_index[i+1]+self.tou_number]=\
                np.roll(self.state[:,self.bss_begin_index[i+1]:self.bss_begin_index[i+1]+self.tou_number],-1,axis=1)
            self.virtual_state[:,self.bss_begin_index[i+1]:self.bss_begin_index[i+1]+self.tou_number]=\
                np.roll(self.virtual_state[:,self.bss_begin_index[i+1]:self.bss_begin_index[i+1]+self.tou_number],-1,axis=1)
            self.MILP_state[:,self.bss_begin_index[i+1]:self.bss_begin_index[i+1]+self.tou_number]=\
                np.roll(self.MILP_state[:,self.bss_begin_index[i+1]:self.bss_begin_index[i+1]+self.tou_number],-1,axis=1)
            time_slot=(self.time_slot_now+self.tou_number)%self.Time_Period_Num
            self.state[:,self.bss_begin_index[i+1]+self.tou_number-1]=self.TOU[time_slot]
            self.virtual_state[:,self.bss_begin_index[i+1]+self.tou_number-1]=self.TOU[time_slot]
            self.MILP_state[:,self.bss_begin_index[i+1]+self.tou_number-1]=self.TOU[time_slot]
                # self.Demand[_batch,i]=np.roll(self.Demand[_batch,i],-1)
                # self.Demand[_batch,i,-1]=self.Demand_Ins[_batch,i,self.time_slot_now]
    def calculate_avail_an(self):
    
        self.avail_action=np.ones([self.batch_size,self.BSS_Number,self.discrete_number+1])
        power_need_to_charge=[ (self.parameter.SOCmax-self.state[:,self.bss_begin_index[bss_index]:self.bss_begin_index[bss_index]+self.Battery_Number[bss_index]])/self.parameter.charging_efficiency* \
                self.parameter.Battery_Volume for bss_index in range(self.BSS_Number)]
        # cumsum_power_can_charge=[]
        transfered_power=np.ones([self.batch_size,self.BSS_Number])*self.parameter.Power*self.Charging_Slot_Number
        truly_allocated_power=np.zeros([self.batch_size,self.BSS_Number])
        for i in range(self.BSS_Number):
            power_need_to_charge[i][power_need_to_charge[i]>self.parameter.Power]=self.parameter.Power
            cumsum_power_can_charge_i=np.cumsum(power_need_to_charge[i],axis=1)#最后一维代表着最多能往这个电站冲的电量
            full_battery=np.zeros(self.batch_size,dtype=np.int32)
            for _batch in range(self.batch_size):
                full_battery[_batch]=np.searchsorted(cumsum_power_can_charge_i[_batch],5e-3, side='right') #((batch,1)
            temp_index=np.column_stack((self.tool_batch,np.minimum(self.Charging_Slot_Number[i]-1+full_battery,self.Battery_Number[i]-1)))
            truly_allocated_power[:,i]=np.minimum(cumsum_power_can_charge_i[temp_index[:,0],temp_index[:,1]],transfered_power[:,i])
        avail_indexs=np.ceil(truly_allocated_power*self.discrete_number/self.parameter.Power/self.Battery_Number)+1#直到这个都是可以的，再往上才不行
        indices = np.arange(self.avail_action.shape[2]) 
        mask = indices >= avail_indexs[:, :, np.newaxis]
        if self.if_discrete:
            
            self.avail_action[mask]=0
        else:
            self.avail_action=truly_allocated_power
            # self.avail_action[mask]=0
        return self.avail_action
        # print()
    
    def allocate_power(self,input_power,if_virtual=False,if_MILP=False):
        # power=input_power not virtual--state virtual not milp -virtaul-state  virtual milp milpstate
        remain_power=input_power
        if not if_virtual:
            power_need_to_charge=[ (self.parameter.SOCmax-self.state[:,self.bss_begin_index[bss_index]:self.bss_begin_index[bss_index]+self.Battery_Number[bss_index]])/self.parameter.charging_efficiency* \
                    self.parameter.Battery_Volume for bss_index in range(self.BSS_Number)]
        elif not if_MILP:
            power_need_to_charge=[ (self.parameter.SOCmax-self.virtual_state[:,self.bss_begin_index[bss_index]:self.bss_begin_index[bss_index]+self.Battery_Number[bss_index]])/self.parameter.charging_efficiency* \
                    self.parameter.Battery_Volume for bss_index in range(self.BSS_Number)]
        else:
            power_need_to_charge=[ (self.parameter.SOCmax-self.MILP_state[:,self.bss_begin_index[bss_index]:self.bss_begin_index[bss_index]+self.Battery_Number[bss_index]])/self.parameter.charging_efficiency* \
                    self.parameter.Battery_Volume for bss_index in range(self.BSS_Number)]
        truly_allocated_power=np.zeros(input_power.shape)
        for i in range(self.BSS_Number):
            power_need_to_charge[i][power_need_to_charge[i]>self.parameter.Power]=self.parameter.Power
            cumsum_power_can_charge_i=np.cumsum(power_need_to_charge[i],axis=1)#最后一维代表着最多能往这个电站冲的电量
            full_battery=np.zeros(self.batch_size,dtype=np.int32)
            for _batch in range(self.batch_size):
                full_battery[_batch]=np.searchsorted(cumsum_power_can_charge_i[_batch],5e-3, side='right') #((batch,1)
            temp_index=np.column_stack((self.tool_batch,np.minimum(self.Charging_Slot_Number[i]-1+full_battery,self.Battery_Number[i]-1)))
            truly_allocated_power[:,i]=np.minimum(cumsum_power_can_charge_i[temp_index[:,0],temp_index[:,1]],input_power[:,i])
        return truly_allocated_power
    def _update_state_according_to_power(self,time_slot,if_virtual=False,if_MILP=False):
        for i in range(self.BSS_Number):
            if not if_virtual:
                Batch_SOC=self.Power[:,i,time_slot]*self.parameter.charging_efficiency \
                /self.parameter.Battery_Volume#够充多少SOC的电量

            elif not if_MILP:
                Batch_SOC=self.Heu_Power[:,i,time_slot]*self.parameter.charging_efficiency \
                /self.parameter.Battery_Volume
            else:
                Batch_SOC=self.MILP_Power_True[:,i,time_slot]*self.parameter.charging_efficiency \
                /self.parameter.Battery_Volume
            for j in range(self.Battery_Number[i]):
                
                
                if not if_virtual:
                    used_soc=np.minimum(np.minimum(Batch_SOC,self.parameter.SOCmax-self.state[:,self.bss_begin_index[i]+j]\
                    ),self.parameter.Power*self.parameter.charging_efficiency/self.parameter.Battery_Volume)
                    self.state[:,self.bss_begin_index[i]+j]+=used_soc
                    Batch_SOC=Batch_SOC-used_soc
                elif not if_MILP:
                    used_soc=np.minimum(np.minimum(Batch_SOC,self.parameter.SOCmax-self.virtual_state[:,self.bss_begin_index[i]+j]\
                    ),self.parameter.Power*self.parameter.charging_efficiency/self.parameter.Battery_Volume)
                    self.virtual_state[:,self.bss_begin_index[i]+j]+=used_soc
                    Batch_SOC=Batch_SOC-used_soc
                else:
                    used_soc=np.minimum(np.minimum(Batch_SOC,self.parameter.SOCmax-self.MILP_state[:,self.bss_begin_index[i]+j]\
                    ),self.parameter.Power*self.parameter.charging_efficiency/self.parameter.Battery_Volume)
                    self.MILP_state[:,self.bss_begin_index[i]+j]+=used_soc
                    Batch_SOC=Batch_SOC-used_soc
        # cumsum_power_can_charge=[]
    def sort_state_soc(self):
        before_soc_states=0
        for bss_index in range(self.BSS_Number):
            self.state[:,before_soc_states:before_soc_states+self.Battery_Number[bss_index]] = np.sort(self.state[:,before_soc_states:before_soc_states+self.Battery_Number[bss_index]],axis=1)[:,::-1] 
            self.virtual_state[:,before_soc_states:before_soc_states+self.Battery_Number[bss_index]] = np.sort(self.virtual_state[:,before_soc_states:before_soc_states+self.Battery_Number[bss_index]],axis=1)[:,::-1] 
            self.MILP_state[:,before_soc_states:before_soc_states+self.Battery_Number[bss_index]] = np.sort(self.MILP_state[:,before_soc_states:before_soc_states+self.Battery_Number[bss_index]],axis=1)[:,::-1]
            before_soc_states+=(self.Battery_Number[bss_index]+self.demand_number)
    def reset(
        self,
        evaluate: bool = False,
        calculate_base:bool=False,
        *,
        
        options: Optional[dict] = None,
    ):

        
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        self.evaluate=evaluate
        self.Reward_DR=np.zeros(self.batch_size)
        self.Reward_Sepa=np.zeros([self.batch_size,self.BSS_Number])
        self.Heu_Reward_DR=np.zeros(self.batch_size)
        self.Heu_Reward_Sepa=np.zeros([self.batch_size,self.BSS_Number])
        self.MILP_Reward_DR=np.zeros(self.batch_size)
        self.MILP_Reward_Sepa=np.zeros([self.batch_size,self.BSS_Number])
        self.state= np.repeat(np.array(self.low, dtype=np.float32)[np.newaxis],self.batch_size,axis=0)
        if calculate_base:
            self.random_day=np.random.choice(np.arange(np.size(self.Demand, 0)), self.batch_size)
        else:
            if evaluate:
        # 选择 Demand 的后 8 个，重复 n 次直到总大小为 batch_size
                self.random_day = np.repeat(np.arange(np.size(self.Demand, 0) - 8, np.size(self.Demand, 0)), self.batch_size // 8)
                if self.batch_size % 8 != 0:  # 如果 batch_size 不是 8 的倍数，需要额外添加一些元素
                    self.random_day = np.concatenate([self.random_day, np.random.choice(np.arange(np.size(self.Demand, 0) - 8, np.size(self.Demand, 0)), self.batch_size % 8)])
            else:
        # 随机选择 Demand 的前 22 个，可重复地总共选 batch_size 个
                self.random_day = np.random.choice(np.arange(np.size(self.Demand, 0) - 8), self.batch_size)
        count_full=0
        before_soc_states=0
        for bss_index in range(self.BSS_Number):
            count_full=0
            for i in range(self.Battery_Number[bss_index]):
                if count_full<self.parameter.Init_Battery:
                    self.state[:,before_soc_states+i]=self.parameter.SOCmax
                    count_full+=1
                else:
                    
                    self.state[:,before_soc_states+i]=self.parameter.SOCinit_min+(self.parameter.SOCinit_max-self.parameter.SOCinit_min)*np.random.rand(self.batch_size)
        
            #-----------set demand----------
            if self.demand_number>0:
                if self.demand_type=="Past":
                    self.state[:,before_soc_states+self.Battery_Number[bss_index]:before_soc_states+self.Battery_Number[bss_index]+self.demand_number]=self.Demand[self.random_day,bss_index,-self.demand_number:]
                else:
                    self.state[:,before_soc_states+self.Battery_Number[bss_index]:before_soc_states+self.Battery_Number[bss_index]+self.demand_number]=np.mean(self.Demand[self.random_day,bss_index,:self.demand_number],0)
            before_soc_states+=self.Battery_Number[bss_index]+self.demand_number
        

        #-----------set TOU---------- 只有一种可能，看未来的
        if self.tou_number>0:
            # if self.tou_number==1:
            #     self.state[:,-self.tou_number]=self.TOU[0]
            # else:
                
            self.state[:,self.bss_begin_index[bss_index+1]:self.bss_begin_index[bss_index+1]+self.tou_number]=self.TOU[:self.tou_number]
           
        self.time_slot_now=0
        # self.state[:,-1]=0
        
        self.virtual_state=self.state.copy()
        self.MILP_state=self.state.copy()
        self.sort_state_soc()
        
        # self.calculate_avail_an()
        # self.state[0:self.parameter.Battery_Number] = np.sort(self.state[0:self.parameter.Battery_Number])[::-1] 
        # start_index=self.parameter.Battery_Number
        # if self.num_demand>0:
        #     self.state[start_index:start_index+self.num_demand]= np.average(self.Demand[:,:self.num_demand],0)
        
        # start_index=self.parameter.Battery_Number+self.num_demand
        # if self.num_price>0:
        #     self.state[start_index:start_index+self.num_price]= self.TOU[:self.num_price]
        

        return self.state
        
    def get_obs(self):
        # before_soc=0
        obs_n={}
        for i in range(self.BSS_Number):
            obs_n[f"{i}"]=np.column_stack((self.state[:,self.bss_begin_index[i]:self.bss_begin_index[i+1]],self.state[:,-(self.time_used_number+self.tou_number):]))
            
            # before_soc+=self.Battery_Number[i]

        return obs_n
    def get_group_state(self,group):
        group_state={}
        for i in range(len(group)):
            temp=np.column_stack([self.state[:,self.bss_begin_index[group[i][_in]]:self.bss_begin_index[group[i][_in]+1]] for _in in range(len(group[i]))])
            group_state[f"{i}"]=np.column_stack((temp,self.state[:,-self.time_used_number:]))
        return group_state
    def get_state(self):
        return self.state
    def render(self):
        pass
    
    def close(self):
        pass
    




if __name__ =="__main__":
    
    abs_path='D:\Myfiles\vscode_files\cleanrl_BSS'
    IF=InstanceGenerator(0,1,24)
    np.random.seed(1)
    BSSs=IF.load_inst(abs_path+"./res/test_inst/B5-0.json")
    BSSs.parameter.Power=BSSs.parameter.Power/(len(BSSs.TOU)/24)
    BSSs.discrete_number=10
    Env=MultipleBSS(BSSs)
    Env.reset()
    for i in range(96):
        Env.step([10 for i in range(5)])
    # print()