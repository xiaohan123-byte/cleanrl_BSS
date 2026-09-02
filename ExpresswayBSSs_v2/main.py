'''
主要运行流程
定义需要的超参数
超参数网格搜索
'''

import os
import random
import time
import logging
from dataclasses import dataclass # 用于定义超参数的类
import tyro   # 用于解析命令行参数

# 定义超参数
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] #ppo
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True # 是否使用确定性的torch后端，设置为True可以确保实验的可重复性
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False # 其实这种小任务cpu更快
    """if toggled, cuda will be enabled by default"""
    track: bool = False # 是否使用Weights and Biases（WandB）进行实验跟踪
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""


    # Algorithm specific arguments
    env_id: str = "CartPole-v1"  
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4 # 并行环境的数量   如果是cpu采集的，可以多一些？
    """the number of parallel game environments"""
    num_steps: int = 128 # 表示每个环境在每次策略更新前运行的步数，代表了每个环境在每次策略更新前采集的数据量。总的批量大小将是 num_envs * num_steps。
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True # 设置学习率是否退火，即是否在训练过程中逐渐减小学习率
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99 # 折扣因子，用于计算未来奖励的折扣值 如果我希望折扣因子是变动的，如何修改源代码？
    """the discount factor gamma"""
    gae_lambda: float = 0.95 # GAE（Generalized Advantage Estimation）的λ参数，用于计算优势函数
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4 # 将采样的数据分成多少个小批量进行训练
    """the number of mini-batches"""
    update_epochs: int = 4 # 表示每个策略更新周期中，对策略网络进行几次梯度更新
    """the K epochs to update the policy"""
    norm_adv: bool = True # 是否对优势函数进行归一化处理
    """Toggles advantages normalization"""
    clip_coef: float = 0.2 # PPO 算法中的裁剪系数，用于限制策略更新的幅度
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False # 是否对价值函数的损失进行裁剪
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01 # 熵正则化系数，用于控制策略的探索性
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


if __name__=="__main__":
    args = tyro.cli(Args) # 解析命令行参数，生成Args实例

    # dir
    data_train_dir = "data/train"
    data_test_dir = "data/test"
    out_dir = "outputs/"
    run_dir = "runs/"
    log_dir = "logs/"

    



