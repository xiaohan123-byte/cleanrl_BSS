# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass # 用于定义超参数的类

import gymnasium as gym
from gymnasium import spaces # 用于定义环境的动作空间和状态空间
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro   # 用于解析命令行参数
from torch.distributions.categorical import Categorical # 用于离散动作空间
from torch.utils.tensorboard import SummaryWriter    # 在线显示训练中的指标

# 定义超参数
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] #ppo
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True # 是否使用确定性的torch后端，设置为True可以确保实验的可重复性
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
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
    env_id: str = "ExpresswayBSS"   # 改成expresswayBSS
    """the id of the environment: ExpresswayBSS/CartPole-v1"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4 # 并行环境的数量   如果是cpu采集的，可以多一些？
    """the number of parallel game environments"""
    num_steps: int = 128 # 表示每个环境在每次策略更新前运行的步数
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
    clip_vloss: bool = True # 是否对价值函数的损失进行裁剪
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


## 自定义环境
###################################################################################
class MyCustomEnv(gym.Env):
    """
    针对现实场景的自定义环境，从 gym.Env 继承，实现自定义的环境逻辑。
    """
    def __init__(self, config=None):
        super(MyCustomEnv, self).__init__()
        
        # 1. 定义动作空间 (Action Space)
        # 例如：控制一个阀门的开度 (-1.0 到 1.0) -> 连续空间
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 或者：开关三个机器 (0, 1, 2) -> 离散空间
        self.action_space = spaces.Discrete(3)

        # 2. 定义观测空间 (Observation Space)
        # 例如：温度、压力、流量等 5 个传感器数据
        # 一定要定义好数据的上下界，如果不知道就写 -inf 到 inf
        self.observation_space = spaces.Box(
            low=0, 
            high=21, 
            shape=(5,), 
            dtype=np.float32
        )

        # 初始化内部状态变量
        self.state = None

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态
        """
        # 必须调用 super().reset(seed=seed) 来处理随机种子
        super().reset(seed=seed)
        
        # TODO: 在这里编写你的业务逻辑，获取初始状态
        self.state = np.zeros(5, dtype=np.float32) 
        
        # 获取初始观测值
        observation = self.state
        info = {} # 可以返回一些辅助信息，不参与训练
        
        return observation, info

    def step(self, action):
        """
        执行动作，返回 (next_obs, reward, terminated, truncated, info)
        """
        # TODO: 1. 根据 action 更新你的现实场景或仿真逻辑
        # self.state = simulation.update(action) ...
        
        # TODO: 2. 计算 Reward (非常关键！)
        reward = -abs(self.state[0] - target_value) # 举例：越接近目标，惩罚越小
        
        # TODO: 3. 判断是否结束
        terminated = False # 任务是否完成（如到达终点）
        if reward > -0.1:
            terminated = True
            
        truncated = False # 是否超时（通常由 Wrapper 处理，这里设为 False 即可）
        
        # TODO: 4. 获取新的观测
        observation = self.state.astype(np.float32)
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        (可选) 用于可视化，如果是现实数据流，可以打印日志或画图
        """
        pass

    def close(self):
        """
        (可选) 清理资源，如关闭数据库连接或硬件端口
        """
        pass



###################################################################################

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id) # 就是环境名称
        env = gym.wrappers.RecordEpisodeStatistics(env) # 记录每一轮的统计信息，比如总奖励、长度等
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args) # 解析命令行参数，生成Args实例
    args.batch_size = int(args.num_envs * args.num_steps) # 计算总的批量大小
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 计算每个小批量的大小
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # 在这里设置BSS环境
    # 注意，是创建了多个并行环境
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    
    # 确保动作空间是离散的
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # 初始化actor和critic网络参数---创建agent的时候
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # 训练主循环
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        # 设置退火的学习率
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 数据采样---每个时间步长，从环境中采样一个动作，执行并记录奖励、_done标志、价值函数预测
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)


        # 计算优势与回报（使用GAE）
        # bootstrap value if not done
        with torch.no_grad(): # 禁用梯度计算，因为我们只需要前向传播来计算价值函数
            next_value = agent.get_value(next_obs).reshape(1, -1) # 预测最后一个观测的价值，作为未来奖励的估计
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # 将收集得到的训练数据展平处理，方便批量训练
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 更新策略和价值网络
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size) # 生成一个从0到batch_size-1的数组，用于随机采样
        clipfracs = [] # 记录每个minibatch的动作概率比率被裁剪的比例，用于监控训练进度
        for epoch in range(args.update_epochs): # 迭代训练
            np.random.shuffle(b_inds) # 每次迭代前打乱数据顺序
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
