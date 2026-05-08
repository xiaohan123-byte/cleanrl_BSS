# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
import logging
from dataclasses import dataclass # 用于定义超参数的类

import gymnasium as gym

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


def make_env(env_id, idx, capture_video, run_name):
    '''
    创建环境的函数，返回一个函数（thunk），
    当调用这个函数时会创建并返回一个环境实例。
    这个设计模式允许我们在需要时才创建环境实例，而不是在程序开始时就创建所有环境实例。
    SyncVectorEnv 内部会分别调用这 4 个 thunk 函数，才真正创建环境实例。
    '''
    def thunk():
        # 如果capture_video为True，并且idx为0（即第一个环境）
        # 则创建一个带有视频录制功能的环境实例，并将视频保存到指定的目录中。
        if capture_video and idx == 0:
            # 创建环境实例，并设置render_mode为"rgb_array"，以便能够捕获视频帧
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # 将环境包装在RecordEpisodeStatistics中，以便记录每个episode的统计信息，如总奖励和episode长度。这些统计信息可以在训练过程中用于监控和分析agent的表现。
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    '''
    定义了一个神经网络类Agent，继承自torch.nn.Module。
    这个类包含了一个critic网络和一个actor网络，分别用于估计状态价值和选择动作。critic网络输出一个标量值，actor网络输出一个动作概率分布。
     - critic网络由三个线性层组成，输入是环境的观测空间维度，输出是一个标量值。每个线性层后面都跟着一个Tanh激活函数。
     - actor网络也由三个线性层组成，输入是环境的观测空间维度，输出是一个动作概率分布。每个线性层后面也跟着一个Tanh激活函数，最后一层的输出维度是动作空间的维度。
     - get_value方法用于计算给定状态的价值函数值，get_action_and_value方法用于计算给定状态的动作概率分布、动作的对数概率、熵以及状态的价值函数值。
    '''
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0), # critic网络的最后一层输出一个标量值，表示状态的价值函数值，因此输出维度为1，权重初始化的标准差设置为1.0
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01), # actor网络的最后一层输出动作概率分布，因此输出维度为动作空间的维度，权重初始化的标准差设置为0.01
        )

    def get_value(self, x):
        '''
        计算给定状态的价值函数值，输入x是环境的观测，输出是一个标量值，表示状态的价值函数值。
         - 这个方法在训练过程中被调用，用于计算当前状态的价值函数值，以便在计算优势函数和更新策略网络时使用。
         - 这个方法也可以在测试过程中被调用，用于评估当前状态的价值函数值，以便在选择动作时使用。
        '''
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        '''
        计算给定状态的动作概率分布、动作的对数概率、熵以及状态的价值函数值。
         - 输入x是环境的观测，action是可选的动作，如果提供了action，则计算该动作的对数概率和熵；如果没有提供action，则从动作概率分布中采样一个动作。
         - 输出是一个元组，包含动作、动作的对数概率、熵以及状态的价值函数值。这个方法在训练过程中被调用，用于计算当前状态的动作概率分布、动作的对数概率、熵以及状态的价值函数值，以便在计算损失函数和更新策略网络时使用。这个方法也可以在测试过程中被调用，用于选择动作和评估当前状态的价值函数值。
         - 这个方法首先通过actor网络计算动作的logits，然后使用Categorical分布来计算动作的概率分布。如果没有提供action，则从这个分布中采样一个动作。最后返回动作、动作的对数概率、熵以及状态的价值函数值。
         - 这个方法的设计使得它既可以用于训练过程中的策略更新，也可以用于测试过程中的动作选择和状态评估，具有很好的灵活性和复用性。
        '''
        logits = self.actor(x)  # 状态x的logits，即未归一化的动作分数
        probs = Categorical(logits=logits)  # 使用Categorical做softmax来计算动作的概率分布
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args) # 解析命令行参数，生成Args实例
    args.batch_size = int(args.num_envs * args.num_steps) # 计算总的批量大小
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 计算每个小批量的大小
    args.num_iterations = args.total_timesteps // args.batch_size # 计算总的训练迭代次数
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # 将终端输出同步写入 logs/<run_name>.log
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("ppo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(f"logs/{run_name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # track取的是False，所以不会执行下面的代码块
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
    # 创建一个SummaryWriter实例，用于记录训练过程中的指标和超参数，以便在TensorBoard中可视化
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
    logger.info(f"Training on device: {device}")

    # env setup
    # 向量化环境，内部运行多个环境实例，并行采集数据，提高训练效率。每个环境实例由make_env函数创建，参数idx用于区分不同的环境实例，capture_video参数用于控制是否录制视频。
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )

    # 确保环境的动作空间是离散的，因为这个实现只支持离散动作空间。如果环境的动作空间不是离散的，程序会抛出一个AssertionError，并提示用户只能使用离散动作空间。
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # 初始化actor和critic网络参数---创建agent的时候
    agent = Agent(envs).to(device)
    # 使用Adam优化器来更新agent的参数，学习率和epsilon值根据args中的设置进行配置。Adam优化器是一种常用的梯度下降优化算法，适用于训练神经网络。
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # 状态初始化
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed) # 重置环境，获取四个环境的初始状态，并设置随机种子以确保实验的可重复性
    next_obs = torch.Tensor(next_obs).to(device) # 将初始状态转换为PyTorch张量，并将其移动到指定的设备（CPU或GPU）上
    next_done = torch.zeros(args.num_envs).to(device) # 四个环境的_done标志，初始值为False，表示每个环境都没有结束
    last_log_step = 0
    latest_episodic_return = None

    # 训练主循环
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        # 设置退火的学习率
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 数据采样---每个时间步长，同时从四个并行环境中采样一个动作，执行并记录奖励、_done标志、价值函数预测
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
                        latest_episodic_return = info["episode"]["r"]
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            if global_step - last_log_step >= 1000:
                sps = int(global_step / (time.time() - start_time))
                if latest_episodic_return is not None:
                    logger.info(f"global_step={global_step}, episodic_return={latest_episodic_return}")
                logger.info(f"SPS: {sps}")
                last_log_step = global_step


        # 计算优势与回报（使用GAE）
        # bootstrap value if not done
        with torch.no_grad(): # 禁用梯度计算，因为我们只需要前向传播来计算价值函数
            next_value = agent.get_value(next_obs).reshape(1, -1) # 预测最后一个观测的价值，作为未来奖励的估计【转化为一行四列的形式】
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            # 反向遍历是因为 按照GAE公式，t时刻的优势函数计算需要用到t+1时刻的价值函数预测和优势函数值，
            # 所以必须从后往前计算才能保证在计算t时刻的优势函数时，t+1时刻的值已经计算好了。
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    # 如果t是最后一个时间步长，使用bootstrap value来计算优势函数。bootstrap value是指在当前策略下，对未来奖励的估计值，通常是通过价值函数网络预测得到的。
                    nextnonterminal = 1.0 - next_done
                    # 如果环境在t+1时刻没有结束（即next_done为False），则nextnonterminal为1.0，表示未来奖励仍然有效；
                    # 如果环境在t+1时刻结束了（即next_done为True），则nextnonterminal为0.0，表示未来奖励不再有效。
                    nextvalues = next_value
                    # 这个bootstrap value将被用来计算t时刻的优势函数值，确保在环境结束时，优势函数能够正确地反映未来奖励的估计值。
                else:
                    # 如果t不是最后一个时间步长，使用t+1时刻的实际奖励和价值函数预测来计算优势函数。此时，nextnonterminal和nextvalues分别取自t+1时刻的dones和values。
                    nextnonterminal = 1.0 - dones[t + 1]
                    # 如果环境在t+1时刻没有结束（即dones[t + 1]为False），则nextnonterminal为1.0，表示未来奖励仍然有效；
                    # 如果环境在t+1时刻结束了（即dones[t + 1]为True），则nextnonterminal为0.0，表示未来奖励不再有效。
                    nextvalues = values[t + 1]
                    # 这个实际奖励和价值函数预测将被用来计算t时刻的优势函数值，确保在环境结束时，优势函数能够正确地反映未来奖励的实际情况。

                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                # TD误差项：实际得到的比预期好多少
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                # 这个公式中，delta是当前时间步长的TD误差，表示实际得到的奖励与预期奖励之间的差距。
                # lastgaelam是上一个时间步长的优势函数值，乘以折扣因子gamma和GAE参数lambda后加上当前的TD误差delta，得到当前时间步长的优势函数值。
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
        for epoch in range(args.update_epochs): # 在每个策略更新周期中，对策略网络进行4次梯度更新
            np.random.shuffle(b_inds) # 每次迭代前打乱数据顺序
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end] # 从打乱后的索引数组中取出一个minibatch的索引，用于从训练数据中选择对应的小批量数据进行训练
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds] # 计算新的动作概率与旧的动作概率的对数比率，表示当前策略相对于旧策略的变化程度
                ratio = logratio.exp() # 计算动作概率比率，表示当前策略相对于旧策略的变化程度。这个比率将被用来计算PPO算法中的损失函数，以限制策略更新的幅度，确保训练的稳定性。

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
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
