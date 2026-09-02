# `main.tex` 科研论文成稿化修订执行规范

本文件是后续修改 `paper/main.tex` 的唯一执行依据。它将已经确认的论文结构、模型接口、正文与附录分工、图位处理和验收标准固化为确定规则。执行时不得重新引入本文件已排除的旧逻辑，也不得在正文中描述修订过程。

## 0. 执行边界与写作原则

### 0.1 文件边界

- 后续实施只修改 `paper/main.tex`。
- 不修改 `paper/figures/MPC-RL.png`、`paper/figures/rolling_horizon.png` 或其他旧图片。
- 不覆盖已跟踪的 `paper/main.pdf`；编译产物写入临时目录。
- 不修改 `reference.bib`、`SJTUReport.sty`、代码、数据或其他论文文件。
- 不新增未经核实的参考文献、实验数据、性能结论或运营机制。

### 0.2 最终论文口吻

论文正文只陈述最终研究问题、模型和方法，不出现以下可见内容：

- “原稿”“上一版”“本次修订”“修改为”“继续保留”等版本叙述；
- “待补充”“在此填写”“编辑意见”“以后加入”等工作过程文字；
- 方括号或圆括号中的编辑批注；
- 对旧模型按时段处理方式的批评性修订说明。

计划文件可以说明哪些原文应保留或重写，但这些说明不得复制进论文正文。

### 0.3 原文保留优先级

1. 与最终机制完全一致的原文字句，在改善语句和符号一致性后保留。
2. 概念有效但时间、用户状态或符号不一致的段落，采用局部改写。
3. 与在途剩余路径优化、连续事件、等待携带或真实记账冲突的内容，删除并重写。
4. 候选网络四步生成法、日前路径三步规则、综合目标结构、critic 终端价值思路和 PPO/GAE 内容属于优先保留部分。

### 0.4 可读性硬约束

- 正文只呈现研究机制和核心等式。
- 候选事件枚举、indicator、Big-\(M\)、槽位匹配、事件排序和边界线性化放入附录。
- 路径公式沿用预约用户索引 \((p,k)\)，不引入全局 \(u=(p,k)\) 别名。
- 站内服务使用通用请求索引 \(r\)，避免在正文反复出现预约和随机用户的复合角标。
- 正文不出现 `dec/fix/wait/out` 多重复合集合或四重下标的匹配变量。
- 正文符号表控制在约 12--15 个核心条目；只在附录使用的符号放入附录符号说明。
- 每个符号首次出现时就近解释；同一符号只承担一种语义。
- 单个正文公式尽量控制在两行以内，长求和、完整变量域和线性化展开移入附录。

## 1. 最终论文结构

### 1.1 必须采用的 LaTeX 层级

```latex
\section{Introduction}

\section{Problem Statement}

\section{MPC Formulation}
% 本节开头使用自然段介绍滚动信息、调用顺序和连续执行
\subsection{Offline Candidate networks by SOC Range}
\subsection{Day-Ahead Baseline Path Generation}
\subsection{Optimization Model}
\paragraph{Decision variables.}
\paragraph{Objective function.}
\paragraph{Constraints.}

\section{RL Formulation}
\subsection{Markov decision process}
\subsection{RL algorithm}

\section{Numerical experiments}

\section{Conclusion}

\appendix
\section{Candidate Events and Effective Arrival Chains}
\section{Waiting, Priority, and Timeout Constraints}
\section{Continuous Charging and Slot Matching}
\section{Publication, Boundary State, and Accounting}
```

### 1.2 结构迁移规则

- 将当前 `\section{Problem Statement and MPC Formulation}` 拆成两个独立的一级章节。
- 删除 `\subsubsection{Rolling-Horizon Information and Execution}` 标题，其有效内容并入 `MPC Formulation` 开头。
- `Offline Candidate networks by SOC Range`、`Day-Ahead Baseline Path Generation` 和 `Optimization Model` 保持标题名称不变，从 `subsubsection` 提升为 `subsection`。
- `Optimization Model` 内只使用三个 `paragraph` 标题，不再增加小节层级。
- `RL Formulation` 保留现有两个 `subsection`；State、Action、Reference rollout、Transition 和 Reward 使用 `paragraph`。
- 将 `Experiments` 改为 `Numerical experiments`，不增加 `Model Validation`。
- Introduction 的全文结构段按上述六个正文一级章节重新编号。

## 2. 模型与符号的最终口径

本节中的规则是后续写作的模型接口，不是可选方案。

### 2.1 时间、区间与预测边界

外层区间长度记为 \(\sigma\)，最大等待时间记为 \(\Delta t\)：

\[
t_q=q\sigma,
\qquad
\mathcal I_q=[t_q,t_{q+1}),
\qquad
\mathcal T_\ell=\{\ell,\ldots,\ell+H-1\}.
\]

- \(\mathcal T_\ell\) 恰好包含 \(H\) 个区间，预测域为 \([t_\ell,t_{\ell+H})\)。
- \(\ell,q\) 是区间索引，\(t_\ell,t_q\) 是连续时刻。
- 预约与随机请求的到站时刻均为外生连续值；区间索引只用于归组、价格和能量限额。
- 恰好发生在 \(t_{\ell+H}\) 的到站、充满、服务或超时事件由下一轮处理。
- 终端 SOC 使用事件前快照 \(S_{i,b}(t_{\ell+H}^{-})\)，端点事件不得跨轮重复执行或结算。

### 2.2 用户状态与剩余路径

- 保留未到用户集合 \(\mathcal K_\ell^{\mathrm{fut},p}\)。
- 使用在途集合 \(\mathcal K_\ell^{\mathrm{enr},p}\)，不再使用 `arr/fix` 作为路径决策划分。
- 站内等待用户属于在途集合，附加当前站点、请求 ID、原到站时刻和截止时刻，不另立路径用户类型。
- 未到用户从入口 \(o^p\) 出发，路径调整基准始终是日前初始发布路径。
- 在途用户从实时虚拟起点 \(o_\ell^{\mathrm{cur},p,k}\) 出发，路径调整基准是最近一次实际发布路径的未执行站序。
- 已执行路径前缀不进入本轮决策或路径差异比较。
- 虚拟起点移动而剩余站序不变时，不产生路径调整成本。
- 在途剩余站序变化时，在本轮边界立即发布，结算一次路径调整成本，并将新站序作为下一轮参考。
- 未到用户的内部预测路径不发布、不覆盖日前路径，也不进入当前实际 reward。
- 已在站等待的当前请求必须保留，不能通过重新选择路径取消。
- 上游预约请求超时后，下游预约事件失活，只结算真实发生的一次预约失败。

### 2.3 候选网络与退回 SOC

- 离线 SOC 分档网络继续提供下游站间连接骨架。
- 在线剩余网络删除已经经过的节点和弧，并从实时虚拟起点补充首段弧。
- 虚拟首段依据实时位置、实时 SOC 和求解前已知的连续 ETA 筛选；不建立随机交通模型。
- 最小换电间距 \(\underline D\) 以入口或最近一次真实换电位置为基准，不把每轮移动的虚拟起点解释为刚完成换电。
- 路径变量继续使用 \(y_{i,j}^{p,k}\)，不增加滚动轮次下标。
- 附录预约候选事件保持 \(\varepsilon=(p,k,j,i)\)，以入弧 \((j,i)\) 唯一确定退回 SOC。
- 未到用户首段使用有效入口 SOC，在途首段使用实时 SOC，后续站间弧使用上一站成功换得的满电 SOC。

### 2.4 路径调整与发布

正文保留站点访问量和固定一次调整成本：

\[
x_{A,i}^{p,k}
=
\sum_{j:(j,i)\in\mathcal A^{p,k}}y_{j,i}^{p,k},
\qquad
C_{\mathrm{adj}}
=
\kappa\sum_{p}\sum_{k\in\mathcal K_\ell^p}d_{p,k}.
\]

- 未到用户的参考访问量来自日前初始路径。
- 在途用户的参考访问量来自最近发布的未执行站序。
- 比较范围为当前候选实体站点和参考未执行站点的并集；缺失访问量补零。
- 实际 reward 只统计本轮真实发布的在途路径调整。

### 2.5 等待、优先级与同时刻顺序

通用请求 \(r\) 使用：

\[
t_r^{\mathrm{ddl}}
=
t_r^{\mathrm{arr}}+\Delta t.
\]

- 服务必须满足 \(t_r^{\mathrm{arr}}\le t_r^{\mathrm{sw}}\le t_r^{\mathrm{ddl}}\)。
- 预约优先于随机用户。
- 两类用户内部均按原到站时刻升序，同一时刻按固定用户编号升序。
- 不为尚未到站的预约用户扣留当前可用电池。
- 每次服务使用编号最小的服务就绪槽位。
- 同一时刻的固定处理顺序为：充电完成 → 登记到站 → 队列分配 → 对仍未服务者判超时。
- 截止时刻恰好有电池充满时，先允许服务，再判定超时。
- 携带请求保留原请求 ID、原到站时刻和原截止时刻，跨轮不得重新计时。
- 换电为瞬时事件，不引入工位、作业时长或服务结束变量。
- 预约候选事件使用 \(s_\varepsilon^A+f_\varepsilon^A+\omega_\varepsilon^A=a_\varepsilon^A\)。
- 随机请求保留服务变量 \(z_r\)；跨边界未决使用 \(\omega_r^R\)，随机流失由 \(1-z_r-\omega_r^R\) 派生，不增加随机流失成本。

### 2.6 连续充电与槽位状态

- Actor 对每个外层区间输出逐槽固定请求 \(\widehat P_{i,b,q}\)，它是本轮 MPC 的已知参数。
- 请求投影固定为

\[
0\le \widehat P_{i,b,q}\le\overline P_i,
\qquad
\sigma\sum_{b\in\mathcal B_i}\widehat P_{i,b,q}
\le\overline E_{i,q}.
\]

- 电池未满时，实际功率等于当前区间请求功率；达到 SOC 1 后立即停止充电。
- 同一槽位换入低 SOC 电池后，立即恢复当前区间的同一请求功率。
- 删除 \(p_i^{\mathrm{tol}}\)、满电补足分支、actor 功率余量预留和欠充视为满电。
- 相邻槽位替换事件之间满足

\[
S_{i,b}(t_2)-S_{i,b}(t_1)
=
\frac{\eta_i}{E_B}
\int_{t_1}^{t_2}P_{i,b}(t)\,\mathrm dt.
\]

- 给定请求功率后，事件段能量是“已知功率系数 × 连续持续时间”，不产生两个自由连续变量的乘积。
- 理论服务就绪阈值严格为 SOC 1；求解器数值容差不进入能量、收入或续航公式。
- \(b\) 表示槽位，不是永久物理电池编号；换电后槽位 SOC 立即写入退回电池 SOC。
- 同一槽位可在一个外层区间内多次服务，但任意两次交付之间必须重新达到 SOC 1。
- 附录采用事件位置驱动的 MILP 表达，展开连续事件时刻、顺序二元量和 indicator/Big-\(M\) 约束。

### 2.7 目标函数与实际记账

综合目标保持：

\[
\max\quad
I^{\mathrm{MPC}}
-C_{\mathrm{ch}}
-C_{\mathrm{adj}}
-C_{\mathrm{fail}}
+\beta\Phi^{\mathrm{RL}}.
\]

- 服务收益按实际或预测服务时刻所在价格区间取 \(\pi_{i,q}^{\mathrm{sw}}\)。
- 充电成本按 \(e_i(t)P_{i,b}(t)\) 的实际积分计算。
- 路径调整成本按实际发布时刻结算。
- 预约失败成本按截止时刻所在执行区间结算。
- 跨轮服务、失败和发布均只结算一次。
- 不增加等待成本、随机流失成本或隐含等待最小化次级目标。

### 2.8 Reference rollout、实际执行与终端价值

调用顺序固定为：参考路径 → actor rollout → critic 参数 → MPC 求解。

- 在途用户的参考路径是最近发布的剩余站序。
- 未到用户优先沿用上一轮内部参考路径；该路径不存在或几何/SOC 不可行时，使用日前路径并按当前已知信息修复。
- 参考路径修复只使用求解前已知的位置、SOC、ETA 和候选网络，不依赖尚未生成的 actor 功率或本轮优化变量。
- 参考执行采用与预测和真实执行相同的优先、等待、超时、充电和槽位替换规则，并允许预约软失败。
- 本轮最优 \(y,\alpha,z\) 不得进入 actor rollout 输入。
- 实际执行事件只来自真实到站和携带队列；预测偏差不触发停止执行或预测—实际终态强制相等。
- 终端状态包含逐槽 SOC 和仍未超时的等待队列。
- \(\lambda^S\) 只表示终端 SOC 的一阶边际价值。
- 服务尚未发生、退回 SOC 尚未写入终端库存的未来预约交付统一进入 \(\Delta V_i^{\mathrm{out}}(\rho)\) 近似。
- 每个未来交付只估值一次；已在域内服务并写入终端 SOC 时，对应 pending 指示为 0。
- 不另建 critic；终端未决价值对未来排队和超时风险的描述是近似，作为 Conclusion 中的模型边界说明。

### 2.9 日前接纳

- 保留按预约提交顺序处理。
- 保留候选站库存余量排序、靠近出口的并列规则和下游可达性检查。
- 不能生成完整入口—出口路径时拒绝预约。
- 保留严格接纳：预计到站时必须已经存在满电池，不利用 \(\Delta t\) 扩大接纳量。
- 日前库存递推与在线模型使用相同的连续充电、达到 SOC 1 才服务、即时槽位替换和退回后立即充电规则。
- 日前成功路径作为初始发布路径和未到用户的固定调整基准。

## 3. 逐章节修改规范

### 3.1 Abstract

- 替换“在此填写摘要内容”。
- 使用 4--5 句方法型摘要，顺序为：研究场景 → 多次换电和库存耦合 → MPC–RL 框架 → 连续等待与充电 → critic 终端价值。
- 不出现“显著提高”“优于基准”等未经实验支持的表述。

### 3.2 Introduction

按以下顺序组织：

1. 长途高速用户可能多次换电，前次站点选择影响下游可达站点和剩余站序。
2. 车辆 SOC 影响续航和首站，换出满电池并退回低 SOC 电池使用户路径与多站库存产生时空耦合。
3. 预约与随机用户具有不同的信息条件和服务关系；在途预约的未执行路径随实时位置、SOC 和 ETA 更新。
4. MPC 负责有限域内的离散剩余路径和连续服务协调，RL actor 提供逐槽请求功率，critic 提供终端价值。
5. 保留原 critic SOC 梯度和域外交付价值思路，但不在 Introduction 展开复杂公式。
6. 贡献内容不渲染，只保留下列源码注释：

```latex
% CONTRIBUTIONS_PLACEHOLDER_BEGIN
% 待作者补充本文贡献
% CONTRIBUTIONS_PLACEHOLDER_END
```

7. 全文结构段按第 2 节 Problem Statement、第 3 节 MPC、第 4 节 RL、第 5 节 Numerical experiments、第 6 节 Conclusion 和附录更新。

框架图占位放在 MPC–RL 分工首次完整说明之后。

### 3.3 Problem Statement

本节只陈述研究系统和运营逻辑，不展开求解器辅助变量：

- 单向高速 O-D 网络、沿线站点、槽位和 SOC；
- 多次换电和路径—库存耦合；
- 预约用户、随机用户及其信息；
- 未到、在途、站内等待三种运行状态；
- 外层区间、连续到站、等待截止和预测域；
- 预约优先、类内到站顺序和不为未来请求预留当前电池；
- 运营者协调剩余路径、充电、服务和库存的任务；
- 约 12--15 项正文核心符号表。

删除以下旧表述：进入后路径固定、每时段先充后换、退回电池下一时段才充电、随机用户到达时段必完成、每轮只发布新进入用户路径。

### 3.4 MPC Formulation 开头

删除 `Rolling-Horizon Information and Execution` 标题，使用 2--4 个自然段说明：

1. 读取观测 SOC、携带队列、实时车辆状态、已发布剩余路径、随机预测和电价；
2. 确定参考路径，完成 actor rollout 和 critic 参数计算；
3. 构造用户剩余网络，由 MPC 优化路径和连续事件安排；
4. 发布发生变化的在途剩余路径，只执行首个外层区间，并携带队列和物理状态进入下一轮。

滚动时域图占位放在这些说明之后、候选网络小节之前。

### 3.5 Offline Candidate networks by SOC Range

高度保留当前正文的 expanded-network 解释、节点/弧、SOC 分档、\(\underline D\) 和四步生成流程：

1. 生成原始弧；
2. 枚举完整可行路径；
3. 剪枝过近换电弧；
4. 合并保留路径形成候选网络。

补充在线虚拟起点、已执行节点删除、实时 SOC 首段筛选和基于真实换电历史的距离解释。统一使用 \(\underline D\)，删除 \(D^{\mathrm{pref}}\)。

### 3.6 Day-Ahead Baseline Path Generation

高度保留当前正文的预约处理顺序、三步选站规则、拒绝条件和 \(\bar y_A\) 发布机制。只重写库存物理递推，使其采用最终连续充电、满电停充、退回后立即充电和严格到站可用接纳规则。

### 3.7 Optimization Model

#### Decision variables

- 正文列出 \(y_{i,j}^{p,k}\)、\(x_{A,i}^{p,k}\)、\(d_{p,k}\)、通用请求的服务/超时/未决结果、连续服务时刻和 \(S_{i,b}(t)\)。
- \(\widehat P_{i,b,q}\) 是 RL 已知参数，\(P_{i,b}(t)\) 由确定性充电规则派生，不写成 MPC 可自由选择的功率。
- \(\alpha\)、事件位置、顺序二元量和 Big-\(M\) 变量只在附录定义。

#### Objective function

按服务收益、连续充电成本、固定一次路径调整成本、预约超时成本和 RL 终端价值依次定义，最后给出综合目标。正文不重复实际 reward 的详细事件求和。

#### Constraints

正文按连续编号给出代表性关系：

1. 剩余路径流平衡与实时 SOC 可达；
2. 按用户状态区分参考路径的调整识别；
3. 请求服务窗和服务/超时/未决守恒；
4. 连续 SOC 积分和换电后的 SOC 跳变；
5. 预约优先、类内 FCFS 和服务就绪；
6. 终端未决、初始状态和变量范围。

每组实现级展开明确引用相应附录，不在正文重复。

### 3.8 RL Formulation

#### Markov decision process

- **State**：逐槽 SOC、未到/在途用户信息、实时位置与 SOC、最近发布剩余路径、预约和随机等待队列、原到站时刻、剩余耐心、实际随机历史、预测请求、电价和时间特征。
- **Action**：每个外层区间的逐槽请求功率，使用最终可行投影。
- **Reference rollout**：按固定调用顺序生成 \(H\) 个区间的已知功率参数。
- **Transition**：`Exec_q` 是区间内连续事件执行器，只使用真实到站、携带队列、已发布路径、当前请求功率和观测 SOC。
- **Reward**：保持 \(r_\ell=I_\ell^{\mathrm{act}}-C_{\mathrm{ch},\ell}-C_{\mathrm{adj},\ell}-C_{\mathrm{fail},\ell}\)，各项按真实发生时刻结算。

#### RL algorithm

保留 PPO、critic loss、clipped objective 和 GAE，压缩通用算法介绍，只更新状态、动作和 reward 接口。外层 RL 步长保持固定 \(\sigma\)，不改为事件触发 RL。

### 3.9 Numerical experiments

只保留可见标题和下列不渲染的源码注释：

```latex
% NUMERICAL_EXPERIMENTS_PLACEHOLDER_BEGIN
% 待作者提供实验设置、数据与结果
% NUMERICAL_EXPERIMENTS_PLACEHOLDER_END
```

不写 `Model Validation`、逻辑验证表、实验设计、对照结果或效果声明。

### 3.10 Conclusion

- 总结剩余路径滚动优化、连续事件执行、等待携带和 MPC–RL 信息接口。
- 说明 ETA 为外生信息、换电瞬时、终端未决价值采用近似。
- 不写未经数值实验支持的性能结论。

## 4. 正文与附录分工

### Appendix A: Candidate Events and Effective Arrival Chains

- 固定候选预约事件 \(\varepsilon=(p,k,j,i)\)，集合不能由未知 \(y=1\) 定义。
- 预约路径激活量 \(\delta_\varepsilon=y_{j,i}^{p,k}\)。
- 随机请求存在量固定为 1，使用 \(z_r\) 控制服务。
- 展开退回 SOC、首站和后续站事件、上游成功链、携带请求初始化。
- 取消 `dec/fix` 全用户路径划分，真实已执行状态作为参数继承。

### Appendix B: Waiting, Priority, and Timeout Constraints

- 展开连续到站、服务、截止和跨边界未决变量。
- 展开 \(s+f+\omega=a\) 及随机流失派生关系。
- 展开预约优先、类内到站顺序、同刻编号和未来预约不占用当前电池。
- 展开同时刻事件顺序和截止时刻允许服务规则。
- 给出 indicator/Big-\(M\) 形式和合法界限。

### Appendix C: Continuous Charging and Slot Matching

- 展开事件位置、相邻事件持续时间和已知请求功率下的 SOC 递推。
- 展开满电停充、换入恢复、严格 SOC 1 服务就绪。
- 展开槽位匹配、最小可用槽位编号和服务后的 SOC 写入。
- 删除每槽每区间一次服务约束，允许真实重新充满后的重复服务。
- 展开站级区间能量上限和事件位置 MILP 表达。

### Appendix D: Publication, Boundary State, and Accounting

- 展开未到初始路径与在途最近发布路径的参考对齐。
- 展开路径差异指示和实际发布后参考更新。
- 展开跨边界等待、pending 交付、终端存活和防重复估值。
- 展开真实服务、充电、超时和发布成本的发生时刻记账。

## 5. 图片占位与绘图 Prompt

### 5.1 通用规则

- 删除两处旧 PNG 的 `\includegraphics` 引用，但不删除图片文件。
- 不新增 TikZ。
- PDF 显示中性 `Figure placeholder` 占位框和正式图注。
- 完整 prompt 逐行以 `%` 开头，位于对应图位的 `FIGURE_PROMPT_BEGIN/END` 注释块，不进入 PDF。
- 后续插入专业图片时只替换占位框，保留 caption、label 和正文引用。

### 5.2 `fig:MPC-RL`

位置：Introduction 中首次完整说明 MPC–RL 协同框架之后。

```latex
\begin{figure}[!htbp]
  \centering
  % FIGURE_PROMPT_BEGIN: fig:MPC-RL
  % 绘制一张适用于运筹优化与智能交通学术论文的 MPC–RL 协同框架图。白色背景，扁平化矢量风格，横向 16:9，配色仅使用深蓝、青绿色、橙色和中性灰，无渐变、阴影、3D 或装饰图标。
  % 图中从左到右展示：当前观测状态（逐槽电池 SOC、预约与随机等待队列及剩余耐心、未到和在途预约信息、实时位置与 ETA、电价和随机需求预测）；基于已知信息预先确定的参考路径与参考执行轨迹；RL actor 输出预测域内逐槽请求功率，critic 输出终端 SOC 边际价值和域外未决交付价值；MPC 在给定 RL 信息后优化预约用户剩余路径和站内连续事件安排；确定性执行器在当前外层区间内按真实到站、充电完成、换电和超时事件连续执行；输出实际 reward、更新后的电池状态、等待队列和已发布剩余路径，并反馈至下一轮。
  % 明确区分外层固定间隔决策与区间内部连续事件，强调参考路径先确定、随后 actor rollout、最后 MPC 求解，不出现 RL–MPC 内循环。所有文字使用简洁中文，字体统一，箭头方向明确，模块不超过六个，适合缩放后在 A4 论文中阅读。输出优先为可编辑 SVG/PDF 或高分辨率透明背景图片。
  % FIGURE_PROMPT_END: fig:MPC-RL
  \fbox{\parbox[c][0.52\textwidth][c]{0.92\textwidth}{%
    \centering\small Figure placeholder}}
  \caption{MPC--RL 协同决策、连续事件执行与状态反馈框架}
  \label{fig:MPC-RL}
\end{figure}
```

### 5.3 `fig:rolling_horizon`

位置：MPC Formulation 开场说明之后、候选网络小节之前。

```latex
\begin{figure}[!htbp]
  \centering
  % FIGURE_PROMPT_BEGIN: fig:rolling_horizon
  % 绘制一张适用于学术论文的横向滚动时域示意图。白色背景、简洁矢量风格，宽高比约 2:1，使用深蓝表示预测域、橙色表示当前执行区间、绿色表示跨边界携带状态、灰色表示已执行历史。
  % 顶部为时间轴，清晰标出 $t_{\ell-1}$、$t_\ell$、$t_{\ell+1}$ 和 $t_{\ell+H}$，预测域采用半开区间 $[t_\ell,t_{\ell+H})$，突出只执行首个外层区间。下方设置三条泳道：第一条为未到预约用户，显示其内部路径可滚动优化但仍相对初始发布路径评价；第二条为在途预约用户，显示从实时虚拟起点重优化剩余站序，路径变化后发布并更新比较基准；第三条为站内等待队列，显示原始到站时刻和截止时刻跨滚动边界保持不变。
  % 当前执行区间内部使用少量时间标记展示连续到站、充电完成、立即换电和超时事件；预测终点处显示尚未截止的请求作为未决状态传入下一轮。不得出现“已发布路径固定不再优化”“整时段统一换电”或“随机用户必须当期完成”等旧逻辑。文字精炼、层级清楚、无大段说明，适合 A4 论文单栏宽度阅读。输出优先为可编辑 SVG/PDF 或高分辨率图片。
  % FIGURE_PROMPT_END: fig:rolling_horizon
  \fbox{\parbox[c][0.46\textwidth][c]{0.92\textwidth}{%
    \centering\small Figure placeholder}}
  \caption{滚动时域中的剩余路径优化、连续事件执行与等待请求携带}
  \label{fig:rolling_horizon}
\end{figure}
```

## 6. 标签迁移与原文处理矩阵

### 6.1 保留的核心标签

- 图表：`fig:MPC-RL`、`fig:rolling_horizon`、`tab:notation`。
- 目标和成本：`eq:objective`、`eq:income`、`eq:chargingcost`、`eq:adjustment_cost`、`eq:reservation_failure_cost`。
- 路径：`eq:flow`、`eq:station_visit_indicator`、`eq:path_adjustment_indicator`、`eq:return_soc`。
- 终端价值：`eq:marginal_value_soc`、`eq:outside_swap_value`、`eq:rl_terminal_value`。
- RL：state、action、rollout、transition、reward、realized-cost、PPO 和 GAE 的现有有效标签。

### 6.2 移入附录并重写的标签

- `eq:swap_event_sets`、`eq:event_activation_and_soc`；
- `eq:reservation_service_failure`、`eq:reservation_priority`、`eq:reservation_service_order`；
- `eq:battery_event_assignment`、`eq:continuous_swap_transition`；
- `eq:initial_battery_soc`、变量域和边界 pending 相关标签。

这些标签只在新公式与原概念保持一致时复用；语义发生根本变化时使用新标签。

### 6.3 删除或替换的旧标签

- 删除 `eq:power_saturation`、`eq:defer_completion`、`eq:complete_charging`。
- 旧区间批量 `eq:continuous_charging_transition` 替换为 `eq:event_charging_transition`。
- `eq:reservation_alive_chain_dec` 和 `eq:reservation_alive_chain_fix` 合并为 `eq:reservation_alive_chain`。
- 删除旧的 `eq:total_reservation_swap`、`eq:full_battery_count`、`eq:service_ready_relation` 和 `eq:available_battery_count`。
- 删除 `eq:first_stage_physical_crosscheck` 及旧区间计数型实际服务公式。
- 最终全量核对 `\label`、`\eqref`、`\ref` 和 `\autoref`，不得保留悬空引用或重复标签。

### 6.4 原文处理矩阵

| 处理方式 | 内容 |
| --- | --- |
| 高度保留 | 候选网络四步流程、日前路径三步规则、综合目标、critic SOC 梯度和域外价值、PPO/GAE |
| 局部改写 | Introduction 的场景与框架段、Problem Statement 的网络和用户段、路径流与调整识别、RL state/action/reward |
| 实质重写 | 滚动用户集合、连续等待、事件服务、充电和槽位状态、边界未决、真实执行算子 |
| 删除 | 可见占位、编辑批注、旧 PNG 引用、整时段先充后换、退回电池下一时段充电、随机请求当期完成、进入后路径固定、`arr/fix` 路径划分、补足容差和预测—实际终态强制相等 |

## 7. 后续实施顺序

1. 在 `main.tex` 中建立六个正文 section、四个附录 section 和源码占位，不先删除可复用段落。
2. 修改 Abstract、Introduction 和 Problem Statement，完成科研论文口吻和核心符号表。
3. 迁移并局部修改候选网络和日前路径内容。
4. 重写 Optimization Model 的核心变量、目标和六组代表性约束。
5. 将候选事件、有效到达链、等待、优先级、槽位匹配和边界展开迁入四个附录。
6. 更新 RL state、action、reference rollout、transition 和 reward，保留并压缩 PPO/GAE。
7. 写 Conclusion，加入两张图的占位、正式图注和完整 prompt。
8. 统一符号、章节编号、标签、交叉引用和全文口吻。
9. 执行静态检查、临时目录 XeLaTeX 构建和逐页视觉检查。

## 8. 验收标准

### 8.1 结构检查

- 正文恰有六个一级 section。
- `Problem Statement` 与 `MPC Formulation` 分开。
- `MPC Formulation` 恰有三个 subsection。
- `Rolling-Horizon Information and Execution` 标题不存在。
- `Numerical experiments` 不包含 `Model Validation`、逻辑验证表或可见占位文字。
- Conclusion 后存在四个 appendix section。

### 8.2 静态残留检查

在 `main.tex` 中清除：

- `Problem Statement and MPC Formulation`；
- `\mathcal K.*arr`、`\mathcal K.*fix` 路径划分；
- `D^{\mathrm{pref}}`；
- 时段长度使用的 `\Delta`；
- `p_i^{\mathrm{tol}}`；
- “先充电、后换电”“下一时段开始充电”“到达时段内完成”；
- 旧 PNG 的 `\includegraphics` 路径；
- 可见的“在此填写”“本节待补充”“原稿”“上一版”“本次修订”；
- `【`、`】` 和编辑性圆括号内容。

### 8.3 图位检查

- 恰有两个 `FIGURE_PROMPT_BEGIN` 和两个对应的 `FIGURE_PROMPT_END`。
- 两个 prompt 内容完整，且 PDF 文本提取不包含 prompt 中的长句。
- 两个占位框、caption、label 和正文引用正确。
- 旧 PNG 文件保留在目录中，但不被 `main.tex` 引用。

### 8.4 数学一致性场景

以下场景只用于内部核查，不写入 Numerical experiments：

1. 电池在请求到站前已经满电：按当前队列顺序立即服务。
2. 电池在截止时刻恰好充满：允许服务后再判超时。
3. 电池在截止时刻之后充满：请求超时，不得借区间净库存提前服务。
4. 随机用户已经到站而预约尚未到站：不为未来预约扣留电池。
5. 两类用户同时等待：预约优先，类内按到站时间和编号。
6. 同一槽位在一个区间内两次服务：中间必须真实重新充满。
7. 等待跨滚动边界：请求 ID、到站时刻和截止时刻不重置。
8. 等待窗跨预测终点：保留未决，不提前判失败。
9. 未到用户多轮内部改路：始终相对日前初始路径评价。
10. 虚拟起点移动但剩余站序不变：不产生路径调整。
11. 上游预约超时：下游事件取消，只结算一次失败。
12. 实际随机到达偏离预测：按真实事件执行，不停止系统。
13. 上期到站请求在本期服务：收入只在实际服务区间结算一次。
14. 同一未来交付：不得同时进入终端 SOC 和域外价值。

### 8.5 构建与视觉检查

从仓库根目录使用临时输出目录构建：

```powershell
$paperVerify = Join-Path (Resolve-Path '.\tmp') ('paper-verify-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $paperVerify | Out-Null
Set-Location .\paper
latexmk -xelatex -interaction=nonstopmode -file-line-error -halt-on-error -outdir=$paperVerify main.tex
```

构建结果必须满足：

- 无编译错误、undefined control sequence、missing file；
- 无 undefined reference、duplicate label、missing glyph；
- 无 overfull box；underfull 逐项检查并消除明显排版问题；
- 符号表、长公式、占位框、附录和分页在渲染 PNG 中清晰可读；
- 不覆盖 `paper/main.pdf`。

### 8.6 本执行规范自身的完整性

- 本文件不保留需要后续执行者选择的技术分支。
- 所有关键时间、队列、充电、发布、定价、终端和日前接纳规则均为确定口径。
- 两张图的完整 prompt、正式图注、label 和占位模板均可直接复制。
- 后续修改 `main.tex` 时无需重新查阅聊天记录。
