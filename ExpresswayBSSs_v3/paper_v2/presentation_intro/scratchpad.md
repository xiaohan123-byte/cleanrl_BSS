# 引言汇报制作记录

- 范围：仅引言，合计 6 页，不另设封面。
- 风格：红色学术；16:9；白底、深红、灰色辅助；以原生文字和形状表达图示。
- 听众：学术组会/论文汇报；讲稿约 3 分钟，每页一个主旨。
- 主线：背景 → 预约与联合优化 → 滚动应对不确定性 → 终端价值 → 贡献 → 正文章节。
- 输出：自包含 HTML、原生可编辑 PPTX、逐页讲稿、预览图。

## 标题序列

1. 研究背景：高速换电与充电调度
2. 研究问题：预约路径与充电联合优化
3. 动态决策：滚动更新路径与充电计划
4. 末端效应：终端资源与未完成服务
5. 研究贡献：模型、方法与管理分析
6. 文章结构：从运营场景到数值验证

## 内容依据与校准

- 主依据：../main.tex；../sections/01_introduction.tex、02_problem_description.tex、03_model.tex、04_terminal_value_learning.tex、05_numerical_experiments.tex、06_conclusion.tex。
- 章节编号以 main.tex 的实际顺序为准：第2节问题描述，第3节模型，第4节终端价值，第5节数值实验，第6节结论。引言末尾的“第2节文献综述…第7节结论”尚未与正文同步，本次不据此制图。
- SOC 首次以“电池荷电状态”解释。换电路径表示沿途换电站序列。
- 模型以净收益为目标，计入充电、路径调整、预约失败等成本；汇报仅作概念介绍，不把目标简化成仅最小化电费。
- 路径和充电更新采用相对频率表述，避免正文默认5/15分钟与当前实验15分钟执行步长的版本差异。
- RL 终端价值同时考虑电池 SOC 分布和未完成服务；基于完整运营轨迹的后续累计净收益拟合，分段线性函数嵌入混合整数优化。
- 当前论文 RL 对照、联合优化消融与敏感性分析尚待补齐。管理分析写为研究目标，不宣称已证实收益增幅或配置结论。
- H+3 为示意性的相对时间标签，不对应具体分钟数。

## 数据来源

IEA, Global EV Outlook 2026, “Trends in electric cars”：截至2025年底，中国约4400万辆 electric cars。按报告口径，包含纯电动与插电式混合动力汽车；不是可换电车辆规模。

https://www.iea.org/reports/global-ev-outlook-2026/trends-in-electric-cars

## 工具

采用用户提供的 baoyu-design 的 make-a-deck、speaker-notes、editable PPTX export 工作流及 deck-stage 组件。
仓库：https://github.com/JimLiu/baoyu-design
已检查 ppt-master 的入口与工作流说明；本稿选择 baoyu-design 作为单一制作流程。

## 检查清单

- [x] 6页正文和6段讲稿
- [x] HTML所有页面视觉检查：无文字越界、无运行错误；翻页和讲稿显示通过；独立文件6页正常加载
- [x] PPTX原生对象和讲稿检查：6页分别28/40/33/38/24/28个原生形状，图片对象均为0；6段备注完整
- [x] PPTX打开及渲染检查：本机 Microsoft PowerPoint 正常打开，逐页导出1920×1080预览并目视核验，生成6页PDF

## 导出记录

baoyu-design 的 npm 依赖通过公共 npm 镜像安装在临时目录。渲染器复用本机 Microsoft Edge，未另行安装 Chromium。导出无警告。预览过程中修正了 deck-stage 默认白色文字的继承问题，成稿使用显式文字色。PowerPoint 只打开并关闭了本次文件。
