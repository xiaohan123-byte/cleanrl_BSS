# 引言汇报 · 红色学术 · 6页

以 paper_v2 当前论文与用户给定草稿为依据。共6页，不另设封面；讲稿建议总时长约3分钟。

## 成品

- [可编辑 PowerPoint](引言汇报_红色学术_6页.pptx)：原生文本、形状和线条；每页备注已写入讲稿。
- [独立 HTML 演示稿](引言汇报.html)：无需网络，可直接用浏览器打开；方向键翻页，F 全屏，N 显示/隐藏讲稿。
- [逐页讲稿](逐页讲稿.md)：每页一段，60–115字左右。
- [PDF 预览](引言汇报_预览.pdf)：由 Microsoft PowerPoint 导出。
- [六页总览](六页预览.png)：来自实际 PPT 渲染。

## 页序

1. 高速换电与充电调度
2. 预约路径与充电联合优化
3. 滚动更新路径与充电计划
4. 终端资源与未完成服务
5. 模型、方法与管理分析
6. 从运营场景到数值验证

## 修改说明

最方便的修改方式是直接编辑 PPTX。HTML 源文件为 intro.html，配套 deck-stage.js；运行 prepare_delivery.py 可更新独立 HTML、讲稿和导出配置。PPTX 与 HTML 是独立成品，修改其中一个不会自动修改另一个。

HTML 使用 Microsoft YaHei 字体，Windows 下可直接显示；其他系统可能采用系统回退字体。

## 内容依据

具体内容校准与工具出处记录于 scratchpad.md。文章结构按 main.tex 实际章节顺序整理；管理分析尚待实验验证。背景4400万辆来自 [IEA Global EV Outlook 2026](https://www.iea.org/reports/global-ev-outlook-2026/trends-in-electric-cars)，按报告口径包含纯电动与插电式混合动力，不是可换电车辆数。

制作流程与 HTML 演示组件来自用户提供的 [baoyu-design](https://github.com/JimLiu/baoyu-design)。验证记录保存在 preview/。
