# Skill Info: Bayesian Optimization for Electrocatalyst Development

## 名称
bayesian-optimization-electrocatalyst

## 作者
- 作者（Author）：Yu Peng
- 维护者（Maintainer）：Yu Peng
- 联系方式（可选）：your.email@example.com
- ORCID（可选）：
- GitHub：https://github.com/ANDYPENG09

> 说明：本技能按 MIT 协议开源。署名信息（author）在以下三处保持一致：
> SKILL.md 前置 author 字段、本文件作者小节、以及 LICENSE 版权行。

## 许可证
MIT License —— 详见仓库根目录 LICENSE 文件。
允许他人自由使用、复制、修改、合并、发布、分发、再许可及销售本软件，
须保留版权声明与许可声明。软件按原样提供，不附任何担保。

## 描述
面向电催化剂开发的贝叶斯优化（BO）闭环技能。以高斯过程（Matérn 5/2 + ARD）
为代理模型，支持 EI / PI / UCB 采集函数，以及约束（§11.2）、批量（§11.3）、
多目标（§11.7）扩展。基于已有实验数据推荐下一轮实验方案，并输出预测均值、
95% 置信区间与收敛分析。理论依据 Roman Garnett《Bayesian Optimization》
（Cambridge University Press）。

## 触发条件
- 贝叶斯优化 / BO
- 下一轮实验 / 推荐实验
- 催化剂优化 高斯过程 / GP 采集函数 / EI / UCB
- 过电位 / Tafel / ECSA / 法拉第效率 优化
- 多目标 催化

## 依赖
- Python >= 3.8
- scikit-optimize, pyyaml, numpy, pandas, scipy

## 目录结构
```
bayesian-optimization-electrocatalyst/
├── SKILL.md                  闭环工作流 + TRACE 规范
├── LICENSE                   MIT 许可证
├── skill-info.md             本文件（署名 / 许可证 / 元数据）
├── _meta.json                平台元数据（SkillHub / ClawHub 导入用）
├── references/
│   ├── bo_theory.md          Garnett 原著理论手册（带公式编号）
│   └── electrocatalyst_metrics.md  电化学指标与约束编码
├── scripts/
│   ├── bo_pipeline.py        主闭环（GP → EI/PI/UCB → 推荐 → 收敛）
│   ├── trace_utils.py        TRACE 调用链追踪器
│   └── data_io.py            CSV / WorkBuddy / Notion / ima 多源归一
└── assets/
	├── config.yaml           通用变量 / 目标 / 约束 / 采集参数
	├── experiment_template.csv  9 点合成种子数据（无个人数据）
	├── config_ptco_example.yaml   PtCo L1₀ 文献参数示例配置（7-D，合成）
	└── experiment_ptco_example.csv  PtCo 9 点合成种子数据（文献参数范围）
```

## 版本
1.0.0
