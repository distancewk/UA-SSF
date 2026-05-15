# AGENTS.md

## 项目概览

本项目是基于 SSF（Strategic Selection and Forgetting）的网络入侵检测持续学习实验项目，当前重点方向是 UA-SSF（Uncertainty-Adaptive Strategic Selection and Forgetting）。

原始 SSF 论文为：

- Continual Learning with Strategic Selection and Forgetting for Network Intrusion Detection
- INFOCOM 2025

当前仓库在原始 SSF 基础上加入了面向投稿探索的改进，包括：

- 不确定性估计与不确定性加权训练
- 自适应漂移响应
- 选择性知识蒸馏
- CICIDS-2017 数据集支持
- 运行日志与失败原因分析

项目当前更像是一个“可继续打磨为 CCF-C 投稿的研究原型”，还不是完全整理好的论文复现实验仓库。

## 主要文件

| 路径 | 说明 |
| --- | --- |
| `ssf.py` | 主训练脚本，包含离线训练、在线持续学习、漂移检测、缓冲池更新和最终评估流程 |
| `utils.py` | 模型结构、数据处理、评估函数、样本选择、漂移检测、不确定性估计等核心工具函数 |
| `README.md` | 原始 SSF 项目说明和基础运行命令 |
| `requirements.txt` | Python 依赖列表 |
| `preprocess_cicids2017.py` | CICIDS-2017 数据预处理脚本 |
| `analysis_results.md` | UA-SSF 早期实验失败原因和改进建议分析 |
| `innovation_proposals.md` | 面向 CCF-C 投稿的创新方案草案 |
| `implementation_plan.md` | UA-SSF 实现计划 |
| `plan_d_analysis.md` | 方案 D（UA-SSF）合理性与可行性分析 |
| `ssf运行日志/` | 多次实验运行日志，包含 SSF 与 UA-SSF 的指标和窗口级行为 |

## 数据目录

| 路径 | 说明 |
| --- | --- |
| `NSL_pre_data/` | NSL-KDD 预处理数据 |
| `UNSW_pre_data/` | UNSW-NB15 预处理数据 |
| `CICIDS-2017/` | CICIDS-2017 预处理数据 |

这些目录通常体积较大，除非用户明确要求，不要删除、重命名或重新生成。

## 运行环境

项目使用 PyTorch 实现。原始 README 中的参考环境为：

- Ubuntu 20.04
- NVIDIA RTX 3090
- CUDA 11.7
- PyTorch 1.13.1
- Anaconda3

当前本地项目位于 macOS 工作区，路径为：

```bash
/Users/distancewk/Downloads/UA-SSF
```

安装依赖：

```bash
pip install -r requirements.txt
```

如需在无 GPU 环境运行，可使用：

```bash
python ssf.py --cuda cpu
```

## 常用运行命令

原始 SSF 模式：

```bash
python ssf.py --dataset nsl --mode ssf --epochs 200 --epoch_1 20 --sample_interval 5000 --num_labeled_sample 50 --opt_old_lr 100 --opt_new_lr 8 --new_sample_weight 3
```

```bash
python ssf.py --dataset unsw --mode ssf --epochs 200 --epoch_1 180 --sample_interval 20000 --num_labeled_sample 200 --opt_old_lr 24 --opt_new_lr 50 --new_sample_weight 60
```

UA-SSF 模式：

```bash
python ssf.py --dataset nsl --mode ua-ssf --epochs 200 --epoch_1 20 --sample_interval 5000 --num_labeled_sample 50 --opt_old_lr 100 --opt_new_lr 8 --new_sample_weight 3
```

```bash
python ssf.py --dataset unsw --mode ua-ssf --epochs 200 --epoch_1 180 --sample_interval 20000 --num_labeled_sample 200 --opt_old_lr 24 --opt_new_lr 50 --new_sample_weight 60
```

CICIDS-2017 示例：

```bash
python ssf.py --dataset cicids --mode ua-ssf --epoch_1 1 --percent 0.8 --cuda cpu
```

注意：CICIDS-2017 在 `ssf.py` 中有自动 epoch 调整逻辑，默认 `epochs=4` 时会自动调整到 30。

## 当前实验现状

根据现有运行日志，当前比较重要的结果如下：

| 数据集 | SSF 平均 F1 | UA-SSF 平均 F1 | 当前判断 |
| --- | ---: | ---: | --- |
| NSL-KDD | 0.9155 | 0.9100 | UA-SSF 略低于 SSF |
| UNSW-NB15 | 0.9097 | 0.9120 | UA-SSF 略高于 SSF，但提升较小 |
| CICIDS-2017 | 暂缺同设置 SSF 对照 | 0.9826 | UA-SSF 单独结果较好，但需要补公平基线 |

这些结果来自：

- `ssf运行日志/ssf_nids_log.md`
- `ssf运行日志/ua-ssf.md`
- `ssf运行日志/ua-ssf_cicids_20260511_092851.md`

## 重要研究判断

当前项目具备继续冲击 CCF-C 类会议的潜力，但还不建议直接投稿。主要原因：

1. 原始 SSF 已是 INFOCOM 2025 工作，UA-SSF 属于增量改进，需要非常扎实的实验证据。
2. NSL-KDD 上 UA-SSF 目前略低于 SSF，UNSW-NB15 上提升幅度较小。
3. CICIDS-2017 结果较强，但缺少同设置下的 SSF 与经典持续学习 baseline。
4. 需要补充消融实验、显著性检验、更多 baseline 和更清晰的论文叙事。

建议后续论文主线不要只写“改进 SSF”，而应突出：

- uncertainty-aware continual NIDS
- adaptive drift response
- selective distillation under concept drift
- 避免 buffer collapse 和 pseudo-label poisoning 的稳健机制

## 后续优先事项

建议按以下顺序推进：

1. 补齐 CICIDS-2017 上的 SSF 同设置实验。
2. 增加 baseline：Fine-tuning、LwF、ER、EWC、原始 SSF。
3. 做消融实验：去掉不确定性加权、去掉选择性蒸馏、去掉自适应漂移响应。
4. 对 5 个 seed 的结果报告均值、标准差和显著性检验。
5. 整理运行日志，把失败实验与最终有效方案分开归档。
6. 将代码中的研究性注释收敛为论文可复现实验配置。
7. 统一命名：明确 `SSF`、`UA-SSF`、`U-SSF`、`A-SSF` 等实验变体。

## 协作注意事项

- 当前工作区有未提交修改：`ssf.py` 和 `utils.py` 已被修改，处理时不要随意回滚。
- `ssf运行日志/` 是重要实验记录目录，不要删除。
- 运行完整实验可能耗时较长，尤其是 `epochs=200`、`epoch_1=180` 的配置。
- 修改核心算法前，应先明确对应消融实验，否则容易让实验结果无法解释。
- 新增结果时，建议记录完整命令、seed、dataset、mode、关键超参数和最终指标。
- 如果要清理仓库，优先移动或归档日志，不要直接删除。

## 代码风格建议

- 保持 `ssf.py` 作为主入口，避免把实验逻辑分散到多个不可追踪脚本。
- 将可复用逻辑放入 `utils.py`。
- 新增参数应通过 `argparse` 暴露，并在日志开头打印完整命令。
- 对研究性机制增加简短中文注释，但避免大段叙事性注释堆在训练循环中。
- 对新数据集支持，应同时补充数据预处理、输入维度、默认训练配置和 README 说明。

## 投稿前检查清单

- [ ] 三个数据集均有 SSF 与 UA-SSF 的公平对照。
- [ ] 至少包含 3-5 个经典 baseline。
- [ ] 每组结果至少 5 个 seed。
- [ ] 报告均值、标准差和显著性检验。
- [ ] 消融实验能够支撑每个核心模块的必要性。
- [ ] 论文图表能解释漂移强度、buffer 状态、不确定性权重和性能变化之间的关系。
- [ ] README、运行命令、日志命名和代码参数保持一致。
- [ ] 仓库中失败日志、临时分析和最终复现实验有清晰区分。
