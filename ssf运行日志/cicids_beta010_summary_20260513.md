# CICIDS-2017 beta=0.10 消融与公平对照汇总

日期：2026-05-13

## 实验设置

本轮实验固定 CICIDS-2017、5 个 seed、同一训练配置，用于形成 CICIDS 上的第一版最小闭环结果。

公共命令参数：

```bash
python ssf.py --dataset cicids --epochs 30 --epoch_1 1 --percent 0.8 --sample_interval 20000 --num_labeled_sample 200 --opt_old_lr 1 --opt_new_lr 50 --new_sample_weight 100 --seed 5011 --seed_round 5 --cuda cpu
```

数据规模：

| Split | Rows | Labels |
| --- | ---: | --- |
| Train | 350000 | normal=290893, attack=59107 |
| Test | 150000 | normal=124669, attack=25331 |

特征数：52

## 已完成实验

| 实验名 | mode | 关键开关 | 状态 |
| --- | --- | --- | --- |
| `SSF-cicids-sameSetting` | `ssf` | 原始 SSF，同设置公平对照 | 有效 |
| `UA-beta010-full` | `ua-ssf` | 完整 UA-SSF，`uncertainty_beta=0.10` | 有效 |
| `UA-beta010-noUW` | `ua-ssf` | `--disable_uncertainty_weight` | 有效 |
| `UA-beta010-noADR` | `ua-ssf` | `--disable_adaptive_drift` | 有效 |
| `UA-beta010-noSD` | `ua-ssf` | `--disable_selective_distillation` | 已跑，但不作为有效 C3 消融 |

`UA-beta010-noSD` 与 `UA-beta010-full` 的结果完全一致。代码中 CICIDS 的选择性蒸馏仅在 `severity > 0.12` 时进入 loss，而本轮 CICIDS 窗口 severity 主要位于 `0.02-0.09`，因此 C3 在当前 gate 下未实际触发。

## 主结果表

| 方法 | Before F1 | After Acc | After Pre | After Rec | After F1 | F1 std | Drift windows | Avg severity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSF same setting | 0.9843 ± 0.0024 | 0.9937 ± 0.0012 | 0.9856 ± 0.0055 | 0.9771 ± 0.0079 | 0.9813 ± 0.0036 | 0.0036 | 17.20 ± 1.48 | 0.7818 ± 0.0674 |
| UA-SSF full | 0.9863 ± 0.0009 | 0.9950 ± 0.0004 | 0.9903 ± 0.0032 | 0.9797 ± 0.0021 | 0.9850 ± 0.0011 | 0.0011 | 9.80 ± 1.79 | 0.0413 ± 0.0048 |
| UA-SSF noUW | 0.9863 ± 0.0009 | 0.9944 ± 0.0021 | 0.9856 ± 0.0121 | 0.9813 ± 0.0015 | 0.9834 ± 0.0061 | 0.0061 | 10.00 ± 1.58 | 0.0414 ± 0.0050 |
| UA-SSF noADR | 0.9863 ± 0.0009 | 0.9943 ± 0.0011 | 0.9877 ± 0.0046 | 0.9783 ± 0.0035 | 0.9829 ± 0.0033 | 0.0033 | 16.80 ± 1.64 | 0.7636 ± 0.0747 |

## 相对提升

以 `SSF-cicids-sameSetting` 为公平 baseline：

| 指标 | SSF | UA-SSF full | 差值 |
| --- | ---: | ---: | ---: |
| After Acc | 0.9937 | 0.9950 | +0.0013 |
| After Pre | 0.9856 | 0.9903 | +0.0047 |
| After Rec | 0.9771 | 0.9797 | +0.0026 |
| After F1 | 0.9813 | 0.9850 | +0.0037 |
| F1 std | 0.0036 | 0.0011 | -0.0025 |
| Drift windows | 17.20 | 9.80 | -7.40 |
| Avg severity | 0.7818 | 0.0413 | -0.7405 |

核心结论：UA-SSF 在 CICIDS-2017 上相较同设置 SSF 获得 `+0.0037` After F1，并显著降低跨 seed 波动与过度漂移响应。这个结果比单看 F1 均值更有价值，因为它同时支持“性能提升”和“漂移响应更稳定”两条叙事。

## 消融结论

| 消融 | After F1 变化 | 现象 | 可写结论 |
| --- | ---: | --- | --- |
| 去掉不确定性加权 noUW | 0.9850 -> 0.9834 | 均值下降，F1 std 从 0.0011 增至 0.0061 | 不确定性加权主要贡献稳定性，并减少异常 seed 的风险 |
| 去掉自适应漂移 noADR | 0.9850 -> 0.9829 | Drift windows 从 9.80 增至 16.80，severity 接近 SSF | 自适应漂移响应是抑制过度 drift 的关键模块 |
| 去掉选择性蒸馏 noSD | 0.9850 -> 0.9850 | 与 full 完全一致 | 当前 CICIDS severity gate 下 C3 未激活，不能作为有效消融证据 |

## 每个 seed 的 After F1

| Seed | SSF | UA full | noUW | noADR |
| ---: | ---: | ---: | ---: | ---: |
| 5011 | 0.9854 | 0.9860 | 0.9870 | 0.9846 |
| 5012 | 0.9839 | 0.9836 | 0.9871 | 0.9810 |
| 5013 | 0.9766 | 0.9863 | 0.9726 | 0.9864 |
| 5014 | 0.9787 | 0.9843 | 0.9855 | 0.9782 |
| 5015 | 0.9822 | 0.9848 | 0.9849 | 0.9845 |

## 推荐论文表述

在 CICIDS-2017 上，UA-SSF 在相同训练预算和缓冲区配置下优于原始 SSF。相比 SSF，UA-SSF 将 After CL F1 从 `0.9813 ± 0.0036` 提升至 `0.9850 ± 0.0011`，同时将平均 drift 窗口数从 `17.20` 降至 `9.80`。这表明 UA-SSF 的收益不仅来自最终分类性能，还来自更保守、更稳定的漂移响应。

消融实验显示，去掉不确定性加权会使 F1 方差明显增大，去掉自适应漂移响应会使 drift 行为退化到接近原始 SSF。选择性蒸馏在当前 CICIDS gate 下未触发，因此本轮结果暂不支持对 C3 单独贡献作强结论。

## 后续最小动作

1. 当前 CICIDS 主实验表可采用：`SSF same setting`、`UA-SSF full`、`noUW`、`noADR`。
2. `noSD` 结果保留为诊断记录，不放入正式消融主表，或在表下注明 “inactive under current severity gate”。
3. 若后续需要证明 C3，应降低 CICIDS 的 C3 gate，例如从 `severity > 0.12` 调整到 `severity > 0.05`，然后只重跑 `UA-beta010-full` 与 `UA-beta010-noSD`。
4. 投稿前仍需在 NSL-KDD、UNSW-NB15 上补齐同样格式的公平对照和消融表。
