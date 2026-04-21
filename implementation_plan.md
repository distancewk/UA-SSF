# UA-SSF 实现计划：不确定性自适应的策略性选择与遗忘

将 SSF 改进为 UA-SSF（Uncertainty-Adaptive SSF），包含两个核心模块：
- **Part B**：自适应漂移响应（先实现，风险低）
- **Part A**：MC Dropout 不确定性引导的掩码优化（后实现，是主要创新点）

## User Review Required

> [!IMPORTANT]
> 以下设计决策需要您确认：
> 1. **新增命令行参数**：`--dropout_rate`、`--mc_forwards`、`--uncertainty_alpha`、`--base_lwf_lambda` — 是否同意这些参数名称和默认值？
> 2. **是否保留原始 SSF 的运行模式**：建议通过 `--mode` 参数切换 `ssf` / `ua-ssf`，以便做消融对比实验，您是否同意？
> 3. **训练循环合并策略**：统一后知识蒸馏始终存在（仅权重不同），漂移时 `lwf_lambda≈0`，无漂移时 `lwf_lambda=0.5`。这会略微影响原有 drift 分支的行为（原来完全无蒸馏）。

## Proposed Changes

### Phase 1: Part B — 自适应漂移响应

实现优先级最高，因为它风险最低、效果最确定，且能立即消除代码重复。

---

#### [MODIFY] [utils.py](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/utils.py)

**改动 1：新增 `detect_drift_with_severity` 函数**（在 [L694](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/utils.py#L694) 之后）

```python
def detect_drift_with_severity(new_data, control_data, window_size, drift_threshold):
    """
    漂移严重程度量化检测模块：
    将原二元漂移检测（drift / no-drift）扩展为连续的漂移程度 [0, 1]。
    利用 KS 检验的 p-value 经 -log10 映射量化漂移强度：
      - severity = 0.0 表示无漂移
      - severity → 1.0 表示极端漂移
    """
    for i in range(0, len(new_data), window_size):
        window_data = new_data[i:i + window_size]
        if len(window_data) < window_size:
            break
        ks_statistic, p_value = ks_2samp(control_data.cpu().numpy(), 
                                          window_data.cpu().numpy())
        if p_value >= drift_threshold:
            severity = 0.0
            print(f"No drift in window {i // window_size + 1} "
                  f"(p-value: {p_value:.6f}, severity: {severity:.4f})")
        else:
            severity = min(1.0, -np.log10(p_value + 1e-10) / 10.0)
            print(f"!!! Drift in window {i // window_size + 1} "
                  f"(p-value: {p_value:.6f}, severity: {severity:.4f})")
            return severity, p_value, True
    return 0.0, 1.0, False


def adaptive_params(severity, base_lwf_lambda=0.5, base_epochs=20):
    """
    根据漂移程度自适应调整训练超参数：
      - 漂移越强 → 知识蒸馏权重越低（忘掉旧知识的约束）
      - 漂移越强 → 训练 epoch 越多（充分学习新分布）
    """
    lwf_lambda = base_lwf_lambda * (1.0 - severity)
    epochs = max(1, int(base_epochs * (0.5 + severity)))
    return lwf_lambda, epochs
```

---

#### [MODIFY] [ssf.py](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/ssf.py)

**改动 1：新增命令行参数**（在 [L35](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/ssf.py#L35) 之后）

```python
parser.add_argument("--mode", type=str, default='ua-ssf', 
                    choices=['ssf', 'ua-ssf'],
                    help="运行模式：ssf=原始方法, ua-ssf=不确定性自适应改进方法")
parser.add_argument("--base_lwf_lambda", type=float, default=0.5,
                    help="知识蒸馏基础权重（自适应模式下的最大值）")
```

**改动 2：替换漂移检测调用**（[L234](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/ssf.py#L234) 和 [L248](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/ssf.py#L248)）

原始代码（NSL 分支）:
```python
drift = detect_drift(pdf11_probe, pdf1_probe, sample_interval, drift_threshold)
```
替换为:
```python
if mode == 'ua-ssf':
    severity, p_val, drift = detect_drift_with_severity(
        pdf11_probe, pdf1_probe, sample_interval, drift_threshold)
    adaptive_lwf, adaptive_epoch_1 = adaptive_params(
        severity, base_lwf_lambda, epoch_1)
else:
    drift = detect_drift(pdf11_probe, pdf1_probe, sample_interval, drift_threshold)
    severity = 1.0 if drift else 0.0
    adaptive_lwf = 0.0 if drift else lwf_lambda
    adaptive_epoch_1 = epoch_1
```

UNSW 分支做同样的替换（[L248](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/ssf.py#L248)）。

**改动 3：合并训练循环**（[L289-365](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/ssf.py#L289-L365)）

将两段几乎完全重复的 `if drift: ... else: ...` 训练循环合并为**一段统一循环**：

```python
        for epoch in range(adaptive_epoch_1):
            if epoch % 50 == 0:
                print('epoch = ', epoch)
            for j, data in enumerate(train_loader, 0):
                inputs, labels, new_sample_mask = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                new_sample_mask = new_sample_mask.to(device)
                normal_new_mask = new_sample_mask[labels == 0]
                
                optimizer.zero_grad()
                
                if dataset == 'nsl':
                    features, recon_vec = model(inputs)
                else:
                    features, recon_vec, classifications = model(inputs)

                con_loss = criterion(recon_vec, labels)
                weighted_con_loss = con_loss * ((1 - normal_new_mask) + normal_new_mask * new_sample_weight)

                if dataset == 'nsl':
                    weighted_loss = weighted_con_loss.mean()
                else:
                    classification_loss = classification_criterion(classifications.squeeze(), labels.float())
                    weighted_classification_loss = classification_loss * ((1 - new_sample_mask) + new_sample_mask * new_sample_weight)
                    weighted_loss = weighted_con_loss.mean() + weighted_classification_loss.mean()

                # 统一的知识蒸馏：权重由 adaptive_lwf 控制
                # - severity=0 (无漂移): adaptive_lwf=base_lwf_lambda (强蒸馏)
                # - severity=1 (强漂移): adaptive_lwf≈0 (几乎无蒸馏)
                if adaptive_lwf > 1e-6:
                    if dataset == 'nsl':
                        with torch.no_grad():
                            teacher_features, teacher_recon_vec = teacher_model(inputs)
                        distillation_loss = F.mse_loss(recon_vec, teacher_recon_vec)
                    else:
                        with torch.no_grad():
                            teacher_features, teacher_recon_vec, teacher_logits = teacher_model(inputs)
                        distillation_loss = F.mse_loss(classifications, teacher_logits)
                    
                    total_loss = weighted_loss + adaptive_lwf * distillation_loss
                else:
                    total_loss = weighted_loss

                total_loss.backward()
                optimizer.step()
```

> [!TIP]
> 通过 `if adaptive_lwf > 1e-6` 的判断，确保在 `severity=1.0`（极端漂移）时完全跳过蒸馏计算，与原始 SSF 在 drift=True 时的行为**完全一致**，保证向后兼容。

---

### Phase 2: Part A — MC Dropout 不确定性引导

在 Phase 1 完成并验证后实施。

---

#### [MODIFY] [utils.py](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/utils.py)

**改动 1：为 AE 添加 Dropout 支持**（[L450-479](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/utils.py#L450-L479)）

新增 `AE_dropout` 类（保留原始 `AE` 不修改，以便对比实验）：

```python
class AE_dropout(nn.Module):
    """带 Dropout 的自编码器，支持 MC Dropout 不确定性估计"""
    def __init__(self, input_dim, dropout_rate=0.1):
        super(AE_dropout, self).__init__()
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))
        second_fourth_layer_size = nearest_power_of_2 // 2
        third_layer_size = nearest_power_of_2 // 4

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode
```

**改动 2：为 AE_classifier 添加 Dropout 版本**（[L28-63](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/utils.py#L28-L63)）

同理新增 `AE_classifier_dropout` 类，在 encoder 和 decoder 各加一层 Dropout。

**改动 3：新增 `estimate_uncertainty` 函数**（[L108](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/utils.py#L108) 之后）

```python
def estimate_uncertainty(model, x, n_forward=10, batch_size=512):
    """
    MC Dropout 不确定性估计：
    在保持 Dropout 开启的状态下进行 n_forward 次前向传播，
    用输出的标准差衡量模型预测的不确定性。
    高不确定性 → 样本处于决策边界 → 信息量大。
    """
    was_training = model.training
    model.train()  # 保持 Dropout 开启
    
    all_uncertainties = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        predictions = []
        for _ in range(n_forward):
            with torch.no_grad():
                output = model(batch)
                recon = output[1]  # 取重建向量（兼容 AE 和 AE_classifier）
                predictions.append(recon)
        predictions = torch.stack(predictions)  # (n_forward, batch, dim)
        uncertainty = predictions.std(dim=0).mean(dim=1)  # 每个样本一个标量
        all_uncertainties.append(uncertainty)
    
    if not was_training:
        model.eval()
    return torch.cat(all_uncertainties)
```

**改动 4：修改 `optimize_old_mask` 签名和逻辑**（[L115-151](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/utils.py#L115-L151)）

```diff
- def optimize_old_mask(control_res, treatment_res, device, initialization='0-1', num_bins=10, lr=100, steps=100):
+ def optimize_old_mask(control_res, treatment_res, device, initialization='0-1', num_bins=10, lr=100, steps=100,
+                       uncertainty=None, uncertainty_alpha=0.1):

      # ... 原有逻辑不变 ...

-     Loss = Accuracy_Loss_c
+     if uncertainty is not None:
+         # 不确定性正则：鼓励高不确定性旧样本的 M_c 更低（优先被遗忘）
+         uncertainty_norm = uncertainty / (uncertainty.max() + 1e-8)
+         uncertainty_reg = uncertainty_alpha * torch.sum(M_c * uncertainty_norm)
+         Loss = Accuracy_Loss_c + uncertainty_reg
+     else:
+         Loss = Accuracy_Loss_c
```

**改动 5：修改 `optimize_new_mask` 签名和逻辑**（[L159-202](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/utils.py#L159-L202)）

```diff
- def optimize_new_mask(control_res, treatment_res, M_c, device, initialization='0-1', num_bins=10, lr=0.1, steps=100):
+ def optimize_new_mask(control_res, treatment_res, M_c, device, initialization='0-1', num_bins=10, lr=0.1, steps=100,
+                       uncertainty=None, uncertainty_alpha=0.1):

      # ... 原有逻辑不变 ...

-     Loss = Drift_Loss_t
+     if uncertainty is not None:
+         # 不确定性正则：鼓励高不确定性新样本的 M_t 更高（优先被选中）
+         uncertainty_norm = uncertainty / (uncertainty.max() + 1e-8)
+         uncertainty_reg = -uncertainty_alpha * torch.sum(M_t * uncertainty_norm)  # 注意负号
+         Loss = Drift_Loss_t + uncertainty_reg
+     else:
+         Loss = Drift_Loss_t
```

> [!IMPORTANT]
> **注意 M_c 和 M_t 的正则方向相反**：
> - M_c（旧数据）：高不确定性 → 正则推低 M_c → 优先遗忘 → 用**正号** `+alpha * M_c * u`
> - M_t（新数据）：高不确定性 → 正则推高 M_t → 优先入选 → 用**负号** `-alpha * M_t * u`

---

#### [MODIFY] [ssf.py](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/ssf.py)

**改动 1：新增 UA-SSF 命令行参数**（在 Phase 1 的参数之后）

```python
parser.add_argument("--dropout_rate", type=float, default=0.1,
                    help="MC Dropout 的丢弃率")
parser.add_argument("--mc_forwards", type=int, default=10,
                    help="MC Dropout 前向传播次数")
parser.add_argument("--uncertainty_alpha", type=float, default=0.1,
                    help="不确定性正则项权重")
```

**改动 2：模型初始化根据 mode 选择**（[L131-136](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/ssf.py#L131-L136)）

```python
    if dataset == 'nsl':
        if mode == 'ua-ssf':
            model = AE_dropout(input_dim, dropout_rate).to(device)
            teacher_model = AE_dropout(input_dim, dropout_rate).to(device)
        else:
            model = AE(input_dim).to(device)
            teacher_model = AE(input_dim).to(device)
    else:
        if mode == 'ua-ssf':
            model = AE_classifier_dropout(input_dim, dropout_rate).to(device)
            teacher_model = AE_classifier_dropout(input_dim, dropout_rate).to(device)
        else:
            model = AE_classifier(input_dim).to(device)
            teacher_model = AE_classifier(input_dim).to(device)
```

**改动 3：在掩码优化前估计不确定性**（[L254-256](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/ssf.py#L254-L256)）

```python
        # UA-SSF: 在掩码优化前估计不确定性
        if mode == 'ua-ssf':
            uncertainty_old = estimate_uncertainty(model, x_train_this_epoch, mc_forwards)
            uncertainty_new = estimate_uncertainty(model, x_test_this_epoch, mc_forwards)
        else:
            uncertainty_old = None
            uncertainty_new = None

        M_c = optimize_old_mask(control_res, treatment_res, device, 
                                initialization=old_init, lr=opt_old_lr,
                                uncertainty=uncertainty_old, 
                                uncertainty_alpha=uncertainty_alpha)
        M_t = optimize_new_mask(control_res, treatment_res, M_c, device, 
                                initialization=new_init, lr=opt_new_lr,
                                uncertainty=uncertainty_new, 
                                uncertainty_alpha=uncertainty_alpha)
```

---

### Phase 3: 验证与实验

#### 3.1 正确性验证

- **SSF 向后兼容测试**：用 `--mode ssf` 运行，确认结果与原始代码完全一致
- **梯度流检查**：通过 `M_c.grad` 和 `M_t.grad` 确认不确定性正则项正确参与梯度计算
- **severity 范围验证**：打印每次漂移检测的 `severity` 值，确认在 `[0, 1]` 范围内

#### 3.2 实验矩阵

| # | 实验名 | 命令 |
|---|--------|------|
| 1 | SSF 基线 | `python ssf.py --mode ssf --dataset nsl` |
| 2 | SSF 基线 | `python ssf.py --mode ssf --dataset unsw` |
| 3 | UA-SSF 完整 | `python ssf.py --mode ua-ssf --dataset nsl` |
| 4 | UA-SSF 完整 | `python ssf.py --mode ua-ssf --dataset unsw` |
| 5 | 消融：仅自适应 | `python ssf.py --mode ua-ssf --uncertainty_alpha 0 --dataset nsl` |
| 6 | 消融：仅不确定性 | `python ssf.py --mode ua-ssf --base_lwf_lambda 0.5 --dataset nsl`（固定 lwf） |
| 7 | α 敏感性 | `--uncertainty_alpha` ∈ {0.01, 0.05, 0.1, 0.5, 1.0} |
| 8 | n_forward 敏感性 | `--mc_forwards` ∈ {5, 10, 20, 30} |

## Open Questions

> [!WARNING]
> 1. **原始 SSF 的 `epoch_1` 默认值为 1**，而 `adaptive_params` 的 `base_epochs` 设为 20 会导致 UA-SSF 训练量远大于原始 SSF。建议将 `adaptive_params` 的 `base_epochs` 直接使用 `epoch_1` 参数值，您是否同意？
> 2. **样本选择分支** `if drift:` 处理（[L261-278](file:///Users/distancewk/Downloads/SSF-Strategic-Selection-and-Forgetting-main/ssf.py#L261-L278)）：在 UA-SSF 中是否也要统一？还是保留原始的二元处理（仅统一训练循环）？建议保留原样，因为样本选择策略本身是另一个维度的问题。

## Verification Plan

### Automated Tests

1. `python ssf.py --mode ssf --dataset unsw --cuda 0 --seed_round 1` — 验证原始 SSF 结果不变
2. `python ssf.py --mode ua-ssf --dataset unsw --cuda 0 --seed_round 1` — 验证 UA-SSF 能正常运行
3. 对比两次运行的 Accuracy / F1 / Precision / Recall

### Manual Verification

- 检查 severity 值的打印输出是否合理
- 检查 `M_c.grad` 是否包含不确定性正则项的贡献
- 对比 UA-SSF vs SSF 的性能指标差异
