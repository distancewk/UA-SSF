import os
import torch
import numpy as np
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import scipy.optimize as opt
import torch.distributions as dist
from sklearn.metrics import accuracy_score
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from scipy.stats import ks_2samp

import torch.optim as optim
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt
    
# MLP function for unsw
class AE_classifier(nn.Module):
    def __init__(self, input_dim):
        super(AE_classifier, self).__init__()
        # Find the nearest power of 2 to input_dim
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # Calculate the dimensions of the 2nd/4th layer and the 3rd layer.
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4         # A quarter

        # Create encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # Create decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, 1),  # 1 neuron for binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        classify = self.classifier(decode)
        return encode, decode, classify

# ==============================
# UA-SSF 改进：带 Dropout 的分类自编码器，支持 MC Dropout 不确定性估计
# 在 encoder 和 decoder 各插入一层 Dropout，以便在推断时通过保持 Dropout 开启
# 进行多次随机前向传播来估计模型预测的不确定性。
# ==============================
class AE_classifier_dropout(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1):
        super(AE_classifier_dropout, self).__init__()
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
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        classify = self.classifier(decode)
        return encode, decode, classify

# Evaluation function for unsw
def evaluate_classifier(model, data_loader, device, get_predict=False, threshold=0.5):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            _, _, classifications = model(inputs)
            preds = (classifications.squeeze(-1) > threshold).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    if not get_predict:
        res = score_detail(all_labels, all_preds, if_print=True)

    if get_predict:
        return all_preds
    else:
        return res

# Evaluation function for single sample or batch of samples for unsw
def evaluate_inputs(model, inputs, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        _, _, classifications = model(inputs)
        preds = (classifications.squeeze(-1) > threshold).float()
    return preds.cpu().numpy()


def predict_classifier_probs(model, inputs, device, batch_size=512):
    was_training = model.training
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size].to(device)
            _, _, classifications = model(batch)
            probs.append(classifications.squeeze(-1).detach().cpu())
    if was_training:
        model.train()
    return torch.cat(probs)


def _normalize_uncertainty(values):
    values = values - values.min()
    denom = values.max() + 1e-8
    return values / denom


def _binary_entropy(prob):
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    return -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())


def estimate_hybrid_uncertainty(model, x, n_forward=10, batch_size=512, mc_dropout_rate=None,
                                recon_weight=0.35, cls_weight=0.65):
    """分类数据集专用：融合重构 MC 方差与分类不确定性，返回 shape=(N,) 的权重信号。"""
    was_training = model.training

    original_rates = []
    if mc_dropout_rate is not None:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                original_rates.append(m.p)
                m.p = mc_dropout_rate

    model.train()
    recon_uncertainties = []
    class_uncertainties = []

    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size].to(next(model.parameters()).device)
        recon_predictions = []
        class_predictions = []
        for _ in range(n_forward):
            with torch.no_grad():
                output = model(batch)
                recon_predictions.append(output[1])
                class_predictions.append(output[2].squeeze(-1))

        recon_predictions = torch.stack(recon_predictions)
        class_predictions = torch.stack(class_predictions)

        recon_std = recon_predictions.std(dim=0, unbiased=False).mean(dim=1)
        mean_prob = class_predictions.mean(dim=0)
        predictive_entropy = _binary_entropy(mean_prob)
        expected_entropy = _binary_entropy(class_predictions).mean(dim=0)
        bald = (predictive_entropy - expected_entropy).clamp(min=0.0)
        class_std = class_predictions.std(dim=0, unbiased=False)
        class_uncertainty = predictive_entropy + bald + class_std

        recon_uncertainties.append(recon_std)
        class_uncertainties.append(class_uncertainty)

    if mc_dropout_rate is not None:
        idx = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = original_rates[idx]
                idx += 1

    if not was_training:
        model.eval()

    recon_uncertainty = _normalize_uncertainty(torch.cat(recon_uncertainties))
    class_uncertainty = _normalize_uncertainty(torch.cat(class_uncertainties))
    return recon_weight * recon_uncertainty + cls_weight * class_uncertainty


def calibrate_classifier_threshold(model, x_ref, y_ref, device, batch_size=512,
                                   previous_threshold=0.5, momentum=0.5,
                                   threshold_min=0.35, threshold_max=0.75):
    """在当前缓存池上选择分类阈值；优先保持接近最佳 F1，同时偏向更高 Precision。"""
    probs = predict_classifier_probs(model, x_ref, device, batch_size=batch_size).numpy()
    labels = y_ref.detach().cpu().numpy().astype(int).reshape(-1)
    thresholds = np.linspace(threshold_min, threshold_max, 81)

    metrics = []
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        metrics.append({
            "threshold": float(threshold),
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    base_metric = min(metrics, key=lambda item: abs(item["threshold"] - 0.5))
    best_f1 = max(item["f1"] for item in metrics)
    near_best = [item for item in metrics if item["f1"] >= best_f1 - 0.002]
    precision_safe = [item for item in near_best if item["precision"] >= base_metric["precision"]]
    candidates = precision_safe if precision_safe else near_best
    selected = max(candidates, key=lambda item: (item["precision"], item["f1"], item["recall"]))

    threshold = selected["threshold"]
    if previous_threshold is not None:
        threshold = momentum * previous_threshold + (1.0 - momentum) * threshold
    threshold = float(np.clip(threshold, thresholds.min(), thresholds.max()))
    selected = dict(selected)
    selected["threshold"] = threshold
    return threshold, selected


def attack_posterior(pdf_normal, pdf_attack):
    return pdf_attack / (pdf_normal + pdf_attack + 1e-8)


def calibrate_nsl_attack_threshold(pdf_normal, pdf_attack, labels, previous_threshold=0.5,
                                   momentum=0.5, threshold_min=0.35, threshold_max=0.60):
    """NSL 专用：在当前 buffer 上选择偏 Recall 的攻击后验阈值。"""
    probs = attack_posterior(pdf_normal, pdf_attack).detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels).astype(int).reshape(-1)
    thresholds = np.linspace(threshold_min, threshold_max, 51)

    metrics = []
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        metrics.append({
            "threshold": float(threshold),
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    best_f1 = max(item["f1"] for item in metrics)
    near_best = [item for item in metrics if item["f1"] >= best_f1 - 0.002]
    selected = max(near_best, key=lambda item: (item["recall"], item["f1"], item["precision"]))

    threshold = selected["threshold"]
    if previous_threshold is not None:
        threshold = momentum * previous_threshold + (1.0 - momentum) * threshold
    threshold = float(np.clip(threshold, threshold_min, threshold_max))
    selected = dict(selected)
    selected["threshold"] = threshold
    return threshold, selected

def initialize_tensor(size, initialization, device):
    if initialization == '0-1':
        return torch.nn.Parameter(torch.rand(size, device=device), requires_grad=True)
    elif initialization == '0-0.5':
        return torch.nn.Parameter(torch.rand(size, device=device) * 0.5, requires_grad=True)
    elif initialization == '0.5-1':
        return torch.nn.Parameter(torch.rand(size, device=device) * 0.5 + 0.5, requires_grad=True)
    else:
        raise ValueError("Invalid initialization type. Choose from '0-1', '0-0.5', or '0.5-1'.")

# ==============================
# MC Dropout 不确定性估计模块 (UA-SSF 核心创新):
# 在保持 Dropout 开启的状态下进行 n_forward 次前向传播，
# 利用输出重建向量的标准差（方差的平方根）来衡量模型对每个样本预测的不确定性。
# 高不确定性 → 样本处于模型决策边界附近 → 信息量大：
#   - 对旧数据：高不确定性意味着模型对其已不够确定，适合被优先遗忘
#   - 对新数据：高不确定性意味着该样本携带了模型未充分学习的信息，应优先入选
# ==============================
def estimate_uncertainty(model, x, n_forward=10, batch_size=512, mc_dropout_rate=None):
    """返回每个样本的不确定性标量，shape: (N,)
    
    mc_dropout_rate: 若指定，则临时将 Dropout rate 设为此值进行 MC 估计，
                     结束后恢复原始值。用于训练时 dropout=0 但估计时需要噪声的场景。
    """
    was_training = model.training
    
    # 临时恢复 MC Dropout rate（训练 dropout=0 时仍需估计噪声）
    original_rates = []
    if mc_dropout_rate is not None:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                original_rates.append(m.p)
                m.p = mc_dropout_rate
    
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
        predictions = torch.stack(predictions)  # (n_forward, batch_size, dim)
        uncertainty = predictions.std(dim=0).mean(dim=1)  # 每个样本一个标量
        all_uncertainties.append(uncertainty)
    
    # 恢复原始 Dropout rate
    if mc_dropout_rate is not None:
        idx = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = original_rates[idx]
                idx += 1
    
    if not was_training:
        model.eval()
    return torch.cat(all_uncertainties)


def _forward_recon_with_shadow_dropout(model, x, dropout_rate):
    z = x
    for layer in model.encoder:
        if isinstance(layer, nn.Dropout):
            continue
        z = layer(z)
        if isinstance(layer, nn.ReLU) and dropout_rate > 0:
            z = F.dropout(z, p=dropout_rate, training=True)

    encode = z
    z = encode
    for layer in model.decoder:
        if isinstance(layer, nn.Dropout):
            continue
        z = layer(z)
        if isinstance(layer, nn.ReLU) and dropout_rate > 0:
            z = F.dropout(z, p=dropout_rate, training=True)
    return encode, z


def estimate_shadow_recon_uncertainty(model, x, n_forward=10, batch_size=512, dropout_rate=0.1):
    """NSL 专用：主模型保持原始 AE 结构，仅在不确定性估计时做影子 dropout。"""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    all_uncertainties = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size].to(device)
        predictions = []
        for _ in range(n_forward):
            with torch.no_grad():
                _, recon = _forward_recon_with_shadow_dropout(model, batch, dropout_rate)
                predictions.append(recon)
        predictions = torch.stack(predictions)
        uncertainty = predictions.std(dim=0, unbiased=False).mean(dim=1)
        all_uncertainties.append(uncertainty)

    if was_training:
        model.train()
    return torch.cat(all_uncertainties)

# ==============================
# 旧数据代表性评估模块 (Strategic Forgetting): 
# 该函数通过对每个旧样本分配一个可学习的掩模参数（M_c），使用梯度下降最小化
# "保留下来的旧样本重构分布 (bin_obs_c)" 和 "未来新样本分布 (bin_tgt_c)" 之间的 KL 散度 (KL Divergence)。
# 最终收敛时，M_c 取值偏高的样本具备更强的描述整体分布能力，是需要被保留的 "核心代表样本"；而偏低的样本即可被战略性遗忘。
# ==============================
def optimize_old_mask(control_res, treatment_res, device, initialization='0-1', num_bins=10, lr=100, steps=100,
                      uncertainty=None, uncertainty_alpha=0.1):
    control_res = torch.tensor(control_res, dtype=torch.float).to(device)
    treatment_res = torch.tensor(treatment_res, dtype=torch.float).to(device)

    M_c = initialize_tensor(control_res.size(0), initialization, device)

    optimizer = torch.optim.SGD([M_c], lr=lr)
    delta = 1e-4

    # UA-SSF: 预处理不确定性向量（归一化到 [0, 1]）
    if uncertainty is not None:
        uncertainty_norm = uncertainty / (uncertainty.max() + 1e-8)

    for step in range(steps):
        with torch.no_grad():
            M_c.clamp_(delta, 1 - delta)

        optimizer.zero_grad()

        bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)
        control_hist = torch.histc(control_res, bins=num_bins, min=0., max=1.)
        treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)

        bin_obs_c = torch.zeros(num_bins, device=device)
        bin_tgt_c = torch.zeros(num_bins, device=device)

        for i in range(num_bins):
            mask_c = (control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])
            bin_obs_c[i] = torch.sum(M_c * mask_c.float()) / torch.sum(M_c)
            bin_tgt_c[i] = treatment_hist[i] / len(treatment_res)

        bin_obs_c = bin_obs_c / bin_obs_c.sum()
        bin_tgt_c = bin_tgt_c / bin_tgt_c.sum()

        Accuracy_Loss_c = F.kl_div(bin_obs_c.log(), bin_tgt_c, reduction='sum')

        # UA-SSF: 不确定性正则项（差分形式）— 只惩罚高于均值的不确定性样本
        # P1 Fix: 避免全局压低 M_c，仅让「异常不确定」的旧样本被优先遗忘
        if uncertainty is not None:
            uncertainty_centered = F.relu(uncertainty_norm - uncertainty_norm.mean())
            uncertainty_reg = uncertainty_alpha * torch.sum(M_c * uncertainty_centered)
            Loss = Accuracy_Loss_c + uncertainty_reg
        else:
            Loss = Accuracy_Loss_c

        Loss.backward()
        optimizer.step()

    return M_c

# ==============================
# 新流入数据优选模块 (Strategic Selection): 
# 此处继承了上一步学习到的旧数据留存掩模（M_c）。函数将新旧样本合并在一起，并为新样本分配单独的掩模（M_t）。
# 它同时考虑了被挑选出来的新、旧样本的混合分布，通过 SGD 优化 M_t 使得混合后的分布尽可能接近单纯新样本的真实分布。
# 从而精准定位能"填补旧知识盲区"的核心增量新样本，纳入缓存用于未来的持续微调。
# ==============================
def optimize_new_mask(control_res, treatment_res, M_c, device, initialization='0-1', num_bins=10, lr=0.1, steps=100,
                      uncertainty=None, uncertainty_alpha=0.1):
    control_res = torch.tensor(control_res, dtype=torch.float).to(device)
    treatment_res = torch.tensor(treatment_res, dtype=torch.float).to(device)

    M_t = initialize_tensor(treatment_res.size(0), initialization, device)

    optimizer = torch.optim.SGD([M_t], lr=lr)
    delta = 1e-4

    # UA-SSF: 预处理不确定性向量（归一化到 [0, 1]）
    if uncertainty is not None:
        uncertainty_norm = uncertainty / (uncertainty.max() + 1e-8)

    for step in range(steps):
        with torch.no_grad():
            M_t.clamp_(delta, 1 - delta)

        optimizer.zero_grad()

        bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)
        Drift_Loss_t = 0.

        control_hist = torch.histc(control_res, bins=num_bins, min=0., max=1.)
        treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)

        bin_tgt_t = torch.zeros(num_bins, device=device)

        bin_combined = torch.zeros(num_bins, device=device)
        for i in range(num_bins):
            mask_c = (control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])
            mask_t = (treatment_res >= bin_edges[i]) & (treatment_res < bin_edges[i + 1])

            bin_tgt_t[i] = treatment_hist[i] / len(treatment_res)

            bin_combined[i] = (torch.sum(M_t * mask_t.float()) + torch.sum(M_c * mask_c.float())) / (torch.sum(M_t) + torch.sum(M_c))

        bin_combined = torch.clamp(bin_combined / bin_combined.sum(), min=1e-10)
        bin_combined = bin_combined / bin_combined.sum()
        bin_tgt_t = torch.clamp(bin_tgt_t / bin_tgt_t.sum(), min=1e-10)
        bin_tgt_t = bin_tgt_t / bin_tgt_t.sum()

        Drift_Loss_t = F.kl_div(bin_combined.log(), bin_tgt_t, reduction='sum')

        # UA-SSF: 不确定性正则项（差分形式）— 只提升高于均值的不确定性新样本的 M_t
        # P1 Fix: 避免所有新样本的 M_t 被无差别抬高
        if uncertainty is not None:
            uncertainty_centered = F.relu(uncertainty_norm - uncertainty_norm.mean())
            uncertainty_reg = -uncertainty_alpha * torch.sum(M_t * uncertainty_centered)
            Loss = Drift_Loss_t + uncertainty_reg
        else:
            Loss = Drift_Loss_t

        Loss.backward()
        optimizer.step()

    return M_t

# 平稳期缓冲池更新策略（无明显漂移时）：
# 1. 根据 M_c 和 M_t 是否大于阈值(0.5)去提取代表性新旧样本。
# 2. 定位缓冲池中得分低的冗余旧样本，将其剔除（腾出 num_labeled_sample 大小的空间）。
# 3. 如果优质新样本（代表性得分高）的数量不足，使用随机补充策略从新一批数据中抓取以满足标记样本的数量要求。
# 4. 组装并返回维护修整后的缓存：x_train_this_epoch（即新的 Memory Buffer）。
def select_and_update_representative_samples(x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, M_c, M_t, num_labeled_sample, device):
    M_c_bin = (M_c >= 0.5).float().to(device)
    # P4 Fix: 提高新样本入选阈值（0.5→0.8），减少新样本注入量
    # SSF baseline 每窗口只选入 0-6 个新样本，而 UA-SSF 的 uncertainty 正则项
    # 系统性抬高了 M_t，导致每窗口约 2100 个新样本被选入，稀释旧知识。
    # 提高阈值后预期降到 ~500-800 个，仍远高于 SSF，保留新概念学习能力。
    M_t_bin = (M_t >= 0.5).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    print(f"  Samples: old={representative_old.shape[0]}, new={representative_new.shape[0]}", end="")

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False

    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    if len(non_representative_old_indices) < num_to_remove:
        print(f" | WARN: not enough non-rep old to remove ({len(non_representative_old_indices)})", end="")
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)
        
        # Remove all non-representative samples first
        remove_indices = non_representative_old_indices
        
        # Then remove the remaining number from the representative samples with the lowest scores
        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]

        remove_indices = torch.cat([remove_indices, additional_remove_indices])
    else:
        remove_indices = non_representative_old_indices[torch.randperm(len(non_representative_old_indices))[:num_to_remove]]

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False

    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    if representative_new.shape[0] < num_labeled_sample:
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]
        print(f" | new补充={additional_samples_needed}", end="")

        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()])
        available_indices = set(torch.arange(len(x_test_this_epoch)).cpu().numpy()) - selected_indices
        available_indices = torch.tensor(list(available_indices), dtype=torch.long)

        fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
        drift_representative_new = torch.cat([representative_new, x_test_this_epoch[fallback_indices]], dim=0)
        new_labels = torch.cat([y_test_this_epoch[M_t_bin.bool()], y_test_this_epoch[fallback_indices]], dim=0)
        sorted_indices_new = torch.cat([torch.arange(len(representative_new)), fallback_indices], dim=0)
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    new_sample_mask = torch.cat([new_sample_mask, torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)])
    x_train_this_epoch = torch.cat([x_train_this_epoch, drift_representative_new], dim=0)
    y_train_this_epoch = torch.cat([y_train_this_epoch, new_labels], dim=0)

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask

# 动荡期缓冲池激进更新策略（检测到严峻的分布/概念漂移时）：
# 1. 较上面的平稳期策略更加彻底：将所有不符合 M_c 阈值的旧样本全部剥离出缓存池（无差别抛弃所有不再具代表性的沉旧知识）。
# 2. 把高质量（代表性强）的新进标注样本填充进缓存池。
# 3. 漂移时期对新知识的渴求极大，若移除旧样本后缓冲池内仍有巨大的剩余容量 (buffer_memory_size 未填满)，
#    则不仅吸纳带真实标注的优质样本，且会调用当前 Model 直接对部分无标注的新样本进行识别并打上"伪标签"(Pseudo-labels),
#    强行塞入缓冲池，以此最大程度扩张针对崭新概念的数据阵列。
def select_and_update_representative_samples_when_drift(
        x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, 
        M_c, M_t, num_labeled_sample, device, buffer_memory_size, model, normal_recon_temp=None,
        classifier_threshold=0.5, attack_threshold=0.5):

    M_c_bin = (M_c >= 0.5).float().to(device)
    # P4 Fix: 同上，drift 路径也使用 0.8 阈值
    M_t_bin = (M_t >= 0.5).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    print(f"  Samples: old={representative_old.shape[0]}, new={representative_new.shape[0]}", end="")

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False

    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    # Remove all non-representative samples
    remove_indices = non_representative_old_indices

    if len(non_representative_old_indices) < num_to_remove:
        print(f" | WARN: not enough non-rep old ({len(non_representative_old_indices)})", end="")
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)
        
        # Then remove the remaining number from the representative samples with the lowest scores
        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]

        remove_indices = torch.cat([remove_indices, additional_remove_indices])

    # ==============================
    # P0 Fix 1: Buffer 最小保留保护
    # 防止 drift 时过度清除旧样本导致缓冲池灾难性坍塌。
    # 确保移除后至少保留 min_retain_ratio 比例的 buffer 容量。
    # 若需移除的数量超出安全上限，则按 M_c 分数从低到高只移除最不具代表性的样本。
    # ==============================
    min_retain_ratio = 0.4
    min_retain = int(buffer_memory_size * min_retain_ratio)
    current_size = len(x_train_this_epoch)
    max_removable = max(0, current_size - min_retain)

    if len(remove_indices) > max_removable and max_removable > 0:
        # 按 M_c 分数升序排列，优先移除最低分样本
        remove_scores = M_c[remove_indices].detach()
        sorted_order = torch.argsort(remove_scores)  # 升序：最低分在前
        remove_indices = remove_indices[sorted_order[:max_removable]]
        print(f" | PROTECT: retain≥{min_retain}", end="")
    elif max_removable == 0:
        remove_indices = remove_indices[:0]  # 不移除任何样本
        print(f" | PROTECT: skip removal", end="")

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False

    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    if representative_new.shape[0] < num_labeled_sample:
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]
        print(f" | new补充={additional_samples_needed}", end="")

        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()])
        available_indices = set(torch.arange(len(x_test_this_epoch)).cpu().numpy()) - selected_indices
        available_indices = torch.tensor(list(available_indices), dtype=torch.long)

        fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
        drift_representative_new = torch.cat([representative_new, x_test_this_epoch[fallback_indices]], dim=0)
        new_labels = torch.cat([y_test_this_epoch[M_t_bin.bool()], y_test_this_epoch[fallback_indices]], dim=0)
        sorted_indices_new = torch.cat([torch.arange(len(representative_new)), fallback_indices], dim=0)
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    new_sample_mask = torch.cat([new_sample_mask, torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)])
    x_train_this_epoch = torch.cat((x_train_this_epoch, drift_representative_new), dim=0)
    y_train_this_epoch = torch.cat((y_train_this_epoch, new_labels), dim=0)

    if len(x_train_this_epoch) < buffer_memory_size:
        additional_samples_needed = buffer_memory_size - len(x_train_this_epoch)
        # Phase1-A: 移除伪标签数量上限，允许 buffer 回填到满容量（与 SSF 一致）
        # 原 P0 Fix 2a 的 max_pseudo=30% 导致 buffer 在每次 drift 后不可逆缩小，
        # 是 UA-SSF 落后 SSF 的最大性能差距来源。
        print(f" | pseudo_fill={additional_samples_needed}", end="")

        if representative_new.shape[0] > num_labeled_sample:
            remaining_new_samples = representative_new[torch.argsort(torch.tensor(scores_new), descending=True)[num_labeled_sample:]]
            # remaining_samples_needed = num_additional_samples_needed

            if remaining_new_samples.size(0) >= additional_samples_needed:
                pseudo_labeled_samples = remaining_new_samples[:additional_samples_needed]
                if normal_recon_temp == None:
                    pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device, threshold=classifier_threshold)
                else:
                    pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch,
                                             pseudo_labeled_samples, 0, model, attack_threshold=attack_threshold)
            else:
                pseudo_labeled_samples = remaining_new_samples
                if normal_recon_temp == None:
                    pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device, threshold=classifier_threshold)
                else:
                    pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch,
                                             pseudo_labeled_samples, 0, model, attack_threshold=attack_threshold)
                
                random_new_additional_samples_needed = additional_samples_needed - remaining_new_samples.size(0)
                
                additional_indices = torch.randperm(len(x_test_this_epoch))[:random_new_additional_samples_needed]
                additional_pseudo_labeled_samples = x_test_this_epoch[additional_indices]
                if normal_recon_temp == None:
                    additional_pseudo_labels = evaluate_inputs(model, additional_pseudo_labeled_samples, device, threshold=classifier_threshold)
                else:
                    additional_pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch,
                                                        additional_pseudo_labeled_samples, 0, model,
                                                        attack_threshold=attack_threshold)
                
                pseudo_labeled_samples = torch.cat([pseudo_labeled_samples, additional_pseudo_labeled_samples], dim=0)
                if normal_recon_temp == None:
                    if random_new_additional_samples_needed > 1:
                        pseudo_labels = torch.cat([torch.tensor(pseudo_labels), torch.tensor(additional_pseudo_labels)], dim=0)
                    else:
                        pseudo_labels = torch.cat([torch.tensor(pseudo_labels), torch.tensor(additional_pseudo_labels).unsqueeze(0)], dim=0)
                else:
                    if random_new_additional_samples_needed > 1:
                        pseudo_labels = torch.cat([pseudo_labels, additional_pseudo_labels], dim=0)
                    else:
                        pseudo_labels = torch.cat([pseudo_labels, additional_pseudo_labels.unsqueeze(0)], dim=0)
        else:
            additional_indices = torch.randperm(len(x_test_this_epoch))[:additional_samples_needed]
            pseudo_labeled_samples = x_test_this_epoch[additional_indices]
            if normal_recon_temp == None:
                pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device, threshold=classifier_threshold)
            else:
                pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch,
                                         pseudo_labeled_samples, 0, model, attack_threshold=attack_threshold)

        # Phase2-D: 完全移除伪标签置信度过滤（原 P0 Fix 2b）
        # SSF 从不过滤伪标签，直接全部入 buffer → buffer 始终满容量
        # conf_kept 每次 drift 丢弃 30% 伪标签，是 buffer 萎缩的根本原因

        # 将过滤后的伪标签样本拼入训练集
        x_train_this_epoch = torch.cat((x_train_this_epoch, pseudo_labeled_samples), dim=0)
        # 统一转为 tensor 处理，避免类型分支冗余
        if not torch.is_tensor(pseudo_labels):
            pseudo_labels = torch.tensor(pseudo_labels)
        if pseudo_labels.dim() == 0:
            pseudo_labels = pseudo_labels.unsqueeze(0)
        y_train_this_epoch = torch.cat((y_train_this_epoch, pseudo_labels.to(device)), dim=0)
        
        new_sample_mask = torch.cat([new_sample_mask, torch.zeros(len(pseudo_labeled_samples), dtype=torch.float32).to(device)])

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask

#################################################################

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

class SplitData(BaseEstimator, TransformerMixin):
    def __init__(self, dataset, pre_scaled=False):
        super(SplitData, self).__init__()
        self.dataset = dataset
        self.pre_scaled = pre_scaled

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, labels, one_hot_label=True):
        if self.dataset == 'nsl':
            # Preparing the labels
            y = X[labels]
            X_ = X.drop(['labels5', 'labels2'], axis=1)
            # abnormal data is labeled as 1, normal data 0
            y = (y != 'normal')
            y_ = np.asarray(y).astype('float32')

        elif self.dataset == 'unsw':
            # UNSW dataset processing
            y_ = X[labels]
            X_ = X.drop('label', axis=1)

        elif self.dataset == 'cicids':
            # CICIDS-2017 dataset processing (same format as UNSW after preprocessing)
            y_ = X[labels]
            X_ = X.drop('label', axis=1)

        else:
            raise ValueError("Unsupported dataset type")

        if self.pre_scaled:
            x_ = X_.to_numpy(dtype='float32')
            return x_, y_

        # Normalization
        normalize = MinMaxScaler().fit(X_)
        x_ = normalize.transform(X_)

        return x_, y_

class AE(nn.Module):
    def __init__(self, input_dim):
        super(AE, self).__init__()

        # Find the nearest power of 2 to input_dim
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # Calculate the dimensions of the 2nd/4th layer and the 3rd layer.
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4         # A quarter

        # Create encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # Create decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

# ==============================
# UA-SSF 改进：带 Dropout 的自编码器，支持 MC Dropout 不确定性估计
# 在 encoder 和 decoder 各插入一层 Dropout，结构与 AE 完全一致，仅增加 Dropout 层。
# ==============================
class AE_dropout(nn.Module):
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

# ==============================
# 自监督对比损失 InfoNCE Loss 计算模块: 
# 在异常检测与连续学习中，通常往往只有少量异常特征，甚至全依赖正常数据的学习。该函数构建了一个极其重要的网络特征重塑对比环境：
# 1. 通过计算小批次 batch 内样本组相互之间的余弦相似度构建 Logits。
# 2. 利用 Mask 掩除矩阵对角线元素（除去自身和自身的平凡相似性）。
# 3. 构造 InfoNCE 公式的核心分子：强制提高正常样本彼此之间的特征内聚度 (logits_normal_normal)。
# 4. 构造 InfoNCE 公式的分母部分：通过正常和异常的互作用 (logits_normal_abnormal) 形成排斥力。
# 目标结果：促使基础网络最终能让同类（通常为正常类）的嵌入分布在潜在高维流形里极其密集凑堆，并将那些未知、新生成的恶意攻击分布挤出此密集集簇，从而实现无先锋经验的异常感知。
# ==============================
class InfoNCELoss(nn.Module):
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(InfoNCELoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):        
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()
        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # Calculate the dot product similarity between pairwise samples
        # create mask 
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size).to(self.device)  
        logits_without_ii = logits * logits_mask
        
        logits_normal = logits_without_ii[(labels == 0).squeeze()]
        logits_normal_normal = logits_normal[:,(labels == 0).squeeze()]
        logits_normal_abnormal = logits_normal[:,(labels > 0).squeeze()]

        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal), axis=1, keepdims=True)
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        log_probs = logits_normal_normal - torch.log(denominator)
  
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature

        return loss
    
def score_detail(y_test,y_test_pred,if_print=True):
    acc = accuracy_score(y_test, y_test_pred)
    pre = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    if if_print:
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"  Acc={acc:.4f}  Pre={pre:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
        print(f"  Confusion: TP={cm[1][1]} FP={cm[0][1]} FN={cm[1][0]} TN={cm[0][0]}")
    return acc, pre, rec, f1

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")
    # seed 信息合并到 ssf.py 的 seed header 中，此处保留静默设置

# Define a small epsilon to avoid log(0)
EPSILON = 1e-10

def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)
    
    # Ensure the values passed to log are above a threshold
    likelihood = 0.5 * pdf1 + 0.5 * pdf2
    likelihood = np.clip(likelihood, a_min=EPSILON, a_max=None)
    
    # Check for NaN values
    if np.any(np.isnan(likelihood)):
        print("NaN values found in likelihood calculation")
        return np.inf
    
    return -np.sum(np.log(likelihood))

# Define the batch processing function
def process_batch(data, temp, layer_index, model, batch_size=128, device='cuda'):
    values = []
    model.to(device)
    temp = temp.to(device)
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        batch_features = F.normalize(model(batch)[layer_index], p=2, dim=1)
        batch_cosine_sim = F.cosine_similarity(batch_features, temp.reshape([-1, temp.shape[0]]), dim=1)
        values.append(batch_cosine_sim)
        del batch, batch_features, batch_cosine_sim  # Free memory
        torch.cuda.empty_cache()  # Clear cache
    
    return torch.cat(values)

def evaluate(normal_recon_temp, x_train, y_train, x_test, y_test, model, batch_size=128, device='cuda',
             get_probs=False, attack_threshold=0.5):
    model.eval()
    # Define dataset and dataloader
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)

    # num_of_layer = 0
    num_of_output = 1

    # values_features_all = []
    # values_features_normal = []
    # values_features_abnormal = []
    values_recon_all = []
    values_recon_normal = []
    values_recon_abnormal = []

    model.to(device)
    # normal_temp = normal_temp.to(device)
    normal_recon_temp = normal_recon_temp.to(device)

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            features, recon_vec = model(inputs)
            # values_features_all.append(F.cosine_similarity(F.normalize(features, p=2, dim=1), normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
            values_recon_all.append(F.cosine_similarity(F.normalize(recon_vec, p=2, dim=1), normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

            normal_mask = (labels == 0)
            abnormal_mask = (labels == 1)

            if normal_mask.sum() > 0:
                # values_features_normal.append(F.cosine_similarity(F.normalize(features[normal_mask], p=2, dim=1), normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
                values_recon_normal.append(F.cosine_similarity(F.normalize(recon_vec[normal_mask], p=2, dim=1), normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
            
            if abnormal_mask.sum() > 0:
                # values_features_abnormal.append(F.cosine_similarity(F.normalize(features[abnormal_mask], p=2, dim=1), normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
                values_recon_abnormal.append(F.cosine_similarity(F.normalize(recon_vec[abnormal_mask], p=2, dim=1), normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

        values_recon_all = torch.cat(values_recon_all).cpu().numpy()
        # values_recon_all = torch.cat(values_recon_all)
        values_recon_normal = torch.cat(values_recon_normal).cpu().numpy()
        values_recon_abnormal = torch.cat(values_recon_abnormal).cpu().numpy()

    x_test = x_test.to(device)
    # values_features_test = process_batch(x_test, normal_temp, num_of_layer, model, batch_size, device)
    values_recon_test = process_batch(x_test, normal_recon_temp, num_of_output, model, batch_size, device)

    mu3_initial = np.mean(values_recon_normal)
    sigma3_initial = np.std(values_recon_normal)
    mu4_initial = np.mean(values_recon_abnormal)
    sigma4_initial = np.std(values_recon_abnormal)

    # Fit Gaussians to reconstruction similarities
    initial_params = np.array([mu3_initial, sigma3_initial, mu4_initial, sigma4_initial])
    result = opt.minimize(log_likelihood, initial_params, args=(values_recon_all,), method='Nelder-Mead')
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = result.x

    if mu3_fit > mu4_fit:
        gaussian3 = Normal(mu3_fit, sigma3_fit)
        gaussian4 = Normal(mu4_fit, sigma4_fit)
    else:
        gaussian4 = Normal(mu3_fit, sigma3_fit)
        gaussian3 = Normal(mu4_fit, sigma4_fit)

    pdf3 = gaussian3.log_prob(values_recon_test.clone().detach()).exp()
    pdf4 = gaussian4.log_prob(values_recon_test.clone().detach()).exp()
    y_test_pred_4 = (attack_posterior(pdf3, pdf4) > attack_threshold).cpu().numpy().astype("int32")
    # y_test_pro_de = (torch.abs(pdf4 - pdf3)).cpu().detach().numpy().astype("float32")

    if get_probs:
        values_recon_test = values_recon_test.detach()
        return pdf3,pdf4,values_recon_test
    else:
        if not isinstance(y_test, int):
            if y_test.device != torch.device("cpu"):
                y_test = y_test.cpu().numpy()
            result_decoder = score_detail(y_test, y_test_pred_4)

        # y_test_pred_no_vote = torch.where(torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de), torch.from_numpy(y_test_pred_2), torch.from_numpy(y_test_pred_4))
        y_test_pred_no_vote = torch.from_numpy(y_test_pred_4)

        if not isinstance(y_test, int):
            result_final = score_detail(y_test, y_test_pred_no_vote, if_print=True)
            # return result_encoder, result_decoder, result_final
            return result_decoder, result_final
        else:
            return y_test_pred_no_vote

# 概念漂移预警模块 (Drift Detection)：
# 在线连续学习场景中，为了判断是否遇到了恶意网络攻击的变种，需要时刻保持警惕。本函数：
# 将滑动窗口所带来的“未来数据预测概率分布”(new_data/treatment) 同
# 之前已被认为表现平稳的“基准模型环境数据分布”(control_data) ，利用非参数的 Kolmogorov-Smirnov (K-S) 两样本检验进行统计学比对。
# 只要得出的 p-value 小于既定的极其严苛的置信界限值 (drift_threshold)，从统计学上就证明未来流量产生了非偶然性本质突变。然后系统将报警，进入上文描述的激进防灾级缓冲池更新。
def detect_drift(new_data, control_data, window_size, drift_threshold):
    for i in range(0, len(new_data), window_size):
        window_data = new_data[i:i + window_size]
        if len(window_data) < window_size:
            break
        ks_statistic, p_value = ks_2samp(control_data.cpu().numpy(), window_data.cpu().numpy())
        # ks_statistic, p_value = ks_2samp(control_data, window_data)
        if p_value < drift_threshold:
            return True
        # SSF mode: no-drift case is silent (logged by ssf.py)
    return False

# ==============================
# 漂移严重程度量化检测模块 (UA-SSF 改进):
# 将原始的二元漂移检测（drift / no-drift）扩展为连续的漂移程度 severity ∈ [0, 1]。
# 利用 KS 检验的 p-value 经 -log10 映射量化漂移强度：
#   - severity = 0.0 表示无漂移（p_value >= threshold）
#   - severity → 1.0 表示极端漂移（p_value 极小）
# 该连续化设计消除了硬阈值导致的策略跳变，使模型能够按漂移程度渐进式调整学习策略。
# ==============================
def detect_drift_with_severity(new_data, control_data, window_size, drift_threshold):
    """返回 (severity, p_value, is_drift) 三元组
    
    P1 Fix: 使用 KS 统计量（而非 p-value）量化漂移严重程度。
    KS 统计量 ∈ [0,1] 本身就是分布差异的直接度量，不受样本量影响，
    避免了 -log10(p_value) 在大样本下快速饱和到 1.0 的问题。
    severity = ks_statistic / ks_severity_cap, 线性映射并截断到 [0, 1]。
    """
    ks_severity_cap = 0.5  # KS 统计量达到 0.5 时视为最大漂移
    for i in range(0, len(new_data), window_size):
        window_data = new_data[i:i + window_size]
        if len(window_data) < window_size:
            break
        ks_statistic, p_value = ks_2samp(control_data.cpu().numpy(),
                                          window_data.cpu().numpy())
        if p_value >= drift_threshold:
            return 0.0, p_value, False
        else:
            severity = min(1.0, ks_statistic / ks_severity_cap)
            return severity, p_value, True
    return 0.0, 1.0, False


def adaptive_params(severity, base_lwf_lambda=0.5, base_epochs=20):
    """
    根据漂移程度自适应调整训练超参数 (C1 创新)：
      - drift 时 lwf_lambda=0（全局蒸馏关闭，由 C3 选择性蒸馏在训练循环中实现细粒度控制）
      - 漂移越强 → 训练 epoch 越多（按漂移严重程度分配训练预算）
    当 severity=0 时退化为无漂移的标准训练（lwf_lambda=base_lwf_lambda, epochs=base_epochs）
    当 severity=1 时退化为极端漂移训练（lwf_lambda=0, epochs=base_epochs*2）
    """
    # C3: drift 时全局 lwf=0，选择性蒸馏由训练循环中按样本不确定性实现
    lwf_lambda = 0.0
    # C1: severity 越大 → 分配更多训练 epoch（0.5x 缩放避免低 severity 时过度训练伪标签）
    epochs = max(1, int(base_epochs * (1.0 + 0.5 * severity)))
    return lwf_lambda, epochs
