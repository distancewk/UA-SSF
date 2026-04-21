import torch
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from utils import *

import argparse
import warnings

warnings.filterwarnings('ignore')

# 解析命令行参数，设置模型的核心超参数：
# --dataset: 所使用的数据集名称 (nsl 或 unsw)
# --epochs: 离线基础训练阶段的总轮数
# --epoch_1: 在线持续学习时的微调训练轮数
# --percent: 将训练数据截断为流式在线数据的比例 (即模拟未来的数据流)
# --sample_interval: 模拟数据流每次到来的样本批次大小（滑动窗口大小）
# --num_labeled_sample: 从缓存池和新数据中挑选出的带标签代表性样本的个数
# --opt_new_lr / --opt_old_lr: 用于优化新/旧数据代表性掩模 (Mask) 的学习率
# --new_sample_weight: 在线再训练时赋予代表性新样本的额外损失权重，防止模型对新学习目标欠拟合
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataset", type=str, default='unsw')
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--epoch_1", type=int, default=1)
parser.add_argument("--percent", type=float, default=0.8)
parser.add_argument("--sample_interval", type=int, default=20000)
parser.add_argument("--cuda", type=str, default="2")
parser.add_argument("--num_labeled_sample", type=int, default=200)
parser.add_argument("--opt_new_lr", type=float, default=50.0)
parser.add_argument("--opt_old_lr", type=float, default=1.0)
parser.add_argument("--new_sample_weight", type=float, default=100.0, help="Weight for new samples in the loss calculation")
parser.add_argument("--mode", type=str, default='ssf', choices=['ssf', 'ua-ssf'],
                    help="Running mode: ssf=original, ua-ssf=uncertainty-adaptive")
parser.add_argument("--base_lwf_lambda", type=float, default=0.5,
                    help="Base knowledge distillation weight (max value in adaptive mode)")
parser.add_argument("--dropout_rate", type=float, default=0.1,
                    help="MC Dropout rate")
parser.add_argument("--mc_forwards", type=int, default=10,
                    help="Number of MC Dropout forward passes")
parser.add_argument("--uncertainty_alpha", type=float, default=0.1,
                    help="Uncertainty regularization weight")


args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs
epoch_1 = args.epoch_1
percent = args.percent
sample_interval = args.sample_interval
cuda_num = args.cuda
num_labeled_sample = args.num_labeled_sample
opt_new_lr = args.opt_new_lr
opt_old_lr = args.opt_old_lr
new_sample_weight = args.new_sample_weight
mode = args.mode
base_lwf_lambda = args.base_lwf_lambda
dropout_rate = args.dropout_rate
mc_forwards = args.mc_forwards
uncertainty_alpha = args.uncertainty_alpha

seed = 5011
seed_round = 5
old_init = '0.5-1'
new_init = '0-0.5'
lwf_lambda = 0.5

tem = 0.02
bs = 128
drift_threshold = 0.05

if dataset == 'nsl':
    input_dim = 121
else:
    input_dim = 196


if dataset == 'nsl':
    KDDTrain_dataset_path   = "NSL_pre_data/PKDDTrain+.csv"
    KDDTest_dataset_path    = "NSL_pre_data/PKDDTest+.csv"

    KDDTrain   =  load_data(KDDTrain_dataset_path)
    KDDTest    =  load_data(KDDTest_dataset_path)

    splitter_nsl = SplitData(dataset='nsl')
else:
    UNSWTrain_dataset_path   = "UNSW_pre_data/UNSWTrain.csv"
    UNSWTest_dataset_path    = "UNSW_pre_data/UNSWTest.csv"

    UNSWTrain   =  load_data(UNSWTrain_dataset_path)
    UNSWTest    =  load_data(UNSWTest_dataset_path)

    splitter_unsw = SplitData(dataset='unsw')

device = torch.device("cuda:"+cuda_num if torch.cuda.is_available() else "cpu")

criterion = InfoNCELoss(device, tem)
if dataset != 'nsl':
    classification_criterion = nn.BCELoss(reduction='none')

for i in range(seed_round):
    # Set the seed for the random number generator for this iteration
    # 设置当前轮次的随机种子，确保实验可重复
    # setup_seed(seed+i)
    current_seed = seed+i
    setup_seed(current_seed)
    print(f"Current seed: {current_seed}")

    if dataset == 'nsl':
        # Transform the data
        x_train, y_train = splitter_nsl.transform(KDDTrain, labels='labels2')
        x_test, y_test = splitter_nsl.transform(KDDTest, labels='labels2')
    else:
        # Transform the data
        x_train, y_train = splitter_unsw.transform(UNSWTrain, labels='label')
        x_test, y_test = splitter_unsw.transform(UNSWTest, labels='label')

    # Convert to torch tensors
    x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
    x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)
    print('shape of x_train ', x_train.shape)
    print('shape of x_test is ', x_test.shape)

    torch.cuda.empty_cache()

    # 将初始大批量训练集再次切分：
    # `online_x_train` 作为模型起步的基础历史数据
    # `online_x_test` 用作后续模拟随时间流式到达的增量新数据
    online_x_train, online_x_test, online_y_train, online_y_test = train_test_split(x_train, y_train, test_size=percent)
    
    # 按照基础数据的规模，计算并分配持续学习缓存池（Memory Buffer）的容量上限
    memory = x_train.shape[0] * (1-percent)
    print('size of memory is ', memory)
    memory = math.floor(memory)

    train_ds = TensorDataset(online_x_train, online_y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=bs, shuffle=True)
    
    num_of_first_train = online_x_train.shape[0]

    # UA-SSF: model initialization based on mode
    # ua-ssf mode uses Dropout model variants for MC Dropout uncertainty estimation
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

    optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

    model.train()
    # 初始离线训练阶段（使用当前已有的历史数据进行基础模型训练）
    for epoch in range(epochs):
        if epoch % 50 == 0 or epoch == 0:
            print('seed = ', (seed+i), ', first round: epoch = ', epoch)
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)

            labels = labels.to(device)
            optimizer.zero_grad()

            # 前向传播：NSL数据集仅使用自编码器，UNSW额外包含分类头
            if dataset == 'nsl':
                features, recon_vec = model(inputs)
            else:
                features, recon_vec, classifications = model(inputs)
            
            # 使用 InfoNCE 对比学习损失拉近同类（正常）样本特征，推远异类特征
            con_loss = criterion(recon_vec,labels)
            
            if dataset == 'nsl':
                loss = con_loss.mean()
            else:
                # 对具备分类头的网络，叠加标准的二元交叉熵损失 (BCE) 取代单一重构
                classification_loss = classification_criterion(classifications.squeeze(), labels.float())
                loss = con_loss.mean() + classification_loss.mean()

            loss.backward()
            optimizer.step()

    teacher_model.load_state_dict(model.state_dict())  # Initialize teacher model

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    online_x_train, online_y_train  = online_x_train.to(device), online_y_train.to(device)

    x_train_this_epoch, x_test_left_epoch, y_train_this_epoch, y_test_left_epoch = online_x_train.clone(), online_x_test.clone().to(device), online_y_train.clone(), online_y_test.clone()
    
    permutation = torch.randperm(x_test.size(0))

    # Apply this permutation to both tensors to shuffle them in unison.
    x_test = x_test[permutation]
    y_test = y_test[permutation]
    # Concatenating x_test and x_test_left_epoch
    x_test_left_epoch = torch.cat((x_test_left_epoch, x_test), dim=0)
    print('shape of x_test_left_epoch is ', x_test_left_epoch.shape)
    # Concatenating y_test and y_test_left_epoch
    y_test_left_epoch = torch.cat((y_test_left_epoch, y_test), dim=0)

    if dataset == 'nsl':
        normal_recon_temp = torch.mean(F.normalize(model(online_x_train[(online_y_train == 0).squeeze()])[1], p=2, dim=1), dim=0)
        y_pred_before_online = evaluate(normal_recon_temp, online_x_train, online_y_train, x_test_left_epoch, 0, model)
        y_pred_before_online = y_pred_before_online.cpu().numpy()
        print('--------------------------- performance_before_continual_training -----------------------------------')
        performance_before = score_detail(y_test_left_epoch[-x_test.shape[0]:].numpy(), y_pred_before_online[-x_test.shape[0]:].astype("int32"))
    else:
        print('--------------------------- performance_before_contunual_training -----------------------------------')
        test_ds = TensorDataset(x_test, y_test)  # Replace with your test data
        test_loader = DataLoader(dataset=test_ds, batch_size=bs, shuffle=False)
        performance_before = evaluate_classifier(model, test_loader, device)

####################### start online training #######################
    # 开始在线/持续学习阶段，模拟流式数据到达的过程
    model = model.to(device)

    count = 0
    y_train_detection = y_train_this_epoch

    start_idx = 0
    y_train_detection = torch.empty(0, dtype=torch.long, device=device)
    labeled_indices = []

    while start_idx < len(x_test_left_epoch):
        print('seed = ', (seed+i), ', i = ', count)
        count += 1
        
        end_idx = min(start_idx + sample_interval, len(x_test_left_epoch))
        x_test_this_epoch = x_test_left_epoch[start_idx:end_idx]
        y_test_this_epoch = y_test_left_epoch[start_idx:end_idx]

        model = model.to(device)

        start_idx += sample_interval

        if dataset == 'nsl':
            # must compute the normal_temp and normal_recon_temp again, because the model has been updated
            model.eval()  # UA-SSF: ensure Dropout is disabled for deterministic template computation
            normal_recon_temp = torch.mean(F.normalize(model(x_train_this_epoch[(y_train_this_epoch == 0).squeeze()])[1], p=2, dim=1), dim=0)
            pdf1, pdf2, values1 = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, x_train_this_epoch, y_train_this_epoch, model, get_probs=True)
            pdf11, pdf22, values11 = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, 0, model, get_probs=True)
            
            pdf1_probe = pdf1 / (pdf1 + pdf2)
            pdf11_probe = pdf11 / (pdf11 + pdf22)

            # UA-SSF: 使用漂移严重程度量化替代二元检测
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
            
            control_res = pdf1_probe.cpu().numpy()
            treatment_res = pdf11_probe.cpu().numpy()
        else:
            # Get logits for drift detection
            with torch.no_grad():
                _, _, test_logits = model(x_test_this_epoch)
                test_logits = test_logits.squeeze()

            with torch.no_grad():
                _, _, train_logits = model(x_train_this_epoch)
                train_logits = train_logits.squeeze()
            
            # UA-SSF: 使用漂移严重程度量化替代二元检测
            if mode == 'ua-ssf':
                severity, p_val, drift = detect_drift_with_severity(
                    test_logits, train_logits, sample_interval, drift_threshold)
                adaptive_lwf, adaptive_epoch_1 = adaptive_params(
                    severity, base_lwf_lambda, epoch_1)
            else:
                drift = detect_drift(test_logits, train_logits, sample_interval, drift_threshold)
                severity = 1.0 if drift else 0.0
                adaptive_lwf = 0.0 if drift else lwf_lambda
                adaptive_epoch_1 = epoch_1

            control_res = train_logits.cpu().numpy()
            treatment_res = test_logits.cpu().numpy()

        # UA-SSF: estimate uncertainty before mask optimization
        # MC Dropout estimates prediction uncertainty for each sample via multiple stochastic forward passes,
        # injecting this as prior knowledge into the mask optimization process.
        if mode == 'ua-ssf':
            uncertainty_old = estimate_uncertainty(model, x_train_this_epoch, mc_forwards)
            uncertainty_new = estimate_uncertainty(model, x_test_this_epoch, mc_forwards)
        else:
            uncertainty_old = None
            uncertainty_new = None

        M_c = optimize_old_mask(control_res, treatment_res, device, initialization=old_init, lr=opt_old_lr,
                                uncertainty=uncertainty_old, uncertainty_alpha=uncertainty_alpha)
        M_t = optimize_new_mask(control_res, treatment_res, M_c, device, initialization=new_init, lr=opt_new_lr,
                                uncertainty=uncertainty_new, uncertainty_alpha=uncertainty_alpha)

        y_test_this_epoch = y_test_this_epoch.to(device)

        # 判断并应用样本更新与遗忘策略
        if drift:
            print("Drift detected, update the model...")
            # 检测到概念漂移 (Drift)：当前窗口内新到达的样本分布与历史分布存在统计显著差异
            # 策略：调用激进的更新机制，不仅淘汰无代表性的旧样本，如果缓冲池有余量，
            # 还会强行为额外的新样本打上伪标签（Pseudo-label）放入缓冲池，尽快适应新的概念分布。
            if dataset == 'nsl':
                x_train_this_epoch, y_train_this_epoch, labeled_indices_current, new_mask = select_and_update_representative_samples_when_drift(
                    x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, M_c, M_t, num_labeled_sample, device, memory, model, normal_recon_temp
                )
            else:
                x_train_this_epoch, y_train_this_epoch, labeled_indices_current, new_mask = select_and_update_representative_samples_when_drift(
                    x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, M_c, M_t, num_labeled_sample, device, memory, model
                )
        else:
            # Select and update representative samples
            x_train_this_epoch, y_train_this_epoch, labeled_indices_current, new_mask = select_and_update_representative_samples(
                x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, M_c, M_t, num_labeled_sample, device
            )

        labeled_indices.append(start_idx - sample_interval + labeled_indices_current.cpu().numpy())

        # 使用更新后的缓存池数据针对当前模型进行在线再训练 (Re-training model)
        train_ds = TensorDataset(x_train_this_epoch, y_train_this_epoch, new_mask)
        train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
        
        model = model.to(device)
        model.train()

        # ==============================
        # 统一训练循环 (UA-SSF 改进):
        # 消除了原始 SSF 中 drift/no-drift 两段几乎完全重复的训练代码。
        # 知识蒸馏权重 adaptive_lwf 由漂移严重程度连续控制：
        #   - severity=0 (无漂移): adaptive_lwf=base_lwf_lambda → 强蒸馏，防止遗忘
        #   - severity→1 (强漂移): adaptive_lwf→0 → 几乎无蒸馏，快速适应
        # 当 adaptive_lwf < 1e-6 时完全跳过蒸馏计算，与原始 SSF drift=True 行为一致。
        # ==============================
        print(f"Training with adaptive_lwf={adaptive_lwf:.4f}, epochs={adaptive_epoch_1}, severity={severity:.4f}")
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

                # 统一的知识蒸馏：权重由 adaptive_lwf 连续控制
                # 当 adaptive_lwf 极小时跳过蒸馏，与原始 SSF drift=True 行为一致
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
        
        teacher_model.load_state_dict(model.state_dict())  # Update teacher model

        if dataset == 'nsl':
            # normal_temp = torch.mean(F.normalize(model(x_train_this_epoch[(y_train_this_epoch == 0).squeeze()])[0], p=2, dim=1), dim=0)
            normal_recon_temp = torch.mean(F.normalize(model(x_train_this_epoch[(y_train_this_epoch == 0).squeeze()])[1], p=2, dim=1), dim=0)
            predict_label = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, 0, model)
        else:
            test_ds = TensorDataset(x_test_this_epoch, y_test_this_epoch)  # Replace with your test data
            test_loader = DataLoader(dataset=test_ds, batch_size=bs, shuffle=False)
            predict_label = evaluate_classifier(model, test_loader, device, get_predict=True)
        
        y_train_detection = torch.cat((y_train_detection.to(device), torch.tensor(predict_label).to(device)))

################### test the performance after online training ###################

    test_size = len(x_test)
    total_size = len(x_test_left_epoch)

    # Convert list of numpy arrays to a single numpy array
    all_labeled_indices = np.hstack(labeled_indices)

    # Mask to denote indexes with true label
    mask = np.ones(total_size, dtype=bool)
    mask[all_labeled_indices] = False

    # get the mask to denote indexes that have been true-labeled in the test dataset
    test_mask = mask[-test_size:]

    # obtain true and pseudo labels in test dataset
    y_test_left_pseudo = y_train_detection[-test_size:][test_mask].to(device)
    y_test_left_true = y_test_left_epoch[-test_size:][test_mask].to(device)
    
    print('--------------------------- performance_after_continual_training -----------------------------------')
    performance_test = score_detail(y_test_left_true.cpu().numpy(), y_test_left_pseudo.cpu().numpy())



