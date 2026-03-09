"""
训练模块
========
包含 Neural ODE 和 LSTM 两种模型的训练逻辑。
支持分段训练、梯度裁剪、学习率调度等策略。
"""

import torch
import torch.nn as nn
import numpy as np
import time

from neural_ode import NeuralODE
from lstm_model import LorenzLSTM, create_dataloader


# ======================== Neural ODE 训练 ========================

def train_neural_ode(
    train_data,
    t_train,
    device="cpu",
    hidden_dim=64,
    n_layers=3,
    n_epochs=300,
    segment_len=50,
    lr=1e-3,
    weight_decay=1e-5,
    grad_clip=1.0,
    print_every=20,
):
    """
    训练 Neural ODE 模型。

    核心思路（分段训练）：
    - 将长轨迹切成多个短片段（segment）
    - 每个片段以第一个点为初始条件，让 ODE 积分到片段末尾
    - 计算积分结果与真实轨迹的 MSE 损失
    - 这样避免了一次性预测过长轨迹导致的梯度消失/爆炸

    参数:
        train_data   : 训练轨迹, shape (N, 3)，numpy数组（已标准化）
        t_train      : 对应的时间数组, shape (N,)
        device       : 'cpu' 或 'cuda'
        hidden_dim   : 网络隐藏层维度
        n_layers     : 网络层数
        n_epochs     : 训练轮数
        segment_len  : 训练片段长度（时间步数）
        lr           : 初始学习率
        weight_decay : L2正则化系数
        grad_clip    : 梯度裁剪阈值
        print_every  : 每隔多少轮打印一次

    返回:
        model   : 训练好的模型
        losses  : 训练损失记录列表
    """
    print("=" * 50)
    print("    Neural ODE 训练")
    print("=" * 50)

    # 初始化模型
    model = NeuralODE(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    # 准备数据
    data_tensor = torch.FloatTensor(train_data).to(device)   # (N, 3)
    dt = t_train[1] - t_train[0]

    # 优化器和学习率调度
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    criterion = nn.MSELoss()
    losses = []
    best_loss = float("inf")
    best_state = None

    # 计算可用的片段数量
    n_segments = len(train_data) // segment_len
    print(f"数据点数: {len(train_data)}, 片段长度: {segment_len}, 片段数: {n_segments}")
    print(f"开始训练 ({n_epochs} epochs)...\n")

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        # 随机打乱片段顺序
        indices = np.random.permutation(n_segments)

        for idx in indices:
            # 取出一个片段
            start = idx * segment_len
            end = start + segment_len
            segment_data = data_tensor[start:end]         # (segment_len, 3)
            segment_t = torch.linspace(0, (segment_len - 1) * dt, segment_len).to(device)

            # 前向传播：从片段起点积分
            x0 = segment_data[0]                           # (3,)
            pred_traj = model(x0, segment_t)               # (segment_len, 3)

            # 计算损失
            loss = criterion(pred_traj, segment_data)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止混沌系统导致梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / n_segments
        losses.append(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 打印训练信息
        if epoch % print_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:>4d}/{n_epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Best: {best_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.1f}s")

    # 加载最佳模型参数
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"\n训练完成! 最终 Best Loss: {best_loss:.6f}")

    return model, losses


# ======================== LSTM 训练 ========================

def train_lstm(
    train_data,
    device="cpu",
    hidden_dim=64,
    n_layers=2,
    window_size=20,
    batch_size=64,
    n_epochs=100,
    lr=1e-3,
    grad_clip=1.0,
    print_every=10,
):
    """
    训练 LSTM 时序预测模型。

    参数:
        train_data  : 训练轨迹, shape (N, 3)，numpy数组（已标准化）
        device      : 'cpu' 或 'cuda'
        hidden_dim  : LSTM隐藏维度
        n_layers    : LSTM层数
        window_size : 输入窗口大小
        batch_size  : 批大小
        n_epochs    : 训练轮数
        lr          : 学习率
        grad_clip   : 梯度裁剪
        print_every : 打印间隔

    返回:
        model  : 训练好的模型
        losses : 损失记录
    """
    print("\n" + "=" * 50)
    print("    LSTM 训练")
    print("=" * 50)

    # 初始化模型
    model = LorenzLSTM(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    # 创建DataLoader
    train_loader = create_dataloader(
        train_data, window_size=window_size, batch_size=batch_size, shuffle=True
    )
    print(f"训练样本数: {len(train_loader.dataset)}, 批次数: {len(train_loader)}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    losses = []
    best_loss = float("inf")
    best_state = None

    print(f"开始训练 ({n_epochs} epochs)...\n")
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)   # (batch, window, 3)
            batch_y = batch_y.to(device)   # (batch, 3)

            pred = model(batch_x)           # (batch, 3)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % print_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:>4d}/{n_epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Best: {best_loss:.6f} | "
                  f"Time: {elapsed:.1f}s")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"\n训练完成! 最终 Best Loss: {best_loss:.6f}")

    return model, losses