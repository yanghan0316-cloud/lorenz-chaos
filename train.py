"""
训练模块
========
包含 Neural ODE 和 LSTM 两种模型的训练逻辑。
支持批并行ODE求解、课程学习、梯度裁剪、学习率调度等策略。
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
    训练 Neural ODE 模型（课程学习 + 批并行）。

    核心策略：
    - 课程学习：从短片段逐步过渡到长片段，先学局部动力学再学全局
    - 批并行：每个阶段将所有片段打包成 batch，一次性送入 ODE 求解器
    - 随机采样：每个 epoch 随机选取片段起点，增加数据多样性

    参数:
        train_data   : 训练轨迹, shape (N, 3)，numpy数组（已标准化）
        t_train      : 对应的时间数组, shape (N,)
        device       : 'cpu' 或 'cuda'
        hidden_dim   : 网络隐藏层维度
        n_layers     : 网络层数
        n_epochs     : 总训练轮数（会按比例分配到各阶段）
        segment_len  : 最终目标片段长度（时间步数）
        lr           : 基础学习率
        weight_decay : L2正则化系数
        grad_clip    : 梯度裁剪阈值
        print_every  : 每隔多少轮打印一次

    返回:
        model   : 训练好的模型
        losses  : 训练损失记录列表
    """
    print("=" * 50)
    print("    Neural ODE 训练（课程学习 + 批并行）")
    print("=" * 50)

    # 初始化模型
    model = NeuralODE(hidden_dim=hidden_dim, n_layers=n_layers, solver="rk4").to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    # 准备数据
    data_tensor = torch.FloatTensor(train_data).to(device)   # (N, 3)
    dt = t_train[1] - t_train[0]
    N = len(train_data)

    criterion = nn.MSELoss()
    losses = []
    best_loss = float("inf")
    best_state = None

    # ---- 课程学习：定义三个训练阶段 ----
    # (片段长度, epoch数, 学习率, batch中的片段数)
    stages = [
        (10,          n_epochs // 5,       lr * 3,   min(N // 10, 256)),
        (25,          n_epochs // 5,       lr * 2,   min(N // 25, 192)),
        (segment_len, n_epochs * 3 // 5,   lr,       min(N // segment_len, 128)),
    ]

    print(f"数据点数: {N}, 目标片段长度: {segment_len}")
    print(f"课程学习策略:")
    for i, (seg, eps, lr_s, bs) in enumerate(stages):
        print(f"  阶段{i+1}: 片段={seg}步({seg*dt:.2f}s), "
              f"epochs={eps}, lr={lr_s:.1e}, batch={bs}")
    print()

    start_time = time.time()
    total_epoch = 0

    for stage_idx, (cur_seg_len, stage_epochs, stage_lr, batch_size) in enumerate(stages):
        print(f"--- 阶段 {stage_idx + 1}/{len(stages)}: 片段长度={cur_seg_len} ---")

        # 每个阶段重建优化器
        optimizer = torch.optim.Adam(
            model.parameters(), lr=stage_lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_epochs
        )

        # 预计算该阶段的时间序列
        segment_t = torch.linspace(0, (cur_seg_len - 1) * dt, cur_seg_len).to(device)

        for epoch in range(1, stage_epochs + 1):
            total_epoch += 1
            model.train()

            # 随机采样片段起点
            max_start = N - cur_seg_len
            starts = np.random.randint(0, max_start, size=batch_size)

            # 构建 batch 数据: (batch_size, cur_seg_len, 3)
            batch_segments = torch.stack([
                data_tensor[s: s + cur_seg_len] for s in starts
            ])  # (batch_size, cur_seg_len, 3)

            # 取所有片段的起点: (batch_size, 3)
            x0_batch = batch_segments[:, 0, :]

            # 批并行 ODE 求解
            # 输出: (cur_seg_len, batch_size, 3)
            pred_traj = model(x0_batch, segment_t)

            # 转置为 (batch_size, cur_seg_len, 3) 以匹配目标
            pred_traj = pred_traj.transpose(0, 1)

            # 计算损失
            loss = criterion(pred_traj, batch_segments)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            avg_loss = loss.item()
            losses.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % print_every == 0 or epoch == 1:
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {total_epoch:>4d} (阶段内{epoch:>3d}/{stage_epochs}) | "
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