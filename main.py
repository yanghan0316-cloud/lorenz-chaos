"""
洛伦兹混沌系统的神经网络动态演变与预测
========================================
主程序入口 — 串联数据生成 → 训练 → 预测 → 可视化全流程

使用方法:
    python main.py              # 首次运行：训练 + 预测 + 出图
    python main.py --skip-train # 跳过训练：加载已有模型，只做预测和出图

可通过修改下方 CONFIG 字典调整所有超参数。
"""

import os
import sys
import torch
import numpy as np

from lorenz_system import generate_trajectory, normalize_data, denormalize_data
from neural_ode import NeuralODE
from lstm_model import LorenzLSTM
from train import train_neural_ode, train_lstm
from visualize import (
    plot_attractor_3d,
    plot_time_series,
    plot_prediction_comparison,
    plot_3d_comparison,
    plot_prediction_error,
    plot_training_loss,
    plot_sensitivity,
    plot_phase_portraits,
)


# ======================== 全局配置 ========================

CONFIG = {
    # --- 数据 ---
    "initial_state": (1.0, 1.0, 1.0),   # 初始条件
    "t_span": (0, 40),                    # 总时间范围
    "dt": 0.01,                           # 时间步长
    "train_ratio": 0.8,                   # 训练集比例

    # --- Neural ODE ---
    "node_hidden_dim": 64,                # 隐藏层维度
    "node_n_layers": 3,                   # 隐藏层数
    "node_epochs": 300,                   # 训练轮数
    "node_segment_len": 50,               # 分段训练片段长度
    "node_lr": 1e-3,                      # 学习率

    # --- LSTM ---
    "lstm_hidden_dim": 64,                # LSTM隐藏维度
    "lstm_n_layers": 2,                   # LSTM层数
    "lstm_window_size": 20,               # 输入窗口长度
    "lstm_batch_size": 64,                # 批大小
    "lstm_epochs": 100,                   # 训练轮数
    "lstm_lr": 1e-3,                      # 学习率

    # --- 输出 ---
    "output_dir": "results",              # 图片保存目录
}


def main():
    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    save = lambda name: os.path.join(CONFIG["output_dir"], name)

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ============================================================
    # 第一步：数据生成
    # ============================================================
    print("=" * 60)
    print("  第一步：生成洛伦兹系统轨迹数据")
    print("=" * 60)

    t_all, data_all = generate_trajectory(
        initial_state=CONFIG["initial_state"],
        t_span=CONFIG["t_span"],
        dt=CONFIG["dt"],
    )

    print(f"数据形状: {data_all.shape}")
    print(f"时间范围: {t_all[0]:.2f} ~ {t_all[-1]:.2f}")
    print(f"总数据点: {len(t_all)}")

    # 划分训练集/测试集
    split = int(len(t_all) * CONFIG["train_ratio"])
    t_train, data_train = t_all[:split], data_all[:split]
    t_test, data_test = t_all[split:], data_all[split:]
    print(f"训练集: {len(t_train)} 点, 测试集: {len(t_test)} 点\n")

    # 标准化（使用训练集的统计量）
    data_train_norm, mean, std = normalize_data(data_train)
    data_test_norm = (data_test - mean) / std
    data_all_norm = (data_all - mean) / std

    # ============================================================
    # 第二步：可视化原始数据
    # ============================================================
    print("=" * 60)
    print("  第二步：可视化原始洛伦兹系统")
    print("=" * 60)

    print("绘制 3D 吸引子...")
    plot_attractor_3d(data_all, save_path=save("01_attractor_3d.png"))

    print("绘制时间序列...")
    plot_time_series(t_all, data_all, save_path=save("02_time_series.png"))

    print("绘制二维相图...")
    plot_phase_portraits(data_all, save_path=save("03_phase_portraits.png"))

    # 初始条件敏感性展示
    print("绘制蝴蝶效应演示...")
    sensitivity_trajs = []
    sensitivity_labels = []
    base_state = np.array([1.0, 1.0, 1.0])
    perturbations = [0, 1e-8, 1e-6, 1e-4]

    for eps in perturbations:
        init = tuple(base_state + eps)
        t_s, d_s = generate_trajectory(initial_state=init, t_span=(0, 30), dt=0.01)
        sensitivity_trajs.append((t_s, d_s))
        if eps == 0:
            sensitivity_labels.append("基准轨迹")
        else:
            sensitivity_labels.append(f"扰动 ε={eps:.0e}")

    plot_sensitivity(sensitivity_trajs, sensitivity_labels, save_path=save("04_sensitivity.png"))

    # ============================================================
    # 第三步 & 第四步：训练或加载模型
    # ============================================================
    skip_train = "--skip-train" in sys.argv

    node_model_path = save("neural_ode_model.pth")
    lstm_model_path = save("lstm_model.pth")
    losses_path = save("training_losses.npz")

    if skip_train and os.path.exists(node_model_path) and os.path.exists(lstm_model_path):
        # ---- 加载已有模型，跳过训练 ----
        print("\n" + "=" * 60)
        print("  跳过训练：加载已保存的模型")
        print("=" * 60)

        node_model = NeuralODE(
            hidden_dim=CONFIG["node_hidden_dim"],
            n_layers=CONFIG["node_n_layers"],
            solver="rk4"
        ).to(device)
        node_model.load_state_dict(torch.load(node_model_path, map_location=device))
        node_model.eval()
        print(f"  Neural ODE 已加载: {node_model_path}")

        lstm_model = LorenzLSTM(
            hidden_dim=CONFIG["lstm_hidden_dim"],
            n_layers=CONFIG["lstm_n_layers"],
        ).to(device)
        lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
        lstm_model.eval()
        print(f"  LSTM 已加载: {lstm_model_path}")

        # 尝试加载训练损失记录（用于画损失曲线）
        if os.path.exists(losses_path):
            losses_data = np.load(losses_path, allow_pickle=True)
            node_losses = losses_data["node_losses"].tolist()
            lstm_losses = losses_data["lstm_losses"].tolist()
        else:
            node_losses, lstm_losses = [], []

    else:
        # ---- 正常训练 ----
        if skip_train:
            print("\n  未找到已保存的模型，将重新训练...\n")

        print("\n" + "=" * 60)
        print("  第三步：训练 Neural ODE")
        print("=" * 60)

        node_model, node_losses = train_neural_ode(
            train_data=data_train_norm,
            t_train=t_train,
            device=device,
            hidden_dim=CONFIG["node_hidden_dim"],
            n_layers=CONFIG["node_n_layers"],
            n_epochs=CONFIG["node_epochs"],
            segment_len=CONFIG["node_segment_len"],
            lr=CONFIG["node_lr"],
        )

        print("\n" + "=" * 60)
        print("  第四步：训练 LSTM")
        print("=" * 60)

        lstm_model, lstm_losses = train_lstm(
            train_data=data_train_norm,
            device=device,
            hidden_dim=CONFIG["lstm_hidden_dim"],
            n_layers=CONFIG["lstm_n_layers"],
            window_size=CONFIG["lstm_window_size"],
            batch_size=CONFIG["lstm_batch_size"],
            n_epochs=CONFIG["lstm_epochs"],
            lr=CONFIG["lstm_lr"],
        )

        # 保存训练损失（供下次 skip-train 时画图用）
        np.savez(
            losses_path,
            node_losses=np.array(node_losses),
            lstm_losses=np.array(lstm_losses),
        )

    # ============================================================
    # 第五步：测试集预测
    # ============================================================
    print("\n" + "=" * 60)
    print("  第五步：测试集预测")
    print("=" * 60)

    # --- Neural ODE 预测 ---
    print("Neural ODE 预测中...")
    x0_test = torch.FloatTensor(data_test_norm[0]).to(device)
    t_test_tensor = torch.FloatTensor(t_test - t_test[0]).to(device)
    node_pred_norm = node_model.predict(x0_test, t_test_tensor).cpu().numpy()
    node_pred = denormalize_data(node_pred_norm, mean, std)
    print(f"  Neural ODE 预测完成, 形状: {node_pred.shape}")

    # --- LSTM 预测 ---
    print("LSTM 预测中...")
    ws = CONFIG["lstm_window_size"]
    # 从训练集末尾取初始窗口
    initial_window = data_train_norm[-ws:]
    lstm_pred_norm = lstm_model.predict_trajectory(
        initial_window, n_steps=len(t_test), device=device
    )
    lstm_pred = denormalize_data(lstm_pred_norm, mean, std)
    print(f"  LSTM 预测完成, 形状: {lstm_pred.shape}")

    # ============================================================
    # 第六步：结果可视化
    # ============================================================
    print("\n" + "=" * 60)
    print("  第六步：结果可视化")
    print("=" * 60)

    pred_dict = {"Neural ODE": node_pred, "LSTM": lstm_pred}

    # 训练损失对比
    if node_losses and lstm_losses:
        print("绘制训练损失曲线...")
        plot_training_loss(
            {"Neural ODE": node_losses, "LSTM": lstm_losses},
            save_path=save("05_training_loss.png"),
        )
    else:
        print("无训练损失记录，跳过损失曲线绘制")

    # 时间序列预测对比
    print("绘制预测对比图...")
    plot_prediction_comparison(
        t_test, data_test, pred_dict,
        title="测试集预测对比",
        save_path=save("06_prediction_comparison.png"),
    )

    # 3D 轨迹对比
    print("绘制 3D 对比图...")
    plot_3d_comparison(
        data_test, node_pred,
        model_name="Neural ODE",
        save_path=save("07_3d_node_comparison.png"),
    )
    plot_3d_comparison(
        data_test, lstm_pred,
        model_name="LSTM",
        save_path=save("08_3d_lstm_comparison.png"),
    )

    # 误差分析
    print("绘制误差分析图...")
    plot_prediction_error(
        t_test, data_test, pred_dict,
        save_path=save("09_prediction_error.png"),
    )

    # ============================================================
    # 第七步：定量评估
    # ============================================================
    print("\n" + "=" * 60)
    print("  第七步：定量评估")
    print("=" * 60)

    for name, pred in pred_dict.items():
        n = min(len(data_test), len(pred))
        mse = np.mean((data_test[:n] - pred[:n]) ** 2)
        mae = np.mean(np.abs(data_test[:n] - pred[:n]))

        # 有效预测时间（误差首次超过吸引子尺度的时间）
        errors = np.linalg.norm(data_test[:n] - pred[:n], axis=1)
        attractor_size = np.std(data_all) * 2  # 吸引子的大致"直径"
        valid_mask = errors < attractor_size
        if valid_mask.any():
            # 找到第一个误差超标的位置
            invalid_indices = np.where(~valid_mask)[0]
            if len(invalid_indices) > 0:
                valid_time = (invalid_indices[0]) * CONFIG["dt"]
            else:
                valid_time = n * CONFIG["dt"]
        else:
            valid_time = 0

        print(f"\n  {name}:")
        print(f"    MSE:  {mse:.6f}")
        print(f"    MAE:  {mae:.6f}")
        print(f"    有效预测时间: ~{valid_time:.2f} 个时间单位")

    # ============================================================
    # 保存模型（仅在训练模式下）
    # ============================================================
    if not skip_train:
        print("\n" + "=" * 60)
        print("  保存模型")
        print("=" * 60)

        torch.save(node_model.state_dict(), save("neural_ode_model.pth"))
        torch.save(lstm_model.state_dict(), save("lstm_model.pth"))
        np.savez(save("normalization_params.npz"), mean=mean, std=std)
        print("模型和参数已保存到", CONFIG["output_dir"])

    print("\n" + "=" * 60)
    print("  全部完成!")
    print("=" * 60)
    print(f"\n所有图片和模型已保存到 ./{CONFIG['output_dir']}/ 目录")


if __name__ == "__main__":
    main()