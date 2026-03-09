"""
可视化模块
==========
包含洛伦兹系统分析所需的所有可视化函数：
- 3D 吸引子
- 时间序列对比
- 预测误差分析
- 训练损失曲线
- 初始条件敏感性展示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

plt.style.use("seaborn-v0_8-whitegrid")

# 全局设置：支持中文 + 美观样式
# Windows优先使用微软雅黑
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",  # Windows 微软雅黑
    "SimHei",           # Windows 黑体
]
matplotlib.rcParams["axes.unicode_minus"] = False
# 清除字体缓存，确保新设置生效
matplotlib.font_manager._load_fontmanager(try_read_cache=False)
plt.style.use("seaborn-v0_8-whitegrid")


# ======================== 1. 3D 吸引子 ========================

def plot_attractor_3d(data, title="洛伦兹吸引子", save_path=None):
    """
    绘制经典的3D蝴蝶形吸引子。

    参数:
        data      : shape (N, 3) 的轨迹数据
        title     : 图标题
        save_path : 保存路径（None则显示）
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 用颜色渐变表示时间演化
    N = len(data)
    colors = cm.viridis(np.linspace(0, 1, N))

    # 分段绘制以实现颜色渐变
    for i in range(0, N - 1, 5):
        end = min(i + 6, N)
        ax.plot(
            data[i:end, 0], data[i:end, 1], data[i:end, 2],
            color=colors[i], linewidth=0.5, alpha=0.8,
        )

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.view_init(elev=25, azim=130)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {save_path}")
    plt.show()


# ======================== 2. 时间序列图 ========================

def plot_time_series(t, data, title="洛伦兹系统时间序列", save_path=None):
    """
    分别绘制 x(t), y(t), z(t) 的时间演化曲线。
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    labels = ["x(t)", "y(t)", "z(t)"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(t, data[:, i], color=color, linewidth=0.8, alpha=0.9)
        ax.set_ylabel(label, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("时间 t", fontsize=13)
    axes[0].set_title(title, fontsize=15, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {save_path}")
    plt.show()


# ======================== 3. 预测 vs 真实对比 ========================

def plot_prediction_comparison(
    t, true_data, pred_data_dict, title="预测轨迹对比", save_path=None
):
    """
    在同一张图中对比真实轨迹和多个模型的预测轨迹。

    参数:
        t              : 时间数组
        true_data      : 真实轨迹, shape (N, 3)
        pred_data_dict : {"模型名": 预测数据} 字典，预测数据 shape (N, 3)
        title          : 标题
        save_path      : 保存路径
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    var_names = ["x(t)", "y(t)", "z(t)"]

    pred_colors = ["#FF5722", "#9C27B0", "#FF9800"]

    for i, (ax, var) in enumerate(zip(axes, var_names)):
        # 真实轨迹
        ax.plot(t, true_data[:, i], "b-", linewidth=1.2, alpha=0.8, label="真实轨迹")

        # 各模型预测
        for j, (name, pred) in enumerate(pred_data_dict.items()):
            ax.plot(
                t[:len(pred)], pred[:, i],
                "--", color=pred_colors[j % len(pred_colors)],
                linewidth=1.2, alpha=0.8, label=f"{name} 预测",
            )

        ax.set_ylabel(var, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("时间 t", fontsize=13)
    axes[0].set_title(title, fontsize=15, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {save_path}")
    plt.show()


# ======================== 4. 3D 预测对比 ========================

def plot_3d_comparison(true_data, pred_data, model_name="Neural ODE", save_path=None):
    """
    在3D空间中对比真实轨迹和预测轨迹。
    """
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 真实轨迹
    ax.plot(
        true_data[:, 0], true_data[:, 1], true_data[:, 2],
        "b-", linewidth=0.6, alpha=0.5, label="真实轨迹",
    )

    # 预测轨迹
    ax.plot(
        pred_data[:, 0], pred_data[:, 1], pred_data[:, 2],
        "r-", linewidth=0.8, alpha=0.8, label=f"{model_name} 预测",
    )

    # 标记起点
    ax.scatter(*true_data[0], c="green", s=100, marker="o", label="起点", zorder=5)

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.set_title(f"3D轨迹对比 — {model_name}", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.view_init(elev=25, azim=130)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {save_path}")
    plt.show()


# ======================== 5. 预测误差分析 ========================

def plot_prediction_error(t, true_data, pred_data_dict, save_path=None):
    """
    绘制各模型的预测误差随时间的演化。
    展示混沌系统的"有效预测时间窗口"。
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    colors = ["#FF5722", "#9C27B0", "#FF9800"]

    # 上图：欧氏距离误差
    ax = axes[0]
    for j, (name, pred) in enumerate(pred_data_dict.items()):
        n = min(len(true_data), len(pred))
        error = np.linalg.norm(true_data[:n] - pred[:n], axis=1)
        ax.plot(t[:n], error, color=colors[j % len(colors)], linewidth=1.5, label=name)

    ax.set_ylabel("欧氏距离误差", fontsize=13)
    ax.set_title("预测误差随时间的演化", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 下图：对数尺度（更清楚地看到误差指数增长）
    ax = axes[1]
    for j, (name, pred) in enumerate(pred_data_dict.items()):
        n = min(len(true_data), len(pred))
        error = np.linalg.norm(true_data[:n] - pred[:n], axis=1)
        error = np.clip(error, 1e-10, None)  # 避免log(0)
        ax.semilogy(t[:n], error, color=colors[j % len(colors)], linewidth=1.5, label=name)

    ax.set_xlabel("时间 t", fontsize=13)
    ax.set_ylabel("误差 (对数尺度)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {save_path}")
    plt.show()


# ======================== 6. 训练损失曲线 ========================

def plot_training_loss(loss_dict, save_path=None):
    """
    绘制训练损失曲线。

    参数:
        loss_dict : {"模型名": [loss列表]} 字典
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#FF5722"]

    for i, (name, losses) in enumerate(loss_dict.items()):
        epochs = range(1, len(losses) + 1)
        ax.semilogy(epochs, losses, color=colors[i % len(colors)], linewidth=2, label=name)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss (对数尺度)", fontsize=13)
    ax.set_title("训练损失曲线", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {save_path}")
    plt.show()


# ======================== 7. 初始条件敏感性 ========================

def plot_sensitivity(trajectories, labels, save_path=None):
    """
    展示初始条件敏感性：微小的初始差异如何导致轨迹发散。

    参数:
        trajectories : 列表，每个元素是 (t, data) 元组
        labels       : 对应的标签列表
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    var_names = ["x(t)", "y(t)", "z(t)"]
    cmap = cm.get_cmap("tab10")

    for i, (ax, var) in enumerate(zip(axes, var_names)):
        for j, ((t, data), label) in enumerate(zip(trajectories, labels)):
            ax.plot(t, data[:, i], color=cmap(j), linewidth=1, alpha=0.8, label=label)
        ax.set_ylabel(var, fontsize=13, fontweight="bold")
        if i == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("时间 t", fontsize=13)
    axes[0].set_title("初始条件敏感性 — 蝴蝶效应", fontsize=15, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {save_path}")
    plt.show()


# ======================== 8. 二维相图 ========================

def plot_phase_portraits(data, save_path=None):
    """
    绘制三组二维相图：X-Y, X-Z, Y-Z。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]

    for ax, (i, j, xi, xj) in zip(axes, pairs):
        N = len(data)
        colors = cm.viridis(np.linspace(0, 1, N))
        for k in range(0, N - 1, 5):
            end = min(k + 6, N)
            ax.plot(
                data[k:end, i], data[k:end, j],
                color=colors[k], linewidth=0.4, alpha=0.8,
            )
        ax.set_xlabel(xi, fontsize=12)
        ax.set_ylabel(xj, fontsize=12)
        ax.set_title(f"{xi}-{xj} 相图", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.suptitle("洛伦兹系统二维相图", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {save_path}")
    plt.show()