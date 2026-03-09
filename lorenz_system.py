"""
洛伦兹混沌系统 - 方程定义与数据生成
===================================
使用经典参数 σ=10, ρ=28, β=8/3，通过 scipy 的 RK45 求解器生成高精度轨迹数据。
"""

import numpy as np
from scipy.integrate import solve_ivp


# ======================== 洛伦兹方程 ========================

def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    洛伦兹方程组的右端函数。

    参数:
        t     : 时间（ODE求解器需要，但洛伦兹系统是自治的，不显含t）
        state : [x, y, z] 三个状态变量
        sigma : Prandtl数，默认10
        rho   : Rayleigh数，默认28
        beta  : 几何因子，默认8/3

    返回:
        [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


# ======================== 轨迹数据生成 ========================

def generate_trajectory(
    initial_state=(1.0, 1.0, 1.0),
    t_span=(0, 30),
    dt=0.01,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
):
    """
    生成洛伦兹系统的轨迹数据。

    参数:
        initial_state : 初始状态 (x0, y0, z0)
        t_span        : 时间范围 (起始, 结束)
        dt            : 采样时间步长
        sigma, rho, beta : 洛伦兹参数

    返回:
        t    : 时间数组, shape (N,)
        data : 轨迹数组, shape (N, 3)，列分别为 x, y, z
    """
    # 生成评估时间点
    t_eval = np.arange(t_span[0], t_span[1], dt)

    # 使用 RK45（自适应四阶龙格库塔）求解
    sol = solve_ivp(
        fun=lambda t, s: lorenz(t, s, sigma, rho, beta),
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,  # 高精度，确保轨迹可靠
        atol=1e-12,
    )

    t = sol.t              # shape (N,)
    data = sol.y.T         # shape (N, 3)，每行是 [x, y, z]

    return t, data


def generate_multi_trajectory(n_trajectories=5, dt=0.01, t_span=(0, 30), seed=42):
    """
    从不同初始条件生成多条轨迹，用于增强训练数据的多样性。

    参数:
        n_trajectories : 轨迹条数
        dt             : 时间步长
        t_span         : 时间范围
        seed           : 随机种子

    返回:
        trajectories : 列表，每个元素是 (t, data)
    """
    rng = np.random.RandomState(seed)
    trajectories = []

    for i in range(n_trajectories):
        # 在吸引子附近随机选取初始条件
        x0 = rng.uniform(-15, 15)
        y0 = rng.uniform(-15, 15)
        z0 = rng.uniform(10, 40)
        initial_state = (x0, y0, z0)

        t, data = generate_trajectory(
            initial_state=initial_state, t_span=t_span, dt=dt
        )
        trajectories.append((t, data))
        print(f"  轨迹 {i + 1}/{n_trajectories} 生成完毕, "
              f"初始条件: ({x0:.2f}, {y0:.2f}, {z0:.2f}), "
              f"数据点数: {len(t)}")

    return trajectories


# ======================== 数据预处理 ========================

def normalize_data(data):
    """
    Z-Score标准化，返回标准化数据及统计量。

    参数:
        data : shape (N, 3) 的原始轨迹

    返回:
        normalized : 标准化后的数据
        mean       : 均值 shape (3,)
        std        : 标准差 shape (3,)
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    normalized = (data - mean) / std
    return normalized, mean, std


def denormalize_data(normalized, mean, std):
    """将标准化数据还原为原始尺度。"""
    return normalized * std + mean


# ======================== 测试 ========================

if __name__ == "__main__":
    print("=== 洛伦兹系统数据生成测试 ===\n")

    t, data = generate_trajectory()
    print(f"时间范围: {t[0]:.2f} ~ {t[-1]:.2f}")
    print(f"数据形状: {data.shape}")
    print(f"x 范围: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]")
    print(f"y 范围: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")
    print(f"z 范围: [{data[:, 2].min():.2f}, {data[:, 2].max():.2f}]")

    norm_data, mean, std = normalize_data(data)
    print(f"\n标准化后均值: {norm_data.mean(axis=0)}")
    print(f"标准化后标准差: {norm_data.std(axis=0)}")