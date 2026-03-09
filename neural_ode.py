"""
Neural ODE 模型
===============
用多层感知器(MLP)拟合洛伦兹系统的动力学函数 f(x,y,z)，
再通过 torchdiffeq 的 ODE 求解器进行轨迹积分。
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint  # 使用伴随方法，内存O(1)


# ======================== 动力学网络 ========================

class LorenzDynamics(nn.Module):
    """
    学习洛伦兹系统的动力学函数：dx/dt = f_θ(x)

    该网络的输入是当前状态 [x, y, z]（3维），
    输出是状态的时间导数 [dx/dt, dy/dt, dz/dt]（3维）。
    本质上是在学习相空间中的"速度场"。
    """

    def __init__(self, hidden_dim=64, n_layers=3):
        """
        参数:
            hidden_dim : 每个隐藏层的神经元数量
            n_layers   : 隐藏层数量
        """
        super().__init__()

        layers = []

        # 输入层: 3 → hidden_dim
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.SiLU())  # SiLU(x) = x * sigmoid(x)，比ReLU更平滑

        # 隐藏层: hidden_dim → hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        # 输出层: hidden_dim → 3
        layers.append(nn.Linear(hidden_dim, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        """
        ODE求解器要求的接口: dx/dt = f(t, x)

        参数:
            t : 当前时间（洛伦兹是自治系统，不显含t，但接口需要）
            x : 当前状态, shape (batch, 3) 或 (3,)

        返回:
            dx/dt, shape 与 x 相同
        """
        return self.net(x)


# ======================== Neural ODE 封装 ========================

class NeuralODE(nn.Module):
    """
    完整的 Neural ODE 模型。
    将动力学网络与 ODE 求解器组合在一起。
    """

    def __init__(self, hidden_dim=64, n_layers=3, solver="dopri5", rtol=1e-5, atol=1e-6):
        """
        参数:
            hidden_dim : 动力学网络隐藏层维度
            n_layers   : 动力学网络隐藏层数
            solver     : ODE求解器类型
                         - 'dopri5': 自适应步长的Dormand-Prince法（推荐，精度高）
                         - 'rk4': 固定步长四阶龙格库塔
                         - 'euler': 欧拉法（最快但精度低）
            rtol, atol : 求解器的相对/绝对误差容限
        """
        super().__init__()
        self.dynamics = LorenzDynamics(hidden_dim, n_layers)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(self, x0, t_span):
        """
        给定初始状态，积分得到整条轨迹。

        参数:
            x0     : 初始状态, shape (batch, 3) 或 (3,)
            t_span : 时间点序列, shape (T,)

        返回:
            trajectory : 预测轨迹, shape (T, batch, 3) 或 (T, 3)
        """
        trajectory = odeint(
            self.dynamics,
            x0,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        return trajectory

    def predict(self, x0, t_span):
        """推理模式的预测，不计算梯度。"""
        self.eval()
        with torch.no_grad():
            return self.forward(x0, t_span)


# ======================== 测试 ========================

if __name__ == "__main__":
    print("=== Neural ODE 模型测试 ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    model = NeuralODE(hidden_dim=64, n_layers=3).to(device)

    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    # 测试前向传播
    x0 = torch.tensor([1.0, 1.0, 1.0], device=device)
    t_span = torch.linspace(0, 1, 100, device=device)
    trajectory = model(x0, t_span)
    print(f"输入初始状态: {x0.shape}")
    print(f"输出轨迹形状: {trajectory.shape}")  # 期望 (100, 3)

    # 测试batch模式
    x0_batch = torch.randn(8, 3, device=device)
    trajectory_batch = model(x0_batch, t_span)
    print(f"Batch输入形状: {x0_batch.shape}")
    print(f"Batch输出形状: {trajectory_batch.shape}")  # 期望 (100, 8, 3)