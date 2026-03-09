"""
LSTM 时序预测模型
================
将洛伦兹轨迹视为时间序列，用滑动窗口构建数据集，
LSTM 从历史片段预测下一个时间步的状态。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ======================== 滑动窗口数据集 ========================

class LorenzSequenceDataset(Dataset):
    """
    滑动窗口数据集：从连续轨迹中提取 (输入窗口, 目标) 对。

    例如 window_size=20 时：
        输入: data[i : i+20]      → shape (20, 3)
        目标: data[i+20]          → shape (3,)
    """

    def __init__(self, data, window_size=20):
        """
        参数:
            data        : 轨迹数据, shape (N, 3)，numpy数组
            window_size : 输入窗口长度
        """
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]      # (window, 3)
        y = self.data[idx + self.window_size]             # (3,)
        return x, y


# ======================== LSTM 模型 ========================

class LorenzLSTM(nn.Module):
    """
    LSTM 时序预测模型。

    结构: 输入(3) → LSTM(hidden) → 全连接(hidden→3) → 输出(3)
    """

    def __init__(self, input_dim=3, hidden_dim=64, n_layers=2, dropout=0.1):
        """
        参数:
            input_dim  : 输入特征维度（x, y, z = 3）
            hidden_dim : LSTM隐藏层维度
            n_layers   : LSTM堆叠层数
            dropout    : Dropout比率（仅在多层LSTM之间）
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,          # 输入形状 (batch, seq, feature)
            dropout=dropout if n_layers > 1 else 0,
        )

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, input_dim),
        )

    def forward(self, x):
        """
        参数:
            x : 输入序列, shape (batch, window_size, 3)

        返回:
            pred : 预测的下一个状态, shape (batch, 3)
        """
        # LSTM编码
        lstm_out, _ = self.lstm(x)          # (batch, window, hidden)
        last_hidden = lstm_out[:, -1, :]    # 取最后一个时间步 (batch, hidden)

        # 全连接预测
        pred = self.fc(last_hidden)         # (batch, 3)
        return pred

    def predict_trajectory(self, initial_window, n_steps, device="cpu"):
        """
        自回归多步预测：用自己的预测作为下一步的输入。

        参数:
            initial_window : 初始窗口, shape (window_size, 3)，numpy或tensor
            n_steps        : 向前预测的步数
            device         : 计算设备

        返回:
            predictions : 预测轨迹, shape (n_steps, 3)，numpy数组
        """
        self.eval()
        with torch.no_grad():
            if isinstance(initial_window, np.ndarray):
                window = torch.FloatTensor(initial_window).to(device)
            else:
                window = initial_window.clone().to(device)

            predictions = []

            for _ in range(n_steps):
                # 取最近的 window_size 步作为输入
                x = window.unsqueeze(0)       # (1, window, 3)
                pred = self.forward(x)         # (1, 3)
                predictions.append(pred.squeeze(0).cpu().numpy())

                # 滑动窗口：去掉最早的，加入新预测的
                window = torch.cat([window[1:], pred], dim=0)

            return np.array(predictions)


# ======================== 工具函数 ========================

def create_dataloader(data, window_size=20, batch_size=64, shuffle=True):
    """
    创建DataLoader的便捷函数。

    参数:
        data        : numpy数组, shape (N, 3)
        window_size : 滑动窗口大小
        batch_size  : 批大小
        shuffle     : 是否打乱

    返回:
        DataLoader 对象
    """
    dataset = LorenzSequenceDataset(data, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ======================== 测试 ========================

if __name__ == "__main__":
    print("=== LSTM 模型测试 ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    model = LorenzLSTM(hidden_dim=64, n_layers=2).to(device)

    # 参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    # 测试前向传播
    fake_data = np.random.randn(1000, 3).astype(np.float32)
    loader = create_dataloader(fake_data, window_size=20, batch_size=32)

    batch_x, batch_y = next(iter(loader))
    batch_x = batch_x.to(device)
    pred = model(batch_x)
    print(f"输入形状: {batch_x.shape}")     # (32, 20, 3)
    print(f"输出形状: {pred.shape}")         # (32, 3)

    # 测试自回归预测
    window = fake_data[:20]
    preds = model.predict_trajectory(window, n_steps=100, device=device)
    print(f"自回归预测形状: {preds.shape}")  # (100, 3)