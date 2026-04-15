import torch
from torch import nn
import torch.fft
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import scipy.io as sio
import time
from sklearn.metrics import mean_squared_error

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.device_count())

#######################################################################
ENCO_DIM = 6  ## number of variables
WINDOW_SIZE = 1024
HID_DIM = 256 ## dimension of hidden vector
DOWN_FACTOR = int(WINDOW_SIZE/HID_DIM)
HORIZON = 8
BATCH_SIZE = 128
##################################################################
#### Load and prepare data
def minmax_norm(X, min_des, max_des):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    temp = (X - min_val) / (max_val - min_val) * (max_des - min_des) + min_des
    return temp, min_val.reshape(1, -1), max_val.reshape(1, -1)

def inverse_minmax_norm_multi(X_norm, min_val, max_val, min_des=-1, max_des=1):
    """
    将 [-1, 1] 区间的归一化数据还原为原始数据范围
    :param X_norm: [N, C]，归一化后的数据
    :param min_val: 原始最小值，[1, C]
    :param max_val: 原始最大值，[1, C]
    :param min_des: 归一化目标区间最小值（默认 -1）
    :param max_des: 归一化目标区间最大值（默认 1）
    """
    return ((X_norm - min_des) / (max_des - min_des)) * (max_val - min_val) + min_val

# def gamma_freq_loss(y_pred, gamma=0.5, weight=0.001):
#     # y_pred: [B, N, T]
#     freq = torch.fft.rfft(y_pred, dim=-1)
#     mag = torch.abs(freq)
#     gamma_penalty = torch.pow(mag + 1e-6, gamma)
#     loss_freq = weight * torch.mean(torch.sum(gamma_penalty, dim=[1, 2]))
#     return loss_freq

def gamma_norm_loss(y_pred, gamma=0.5, gamma_weight=0.001):
    """
    对反演信号 y_pred 使用 γ-范数约束，抑制冗余响应
    y_pred: [B, N, T] - N条路径的预测信号
    """
    eps = 1e-6  # 避免梯度爆炸
    gamma_term = torch.pow(torch.abs(y_pred) + eps, gamma)
    gamma_loss = gamma_weight * torch.mean(torch.sum(gamma_term, dim=[1, 2]))  # sum over N and T
    return gamma_loss

def fastX(X, X_test, window_size, horizon):
    """
    Segment the input tensor based on the given window_size and horizon.
    Parameters:
    X (torch.Tensor): Input tensor.
    window_size (int): Size of the window for segmenting X.
    horizon (int): Forecasting horizon.
    Returns:
    torch.Tensor, torch.Tensor: Segmented tensors x and x_ref.
    """
    X = X.unsqueeze(0)
    X_test = X_test.unsqueeze(0)
    n = X.shape[1]
    if n - window_size - horizon >= 0:
        x = torch.cat([X[:, i:i + window_size, :] for i in range(n - window_size - horizon + 1)], dim=0)
        x_ref = torch.cat(
            [X_test[:, i + window_size:i + window_size + horizon, :] for i in range(n - window_size - horizon + 1)], dim=0)
    else:
        x, x_ref = torch.empty(1), torch.empty(1)
    return x, x_ref

def reconstruct_signal_from_windows(predictions, original_length, window_size, horizon, crop="valid", use_window=True):
    """
    将滑窗预测结果还原为原始信号长度（重叠区域做加权平均处理，可选 Hann 窗）

    参数：
        predictions: Tensor [num_windows, num_channels, horizon]
        original_length: 原始信号长度
        window_size: 输入滑窗长度（用于定位起始位置）
        horizon: 每个窗口预测长度
        crop: 'valid' | 'same' | 'full'，裁剪策略
        use_window: 是否使用 hann window 加权重叠区域

    返回：
        Tensor [T, num_channels]：重构后的完整预测时序
    """
    num_windows, num_channels, pred_len = predictions.shape
    assert pred_len == horizon, f"预测长度应为 horizon={horizon}，实际为 {pred_len}"

    # Hann window: [horizon] → [horizon, 1] for broadcasting
    if use_window:
        window_fn = torch.hann_window(horizon).view(horizon, 1).to(predictions.device)
    else:
        window_fn = torch.ones(horizon, 1).to(predictions.device)

    # 初始化累计张量
    output = torch.zeros((original_length, num_channels), device=predictions.device)
    counts = torch.zeros((original_length, num_channels), device=predictions.device)

    for i in range(num_windows):
        start = i + window_size
        end = start + horizon
        if end > original_length:
            break

        segment = predictions[i].transpose(0, 1)  # [horizon, num_channels]
        segment_weighted = segment * window_fn  # 每个窗口加权

        output[start:end, :] += segment_weighted
        counts[start:end, :] += window_fn  # 同样加权

    counts[counts == 0] = 1e-6  # 防止除以零
    full_output = output / counts

    # 根据裁剪策略裁剪
    if crop == 'valid':
        start_idx = 2 * window_size
        end_idx = original_length - 2 * window_size - 1
        effective_output = full_output[start_idx:end_idx, :]
    elif crop == 'same':
        target_len = original_length
        current_len = full_output.shape[0]
        if current_len >= target_len:
            effective_output = full_output[:target_len, :]
        else:
            pad = torch.zeros((target_len - current_len, num_channels), device=predictions.device)
            effective_output = torch.cat([full_output, pad], dim=0)
    elif crop == 'full':
        effective_output = full_output
    else:
        raise ValueError(f"Unknown crop mode: {crop}. Choose from ['valid', 'same', 'full'].")

    return effective_output

# df = pd.read_csv('data.csv', sep=',', header=0) #### Need to change path here
# x_all = df.to_numpy()
import scipy.io as io
import glob
import os.path as osp
path = "data/geardata/dec1200rpm"  # load data
fault_dirs = glob.glob(osp.join(path, f'G*.mat'))
a_measured = io.loadmat(fault_dirs[0])["a_measured"][0:1600000, 0]

path1 = "data/geardata/dec1200rpm"
# fault_dirs1 = glob.glob(osp.join(path1, f'Z*.mat'))
fault_dirs1 = glob.glob(osp.join(path1, f'G*.mat'))
contribution_td = io.loadmat(fault_dirs1[0])["Contribution_TD"][:, 0:320000].T

x_all = a_measured[0:320000][::2].reshape(-1, 1)
x_all_o_test = a_measured[320000:640000][::2].reshape(-1, 1)

x_all, _, _ = minmax_norm(x_all, -1, 1)
x_all_o_test, x_min, x_max = minmax_norm(x_all_o_test, -1, 1)
x_all_1, _, _ = minmax_norm(contribution_td[0:160000], -1, 1)
x_all_t_test, path_min, path_max = minmax_norm(contribution_td[160000:], -1, 1)

x_all_t, x_all_ref_t = fastX(torch.from_numpy(x_all).float(), torch.from_numpy(x_all_1).float(), WINDOW_SIZE, HORIZON)
x_test, x_test_ref = fastX(torch.from_numpy(x_all_o_test).float(), torch.from_numpy(x_all_t_test).float(), WINDOW_SIZE, HORIZON)

init_dataset = torch.utils.data.TensorDataset(x_all_t.permute(0, 2, 1), x_all_ref_t.permute(0, 2, 1))
test_dataset = torch.utils.data.TensorDataset(x_test.permute(0, 2, 1), x_test_ref.permute(0, 2, 1))
training_size = int(np.round(x_all_t.shape[0] * 0.8))
train_dataset, val_dataset = torch.utils.data.random_split(init_dataset,
                                                           [training_size, x_all_t.shape[0] - training_size],
                                                           generator=torch.Generator().manual_seed(42))
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)
##################################################################
#### Load adjacency matrix here
adj = torch.tensor([
        # P1   P2   P3   P4   P5   P6
        [0.9967, 0.0923, 0.0003, 0.0025, 0.0000, 0.0003],  # P1
        [0.0923, 0.7618, 0.0016, 0.0140, 0.0002, 0.0015],  # P2
        [0.0003, 0.0016, 0.3344, 0.0080, 0.0001, 0.0005],  # P3
        [0.0025, 0.0140, 0.0080, 0.2695, 0.0005, 0.0043],  # P4
        [0.0000, 0.0002, 0.0001, 0.0005, 0.5018, 0.0187],  # P5
        [0.0003, 0.0015, 0.0005, 0.0043, 0.0187, 0.3736]  # P6
    ], dtype=torch.float64)

# 通过主对角线掩码直接置零
diag_mask = torch.eye(6, dtype=torch.bool)  # 主对角线掩码
adj_matrix = adj.clone()
adj_matrix[diag_mask] = 0  # 主对角线置零

# 保证每列和为1(因果强度归一化)
adj_matrix = adj_matrix / adj_matrix.sum(dim=0, keepdim=True)
# 获取下三角部分（包含主对角线）
lower_triangular = torch.tril(adj_matrix)
# 筛选 >0.5 的值（不满足条件的置零）
result = lower_triangular * (lower_triangular > 0.5).to(torch.float64)
A_t = result.to(device)


# ========== Causal Module ========== #
class MechanicalCausalEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, Q_t):  # Q_t: [N, 3], H, E, V
        return torch.tanh(self.linear(Q_t))  # [N, hidden_dim]


class MechanicalCausalDecoder(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=4):  # α, β, γ, η
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, H_tilde):
        return torch.sigmoid(self.linear(H_tilde))  # [N, 4]

def mechanical_dynamics(Q_t, P_t, adj, noise_std=0.01):
    """
        Q_t: [B*N, 3] - current causal states: H, E, V
        P_t: [B*N, 4] - inferred parameters: α, β, γ, η
        adj: [N, N] - normalized adjacency matrix (learned or fixed)
        """
    B_N = Q_t.shape[0]
    N = adj.shape[1]
    B = B_N // N

    H, E, V = Q_t[:, 0], Q_t[:, 1], Q_t[:, 2]
    α, β, γ, η = P_t[:, 0], P_t[:, 1], P_t[:, 2], P_t[:, 3]

    # 1. reshape E for coupling: [B, N]
    E_mat = E.view(B, N)
    # 2. compute neighbor influence: [B, N]
    E_coupled = torch.bmm(adj, E_mat.unsqueeze(-1)).squeeze(-1)  # [B, N]
    E_coupled = E_coupled.view(B * N)  # Flatten to match rest

    # 3. compute deltas
    dH = -α * H
    dE = β * H - γ * E + 0.1 * E_coupled  # coupled energy propagation
    # dE = β * H - γ * E  # coupled energy propagation
    dV = η * E + torch.randn_like(E) * noise_std

    H_new = H + dH
    E_new = E + dE
    V_new = V + dV

    return torch.stack([H_new, E_new, V_new], dim=1)  # [B*N, 3]

def path_sparsity_loss(V_t, sparsity_weight=0.001):
    """
    V_t: [B, N, 1 or 3] - vibration activation levels per path
    sparsity_weight: penalty coefficient
    """
    # 如果 V 是 3 维向量状态：[B, N, 3]，取第2维通道 (index=2) 表示 V
    if V_t.shape[-1] == 3:
        V_scalar = V_t[:, :, 2]
    else:
        V_scalar = V_t.squeeze(-1)

    # L1正则惩罚项（鼓励路径稀疏激活）
    loss_sparse = sparsity_weight * torch.mean(torch.abs(V_scalar))
    return loss_sparse


# 构建邻接估计模块
class AdjacencyLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc_q = nn.Linear(input_dim, hidden_dim)
        self.fc_k = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x: [B, C, T]
        q = self.fc_q(x)  # [B, C, d]
        k = self.fc_k(x)  # [B, C, d]
        scores = torch.matmul(q, k.transpose(1, 2))  # [B, C, C]
        adj = self.softmax(scores)  # 每行归一化
        return adj


###########################################################
class CNN_1D(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv1d(1, 64, kernel_size=DOWN_FACTOR//4, stride=DOWN_FACTOR//4, groups=1)
        # self.conv2 = nn.Conv1d(64, 64, kernel_size=DOWN_FACTOR//4, stride=DOWN_FACTOR//4, groups=16, dilation=1)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=DOWN_FACTOR//2, stride=DOWN_FACTOR//2, groups=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=DOWN_FACTOR//2, stride=DOWN_FACTOR//2, groups=16, dilation=1)
        self.conv3 = nn.Conv1d(64, 1, kernel_size=1, stride=1, groups=1)

    def forward(self, x):
        h = torch.tanh(self.conv1(x))
        h = torch.tanh(self.conv2(h))
        h = torch.tanh(self.conv3(h))
        return h

class BiLSTMPathPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=6):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 输出6个路径分量

    def forward(self, x):
        x, _ = self.bilstm(x)
        return self.fc(x)

class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.0,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=True)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=True)
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        batch_size, n_nodes = h.shape[0], h.shape[1]
        g_l = self.linear_l(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        g_l_repeat = g_l.repeat(1, n_nodes, 1, 1)  # 将每个元素重复repeats次数
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=1)  # 重复张量的元素
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(batch_size, n_nodes, n_nodes, self.n_heads, self.n_hidden)
        # adj_mat_repeat = adj_mat.repeat(batch_size, 1, 1)[:, :, :, None]
        adj_mat_repeat = adj_mat.unsqueeze(-1)  # [B, C, C, 1]

        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)
        e = e.masked_fill(adj_mat_repeat == 0, float(-1e20))  # 根据布尔掩码mask将张量中指定位置的元素替换为特定值
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('cijh,cjhf->cihf', a, g_r)  # 执行爱因斯坦求和约定
        if self.is_concat:
            return a, attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        else:
            return a, attn_res.mean(dim=2)

def frequency_loss(pred, target):
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    return torch.mean((torch.abs(pred_fft) - torch.abs(target_fft)) ** 2)

class PGCA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(ENCO_DIM))
        self.affine_bias = nn.Parameter(torch.zeros(ENCO_DIM))
        # self.fc_6 = nn.Linear(1, ENCO_DIM)  # 输出6个路径分量
        # 替代 fc_6：1D 卷积分解通道
        self.path_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, ENCO_DIM, kernel_size=1)
        )

        # self.bilstm = BiLSTMPathPredictor()
        self.networks = nn.ModuleList([CNN_1D() for _ in range(ENCO_DIM)])
        # self.networks = nn.ModuleList([FourierConv1D() for _ in range(ENCO_DIM)])

        self.Gconv1 = GraphAttentionV2Layer(in_channels, hidden_channels, 1,
                                            is_concat=True)
        self.out = nn.Conv1d(ENCO_DIM, ENCO_DIM, kernel_size=HID_DIM, stride=HID_DIM, groups=ENCO_DIM)

        self.causal_encoder = MechanicalCausalEncoder(3, hidden_channels)
        self.causal_decoder = MechanicalCausalDecoder(hidden_channels, 4)

        self.adj_net = AdjacencyLearner(input_dim=WINDOW_SIZE, hidden_dim=32)

    def forward(self, x, edge_index, intervent=False, do=None):
        #### Apply normalize
        mean = torch.mean(x, dim=2, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False))
        x = x - mean
        x = x / std
        # x = x.permute(0, 2, 1)
        # x = self.fc_6(x)
        # x = x * self.affine_weight[None, None, :] + self.affine_bias[None, None, :]
        # x = x.permute(0, 2, 1)
        x = self.path_encoder(x)  # [B, 6, T]
        x = x * self.affine_weight.view(1, -1, 1) + self.affine_bias.view(1, -1, 1)

        B = x.shape[0]
        Q_t = torch.tensor([1.0, 0.0, 0.0], device=x.device).repeat(B * ENCO_DIM, 1)

        adj_soft = self.adj_net(x)  # [B, C, C]
        adj_bin = (adj_soft > 0.5).float()  # 可选：mask处理或保留 soft attention

        if intervent:
            adj_bin_cf = adj_bin.clone()
            adj_bin_cf[:, do, :] = 0
            adj_bin_cf[:, :, do] = 0
            adj_bin = adj_bin_cf

        x_pred_all = None
        for step in range(HORIZON):  ## For multi-step ahead prediction
            h = torch.cat([self.networks[i](x[:, i:i + 1, :]) for i in range(ENCO_DIM)], dim=1)
            _, h_temp = self.Gconv1(h, adj_bin)

            H_causal = self.causal_encoder(Q_t).view(B, ENCO_DIM, -1)
            h = h + torch.tanh(h_temp) + H_causal

            # h = h + torch.tanh(h_temp)
            x_pred = self.out(h)
            x = torch.cat((x[:, :, 1:], x_pred), dim=2)

            H_tilde = h
            P_t = self.causal_decoder(H_tilde.view(B * ENCO_DIM, -1))
            Q_t = mechanical_dynamics(Q_t, P_t, adj_bin)

            x_pred_all = torch.cat((x_pred_all, x_pred), dim=2) if x_pred_all is not None else x_pred

        #### Apply denormalization
        out = x_pred_all.permute(0, 2, 1)
        out = (out - self.affine_bias[None, None, :]) / self.affine_weight[None, None, :]
        out = out.permute(0, 2, 1)
        out = out * std + mean
        return out, Q_t.view(B, ENCO_DIM, 3), P_t.view(B, ENCO_DIM, 4), adj_soft

def model_training(model, train_dl, lr):
    avg_loss = 0
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for i, (x, x_ref) in enumerate(train_dl):
        x, x_ref = x.to(device), x_ref.to(device)
        # plt.plot(torch.sum(x_ref, 1).cpu().numpy()[0])
        # plt.show()
        optimizer.zero_grad()
        output, Q_t, _, _ = model(x, A_t, intervent=False, do=None)
        # loss = criterion(output, x_ref)
        loss = sum([criterion(output[:, i, :], x_ref[:, i, :]) for i in range(ENCO_DIM)])
        loss_sparse = path_sparsity_loss(Q_t)
        # loss = loss + gamma_norm_loss(output)
        loss = loss + 0.2 * sum([frequency_loss(output[:, i, :], x_ref[:, i, :]) for i in range(ENCO_DIM)])
        loss = loss + loss_sparse
        # loss = loss + criterion(torch.sum(output, 1), torch.sum(x_ref, 1))
        # loss = loss + frequency_smoothness_loss(output)
        loss.backward()
        optimizer.step()
        avg_loss += float(loss)
    return avg_loss / (i + 1)


def evaluation(model, val_dl):
    avg_loss = 0
    loss_metric = nn.MSELoss()
    model.eval()
    for i, (x, x_ref) in enumerate(val_dl):
        x, x_ref = x.to(device), x_ref.to(device)
        output, _, _, _ = model(x, A_t, intervent=False, do=None)
        # loss = loss_metric(output, x_ref)
        loss = sum([loss_metric(output[:, i, :], x_ref[:, i, :]) for i in range(ENCO_DIM)])
        # loss = loss + gamma_norm_loss(output)
        avg_loss += float(loss)
    return avg_loss / (i + 1)


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def compute_rel_metric(out_base, out_mod):
    """
    计算相对变化度量(Relative Difference Metric)

    参数:
        out_base (torch.Tensor): 原始输出 (预测值或中间响应)
        out_mod (torch.Tensor): 干预后的输出

    返回:
        diff (float): 相对变化 (越大表示干预对输出影响越强)
    """
    # 取绝对差值的均值
    num = torch.mean(torch.abs(out_base - out_mod)).item()

    # 对原始输出的均值做归一化，防止除零
    den = torch.mean(torch.abs(out_base)).item() + 1e-12

    # 相对变化比值
    diff = (num / den)*10
    return diff


model = PGCA(HID_DIM, HID_DIM).to(device)
LEARNING_RATE = 0.001
EPOCHS = 50
BEST_VAL = float('inf')

loss_train = np.zeros((EPOCHS, 1))
loss_val = np.zeros((EPOCHS, 1))

train_epoch_time = np.zeros((EPOCHS, 1))
val_epoch_time = np.zeros((EPOCHS, 1))
total_epoch_time = np.zeros((EPOCHS, 1))

check_every = 5
check_thre = 50
lookback = 5
best_it = None
for epoch in range(EPOCHS):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    loss_train_temp = model_training(model, train_dl, LEARNING_RATE)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    loss_val_temp = evaluation(model, val_dl)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    loss_train[epoch] = loss_train_temp
    loss_val[epoch] = loss_val_temp

    train_epoch_time[epoch] = t1 - t0
    val_epoch_time[epoch] = t2 - t1
    total_epoch_time[epoch] = t2 - t0

    if (epoch + 1) % check_every == 0:
        print(('-' * 10 + 'Iter = %d' + '-' * 10) % (epoch + 1))
        print('Train MSE = ' + str(loss_train[epoch]))
        print('Validation MSE = ' + str(loss_val[epoch]))
        print('Train epoch time (s) = ' + str(train_epoch_time[epoch]))
        print('Validation time (s) = ' + str(val_epoch_time[epoch]))
        print('Total epoch time (s) = ' + str(total_epoch_time[epoch]))

    if loss_val_temp < BEST_VAL:
        BEST_VAL = loss_val_temp
        best_it = epoch
        best_model = deepcopy(model)
    elif (epoch - best_it) == 20:
        print('Stopping early')
        break

    restore_parameters(model, best_model)

    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_dim': HID_DIM,
        'window_size': WINDOW_SIZE,
        'horizon': HORIZON
    }, 'checkpoints/best_model_all.pth')

valid_train_time = train_epoch_time[:epoch + 1]
valid_val_time = val_epoch_time[:epoch + 1]
valid_total_time = total_epoch_time[:epoch + 1]

print("=" * 60)
print(f"Average Train Epoch time (s): {np.mean(valid_train_time):.6f}")
print(f"Average Validation time (s): {np.mean(valid_val_time):.6f}")
print(f"Average Total Epoch time (s): {np.mean(valid_total_time):.6f}")
print("=" * 60)

# 加载训练中保存的最佳模型参数
restore_parameters(model, best_model)
model.eval()
all_preds = []
all_q = []
all_p = []
all_adj = []

# 增加因果干预
intervention_type = "zero"
cde_list = []
all_cde = []

with torch.no_grad():
    for x, _ in test_dl:
        x = x.to(device)

        x_cf = x.clone()

        pred, qt_pred, pt_pred, adj_soft = model(x, A_t, intervent=False, do=None)  # [B, HORIZON, ENCO_DIM]
        # pred_cf_0, _, _, _ = model(x_cf, A_t, intervent=False, do=0)
        # pred_cf_1, _, _, _ = model(x_cf, A_t, intervent=False, do=1)
        # pred_cf_2, _, _, _ = model(x_cf, A_t, intervent=False, do=2)
        # pred_cf_3, _, _, _ = model(x_cf, A_t, intervent=False, do=3)
        # pred_cf_4, _, _, _ = model(x_cf, A_t, intervent=False, do=4)
        # pred_cf_5, _, _, _ = model(x_cf, A_t, intervent=False, do=5)

        all_preds.append(pred.cpu())
        all_q.append(qt_pred.cpu())
        all_p.append(pt_pred.cpu())
        all_adj.append(adj_soft.cpu())

    #     # 计算每个路径的相对变化指标
    #     rel_metrics = []
    #     for cf_pred in [pred_cf_0, pred_cf_1, pred_cf_2, pred_cf_3, pred_cf_4, pred_cf_5]:
    #         diff = compute_rel_metric(pred, cf_pred)
    #         rel_metrics.append(diff)
    #
    #     all_cde.append(rel_metrics)
    #
    # all_cde = np.array(all_cde)
    # sio.savemat("all_cde.mat", {
    #     "all_cde": all_cde
    # })
    # print(r"每个路径干预结果已保存为 all_cde.mat")
    # all_cde = np.mean(all_cde, 0)
    # print("cde_list", all_cde)

all_preds = torch.cat(all_preds, dim=0)  # [num_windows, HORIZON, ENCO_DIM]
all_q = torch.cat(all_q, dim=0)  # [N_win, C, 3] 每个通道的因果状态
all_p = torch.cat(all_p, dim=0)  # [N_win, C, 3] 每个通道的因果状态
all_adj = torch.cat(all_adj, dim=0).numpy()         # shape: [N_win, C, C]

reconstructed = reconstruct_signal_from_windows(all_preds, original_length=all_preds.shape[0],
                                                window_size=WINDOW_SIZE, horizon=HORIZON, crop="valid")
reconstructed_ins = inverse_minmax_norm_multi(reconstructed, path_min, path_max)

def plot_HEV_paths(q_state, path_idx):
    H = q_state[:, path_idx, 0]
    E = q_state[:, path_idx, 1]
    V = q_state[:, path_idx, 2]

    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1)
    plt.plot(H, label='Health (H)', color='green')
    plt.title(f'Path {path_idx+1} - Health')
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(E, label='Energy (E)', color='orange')
    plt.title(f'Path {path_idx+1} - Energy')
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(V, label='Vibration (V)', color='red')
    plt.title(f'Path {path_idx+1} - Vibration')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 使用相同的裁剪策略对 Ground Truth 进行对齐
start_idx = 2 * WINDOW_SIZE
end_idx = contribution_td.shape[0] - WINDOW_SIZE - 1
gt_clip = x_all_t_test[start_idx:end_idx]  # 与 reconstructed 对齐

# 绘图展示前5000点对比
plt.figure(figsize=(12, 4))
plt.plot(reconstructed[2048:5000+2048], label='Predicted')
plt.plot(gt_clip[:5000, 0], label='Ground Truth', alpha=0.7)
plt.title("Prediction vs Ground Truth (first 5000 samples)")
plt.xlabel("Time Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# sio.savemat("restored_prediction.mat", {"prediction": reconstructed_ins})

for ch in range(ENCO_DIM):
    plot_HEV_paths(all_q, path_idx=ch)

# 保存所有通道的 Q, P, 预测结果
sio.savemat("restored_prediction.mat", {
    "prediction": reconstructed_ins.numpy(),
    "q_state": all_q.numpy(),
    "p_state": all_p.numpy()
})
print(r"预测结果已保存为 restored_prediction.mat")

sio.savemat("all_adj.mat", {"all_adj": all_adj})
print("已保存为 all_adj.mat")


