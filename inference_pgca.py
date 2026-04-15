import os
import glob
import time
import numpy as np
import scipy.io as io
import scipy.io as sio
import torch
from torch import nn

# =========================
# Device
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Config
# =========================
ENCO_DIM = 6
BATCH_SIZE = 128

# checkpoint path
CKPT_PATH = "checkpoints/best_model_all.pth"

# data path 
DATA_PATH = "data/geardata/dec1200rpm"

# =========================
# Utils
# =========================
def minmax_norm(X, min_des, max_des):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    temp = (X - min_val) / (max_val - min_val) * (max_des - min_des) + min_des
    return temp, min_val.reshape(1, -1), max_val.reshape(1, -1)

def inverse_minmax_norm_multi(X_norm, min_val, max_val, min_des=-1, max_des=1):
    return ((X_norm - min_des) / (max_des - min_des)) * (max_val - min_val) + min_val

def fastX(X, X_test, window_size, horizon):
    X = X.unsqueeze(0)
    X_test = X_test.unsqueeze(0)
    n = X.shape[1]
    if n - window_size - horizon >= 0:
        x = torch.cat([X[:, i:i + window_size, :] for i in range(n - window_size - horizon + 1)], dim=0)
        x_ref = torch.cat(
            [X_test[:, i + window_size:i + window_size + horizon, :] for i in range(n - window_size - horizon + 1)],
            dim=0
        )
    else:
        x, x_ref = torch.empty(1), torch.empty(1)
    return x, x_ref

def reconstruct_signal_from_windows(predictions, original_length, window_size, horizon, crop="valid", use_window=True):
    num_windows, num_channels, pred_len = predictions.shape
    assert pred_len == horizon, f"预测长度应为 horizon={horizon}，实际为 {pred_len}"

    if use_window:
        window_fn = torch.hann_window(horizon).view(horizon, 1).to(predictions.device)
    else:
        window_fn = torch.ones(horizon, 1).to(predictions.device)

    output = torch.zeros((original_length, num_channels), device=predictions.device)
    counts = torch.zeros((original_length, num_channels), device=predictions.device)

    for i in range(num_windows):
        start = i + window_size
        end = start + horizon
        if end > original_length:
            break

        segment = predictions[i].transpose(0, 1)  # [horizon, num_channels]
        segment_weighted = segment * window_fn

        output[start:end, :] += segment_weighted
        counts[start:end, :] += window_fn

    counts[counts == 0] = 1e-6
    full_output = output / counts

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

# =========================
# Causal / graph modules
# =========================
class MechanicalCausalEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, Q_t):
        return torch.tanh(self.linear(Q_t))

class MechanicalCausalDecoder(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=4):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, H_tilde):
        return torch.sigmoid(self.linear(H_tilde))

def mechanical_dynamics(Q_t, P_t, adj, noise_std=0.01):
    """
    Q_t: [B*N, 3]
    P_t: [B*N, 4]
    adj: [B, N, N] or [N, N]
    """
    B_N = Q_t.shape[0]

    if adj.dim() == 2:
        N = adj.shape[1]
        B = B_N // N
        adj_batch = adj.unsqueeze(0).repeat(B, 1, 1)
    else:
        B = adj.shape[0]
        N = adj.shape[1]
        adj_batch = adj

    H, E, V = Q_t[:, 0], Q_t[:, 1], Q_t[:, 2]
    alpha, beta, gamma, eta = P_t[:, 0], P_t[:, 1], P_t[:, 2], P_t[:, 3]

    E_mat = E.view(B, N)
    E_coupled = torch.bmm(adj_batch, E_mat.unsqueeze(-1)).squeeze(-1)
    E_coupled = E_coupled.view(B * N)

    dH = -alpha * H
    dE = beta * H - gamma * E + 0.1 * E_coupled
    dV = eta * E + torch.randn_like(E) * noise_std

    H_new = H + dH
    E_new = E + dE
    V_new = V + dV

    return torch.stack([H_new, E_new, V_new], dim=1)

class AdjacencyLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc_q = nn.Linear(input_dim, hidden_dim)
        self.fc_k = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x: [B, C, T]
        q = self.fc_q(x)
        k = self.fc_k(x)
        scores = torch.matmul(q, k.transpose(1, 2))
        adj = self.softmax(scores)
        return adj

class CNN_1D(nn.Module):
    def __init__(self, window_size, hid_dim):
        super().__init__()
        down_factor = int(window_size / hid_dim)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=down_factor // 2, stride=down_factor // 2, groups=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=down_factor // 2, stride=down_factor // 2, groups=16, dilation=1)
        self.conv3 = nn.Conv1d(64, 1, kernel_size=1, stride=1, groups=1)

    def forward(self, x):
        h = torch.tanh(self.conv1(x))
        h = torch.tanh(self.conv2(h))
        h = torch.tanh(self.conv3(h))
        return h

class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features, out_features, n_heads,
                 is_concat=True, dropout=0.0, leaky_relu_negative_slope=0.2,
                 share_weights=False):
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

    def forward(self, h, adj_mat):
        batch_size, n_nodes = h.shape[0], h.shape[1]
        g_l = self.linear_l(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)

        g_l_repeat = g_l.repeat(1, n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=1)

        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(batch_size, n_nodes, n_nodes, self.n_heads, self.n_hidden)

        adj_mat_repeat = adj_mat.unsqueeze(-1)
        e = self.attn(self.activation(g_sum)).squeeze(-1)
        e = e.masked_fill(adj_mat_repeat == 0, float(-1e20))
        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum('cijh,cjhf->cihf', a, g_r)
        if self.is_concat:
            return a, attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        else:
            return a, attn_res.mean(dim=2)

class CNN_GAT(nn.Module):
    def __init__(self, window_size, horizon, enco_dim, hid_dim):
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        self.enco_dim = enco_dim
        self.hid_dim = hid_dim

        self.affine_weight = nn.Parameter(torch.ones(enco_dim))
        self.affine_bias = nn.Parameter(torch.zeros(enco_dim))

        self.path_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, enco_dim, kernel_size=1)
        )

        self.networks = nn.ModuleList([CNN_1D(window_size, hid_dim) for _ in range(enco_dim)])
        self.Gconv1 = GraphAttentionV2Layer(hid_dim, hid_dim, 1, is_concat=True)
        self.out = nn.Conv1d(enco_dim, enco_dim, kernel_size=hid_dim, stride=hid_dim, groups=enco_dim)

        self.causal_encoder = MechanicalCausalEncoder(3, hid_dim)
        self.causal_decoder = MechanicalCausalDecoder(hid_dim, 4)

        self.adj_net = AdjacencyLearner(input_dim=window_size, hidden_dim=32)

    def forward(self, x, edge_index=None, intervent=False, do=None):
        mean = torch.mean(x, dim=2, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False))
        x = (x - mean) / std

        x = self.path_encoder(x)
        x = x * self.affine_weight.view(1, -1, 1) + self.affine_bias.view(1, -1, 1)

        B = x.shape[0]
        Q_t = torch.tensor([1.0, 0.0, 0.0], device=x.device).repeat(B * self.enco_dim, 1)

        adj_soft = self.adj_net(x)           # [B, C, C]
        adj_bin = (adj_soft > 0.5).float()   # [B, C, C]

        if intervent:
            adj_bin_cf = adj_bin.clone()
            adj_bin_cf[:, do, :] = 0
            adj_bin_cf[:, :, do] = 0
            adj_bin = adj_bin_cf

        x_pred_all = None
        P_t = None

        for _ in range(self.horizon):
            h = torch.cat([self.networks[i](x[:, i:i + 1, :]) for i in range(self.enco_dim)], dim=1)
            _, h_temp = self.Gconv1(h, adj_bin)

            H_causal = self.causal_encoder(Q_t).view(B, self.enco_dim, -1)
            h = h + torch.tanh(h_temp) + H_causal

            x_pred = self.out(h)
            x = torch.cat((x[:, :, 1:], x_pred), dim=2)

            H_tilde = h
            P_t = self.causal_decoder(H_tilde.view(B * self.enco_dim, -1))
            Q_t = mechanical_dynamics(Q_t, P_t, adj_bin)

            x_pred_all = torch.cat((x_pred_all, x_pred), dim=2) if x_pred_all is not None else x_pred

        out = x_pred_all.permute(0, 2, 1)
        out = (out - self.affine_bias[None, None, :]) / self.affine_weight[None, None, :]
        out = out.permute(0, 2, 1)
        out = out * std + mean

        return out, Q_t.view(B, self.enco_dim, 3), P_t.view(B, self.enco_dim, 4), adj_soft


# =========================
# Load checkpoint
# =========================
assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
ckpt = torch.load(CKPT_PATH, map_location=device)

HID_DIM = ckpt["encoder_dim"]
WINDOW_SIZE = ckpt["window_size"]
HORIZON = ckpt["horizon"]

print(f"Loaded checkpoint from: {CKPT_PATH}")
print(f"HID_DIM={HID_DIM}, WINDOW_SIZE={WINDOW_SIZE}, HORIZON={HORIZON}")

# =========================
# Build model
# =========================
model = CNN_GAT(
    window_size=WINDOW_SIZE,
    horizon=HORIZON,
    enco_dim=ENCO_DIM,
    hid_dim=HID_DIM
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# =========================
# Prepare test data
# =========================
fault_dirs = glob.glob(os.path.join(DATA_PATH, "G*.mat"))
fault_dirs = sorted(fault_dirs)
assert len(fault_dirs) > 1, f"Not enough .mat files found in {DATA_PATH}"

a_measured = io.loadmat(fault_dirs[1])["a_measured"][0:1600000, 0]
contribution_td = io.loadmat(fault_dirs[1])["Contribution_TD"][:, 0:320000].T

x_all = a_measured[0:320000][::2].reshape(-1, 1)
x_all_o_test = a_measured[320000:640000][::2].reshape(-1, 1)

x_all, _, _ = minmax_norm(x_all, -1, 1)
x_all_o_test, x_min, x_max = minmax_norm(x_all_o_test, -1, 1)
x_all_1, _, _ = minmax_norm(contribution_td[0:160000], -1, 1)
x_all_t_test, path_min, path_max = minmax_norm(contribution_td[160000:], -1, 1)

x_all_t, x_all_ref_t = fastX(
    torch.from_numpy(x_all).float(),
    torch.from_numpy(x_all_1).float(),
    WINDOW_SIZE,
    HORIZON
)
x_test, x_test_ref = fastX(
    torch.from_numpy(x_all_o_test).float(),
    torch.from_numpy(x_all_t_test).float(),
    WINDOW_SIZE,
    HORIZON
)

test_dataset = torch.utils.data.TensorDataset(
    x_test.permute(0, 2, 1),
    x_test_ref.permute(0, 2, 1)
)
test_dl = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
    shuffle=False
)

# =========================
# Inference + timing
# =========================
all_preds = []
all_q = []
all_p = []
all_adj = []

num_batches = 0
num_windows = 0
total_infer_time_s = 0.0

with torch.no_grad():
    for x, _ in test_dl:
        x = x.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pred, qt_pred, pt_pred, adj_soft = model(x, None, intervent=False, do=None)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        batch_time = t1 - t0
        total_infer_time_s += batch_time
        num_batches += 1
        num_windows += x.size(0)

        all_preds.append(pred.cpu())
        all_q.append(qt_pred.cpu())
        all_p.append(pt_pred.cpu())
        all_adj.append(adj_soft.cpu())

# =========================
# Timing report
# =========================
total_infer_time_ms = total_infer_time_s * 1000.0
avg_batch_time_ms = total_infer_time_ms / max(num_batches, 1)
avg_window_time_ms = total_infer_time_ms / max(num_windows, 1)

print("=" * 60)
print(f"Total inference time (ms): {total_infer_time_ms:.6f}")
print(f"Average inference time per batch (ms): {avg_batch_time_ms:.6f}")
print(f"Average inference time per window (ms): {avg_window_time_ms:.6f}")
print(f"Number of batches: {num_batches}")
print(f"Number of windows: {num_windows}")
print("=" * 60)

# =========================
# Optional: save outputs
# =========================
all_preds = torch.cat(all_preds, dim=0)
all_q = torch.cat(all_q, dim=0)
all_p = torch.cat(all_p, dim=0)
all_adj = torch.cat(all_adj, dim=0).numpy()

reconstructed = reconstruct_signal_from_windows(
    all_preds,
    original_length=all_preds.shape[0],
    window_size=WINDOW_SIZE,
    horizon=HORIZON,
    crop="valid"
)
reconstructed_ins = inverse_minmax_norm_multi(reconstructed.cpu().numpy(), path_min, path_max)

sio.savemat("inference_results.mat", {
    "prediction": reconstructed_ins,
    "q_state": all_q.numpy(),
    "p_state": all_p.numpy(),
    "all_adj": all_adj,
    "total_inference_time_ms": np.array([[total_infer_time_ms]], dtype=np.float64),
    "avg_batch_time_ms": np.array([[avg_batch_time_ms]], dtype=np.float64),
    "avg_window_time_ms": np.array([[avg_window_time_ms]], dtype=np.float64),
})

print("Saved inference results to inference_results.mat")