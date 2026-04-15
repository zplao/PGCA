# interpretability_card.py
# -*- coding: utf-8 -*-
"""
Interpretability Card utility.
Provides metric computations and visualization for the LaTeX-specified metrics:
EC, PDC, BC, CDE, IM, INV, SPS, MSS, DU.
Works with numpy arrays or torch tensors (auto-converts).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from math import ceil
from scipy.stats import spearmanr
import scipy.io as sio


try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# -----------------------
# Helpers: conversion
# -----------------------
def _to_np(x):
    """Convert numpy or torch tensor to numpy 1d array (float)."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def ec_scale_inv(x):
    # x: scalar or numpy array, EC_raw >=0
    return x / (1.0 + x)


# -----------------------
# Metric functions
# -----------------------
def energy_closure(y, y_ps):
    """EC = sum_p ||y_p||^2 / ||y||^2"""
    y = _to_np(y).ravel()
    y_ps = [_to_np(p).ravel() for p in y_ps]
    num = sum(np.linalg.norm(p)**2 for p in y_ps)
    den = np.linalg.norm(y)**2 + 1e-12
    return float(ec_scale_inv(num / den))

def pdc_per_path(y_ps, yhat_ps, group_delay_errors, alpha=1.0):
    """
    Compute PDC per path.
    PDC_p = 0.5*(Coh(y_p, yhat_p) + exp(-alpha*|Delta_tau|))
    Coh approximated by squared Pearson correlation (proxy for mag-sq coherence).
    """
    pdc = []
    for p, phat, dt in zip(y_ps, yhat_ps, group_delay_errors):
        p = _to_np(p).ravel(); phat = _to_np(phat).ravel()
        # plt.plot(p)
        # plt.plot(phat)
        # plt.show()
        if p.std() == 0 or phat.std() == 0:
            coh = 0.0
        else:
            coh = np.corrcoef(p, phat)[0,1]**2
            coh = float(np.clip(coh, 0.0, 1.0))
        pdc_p = 0.5 * (coh + np.exp(-alpha * abs(float(dt))))
        pdc.append(float(pdc_p))
    return np.array(pdc)

def bandwise_consistency(Sy, Sphys, Omega_idx):
    """
    BC = sum_{omega in Omega} min(Sy, Sphys) / sum Sphys over Omega
    Inputs are 1D spectra; Omega_idx are indices (iterable).
    """
    Sy = _to_np(Sy)
    Sphys = _to_np(Sphys)
    Omega_idx = np.asarray(Omega_idx, dtype=int)
    mins = np.minimum(Sy[Omega_idx], Sphys[Omega_idx])
    den = Sphys[Omega_idx].sum() + 1e-12
    return float(mins.sum() / den)

def counterfactual_deletion_curve(model_loss_pre, model_loss_post_list):
    """
    Given baseline loss scalar and list of post-deletion losses (ordered by deletion),
    returns relative changes: (post - pre)/pre
    """
    pre = float(model_loss_pre)
    posts = _to_np(model_loss_post_list).astype(float)
    rel = (posts - pre) / (pre + 1e-12)
    return rel

def intervention_monotonicity(ms, ss):
    """Spearman correlation between magnitudes m and scores s, rescaled to [0,1] by (rho+1)/2"""
    ms = _to_np(ms).ravel(); ss = _to_np(ss).ravel()
    if len(ms) < 2:
        return 0.5
    rho, _ = spearmanr(ms, ss)
    if np.isnan(rho):
        rho = 0.0
    return float((rho + 1.0) / 2.0)

def jaccard(set_a, set_b):
    a = set(set_a); b = set(set_b)
    if len(a | b) == 0:
        return 1.0
    return float(len(a & b) / len(a | b))

def invariance_metric(edge_sets_by_domain):
    """Average pairwise Jaccard over all domain pairs."""
    domains = sorted(edge_sets_by_domain.keys())
    pairs = list(combinations(domains, 2))
    if len(pairs) == 0:
        return 1.0
    vals = [jaccard(edge_sets_by_domain[d1], edge_sets_by_domain[d2]) for d1,d2 in pairs]
    return float(np.mean(vals))

def sparsity_and_mss(A_all_edges, kept_edges_mask, edge_contributions, eta=0.8):
    """
    SPS = 1 - |E_kept|/|E_all|
    MSS: greedy accumulate edges sorted by contribution until accumulated contribution >= eta * total_contribution.
    returns (SPS, MSS, kept_count, mstar)
    MSS reported as 1 - m*/|E_all|
    """
    total_edges = len(A_all_edges)
    kept_count = int(np.sum(_to_np(kept_edges_mask)))
    sps = 1.0 - (kept_count / (total_edges + 1e-12))

    contrib = _to_np(edge_contributions).astype(float)
    total = contrib.sum() + 1e-12
    target = eta * total
    sorted_idx = np.argsort(contrib)[::-1]
    cumsum = 0.0
    mstar = 0
    for idx in sorted_idx:
        cumsum += contrib[idx]
        mstar += 1
        if cumsum >= target:
            break
    mstar = min(mstar, total_edges)
    mss = 1.0 - (mstar / (total_edges + 1e-12))
    return float(sps), float(mss), int(kept_count), int(mstar)

def decision_utility(cost_baseline, cost_with_expl):
    base = float(np.mean(_to_np(cost_baseline)))
    with_expl = float(np.mean(_to_np(cost_with_expl)))
    if base == 0:
        return 0.0
    return float((base - with_expl) / (base + 1e-12))

# -----------------------
# Visualization
# -----------------------
def visualize_interpretability_card(inputs, show=True, save_prefix=None):
    """
    inputs: dict with keys (recommended):
      y, y_ps, yhat_ps, group_delay_errors, Sy, Sphys, Omega_idx,
      model_loss_pre, model_loss_post_list, m_list, s_list,
      edge_sets_by_domain, A_all_edges, kept_edges_mask, edge_contributions,
      cost_baseline, cost_with_expl, Ts

    Returns: (df, metrics_dict, extras_dict)
    Saves figures with save_prefix if provided.
    """
    # Fallback defaults
    y = inputs.get('y', np.zeros(1024))
    y_ps = inputs.get('y_ps', [np.zeros_like(y)])
    yhat_ps = inputs.get('yhat_ps', y_ps)
    gdel = inputs.get('group_delay_errors', [0.0]*len(y_ps))
    Sy = inputs.get('Sy', np.abs(np.fft.rfft(_to_np(y)))**2)
    Sphys = inputs.get('Sphys', Sy.copy())
    Omega_idx = inputs.get('Omega_idx', np.arange(min(len(Sy), len(Sphys))))
    model_loss_pre = inputs.get('model_loss_pre', 1.0)
    model_loss_post_list = inputs.get('model_loss_post_list', np.linspace(1.0,1.5,len(y_ps)))
    m_list = inputs.get('m_list', np.linspace(0.1,1.0,10))
    s_list = inputs.get('s_list', np.linspace(0.1,1.0,10) + np.random.randn(10)*0.05)
    # edge_sets_by_domain = inputs.get('edge_sets_by_domain', {'d0':[0,1,2], 'd1':[0,2,3]})
    A_all_edges = inputs.get('A_all_edges', list(range(20)))
    kept_edges_mask = inputs.get('kept_edges_mask', np.array([i<5 for i in A_all_edges]))
    edge_contributions = inputs.get('edge_contributions', np.abs(np.random.rand(len(A_all_edges))))
    # cost_baseline = inputs.get('cost_baseline', np.random.rand(100)*100.0 + 500)
    # cost_with_expl = inputs.get('cost_with_expl', cost_baseline * (1 - 0.12))
    Ts = inputs.get('Ts', 32000)

    # Compute metrics
    EC = energy_closure(y, y_ps)
    pdc_arr = pdc_per_path(y_ps, yhat_ps, gdel, alpha=1.0 / (Ts + 1e-12))
    PDC = float(np.mean(pdc_arr))
    BC = bandwise_consistency(Sy, Sphys, Omega_idx)
    # cde_rel = counterfactual_deletion_curve(model_loss_pre, model_loss_post_list)
    # CDE_summary = float(np.mean(cde_rel))
    # IM = intervention_monotonicity(m_list, s_list)
    # # INV = invariance_metric(edge_sets_by_domain)
    # SPS, MSS, kept_count, mstar = sparsity_and_mss(A_all_edges, kept_edges_mask, edge_contributions, eta=0.8)
    # DU = decision_utility(cost_baseline, cost_with_expl)

    metrics = {
        'Energy Closure (EC)': EC,
        'PDC (avg)': PDC,
        'Bandwise Consistency (BC)': BC,
        # 'CDE (avg rel loss)': np.clip(CDE_summary, 0.0, 1.0),
        # 'Intervention Monotonicity (IM)': IM,
        # # 'Invariance (INV)': INV,
        # 'Sparsity (SPS)': SPS,
        # 'Minimal Sufficient Set (MSS)': MSS,
        # 'Decision Utility (DU)': DU
    }
    # for k in list(metrics.keys()):
    #     metrics[k] = float(np.clip(metrics[k], 0.0, 1.0))

    df = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])

    # Print or display table
    # try:
    #     from caas_jupyter_tools import display_dataframe_to_user
    #     display_dataframe_to_user("Interpretability Metrics", df)
    # except Exception:
    #     print(df)

    # ---------- Plots (each figure single plot) ----------
    # 1) Overview bar chart
    plt.figure(figsize=(10,5))
    names = df.index.tolist()
    vals = df['value'].values
    y_pos = np.arange(len(names))
    plt.barh(y_pos, vals)
    plt.xlim(0,1)
    plt.yticks(y_pos, names)
    plt.xlabel('Scaled metric value [0,1]')
    plt.title('Interpretability Card — Metrics Overview')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_metrics_overview.png", dpi=150)
    if show:
        plt.show()
    plt.close()

    # 2) PDC per path
    plt.figure(figsize=(7,3))
    idx = np.arange(len(pdc_arr))
    plt.bar(idx, pdc_arr)
    plt.ylim(0,1)
    plt.xlabel('Path index')
    plt.ylabel('PDC per-path')
    plt.title('Phase/Delay Consistency (per key path)')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_pdc_per_path.png", dpi=150)
    if show:
        plt.show()
    plt.close()

    # 3) Energy fractions
    energies = np.array([np.linalg.norm(_to_np(p))**2 for p in y_ps])
    total_energy = energies.sum() + 1e-12
    rel = energies / total_energy
    plt.figure(figsize=(7,3))
    plt.bar(np.arange(len(rel)), rel)
    plt.ylim(0,1)
    plt.xlabel('Path index')
    plt.ylabel('Fraction of reconstructed energy')
    plt.title('Energy Closure — path energy fractions')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_energy_fractions.png", dpi=150)
    if show:
        plt.show()
    plt.close()

    # 4) CDE curve
    # plt.figure(figsize=(7,3))
    # x = np.arange(len(cde_rel)) + 1
    # plt.plot(x, cde_rel, marker='o')
    # plt.xlabel('Deletion step (increasing edges removed)')
    # plt.ylabel('Relative change in loss (post-pre)/pre')
    # plt.title('Counterfactual Deletion Curve (CDE)')
    # plt.axhline(0, linestyle='--')
    # plt.tight_layout()
    # if save_prefix:
    #     plt.savefig(f"{save_prefix}_cde_curve.png", dpi=150)
    # if show:
    #     plt.show()
    # plt.close()

    # # 5) Invariance matrix
    # domains = sorted(edge_sets_by_domain.keys())
    # n = len(domains)
    # mat = None
    # if n > 1:
    #     mat = np.zeros((n,n))
    #     for i,d1 in enumerate(domains):
    #         for j,d2 in enumerate(domains):
    #             mat[i,j] = jaccard(edge_sets_by_domain[d1], edge_sets_by_domain[d2])
    #     plt.figure(figsize=(4,4))
    #     plt.imshow(mat, aspect='equal')
    #     plt.colorbar(label='Jaccard')
    #     plt.xticks(np.arange(n), domains, rotation=45)
    #     plt.yticks(np.arange(n), domains)
    #     plt.title('Pairwise Jaccard of Top-k Edge Sets (Invariance)')
    #     plt.tight_layout()
    #     if save_prefix:
    #         plt.savefig(f"{save_prefix}_invariance_matrix.png", dpi=150)
    #     if show:
    #         plt.show()
    #     plt.close()

    # 6) Sparsity / MSS sizes
    # plt.figure(figsize=(6,3))
    # labels = ['Kept edges', 'm* (MSS)']
    # values = [kept_count, mstar]
    # plt.bar(labels, values)
    # plt.ylabel('Number of edges')
    # plt.title('Sparsity and Minimal Sufficient Set sizes')
    # plt.tight_layout()
    # if save_prefix:
    #     plt.savefig(f"{save_prefix}_sparsity_mss.png", dpi=150)
    # if show:
    #     plt.show()
    # plt.close()

    # # 7) Decision utility boxplots
    # plt.figure(figsize=(6,3))
    # plt.boxplot([_to_np(cost_baseline), _to_np(cost_with_expl)], labels=['Baseline','With explanation'])
    # plt.ylabel('Cost')
    # plt.title('Decision Utility — cost distributions')
    # plt.tight_layout()
    # if save_prefix:
    #     plt.savefig(f"{save_prefix}_decision_utility.png", dpi=150)
    # if show:
    #     plt.show()
    # plt.close()

    extras = {
        'pdc_per_path': pdc_arr,
        # 'cde_curve': cde_rel,
        'energy_fractions': rel
        # 'invariance_matrix': mat
    }

    return df, metrics, extras

# -----------------------
# Example / demo main
# -----------------------
def main_demo():
    """Demo using synthetic data. Replace these arrays with your real outputs."""
    np.random.seed(0)
    N = 2048
    y = np.random.randn(N)
    # make synthetic path contributions (6 paths)
    y_ps = [np.random.randn(N)*f for f in [0.8, 0.4, 0.2, 0.1, 0.05, 0.02]]
    yhat_ps = [p + 0.05*np.random.randn(N) for p in y_ps]
    gdel = np.random.randn(len(y_ps))*0.001
    Sy = np.abs(np.fft.rfft(y))**2
    Sphys = Sy * (1 + 0.2*np.random.randn(len(Sy)))
    Omega_idx = np.array([10, 22, 34, 56, 78])

    inputs = {
        'y': y,
        'y_ps': y_ps,
        'yhat_ps': yhat_ps,
        'group_delay_errors': gdel,
        'Sy': Sy,
        'Sphys': np.abs(Sphys),
        'Omega_idx': Omega_idx,
        'model_loss_pre': 0.8,
        'model_loss_post_list': np.linspace(0.8, 1.4, 12),
        'm_list': np.linspace(0.1, 1.0, 12),
        's_list': np.linspace(0.05, 0.95, 12) + 0.05*np.random.randn(12),
        'edge_sets_by_domain': {'low_speed':[0,1,2,3], 'high_speed':[0,2,3,4], 'high_load':[0,1,3,5]},
        'A_all_edges': list(range(20)),
        'kept_edges_mask': np.array([i<6 for i in range(20)]),
        'edge_contributions': np.abs(np.random.rand(20)),
        'cost_baseline': np.random.rand(200)*200 + 1000,
        'cost_with_expl': (np.random.rand(200)*200 + 1000) * 0.88,
        'Ts': 1/16000
    }

    # Run visualization and save outputs with prefix
    df, metrics, extras = visualize_interpretability_card(inputs, show=True, save_prefix='interpret_card_demo')
    print("\nMetrics:\n", df)

def main():
    # 加载数据
    deta = 3000
    # data_path = "D:\华工\博士阶段\博一\下学期\因果信号分解小论文\图\反演性能评估\所提方法-输出轴1.mat"
    # data_path1 = "D:\华工\博士阶段\博一\下学期\因果信号分解小论文\图\反演性能评估\BiLSTM-输出轴.mat"
    data_path = "D:\华工\博士阶段\博一\下学期\因果信号分解小论文\图\反演性能评估\所提方法-中间轴.mat"
    data_path1 = "D:\Projects\CAL-decoup\可解释指标对比方法\中间轴磨损-无FEV/restored_prediction.mat"
    y = sio.loadmat(data_path)["a_measured"]
    y = y[deta + 320000:deta + 320000+40960:2,:].squeeze()
    # 归一化
    y_min = y.min(axis=0)
    y_max = y.max(axis=0)
    y_norm = (y - y_min) / (y_max - y_min + 1e-12)


    y_ps_t = sio.loadmat(data_path1)["prediction"][deta + 2048:deta + 2048+20480,:]
    # 归一化
    y_ps_t_min = y_ps_t.min(axis=0)
    y_ps_t_max = y_ps_t.max(axis=0)
    y_ps_t_norm = (y_ps_t - y_ps_t_min) / (y_ps_t_max - y_ps_t_min + 1e-12)
    y_ps = [p for p in y_ps_t_norm.T]

    yhat_ps_t = sio.loadmat(data_path)["Contribution_TD"][:, deta + 160000:deta + 160000+20480]
    # 归一化
    yhat_ps_t_min = yhat_ps_t.min(axis=1)
    yhat_ps_t_max = yhat_ps_t.max(axis=1)
    yhat_ps_t_norm = (yhat_ps_t.T - yhat_ps_t_min) / (yhat_ps_t_max - yhat_ps_t_min + 1e-12)
    yhat_ps = [p for p in yhat_ps_t_norm.T]

    gdel = np.random.randn(len(y_ps))*(1/32000)
    Sy = np.abs(np.fft.rfft(np.sum(y_ps_t,1))) ** 2
    y_phys = sio.loadmat(data_path)["A_reconstructed"][:, deta + 2048:deta + 2048+20480].squeeze()
    Sphys = np.abs(np.fft.rfft(y_phys)) ** 2
    Omega_idx = np.array([14, 20, 462, 476, 490, 540, 560, 580, 938, 952, 966, 1100, 1120, 1140, 1428, 1680])
                          # 1904, 2240, 2380, 2800, 2856, 3332, 3360, 3808, 3920, 4284, 4480, 4760, 5040, 5600])


    np.random.seed(0)
    N = 2048
    # y = np.random.randn(N)
    # # make synthetic path contributions (6 paths)
    # y_ps = [np.random.randn(N)*f for f in [0.8, 0.4, 0.2, 0.1, 0.05, 0.02]]
    # yhat_ps = [p + 0.05*np.random.randn(N) for p in y_ps]
    # gdel = np.random.randn(len(y_ps))*0.001
    # Sy = np.abs(np.fft.rfft(y))**2
    # Sphys = Sy * (1 + 0.2*np.random.randn(len(Sy)))
    # Omega_idx = np.array([10, 22, 34, 56, 78])

    inputs = {
        'y': y_norm,
        'y_ps': y_ps,
        'yhat_ps': yhat_ps,
        'group_delay_errors': gdel,
        'Sy': Sy,
        'Sphys': np.abs(Sphys),
        'Omega_idx': Omega_idx,
        # 'model_loss_pre': 0.8,
        # 'model_loss_post_list': np.linspace(0.8, 1.4, 12),
        # 'm_list': np.linspace(0.1, 1.0, 12),
        # 's_list': np.linspace(0.05, 0.95, 12) + 0.05*np.random.randn(12),
        # 'edge_sets_by_domain': {'low_speed':[0,1,2,3], 'high_speed':[0,2,3,4], 'high_load':[0,1,3,5]},
        # 'A_all_edges': list(range(20)),
        # 'kept_edges_mask': np.array([i<6 for i in range(20)]),
        # 'edge_contributions': np.abs(np.random.rand(20)),
        # 'cost_baseline': np.random.rand(200)*200 + 1000,
        # 'cost_with_expl': (np.random.rand(200)*200 + 1000) * 0.88,
        # 'Ts': 1/16000
    }

    # Run visualization and save outputs with prefix
    df, metrics, extras = visualize_interpretability_card(inputs, show=True, save_prefix='interpret_card_demo')
    print("\nMetrics:\n", df)

if __name__ == "__main__":
    main()
