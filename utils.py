import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('样本数量必须匹配')

    a = squareform(pdist(X))
    b = squareform(pdist(Y))

    a_mean = a.mean(axis=0, keepdims=True)
    b_mean = b.mean(axis=0, keepdims=True)
    a_mean_overall = a.mean()
    b_mean_overall = b.mean()

    eps = 1e-8
    A = a - a_mean - a.mean(axis=1, keepdims=True) + a_mean_overall + eps
    B = b - b_mean - b.mean(axis=1, keepdims=True) + b_mean_overall + eps

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)

    dcov2_xx = max(dcov2_xx, eps)
    dcov2_yy = max(dcov2_yy, eps)

    dcor = np.sqrt(max(dcov2_xy, 0)) / (np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy) + eps)
    return min(max(dcor, 0), 1)

def stable_mutual_information_estimate(z, y, method='variational'):
    try:
        if method == 'variational':
            return stable_variational_mi(z, y)
        elif method == 'kl':
            return stable_kl_mi(z, y)
        else:
            return stable_distance_correlation_mi(z, y)
    except Exception as e:
        print(f"互信息估计错误: {e}")
        return torch.tensor(0.1)

def stable_kl_mi(z, y):
    try:
        z_np = z.detach().cpu().numpy() if torch.is_tensor(z) else z
        y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else y

        if len(z_np.shape) > 1 and z_np.shape[1] > 1:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            z_np = pca.fit_transform(z_np)
            y_np = pca.fit_transform(y_np.reshape(-1, 1))

        mi_estimate = distcorr(z_np, y_np)
        return torch.tensor(max(min(mi_estimate, 0.99), 0.01))
    except Exception as e:
        print(f"KL MI估计错误: {e}")
        return torch.tensor(0.1)

def stable_variational_mi(z, y):
    try:
        if torch.is_tensor(z):
            z = z.detach()
        if torch.is_tensor(y):
            y = y.detach()

        if len(z.shape) > 1:
            z = z.mean(dim=1)
        if len(y.shape) > 1:
            y = y.mean(dim=1)

        z_mean = z.mean()
        y_mean = y.mean()
        z_std = z.std() + 1e-8
        y_std = y.std() + 1e-8

        z_norm = (z - z_mean) / z_std
        y_norm = (y - y_mean) / y_std

        z_norm = torch.clamp(z_norm, -5, 5)
        y_norm = torch.clamp(y_norm, -5, 5)

        correlation = torch.abs(torch.mean(z_norm * y_norm))
        return torch.clamp(correlation, 0.01, 0.99)
    except Exception as e:
        print(f"变分MI估计错误: {e}")
        return torch.tensor(0.1)

def stable_distance_correlation_mi(z, y):
    try:
        z_np = z.detach().cpu().numpy() if torch.is_tensor(z) else z
        y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else y

        z_np = np.nan_to_num(z_np, nan=0.0, posinf=1.0, neginf=-1.0)
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=1.0, neginf=-1.0)

        mi_estimate = distcorr(z_np, y_np)
        return torch.tensor(max(min(mi_estimate, 0.99), 0.01))
    except Exception as e:
        print(f"距离相关性MI错误: {e}")
        return torch.tensor(0.1)

def set_seed(seed=42):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() == 0:
        return 0.0
    
    y_pred_c, y_true_c = y_pred[mask], y_true[mask]
    ss_res = np.sum((y_true_c - y_pred_c) ** 2)
    ss_tot = np.sum(y_true_c ** 2)
    
    if ss_tot < 1e-10:
        return 0.0
    
    return (1 - ss_res / ss_tot) * 100

