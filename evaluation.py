import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from losses import stable_kl_divergence


def compute_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    pred_flat = predicted.flatten()
    actual_flat = actual.flatten()
    mask = np.isfinite(pred_flat) & np.isfinite(actual_flat)
    if mask.sum() < 2:
        return 0.0
    pred_clean = pred_flat[mask]
    actual_clean = actual_flat[mask]
    correlation = np.corrcoef(pred_clean, actual_clean)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def compute_rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    pred_flat = predicted.flatten()
    actual_flat = actual.flatten()
    mask = np.isfinite(pred_flat) & np.isfinite(actual_flat)
    if mask.sum() < 2:
        return 0.0
    pred_clean = pred_flat[mask]
    actual_clean = actual_flat[mask]
    pred_rank = pd.Series(pred_clean).rank()
    actual_rank = pd.Series(actual_clean).rank()
    correlation = np.corrcoef(pred_rank, actual_rank)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def compute_icir(predicted: np.ndarray, actual: np.ndarray) -> float:
    T, N = predicted.shape
    ic_series = []
    for t in range(T):
        pred_t = predicted[t]
        actual_t = actual[t]
        mask = np.isfinite(pred_t) & np.isfinite(actual_t)
        if mask.sum() < 2:
            continue
        pred_clean = pred_t[mask]
        actual_clean = actual_t[mask]
        correlation = np.corrcoef(pred_clean, actual_clean)[0, 1]
        if not np.isnan(correlation):
            ic_series.append(correlation)
    if len(ic_series) == 0:
        return 0.0
    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    return ic_mean / ic_std if ic_std > 1e-10 else 0.0


def compute_r2_total(mu_total: np.ndarray, returns_true: np.ndarray) -> float:
    mu_flat = mu_total.flatten()
    returns_flat = returns_true.flatten()

    mask = np.isfinite(mu_flat) & np.isfinite(returns_flat)
    if mask.sum() == 0:
        return 0.0
    mu_clean = mu_flat[mask]
    returns_clean = returns_flat[mask]
    ss_res = np.sum((returns_clean - mu_clean) ** 2)
    ss_tot = np.sum(returns_clean ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return (1 - ss_res / ss_tot) * 100


def compute_r2_pred(mu_pred: np.ndarray, returns_true: np.ndarray) -> float:
    mu_flat = mu_pred.flatten()
    returns_flat = returns_true.flatten()

    mask = np.isfinite(mu_flat) & np.isfinite(returns_flat)
    if mask.sum() == 0:
        return 0.0
    mu_clean = mu_flat[mask]
    returns_clean = returns_flat[mask]
    ss_res = np.sum((returns_clean - mu_clean) ** 2)
    ss_tot = np.sum(returns_clean ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return (1 - ss_res / ss_tot) * 100


def evaluate_stable_model(model, test_loader, config):
    model.eval()
    device = config.device
    total_loss = 0.0
    n_samples = 0

    is_diffusion = hasattr(model, 'diffusion')

    with torch.no_grad():
        for batch_data in test_loader:
            try:
                if len(batch_data) == 6:
                    x_batch, y_batch, r_hist_batch, x_t_prev, r_t_prev, r_hist_t_prev = batch_data
                else:
                    x_batch, y_batch, r_hist_batch = batch_data
                    x_t_prev, r_t_prev, r_hist_t_prev = None, None, None

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                r_hist_batch = r_hist_batch.to(device)
                x_flat = x_batch[:, -1, :]
                x_flat = torch.clamp(x_flat, -10, 10)

                if is_diffusion:
                    f_prev = None
                    if x_t_prev is not None:
                        x_t_prev = x_t_prev.to(device)
                        outputs_prev = model(x_t_prev, r_t=r_t_prev.to(device) if r_t_prev is not None else None,
                                             mode='inference')
                        f_prev = outputs_prev['f_t']

                    outputs = model(x_batch, r_t=y_batch, f_prev=f_prev, mode='train')
                    batch_loss = outputs['diffusion_loss'].item()
                else:
                    recon, mu, logvar = model(r=r_hist_batch, x=x_flat, sample_z=False)
                    recon_loss = F.mse_loss(recon, x_flat)
                    kl_loss = stable_kl_divergence(mu, logvar)
                    batch_loss = (recon_loss + kl_loss).item()

                total_loss += batch_loss * x_batch.size(0)
                n_samples += x_batch.size(0)

            except Exception as e:
                print(f"Batch evaluation error: {e}")
                continue

    return total_loss / n_samples if n_samples > 0 else float('inf')


@torch.no_grad()
def test_model(model, test_loader, config):
    model.eval()
    device = config.device

    all_predictions_total = []
    all_predictions_pred = []
    all_targets = []
    all_time_indices = []
    all_stock_indices = []

    historical_ft_dict = {}

    is_diffusion = hasattr(model, 'diffusion')

    print("\nStarting testing...")
    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
        try:
            if len(batch_data) == 6:
                x_batch, y_batch, r_hist_batch, x_t_prev, r_t_prev, r_hist_t_prev = batch_data
            else:
                x_batch, y_batch, r_hist_batch = batch_data
                x_t_prev, r_t_prev, r_hist_t_prev = None, None, None

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            r_hist_batch = r_hist_batch.to(device)
            x_flat = x_batch[:, -1, :]

            batch_size = x_batch.size(0)

            dataset = test_loader.dataset
            if hasattr(dataset, 'time_indices') and dataset.time_indices is not None:
                start_idx = batch_idx * test_loader.batch_size
                end_idx = min(start_idx + batch_size, len(dataset))
                batch_time_indices = dataset.time_indices[start_idx:end_idx]
                batch_stock_indices = dataset.stock_indices[start_idx:end_idx] if hasattr(dataset,
                                                                                          'stock_indices') and dataset.stock_indices is not None else None
            else:
                batch_time_indices = [batch_idx] * batch_size
                batch_stock_indices = list(range(batch_size))

            if is_diffusion:
                model.reset_f_prev_buffer()
                predictions_total, _ = model.predict(x_batch, r_t=r_hist_batch)
                model.reset_f_prev_buffer()
            else:
                predictions_total = model.predict(x_batch, x_flat, y=None, r=r_hist_batch)

            predictions_pred = []

            for i in range(batch_size):
                time_idx = batch_time_indices[i] if i < len(batch_time_indices) else batch_idx

                x_seq_i = x_batch[i:i + 1]
                x_flat_i = x_flat[i:i + 1]
                r_hist_i = r_hist_batch[i:i + 1]

                if is_diffusion:
                    model.reset_f_prev_buffer()
                    f_prev = None
                    outputs = model(x_seq_i, r_t=r_hist_i, f_prev=f_prev, mode='inference')
                    ft_curr = outputs['f_t'][0].cpu().numpy()
                    model.reset_f_prev_buffer()
                else:
                    recon_ft_curr, _, _ = model.forward(r=r_hist_i, x=x_flat_i, sample_z=False)
                    ft_curr = recon_ft_curr[0].cpu().numpy()

                if time_idx not in historical_ft_dict:
                    historical_ft_dict[time_idx] = []
                historical_ft_dict[time_idx].append(ft_curr)

                hist_ft_list = []
                for prev_time in sorted(historical_ft_dict.keys()):
                    if prev_time < time_idx:
                        hist_ft_list.extend(historical_ft_dict[prev_time])

                if len(hist_ft_list) > 10:
                    hist_ft_list = hist_ft_list[-10:]

                if len(hist_ft_list) > 0:
                    ft_bar = np.mean(hist_ft_list, axis=0)
                else:
                    ft_bar = ft_curr

                z = model.get_z_from_x(x_seq_i)
                z_for_beta = z.unsqueeze(1)
                beta_z_factor = model.beta_network(z_for_beta)
                beta_z_factor = beta_z_factor.squeeze(1)

                if is_diffusion:
                    if ft_bar.shape[0] == model.factor_dim:
                        ft_bar_tensor = torch.FloatTensor(ft_bar).unsqueeze(0).to(device)
                        q_pred = (beta_z_factor * ft_bar_tensor).sum(dim=1)
                    else:
                        beta_z = model.beta_projection(beta_z_factor)
                        ft_bar_tensor = torch.FloatTensor(ft_bar).unsqueeze(0).to(device)
                        q_pred = (beta_z * ft_bar_tensor).sum(dim=1)
                else:
                    beta_z = model.beta_projection(beta_z_factor)
                    ft_bar_tensor = torch.FloatTensor(ft_bar).unsqueeze(0).to(device)
                    q_pred = (beta_z * ft_bar_tensor).sum(dim=1)

                predictions_pred.append(q_pred.item())

            pred_total_np = predictions_total.cpu().numpy().flatten()
            pred_pred_np = np.array(predictions_pred)
            actual_np = y_batch.cpu().numpy().flatten()

            all_predictions_total.extend(pred_total_np)
            all_predictions_pred.extend(pred_pred_np)
            all_targets.extend(actual_np)
            all_time_indices.extend(batch_time_indices[:batch_size])
            if batch_stock_indices is not None:
                all_stock_indices.extend(batch_stock_indices[:batch_size])

        except Exception as e:
            print(f"Test batch {batch_idx} error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_predictions_total) == 0:
        print("Error: No predictions generated")
        return None

    all_predictions_total = np.array(all_predictions_total)
    all_predictions_pred = np.array(all_predictions_pred)
    all_targets = np.array(all_targets)
    all_time_indices = np.array(all_time_indices)

    if len(all_time_indices) > 0 and len(np.unique(all_time_indices)) > 0:
        unique_times = np.unique(all_time_indices)
        T = len(unique_times)

        max_stocks_per_time = 0
        for t in unique_times:
            mask = all_time_indices == t
            n_stocks = mask.sum()
            max_stocks_per_time = max(max_stocks_per_time, n_stocks)

        if max_stocks_per_time < 2:
            print("Warning: Only 1 stock per time point, cannot correctly calculate IC, ICIR, RankIC.")
            print("      These metrics require multiple stocks per time point (N>1) to calculate.")
            print("      Current data may be shuffled, or data loading caused time information loss.")
            pred_2d = all_predictions_total.reshape(1, -1)
            actual_2d = all_targets.reshape(1, -1)
            ic = compute_ic(pred_2d, actual_2d)
            rank_ic = compute_rank_ic(pred_2d, actual_2d)
            icir = 0.0
        else:
            pred_by_time = []
            actual_by_time = []

            for t in unique_times:
                mask = all_time_indices == t
                pred_t = all_predictions_total[mask]
                actual_t = all_targets[mask]

                if len(pred_t) < max_stocks_per_time:
                    padding = np.full(max_stocks_per_time - len(pred_t), np.nan)
                    pred_t = np.concatenate([pred_t, padding])
                    actual_t = np.concatenate([actual_t, padding])

                pred_by_time.append(pred_t)
                actual_by_time.append(actual_t)

            pred_2d = np.array(pred_by_time)
            actual_2d = np.array(actual_by_time)

            ic = compute_ic(pred_2d, actual_2d)
            icir = compute_icir(pred_2d, actual_2d)
            rank_ic = compute_rank_ic(pred_2d, actual_2d)
    else:
        print("Warning: No time index information, cannot organize data by time.")
        print("      Calculation of IC, ICIR, RankIC may be inaccurate.")
        pred_2d = all_predictions_total.reshape(1, -1)
        actual_2d = all_targets.reshape(1, -1)
        ic = compute_ic(pred_2d, actual_2d)
        rank_ic = compute_rank_ic(pred_2d, actual_2d)
        icir = 0.0

    r2_total = compute_r2_total(all_predictions_total, all_targets)
    r2_pred = compute_r2_pred(all_predictions_pred, all_targets)

    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    print(f"IC:       {ic:.6f}")
    print(f"ICIR:     {icir:.6f}")
    print(f"RankIC:   {rank_ic:.6f}")
    print(f"R²_total: {r2_total:.6f}%")
    print(f"R²_pred:  {r2_pred:.6f}%")
    print("=" * 50)

    return {
        'ic': ic,
        'icir': icir,
        'rank_ic': rank_ic,
        'r2_total': r2_total,
        'r2_pred': r2_pred
    }