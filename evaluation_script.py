import torch
import numpy as np
import pandas as pd
import os
import sys
import json
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import StableConfig
from data_preprocessing import StableDataPreprocessor, StableStockDataset
from models import StableCaRIVAE, StableCaRIVAEWithDiffusion
from evaluation import test_model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        else:
            return super().default(obj)

def compute_daily_metrics(predicted_total, predicted_pred, actual_returns, time_indices=None):
    num_samples = len(predicted_total)

    if time_indices is None:
        time_indices = np.zeros(num_samples)

    unique_times = np.unique(time_indices)

    daily_metrics = {
        'time_index': [],
        'ic_total': [],
        'rank_ic_total': [],
        'r2_total': [],
        'r2_pred': [],
        'num_samples': []
    }

    for t in tqdm(unique_times, desc="Computing daily metrics", leave=False):
        mask = time_indices == t
        if mask.sum() < 2:
            continue

        pred_total_t = predicted_total[mask]
        pred_pred_t = predicted_pred[mask]
        actual_t = actual_returns[mask]

        ic_t = compute_ic(pred_total_t, actual_t)
        rank_ic_t = compute_rank_ic(pred_total_t, actual_t)
        r2_total_t = compute_r2(pred_total_t, actual_t)
        r2_pred_t = compute_r2(pred_pred_t, actual_t)

        daily_metrics['time_index'].append(t)
        daily_metrics['ic_total'].append(ic_t)
        daily_metrics['rank_ic_total'].append(rank_ic_t)
        daily_metrics['r2_total'].append(r2_total_t)
        daily_metrics['r2_pred'].append(r2_pred_t)
        daily_metrics['num_samples'].append(mask.sum())

    return daily_metrics

def compute_aggregate_metrics(daily_metrics):
    metrics = ['ic_total', 'rank_ic_total', 'r2_total', 'r2_pred']
    aggregate_metrics = {}

    for metric in tqdm(metrics, desc="Computing aggregate metrics", leave=False):
        values = np.array(daily_metrics[metric])

        valid_mask = np.isfinite(values)
        if valid_mask.sum() == 0:
            aggregate_metrics[metric] = {
                'mean': 0.0,
                'std': 0.0,
                'worst': 0.0,
                'num_valid': 0
            }
            continue

        valid_values = values[valid_mask]

        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)

        if metric.startswith('r2'):
            worst_val = np.min(valid_values)
        else:
            worst_val = np.min(valid_values)

        aggregate_metrics[metric] = {
            'mean': mean_val,
            'std': std_val,
            'worst': worst_val,
            'num_valid': len(valid_values)
        }

    return aggregate_metrics

def compute_ood_drop(train_metrics, test_metrics):
    ood_drop = {}
    metrics = ['ic_total', 'rank_ic_total', 'r2_total', 'r2_pred']

    for metric in tqdm(metrics, desc="Computing OOD drop", leave=False):
        train_mean = train_metrics[metric]['mean']
        test_mean = test_metrics[metric]['mean']

        if abs(train_mean) < 1e-10:
            ood_drop[metric] = 0.0 if abs(test_mean - train_mean) < 1e-10 else 1.0
        else:
            ood_drop[metric] = abs(test_mean - train_mean) / abs(train_mean)

    return ood_drop

def compute_ic(predicted, actual):
    mask = np.isfinite(predicted) & np.isfinite(actual)
    if mask.sum() < 2:
        return 0.0

    pred_clean = predicted[mask]
    actual_clean = actual[mask]

    correlation = np.corrcoef(pred_clean, actual_clean)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0

def compute_rank_ic(predicted, actual):
    mask = np.isfinite(predicted) & np.isfinite(actual)
    if mask.sum() < 2:
        return 0.0

    pred_clean = predicted[mask]
    actual_clean = actual[mask]

    pred_rank = pd.Series(pred_clean).rank()
    actual_rank = pd.Series(actual_clean).rank()

    correlation = np.corrcoef(pred_rank, actual_rank)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0

def compute_r2(predicted, actual):
    mask = np.isfinite(predicted) & np.isfinite(actual)
    if mask.sum() == 0:
        return 0.0

    pred_clean = predicted[mask]
    actual_clean = actual[mask]

    ss_res = np.sum((actual_clean - pred_clean) ** 2)
    ss_tot = np.sum(actual_clean ** 2)

    if ss_tot < 1e-10:
        return 0.0

    r2 = (1 - ss_res / ss_tot) * 100
    return r2

def load_model(config, model_path, input_dim, use_diffusion=True):
    device = config.device

    if use_diffusion:
        print("Loading Diffusion model...")
        model = StableCaRIVAEWithDiffusion(
            input_dim=input_dim,
            hidden_dims=config.vae_hidden_dims,
            z_dim=config.z_dim,
            factor_dim=config.factor_dim,
            config=config
        )
    else:
        print("Loading VAE model...")
        model = StableCaRIVAE(
            input_dim=input_dim,
            hidden_dims=config.vae_hidden_dims,
            z_dim=config.z_dim,
            factor_dim=config.factor_dim,
            config=config,
            use_r_and_x=True
        )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Loading with weights_only=False failed: {e}")
        print("Trying legacy loading method...")
        checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Model loaded: {model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif hasattr(obj, 'dtype'):
        return float(obj) if np.isscalar(obj) else obj.tolist()
    else:
        return obj

def get_historical_factor_mean(model, x_seq, r_hist, time_index, historical_factors, window_size=10):
    if hasattr(model, 'diffusion'):
        model.reset_f_prev_buffer()
        outputs = model(x_seq, r_t=r_hist.unsqueeze(0), mode='inference')
        f_current = outputs['f_t'].detach().cpu().numpy()[0]
        model.reset_f_prev_buffer()
    else:
        x_flat = x_seq[:, -1, :]
        recon_ft, _, _ = model(r=r_hist.unsqueeze(0), x=x_flat, sample_z=False)
        f_current = recon_ft.detach().cpu().numpy()[0]

    historical_factors[time_index] = f_current

    historical_times = [t for t in historical_factors.keys() if t < time_index]
    historical_times_sorted = sorted(historical_times)

    if len(historical_times_sorted) == 0:
        return f_current

    recent_times = historical_times_sorted[-window_size:]
    recent_factors = [historical_factors[t] for t in recent_times]

    f_bar = np.mean(recent_factors, axis=0)
    return f_bar

def enhanced_test_model(model, test_loader, config, time_indices=None):
    model.eval()
    device = config.device

    all_predictions_total = []
    all_predictions_pred = []
    all_targets = []
    all_time_indices = []

    is_diffusion = hasattr(model, 'diffusion')

    historical_factors = {}

    print("Starting enhanced testing...")

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing progress")

    with torch.no_grad():
        for batch_idx, batch_data in progress_bar:
            try:
                if len(batch_data) == 6:
                    x_batch, y_batch, r_hist_batch, x_t_prev, r_t_prev, r_hist_t_prev = batch_data
                else:
                    x_batch, y_batch, r_hist_batch = batch_data
                    x_t_prev, r_t_prev, r_hist_t_prev = None, None, None

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                r_hist_batch = r_hist_batch.to(device)

                batch_size = x_batch.size(0)

                dataset = test_loader.dataset
                if hasattr(dataset, 'time_indices') and dataset.time_indices is not None:
                    start_idx = batch_idx * test_loader.batch_size
                    end_idx = min(start_idx + batch_size, len(dataset))
                    batch_time_indices = dataset.time_indices[start_idx:end_idx]
                else:
                    batch_time_indices = [batch_idx] * batch_size

                batch_predictions_total = []
                batch_predictions_pred = []

                sample_progress = tqdm(range(batch_size), desc=f"Batch {batch_idx} processing samples", leave=False)

                for i in sample_progress:
                    x_seq_i = x_batch[i:i + 1]
                    y_i = y_batch[i:i + 1]
                    r_hist_i = r_hist_batch[i:i + 1]
                    time_idx = batch_time_indices[i] if i < len(batch_time_indices) else batch_idx

                    if is_diffusion:
                        model.reset_f_prev_buffer()
                        pred_total, _ = model.predict(x_seq_i, r_t=r_hist_i)
                        model.reset_f_prev_buffer()
                    else:
                        x_flat_i = x_seq_i[:, -1, :]
                        pred_total = model.predict(x_seq_i, x_flat_i, y=None, r=r_hist_i)

                    pred_total_np = pred_total.cpu().numpy().flatten()[0]
                    batch_predictions_total.append(pred_total_np)

                    f_bar = get_historical_factor_mean(
                        model, x_seq_i, r_hist_i, time_idx, historical_factors
                    )

                    z = model.get_z_from_x(x_seq_i)
                    z_for_beta = z.unsqueeze(1)

                    if is_diffusion:
                        beta_z_factor = model.beta_network(z_for_beta)
                        beta_z_factor = beta_z_factor.squeeze(1)

                        f_bar_tensor = torch.FloatTensor(f_bar).unsqueeze(0).to(device)

                        pred_pred = (beta_z_factor * f_bar_tensor).sum(dim=1)
                    else:
                        beta_z_factor = model.beta_network(z_for_beta)
                        beta_z_factor = beta_z_factor.squeeze(1)
                        beta_z = model.beta_projection(beta_z_factor)

                        f_bar_tensor = torch.FloatTensor(f_bar).unsqueeze(0).to(device)

                        pred_pred = (beta_z * f_bar_tensor).sum(dim=1)

                    pred_pred_np = pred_pred.cpu().numpy().flatten()[0]
                    batch_predictions_pred.append(pred_pred_np)

                    sample_progress.set_description(f"Batch {batch_idx} sample {i + 1}/{batch_size}")

                actual_np = y_batch.cpu().numpy().flatten()

                all_predictions_total.extend(batch_predictions_total)
                all_predictions_pred.extend(batch_predictions_pred)
                all_targets.extend(actual_np)
                all_time_indices.extend(batch_time_indices[:batch_size])

                progress_bar.set_description(f"Testing progress [{batch_idx + 1}/{len(test_loader)}]")

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

    print("Computing daily metrics...")
    daily_metrics = compute_daily_metrics(
        all_predictions_total, all_predictions_pred, all_targets, all_time_indices
    )

    print("Computing aggregate metrics...")
    aggregate_metrics = compute_aggregate_metrics(daily_metrics)

    print("Computing global metrics...")
    ic = compute_ic(all_predictions_total, all_targets)
    rank_ic = compute_rank_ic(all_predictions_total, all_targets)
    r2_total = compute_r2(all_predictions_total, all_targets)
    r2_pred = compute_r2(all_predictions_pred, all_targets)

    print("Computing ICIR...")
    if len(np.unique(all_time_indices)) > 1:
        unique_times = np.unique(all_time_indices)
        ic_series = []

        for t in tqdm(unique_times, desc="Computing IC series", leave=False):
            mask = all_time_indices == t
            if mask.sum() >= 2:
                pred_t = all_predictions_total[mask]
                actual_t = all_targets[mask]
                ic_t = compute_ic(pred_t, actual_t)
                if not np.isnan(ic_t):
                    ic_series.append(ic_t)

        if len(ic_series) > 0:
            ic_mean = np.mean(ic_series)
            ic_std = np.std(ic_series)
            icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
        else:
            icir = 0.0
    else:
        icir = 0.0

    return {
        'ic': ic,
        'icir': icir,
        'rank_ic': rank_ic,
        'r2_total': r2_total,
        'r2_pred': r2_pred,
        'daily_metrics': daily_metrics,
        'aggregate_metrics': aggregate_metrics,
        'all_predictions_total': all_predictions_total,
        'all_predictions_pred': all_predictions_pred,
        'all_targets': all_targets,
        'all_time_indices': all_time_indices
    }

def main():
    print("=" * 70)
    print("Starting enhanced evaluation metrics computation")
    print("Adding standard deviation, worst environment performance and OOD drop calculation")
    print("Fixing R²_total and R²_pred calculation logic")
    print("Adding progress bar display")
    print("=" * 70)

    config = StableConfig()

    device = config.device
    print(f"Using device: {device}")

    print("\n[1] Data preprocessing...")
    preprocessor = StableDataPreprocessor(config)
    processed_data = preprocessor.preprocess_all_data()

    if len(processed_data['test'][0]) == 0:
        print("Error: Test data is empty, please check data file paths")
        return

    (train_seq, train_target, train_hist_returns, train_time_indices, train_stock_indices) = processed_data['train']
    (val_seq, val_target, val_hist_returns, val_time_indices, val_stock_indices) = processed_data['val']
    (test_seq, test_target, test_hist_returns, test_time_indices, test_stock_indices) = processed_data['test']

    print(f"Data shapes:")
    print(f"  Training set: {train_seq.shape}")
    print(f"  Validation set: {val_seq.shape}")
    print(f"  Test set: {test_seq.shape}")

    print("\n[2] Creating data loaders...")

    train_dataset = StableStockDataset(
        train_seq, train_target, train_hist_returns,
        train_time_indices, train_stock_indices,
        use_temporal_pairs=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_dataset = StableStockDataset(
        test_seq, test_target, test_hist_returns,
        test_time_indices, test_stock_indices,
        use_temporal_pairs=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("\n[3] Loading trained model...")

    diffusion_model_path = "stable_cari_diffusion_transformer_complete.pth"
    vae_model_path = "stable_cari_vae_mi_beta_transformer_complete.pth"

    if os.path.exists(diffusion_model_path):
        model_path = diffusion_model_path
        use_diffusion = True
    elif os.path.exists(vae_model_path):
        model_path = vae_model_path
        use_diffusion = False
    else:
        print("Error: Model file not found, please check paths")
        print(f"Attempted: {diffusion_model_path}")
        print(f"Attempted: {vae_model_path}")
        return

    print(f"Using model file: {model_path}")
    print(f"Model type: {'Diffusion' if use_diffusion else 'VAE'}")

    input_dim = test_seq.shape[-1] if len(test_seq) > 0 else 1

    model = load_model(config, model_path, input_dim, use_diffusion)

    print("\n[4] Computing enhanced evaluation metrics...")

    try:
        print("Computing training set metrics...")
        train_results = enhanced_test_model(model, train_loader, config, train_time_indices)

        print("Computing test set metrics...")
        test_results = enhanced_test_model(model, test_loader, config, test_time_indices)

        if train_results is None or test_results is None:
            print("Error: Metrics computation failed")
            return

        print("Computing OOD drop...")
        ood_drop = compute_ood_drop(
            train_results['aggregate_metrics'],
            test_results['aggregate_metrics']
        )

        print("\n[5] Saving enhanced results to CSV files...")

        enhanced_metrics = []
        metrics_info = [
            ('IC', 'ic_total', 'Information Coefficient'),
            ('RankIC', 'rank_ic_total', 'Rank Information Coefficient'),
            ('R²_total', 'r2_total', 'Total R-squared (using current factors)'),
            ('R²_pred', 'r2_pred', 'Predicted R-squared (using historical factor mean)')
        ]

        for metric_name, metric_key, description in tqdm(metrics_info, desc="Generating enhanced metrics"):
            train_agg = train_results['aggregate_metrics'][metric_key]
            test_agg = test_results['aggregate_metrics'][metric_key]

            enhanced_metrics.append({
                'Metric': metric_name,
                'Description': description,
                'Train_Mean': f"{train_agg['mean']:.6f}",
                'Train_Std': f"{train_agg['std']:.6f}",
                'Train_Worst': f"{train_agg['worst']:.6f}",
                'Train_Valid_Envs': train_agg['num_valid'],
                'Test_Mean': f"{test_agg['mean']:.6f}",
                'Test_Std': f"{test_agg['std']:.6f}",
                'Test_Worst': f"{test_agg['worst']:.6f}",
                'Test_Valid_Envs': test_agg['num_valid'],
                'OOD_Drop': f"{ood_drop[metric_key]:.6f}"
            })

        basic_metrics = {
            'Metric': ['IC', 'ICIR', 'RankIC', 'R²_total', 'R²_pred'],
            'Value': [
                test_results['ic'],
                test_results['icir'],
                test_results['rank_ic'],
                test_results['r2_total'],
                test_results['r2_pred']
            ],
            'Description': [
                'Information Coefficient (IC)',
                'Information Coefficient Information Ratio (ICIR)',
                'Rank Information Coefficient (RankIC)',
                'Total R-squared (using current factors)',
                'Predicted R-squared (using historical factor mean)'
            ]
        }

        basic_df = pd.DataFrame(basic_metrics)
        basic_df['Value_Formatted'] = basic_df['Value'].apply(lambda x: f"{x:.6f}")
        basic_csv_filename = "evaluation_basic_metrics.csv"
        basic_df.to_csv(basic_csv_filename, index=False, encoding='utf-8')

        enhanced_df = pd.DataFrame(enhanced_metrics)
        enhanced_csv_filename = "evaluation_enhanced_metrics.csv"
        enhanced_df.to_csv(enhanced_csv_filename, index=False, encoding='utf-8')

        print("\n[6] Generating daily environment performance CSV file...")

        daily_df = pd.DataFrame({
            'time_index': test_results['daily_metrics']['time_index'],
            'ic_total': test_results['daily_metrics']['ic_total'],
            'rank_ic_total': test_results['daily_metrics']['rank_ic_total'],
            'r2_total': test_results['daily_metrics']['r2_total'],
            'r2_pred': test_results['daily_metrics']['r2_pred'],
            'num_samples': test_results['daily_metrics']['num_samples']
        })

        start_date = datetime(2020, 1, 1)
        daily_df['date'] = [start_date + timedelta(days=int(idx)) for idx in daily_df['time_index']]

        daily_df = daily_df[['time_index', 'date', 'ic_total', 'rank_ic_total', 'r2_total', 'r2_pred', 'num_samples']]

        daily_csv_filename = "daily_environment_metrics.csv"
        daily_df.to_csv(daily_csv_filename, index=False, encoding='utf-8')

        print(f"Basic metrics saved to: {basic_csv_filename}")
        print(f"Enhanced metrics saved to: {enhanced_csv_filename}")
        print(f"Daily environment performance saved to: {daily_csv_filename}")

        print("\n" + "=" * 70)
        print("Enhanced Evaluation Metrics Results")
        print("=" * 70)

        print("\nBasic metrics:")
        print("-" * 50)
        for _, row in basic_df.iterrows():
            print(f"{row['Metric']:>10}: {row['Value_Formatted']} - {row['Description']}")

        print("\nEnhanced metrics (including standard deviation, worst environment and OOD drop):")
        print("-" * 80)
        for metric in enhanced_metrics:
            print(f"\n{metric['Metric']} ({metric['Description']}):")
            print(f"  Training set - Mean: {metric['Train_Mean']}, Std: {metric['Train_Std']}, Worst environment: {metric['Train_Worst']}")
            print(f"  Test set - Mean: {metric['Test_Mean']}, Std: {metric['Test_Std']}, Worst environment: {metric['Test_Worst']}")
            print(f"  OOD Drop: {metric['OOD_Drop']}")

        print("\nR²_total and R²_pred difference analysis:")
        print(f"  R²_total (Test set): {test_results['r2_total']:.6f}%")
        print(f"  R²_pred (Test set): {test_results['r2_pred']:.6f}%")
        diff = abs(test_results['r2_total'] - test_results['r2_pred'])
        print(f"  Difference: {diff:.6f}%")

        print("\n" + "=" * 70)

        print("\n[7] Saving detailed results to JSON file...")

        detailed_results = {
            'config': {
                'model_type': 'Diffusion' if use_diffusion else 'VAE',
                'model_path': model_path,
                'device': str(device)
            },
            'basic_metrics': convert_numpy_types(basic_metrics),
            'enhanced_metrics': convert_numpy_types(enhanced_metrics),
            'ood_drop': convert_numpy_types(ood_drop),
            'train_aggregate_metrics': convert_numpy_types(train_results['aggregate_metrics']),
            'test_aggregate_metrics': convert_numpy_types(test_results['aggregate_metrics']),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        json_filename = "evaluation_detailed_results.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        print(f"Detailed results saved to: {json_filename}")

        print(f"\nBasic metrics CSV file preview:")
        print(basic_df.to_string(index=False))

        print(f"\nEnhanced metrics CSV file preview:")
        print(enhanced_df.to_string(index=False))

        print(f"\nDaily environment performance CSV file first 5 rows preview:")
        print(daily_df.head().to_string(index=False))

    except Exception as e:
        print(f"Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nEnhanced evaluation metrics computation completed!")

if __name__ == "__main__":
    main()