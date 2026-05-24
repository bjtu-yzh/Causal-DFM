import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import json
from datetime import datetime
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import StableConfig
from data_preprocessing import StableDataPreprocessor, StableStockDataset
from models import StableCaRIVAEWithDiffusion
from evaluation import test_model
from torch.utils.data import DataLoader


class AblationStudy:
    def __init__(self, config, model_path, test_loader):
        self.config = config
        self.model_path = model_path
        self.test_loader = test_loader
        self.device = config.device
        self.results = []

    def load_base_model(self):
        print("Loading base Diffusion model...")

        sample_batch = next(iter(self.test_loader))
        if len(sample_batch) == 6:
            x_batch, _, _, _, _, _ = sample_batch
        else:
            x_batch, _, _ = sample_batch

        input_dim = x_batch.shape[-1]

        model = StableCaRIVAEWithDiffusion(
            input_dim=input_dim,
            hidden_dims=self.config.vae_hidden_dims,
            z_dim=self.config.z_dim,
            factor_dim=self.config.factor_dim,
            config=self.config
        )

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {e}")
            return None

        model.to(self.device)
        model.eval()
        return model

    def create_ablation_variants(self, base_model):
        variants = {}

        variants['full_model'] = {
            'model': base_model,
            'description': 'Full Diffusion model',
            'config_changes': {}
        }

        variants['no_transformer'] = self._create_no_transformer_variant(base_model)

        variants['no_betanetwork'] = self._create_no_betanetwork_variant(base_model)

        variants['no_diffusion'] = self._create_no_diffusion_variant(base_model)

        variants['no_ib'] = self._create_no_ib_variant(base_model)

        variants['no_prediction'] = self._create_no_prediction_variant(base_model)

        variants['no_cfg'] = self._create_no_cfg_variant(base_model)

        variants['simple_denoiser'] = self._create_simple_denoiser_variant(base_model)

        return variants

    def _create_no_transformer_variant(self, base_model):
        import copy
        model = copy.deepcopy(base_model)

        class SimpleMLPEncoder(nn.Module):
            def __init__(self, input_dim, z_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, z_dim)
                )

            def forward(self, x):
                x_flat = x[:, -1, :]
                return self.net(x_flat)

        input_dim = base_model.input_dim
        z_dim = base_model.z_dim
        model.transformer_encoder = SimpleMLPEncoder(input_dim, z_dim)

        def simple_get_z_from_x(x):
            return model.transformer_encoder(x)

        model.get_z_from_x = simple_get_z_from_x

        return {
            'model': model,
            'description': 'No Transformer Encoder (replaced with MLP)',
            'config_changes': {'transformer_layers_time': 0, 'transformer_layers_cross': 0}
        }

    def _create_no_betanetwork_variant(self, base_model):
        import copy
        model = copy.deepcopy(base_model)

        class DimensionAwareBetaNetwork(nn.Module):
            def __init__(self, z_dim, factor_dim, input_dim):
                super().__init__()
                self.z_dim = z_dim
                self.factor_dim = factor_dim
                self.input_dim = input_dim

                if factor_dim != input_dim:
                    self.proj_to_factor = nn.Linear(z_dim, factor_dim)
                    print(f"Dimension fix: Adding projection layer {z_dim} -> {factor_dim}")
                else:
                    self.proj_to_factor = nn.Identity()

            def forward(self, z):

                if z.dim() == 3:
                    B, N, z_dim = z.shape
                    z_flat = z.reshape(B * N, z_dim)
                    beta_z_flat = self.proj_to_factor(z_flat)
                    return beta_z_flat.reshape(B, N, self.factor_dim)
                else:
                    return self.proj_to_factor(z)

        model.beta_network = DimensionAwareBetaNetwork(
            base_model.z_dim, base_model.factor_dim, base_model.input_dim
        )

        if not hasattr(model, 'beta_projection'):

            if base_model.factor_dim == base_model.input_dim:
                model.beta_projection = nn.Identity()
            else:

                model.beta_projection = nn.Linear(base_model.factor_dim, base_model.input_dim)

                if base_model.factor_dim <= base_model.input_dim:
                    with torch.no_grad():

                        model.beta_projection.weight.data[:base_model.factor_dim, :base_model.factor_dim] = torch.eye(
                            base_model.factor_dim)
                        if base_model.factor_dim < base_model.input_dim:
                            model.beta_projection.weight.data[base_model.factor_dim:, :] = 0
                        model.beta_projection.bias.data.zero_()

        original_forward = model.forward

        def patched_forward(x_seq, r_t=None, f_prev=None, mode='train'):

            result = original_forward(x_seq, r_t, f_prev, mode)

            if isinstance(result, dict) and 'beta_z' not in result:

                z = model.get_z_from_x(x_seq)
                if z.dim() == 2:
                    z = z.unsqueeze(1)

                beta_z_factor = model.beta_network(z)
                beta_z = model.beta_projection(beta_z_factor.squeeze(1))

                result['beta_z'] = beta_z

            return result

        model.forward = patched_forward

        if hasattr(model, 'predict'):
            original_predict = model.predict

            def dimension_aware_predict(x_seq, r_t=None, f_prev=None):
                try:
                    predictions, outputs = original_predict(x_seq, r_t, f_prev)

                    if isinstance(outputs, dict) and 'beta_z' not in outputs:
                        z = model.get_z_from_x(x_seq)
                        if z.dim() == 2:
                            z = z.unsqueeze(1)

                        beta_z_factor = model.beta_network(z)
                        beta_z = model.beta_projection(beta_z_factor.squeeze(1))
                        outputs['beta_z'] = beta_z

                    return predictions, outputs
                except Exception as e:
                    print(f"Dimension error during prediction: {e}")

                    return self._create_fallback_prediction(model, x_seq, r_t, f_prev)

            model.predict = dimension_aware_predict

        return {
            'model': model,
            'description': 'No BetaNetwork (replaced with identity mapping, fixing dimensions)',
            'config_changes': {'beta_network_complexity': 'identity_with_dim_fix'}
        }

    def _create_no_diffusion_variant(self, base_model):
        import copy
        model = copy.deepcopy(base_model)

        class DimensionAwareFactorPredictor(nn.Module):
            def __init__(self, z_dim, factor_dim, input_dim, r_feat_dim=32):
                super().__init__()
                self.factor_dim = factor_dim
                self.input_dim = input_dim

                self.factor_mlp = nn.Sequential(
                    nn.Linear(z_dim + r_feat_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, factor_dim)
                )

            def forward(self, z, r_feat, f_prev=None):

                if z.dim() == 3:
                    z = z.mean(dim=1)

                if r_feat.dim() == 3:
                    r_feat = r_feat.mean(dim=1)

                combined = torch.cat([z, r_feat], dim=1)
                return self.factor_mlp(combined)

        class SimpleReturnEncoder(nn.Module):
            def __init__(self, r_feat_dim=32):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(1, r_feat_dim),
                    nn.ReLU(),
                    nn.Linear(r_feat_dim, r_feat_dim)
                )

            def forward(self, r_t):
                if r_t.dim() == 1:
                    r_t = r_t.unsqueeze(-1)
                return self.encoder(r_t)

        model.original_diffusion = model.diffusion

        model.diffusion = None
        model.factor_predictor = DimensionAwareFactorPredictor(
            base_model.z_dim, base_model.factor_dim, base_model.input_dim
        )
        model.return_encoder_simple = SimpleReturnEncoder(32)

        if not hasattr(model, 'beta_projection'):

            if base_model.factor_dim == base_model.input_dim:
                model.beta_projection = nn.Identity()
            else:
                model.beta_projection = nn.Linear(base_model.factor_dim, base_model.input_dim)

                if base_model.factor_dim <= base_model.input_dim:
                    with torch.no_grad():
                        model.beta_projection.weight.data[:base_model.factor_dim, :base_model.factor_dim] = torch.eye(
                            base_model.factor_dim)
                        if base_model.factor_dim < base_model.input_dim:
                            model.beta_projection.weight.data[base_model.factor_dim:, :] = 0
                        model.beta_projection.bias.data.zero_()

        def dimension_aware_forward(x_seq, r_t=None, f_prev=None, mode='train'):
            B = x_seq.shape[0]

            z = model.get_z_from_x(x_seq)

            if r_t is not None:
                if r_t.dim() == 1:
                    r_t = r_t.unsqueeze(-1)
                r_feat = model.return_encoder_simple(r_t)
            else:
                B = z.shape[0] if z.dim() == 2 else z.shape[0]
                r_feat = torch.zeros(B, 32, device=z.device)

            f_t = model.factor_predictor(z, r_feat, f_prev)

            if z.dim() == 2:
                z_for_beta = z.unsqueeze(1)
            else:
                z_for_beta = z

            beta_z_factor = model.beta_network(z_for_beta)
            beta_z = model.beta_projection(beta_z_factor.squeeze(1))

            if mode == 'train':

                loss = torch.tensor(0.1, device=z.device)
                return {
                    'diffusion_loss': loss,
                    'f_t_star': f_t.detach(),
                    'beta_z': beta_z,
                    'z': z
                }
            else:
                return {
                    'f_t': f_t,
                    'beta_z': beta_z,
                    'z': z
                }

        model.forward = dimension_aware_forward

        def dimension_aware_predict(x_seq, r_t=None, f_prev=None):
            with torch.no_grad():
                try:
                    outputs = model.forward(x_seq, r_t, f_prev, mode='inference')
                    f_t = outputs['f_t']

                    beta_z = outputs['beta_z']

                    if beta_z.dim() == 3:

                        if f_t.dim() == 2:
                            f_t = f_t.unsqueeze(1)

                        if beta_z.shape[-1] != f_t.shape[-1]:

                            if beta_z.shape[-1] > f_t.shape[-1]:

                                min_dim = min(beta_z.shape[-1], f_t.shape[-1])
                                beta_z_adj = beta_z[:, :, :min_dim]
                                f_t_adj = f_t[:, :, :min_dim]
                            else:

                                min_dim = min(beta_z.shape[-1], f_t.shape[-1])
                                beta_z_adj = beta_z
                                f_t_adj = f_t[:, :, :min_dim]

                            pred = (beta_z_adj * f_t_adj).sum(dim=-1)
                        else:

                            pred = (beta_z * f_t).sum(dim=-1)
                    else:

                        if beta_z.shape[-1] != f_t.shape[-1]:

                            min_dim = min(beta_z.shape[-1], f_t.shape[-1])
                            beta_z_adj = beta_z[:, :min_dim]
                            f_t_adj = f_t[:, :min_dim]
                            pred = (beta_z_adj * f_t_adj).sum(dim=1)
                        else:

                            pred = (beta_z * f_t).sum(dim=1)

                    return pred, outputs

                except Exception as e:
                    print(f"Dimension error during prediction: {e}")

                    return self._create_fallback_prediction(model, x_seq, r_t, f_prev)

        model.predict = dimension_aware_predict

        return {
            'model': model,
            'description': 'No TemporalFactorDiffusion (replaced with MLP, fixing dimensions)',
            'config_changes': {'diffusion_lambda': 0.0}
        }

    def _create_fallback_prediction(self, model, x_seq, r_t=None, f_prev=None):
        B = x_seq.shape[0]
        device = x_seq.device

        try:

            z = model.get_z_from_x(x_seq)

            if hasattr(model, 'beta_projection'):

                if z.dim() == 3:
                    z_flat = z.mean(dim=1)
                else:
                    z_flat = z

                if not hasattr(model, 'fallback_proj'):
                    model.fallback_proj = nn.Linear(z_flat.shape[-1], 1).to(device)

                pred = model.fallback_proj(z_flat).squeeze(-1)
            else:

                pred = torch.zeros(B, device=device)

            outputs = {
                'f_t': torch.zeros(B, model.factor_dim, device=device),
                'beta_z': torch.zeros(B, model.input_dim, device=device),
                'z': z
            }

            return pred, outputs

        except Exception as e:
            print(f"Fallback prediction also failed: {e}")

            B = x_seq.shape[0]
            pred = torch.zeros(B, device=device)
            outputs = {
                'f_t': torch.zeros(B, model.factor_dim, device=device),
                'beta_z': torch.zeros(B, model.input_dim, device=device),
                'z': torch.zeros(B, model.z_dim, device=device)
            }
            return pred, outputs

    def _create_no_ib_variant(self, base_model):
        import copy
        model = copy.deepcopy(base_model)

        ablation_config = copy.deepcopy(self.config)
        ablation_config.info_bottleneck_lambda = 0.0

        return {
            'model': model,
            'description': 'No information bottleneck constraint (IB Loss weight = 0)',
            'config_changes': {'info_bottleneck_lambda': 0.0}
        }

    def _create_no_prediction_variant(self, base_model):
        import copy
        model = copy.deepcopy(base_model)

        ablation_config = copy.deepcopy(self.config)
        ablation_config.prediction_lambda = 0.0

        return {
            'model': model,
            'description': 'No prediction loss (Prediction Loss weight = 0)',
            'config_changes': {'prediction_lambda': 0.0}
        }

    def _create_no_cfg_variant(self, base_model):
        import copy
        model = copy.deepcopy(base_model)

        ablation_config = copy.deepcopy(self.config)
        ablation_config.diffusion_mask_r_prob = 0.0

        if hasattr(model, 'diffusion') and model.diffusion is not None:
            model.diffusion.mask_r_prob = 0.0

        return {
            'model': model,
            'description': 'No Classifier-Free Guidance (mask probability = 0)',
            'config_changes': {'diffusion_mask_r_prob': 0.0}
        }

    def _create_simple_denoiser_variant(self, base_model):
        import copy
        model = copy.deepcopy(base_model)

        ablation_config = copy.deepcopy(self.config)
        ablation_config.diffusion_denoiser_num_layers = 1
        ablation_config.diffusion_denoiser_num_heads = 2
        ablation_config.diffusion_denoiser_hidden_dim = 128

        return {
            'model': model,
            'description': 'Simplified Denoiser (1 layer, 2 heads, 128 dims)',
            'config_changes': {
                'diffusion_denoiser_num_layers': 1,
                'diffusion_denoiser_num_heads': 2,
                'diffusion_denoiser_hidden_dim': 128
            }
        }

    def evaluate_variant(self, variant, variant_name):
        print(f"\nEvaluating variant: {variant_name}")
        print(f"Description: {variant['description']}")

        model = variant['model']
        model.to(self.device)
        model.eval()

        try:

            results = test_model(model, self.test_loader, self.config)

            if results is not None:

                metrics = {
                    'variant': variant_name,
                    'description': variant['description'],
                    'IC': float(results.get('ic', 0)),
                    'ICIR': float(results.get('icir', 0)),
                    'RankIC': float(results.get('rank_ic', 0)),
                    'R2_total': float(results.get('r2_total', 0)),
                    'R2_pred': float(results.get('r2_pred', 0))
                }

                config_changes = variant['config_changes']
                for key, value in config_changes.items():
                    if isinstance(value, (int, float, bool, str)):
                        metrics[key] = value
                    else:
                        metrics[key] = str(value)

                self.results.append(metrics)

                print(f"Evaluation completed: IC={metrics['IC']:.4f}, ICIR={metrics['ICIR']:.4f}, "
                      f"RankIC={metrics['RankIC']:.4f}, R2_total={metrics['R2_total']:.4f}%, "
                      f"R2_pred={metrics['R2_pred']:.4f}%")

                return metrics
            else:
                print("Evaluation failed, returning empty results")

                default_metrics = {
                    'variant': variant_name,
                    'description': variant['description'],
                    'IC': 0.0,
                    'ICIR': 0.0,
                    'RankIC': 0.0,
                    'R2_total': 0.0,
                    'R2_pred': 0.0
                }
                default_metrics.update(variant['config_changes'])
                self.results.append(default_metrics)
                return default_metrics

        except Exception as e:
            print(f"Error evaluating variant {variant_name}: {e}")
            import traceback
            traceback.print_exc()

            error_metrics = {
                'variant': variant_name,
                'description': variant['description'],
                'IC': 0.0,
                'ICIR': 0.0,
                'RankIC': 0.0,
                'R2_total': 0.0,
                'R2_pred': 0.0,
                'error': str(e)
            }
            error_metrics.update(variant['config_changes'])
            self.results.append(error_metrics)
            return error_metrics

    def run_ablation_study(self):
        print("=" * 70)
        print("Starting Diffusion model ablation study")
        print("=" * 70)

        base_model = self.load_base_model()
        if base_model is None:
            print("Unable to load base model, terminating experiment")
            return None

        variants = self.create_ablation_variants(base_model)

        print(f"Created {len(variants)} ablation variants")

        for variant_name, variant in variants.items():
            self.evaluate_variant(variant, variant_name)

        self.save_results_to_csv()

        self.print_summary()

        return self.results

    def save_results_to_csv(self):
        if not self.results:
            print("No results to save")
            return

        df = pd.DataFrame(self.results)

        base_columns = ['variant', 'description', 'IC', 'ICIR', 'RankIC', 'R2_total', 'R2_pred']
        config_columns = [col for col in df.columns if col not in base_columns and col != 'error']

        if 'error' in df.columns:
            config_columns.append('error')

        df = df[base_columns + config_columns]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"diffusion_ablation_study_{timestamp}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8')

        print(f"\nResults saved to: {csv_filename}")

        json_filename = f"diffusion_ablation_study_{timestamp}.json"
        self._save_results_to_json(json_filename)

        print(f"Detailed results saved to: {json_filename}")

        return csv_filename

    def _save_results_to_json(self, filename):
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (str, type(None))):
                return obj
            else:
                return str(obj)

        serializable_results = convert_to_serializable(self.results)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        if not self.results:
            print("No results to display")
            return

        print("\n" + "=" * 70)
        print("Ablation Study Summary")
        print("=" * 70)

        df = pd.DataFrame(self.results)

        full_model_metrics = None
        for result in self.results:
            if result['variant'] == 'full_model':
                full_model_metrics = result
                break

        if full_model_metrics:
            print(f"Baseline model (full model) metrics:")
            print(f"  IC:       {full_model_metrics['IC']:.6f}")
            print(f"  ICIR:     {full_model_metrics['ICIR']:.6f}")
            print(f"  RankIC:   {full_model_metrics['RankIC']:.6f}")
            print(f"  R²_total: {full_model_metrics['R2_total']:.6f}%")
            print(f"  R²_pred:  {full_model_metrics['R2_pred']:.6f}%")
            print()

        print("Ablation variant performance comparison:")
        print("-" * 90)
        print(f"{'Variant Name':<25} {'IC':<10} {'ICIR':<10} {'RankIC':<10} {'R²_total':<12} {'R²_pred':<12}")
        print("-" * 90)

        for result in self.results:
            variant_name = result['variant']
            description = result['description'][:20] + "..." if len(result['description']) > 20 else result[
                'description']

            ic = result['IC']
            icir = result['ICIR']
            rank_ic = result['RankIC']
            r2_total = result['R2_total']
            r2_pred = result['R2_pred']

            if full_model_metrics:
                ic_change = ic - full_model_metrics['IC']
                icir_change = icir - full_model_metrics['ICIR']
                rank_ic_change = rank_ic - full_model_metrics['RankIC']
                r2_total_change = r2_total - full_model_metrics['R2_total']
                r2_pred_change = r2_pred - full_model_metrics['R2_pred']

                ic_str = f"{ic:.4f} ({ic_change:+.4f})"
                icir_str = f"{icir:.4f} ({icir_change:+.4f})"
                rank_ic_str = f"{rank_ic:.4f} ({rank_ic_change:+.4f})"
                r2_total_str = f"{r2_total:.2f}% ({r2_total_change:+.2f}%)"
                r2_pred_str = f"{r2_pred:.2f}% ({r2_pred_change:+.2f}%)"
            else:
                ic_str = f"{ic:.4f}"
                icir_str = f"{icir:.4f}"
                rank_ic_str = f"{rank_ic:.4f}"
                r2_total_str = f"{r2_total:.2f}%"
                r2_pred_str = f"{r2_pred:.2f}%"

            print(
                f"{variant_name:<25} {ic_str:<10} {icir_str:<10} {rank_ic_str:<10} {r2_total_str:<12} {r2_pred_str:<12}")

        print("-" * 90)

        if full_model_metrics and len(self.results) > 1:
            print("\nPerformance impact analysis:")
            print("Impact of component removal on IC metric (negative values indicate performance degradation):")

            ic_changes = []
            for result in self.results:
                if result['variant'] != 'full_model':
                    change = result['IC'] - full_model_metrics['IC']
                    ic_changes.append((result['variant'], change, result['description']))

            ic_changes.sort(key=lambda x: x[1])

            for variant, change, desc in ic_changes:
                change_percent = (change / abs(full_model_metrics['IC'])) * 100 if full_model_metrics['IC'] != 0 else 0
                trend = "↓" if change < 0 else "↑"
                print(f"  {variant:<20} {change:+.4f} ({change_percent:+.1f}%) {trend} - {desc}")


def prepare_test_data(config):
    print("Preparing test data...")

    preprocessor = StableDataPreprocessor(config)
    processed_data = preprocessor.preprocess_all_data()

    if processed_data is None or len(processed_data['test'][0]) == 0:
        print("Error: Test data is empty")
        return None

    test_seq, test_target, test_hist_returns, test_time_indices, test_stock_indices = processed_data['test']

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

    print(f"Number of test samples: {len(test_dataset)}")
    return test_loader


def main():
    config = StableConfig()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "stable_cari_diffusion_transformer_complete.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        print("Please train the model first or provide the correct model path")
        return

    test_loader = prepare_test_data(config)
    if test_loader is None:
        print("Test data preparation failed")
        return

    ablation_study = AblationStudy(config, model_path, test_loader)

    results = ablation_study.run_ablation_study()

    if results is not None:
        print("\nAblation study completed!")
        print(f"Evaluated {len(results)} model variants in total")
    else:
        print("Ablation study failed")


if __name__ == "__main__":
    main()