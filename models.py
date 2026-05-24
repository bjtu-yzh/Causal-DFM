import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_encoder import TemporalCrossSectionEncoder
from prediction_head import BetaNetwork
from utils import stable_mutual_information_estimate
from temporal_factor_diffusion_dit import TemporalFactorDiffusionDiT as TemporalFactorDiffusion


class StableEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.1))
            prev = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, z_dim)
        self.fc_logvar = nn.Linear(prev, z_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = torch.clamp(x, -10, 10)
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10, 10)
        return mu, logvar


class StableDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = z_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, z):
        z = torch.clamp(z, -5, 5)
        return self.net(z)


class StableMutualInformationEstimator(nn.Module):
    def __init__(self, z_dim, y_dim=1, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev = z_dim + y_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, z, y):
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        z = torch.clamp(z, -5, 5)
        y = torch.clamp(y, -5, 5)
        zy = torch.cat([z, y], dim=1)
        return self.net(zy)


class StableCaRIVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, factor_dim, config, use_r_and_x=True):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.factor_dim = factor_dim
        self.config = config
        self.use_r_and_x = use_r_and_x

        vae_input_dim = (1 + input_dim) if use_r_and_x else 1

        self.encoder = StableEncoder(vae_input_dim, hidden_dims, z_dim)
        self.decoder = StableDecoder(z_dim, hidden_dims, input_dim)
        self.mi_estimator = StableMutualInformationEstimator(z_dim)

        self.transformer_encoder = TemporalCrossSectionEncoder(
            input_dim=input_dim,
            z_dim=z_dim,
            nhead=config.transformer_nhead,
            num_layers_time=config.transformer_layers_time,
            num_layers_cross=config.transformer_layers_cross,
            dropout=config.transformer_dropout
        )

        self.beta_network = BetaNetwork(
            z_dim=z_dim,
            factor_dim=factor_dim,
            hidden_dim=None
        )

        self.beta_projection = nn.Sequential(
            nn.Linear(factor_dim, input_dim),
            nn.Tanh()
        )

        self.predictor_y = nn.Linear(z_dim, 1)
        nn.init.xavier_uniform_(self.predictor_y.weight)
        nn.init.constant_(self.predictor_y.bias, 0.1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, r, x=None, sample_z=True):
        if r.dim() == 1:
            r = r.unsqueeze(1)

        if self.use_r_and_x:
            if x is None:
                raise ValueError("When use_r_and_x=True, x parameter must be provided")
            x = torch.clamp(x, -10, 10)
            vae_input = torch.cat([r, x], dim=1)
        else:
            vae_input = r

        vae_input = torch.clamp(vae_input, -10, 10)

        mu, logvar = self.encoder(vae_input)
        mu = torch.clamp(mu, -5, 5)
        logvar = torch.clamp(logvar, -10, 10)

        if sample_z:
            z_internal = self.reparameterize(mu, logvar)
        else:
            z_internal = mu

        z_internal = torch.clamp(z_internal, -5, 5)
        recon = self.decoder(z_internal)

        return recon, mu, logvar

    def get_z_from_x(self, x):
        batch_size, seq_length, P = x.shape
        x_expanded = x.unsqueeze(2)

        z = self.transformer_encoder(x_expanded)

        z = z.squeeze(1)

        return z

    def estimate_mutual_information(self, z, y, method='variational'):
        try:
            if method == 'variational':
                return self.stable_variational_mi_estimate(z, y)
            else:
                return stable_mutual_information_estimate(z, y, method)
        except Exception as e:
            print(f"Mutual information estimation error: {e}")
            return torch.tensor(0.1, device=z.device)

    def stable_variational_mi_estimate(self, z, y):
        try:
            y_pred = self.predictor_y(z)
            if y.dim() == 1:
                y = y.unsqueeze(1)
            mse_loss = F.mse_loss(y_pred, y, reduction='mean')
            mi_estimate = 1.0 / (1.0 + torch.clamp(mse_loss, 1e-8, 100.0))
            return torch.clamp(mi_estimate, 0.01, 0.99)
        except Exception as e:
            print(f"Variational MI error: {e}")
            return torch.tensor(0.1, device=z.device)

    def predict(self, x_seq, x_flat=None, y=None, r=None):
        z = self.get_z_from_x(x_seq)

        z_for_beta = z.unsqueeze(1)
        beta_z_factor = self.beta_network(z_for_beta)
        beta_z_factor = beta_z_factor.squeeze(1)

        beta_z = self.beta_projection(beta_z_factor)

        if x_flat is None:

            x_for_vae = x_seq[:, -1, :]
        else:
            x_for_vae = x_flat

        if r is None:

            if y is None:

                import warnings
                warnings.warn(
                    "predict method did not provide historical returns r, directly using x_flat as ft, this may cause training-test inconsistency")
                ft = x_for_vae
            else:

                import warnings
                warnings.warn("predict method uses y (future returns) as r, data leakage risk exists")
                recon_ft, _, _ = self.forward(r=y, x=x_for_vae, sample_z=False)
                ft = recon_ft
        else:

            recon_ft, _, _ = self.forward(r=r, x=x_for_vae, sample_z=False)
            ft = recon_ft

        ft = torch.clamp(ft, -10, 10)

        q = (beta_z * ft).sum(dim=1)

        return q

    def get_causal_representation(self, x):
        if x.dim() == 3:

            return self.get_z_from_x(x)
        else:

            x = torch.clamp(x, -10, 10)
            x_seq = x.unsqueeze(1)
            return self.get_z_from_x(x_seq)


class StableCaRIVAEWithDiffusion(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, factor_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.factor_dim = factor_dim
        self.config = config

        self.transformer_encoder = TemporalCrossSectionEncoder(
            input_dim=input_dim,
            z_dim=z_dim,
            nhead=config.transformer_nhead,
            num_layers_time=config.transformer_layers_time,
            num_layers_cross=config.transformer_layers_cross,
            dropout=config.transformer_dropout
        )

        self.beta_network = BetaNetwork(
            z_dim=z_dim,
            factor_dim=factor_dim,
            hidden_dim=None
        )

        diffusion_num_timesteps = getattr(config, 'diffusion_num_timesteps', 1000)
        diffusion_ddim_steps = getattr(config, 'diffusion_ddim_steps', 30)
        diffusion_r_feat_dim = getattr(config, 'diffusion_r_feat_dim', 32)
        self.diffusion = TemporalFactorDiffusion(
            factor_dim=factor_dim,
            z_dim=z_dim,
            r_feat_dim=diffusion_r_feat_dim,
            num_timesteps=diffusion_num_timesteps,
            beta_schedule=getattr(config, 'diffusion_beta_schedule', 'linear'),
            denoiser_hidden_dim=getattr(config, 'diffusion_denoiser_hidden_dim', 256),
            denoiser_num_layers=2,
            denoiser_num_heads=4,
            solver_method='cholesky',
            ddim_eta=getattr(config, 'diffusion_ddim_eta', 0.0),
            ddim_steps=diffusion_ddim_steps
        )

        self.diffusion_mask_r_prob = getattr(config, 'diffusion_mask_r_prob', 0.1)

        self.predictor_y = nn.Linear(z_dim, 1)
        nn.init.xavier_uniform_(self.predictor_y.weight)
        nn.init.constant_(self.predictor_y.bias, 0.1)

        self.register_buffer('f_prev_buffer', None)

        self.pred_scale = nn.Parameter(torch.tensor(10.0))

    def get_z_from_x(self, x):
        batch_size, seq_length, P = x.shape
        x_expanded = x.unsqueeze(2)
        z = self.transformer_encoder(x_expanded)
        z = z.squeeze(1)
        return z

    def get_beta_z(self, z):
        if z.dim() == 2:

            z_for_beta = z.unsqueeze(1)
            beta_z_factor = self.beta_network(z_for_beta)
            beta_z_factor = beta_z_factor.squeeze(1)
            return beta_z_factor
        elif z.dim() == 3:

            B, N, _ = z.shape
            z_flat = z.reshape(B * N, -1)
            z_for_beta = z_flat.unsqueeze(1)
            beta_z_factor = self.beta_network(z_for_beta)
            beta_z_factor = beta_z_factor.squeeze(1)
            beta_z_factor = beta_z_factor.reshape(B, N, -1)
            return beta_z_factor
        else:
            raise ValueError(f"Unexpected z dimension: {z.dim()}")

    def forward(self, x_seq, r_t=None, f_prev=None, mode='train'):
        z = self.get_z_from_x(x_seq)

        if r_t is not None and r_t.dim() == 2:

            B, N = r_t.shape
            z_expanded = z.unsqueeze(1).expand(B, N, -1)
            beta_z = self.get_beta_z(z_expanded)
        else:

            beta_z = self.get_beta_z(z)
            if r_t is not None:
                beta_z = beta_z.unsqueeze(1)

        if f_prev is None:
            if mode == 'inference' and self.f_prev_buffer is not None:

                f_prev = self.f_prev_buffer

                B = z.shape[0]
                if f_prev.shape[0] != B:
                    f_prev = torch.randn(B, self.factor_dim, device=z.device, dtype=z.dtype) * 0.1
            else:

                B = z.shape[0]
                f_prev = torch.randn(B, self.factor_dim, device=z.device, dtype=z.dtype) * 0.1

        if mode == 'train':
            if r_t is None:
                raise ValueError("r_t is required in training mode")
            diffusion_loss, f_t_star = self.diffusion(
                beta_z, r_t, z, f_prev.detach(),
                mode='train',
                mask_r_prob=self.diffusion_mask_r_prob
            )
            return {
                'diffusion_loss': diffusion_loss,
                'f_t_star': f_t_star,
                'z': z,
                'beta_z': beta_z
            }
        else:
            f_t = self.diffusion(beta_z, r_t, z, f_prev.detach(), mode='inference')

            self.f_prev_buffer = f_t.detach().clone()
            return {
                'f_t': f_t,
                'z': z,
                'beta_z': beta_z
            }

    def predict(self, x_seq, r_t=None, f_prev=None):
        outputs = self.forward(x_seq, r_t=r_t, f_prev=f_prev, mode='inference')
        f_t = outputs['f_t']
        beta_z = outputs['beta_z']

        if beta_z.dim() == 3:

            pred_r = (beta_z @ f_t.unsqueeze(-1)).squeeze(-1)
        else:

            pred_r = (beta_z @ f_t).squeeze(-1)
            if r_t is not None and r_t.dim() == 2:
                pred_r = pred_r.unsqueeze(1).expand(-1, r_t.shape[1])

        return pred_r, f_t

    def estimate_mutual_information(self, z, y, method='variational'):
        try:
            if method == 'variational':
                return self.stable_variational_mi_estimate(z, y)
            else:
                return stable_mutual_information_estimate(z, y, method)
        except Exception as e:
            print(f"Mutual information estimation error: {e}")
            return torch.tensor(0.1, device=z.device)

    def stable_variational_mi_estimate(self, z, y):
        try:
            y_pred = self.predictor_y(z)
            if y.dim() == 1:
                y = y.unsqueeze(1)
            mse_loss = F.mse_loss(y_pred, y, reduction='mean')
            mi_estimate = 1.0 / (1.0 + torch.clamp(mse_loss, 1e-8, 100.0))
            return torch.clamp(mi_estimate, 0.01, 0.99)
        except Exception as e:
            print(f"Variational MI error: {e}")
            return torch.tensor(0.1, device=z.device)

    def reset_f_prev_buffer(self):
        self.f_prev_buffer = None

    def get_causal_representation(self, x):
        if x.dim() == 3:
            return self.get_z_from_x(x)
        else:
            x = torch.clamp(x, -10, 10)
            x_seq = x.unsqueeze(1)
            return self.get_z_from_x(x_seq)