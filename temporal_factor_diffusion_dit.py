import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
                        
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) * -self.emb)
        self.emb = nn.Parameter(self.emb, requires_grad=False)        

    def forward(self, time):
        device = time.device
        emb = self.emb.to(device)
        emb = time[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class DiTBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout,
            batch_first=True, vdim=d_model
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
                 
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x

class TemporalDenoiserDiT(nn.Module):
    def __init__(
        self, factor_dim, z_dim, r_feat_dim=32,
        hidden_dim=256, num_layers=2, num_heads=4, dropout=0.1
    ):
        super().__init__()
        self.factor_dim = factor_dim
        self.z_dim = z_dim
        self.r_feat_dim = r_feat_dim

        self.return_encoder = nn.Sequential(
            nn.Linear(1, r_feat_dim),
            nn.LayerNorm(r_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(r_feat_dim, r_feat_dim),
            nn.LayerNorm(r_feat_dim)
        )

        total_cond_dim = z_dim + r_feat_dim + factor_dim
        self.condition_fusion = nn.Sequential(
            nn.Linear(total_cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.time_embed = SinusoidalPositionalEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.input_proj = nn.Linear(factor_dim, hidden_dim)

        self.diT_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, factor_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, f_k, timestep, z, r_feat, f_prev, mask_r_feat=False):
        B = f_k.shape[0]

        if isinstance(mask_r_feat, torch.Tensor):
            mask_r_feat_scalar = mask_r_feat.any()
        else:
            mask_r_feat_scalar = mask_r_feat

        if mask_r_feat_scalar and self.training:
            r_feat = torch.zeros_like(r_feat)

        if f_prev is None:
            f_prev = torch.zeros(B, self.factor_dim, device=f_k.device, dtype=f_k.dtype)
        elif f_prev.shape[0] != B:
            f_prev = torch.zeros(B, self.factor_dim, device=f_k.device, dtype=f_k.dtype)

        cond = torch.cat([z, r_feat, f_prev], dim=-1)
        cond_emb = self.condition_fusion(cond)

        t_emb = self.time_proj(self.time_embed(timestep))

        x = self.input_proj(f_k)
        x = x + cond_emb + t_emb

        x = x.unsqueeze(1)                      
        for block in self.diT_blocks:
            x = block(x)
        x = x.squeeze(1)                   

        f_t_pred = self.output_proj(x)

        return f_t_pred

class BatchedPseudoLabelSolver:
    def __init__(self, method='cholesky', lambda_reg=1e-6):
        self.method = method
        self.lambda_reg = lambda_reg

    def solve(self, beta_z, r_t, f_init=None):
        B, N, K = beta_z.shape
        device = beta_z.device
        dtype = beta_z.dtype

        if self.method == 'cholesky':
            return self._cholesky_solve(beta_z, r_t, device, dtype)
        elif self.method == 'lstsq':
            return self._lstsq_solve(beta_z, r_t, device, dtype)
        else:
            return self._cholesky_solve(beta_z, r_t, device, dtype)

    def _cholesky_solve(self, beta_z, r_t, device, dtype):
        B, N, K = beta_z.shape

        if r_t.dim() == 1:
                                    
            r_t_expanded = r_t.unsqueeze(1).expand(-1, N)
        else:
            r_t_expanded = r_t

        f_t_target = torch.zeros(B, K, device=device, dtype=dtype)

        for b in range(B):
            try:
                                          
                result = torch.linalg.lstsq(beta_z[b], r_t_expanded[b].unsqueeze(-1))
                f_t_target[b] = result.solution.squeeze(-1)
            except:
                            
                f_t_target[b] = torch.zeros(K, device=device, dtype=dtype)

        return f_t_target

    def _lstsq_solve(self, beta_z, r_t, device, dtype):
        B, N, K = beta_z.shape
        f_t_target = torch.zeros(B, K, device=device, dtype=dtype)

        for b in range(B):
            try:
                r_b = r_t[b] if r_t.dim() == 2 else r_t[b:b+1]           
                result = torch.linalg.lstsq(beta_z[b], r_b.unsqueeze(-1))
                f_t_target[b] = result.solution.squeeze(-1)
            except:
                f_t_target[b] = torch.zeros(K, device=device, dtype=dtype)

        return f_t_target

class TemporalFactorDiffusionDiT(nn.Module):
    def __init__(
        self,
        factor_dim: int,
        z_dim: int,
        r_feat_dim: int = 32,
        num_timesteps: int = 1000,
        beta_schedule: str = 'linear',
        denoiser_hidden_dim: int = 256,
        denoiser_num_layers: int = 2,         
        denoiser_num_heads: int = 4,          
        solver_method: str = 'cholesky',
        ddim_eta: float = 0.0,
        ddim_steps: int = 30                   
    ):
        super().__init__()
        self.factor_dim = factor_dim
        self.z_dim = z_dim
        self.r_feat_dim = r_feat_dim
        self.num_timesteps = num_timesteps
        self.ddim_eta = ddim_eta
        self.ddim_steps = ddim_steps

        self.return_encoder = nn.Sequential(
            nn.Linear(1, r_feat_dim),
            nn.LayerNorm(r_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(r_feat_dim, r_feat_dim),
            nn.LayerNorm(r_feat_dim)
        )

        self.denoiser = TemporalDenoiserDiT(
            factor_dim=factor_dim,
            z_dim=z_dim,
            r_feat_dim=r_feat_dim,
            hidden_dim=denoiser_hidden_dim,
            num_layers=denoiser_num_layers,
            num_heads=denoiser_num_heads
        )

        self.solver = BatchedPseudoLabelSolver(method=solver_method)

        self.register_buffer('betas', self._get_beta_schedule(beta_schedule))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))

        self.register_buffer('ddim_alphas', self.alphas_cumprod)
        self.register_buffer('ddim_alphas_prev', self.alphas_cumprod_prev)
        self.register_buffer('ddim_sigmas', torch.sqrt((1 - self.ddim_alphas) / self.ddim_alphas))

    def _get_beta_schedule(self, schedule):
        if schedule == 'linear':
            return torch.linspace(0.0001, 0.02, self.num_timesteps)
        elif schedule == 'cosine':
            s = 0.008
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def _encode_returns(self, r_t):
        if r_t.dim() == 1:
            r_t_expanded = r_t.unsqueeze(-1)
            return self.return_encoder(r_t_expanded)
        else:
            B, N = r_t.shape
            r_t_flat = r_t.reshape(-1, 1)
            r_feat_flat = self.return_encoder(r_t_flat)
            return r_feat_flat.reshape(B, N, self.r_feat_dim)

    def q_sample(self, f_start, t, noise=None):
        """前向扩散：x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε"""
        if noise is None:
            noise = torch.randn_like(f_start)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        return sqrt_alphas_cumprod_t * f_start + sqrt_one_minus_alphas_cumprod_t * noise

    def ddim_sample(self, f_T, z, r_feat, f_prev, mask_r_feat=False):
        B = f_T.shape[0]
        f_k = f_T

        if r_feat is not None and r_feat.dim() == 3:
            r_feat = r_feat.mean(dim=1)

        step_indices = torch.linspace(0, self.num_timesteps - 1, self.ddim_steps, dtype=torch.long)

        for i in range(self.ddim_steps - 1, -1, -1):
            t = step_indices[i].item()
            t_tensor = torch.full((B,), t, device=f_T.device, dtype=torch.long)

            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()

            f_t_pred = self.denoiser(f_k, t_tensor, z, r_feat, f_prev, mask_r_feat=mask_r_feat)

            epsilon_pred = (f_k - sqrt_alpha_t * f_t_pred) / sqrt_one_minus_alpha_t

            pred_f0 = f_t_pred                      

            if i > 0:
                t_prev = step_indices[i - 1].item()
                alpha_t_prev = self.alphas_cumprod[t_prev]

                pred_f0_coeff = alpha_t_prev.sqrt()
                dir_xt = (1 - alpha_t_prev - self.ddim_eta ** 2 * (1 - alpha_t)).sqrt()
                noise = self.ddim_eta * torch.randn_like(f_k) if self.ddim_eta > 0 else torch.zeros_like(f_k)

                f_k = pred_f0_coeff * pred_f0 + dir_xt * epsilon_pred + noise
            else:
                f_k = pred_f0

        return f_k

    def compute_loss(self, beta_z, r_t, z, f_prev, mask_r_prob=0.1):
        B = beta_z.shape[0]
        r_feat = self._encode_returns(r_t)

        if z.dim() == 3:
            z_cond = z.mean(dim=1)
        else:
            z_cond = z

        f_t_target = self.solver.solve(beta_z, r_t)

        t = torch.randint(0, self.num_timesteps, (B,), device=beta_z.device)
        noise = torch.randn_like(f_t_target)
        x_t = self.q_sample(f_t_target, t, noise)

        mask_r_feat = (torch.rand(B, device=beta_z.device) < mask_r_prob)

        f_t_pred = self.denoiser(
            x_t, t, z_cond,
            r_feat.mean(dim=1) if r_feat.dim() == 3 else r_feat,
            f_prev, mask_r_feat=mask_r_feat
        )

        diffusion_loss = F.mse_loss(f_t_pred, f_t_target)

        return diffusion_loss, f_t_pred                        

    def forward(self, beta_z, r_t, z, f_prev=None, mode='train', mask_r_prob=0.1):
        B = beta_z.shape[0]

        if f_prev is None:
            f_prev = torch.zeros(B, self.factor_dim, device=beta_z.device, dtype=beta_z.dtype)
        elif f_prev.shape[0] != B:
            f_prev = torch.zeros(B, self.factor_dim, device=beta_z.device, dtype=beta_z.dtype)

        if z.dim() == 3:
            z_cond = z.mean(dim=1)
        else:
            z_cond = z

        if mode == 'train':
            if r_t is None:
                raise ValueError("r_t is required in training mode")
            loss, f_t_pred = self.compute_loss(beta_z, r_t, z_cond, f_prev, mask_r_prob=mask_r_prob)
            return loss, f_t_pred                            
        else:
            if r_t is None:
                r_feat = torch.zeros(B, self.r_feat_dim, device=beta_z.device, dtype=beta_z.dtype)
            else:
                r_feat = self._encode_returns(r_t)
                if r_feat.dim() == 3:
                    r_feat = r_feat.mean(dim=1)

            f_T = torch.randn(B, self.factor_dim, device=beta_z.device, dtype=beta_z.dtype)
            f_t = self.ddim_sample(f_T, z_cond, r_feat, f_prev)
            return f_t

def create_dit_optimized_diffusion(factor_dim, z_dim, config):
    return TemporalFactorDiffusionDiT(
        factor_dim=factor_dim,
        z_dim=z_dim,
        r_feat_dim=getattr(config, 'diffusion_r_feat_dim', 32),
        num_timesteps=getattr(config, 'diffusion_num_timesteps', 1000),
        beta_schedule=getattr(config, 'diffusion_beta_schedule', 'linear'),
        denoiser_hidden_dim=getattr(config, 'diffusion_denoiser_hidden_dim', 256),
        denoiser_num_layers=2,             
        denoiser_num_heads=4,              
        solver_method='cholesky',              
        ddim_eta=getattr(config, 'diffusion_ddim_eta', 0.0),
        ddim_steps=30                         
    )
