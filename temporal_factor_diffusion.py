import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class TemporalDenoiser(nn.Module):
    def __init__(self, factor_dim, z_dim, r_feat_dim=32, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.factor_dim = factor_dim
        self.z_dim = z_dim
        self.r_feat_dim = r_feat_dim
        self.hidden_dim = hidden_dim
        
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
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        self.time_embed = SinusoidalPositionalEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_proj = nn.Sequential(
            nn.Linear(factor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, factor_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
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
                               
            f_prev = torch.zeros(B, f_prev.shape[1], device=f_prev.device, dtype=f_prev.dtype)
        
        cond = torch.cat([z, r_feat, f_prev], dim=-1)                               
        cond_emb = self.condition_fusion(cond)                   
        
        t_emb = self.time_embed(timestep)                   
        t_emb = self.time_proj(t_emb)                   
        
        x = self.input_proj(f_k)                   
        
        x = x + cond_emb + t_emb                   
        
        x = x.unsqueeze(1)                      
        x = self.transformer(x)                      
        x = x.squeeze(1)                   
        
        epsilon_pred = self.output_proj(x)          
        
        return epsilon_pred

class PseudoLabelSolver:
    def __init__(self, method='least_squares', max_iter=100, lr=0.01, tol=1e-6):
        self.method = method
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
    
    def solve(self, beta_z, r_t, f_init=None):
        B, N, K = beta_z.shape
        
        if self.method == 'least_squares':
            return self._least_squares_solve(beta_z, r_t)
        elif self.method == 'gradient_descent':
            return self._gradient_descent_solve(beta_z, r_t, f_init)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _least_squares_solve(self, beta_z, r_t):
        B, N, K = beta_z.shape
        f_t_star = torch.zeros(B, K, device=beta_z.device, dtype=beta_z.dtype)
        
        for b in range(B):
            beta_b = beta_z[b]          
            r_b = r_t[b]       
            
            lambda_reg = 1e-6
            beta_T_beta = beta_b.T @ beta_b          
            reg_matrix = lambda_reg * torch.eye(K, device=beta_b.device, dtype=beta_b.dtype)
            beta_T_r = beta_b.T @ r_b       
            
            try:
                         
                f_b = torch.linalg.solve(beta_T_beta + reg_matrix, beta_T_r)       
                f_t_star[b] = f_b
            except:
                             
                try:
                    f_b = torch.linalg.pinv(beta_T_beta + reg_matrix) @ beta_T_r
                    f_t_star[b] = f_b
                except:
                                  
                    f_t_star[b] = torch.zeros(K, device=beta_b.device, dtype=beta_b.dtype)
        
        return f_t_star
    
    def _gradient_descent_solve(self, beta_z, r_t, f_init=None):
        B, N, K = beta_z.shape
        
        if f_init is None:
            f = torch.zeros(B, K, device=beta_z.device, dtype=beta_z.dtype, requires_grad=True)
        else:
            f = f_init.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([f], lr=self.lr)
        
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            
            r_pred = (beta_z @ f.unsqueeze(1)).squeeze(1)          
            
            loss = F.mse_loss(r_pred, r_t)
            
            if loss.item() < self.tol:
                break
            
            loss.backward()
            optimizer.step()
        
        return f.detach()

class TemporalFactorDiffusion(nn.Module):
    def __init__(
        self,
        factor_dim: int,
        z_dim: int,
        r_feat_dim: int = 32,
        num_timesteps: int = 1000,
        beta_schedule: str = 'linear',
        denoiser_hidden_dim: int = 256,
        denoiser_num_layers: int = 4,
        denoiser_num_heads: int = 8,
        solver_method: str = 'least_squares',
        ddim_eta: float = 0.0,
        ddim_steps: int = 50
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
        
        self.denoiser = TemporalDenoiser(
            factor_dim=factor_dim,
            z_dim=z_dim,
            r_feat_dim=r_feat_dim,
            hidden_dim=denoiser_hidden_dim,
            num_layers=denoiser_num_layers,
            num_heads=denoiser_num_heads
        )
        
        self.solver = PseudoLabelSolver(method=solver_method)
        
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
            r_feat = self.return_encoder(r_t_expanded)                   
            return r_feat
        else:
                                             
            B, N = r_t.shape
            r_t_flat = r_t.reshape(-1, 1)            
            r_feat_flat = self.return_encoder(r_t_flat)                     
            r_feat = r_feat_flat.reshape(B, N, self.r_feat_dim)                      
            return r_feat
    
    def q_sample(self, f_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(f_start)
        
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().unsqueeze(-1)          
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)          
        
        f_t = sqrt_alphas_cumprod_t * f_start + sqrt_one_minus_alphas_cumprod_t * noise
        return f_t
    
    def p_sample(self, f_k, t, z, r_feat, f_prev, mask_r_feat=False):
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(f_k.shape[0])
        
        epsilon_pred = self.denoiser(f_k, t, z, r_feat, f_prev, mask_r_feat=mask_r_feat)
        
        alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
        pred_f0 = (f_k - sqrt_one_minus_alpha_t * epsilon_pred) / sqrt_alpha_t
        
        if t[0] > 0:
            alpha_t_prev = self.alphas_cumprod[t - 1].unsqueeze(-1)
            beta_t = self.betas[t].unsqueeze(-1)
            
            pred_f0_coeff = (alpha_t_prev.sqrt() * beta_t) / (1 - alpha_t)
            f_k_coeff = (self.alphas[t].sqrt() * (1 - alpha_t_prev)) / (1 - alpha_t)
            
            posterior_mean = pred_f0_coeff * pred_f0 + f_k_coeff * f_k
            posterior_variance = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
            
            noise = torch.randn_like(f_k)
            f_prev_sample = posterior_mean + posterior_variance.sqrt() * noise
        else:
            f_prev_sample = pred_f0
        
        return f_prev_sample
    
    def ddim_sample(self, f_T, z, r_feat, f_prev, mask_r_feat=False):
        B = f_T.shape[0]
        f_k = f_T
        
        step_indices = torch.linspace(0, self.num_timesteps - 1, self.ddim_steps, dtype=torch.long)
        
        for i in range(self.ddim_steps - 1, -1, -1):
            t = step_indices[i].item()
            t_tensor = torch.full((B,), t, device=f_T.device, dtype=torch.long)
            
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
            
            epsilon_pred = self.denoiser(f_k, t_tensor, z, r_feat, f_prev, mask_r_feat=mask_r_feat)
            pred_f0 = (f_k - sqrt_one_minus_alpha_t * epsilon_pred) / sqrt_alpha_t
            
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
        
        f_t_star = self.solver.solve(beta_z, r_t)          
        
        t = torch.randint(0, self.num_timesteps, (B,), device=beta_z.device)
        
        epsilon = torch.randn_like(f_t_star)          
        
        f_k = self.q_sample(f_t_star, t, epsilon)          
        
        mask_r_feat = (torch.rand(B, device=beta_z.device) < mask_r_prob)
        epsilon_pred = self.denoiser(f_k, t, z_cond, r_feat.mean(dim=1) if r_feat.dim() == 3 else r_feat, f_prev, mask_r_feat=mask_r_feat)
        
        loss = F.mse_loss(epsilon_pred, epsilon)
        
        return loss, f_t_star
    
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
            loss, f_t_star = self.compute_loss(beta_z, r_t, z_cond, f_prev, mask_r_prob=mask_r_prob)
            return loss, f_t_star
        else:
                  
            if r_t is not None:
                r_feat = self._encode_returns(r_t)                      
                r_feat_cond = r_feat.mean(dim=1)                   
            else:
                                                  
                r_feat_cond = torch.zeros(B, self.r_feat_dim, device=beta_z.device, dtype=beta_z.dtype)
            
            f_T = torch.randn(B, self.factor_dim, device=beta_z.device, dtype=beta_z.dtype)
            f_t = self.ddim_sample(f_T, z_cond, r_feat_cond, f_prev, mask_r_feat=True)
            
            return f_t

