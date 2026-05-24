import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaNetwork(nn.Module):
    def __init__(self, z_dim: int, factor_dim: int, hidden_dim: int = None):
        super().__init__()
        
        self.z_dim = z_dim
        self.factor_dim = factor_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else z_dim
        
        self.net = nn.Sequential(
            nn.Linear(z_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, factor_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, N, z_dim = z.shape
        
        z_flat = z.reshape(B * N, z_dim)                
        
        beta_flat = self.net(z_flat)            
        beta_z = beta_flat.reshape(B, N, self.factor_dim)             
        
        beta_z = torch.tanh(beta_z)
        
        return beta_z

def compute_factor_pricing_returns(beta_z: torch.Tensor, F_t: torch.Tensor) -> torch.Tensor:
    if torch.isnan(beta_z).any() or torch.isnan(F_t).any():
                      
        B, N = beta_z.shape[0], beta_z.shape[1]
        return torch.zeros(B, N, device=beta_z.device, dtype=beta_z.dtype)
    
    beta_z = torch.nan_to_num(beta_z, nan=0.0, posinf=1e6, neginf=-1e6)
    F_t = torch.nan_to_num(F_t, nan=0.0, posinf=1e6, neginf=-1e6)
    
    beta_z = torch.clamp(beta_z, -10.0, 10.0)
    F_t = torch.clamp(F_t, -10.0, 10.0)
    
    F_t_expanded = F_t.unsqueeze(1)             

    y_pred = torch.sum(beta_z * F_t_expanded, dim=-1)          
    
    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    y_pred = torch.clamp(y_pred, -10.0, 10.0)

    return y_pred

def mse_loss(predicted: torch.Tensor, target: torch.Tensor, 
             reduction: str = "mean") -> torch.Tensor:
    mse = F.mse_loss(predicted, target, reduction=reduction)
    
    return mse

def compute_prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mse_loss(predicted, target, reduction="mean")

