import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = None,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_norm = self.norm1(src)
        attn_out, _ = self.self_attn(
            src_norm, src_norm, src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout(attn_out)
        
        src_norm = self.norm2(src)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout(ff_out)
        
        return src

def generate_causal_mask(seq_len: int, device=None):
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask

class TimeTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = None,
                 dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, batch_size_per_group=None):
        B, T, N, P = x.shape
        
        if batch_size_per_group is not None and B > batch_size_per_group:
            outputs = []
            for b_start in range(0, B, batch_size_per_group):
                b_end = min(b_start + batch_size_per_group, B)
                x_group = x[b_start:b_end]                         
                out_group = self._forward_single(x_group)
                outputs.append(out_group)
            return torch.cat(outputs, dim=0)
        else:
            return self._forward_single(x)
    
    def _forward_single(self, x):
        B, T, N, P = x.shape
        
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()                
        x_flat = x_reshaped.reshape(B * N, T, P)               
        
        x_proj = self.input_proj(x_flat)                     
        
        x_pos = self.pos_encoding(x_proj)                     
        
        causal_mask = generate_causal_mask(T, device=x.device)          
        
        out = x_pos
        for layer in self.layers:
            out = layer(out, src_mask=causal_mask)                     
        
        out = self.output_norm(out)                     
        
        out = out.reshape(B, N, T, self.d_model)                      
        out = out.permute(0, 2, 1, 3).contiguous()                      
        
        return out

class CrossSectionTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = None,
                 dropout: float = 0.1, batch_size_per_time: int = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.batch_size_per_time = batch_size_per_time
        
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()
        
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, T, N, d = x.shape
        
        if self.batch_size_per_time is not None and T > self.batch_size_per_time:
                     
            outputs = []
            for t_start in range(0, T, self.batch_size_per_time):
                t_end = min(t_start + self.batch_size_per_time, T)
                x_time_group = x[:, t_start:t_end]                      
                out_time_group = self._forward_time_group(x_time_group)
                outputs.append(out_time_group)
            return torch.cat(outputs, dim=1)
        else:
            return self._forward_time_group(x)
    
    def _forward_time_group(self, x):
        B, group_T, N, d = x.shape
        
        x_reshaped = x.reshape(B * group_T, N, d)                     
        
        x_proj = self.input_proj(x_reshaped)                           
        
        out = x_proj
        for layer in self.layers:
            out = layer(out)                           
        
        out = self.output_norm(out)                           
        
        out = out.reshape(B, group_T, N, self.d_model)
        
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                         
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TemporalCrossSectionEncoder(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, nhead: int = 4,
                 num_layers_time: int = 2, num_layers_cross: int = 2,
                 dropout: float = 0.1, dim_feedforward: int = None,
                 batch_size_per_group: int = None,
                 batch_size_per_time: int = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        
        self.time_transformer = TimeTransformerEncoder(
            input_dim=input_dim,
            d_model=z_dim,
            nhead=nhead,
            num_layers=num_layers_time,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.cross_transformer = CrossSectionTransformerEncoder(
            input_dim=z_dim,
            d_model=z_dim,
            nhead=nhead,
            num_layers=num_layers_cross,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_size_per_time=batch_size_per_time
        )
        
        self.batch_size_per_group = batch_size_per_group
        
    def forward(self, x):
        B, T, N, P = x.shape
        
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        time_out = self.time_transformer(x, batch_size_per_group=self.batch_size_per_group)
        
        time_out = torch.nan_to_num(time_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        cross_out = self.cross_transformer(time_out)
        
        cross_out = torch.nan_to_num(cross_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        z = cross_out[:, -1, :, :]           
        
        return z

