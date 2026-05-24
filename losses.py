import torch
import torch.nn.functional as F

def stable_kl_divergence(mu, logvar):
    try:
        logvar = torch.clamp(logvar, -10, 10)
        mu = torch.clamp(mu, -5, 5)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl = torch.clamp(kl, 0, 100)
        return kl.mean()
    except Exception as e:
        print(f"KL divergence error: {e}")
        return torch.tensor(0.1, device=mu.device)

def stable_information_bottleneck_loss(z, x, y, lambda_ib=0.1, vae_model=None):
    try:
        if vae_model is not None:
            i_zy = vae_model.estimate_mutual_information(z, y)
            z_var = torch.var(z, dim=0).mean()
            z_var = torch.clamp(z_var, 0.1, 10.0)
            ib_loss = z_var - lambda_ib * torch.clamp(i_zy, 0.01, 0.99)
        else:
            z_var = torch.clamp(torch.var(z, dim=0).mean(), 0.1, 10.0)
            ib_loss = lambda_ib * z_var
        return torch.clamp(ib_loss, -10.0, 10.0)
    except Exception as e:
        print(f"Information bottleneck loss error: {e}")
        return torch.tensor(0.0, device=z.device)