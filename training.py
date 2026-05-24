import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses import (
    stable_kl_divergence,
    stable_information_bottleneck_loss
)
from data_preprocessing import StableStockDataset
from models import StableCaRIVAE, StableCaRIVAEWithDiffusion
from utils import set_seed

import numpy
from torch.serialization import add_safe_globals

add_safe_globals([numpy._core.multiarray.scalar])


def stable_train_cari_vae(vae, train_loader, val_loader, config):
    device = config.device
    vae.to(device)

    if hasattr(config, 'optimizer_type') and config.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            vae.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
    else:
        optimizer = torch.optim.Adam(
            vae.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    if hasattr(config, 'scheduler_type') and config.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.vae_epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    train_losses = []
    val_losses = []
    val_ics = []
    best_val_loss = float('inf')
    best_val_ic = -float('inf')

    patience = 30
    patience_counter = 0
    warmup_epochs = getattr(config, 'warmup_epochs', 10)

    for epoch in range(config.vae_epochs):
        vae.train()
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_ib_loss = 0.0
        n_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.vae_epochs}")

        for batch_idx, (x_batch, y_batch, r_hist_batch) in enumerate(progress_bar):
            try:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                r_hist_batch = r_hist_batch.to(device)
                x_flat = x_batch[:, -1, :]
                x_flat = torch.clamp(x_flat, -10, 10)

                optimizer.zero_grad()

                recon, mu, logvar = vae(
                    r=r_hist_batch, x=x_flat, sample_z=True
                )

                z = vae.get_z_from_x(x_batch)

                recon_loss = F.mse_loss(recon, x_flat, reduction='mean')
                kl_loss = stable_kl_divergence(mu, logvar)

                z_for_beta = z.unsqueeze(1)
                beta_z_factor = vae.beta_network(z_for_beta)
                beta_z_factor = beta_z_factor.squeeze(1)

                beta_z = vae.beta_projection(beta_z_factor)

                ft_true = x_flat
                q_pred = (beta_z * ft_true).sum(dim=1)

                pred_loss = F.mse_loss(q_pred, y_batch, reduction='mean')

                ib_loss = stable_information_bottleneck_loss(
                    z, x_flat, y_batch, config.info_bottleneck_lambda, vae
                )

                recon_weight = getattr(config, 'recon_lambda', 1.0)
                kl_weight = getattr(config, 'kl_lambda', 1.0)

                total_loss = (
                        recon_weight * recon_loss +
                        kl_weight * kl_loss +
                        config.prediction_lambda * pred_loss +
                        ib_loss
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), config.gradient_clip)
                optimizer.step()

                batch_size = x_batch.size(0)
                epoch_total_loss += total_loss.item() * batch_size
                epoch_recon_loss += recon_loss.item() * batch_size
                epoch_kl_loss += kl_loss.item() * batch_size
                epoch_pred_loss += pred_loss.item() * batch_size
                epoch_ib_loss += ib_loss.item() * batch_size
                n_batches += batch_size

                if batch_idx % 100 == 0:
                    progress_bar.set_postfix({
                        'Loss': f'{total_loss.item():.6f}',
                        'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                    })

            except Exception as e:
                print(f"Batch {batch_idx} training error: {e}")
                import traceback
                traceback.print_exc()
                continue

        vae.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        n_val_batches = 0

        with torch.no_grad():
            for x_val, y_val, r_hist_val in val_loader:
                try:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    r_hist_val = r_hist_val.to(device)
                    x_val_flat = x_val[:, -1, :]
                    x_val_flat = torch.clamp(x_val_flat, -10, 10)

                    recon_val, mu_val, logvar_val = vae(
                        r=r_hist_val, x=x_val_flat, sample_z=False
                    )

                    recon_loss_val = F.mse_loss(recon_val, x_val_flat)
                    kl_loss_val = stable_kl_divergence(mu_val, logvar_val)
                    val_batch_loss = (recon_loss_val + kl_loss_val).item()

                    val_loss += val_batch_loss * x_val.size(0)
                    n_val_batches += x_val.size(0)

                    z_val = vae.get_z_from_x(x_val)
                    z_for_beta_val = z_val.unsqueeze(1)
                    beta_z_factor_val = vae.beta_network(z_for_beta_val).squeeze(1)
                    beta_z_val = vae.beta_projection(beta_z_factor_val)

                    q_pred_val = (beta_z_val * recon_val).sum(dim=1)

                    val_predictions.extend(q_pred_val.cpu().numpy())
                    val_targets.extend(y_val.cpu().numpy())

                except Exception as e:
                    print(f"Validation batch error: {e}")
                    continue

        if len(val_predictions) > 0 and len(val_targets) > 0:
            import numpy as np
            from scipy.stats import pearsonr

            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)

            mask = np.isfinite(val_predictions) & np.isfinite(val_targets)
            if mask.sum() > 10:
                val_predictions_clean = val_predictions[mask]
                val_targets_clean = val_targets[mask]
                ic_value, _ = pearsonr(val_predictions_clean, val_targets_clean)
                if np.isnan(ic_value):
                    ic_value = 0.0
            else:
                ic_value = 0.0
        else:
            ic_value = 0.0

        val_ics.append(ic_value)

        avg_train_loss = epoch_total_loss / n_batches if n_batches > 0 else float('inf')
        avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else float('inf')

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if hasattr(config, 'scheduler_type') and config.scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_val_loss)

        current_ic = abs(ic_value)

        if epoch < warmup_epochs:
            patience_counter = 0
        else:

            if current_ic > best_val_ic:
                best_val_ic = current_ic
                best_val_loss = avg_val_loss
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': vae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_ic': current_ic,
                }, 'best_stable_cari_vae_mi_beta_transformer.pth')

                print(f"New best IC: {current_ic:.6f} (absolute value)")
            else:
                patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            recon_weight = getattr(config, 'recon_lambda', 1.0)
            kl_weight = getattr(config, 'kl_lambda', 1.0)

            print(f"\n=== Epoch {epoch + 1} ===")
            print(f"Training loss: {avg_train_loss:.6f}")
            print(f"Validation loss: {avg_val_loss:.6f}")
            print(f"Validation IC: {ic_value:.6f} (absolute value: {current_ic:.6f})")
            print(f"Reconstruction loss: {epoch_recon_loss / n_batches:.6f} (weight: {recon_weight})")
            print(f"KL loss: {epoch_kl_loss / n_batches:.6f} (weight: {kl_weight})")
            print(f"Prediction loss: {epoch_pred_loss / n_batches:.6f} (weight: {config.prediction_lambda})")
            print(
                f"Information bottleneck loss: {epoch_ib_loss / n_batches:.6f} (weight: {config.info_bottleneck_lambda})")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.8f}")
            print(f"Best validation IC: {best_val_ic:.6f}")
            print(f"Early stopping counter: {patience_counter}/{patience}")

            avg_recon = epoch_recon_loss / n_batches
            avg_pred = epoch_pred_loss / n_batches
            if avg_recon > 0 and avg_pred > 0:
                ratio = avg_recon / avg_pred
                if ratio > 5:
                    print(f"Warning: Reconstruction loss is {ratio:.1f} times prediction loss")
                elif ratio < 0.2:
                    print(f"Success: Prediction loss dominates training, beneficial for IC optimization")

        if patience_counter >= patience and epoch >= warmup_epochs:
            print(f"\nEarly stopping at epoch {epoch + 1}, best IC: {best_val_ic:.6f}")
            break

    try:
        checkpoint = torch.load('best_stable_cari_vae_mi_beta_transformer.pth', weights_only=False)
    except:

        checkpoint = torch.load('best_stable_cari_vae_mi_beta_transformer.pth',
                                map_location=device, weights_only=False)

    vae.load_state_dict(checkpoint['model_state_dict'])

    return vae, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ics': val_ics,
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': best_val_loss,
        'best_val_ic': best_val_ic
    }


def stable_train_diffusion(model, train_loader, val_loader, config):
    device = config.device
    model.to(device)

    if hasattr(config, 'optimizer_type') and config.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    if hasattr(config, 'scheduler_type') and config.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.vae_epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    train_losses = []
    val_losses = []
    val_ics = []
    best_val_loss = float('inf')
    best_val_ic = -float('inf')
    patience = 25
    patience_counter = 0
    warmup_epochs = getattr(config, 'warmup_epochs', 10)

    for epoch in range(config.vae_epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_diffusion_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_ib_loss = 0.0
        n_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.vae_epochs}")

        for batch_idx, batch_data in enumerate(progress_bar):
            try:

                if len(batch_data) == 6:

                    x_t, r_t, r_hist_t, x_t_prev, r_t_prev, r_hist_t_prev = batch_data
                    x_t = x_t.to(device)
                    r_t = r_t.to(device)
                    x_flat = x_t[:, -1, :]
                    x_flat = torch.clamp(x_flat, -10, 10)

                    f_prev = None
                    if x_t_prev is not None:
                        x_t_prev = x_t_prev.to(device)
                        r_t_prev = r_t_prev.to(device) if r_t_prev is not None else None

                        batch_size = x_t_prev.shape[0]
                        is_cold_start_mask = (x_t_prev.abs().sum(dim=(1, 2)) < 1e-6)

                        if not is_cold_start_mask.all():
                            with torch.no_grad():
                                outputs_prev = model(x_t_prev, r_t=r_t_prev, mode='inference')
                                f_prev_generated = outputs_prev['f_t']

                            f_prev = torch.randn(batch_size, f_prev_generated.shape[1],
                                                 device=f_prev_generated.device,
                                                 dtype=f_prev_generated.dtype) * 0.1
                            f_prev[~is_cold_start_mask] = f_prev_generated[~is_cold_start_mask]
                else:
                    x_batch, y_batch, r_hist_batch = batch_data
                    x_t = x_batch.to(device)
                    r_t = y_batch.to(device)
                    x_flat = x_t[:, -1, :]
                    x_flat = torch.clamp(x_flat, -10, 10)
                    f_prev = None

                optimizer.zero_grad()

                outputs = model(x_t, r_t=r_t, f_prev=f_prev, mode='train')
                diffusion_loss = outputs['diffusion_loss']
                f_t_star = outputs['f_t_star']
                z = outputs['z']
                beta_z = outputs['beta_z']

                if beta_z.dim() == 3:
                    pred_r = torch.einsum('bnk,bk->bn', beta_z, f_t_star)
                    pred_r = pred_r * model.pred_scale

                    if r_t.dim() == 1:
                        if pred_r.shape[1] == 1:
                            pred_r = pred_r.squeeze(1)
                            r_t_expanded = r_t
                        else:
                            r_t_expanded = r_t.unsqueeze(1).expand(-1, pred_r.shape[1])
                    else:
                        r_t_expanded = r_t
                else:
                    pred_r = (beta_z * f_t_star).sum(dim=1)
                    pred_r = pred_r * model.pred_scale

                    if r_t.dim() == 2:
                        r_t_expanded = r_t.mean(dim=1)
                    else:
                        r_t_expanded = r_t

                assert pred_r.shape == r_t_expanded.shape
                pred_loss = F.mse_loss(pred_r, r_t_expanded, reduction='mean')

                ib_loss = stable_information_bottleneck_loss(
                    z, x_flat, r_t_expanded, config.info_bottleneck_lambda, model
                )

                total_loss = (
                        config.diffusion_lambda * diffusion_loss +
                        config.prediction_lambda * pred_loss +
                        ib_loss
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()

                batch_size = x_t.size(0)
                epoch_total_loss += total_loss.item() * batch_size
                epoch_diffusion_loss += diffusion_loss.item() * batch_size
                epoch_pred_loss += pred_loss.item() * batch_size
                epoch_ib_loss += ib_loss.item() * batch_size
                n_batches += batch_size

                if batch_idx % 100 == 0:
                    progress_bar.set_postfix({
                        'Loss': f'{total_loss.item():.6f}',
                        'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                    })

            except Exception as e:
                print(f"Batch {batch_idx} training error: {e}")
                import traceback
                traceback.print_exc()
                continue

        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        n_val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                try:
                    if len(batch_data) == 6:
                        x_t, r_t, r_hist_t, x_t_prev, r_t_prev, r_hist_t_prev = batch_data
                        x_t = x_t.to(device)
                        r_t = r_t.to(device)

                        f_prev = None
                        if x_t_prev is not None:
                            x_t_prev = x_t_prev.to(device)
                            r_t_prev = r_t_prev.to(device) if r_t_prev is not None else None

                            batch_size = x_t_prev.shape[0]
                            is_cold_start_mask = (x_t_prev.abs().sum(dim=(1, 2)) < 1e-6)

                            if not is_cold_start_mask.all():
                                with torch.no_grad():
                                    outputs_prev = model(x_t_prev, r_t=r_t_prev, mode='inference')
                                    f_prev_generated = outputs_prev['f_t']

                                f_prev = torch.randn(batch_size, f_prev_generated.shape[1],
                                                     device=f_prev_generated.device,
                                                     dtype=f_prev_generated.dtype) * 0.1
                                f_prev[~is_cold_start_mask] = f_prev_generated[~is_cold_start_mask]
                    else:
                        x_batch, y_batch, r_hist_batch = batch_data
                        x_t = x_batch.to(device)
                        r_t = y_batch.to(device)
                        f_prev = None

                    outputs = model(x_t, r_t=r_t, f_prev=f_prev, mode='train')
                    val_batch_loss = outputs['diffusion_loss'].item()
                    val_loss += val_batch_loss * x_t.size(0)

                    pred_r, _ = model.predict(x_t, r_t=r_t, f_prev=f_prev)
                    val_predictions.extend(pred_r.cpu().numpy().flatten())
                    val_targets.extend(r_t.cpu().numpy().flatten())

                    n_val_batches += x_t.size(0)

                except Exception as e:
                    print(f"Validation batch error: {e}")
                    continue

        if len(val_predictions) > 0 and len(val_targets) > 0:
            import numpy as np
            from scipy.stats import pearsonr

            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)

            mask = np.isfinite(val_predictions) & np.isfinite(val_targets)
            if mask.sum() > 10:
                val_predictions_clean = val_predictions[mask]
                val_targets_clean = val_targets[mask]
                ic_value, _ = pearsonr(val_predictions_clean, val_targets_clean)
                if np.isnan(ic_value):
                    ic_value = 0.0
            else:
                ic_value = 0.0
        else:
            ic_value = 0.0

        val_ics.append(ic_value)

        avg_train_loss = epoch_total_loss / n_batches if n_batches > 0 else float('inf')
        avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else float('inf')

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if hasattr(config, 'scheduler_type') and config.scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_val_loss)

        current_ic = abs(ic_value)

        if epoch < warmup_epochs:
            patience_counter = 0
        else:
            if current_ic > best_val_ic:
                best_val_ic = current_ic
                best_val_loss = avg_val_loss
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_ic': current_ic,
                }, 'best_stable_cari_diffusion_transformer.pth')

                print(f"New best IC: {current_ic:.6f} (absolute value)")
            else:
                patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n=== Epoch {epoch + 1} ===")
            print(f"Training loss: {avg_train_loss:.6f}")
            print(f"Validation loss: {avg_val_loss:.6f}")
            print(f"Validation IC: {ic_value:.6f} (absolute value: {current_ic:.6f})")
            print(f"Diffusion loss: {epoch_diffusion_loss / n_batches:.6f} (weight: {config.diffusion_lambda})")
            print(f"Prediction loss: {epoch_pred_loss / n_batches:.6f} (weight: {config.prediction_lambda})")
            print(
                f"Information bottleneck loss: {epoch_ib_loss / n_batches:.6f} (weight: {config.info_bottleneck_lambda})")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.8f}")
            print(f"Best validation IC: {best_val_ic:.6f}")
            print(f"Early stopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience and epoch >= warmup_epochs:
            print(f"\nEarly stopping at epoch {epoch + 1}, best IC: {best_val_ic:.6f}")
            break

    try:
        checkpoint = torch.load('best_stable_cari_diffusion_transformer.pth', weights_only=False)
    except:
        checkpoint = torch.load('best_stable_cari_diffusion_transformer.pth',
                                map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ics': val_ics,
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': best_val_loss,
        'best_val_ic': best_val_ic
    }


def complete_stable_training_pipeline(use_diffusion=True):
    from config import StableConfig
    from data_preprocessing import StableDataPreprocessor

    config = StableConfig()
    set_seed(config.seed)

    if use_diffusion:
        print("=== Optimized CaRI Diffusion Training Pipeline (IC Maximization Version) ===")
    else:
        print("=== Optimized CaRI VAE Training Pipeline (IC Maximization Version) ===")
    print(f"Device: {config.device}")
    print(f"Random seed: {config.seed}")
    print(f"IC optimization mode: {getattr(config, 'ic_optimization_focus', False)}")

    try:

        print("\n1. Data preprocessing...")
        preprocessor = StableDataPreprocessor(config)
        processed_data = preprocessor.preprocess_all_data()

        if len(processed_data['train'][0]) == 0:
            print("Error: No available training data")
            return None, None, None, None

        train_dataset = StableStockDataset(
            processed_data['train'][0], processed_data['train'][1], processed_data['train'][2],
            processed_data['train'][3] if len(processed_data['train']) > 3 else None,
            processed_data['train'][4] if len(processed_data['train']) > 4 else None,
            use_temporal_pairs=use_diffusion
        )
        val_dataset = StableStockDataset(
            processed_data['val'][0], processed_data['val'][1], processed_data['val'][2],
            processed_data['val'][3] if len(processed_data['val']) > 3 else None,
            processed_data['val'][4] if len(processed_data['val']) > 4 else None,
            use_temporal_pairs=use_diffusion
        )
        test_dataset = StableStockDataset(
            processed_data['test'][0], processed_data['test'][1], processed_data['test'][2],
            processed_data['test'][3] if len(processed_data['test']) > 3 else None,
            processed_data['test'][4] if len(processed_data['test']) > 4 else None,
            use_temporal_pairs=False
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        input_dim = len(processed_data['feature_columns'])

        if use_diffusion:
            model = StableCaRIVAEWithDiffusion(
                input_dim, config.vae_hidden_dims, config.z_dim, config.factor_dim, config
            )
        else:
            model = StableCaRIVAE(input_dim, config.vae_hidden_dims, config.z_dim, config.factor_dim, config,
                                  use_r_and_x=True)

        print(f"\n2. Model initialization completed")
        print(f"Model type: {'Diffusion' if use_diffusion else 'VAE'}")
        print(f"Input dimension: {input_dim}")
        print(f"Latent dimension: {config.z_dim}")
        print(f"Factor dimension: {config.factor_dim}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        print("\n3. Starting training...")
        if use_diffusion:
            trained_model, training_history = stable_train_diffusion(model, train_loader, val_loader, config)
        else:
            trained_model, training_history = stable_train_cari_vae(model, train_loader, val_loader, config)

        print("\n4. Model evaluation...")
        from evaluation import evaluate_stable_model, test_model
        test_loss = evaluate_stable_model(trained_model, test_loader, config)
        print(f"Test loss: {test_loss:.6f}")

        print("\n4.5. Testing model performance...")
        test_metrics = test_model(trained_model, test_loader, config)

        print("\n5. Saving final results...")
        final_results = {
            'training_history': training_history,
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'config': config.__dict__,
            'feature_columns': processed_data['feature_columns']
        }

        if use_diffusion:
            model_filename = 'stable_cari_diffusion_transformer_complete.pth'
        else:
            model_filename = 'stable_cari_vae_mi_beta_transformer_complete.pth'

        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'results': final_results
        }, model_filename)

        print("\n=== Training completed ===")
        print(f"Best validation IC: {training_history.get('best_val_ic', 0):.6f}")
        print(f"Test loss: {test_loss:.6f}")
        if test_metrics:
            print(f"Test IC: {test_metrics.get('ic', 0):.6f}")

        return trained_model, test_loader, config, test_metrics

    except Exception as e:
        print(f"Training pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None