import torch
from training import complete_stable_training_pipeline

if __name__ == "__main__":

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    use_diffusion = True
    trained_model, test_loader, config, results = complete_stable_training_pipeline(use_diffusion=use_diffusion)

    if trained_model is not None:
        print("\n" + "=" * 50)
        print("Program executed successfully!")
        print("=" * 50)
        if use_diffusion:
            print(f"Model file: stable_cari_diffusion_transformer_complete.pth")
        else:
            print(f"Model file: stable_cari_vae_mi_beta_transformer_complete.pth")
        print("=" * 50)
    else:
        print("\nProgram execution failed, please check the error message")