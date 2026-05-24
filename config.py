import torch
import os

def get_project_root():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    return current_dir

class StableConfig:
    def __init__(self, data_dir=None):
        project_root = get_project_root()
        
        if data_dir is None:
            data_dir = project_root
        
        self.price_and_factor_path = os.path.join(data_dir, "price_and_factor.npy")                  
        self.return_path = os.path.join(data_dir, "return.npy")                
        self.time_stamp_path = os.path.join(data_dir, "time_stamp.npy")           
        
        self.seed = 42
        self.batch_size = 256                        
        self.vae_epochs = 200                    
        self.learning_rate = 5e-4                    
        self.z_dim = 64                   
        self.factor_dim = 24                   
        self.vae_hidden_dims = [256, 128]                   

        self.impute_method = 'mean'
        self.normalize_method = 'standard'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.info_bottleneck_lambda = 0.00005                        
        self.prediction_lambda = 8.0                           
        self.recon_lambda = 0.05                     
        self.kl_lambda = 0.05                    

        self.mi_estimation_method = 'variational'

        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2
        self.seq_length = 15                   

        self.transformer_nhead = 8           
        self.transformer_layers_time = 3          
        self.transformer_layers_cross = 3          
        self.transformer_dropout = 0.2                   

        self.gradient_clip = 0.8                 
        self.weight_decay = 1e-4                

        self.diffusion_num_timesteps = 800                          
        self.diffusion_beta_schedule = 'cosine'                     
        self.diffusion_denoiser_hidden_dim = 512                   
        self.diffusion_denoiser_num_layers = 3                
        self.diffusion_denoiser_num_heads = 8           
        self.diffusion_solver_method = 'least_squares'             
        self.diffusion_ddim_eta = 0.0            
        self.diffusion_ddim_steps = 25                   
        self.diffusion_lambda = 2.0                   

        self.diffusion_r_feat_dim = 64            
        self.diffusion_mask_r_prob = 0.05                 

        self.optimizer_type = 'adamw'              
        self.scheduler_type = 'cosine'                 
        self.warmup_epochs = 10           

        self.ic_optimization_focus = True            
        self.early_stopping_metric = 'ic'                               