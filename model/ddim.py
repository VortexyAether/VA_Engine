import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from Diffusion.utils.diffusion_utils import make_beta_schedule, setup_logging

from tqdm import tqdm

class DDIM(nn.Module):
    def __init__(self,
                 model,
                 ddpm_timestep=1000,
                 ddim_timestep=100,
                 eta=0.0,
                 lr=1e-4,
                 input_parameter = 7,
                 contour_size = 100,
                 load_model_path=None,
                 only_test=False):
        super(DDIM, self).__init__()
        self.contour_size = contour_size
#        self.condition_fc = nn.Linear(input_parameter, contour_size**2)

        self.condition_fc = nn.Sequential(
            nn.Linear(input_parameter, contour_size ** 2),
            nn.Unflatten(dim=1, unflattened_size=(1, contour_size, contour_size))
        )

        self.model = model
        self.path = load_model_path
        self.only_test = only_test
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.loss_fn = nn.MSELoss()
        self.ssim_weight = 0# 0.5  # Weight for SSIM loss

        # SSIM metric (requires torchmetrics: pip install torchmetrics)
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        except ImportError:
            print("Warning: torchmetrics not installed. Install with: pip install torchmetrics")
            self.ssim_metric = None

        self.ddpm_timestep = ddpm_timestep
        self.ddim_timestep = ddim_timestep
        self.eta = eta

        self.ddpm_beta_schedule = make_beta_schedule(ddpm_timestep)
        self.ddim_timesteps = np.asarray(list(range(0, ddpm_timestep, ddpm_timestep // ddim_timestep)))
        self.ddpm_alphas = 1 - self.ddpm_beta_schedule
        self.ddpm_alphas_cumprod = torch.cumprod(self.ddpm_alphas, axis=0)

        self.alphas = self.ddpm_alphas_cumprod[self.ddim_timesteps]
        self.alphas_prev = np.asarray(
            [self.ddpm_alphas_cumprod[0]] + self.ddpm_alphas_cumprod[self.ddim_timesteps[:-1]].tolist())
        self.sigmas = eta * np.sqrt((1 - self.alphas_prev) / (1 - self.alphas) * (1 - self.alphas / self.alphas_prev))

        self.register_buffer('alphas_cumprod', self.ddpm_alphas_cumprod)
        self.register_buffer('sigmas', self.sigmas)
        self.register_buffer('alphas', self.alphas)
        self.register_buffer('alphas_prev', self.alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - self.alphas.cpu()))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def _compute_combined_loss(self, pred, target):
        """Compute MSE + SSIM loss"""
        mse_loss = F.mse_loss(pred, target, reduction="mean")

        if self.ssim_metric is not None:
            # Normalize to [0, 1] for SSIM if data is not already normalized
            pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)

            ssim_val = self.ssim_metric(pred_norm, target_norm)
            ssim_loss = 1.0 - ssim_val  # Convert to loss (lower is better)
            total_loss = mse_loss + self.ssim_weight * ssim_loss
        else:
            ssim_loss = torch.tensor(0.0, device=pred.device)
            total_loss = mse_loss

        return total_loss, mse_loss, ssim_loss

    def load_saved_model(self, path):
        cp = torch.load(path)
        self.model.load_state_dict(cp)

    def make_noise(self, data):
        noise = torch.randn_like(data)
        return noise

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = self.make_noise(x_0)

        alphas = self.alphas

        alpha_hat_t = alphas[t].view(-1, 1, 1, 1)
        sqrt_alpha_hat_t = torch.sqrt(alpha_hat_t)
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - alpha_hat_t)

        x_t = sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise
        return x_t

    @torch.no_grad()
#    def sampling(self, x_T, cond, image):
#    def sampling(self, image, cond):
    def sampling(self, cond):
        time_range = np.flip(self.ddim_timesteps)
        total_step = time_range.shape[0]
        
        # During training: inputs are C_in channels, targets are C_out channels
        # During sampling: we need to create target with C_out channels
        # Extract the number of output channels from the model's final layer
        if hasattr(self.model, 'final') and hasattr(self.model.final[-1], 'out_channels'):
            target_channels = self.model.final[-1].out_channels
        else:
            # Fallback: assume same as condition channels but this might not be correct
            target_channels = cond.shape[1]
        
        x_T = torch.randn(cond.shape[0], target_channels, cond.shape[2], cond.shape[3], device=cond.device)
        
        print(f"cond.shape: {cond.shape}, x_T.shape: {x_T.shape}")
        for i, step in enumerate(time_range):
            index = total_step - i - 1
            x_T = self.p_sample(x_T, index, cond)

        return x_T

    @torch.no_grad()
    def p_sample(self, x, index, cond):
        b = x.shape[0]
        t = torch.full((x.shape[0],), index, device=x.device, dtype=torch.long)
        x = x.to("cuda")

        model_input = torch.cat([x, cond], dim=1)
        e_t = self.model(model_input, t)

        alphas = self.alphas
        alphas_prev = self.alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.sigmas

        a_t = torch.full((b, 1, 1, 1), alphas[index], device="cuda")
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device="cuda")
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device="cuda")
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device="cuda")

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if index == 0:
            return pred_x0
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t

        noise = torch.randn_like(x, device="cuda")
        noise = sigma_t * noise
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev

    def train_step(self, x_0, t, cond):
        noise = self.make_noise(x_0)

        noisy_x = self.q_sample(x_0, t, noise=noise)
        model_input = torch.cat([noisy_x, cond], dim=1).float()
        predicted_noise = self.model(model_input, t)

        # Compute combined loss (MSE + SSIM on noise prediction)
        total_loss, mse_loss, ssim_loss = self._compute_combined_loss(predicted_noise, noise)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {'total': total_loss.item(), 'mse': mse_loss.item(), 'ssim': ssim_loss.item()}


    def validation_step(self, x_0, t, cond):
        noise = self.make_noise(x_0)

        with torch.no_grad():
            #cond = self.condition_fc(cond)
            noisy_x = self.q_sample(x_0, t, noise=noise)
            model_input = torch.cat([noisy_x, cond], dim=1).float()
            predicted_noise = self.model(model_input, t)

            # Compute combined loss (MSE + SSIM on noise prediction)
            total_loss, mse_loss, ssim_loss = self._compute_combined_loss(predicted_noise, noise)

        return {'total': total_loss.item(), 'mse': mse_loss.item(), 'ssim': ssim_loss.item()}

    def train(self, train_data, val_data, epochs, save_path="best_model", log_dir="./logs", transfer=None):
        """
        Training loop with validation for fair comparison with EDM and CNN.

        Args:
            train_data: Training dataloader
            val_data: Validation dataloader
            epochs: Number of training epochs
            save_path: Path to save best model
            log_dir: Directory to save logs
            transfer: Path to pretrained model (optional)
        """
        setup_logging(log_dir)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        best_val_loss = float('inf')

        # History for tracking (now includes validation)
        history = {
            'train_loss': [], 'val_loss': [],
            'train_mse': [], 'val_mse': [],
            'train_ssim': [], 'val_ssim': [],
            'learning_rate': []
        }

        # Open log file
        log_file_path = os.path.join(log_dir, 'training_log.txt')
        log_file = open(log_file_path, 'w')
        log_file.write(f"DDIM Training Log (with Validation)\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Total Epochs: {epochs}\n")
        log_file.write(f"DDPM Timesteps: {self.ddpm_timestep}\n")
        log_file.write(f"DDIM Timesteps: {self.ddim_timestep}\n")
        log_file.write(f"Eta: {self.eta}\n")
        log_file.write(f"{'='*60}\n\n")

        for epoch in range(epochs):
            # Training phase
            train_losses = {'total': [], 'mse': [], 'ssim': []}
            self.model.train()
            for batch in tqdm(train_data, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
                inputs = batch['input'].to("cuda")
                targets = batch['target'].to("cuda")
                t = torch.randint(0, self.ddim_timestep, (inputs.shape[0],)).to("cuda")

                loss_dict = self.train_step(targets, t, inputs)

                train_losses['total'].append(loss_dict['total'])
                train_losses['mse'].append(loss_dict['mse'])
                train_losses['ssim'].append(loss_dict['ssim'])

            # Validation phase
            val_losses = {'total': [], 'mse': [], 'ssim': []}
            self.model.eval()
            for batch in tqdm(val_data, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                inputs = batch['input'].to("cuda")
                targets = batch['target'].to("cuda")
                t = torch.randint(0, self.ddim_timestep, (inputs.shape[0],)).to("cuda")

                loss_dict = self.validation_step(targets, t, inputs)

                val_losses['total'].append(loss_dict['total'])
                val_losses['mse'].append(loss_dict['mse'])
                val_losses['ssim'].append(loss_dict['ssim'])

            # Calculate averages
            avg_train_loss = sum(train_losses['total']) / len(train_losses['total'])
            avg_train_mse = sum(train_losses['mse']) / len(train_losses['mse'])
            avg_train_ssim = sum(train_losses['ssim']) / len(train_losses['ssim'])

            avg_val_loss = sum(val_losses['total']) / len(val_losses['total'])
            avg_val_mse = sum(val_losses['mse']) / len(val_losses['mse'])
            avg_val_ssim = sum(val_losses['ssim']) / len(val_losses['ssim'])

            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_mse'].append(avg_train_mse)
            history['val_mse'].append(avg_val_mse)
            history['train_ssim'].append(avg_train_ssim)
            history['val_ssim'].append(avg_val_ssim)

            # Scheduler step (use validation loss)
            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)

            # Logging
            log_msg = (f"Epoch {epoch + 1}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.6f} (MSE: {avg_train_mse:.6f}, SSIM: {avg_train_ssim:.6f}), "
                      f"Val Loss: {avg_val_loss:.6f} (MSE: {avg_val_mse:.6f}, SSIM: {avg_val_ssim:.6f}), "
                      f"LR: {current_lr:.2e}")
            print(log_msg)
            log_file.write(log_msg + '\n')
            log_file.flush()

            logging.info(log_msg)

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), f'{save_path}')
                save_msg = f"  → Saved best model (val_loss: {best_val_loss:.6f})"
                print(save_msg)
                log_file.write(save_msg + '\n')
                log_file.flush()

        log_file.close()
        print(f"\nTraining log saved to: {log_file_path}")

        # Save loss plot (with validation curves)
        import matplotlib.pyplot as plt
        plot_path = os.path.join(log_dir, 'loss_curves.png')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs_range = range(1, epochs + 1)

        # Total Loss plot
        ax1 = axes[0, 0]
        ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Total Loss', linewidth=2, markersize=4)
        ax1.plot(epochs_range, history['val_loss'], 'r-s', label='Val Total Loss', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Total Loss (MSE + SSIM)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # MSE Loss plot
        ax2 = axes[0, 1]
        ax2.plot(epochs_range, history['train_mse'], 'b-o', label='Train MSE', linewidth=2, markersize=4)
        ax2.plot(epochs_range, history['val_mse'], 'r-s', label='Val MSE', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MSE Loss', fontsize=12)
        ax2.set_title('MSE Loss', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # SSIM Loss plot
        ax3 = axes[1, 0]
        ax3.plot(epochs_range, history['train_ssim'], 'b-o', label='Train SSIM Loss', linewidth=2, markersize=4)
        ax3.plot(epochs_range, history['val_ssim'], 'r-s', label='Val SSIM Loss', linewidth=2, markersize=4)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('SSIM Loss (1 - SSIM)', fontsize=12)
        ax3.set_title('SSIM Loss', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Learning rate plot
        ax4 = axes[1, 1]
        ax4.plot(epochs_range, history['learning_rate'], 'g-^', linewidth=2, markersize=4)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Learning Rate', fontsize=12)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss curves saved to: {plot_path}")

    def train_unconditional(self, train_data, val_data, epochs, save_path="best_model", log_dir="./logs", use_tensorboard=False):
        """Training loop for unconditional generation (Gaussian noise as condition).

        Args:
            train_data: Training dataloader
            val_data: Validation dataloader
            epochs: Number of training epochs
            save_path: Path to save best model
            log_dir: Directory to save logs
            use_tensorboard: Not used, kept for compatibility
        """
        setup_logging(log_dir)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        best_val_loss = float('inf')

        # History for tracking
        history = {
            'train_loss': [], 'val_loss': [],
            'train_mse': [], 'val_mse': [],
            'train_ssim': [], 'val_ssim': [],
            'learning_rate': []
        }

        # Open log file
        log_file_path = os.path.join(log_dir, 'training_log.txt')
        log_file = open(log_file_path, 'w')
        log_file.write(f"DDIM Unconditional Generation Training Log\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Total Epochs: {epochs}\n")
        log_file.write(f"DDPM Timesteps: {self.ddpm_timestep}\n")
        log_file.write(f"DDIM Timesteps: {self.ddim_timestep}\n")
        log_file.write(f"Eta: {self.eta}\n")
        log_file.write(f"Mode: Unconditional Generation (Gaussian Noise as Condition)\n")
        log_file.write(f"{'='*60}\n\n")

        for epoch in range(epochs):
            # Training phase
            train_losses = {'total': [], 'mse': [], 'ssim': []}
            self.model.train()
            for batch in tqdm(train_data, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
                targets = batch['target'].to("cuda")
                indices = batch['index']  # Get sample indices
                B, C, H, W = targets.shape

                # Generate FIXED Gaussian noise per sample using index as seed
                noise_cond = torch.zeros(B, 1, H, W, device=targets.device)
                for i, idx in enumerate(indices):
                    # Use index as seed for reproducible noise per sample
                    generator = torch.Generator(device=targets.device)
                    generator.manual_seed(int(idx))
                    noise_cond[i] = torch.randn(1, H, W, device=targets.device, generator=generator)

                t = torch.randint(0, self.ddim_timestep, (B,)).to("cuda")
                loss_dict = self.train_step(targets, t, noise_cond)

                train_losses['total'].append(loss_dict['total'])
                train_losses['mse'].append(loss_dict['mse'])
                train_losses['ssim'].append(loss_dict['ssim'])

            # Validation phase
            val_losses = {'total': [], 'mse': [], 'ssim': []}
            self.model.eval()
            for batch in tqdm(val_data, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                targets = batch['target'].to("cuda")
                indices = batch['index']  # Get sample indices
                B, C, H, W = targets.shape

                # Generate FIXED Gaussian noise per sample using index as seed
                noise_cond = torch.zeros(B, 1, H, W, device=targets.device)
                for i, idx in enumerate(indices):
                    # Use index as seed for reproducible noise per sample
                    generator = torch.Generator(device=targets.device)
                    generator.manual_seed(int(idx))
                    noise_cond[i] = torch.randn(1, H, W, device=targets.device, generator=generator)

                t = torch.randint(0, self.ddim_timestep, (B,)).to("cuda")
                loss_dict = self.validation_step(targets, t, noise_cond)

                val_losses['total'].append(loss_dict['total'])
                val_losses['mse'].append(loss_dict['mse'])
                val_losses['ssim'].append(loss_dict['ssim'])

            # Calculate averages
            avg_train_loss = sum(train_losses['total']) / len(train_losses['total'])
            avg_train_mse = sum(train_losses['mse']) / len(train_losses['mse'])
            avg_train_ssim = sum(train_losses['ssim']) / len(train_losses['ssim'])

            avg_val_loss = sum(val_losses['total']) / len(val_losses['total'])
            avg_val_mse = sum(val_losses['mse']) / len(val_losses['mse'])
            avg_val_ssim = sum(val_losses['ssim']) / len(val_losses['ssim'])

            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_mse'].append(avg_train_mse)
            history['val_mse'].append(avg_val_mse)
            history['train_ssim'].append(avg_train_ssim)
            history['val_ssim'].append(avg_val_ssim)

            # Scheduler step
            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)

            # Logging
            log_msg = (f"Epoch {epoch + 1}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.6f} (MSE: {avg_train_mse:.6f}, SSIM: {avg_train_ssim:.6f}), "
                      f"Val Loss: {avg_val_loss:.6f} (MSE: {avg_val_mse:.6f}, SSIM: {avg_val_ssim:.6f}), "
                      f"LR: {current_lr:.2e}")
            print(log_msg)
            log_file.write(log_msg + '\n')
            log_file.flush()
            logging.info(log_msg)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), f'{save_path}')
                save_msg = f"  → Saved best model (val_loss: {best_val_loss:.6f})"
                print(save_msg)
                log_file.write(save_msg + '\n')
                log_file.flush()

        log_file.close()
        print(f"\nTraining log saved to: {log_file_path}")

        # Save loss plot
        import matplotlib.pyplot as plt
        plot_path = os.path.join(log_dir, 'loss_curves.png')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs_range = range(1, epochs + 1)

        ax1 = axes[0, 0]
        ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Total Loss', linewidth=2, markersize=4)
        ax1.plot(epochs_range, history['val_loss'], 'r-s', label='Val Total Loss', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Total Loss (MSE + SSIM)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(epochs_range, history['train_mse'], 'b-o', label='Train MSE', linewidth=2, markersize=4)
        ax2.plot(epochs_range, history['val_mse'], 'r-s', label='Val MSE', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MSE Loss', fontsize=12)
        ax2.set_title('MSE Loss', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(epochs_range, history['train_ssim'], 'b-o', label='Train SSIM Loss', linewidth=2, markersize=4)
        ax3.plot(epochs_range, history['val_ssim'], 'r-s', label='Val SSIM Loss', linewidth=2, markersize=4)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('SSIM Loss (1 - SSIM)', fontsize=12)
        ax3.set_title('SSIM Loss', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.plot(epochs_range, history['learning_rate'], 'g-^', linewidth=2, markersize=4)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Learning Rate', fontsize=12)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss curves saved to: {plot_path}")
