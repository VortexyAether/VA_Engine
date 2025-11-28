import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import math
from einops import rearrange
from tqdm import tqdm
import os
import logging


class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings for conditioning."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization with timestep conditioning."""
    def __init__(self, num_groups, num_channels, time_emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, num_channels * 2)
        )
    
    def forward(self, x, time_emb):
        x = self.norm(x)
        scale_shift = self.time_mlp(time_emb)
        scale_shift = rearrange(scale_shift, 'b c -> b c 1 1')
        scale, shift = scale_shift.chunk(2, dim=1)
        return x * (1 + scale) + shift


class FrequencyAwareBlock(nn.Module):
    """Residual block with frequency-aware adaptive normalization."""
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(groups, out_channels, time_emb_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = AdaptiveGroupNorm(groups, out_channels, time_emb_dim)
        
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.SiLU()
    
    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h, time_emb)
        h = self.activation(h)
        h = self.conv2(h)
        h = self.norm2(h, time_emb)
        return self.activation(h + self.residual(x))


class FrequencyProgressiveUNet(nn.Module):
    """Single UNet with frequency-aware progressive refinement."""
    def __init__(self, in_channels, out_channels, base_channels=64, time_emb_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Encoder
        self.enc1 = FrequencyAwareBlock(in_channels, base_channels, time_emb_dim)  # x_t + condition
        self.enc2 = FrequencyAwareBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = FrequencyAwareBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.enc4 = FrequencyAwareBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = FrequencyAwareBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        
        # Decoder with skip connections
        self.dec4 = FrequencyAwareBlock(base_channels * 16, base_channels * 4, time_emb_dim)
        self.dec3 = FrequencyAwareBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.dec2 = FrequencyAwareBlock(base_channels * 4, base_channels, time_emb_dim)
        self.dec1 = FrequencyAwareBlock(base_channels * 2, base_channels, time_emb_dim)
        
        # Output
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
        # Downsampling and upsampling
        self.downs = nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1) 
                                    for c in [base_channels, base_channels*2, base_channels*4, base_channels*8]])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(c, c, 4, stride=2, padding=1) 
                                 for c in [base_channels*8, base_channels*4, base_channels*2, base_channels]])
    
    def forward(self, x, timestep):
        # Time embedding
        t_emb = self.time_embedding(timestep)
        t_emb = self.time_mlp(t_emb)
        
        # Encoder
        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(self.downs[0](h1), t_emb)
        h3 = self.enc3(self.downs[1](h2), t_emb)
        h4 = self.enc4(self.downs[2](h3), t_emb)
        
        # Bottleneck
        h = self.bottleneck(self.downs[3](h4), t_emb)
        
        # Decoder with skip connections
        h = self.dec4(torch.cat([self.ups[0](h), h4], dim=1), t_emb)
        h = self.dec3(torch.cat([self.ups[1](h), h3], dim=1), t_emb)
        h = self.dec2(torch.cat([self.ups[2](h), h2], dim=1), t_emb)
        h = self.dec1(torch.cat([self.ups[3](h), h1], dim=1), t_emb)
        
        return self.final(h)


class EDM(nn.Module):
    """Energy Dissipation Model with single UNet and AdaGN."""
    
    def __init__(self, unet, num_timesteps=4, lr=1e-4,
                 Lx=0.5*math.pi, Ly=0.5*math.pi,
                 min_freq_ratio=0.02, max_freq_ratio=0.8):
        super().__init__()
        
        self.unet = unet
        self.num_timesteps = num_timesteps
        self.min_freq_ratio = min_freq_ratio
        self.max_freq_ratio = max_freq_ratio
        self.Lx = Lx
        self.Ly = Ly
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.loss_fn = nn.MSELoss()
        self.ssim_weight = 0.5  # Weight for SSIM loss

        # SSIM metric (requires torchmetrics: pip install torchmetrics)
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        except ImportError:
            print("Warning: torchmetrics not installed. Install with: pip install torchmetrics")
            self.ssim_metric = None

    def _compute_combined_loss(self, pred, target):
        """Compute MSE + SSIM loss"""
        mse_loss = self.loss_fn(pred, target)

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

    def _create_freq_mask(self, B, C, H, W, t_scalar, device):
        """Create frequency mask for progressive filtering."""
        if t_scalar == 0:
            return torch.ones((B, C, H, W//2+1), device=device)
        elif t_scalar >= self.num_timesteps:
            return torch.zeros((B, C, H, W//2+1), device=device)
        
        # Compute cutoff ratio
        t_norm = (t_scalar - 1) / (self.num_timesteps - 1) if self.num_timesteps > 1 else 1.0
        log_ratio = math.log(self.max_freq_ratio) - t_norm * (math.log(self.max_freq_ratio) - math.log(self.min_freq_ratio))
        cutoff_ratio = math.exp(log_ratio)
        
        # Create frequency grid
        freqs_h = torch.fft.fftfreq(H, device=device).unsqueeze(1)
        freqs_w = torch.fft.rfftfreq(W, device=device).unsqueeze(0)
        freq_dist = torch.sqrt(freqs_h**2 + freqs_w**2)
        
        # Create mask
        mask = (freq_dist <= cutoff_ratio).float()
        return mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W//2+1)
    
    def q_sample(self, x_0, t):
        """Forward process: Apply frequency filtering (vectorized)."""
        B = x_0.shape[0]
        device = x_0.device
        x_t = torch.zeros_like(x_0)
        
        # Handle different timesteps efficiently
        if torch.is_tensor(t):
            unique_t = torch.unique(t)
            for t_val in unique_t:
                mask_t = (t == t_val)
                if t_val == 0:
                    x_t[mask_t] = x_0[mask_t]
                elif t_val >= self.num_timesteps:
                    x_t[mask_t] = 0
                else:
                    # Apply frequency filtering
                    x_batch = x_0[mask_t]
                    x_freq = torch.fft.rfft2(x_batch)
                    freq_mask = self._create_freq_mask(x_batch.shape[0], x_batch.shape[1], 
                                                       x_batch.shape[-2], x_batch.shape[-1], 
                                                       t_val.item(), device)
                    x_freq_filtered = x_freq * freq_mask
                    x_t[mask_t] = torch.fft.irfft2(x_freq_filtered, s=(x_batch.shape[-2], x_batch.shape[-1]))
        else:
            # Single timestep for all samples
            if t == 0:
                return x_0
            elif t >= self.num_timesteps:
                return torch.zeros_like(x_0)
            else:
                x_freq = torch.fft.rfft2(x_0)
                freq_mask = self._create_freq_mask(B, x_0.shape[1], x_0.shape[2], x_0.shape[3], t, device)
                x_freq_filtered = x_freq * freq_mask
                x_t = torch.fft.irfft2(x_freq_filtered, s=(x_0.shape[-2], x_0.shape[-1]))
        
        return x_t
    
    @torch.no_grad()
    def p_sample(self, x_t, condition, t, device):
        """Single reverse process step.

        Given x_t at timestep t, predict x_{t-1}.
        Training learns: model(x_{t+1}, cond_filtered@(t-1), timestep=t) -> x_t
        So sampling should use: model(x_t, cond_filtered@(t-2), timestep=t-1) -> x_{t-1}
        """
        B = x_t.shape[0]

        # Apply frequency filtering to condition at timestep (t-1)-1 = t-2
        # Because we're passing timestep t-1 to the model, and training uses cond@(timestep-1)
        t_cond = max(0, t - 2)
        cond_filtered = self.q_sample(condition, torch.tensor(t_cond, device=device))

        # Model prediction: model(x_t, cond@(t-2), timestep=t-1) -> x_{t-1}
        # This matches training: model(x_{t+1}, cond@(t-1), timestep=t) -> x_t
        model_input = torch.cat([x_t, cond_filtered], dim=1)
        t_tensor = torch.full((B,), t - 1, device=device, dtype=torch.long)
        x_t_minus_1 = self.unet(model_input, t_tensor)

        return x_t_minus_1
    
    @torch.no_grad()
    def sampling(self, n_samples, device, grid_size, cond):
        """Full sampling process using progressive refinement."""
        # Start from zeros (maximum frequency filtering)  
        # x_t should match the output channels of the UNet
        target_channels = self.unet.final[-1].out_channels  # Get actual output channels from UNet
        x_t = torch.zeros(n_samples, target_channels, grid_size[0], grid_size[1], device=device)
        
        # Initialize list with correct size: [x_T, x_{T-1}, ..., x_1, x_0]
        intermediate_predictions = [None] * (self.num_timesteps + 1)
        intermediate_predictions[self.num_timesteps] = x_t  # x_T (t=num_timesteps)
        
        # Progressive refinement: t=T -> t=T-1 -> ... -> t=1 -> t=0
        for t in reversed(range(1, self.num_timesteps + 1)):
            x_t = self.p_sample(x_t, cond, t, device)  # p_sample(x_t, t) -> x_{t-1}
            intermediate_predictions[t-1] = x_t  # Store x_{t-1} at position t-1
        
        return intermediate_predictions
    
    def train_step(self, targets, cond):
        """Single training step with random timestep sampling."""
        B = targets.shape[0]
        device = targets.device

        # Sample random timesteps
        t = torch.randint(1, self.num_timesteps + 1, (B,), device=device)

        # Get x_t (target state)
        x_t = self.q_sample(targets, t)

        # Get x_{t+1} (input state)
        t_plus_1 = torch.clamp(t + 1, max=self.num_timesteps)
        x_t_plus_1 = self.q_sample(targets, t_plus_1)

        # Get filtered condition at t-1
        t_minus_1 = torch.clamp(t - 1, min=0)
        cond_filtered = self.q_sample(cond, t_minus_1)

        # Forward pass
        model_input = torch.cat([x_t_plus_1, cond_filtered], dim=1)
        pred = self.unet(model_input, t)

        # Compute combined loss (MSE + SSIM)
        total_loss, mse_loss, ssim_loss = self._compute_combined_loss(pred, x_t)

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()

        return {'total': total_loss.item(), 'mse': mse_loss.item(), 'ssim': ssim_loss.item()}
    
    def val_step(self, targets, cond):
        """Validation step (no gradient computation)."""
        self.unet.eval()
        with torch.no_grad():
            B = targets.shape[0]
            device = targets.device

            # Sample random timesteps
            t = torch.randint(1, self.num_timesteps + 1, (B,), device=device)

            # Same as train_step but without optimization
            x_t = self.q_sample(targets, t)
            t_plus_1 = torch.clamp(t + 1, max=self.num_timesteps)
            x_t_plus_1 = self.q_sample(targets, t_plus_1)
            t_minus_1 = torch.clamp(t - 1, min=0)
            cond_filtered = self.q_sample(cond, t_minus_1)

            model_input = torch.cat([x_t_plus_1, cond_filtered], dim=1)
            pred = self.unet(model_input, t)

            # Compute combined loss (MSE + SSIM)
            total_loss, mse_loss, ssim_loss = self._compute_combined_loss(pred, x_t)

        self.unet.train()
        return {'total': total_loss.item(), 'mse': mse_loss.item(), 'ssim': ssim_loss.item()}
    
    def train(self, train_data, val_data, epochs, save_path="best_model.pth", log_dir="./logs", use_tensorboard=False):
        """Training loop with validation.

        Args:
            train_data: Training dataloader
            val_data: Validation dataloader
            epochs: Number of training epochs
            save_path: Path to save best model checkpoint
            log_dir: Directory to save logs and plots
            use_tensorboard: If True, log to TensorBoard (requires tensorboard package)
        """
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        best_val_loss = float('inf')

        # History for plotting
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'train_ssim': [],
            'val_ssim': [],
            'learning_rate': []
        }

        # TensorBoard setup (optional)
        writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard logging enabled. Run: tensorboard --logdir={log_dir}")
            except ImportError:
                print("Warning: tensorboard not installed. Install with: pip install tensorboard")
                use_tensorboard = False

        # Open log file
        log_file_path = os.path.join(log_dir, 'training_log.txt')
        log_file = open(log_file_path, 'w')
        log_file.write(f"EDM Training Log\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Total Epochs: {epochs}\n")
        log_file.write(f"Num Timesteps: {self.num_timesteps}\n")
        log_file.write(f"Min Freq Ratio: {self.min_freq_ratio}\n")
        log_file.write(f"Max Freq Ratio: {self.max_freq_ratio}\n")
        log_file.write(f"TensorBoard: {use_tensorboard}\n")
        log_file.write(f"{'='*60}\n\n")

        for epoch in range(epochs):
            # Training
            self.unet.train()
            train_losses = {'total': [], 'mse': [], 'ssim': []}
            for batch in tqdm(train_data, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
                cond = batch['input'].cuda()
                targets = batch['target'].cuda()
                loss_dict = self.train_step(targets, cond)
                train_losses['total'].append(loss_dict['total'])
                train_losses['mse'].append(loss_dict['mse'])
                train_losses['ssim'].append(loss_dict['ssim'])

            # Validation
            val_losses = {'total': [], 'mse': [], 'ssim': []}
            for batch in tqdm(val_data, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                cond = batch['input'].cuda()
                targets = batch['target'].cuda()
                loss_dict = self.val_step(targets, cond)
                val_losses['total'].append(loss_dict['total'])
                val_losses['mse'].append(loss_dict['mse'])
                val_losses['ssim'].append(loss_dict['ssim'])

            avg_train_loss = sum(train_losses['total']) / len(train_losses['total'])
            avg_val_loss = sum(val_losses['total']) / len(val_losses['total'])
            avg_train_mse = sum(train_losses['mse']) / len(train_losses['mse'])
            avg_val_mse = sum(val_losses['mse']) / len(val_losses['mse'])
            avg_train_ssim = sum(train_losses['ssim']) / len(train_losses['ssim'])
            avg_val_ssim = sum(val_losses['ssim']) / len(val_losses['ssim'])

            # Scheduler step
            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_mse'].append(avg_train_mse)
            history['val_mse'].append(avg_val_mse)
            history['train_ssim'].append(avg_train_ssim)
            history['val_ssim'].append(avg_val_ssim)
            history['learning_rate'].append(current_lr)

            # TensorBoard logging
            if writer is not None:
                writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
                writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
                writer.add_scalar('Loss/train_mse', avg_train_mse, epoch)
                writer.add_scalar('Loss/val_mse', avg_val_mse, epoch)
                writer.add_scalar('Loss/train_ssim', avg_train_ssim, epoch)
                writer.add_scalar('Loss/val_ssim', avg_val_ssim, epoch)
                writer.add_scalar('Learning_Rate', current_lr, epoch)

            # Logging
            log_msg = (f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.6f} (MSE: {avg_train_mse:.6f}, SSIM: {avg_train_ssim:.6f}), "
                      f"Val Loss: {avg_val_loss:.6f} (MSE: {avg_val_mse:.6f}, SSIM: {avg_val_ssim:.6f}), "
                      f"LR: {current_lr:.2e}")
            print(log_msg)
            log_file.write(log_msg + '\n')
            log_file.flush()

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.unet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                }, save_path)
                save_msg = f"  → Saved best model (val_loss: {best_val_loss:.6f})"
                print(save_msg)
                log_file.write(save_msg + '\n')
                log_file.flush()

        log_file.close()
        print(f"\nTraining log saved to: {log_file_path}")

        # Close TensorBoard writer
        if writer is not None:
            writer.close()
            print(f"TensorBoard logs saved to: {log_dir}")

        # Plot loss curves
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


class EDM_AE(EDM):
    """Energy Dissipation Model for Autoencoder (Unsupervised Learning)

    This class extends EDM for unsupervised learning where:
    - Input: heavily filtered timestep contour
    - Output: less filtered timestep contour
    - No conditioning needed, pure autoencoder approach
    - Zero field (max timestep) is excluded from training
    """

    def __init__(self, unet, num_timesteps=4, lr=1e-4,
                 Lx=0.5*math.pi, Ly=0.5*math.pi,
                 min_freq_ratio=0.02, max_freq_ratio=0.8,
                 ssim_weight=0.5, freq_weight=0.1, multistep_weight=0.5):
        super().__init__(unet, num_timesteps, lr, Lx, Ly, min_freq_ratio, max_freq_ratio)

        # Loss weights
        self.ssim_weight = ssim_weight
        self.freq_weight = freq_weight
        self.multistep_weight = multistep_weight

        # Validate that UNet has correct input/output channels for autoencoder
        if hasattr(unet, 'final') and hasattr(unet.final[-1], 'out_channels'):
            out_channels = unet.final[-1].out_channels
            # For autoencoder, input channels should equal output channels
            print(f"EDM_AE initialized: {out_channels} → {out_channels} channels (unsupervised)")
            print(f"Loss: MSE + {ssim_weight}*SSIM + {freq_weight}*Frequency + {multistep_weight}*MultiStep")

    def _compute_ssim(self, img1, img2, window_size=11):
        """Compute SSIM between two images (simplified version)"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Simple average pooling as window
        kernel = torch.ones(1, 1, window_size, window_size, device=img1.device) / (window_size ** 2)

        # Handle multi-channel images by averaging SSIM across channels
        channels = img1.shape[1]
        ssim_per_channel = []

        for c in range(channels):
            img1_c = img1[:, c:c+1, :, :]
            img2_c = img2[:, c:c+1, :, :]

            mu1 = F.conv2d(img1_c, kernel, padding=window_size//2)
            mu2 = F.conv2d(img2_c, kernel, padding=window_size//2)

            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1_c ** 2, kernel, padding=window_size//2) - mu1_sq
            sigma2_sq = F.conv2d(img2_c ** 2, kernel, padding=window_size//2) - mu2_sq
            sigma12 = F.conv2d(img1_c * img2_c, kernel, padding=window_size//2) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            ssim_per_channel.append(ssim_map.mean())

        return torch.stack(ssim_per_channel).mean()

    def _compute_frequency_loss(self, pred, target):
        """Compute frequency domain loss using L1 distance"""
        # Apply FFT
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        # L1 loss on magnitude spectrum
        freq_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

        return freq_loss

    def _compute_combined_loss(self, pred, target):
        """Compute combined loss: MSE + SSIM + Frequency"""
        # MSE loss (base)
        mse_loss = self.loss_fn(pred, target)

        # SSIM loss (structural similarity)
        ssim_val = self._compute_ssim(pred, target)
        ssim_loss = 1.0 - ssim_val  # Convert to loss (lower is better)

        # Frequency domain loss (high-frequency detail preservation)
        freq_loss = self._compute_frequency_loss(pred, target)

        # Combined loss
        total_loss = mse_loss + self.ssim_weight * ssim_loss + self.freq_weight * freq_loss

        return total_loss, mse_loss, ssim_loss, freq_loss
    
    @torch.no_grad()
    def p_sample(self, x_t, t, device):
        """Single reverse process step (unsupervised autoencoder)."""
        B = x_t.shape[0]
        
        # Model prediction (no conditioning needed for unsupervised)
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        x_t_minus_1 = self.unet(x_t, t_tensor)
        
        return x_t_minus_1
    
    @torch.no_grad()
    def sampling(self, n_samples, device, grid_size, start_from_timestep=None):
        """Full sampling process using progressive refinement (unsupervised)
        
        Args:
            n_samples: Number of samples to generate
            device: Device to use
            grid_size: Grid size tuple (H, W)
            start_from_timestep: Starting timestep (if None, start from max timestep-1)
        """
        # Get output channels from UNet
        target_channels = self.unet.final[-1].out_channels
        
        # Start from maximum filtered state (but not zero field)
        if start_from_timestep is None:
            start_timestep = self.num_timesteps - 1  # Exclude zero field
        else:
            start_timestep = min(start_from_timestep, self.num_timesteps - 1)
        
        # Initialize with heavily filtered noise or can start with interpolated low-res
        x_t = torch.randn(n_samples, target_channels, grid_size[0], grid_size[1], device=device)
        # Apply heavy filtering to match starting timestep
        x_t = self.q_sample(x_t, torch.tensor(start_timestep, device=device))
        
        # Initialize list: [x_T-1, x_T-2, ..., x_1, x_0]
        intermediate_predictions = [None] * (start_timestep + 1)
        intermediate_predictions[start_timestep] = x_t
        
        # Progressive refinement: t=start_timestep -> ... -> t=1 -> t=0
        for t in reversed(range(1, start_timestep + 1)):
            x_t = self.p_sample(x_t, t, device)
            intermediate_predictions[t-1] = x_t
        
        return intermediate_predictions
    
    def train_step(self, targets):
        """Single training step with multi-step loss for unsupervised autoencoder."""
        import random

        B = targets.shape[0]
        device = targets.device

        # 1. Single-step loss (original)
        t = torch.randint(1, self.num_timesteps, (B,), device=device)
        x_t = self.q_sample(targets, t)
        t_plus_1 = torch.clamp(t + 1, max=self.num_timesteps - 1)
        x_t_plus_1 = self.q_sample(targets, t_plus_1)
        pred = self.unet(x_t_plus_1, t + 1)

        single_loss, single_mse, single_ssim, single_freq = self._compute_combined_loss(pred, x_t)

        # 2. Multi-step loss (new - to reduce accumulation error)
        # Randomly unroll 2-3 steps (reduced from 3-5 to save memory)
        num_steps = random.randint(2, min(3, self.num_timesteps - 1))
        start_t = random.randint(num_steps, self.num_timesteps - 1)

        # Use smaller batch for multi-step (use half of batch to save memory)
        half_B = B // 2
        targets_subset = targets[:half_B]

        # Start from heavily filtered state
        x_current = self.q_sample(targets_subset, torch.full((half_B,), start_t, device=device, dtype=torch.long))

        # Unroll num_steps
        for step in range(num_steps):
            t_curr = start_t - step
            t_tensor = torch.full((half_B,), t_curr, device=device, dtype=torch.long)
            x_current = self.unet(x_current, t_tensor)

        # Compare with target at (start_t - num_steps)
        target_t = start_t - num_steps
        x_target = self.q_sample(targets_subset, torch.full((half_B,), target_t, device=device, dtype=torch.long))
        multi_loss, multi_mse, multi_ssim, multi_freq = self._compute_combined_loss(x_current, x_target)

        # 3. Combined loss
        total_loss = single_loss + self.multistep_weight * multi_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'mse': single_mse.item(),
            'ssim': single_ssim.item(),
            'freq': single_freq.item(),
            'multi': multi_loss.item(),
            'multi_mse': multi_mse.item()
        }
    
    def val_step(self, targets, compute_full_sampling=False):
        """Validation step for unsupervised autoencoder.

        Args:
            targets: Ground truth targets
            compute_full_sampling: If True, also compute full sampling loss (t=max -> t=0)
        """
        self.unet.eval()
        with torch.no_grad():
            B = targets.shape[0]
            device = targets.device

            # 1. Single-step validation (same as training)
            t = torch.randint(1, self.num_timesteps, (B,), device=device)
            x_t = self.q_sample(targets, t)
            t_plus_1 = torch.clamp(t + 1, max=self.num_timesteps - 1)
            x_t_plus_1 = self.q_sample(targets, t_plus_1)
            pred = self.unet(x_t_plus_1, t + 1)

            # Compute combined loss for single-step
            total_loss, mse_loss, ssim_loss, freq_loss = self._compute_combined_loss(pred, x_t)

            result = {
                'total': total_loss.item(),
                'mse': mse_loss.item(),
                'ssim': ssim_loss.item(),
                'freq': freq_loss.item()
            }

            # 2. Full sampling validation (optional, more expensive)
            if compute_full_sampling:
                # Start from heavily filtered state
                start_t = self.num_timesteps - 1
                x_current = self.q_sample(targets, torch.full((B,), start_t, device=device, dtype=torch.long))

                # Sample from t=max-1 to t=0
                for t_step in reversed(range(1, start_t + 1)):
                    t_tensor = torch.full((B,), t_step, device=device, dtype=torch.long)
                    x_current = self.unet(x_current, t_tensor)

                # Final result should match ground truth (t=0)
                full_total_loss, full_mse, full_ssim, full_freq = self._compute_combined_loss(x_current, targets)

                result['full_sampling_total'] = full_total_loss.item()
                result['full_sampling_mse'] = full_mse.item()
                result['full_sampling_ssim'] = full_ssim.item()
                result['full_sampling_freq'] = full_freq.item()

        self.unet.train()
        return result
    
    def train_autoencoder(self, train_data, val_data, epochs, save_path="best_model.pth", log_dir="./logs"):
        """Training loop for unsupervised autoencoder."""
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        best_val_loss = float('inf')

        print(f"Training EDM_AE (Unsupervised) with {self.num_timesteps} timesteps (excluding zero field)")

        for epoch in range(epochs):
            # Training
            self.unet.train()
            train_losses = {'total': [], 'mse': [], 'ssim': [], 'freq': [], 'multi': [], 'multi_mse': []}
            for batch in tqdm(train_data, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
                # Only use targets for unsupervised learning
                targets = batch['target'].cuda()
                loss_dict = self.train_step(targets)
                for key in train_losses:
                    if key in loss_dict:
                        train_losses[key].append(loss_dict[key])

            # Validation (compute full sampling every 5 epochs)
            compute_full = (epoch % 5 == 0 or epoch == epochs - 1)
            val_losses = {'total': [], 'mse': [], 'ssim': [], 'freq': []}
            val_full_losses = {'total': [], 'mse': [], 'ssim': [], 'freq': []}

            for batch in tqdm(val_data, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                targets = batch['target'].cuda()
                loss_dict = self.val_step(targets, compute_full_sampling=compute_full)

                # Single-step validation
                for key in val_losses:
                    val_losses[key].append(loss_dict[key])

                # Full sampling validation (if computed)
                if compute_full:
                    val_full_losses['total'].append(loss_dict['full_sampling_total'])
                    val_full_losses['mse'].append(loss_dict['full_sampling_mse'])
                    val_full_losses['ssim'].append(loss_dict['full_sampling_ssim'])
                    val_full_losses['freq'].append(loss_dict['full_sampling_freq'])

            # Compute averages
            avg_train_loss = {k: sum(v) / len(v) for k, v in train_losses.items()}
            avg_val_loss = {k: sum(v) / len(v) for k, v in val_losses.items()}

            # Scheduler step (use total loss)
            self.scheduler.step(avg_val_loss['total'])
            current_lr = self.optimizer.param_groups[0]['lr']

            # Logging
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train - Total: {avg_train_loss['total']:.6f}, MSE: {avg_train_loss['mse']:.6f}, "
                  f"SSIM: {avg_train_loss['ssim']:.6f}, Freq: {avg_train_loss['freq']:.6f}, "
                  f"Multi: {avg_train_loss['multi']:.6f}, Multi_MSE: {avg_train_loss['multi_mse']:.6f}")
            print(f"  Val   - Total: {avg_val_loss['total']:.6f}, MSE: {avg_val_loss['mse']:.6f}, "
                  f"SSIM: {avg_val_loss['ssim']:.6f}, Freq: {avg_val_loss['freq']:.6f}")

            # Full sampling validation logging
            if compute_full:
                avg_val_full = {k: sum(v) / len(v) for k, v in val_full_losses.items()}
                print(f"  Full  - Total: {avg_val_full['total']:.6f}, MSE: {avg_val_full['mse']:.6f}, "
                      f"SSIM: {avg_val_full['ssim']:.6f}, Freq: {avg_val_full['freq']:.6f}")

            print(f"  LR: {current_lr:.2e}")

            # Save best model (based on single-step validation)
            if avg_val_loss['total'] < best_val_loss:
                best_val_loss = avg_val_loss['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.unet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                }, save_path)
                print(f"  → Saved best model (val_loss: {best_val_loss:.6f})")

