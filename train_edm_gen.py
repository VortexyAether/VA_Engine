from model.unet import UNet
from model.edm_single import EDM_AE, FrequencyProgressiveUNet
from model.ddim import DDIM
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
import random
import io

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils.s2s import create_snapshot_dataloaders
from utils.plot import plot_q_sample_energy_cascade, plot_energy_cascade_analysis
from utils.hallucination_detector import TurbulenceHallucinationDetector, aggregate_hallucination_results
import argparse


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

############################ Hyperparameter ############################
# Create an argument parser
parser = argparse.ArgumentParser(description='Train EDM for unconditional generation.')
parser.add_argument('--model', type=str, default='edm', choices=['edm', 'ddim', 'cnn'], help='The model to train (edm, ddim, or cnn).')
parser.add_argument('--epochs', type=int, default=10, help='The number of training epochs.')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'load'], help='Set mode to train or load.')
parser.add_argument('--dataset', type=str, default='decaying_snapshot', help='Select the datasets')
parser.add_argument('--num_unets', type=int, default=3, help='The number of timesteps (num_unets + 1).')
parser.add_argument('--max_cutoff', type=float, default=0.8, help='The maximum cutoff frequency')
parser.add_argument('--min_cutoff', type=float, default=0.02, help='The minimum cutoff frequency')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='default learning rate')
parser.add_argument('--train_ratio', type=float, default=0.6, help='train_ratio')
parser.add_argument('--val_ratio', type=float, default=0.2, help='val_ratio')
parser.add_argument('--data_ratio', type=float, default=1.0, help='Ratio of total dataset to use (0.0-1.0). E.g., 0.6 uses 60%% of all data')
parser.add_argument('--target', type=str, default='Y.npy', help='default=Y.npy (target data for generation)')
parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
parser.add_argument('--ssim_weight', type=float, default=0.5, help='Weight for SSIM loss')
parser.add_argument('--freq_weight', type=float, default=0.1, help='Weight for frequency loss')
parser.add_argument('--multistep_weight', type=float, default=0.5, help='Weight for multi-step loss')
args = parser.parse_args()

epochs = args.epochs
dataset = args.dataset
model_name = args.model
num_unets = args.num_unets
batch_size = args.batch_size
learning_rate = args.learning_rate
train_ratio = args.train_ratio
val_ratio = train_ratio/3
test_ratio = 1 - train_ratio - val_ratio

save_result_path = f'test_GEN/result_{model_name}_{dataset}_gen_{train_ratio*5000}'

os.makedirs(save_result_path, exist_ok=True)
with open(os.path.join(save_result_path, 'parameters.txt'), 'w') as f:
    for arg, value in vars(args).items():
        f.write(f'{arg}: {value}\n')


num_timesteps = num_unets + 1
###########################################################################

# initializing model
print(f'Initializing {args.model.upper()} model for unconditional generation....')
torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Dataset load (only target data, no condition needed)
data_dir = f'/home/navier/Dataset/various_CFD/{dataset}/'

# For generation, we only need target data (no input/condition)
train_loader, val_loader, test_loader, dataset = create_snapshot_dataloaders(
        data_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        data_ratio=args.data_ratio,
        condition=args.target,  # Use target as both input and target for autoencoder
        target=args.target
    )


batch_sample= next(iter(train_loader))

# For generation, we only use target data
targets_sample = batch_sample['target']

print(f"Target shape: {targets_sample.shape}")
_, C_out, _, _ = targets_sample.shape


if args.model == 'edm':
    # For unconditional generation, input = output (autoencoder)
    in_channels = C_out  # Only the generated state, no condition
    out_channels = C_out

    unet = FrequencyProgressiveUNet(
        in_channels=in_channels,  # Only state (no condition)
        out_channels=out_channels,
        base_channels=64,
        time_emb_dim=256
    ).cuda()

    # Use EDM_AE for unsupervised/unconditional generation
    model = EDM_AE(
        unet=unet,
        num_timesteps=num_timesteps,
        lr=learning_rate,
        max_freq_ratio=args.max_cutoff,
        min_freq_ratio=args.min_cutoff,
        ssim_weight=args.ssim_weight,
        freq_weight=args.freq_weight,
        multistep_weight=args.multistep_weight
    ).to(device)
    print(f'Successfully initialized EDM_AE model for unconditional generation!\n')

elif args.model == 'ddim':
    # For DDIM generation, we use Gaussian noise as condition
    # Input channels = C_out (noisy target) + C_out (Gaussian noise as condition)
    in_channels = C_out * 2  # noisy target + Gaussian noise condition
    out_channels = C_out
    unet = FrequencyProgressiveUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=64,
        time_emb_dim=256
    ).to(device)
    model = DDIM(model=unet,
                 ddpm_timestep=1000,
                 ddim_timestep=100,
                 lr=learning_rate).to(device)

    print(f'Successfully initialized DDIM model for generation (with Gaussian noise condition)!\n')

elif args.model == 'cnn':
    in_channels = C_out  # For generation, CNN takes noise/initial state
    out_channels = C_out
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    # SSIM metric for CNN
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ssim_weight = 0
        print('Successfully initialized CNN model with MSE + SSIM loss!\n')
    except ImportError:
        print("Warning: torchmetrics not installed. Using MSE loss only.")
        ssim_metric = None
        ssim_weight = 0.0
        print('Successfully initialized CNN model with MSE loss!\n')
else:
    print('You didnt select right model name..')


if args.model == 'edm':
    # --- Monitoring q_sample ---
    print(f"\nMonitoring EDM q_sample for {model.num_timesteps} timesteps...")
    NUM_SAMPLES_TO_MONITOR = 2
    NUM_T_STEPS_TO_PLOT = model.num_timesteps

    # Collect samples from train_loader
    all_targets = []
    for batch in train_loader:
        all_targets.append(batch['target'].to(device))
        total_samples = sum(tgt.shape[0] for tgt in all_targets)
        if total_samples >= NUM_SAMPLES_TO_MONITOR:
            break

    train_targets = torch.cat(all_targets, dim=0)

    if len(train_targets) > NUM_SAMPLES_TO_MONITOR:
        indices = torch.randperm(len(train_targets))[:NUM_SAMPLES_TO_MONITOR]
        train_targets = train_targets[indices]

    x0_samples = train_targets  # (N, C, H, W)

    plot_t_indices = list(range(NUM_T_STEPS_TO_PLOT))

    all_snapshots_by_sample = []
    for i in range(NUM_SAMPLES_TO_MONITOR):
        x0_single_sample = x0_samples[i:i+1]
        snapshots_for_this_sample = []
        print(f"Generating forward process snapshots for sample {i+1}/{NUM_SAMPLES_TO_MONITOR}...")
        for t_idx in tqdm(plot_t_indices, desc=f"Sample {i+1}"):
            t_batch = torch.tensor([t_idx], device=device).long()
            x_t_distorted = model.q_sample(x0_single_sample, t_batch)
            snapshots_for_this_sample.append(x_t_distorted.cpu().detach().numpy())
        all_snapshots_by_sample.append(snapshots_for_this_sample)

    print("\nForward process snapshot generation complete.")

    # --- Plotting q_sample ---
    channel_names = [f'Channel {i+1}' for i in range(C_out)]

    fig, axes = plt.subplots(NUM_SAMPLES_TO_MONITOR * C_out, NUM_T_STEPS_TO_PLOT,
                             figsize=(3 * NUM_T_STEPS_TO_PLOT, 3.2 * NUM_SAMPLES_TO_MONITOR * C_out),
                             squeeze=False, constrained_layout=True)
    fig.suptitle('EDM Forward Process (q_sample) Visualization - Generation', fontsize=20)

    for i in range(NUM_SAMPLES_TO_MONITOR):
        for ch_idx, ch_name in enumerate(channel_names):
            for j, t_idx in enumerate(plot_t_indices):
                ax_row = i * C_out + ch_idx
                ax = axes[ax_row, j]

                img_data = all_snapshots_by_sample[i][j][0, ch_idx, :, :]

                vmin = train_targets[:, ch_idx, :, :].min()
                vmax = train_targets[:, ch_idx, :, :].max()

                im = ax.imshow(img_data, origin='lower', aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)

                fig.colorbar(im, ax=ax, shrink=0.8)

                ax.set_title(f't = {t_idx}', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

                if j == 0:
                    ax.set_ylabel(f'Sample {i+1}\n{ch_name}', rotation=0, size='large', labelpad=40)

    plot_filename = os.path.join(save_result_path, "q_sample_monitoring_result_gen.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Monitoring plot saved to '{plot_filename}'")

    # --- Energy Cascade Analysis for q_sample ---
    print("\nGenerating energy cascade analysis for q_sample forward process...")
    plot_q_sample_energy_cascade(
        q_sample_snapshots=all_snapshots_by_sample[0],
        timesteps=plot_t_indices,
        save_path=save_result_path,
        experiment_name='q_sample_energy_cascade_gen',
        sample_idx=0,
        channel_idx=0
    )
    print("Energy cascade analysis for q_sample completed.")

# --- Start Training ---
if args.mode == 'train':
    print("\nStarting training for unconditional generation...")
    if args.model == 'edm':
        # Use train_autoencoder method for EDM_AE
        model.train_autoencoder(train_loader, val_loader, epochs,
                              f'{save_result_path}/best_model.pth',
                              f'{save_result_path}/logs')
    elif args.model == 'ddim':
        # For DDIM unconditional generation, we need to modify the dataloader
        # to provide Gaussian noise as condition instead of actual input
        # We'll create a wrapper that replaces batch['input'] with Gaussian noise

        # Create custom training function for DDIM with Gaussian noise condition
        import matplotlib.pyplot as plt
        from Diffusion.utils.diffusion_utils import setup_logging

        setup_logging(f'{save_result_path}/logs')
        os.makedirs(save_result_path, exist_ok=True)
        os.makedirs(f'{save_result_path}/logs', exist_ok=True)

        best_val_loss = float('inf')
        history = {
            'train_loss': [], 'val_loss': [],
            'train_mse': [], 'val_mse': [],
            'train_ssim': [], 'val_ssim': [],
            'learning_rate': []
        }

        log_file = open(f'{save_result_path}/logs/training_log.txt', 'w')
        log_file.write(f"DDIM Training Log (Unconditional Generation)\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Total Epochs: {epochs}\n")
        log_file.write(f"Condition: Gaussian Noise\n")
        log_file.write(f"{'='*60}\n\n")

        for epoch in range(epochs):
            # Training
            model.model.train()
            train_losses = {'total': [], 'mse': [], 'ssim': []}

            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
                targets = batch['target'].to(device)
                # Replace input with Gaussian noise as condition
                gaussian_cond = torch.randn_like(targets).to(device)
                t = torch.randint(0, model.ddim_timestep, (targets.shape[0],)).to(device)

                loss_dict = model.train_step(targets, t, gaussian_cond)
                train_losses['total'].append(loss_dict['total'])
                train_losses['mse'].append(loss_dict['mse'])
                train_losses['ssim'].append(loss_dict['ssim'])

            # Validation
            model.model.eval()
            val_losses = {'total': [], 'mse': [], 'ssim': []}

            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                targets = batch['target'].to(device)
                gaussian_cond = torch.randn_like(targets).to(device)
                t = torch.randint(0, model.ddim_timestep, (targets.shape[0],)).to(device)

                loss_dict = model.validation_step(targets, t, gaussian_cond)
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

            # Scheduler
            model.scheduler.step(avg_val_loss)
            current_lr = model.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)

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
                torch.save(model.model.state_dict(), f'{save_result_path}/best_model.pth')
                save_msg = f"  → Saved best model (val_loss: {best_val_loss:.6f})"
                print(save_msg)
                log_file.write(save_msg + '\n')
                log_file.flush()

        log_file.close()
        print(f"Training log saved to: {save_result_path}/logs/training_log.txt")
    elif args.model == 'cnn':
        # CNN Training Loop
        os.makedirs(save_result_path, exist_ok=True)
        log_file = open(f'{save_result_path}/training_log.txt', 'w')

        # History for plotting
        history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': [],
                   'train_ssim': [], 'val_ssim': [], 'learning_rate': []}

        for epoch in range(epochs):
            model.train()
            train_losses = {'total': [], 'mse': [], 'ssim': []}

            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
                # For generation, use noisy input -> clean target
                targets = batch['target'].to(device)
                # Add noise to create input
                noise_level = 0.1
                inputs = targets + torch.randn_like(targets) * noise_level

                optimizer.zero_grad()
                outputs = model(inputs)

                # Compute combined loss
                mse_loss = criterion(outputs, targets)
                if ssim_metric is not None:
                    pred_norm = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-8)
                    target_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
                    ssim_val = ssim_metric(pred_norm, target_norm)
                    ssim_loss = 1.0 - ssim_val
                    total_loss = mse_loss + ssim_weight * ssim_loss
                else:
                    ssim_loss = torch.tensor(0.0, device=device)
                    total_loss = mse_loss

                total_loss.backward()
                optimizer.step()

                train_losses['total'].append(total_loss.item())
                train_losses['mse'].append(mse_loss.item())
                train_losses['ssim'].append(ssim_loss.item())

            # Validation
            model.eval()
            val_losses = {'total': [], 'mse': [], 'ssim': []}
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                    targets = batch['target'].to(device)
                    inputs = targets + torch.randn_like(targets) * noise_level

                    outputs = model(inputs)

                    # Compute combined loss
                    mse_loss = criterion(outputs, targets)
                    if ssim_metric is not None:
                        pred_norm = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-8)
                        target_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
                        ssim_val = ssim_metric(pred_norm, target_norm)
                        ssim_loss = 1.0 - ssim_val
                        total_loss = mse_loss + ssim_weight * ssim_loss
                    else:
                        ssim_loss = torch.tensor(0.0, device=device)
                        total_loss = mse_loss

                    val_losses['total'].append(total_loss.item())
                    val_losses['mse'].append(mse_loss.item())
                    val_losses['ssim'].append(ssim_loss.item())

            avg_train_loss = sum(train_losses['total']) / len(train_losses['total'])
            avg_val_loss = sum(val_losses['total']) / len(val_losses['total'])
            avg_train_mse = sum(train_losses['mse']) / len(train_losses['mse'])
            avg_val_mse = sum(val_losses['mse']) / len(val_losses['mse'])
            avg_train_ssim = sum(train_losses['ssim']) / len(train_losses['ssim'])
            avg_val_ssim = sum(val_losses['ssim']) / len(val_losses['ssim'])

            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_mse'].append(avg_train_mse)
            history['val_mse'].append(avg_val_mse)
            history['train_ssim'].append(avg_train_ssim)
            history['val_ssim'].append(avg_val_ssim)

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)

            log_msg = (f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.6f} (MSE: {avg_train_mse:.6f}, SSIM: {avg_train_ssim:.6f}), "
                      f"Val Loss: {avg_val_loss:.6f} (MSE: {avg_val_mse:.6f}, SSIM: {avg_val_ssim:.6f}), "
                      f"LR: {current_lr:.2e}")
            print(log_msg)
            log_file.write(log_msg + '\n')

        torch.save(model.state_dict(), f'{save_result_path}/best_model.pth')
        print("Saved final model.")
        log_file.close()

        # Plot loss curves
        plot_path = os.path.join(f'{save_result_path}/logs', 'loss_curves.png')
        os.makedirs(f'{save_result_path}/logs', exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs_range = range(1, epochs + 1)

        # Total Loss
        axes[0, 0].plot(epochs_range, history['train_loss'], 'b-o', label='Train Total', linewidth=2, markersize=4)
        axes[0, 0].plot(epochs_range, history['val_loss'], 'r-s', label='Val Total', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss (MSE + SSIM)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MSE Loss
        axes[0, 1].plot(epochs_range, history['train_mse'], 'b-o', label='Train MSE', linewidth=2, markersize=4)
        axes[0, 1].plot(epochs_range, history['val_mse'], 'r-s', label='Val MSE', linewidth=2, markersize=4)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].set_title('MSE Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # SSIM Loss
        axes[1, 0].plot(epochs_range, history['train_ssim'], 'b-o', label='Train SSIM', linewidth=2, markersize=4)
        axes[1, 0].plot(epochs_range, history['val_ssim'], 'r-s', label='Val SSIM', linewidth=2, markersize=4)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM Loss')
        axes[1, 0].set_title('SSIM Loss (1 - SSIM)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 1].plot(epochs_range, history['learning_rate'], 'g-^', linewidth=2, markersize=4)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss curves saved to: {plot_path}")

    print("Training finished.")
else:
    print("\nSkipping training and loading model directly.")


# --- Load Best Model and Generate Samples ---
print("\nLoading best model and generating samples...")
checkpoint_path = f'{save_result_path}/best_model.pth'
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found at {checkpoint_path}")
    import glob
    epochs_available = sorted([int(f.split('_')[-1].split('.')[0]) for f in glob.glob(f'{save_result_path}/best_model.pth')])
    if not epochs_available:
        print("No checkpoints found. Exiting.")
        exit()
    latest_epoch = epochs_available[-1]
    checkpoint_path = f'{save_result_path}/best_model.pth'
    print(f"Using latest found checkpoint: {checkpoint_path}")

if args.model == 'edm':
    checkpoint = torch.load(checkpoint_path)
    model.unet.load_state_dict(checkpoint['model_state_dict'])
    model.unet.eval()
elif args.model == 'ddim':
    model.load_saved_model(checkpoint_path)
    model.model.eval()
elif args.model == 'cnn':
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

# --- Generation Evaluation ---
print("\nStarting generation evaluation...")

num_test_samples = 20  # Generate 20 samples

print(f"Generating {num_test_samples} samples...")

# For generation, we also collect some real samples for reference (NOT as GT)
all_reference_samples = []
samples_collected = 0

for batch in test_loader:
    batch_size = batch['target'].shape[0]
    samples_to_take = min(batch_size, 5 - samples_collected)  # Just 5 reference samples
    all_reference_samples.append(batch['target'][:samples_to_take].to(device))
    samples_collected += samples_to_take
    if samples_collected >= 5:
        break

reference_samples = torch.cat(all_reference_samples, dim=0)[:5]

print(f"Reference samples shape: {reference_samples.shape}")

# Generate samples from scratch (starting from heavily filtered noise)
generated_samples = []
intermediate_generations = []

with torch.no_grad():
    for i in range(num_test_samples):
        if args.model == 'edm':
            # Generate from heavily filtered noise
            generated_frames = model.sampling(
                1, device,
                (reference_samples.shape[-2], reference_samples.shape[-1]),
                start_from_timestep=model.num_timesteps - 1
            )
            final_generation = generated_frames[0]  # t=0 result

            generated_samples.append(final_generation.cpu().numpy())
            intermediate_generations.append([g.cpu().numpy() for g in generated_frames])
        elif args.model == 'ddim':
            # DDIM generation (unconditional with Gaussian noise condition)
            gaussian_cond = torch.randn(1, C_out, reference_samples.shape[-2], reference_samples.shape[-1], device=device)
            final_generation = model.sampling(gaussian_cond)
            generated_samples.append(final_generation.cpu().numpy())
            intermediate_generations.append(None)
        elif args.model == 'cnn':
            # CNN denoising: Start from noisy input
            noise_level = 0.1
            noisy_input = torch.randn(1, C_out, reference_samples.shape[-2], reference_samples.shape[-1], device=device) * noise_level
            final_generation = model(noisy_input)
            generated_samples.append(final_generation.cpu().numpy())
            intermediate_generations.append(None)

# numpy 변환
reference_samples_np = reference_samples.cpu().numpy()

# --- Hallucination Detection ---
print(f"\n{'='*60}")
print("Running Hallucination Detection on Generated Samples...")
print(f"{'='*60}")

# Initialize detector with reference samples
# Use more reference samples for better statistics
all_reference_for_detector = []
ref_collected = 0
for batch in test_loader:
    batch_samples = batch['target'].to(device)
    all_reference_for_detector.append(batch_samples.cpu().numpy())
    ref_collected += batch_samples.shape[0]
    if ref_collected >= 50:  # Use 50 reference samples
        break

reference_for_detector = np.concatenate(all_reference_for_detector, axis=0)[:50]

detector = TurbulenceHallucinationDetector(
    reference_samples=reference_for_detector,
    sample_shape=(reference_samples.shape[-2], reference_samples.shape[-1])
)

# Evaluate all generated samples
print(f"Evaluating {num_test_samples} generated samples for hallucination...")
hallucination_results = []

for i, gen_sample in enumerate(generated_samples):
    sample = gen_sample[0]  # Remove batch dimension
    result = detector.unified_hallucination_score(sample)
    hallucination_results.append(result)

    if (i + 1) % 5 == 0:
        print(f"  Evaluated {i+1}/{num_test_samples} samples...")

# Aggregate results
aggregate_stats = aggregate_hallucination_results(hallucination_results)

print(f"\n{'='*60}")
print("Hallucination Detection Results:")
print(f"{'='*60}")
print(f"Hallucination Rate: {aggregate_stats['hallucination_rate']:.2%}")
print(f"Average Confidence: {aggregate_stats['avg_confidence']:.4f}")
print(f"\nAverage Component Scores:")
for component, score in aggregate_stats['avg_scores'].items():
    std = aggregate_stats['std_scores'][component]
    print(f"  {component:20s}: {score:.4f} ± {std:.4f}")

# Identify worst hallucinations
hallucination_scores = [r['total_score'] for r in hallucination_results]
worst_indices = np.argsort(hallucination_scores)[-5:][::-1]
print(f"\nTop 5 Likely Hallucinations:")
for rank, idx in enumerate(worst_indices, 1):
    result = hallucination_results[idx]
    print(f"  #{rank} Sample {idx+1}: score={result['total_score']:.4f}")

# --- Hallucination Visualization ---
print("\nGenerating hallucination analysis plots...")

# 1. Hallucination Score Distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
fig.suptitle('Hallucination Detection Analysis', fontsize=18, fontweight='bold')

# 1.1 Overall score distribution
ax = axes[0, 0]
ax.hist(hallucination_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(0.4, color='red', linestyle='--', linewidth=2, label='Threshold')
ax.axvline(np.mean(hallucination_scores), color='blue', linestyle='-', linewidth=2,
          label=f'Mean: {np.mean(hallucination_scores):.3f}')
ax.set_xlabel('Hallucination Score')
ax.set_ylabel('Frequency')
ax.set_title(f'Overall Score Distribution\nHallucination Rate: {aggregate_stats["hallucination_rate"]:.1%}')
ax.legend()
ax.grid(True, alpha=0.3)

# 1.2 Component scores comparison
ax = axes[0, 1]
components = ['spectral', 'structure', 'nearest_neighbor', 'statistical']
component_means = [aggregate_stats['avg_scores'][c] for c in components]
component_stds = [aggregate_stats['std_scores'][c] for c in components]

x_pos = np.arange(len(components))
ax.bar(x_pos, component_means, yerr=component_stds, alpha=0.7, color='teal', capsize=5)
ax.set_ylabel('Average Score')
ax.set_title('Component Score Breakdown')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Spectral', 'Structure', 'NN Distance', 'Statistical'], rotation=45, ha='right')
ax.axhline(0.4, color='red', linestyle='--', alpha=0.5, label='Threshold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 1.3 Scatter: Spectral vs Structure
ax = axes[0, 2]
spectral_scores = [r['component_scores']['spectral'] for r in hallucination_results]
structure_scores = [r['component_scores']['structure'] for r in hallucination_results]
is_halluc = [r['is_hallucination'] for r in hallucination_results]

colors = ['red' if h else 'green' for h in is_halluc]
ax.scatter(spectral_scores, structure_scores, c=colors, alpha=0.6, s=50)
ax.set_xlabel('Spectral Score')
ax.set_ylabel('Structure Function Score')
ax.set_title('Physics-based Metrics Correlation')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.6, label='Hallucination'),
    Patch(facecolor='green', alpha=0.6, label='Valid')
]
ax.legend(handles=legend_elements)

# 1.4 Show worst hallucinations
ax = axes[1, 0]
ax.axis('off')
text_content = "Top 5 Worst Hallucinations:\n\n"
for rank, idx in enumerate(worst_indices, 1):
    result = hallucination_results[idx]
    text_content += f"#{rank} Sample {idx+1}:\n"
    text_content += f"  Total: {result['total_score']:.3f}\n"
    text_content += f"  Spectral: {result['component_scores']['spectral']:.3f}\n"
    text_content += f"  Structure: {result['component_scores']['structure']:.3f}\n\n"

ax.text(0.1, 0.9, text_content, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 1.5 Comparison: Real vs Generated (worst sample)
ax = axes[1, 1]
worst_idx = worst_indices[0]
worst_gen = generated_samples[worst_idx][0, 0]
im1 = ax.imshow(worst_gen, cmap='RdBu_r', origin='lower', aspect='equal')
ax.set_title(f'Worst Hallucination (Sample {worst_idx+1})\nScore: {hallucination_results[worst_idx]["total_score"]:.3f}')
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im1, ax=ax, shrink=0.8)

# 1.6 Comparison: Best sample
ax = axes[1, 2]
best_indices = np.argsort(hallucination_scores)[:5]
best_idx = best_indices[0]
best_gen = generated_samples[best_idx][0, 0]
im2 = ax.imshow(best_gen, cmap='RdBu_r', origin='lower', aspect='equal')
ax.set_title(f'Best Sample (Sample {best_idx+1})\nScore: {hallucination_results[best_idx]["total_score"]:.3f}')
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im2, ax=ax, shrink=0.8)

plt.savefig(os.path.join(save_result_path, 'hallucination_analysis.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved hallucination analysis plot")

# 2. Detailed spectral analysis for worst cases
fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
fig.suptitle('Spectral Analysis: Worst Hallucinations vs Reference', fontsize=16, fontweight='bold')

for i, idx in enumerate(worst_indices[:3]):
    # Top row: Generated sample
    ax_img = axes[0, i]
    gen_sample = generated_samples[idx][0, 0]
    im = ax_img.imshow(gen_sample, cmap='RdBu_r', origin='lower', aspect='equal')
    ax_img.set_title(f'Sample {idx+1} (Score: {hallucination_results[idx]["total_score"]:.3f})')
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    plt.colorbar(im, ax=ax_img, shrink=0.8)

    # Bottom row: Energy spectrum
    ax_spec = axes[1, i]
    E_k_gen = detector._compute_energy_spectrum(gen_sample)
    E_k_ref = detector.reference_statistics['energy_spectrum_mean']

    k = np.arange(len(E_k_gen))
    k_ref = np.arange(len(E_k_ref))

    # Plot on log-log scale
    ax_spec.loglog(k[1:], E_k_gen[1:], 'b-', linewidth=2, label='Generated', alpha=0.7)
    ax_spec.loglog(k_ref[1:], E_k_ref[1:], 'r--', linewidth=2, label='Reference Mean', alpha=0.7)

    # Add -5/3 slope reference
    k_inertial = np.arange(10, 50)
    E_kolmogorov = E_k_gen[10] * (k_inertial / 10)**(-5/3)
    ax_spec.loglog(k_inertial, E_kolmogorov, 'k:', linewidth=1.5, label='k^(-5/3)', alpha=0.5)

    ax_spec.set_xlabel('Wavenumber k')
    ax_spec.set_ylabel('Energy E(k)')
    ax_spec.set_title(f'Energy Spectrum')
    ax_spec.legend()
    ax_spec.grid(True, alpha=0.3)

plt.savefig(os.path.join(save_result_path, 'hallucination_spectral_details.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved spectral details plot")

# --- Visualization: Generated Samples Grid ---
print("\nGenerating visualization plots...")

# 1. Show all generated samples in a grid
samples_per_row = 5
num_rows = (num_test_samples + samples_per_row - 1) // samples_per_row

fig, axes = plt.subplots(num_rows, samples_per_row,
                        figsize=(3 * samples_per_row, 3 * num_rows),
                        squeeze=False, constrained_layout=True)

fig.suptitle(f'Generated Samples Grid ({num_test_samples} samples)', fontsize=18, fontweight='bold')

# Calculate global color scale from all generated samples
all_gen_values = np.concatenate([gen[0, 0] for gen in generated_samples])
vmin_global = np.percentile(all_gen_values, 1)
vmax_global = np.percentile(all_gen_values, 99)

for idx in range(num_test_samples):
    row = idx // samples_per_row
    col = idx % samples_per_row
    ax = axes[row, col]

    gen_data = generated_samples[idx][0, 0]  # First channel

    im = ax.imshow(gen_data, cmap='RdBu_r', origin='lower', aspect='equal',
                  vmin=vmin_global, vmax=vmax_global)
    ax.set_title(f'Sample {idx+1}', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)

# Hide empty subplots
for idx in range(num_test_samples, num_rows * samples_per_row):
    row = idx // samples_per_row
    col = idx % samples_per_row
    axes[row, col].axis('off')

plt.savefig(os.path.join(save_result_path, 'generated_samples_grid.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved generated samples grid")

# 2. Compare Generated vs Reference samples
fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
fig.suptitle('Generated vs Reference Samples (for diversity comparison)', fontsize=16, fontweight='bold')

# Top row: 5 reference samples
for i in range(5):
    ax = axes[0, i]
    ref_data = reference_samples_np[i, 0]  # First channel
    im = ax.imshow(ref_data, cmap='RdBu_r', origin='lower', aspect='equal',
                  vmin=vmin_global, vmax=vmax_global)
    ax.set_title(f'Reference {i+1}', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)

# Bottom row: 5 generated samples
for i in range(5):
    ax = axes[1, i]
    gen_data = generated_samples[i][0, 0]  # First channel
    im = ax.imshow(gen_data, cmap='RdBu_r', origin='lower', aspect='equal',
                  vmin=vmin_global, vmax=vmax_global)
    ax.set_title(f'Generated {i+1}', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.savefig(os.path.join(save_result_path, 'generated_vs_reference.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved generated vs reference comparison")

# --- Calculate Generation Quality Metrics ---
print("\nCalculating generation quality statistics...")

# Calculate statistics of generated samples
gen_array = np.array([gen[0] for gen in generated_samples])  # (N, C, H, W)

generation_stats = {
    'mean': np.mean(gen_array, axis=0),  # (C, H, W)
    'std': np.std(gen_array, axis=0),    # (C, H, W)
    'min': np.min(gen_array, axis=0),    # (C, H, W)
    'max': np.max(gen_array, axis=0),    # (C, H, W)
}

# Calculate diversity metrics (variance between samples)
diversity_metrics = {
    'inter_sample_variance': np.var(gen_array, axis=0).mean(),  # Higher = more diverse
    'mean_abs_diff': np.mean([np.mean(np.abs(gen_array[i] - gen_array[j]))
                              for i in range(len(gen_array))
                              for j in range(i+1, len(gen_array))]),
}

# Compare generated distribution with reference distribution
ref_array = reference_samples_np  # (5, C, H, W)

distribution_comparison = {
    'gen_mean': np.mean(gen_array),
    'gen_std': np.std(gen_array),
    'ref_mean': np.mean(ref_array),
    'ref_std': np.std(ref_array),
    'mean_diff': np.abs(np.mean(gen_array) - np.mean(ref_array)),
    'std_diff': np.abs(np.std(gen_array) - np.std(ref_array)),
}

# --- Generation Quality Visualization ---
print("\nGenerating quality analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
fig.suptitle('Generation Quality Analysis', fontsize=16, fontweight='bold')

# 1. Value distribution comparison (Generated vs Reference)
ax = axes[0, 0]
ax.hist(gen_array.flatten(), bins=50, alpha=0.6, color='blue', label='Generated', density=True)
ax.hist(ref_array.flatten(), bins=50, alpha=0.6, color='red', label='Reference', density=True)
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Value Distribution: Generated vs Reference')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Inter-sample variance (diversity)
ax = axes[0, 1]
inter_sample_std = np.std(gen_array, axis=0).mean(axis=(1, 2))  # Std across samples for each channel
ax.bar(range(C_out), inter_sample_std, color='purple', alpha=0.7)
ax.set_xlabel('Channel')
ax.set_ylabel('Inter-Sample Std Dev')
ax.set_title(f'Sample Diversity (Avg: {diversity_metrics["inter_sample_variance"]:.6f})')
ax.set_xticks(range(C_out))
ax.set_xticklabels([f'Ch{i+1}' for i in range(C_out)])
ax.grid(True, alpha=0.3, axis='y')

# 3. Mean and Std comparison
ax = axes[1, 0]
x = ['Generated', 'Reference']
means = [distribution_comparison['gen_mean'], distribution_comparison['ref_mean']]
stds = [distribution_comparison['gen_std'], distribution_comparison['ref_std']]

x_pos = np.arange(len(x))
ax.bar(x_pos - 0.2, means, 0.4, label='Mean', alpha=0.7, color='blue')
ax.bar(x_pos + 0.2, stds, 0.4, label='Std Dev', alpha=0.7, color='orange')
ax.set_ylabel('Value')
ax.set_title('Distribution Statistics')
ax.set_xticks(x_pos)
ax.set_xticklabels(x)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. Spatial statistics
ax = axes[1, 1]
# Calculate spatial mean and std for first channel
spatial_mean = generation_stats['mean'][0]  # (H, W)
im = ax.imshow(spatial_mean, cmap='RdBu_r', origin='lower', aspect='equal')
ax.set_title('Spatial Mean of Generated Samples (Ch1)')
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im, ax=ax, label='Mean Value')

plt.savefig(os.path.join(save_result_path, 'generation_quality_analysis.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved generation quality analysis plot")

# --- Save Test Results ---
print("\nSaving test results for postprocessing...")

results_dict = {
    'model_type': 'unconditional_generation',
    'reference_samples': reference_samples_np,
    'generated_samples': np.array([gen[0] for gen in generated_samples]),
    'intermediate_generations': None,
    'generation_stats': generation_stats,
    'diversity_metrics': diversity_metrics,
    'distribution_comparison': distribution_comparison,
    'hallucination_detection': {
        'aggregate_stats': aggregate_stats,
        'individual_results': hallucination_results,
        'worst_indices': worst_indices.tolist(),
        'best_indices': best_indices.tolist()
    },
    'model_config': {
        'model_name': args.model,
        'num_timesteps': num_timesteps,
        'max_cutoff': args.max_cutoff,
        'min_cutoff': args.min_cutoff,
        'C_out': C_out
    }
}

if args.model == 'edm' and intermediate_generations[0] is not None:
    intermediate_gens = []
    for sample_gens in intermediate_generations:
        sample_intermediate = np.array([step_gen[0] for step_gen in sample_gens])
        intermediate_gens.append(sample_intermediate)
    results_dict['intermediate_generations'] = np.array(intermediate_gens)

np.save(os.path.join(save_result_path, 'generation_results.npy'), results_dict)
print(f"Generation results saved to {os.path.join(save_result_path, 'generation_results.npy')}")

print(f"\n{'='*60}")
print(f"Unconditional Generation Evaluation Complete!")
print(f"{'='*60}")
print(f"Model: {args.model.upper()}")
print(f"Generated {num_test_samples} samples")
print(f"Output channels: {C_out}")

print(f"\n{'='*60}")
print(f"Generation Quality Metrics:")
print(f"{'='*60}")
print(f"\nDistribution Statistics:")
print(f"  Generated - Mean: {distribution_comparison['gen_mean']:.6f}, Std: {distribution_comparison['gen_std']:.6f}")
print(f"  Reference - Mean: {distribution_comparison['ref_mean']:.6f}, Std: {distribution_comparison['ref_std']:.6f}")
print(f"  Difference - Mean: {distribution_comparison['mean_diff']:.6f}, Std: {distribution_comparison['std_diff']:.6f}")

print(f"\nDiversity Metrics:")
print(f"  Inter-sample Variance: {diversity_metrics['inter_sample_variance']:.6f}")
print(f"  Mean Absolute Difference: {diversity_metrics['mean_abs_diff']:.6f}")

print(f"\nGenerated Sample Statistics:")
print(f"  Global Mean: {np.mean(gen_array):.6f}")
print(f"  Global Std: {np.std(gen_array):.6f}")
print(f"  Global Min: {np.min(gen_array):.6f}")
print(f"  Global Max: {np.max(gen_array):.6f}")

print(f"\n{'='*60}")
print(f"Hallucination Detection Summary:")
print(f"{'='*60}")
print(f"  Hallucination Rate: {aggregate_stats['hallucination_rate']:.1%}")
print(f"  Average Score: {aggregate_stats['avg_scores']['total']:.4f} ± {aggregate_stats['std_scores']['total']:.4f}")
print(f"  Worst Sample: #{worst_indices[0]+1} (score: {hallucination_results[worst_indices[0]]['total_score']:.4f})")
print(f"  Best Sample: #{best_indices[0]+1} (score: {hallucination_results[best_indices[0]]['total_score']:.4f})")

print(f"\n{'='*60}")
print(f"Results saved in: {save_result_path}")
print(f"  - generated_samples_grid.png")
print(f"  - generated_vs_reference.png")
print(f"  - generation_quality_analysis.png")
print(f"  - hallucination_analysis.png")
print(f"  - hallucination_spectral_details.png")
print(f"  - generation_results.npy (for postprocessing)")

# --- Energy Cascade Analysis for Generations ---
print(f"\n{'='*60}")
print("Generating energy cascade analysis for generations (Reference vs Generated)...")
print(f"{'='*60}")

model_predictions = {
    'generated': np.array([gen[0] for gen in generated_samples])
}

energy_results = plot_energy_cascade_analysis(
    gt_data=reference_samples_np,
    model_predictions=model_predictions,
    save_path=save_result_path,
    experiment_name=f'energy_cascade_generation'
)

print(f"Energy cascade analysis completed!")
print(f"  - energy_cascade_generation_comparison.png")
print(f"  - Analyzed {energy_results['n_samples_analyzed']} samples")
print(f"  - Note: Comparison is with reference samples, not ground truth")
print(f"{'='*60}")
