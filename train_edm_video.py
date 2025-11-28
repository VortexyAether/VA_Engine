from model.unet import UNet
from model.edm_single import EDM, FrequencyProgressiveUNet
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
import argparse
from PIL import Image


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class VideoSequenceDataset(torch.utils.data.Dataset):
    """
    Video prediction dataset: 3 input timesteps -> 1 output timestep
    Data shape: (timesteps, 1, x_grid, y_grid)
    """
    def __init__(self, data, seq_len=3):
        """
        Parameters:
        data: numpy array of shape (timesteps, 1, x_grid, y_grid)
        seq_len: number of input timesteps (default=3)
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len

    def __len__(self):
        # We can create (total_timesteps - seq_len) samples
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Input: seq_len consecutive timesteps
        input_seq = self.data[idx:idx+self.seq_len]  # (seq_len, C, H, W)
        # Output: next timestep
        target = self.data[idx+self.seq_len]  # (C, H, W)

        # Reshape input: (seq_len, C, H, W) -> (seq_len*C, H, W)
        # For multi-channel data (e.g., u,v,w velocities):
        #   (3, 3, H, W) -> (9, H, W) where channels = [t1_u, t1_v, t1_w, t2_u, t2_v, t2_w, t3_u, t3_v, t3_w]
        # For single-channel data:
        #   (3, 1, H, W) -> (3, H, W)
        num_timesteps, num_channels = input_seq.shape[:2]
        input_seq = input_seq.reshape(num_timesteps * num_channels, *input_seq.shape[2:])

        return {
            'input': input_seq,  # (seq_len*C, H, W)
            'target': target,    # (C, H, W)
            'index': idx
        }


def create_video_dataloaders(data_path, batch_size=8, seq_len=3,
                             train_ratio=0.6, val_ratio=0.2):
    """
    Create train/val/test dataloaders for video prediction

    Parameters:
    data_path: path to Y.npy file
    batch_size: batch size
    seq_len: number of input timesteps
    train_ratio: 0.6 (first 60%)
    val_ratio: 0.2 (middle 20%)
    test_ratio: 0.2 (last 20%)
    """
    # Load data
    data = np.load(data_path).astype(np.float32)
    print(f"Loaded data shape: {data.shape}")

    # Ensure 4D: (timesteps, channels, x_grid, y_grid)
    if len(data.shape) == 3:
        # Single channel data: (T, H, W) -> (T, 1, H, W)
        data = data[:, np.newaxis, :, :]
        print(f"Added channel dimension: {data.shape}")
    elif len(data.shape) == 4:
        # Multi-channel data already in correct format: (T, C, H, W)
        print(f"Multi-channel data detected: {data.shape[1]} channels")
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}. Expected (T, H, W) or (T, C, H, W)")

    total_timesteps = len(data)
    train_end = int(total_timesteps * train_ratio)
    val_end = int(total_timesteps * (train_ratio + val_ratio))

    # Split data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"Data split: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")

    # Create datasets
    train_dataset = VideoSequenceDataset(train_data, seq_len=seq_len)
    val_dataset = VideoSequenceDataset(val_data, seq_len=seq_len)
    test_dataset = VideoSequenceDataset(test_data, seq_len=seq_len)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)

    print(f"Dataset created: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")

    return train_loader, val_loader, test_loader, data

############################ Hyperparameter ############################
# Create an argument parser
parser = argparse.ArgumentParser(description='Train a model for video frame prediction.')
parser.add_argument('--model', type=str, default='edm', choices=['edm', 'cnn', 'ddim'], help='The model to train (edm, cnn, or ddim).')
parser.add_argument('--epochs', type=int, default=10, help='The number of training epochs.')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'load'], help='Set mode to train or load.')
parser.add_argument('--dataset', type=str, default='decaying_snapshot', help='Select the datasets')
parser.add_argument('--num_unets', type=int, default=3, help='The number of U-Net models in the refinement chain.')
parser.add_argument('--max_cutoff', type=float, default=0.8, help='The maximum cutoff frequency')
parser.add_argument('--min_cutoff', type=float, default=0.02, help='The minimum cutoff frequency')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='default learning rate')
parser.add_argument('--train_ratio', type=float, default=0.6, help='train_ratio')
parser.add_argument('--val_ratio', type=float, default=0.2, help='val_ratio')
args = parser.parse_args()

epochs = args.epochs
dataset = args.dataset
model_name = args.model
num_unets = args.num_unets
batch_size = args.batch_size
learning_rate = args.learning_rate
train_ratio = args.train_ratio
val_ratio = args.val_ratio
test_ratio = 1 - train_ratio - val_ratio

save_result_path = f'test/result_{model_name}_{dataset}_video'

os.makedirs(save_result_path, exist_ok=True)
with open(os.path.join(save_result_path, 'parameters.txt'), 'w') as f:
    for arg, value in vars(args).items():
        f.write(f'{arg}: {value}\n')


num_timesteps = num_unets + 1
###########################################################################

# initializing model
print(f'Initializing {args.model.upper()} model....')
torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Dataset load
data_dir = f'/home/navier/Dataset/various_CFD/{dataset}/'
#data_dir = f'/mnt/hdd3/home/ddfe/jjw/Dataset/various_CFD/{dataset}/'
data_path = os.path.join(data_dir, f'{dataset}.npy')

train_loader, val_loader, test_loader, full_data = create_video_dataloaders(
        data_path,
        batch_size=batch_size,
        seq_len=3,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )


batch_sample= next(iter(train_loader))


inputs_sample= batch_sample['input']
targets_sample = batch_sample['target']

print(inputs_sample.shape)
print(targets_sample.shape)
_, C_in, _, _ = inputs_sample.shape
_, C_out, _, _ = targets_sample.shape

print(f"Input shape: {inputs_sample.shape}, Target shape: {targets_sample.shape}")
print(f"C_in: {C_in}, C_out: {C_out}")

if args.model == 'edm':
    # For video prediction with EDM: 3 condition frames + 1 noisy state from previous timestep
    in_channels = C_in + C_out  # 3 (condition) + 1 (noisy state at timestep t)
    out_channels = C_out  # 1 (denoised output)
    unet = FrequencyProgressiveUNet(
        in_channels=in_channels,  # 3 + 1 = 4
        out_channels=out_channels,  # 1
        base_channels=64,
        time_emb_dim=256
    ).cuda()
    model = EDM(unet=unet,
                num_timesteps=num_timesteps,
                lr=learning_rate,
                max_freq_ratio=args.max_cutoff,
                min_freq_ratio=args.min_cutoff).to(device)
    print(f'Successfully initialized EDM model for video prediction!\n')
    print(f'Input channels: {in_channels} (3 condition + 1 noisy state), Output channels: {out_channels}\n')
elif args.model == 'ddim':
    # For video prediction with DDIM: 3 condition frames + 1 noisy state from previous timestep
    in_channels = C_in + C_out  # 3 (condition) + 1 (noisy state at timestep t)
    out_channels = C_out  # 1 (denoised output)
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

    print(f'Successfully initialized DDIM model for video prediction!\n')
    print(f'Input channels: {in_channels} (3 condition + 1 noisy state), Output channels: {out_channels}\n')
elif args.model == 'cnn':
    # CNN is not a diffusion model - just 3 input frames directly
    in_channels = C_in  # 3 condition frames only (no noisy state)
    out_channels = C_out  # 1 output frame
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    print(f'Successfully initialized CNN model for video prediction!\n')
    print(f'Input channels: {in_channels} (3 condition frames only), Output channels: {out_channels}\n')
else:
    print('You didnt select right model name..')



if args.model == 'edm':
# --- Monitoring q_sample ---
    print(f"\nMonitoring EDM q_sample for {model.num_timesteps} DDIM steps...")
    NUM_SAMPLES_TO_MONITOR = 2
    NUM_T_STEPS_TO_PLOT = model.num_timesteps 

    # 이미 텐서이므로 바로 스택만 하면 됨!
    #indices = torch.randint(0, len(train_dataset), (NUM_SAMPLES_TO_MONITOR,))

# 간단하게 추출
    # train_loader에서 배치들을 수집해서 충분한 샘플 모으기
    all_inputs = []
    all_targets = []

    for batch in train_loader:
        all_inputs.append(batch['input'].to(device))
        all_targets.append(batch['target'].to(device))

        # 충분한 샘플이 모이면 중단
        total_samples = sum(inp.shape[0] for inp in all_inputs)
        if total_samples >= NUM_SAMPLES_TO_MONITOR:
            break

      # 모든 배치를 하나로 합치기
    train_inputs = torch.cat(all_inputs, dim=0)   # (total_samples, 2, 1, H, W)
    train_targets = torch.cat(all_targets, dim=0) # (total_samples, 4, 1, H, W)
    
    # 필요한 만큼만 랜덤 선택
    if len(train_inputs) > NUM_SAMPLES_TO_MONITOR:
        indices = torch.randperm(len(train_inputs))[:NUM_SAMPLES_TO_MONITOR]
        train_inputs = train_inputs[indices]
        train_targets = train_targets[indices]
    
    # 모니터링용 샘플
    x0_samples = train_targets    # (10, 4, 1, H, W)
    cond_samples = train_inputs   # (10, 2, 1, H, W)


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

## --- Plotting q_sample ---
#    os.makedirs(save_result_path, exist_ok=True)
# --- Plotting q_sample ---
    os.makedirs(save_result_path, exist_ok=True)
    
    channel_names = [f'Channel {i+1}' for i in range(C_out)]
    
    fig, axes = plt.subplots(NUM_SAMPLES_TO_MONITOR* C_out, NUM_T_STEPS_TO_PLOT,
                             figsize=(3 * NUM_T_STEPS_TO_PLOT, 3.2 * NUM_SAMPLES_TO_MONITOR * C_out),
                             squeeze=False, constrained_layout=True)
    fig.suptitle('EDM Forward Process (q_sample) Visualization', fontsize=20)

    for i in range(NUM_SAMPLES_TO_MONITOR):
        for ch_idx, ch_name in enumerate(channel_names):
            for j, t_idx in enumerate(plot_t_indices):
                ax_row = i * C_out + ch_idx
                ax = axes[ax_row, j]
                
                # Extract the specific channel
                img_data = all_snapshots_by_sample[i][j][0, ch_idx, :, :]
                
                # Determine color limits for this specific channel from the original targets
                vmin = train_targets[:, ch_idx, :, :].min()
                vmax = train_targets[:, ch_idx, :, :].max()
                
                im = ax.imshow(img_data.T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
                
                # Add a colorbar to each subplot
                fig.colorbar(im, ax=ax, shrink=0.8)
                
                ax.set_title(f't = {t_idx}', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                
                if j == 0:
                    ax.set_ylabel(f'Sample {i+1}\n{ch_name}', rotation=0, size='large', labelpad=40)

    plot_filename = os.path.join(save_result_path, "q_sample_monitoring_result.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Monitoring plot saved to '{plot_filename}'")

    # --- Energy Cascade Analysis for q_sample ---
    print("\nGenerating energy cascade analysis for q_sample forward process...")
    # Analyze first sample for energy spectra evolution
    plot_q_sample_energy_cascade(
        q_sample_snapshots=all_snapshots_by_sample[0],  # First sample
        timesteps=plot_t_indices,
        save_path=save_result_path,
        experiment_name='q_sample_energy_cascade',
        sample_idx=0,
        channel_idx=0  # First channel
    )
    print("Energy cascade analysis for q_sample completed.")

# --- Start Training ---
if args.mode == 'train':
    print("\nStarting training...")
    if args.model == 'edm':
        model.train(train_loader, val_loader,  epochs, f'{save_result_path}/best_model.pth', f'{save_result_path}/logs')
    elif args.model == 'ddim':
        model.train(train_loader, val_loader, epochs, f'{save_result_path}/best_model.pth', f'{save_result_path}/logs')
    else:
        os.makedirs(save_result_path, exist_ok=True)
        log_file = open(f'{save_result_path}/training_log.txt', 'w')
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_train_loss = 0.0
            total_val_loss =0.0

            for batch in tqdm(train_loader):
                cond = batch['input'].to(device)
                targets = batch['target'].to(device)

                optimizer.zero_grad()
                #dummy_time = torch.zeros(inputs.size(0), device=device)
                outputs = model(cond)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                
            model.eval()
            for batch in tqdm(val_loader):
                cond = batch['input'].to(device)
                targets = batch['target'].to(device)

                #optimizer.zero_grad()
                #dummy_time = torch.zeros(inputs.size(0), device=device)
                outputs = model(cond)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}")
            log_file.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}\n")
        torch.save(model.state_dict(), f'{save_result_path}/best_model.pth')
        print("Saved final model.")
        log_file.close()
    print("Training finished.")
else:
    print("\nSkipping training and loading model directly.")


#-=--0-------Visualization section-----------_#

# --- Load Best Model and Generate Samples ---
print("\nLoading best model and generating samples...")
# Sanity check: use a specific epoch for loading, you might need to change this
checkpoint_path = f'{save_result_path}/best_model.pth'
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found at {checkpoint_path}, using a fallback or exiting.")
    # Find the latest epoch
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
else:
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

# --- Autoregressive Video Prediction Test ---
print("\nStarting autoregressive video prediction test...")

# Get initial 3 timesteps from test data
# Use the test split from full_data
train_ratio = args.train_ratio
val_ratio = args.val_ratio
total_timesteps = len(full_data)
train_end = int(total_timesteps * train_ratio)
val_end = int(total_timesteps * (train_ratio + val_ratio))
test_data = full_data[val_end:]

# Get first 3 timesteps as initial condition
initial_frames = torch.FloatTensor(test_data[:3]).to(device)  # (3, C, H, W)
print(f"Initial frames shape: {initial_frames.shape}")

# Ground truth for comparison (first 100 timesteps of test set)
num_pred_steps = min(100, len(test_data) - 3)
gt_sequence = test_data[3:3+num_pred_steps]  # (100, C, H, W)
print(f"Predicting {num_pred_steps} timesteps...")

# Autoregressive prediction
predictions_list = []
# Reshape initial frames: (3, C, H, W) -> (3*C, H, W)
seq_len, num_channels = initial_frames.shape[:2]
current_input = initial_frames.reshape(seq_len * num_channels, *initial_frames.shape[2:])
print(f"Reshaped current_input shape: {current_input.shape}")  # (3*C, H, W)

with torch.no_grad():
    for step in tqdm(range(num_pred_steps), desc="Autoregressive prediction"):
        # Prepare input: (1, 3*C, H, W) or (1, C, H, W) for single channel
        input_batch = current_input.unsqueeze(0)

        if args.model == 'edm':
            # EDM sampling
            predicted_frame = model.sampling(1, device,
                                           (input_batch.shape[-2], input_batch.shape[-1]),
                                           cond=input_batch)[0]  # (1, C, H, W)
        elif args.model == 'ddim':
            predicted_frame = model.sampling(input_batch)  # (1, C, H, W)
        else:
            predicted_frame = model(input_batch)  # (1, C, H, W)

        predictions_list.append(predicted_frame.cpu().numpy())

        # Update input: shift sliding window and add new prediction
        # For multi-channel:
        #   current_input: (3*C, H, W) = [t1_ch1, t1_ch2, ..., t2_ch1, t2_ch2, ..., t3_ch1, t3_ch2, ...]
        #   Remove first C channels (t1), add new C channels (pred_t4) at the end
        # For single-channel:
        #   current_input: (3, H, W) = [t1, t2, t3]
        #   Remove first 1 channel (t1), add new 1 channel (pred_t4)
        new_frame = predicted_frame.squeeze(0)  # (C, H, W)
        current_input = torch.cat([current_input[num_channels:], new_frame], dim=0)  # (3*C, H, W)

# Stack predictions
predictions_np = np.array(predictions_list)  # (num_pred_steps, 1, C, H, W)
predictions_np = predictions_np.squeeze(1)  # (num_pred_steps, C, H, W)

print(f"Predictions shape: {predictions_np.shape}")
print(f"Ground truth shape: {gt_sequence.shape}")

# For compatibility with old code structure
test_inputs = initial_frames.unsqueeze(0).cpu()  # (1, 3, 1, H, W)
test_targets = torch.FloatTensor(gt_sequence[:15]).unsqueeze(0) if len(gt_sequence) >= 15 else torch.FloatTensor(gt_sequence).unsqueeze(0)  # dummy
num_test_samples = 1

print(f"Test input shape: {test_inputs.shape}")
print(f"Test target shape: {test_targets.shape}")

# --- Generate GIF ---
print("\nGenerating GIF animation...")


def create_video_gif(predictions, ground_truth, initial_frames, save_path, fps=10):
    """
    Create GIF animation comparing prediction and ground truth

    Parameters:
    predictions: (T, 1, H, W)
    ground_truth: (T, 1, H, W)
    initial_frames: (3, 1, H, W)
    save_path: path to save GIF
    fps: frames per second
    """
    T = len(predictions)
    frames = []

    # Combine initial + predictions for visualization
    all_pred = np.concatenate([initial_frames, predictions], axis=0)  # (T+3, 1, H, W)
    all_gt = np.concatenate([initial_frames, ground_truth], axis=0)  # (T+3, 1, H, W)

    # Normalize for visualization (GT 기준으로만)
    vmin = all_gt.min()
    vmax = all_gt.max()

    for t in range(T):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Prediction
        pred_frame = predictions[t, 0, :, :]
        axes[0].imshow(pred_frame.T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Prediction (t={t+1})', fontsize=14)
        axes[0].axis('off')

        # Ground Truth
        gt_frame = ground_truth[t, 0, :, :]
        axes[1].imshow(gt_frame.T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Ground Truth (t={t+1})', fontsize=14)
        axes[1].axis('off')

        # Error
        error = np.abs(pred_frame - gt_frame)
        im = axes[2].imshow(error.T, origin='lower', cmap='Reds', vmin=0, vmax=(vmax-vmin)*0.3)
        mae = np.mean(error)
        axes[2].set_title(f'Error (MAE={mae:.4f})', fontsize=14)
        axes[2].axis('off')

        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close()
        buf.close()

    # Save as GIF
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=1000//fps, loop=0)
    print(f"GIF saved to {save_path}")


# Create GIF
gif_path = os.path.join(save_result_path, 'video_prediction.gif')
create_video_gif(predictions_np, gt_sequence, test_data[:3], gif_path, fps=10)

# Also create a comparison plot for selected timesteps
print("\nGenerating comparison plots for selected timesteps...")
selected_steps = [0, num_pred_steps//4, num_pred_steps//2, 3*num_pred_steps//4, num_pred_steps-1]

fig, axes = plt.subplots(3, len(selected_steps), figsize=(4*len(selected_steps), 12))

vmin = gt_sequence.min()  # GT 기준으로만
vmax = gt_sequence.max()

for idx, step in enumerate(selected_steps):
    # Prediction
    axes[0, idx].imshow(predictions_np[step, 0, :, :].T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, idx].set_title(f'Pred t={step+1}')
    axes[0, idx].axis('off')

    # Ground truth
    axes[1, idx].imshow(gt_sequence[step, 0, :, :].T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, idx].set_title(f'GT t={step+1}')
    axes[1, idx].axis('off')

    # Error
    error = np.abs(predictions_np[step, 0, :, :] - gt_sequence[step, 0, :, :])
    im = axes[2, idx].imshow(error.T, origin='lower', cmap='Reds', vmin=0, vmax=(vmax-vmin)*0.3)
    mae = np.mean(error)
    axes[2, idx].set_title(f'Error (MAE={mae:.4f})')
    axes[2, idx].axis('off')

axes[0, 0].set_ylabel('Prediction', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Ground Truth', fontsize=14, fontweight='bold')
axes[2, 0].set_ylabel('Error', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_result_path, 'video_prediction_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Comparison plot saved")

# Calculate metrics over all timesteps
all_mae = []
all_mse = []
for t in range(num_pred_steps):
    mae = np.mean(np.abs(predictions_np[t] - gt_sequence[t]))
    mse = np.mean((predictions_np[t] - gt_sequence[t])**2)
    all_mae.append(mae)
    all_mse.append(mse)

# Plot error evolution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, num_pred_steps+1), all_mae, 'o-', linewidth=2, markersize=4)
axes[0].set_xlabel('Timestep', fontsize=12)
axes[0].set_ylabel('MAE', fontsize=12)
axes[0].set_title('MAE Evolution Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, num_pred_steps+1), all_mse, 's-', color='red', linewidth=2, markersize=4)
axes[1].set_xlabel('Timestep', fontsize=12)
axes[1].set_ylabel('MSE', fontsize=12)
axes[1].set_title('MSE Evolution Over Time', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_result_path, 'error_evolution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Error evolution plot saved")

print(f"\n{'='*60}")
print(f"Video Prediction Evaluation Complete!")
print(f"{'='*60}")
print(f"Model: {args.model.upper()}")
print(f"Predicted {num_pred_steps} timesteps autoregressively")
print(f"\nOverall Metrics:")
print(f"  Average MAE: {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
print(f"  Average MSE: {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
print(f"\nResults saved in: {save_result_path}")
print(f"  - video_prediction.gif")
print(f"  - video_prediction_comparison.png")
print(f"  - error_evolution.png")
print(f"{'='*60}")

# All old visualization code below is commented out for video prediction

# --- OLD SNAPSHOT-TO-SNAPSHOT VISUALIZATION CODE (NOT USED FOR VIDEO) ---
if False and args.model == 'edm':
    print("\nGenerating EDM timestep comparison plots...")
    
    # 3개 샘플에 대해 timestep별 비교 visualization
    num_comparison_samples = min(3, num_test_samples)
    
    for sample_idx in range(num_comparison_samples):
        fig, axes = plt.subplots(4, model.num_timesteps + 1, 
                                figsize=(3 * (model.num_timesteps + 1), 12), 
                                squeeze=False, constrained_layout=True)
        
        fig.suptitle(f'EDM Timestep Comparison - Sample {sample_idx + 1}', fontsize=16, fontweight='bold')
        
        # 첫 번째 채널만 시각화 (다채널인 경우)
        channel_idx = 0
        
        for t_idx in range(model.num_timesteps + 1):
            # 1행: GT q_sample at timestep t
            gt_t = gt_q_samples[sample_idx][t_idx][0, channel_idx]  # [1, C, H, W] -> [H, W]
            
            # 2행: Predicted at timestep t (should match GT q_sample at same timestep)
            pred_t = pred_timesteps[sample_idx][t_idx][0, channel_idx]  # [1, C, H, W] -> [H, W]
            
            
            # 3행: Error (Pred - GT)
            error_t = np.abs(pred_t - gt_t)
            
            # 4행: Original GT (reference)
            original_gt = test_targets_np[sample_idx, channel_idx]
            
            # 컬러맵 범위 설정
            vmin = min(gt_t.min(), pred_t.min(), original_gt.min())
            vmax = max(gt_t.max(), pred_t.max(), original_gt.max())
            
            # GT q_sample 플롯
            ax_gt = axes[0, t_idx]
            im = ax_gt.imshow(gt_t.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
            ax_gt.set_title(f't={t_idx}\nGT q_sample', fontsize=10)
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            if t_idx == 0:
                ax_gt.set_ylabel('GT q_sample', fontsize=12, fontweight='bold')
            
            # Prediction 플롯
            ax_pred = axes[1, t_idx]
            im = ax_pred.imshow(pred_t.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
            ax_pred.set_title(f'Prediction', fontsize=10)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            if t_idx == 0:
                ax_pred.set_ylabel('Prediction', fontsize=12, fontweight='bold')
            
            # Error 플롯
            ax_err = axes[2, t_idx]
            im_err = ax_err.imshow(error_t.T, cmap='Reds', origin='lower', vmin=0, vmax=(vmax-vmin)*0.3)
            mae = np.mean(error_t)
            ax_err.set_title(f'MAE={mae:.4f}', fontsize=10)
            ax_err.set_xticks([])
            ax_err.set_yticks([])
            if t_idx == 0:
                ax_err.set_ylabel('|Pred - GT|', fontsize=12, fontweight='bold')
            
            # Original GT (reference) 플롯
            ax_orig = axes[3, t_idx]
            im = ax_orig.imshow(original_gt.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
            if t_idx == (model.num_timesteps + 1) // 2:  # 중앙에만 제목 표시
                ax_orig.set_title(f'Original GT (Reference)', fontsize=10)
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            if t_idx == 0:
                ax_orig.set_ylabel('Original GT', fontsize=12, fontweight='bold')
        
        # 컬러바 추가
        cbar_ax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Value')
        
        cbar_ax2 = fig.add_axes([0.94, 0.3, 0.01, 0.4])
        sm2 = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=(vmax-vmin)*0.3))
        sm2.set_array([])
        fig.colorbar(sm2, cax=cbar_ax2, label='Error')
        
        # 저장
        plt.savefig(os.path.join(save_result_path, f'edm_timestep_comparison_sample{sample_idx+1}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved EDM timestep comparison for sample {sample_idx + 1}")

# --- Visualization: Snapshot-to-Snapshot Results ---
# NOTE: This section is for snapshot-to-snapshot tasks, not video prediction
# Skipping for video prediction task
if False:  # Disabled for video prediction
    print("\nGenerating snapshot-to-snapshot visualization plots...")

    cases_per_page = 5  # 페이지당 케이스 수
    num_pages = 3       # 총 페이지 수

    for page_idx in range(num_pages):
        # 각 케이스는 C_out개 행을 가짐 (채널별로), 4개 열 (Input, GT, Pred, Error)
        fig, axes = plt.subplots(cases_per_page * C_out, 4,
                                figsize=(16, 4 * cases_per_page * C_out),
                                squeeze=False, constrained_layout=True)

    fig.suptitle(f'Snapshot-to-Snapshot Results - Page {page_idx + 1}/{num_pages}',
                 fontsize=18, fontweight='bold')

    for case_idx in range(cases_per_page):
        sample_idx = page_idx * cases_per_page + case_idx
        if sample_idx >= num_test_samples:
            # 빈 subplot들을 숨김
            for ch in range(C_out):
                for col in range(4):
                    axes[case_idx * C_out + ch, col].axis('off')
            continue

        input_data = test_inputs_np[sample_idx]    # (C_in, H, W)
        target_data = test_targets_np[sample_idx]  # (C_out, H, W)
        pred_data = predictions[sample_idx][0]     # (C_out, H, W)

        # 각 출력 채널별로 행 생성
        for ch in range(C_out):
            row_idx = case_idx * C_out + ch

            # GT 기준으로 컬러스케일 설정
            gt_channel = target_data[ch]
            vmin, vmax = gt_channel.min(), gt_channel.max()

            # 1. Input (해당 채널이 있으면 표시, 없으면 평균 또는 첫 번째 채널)
            ax_input = axes[row_idx, 0]
            if ch < C_in:
                input_channel = input_data[ch]
            else:
                # 입력 채널이 출력 채널보다 적으면 첫 번째 채널 또는 평균 사용
                input_channel = input_data[0] if C_in > 0 else np.zeros_like(gt_channel)

            # Input은 자체 컬러스케일 사용
            input_vmin, input_vmax = input_channel.min(), input_channel.max()
            im1 = ax_input.imshow(input_channel.T, cmap='RdBu_r', origin='lower',
                                 vmin=input_vmin, vmax=input_vmax)
            ax_input.set_title(f'Input Ch{min(ch+1, C_in)}', fontsize=11)
            ax_input.set_xticks([])
            ax_input.set_yticks([])

            if ch == 0:  # 첫 번째 채널에만 케이스 번호 표시
                ax_input.set_ylabel(f'Case {sample_idx + 1}', fontsize=12, fontweight='bold')

            # 2. Ground Truth
            ax_gt = axes[row_idx, 1]
            im2 = ax_gt.imshow(gt_channel.T, cmap='RdBu_r', origin='lower',
                              vmin=vmin, vmax=vmax)
            ax_gt.set_title(f'GT Ch{ch+1}', fontsize=11)
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])

            # 3. Prediction (GT와 동일한 컬러스케일)
            ax_pred = axes[row_idx, 2]
            pred_channel = pred_data[ch]
            im3 = ax_pred.imshow(pred_channel.T, cmap='RdBu_r', origin='lower',
                                vmin=vmin, vmax=vmax)
            ax_pred.set_title(f'Pred Ch{ch+1}', fontsize=11)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            # 4. Error
            ax_error = axes[row_idx, 3]
            error = np.abs(pred_channel - gt_channel)
            mae_ch = np.mean(error)
            mse_ch = np.mean(error**2)

            # 오차는 0부터 최대 오차까지
            error_vmax = max(error.max(), (vmax - vmin) * 0.1)  # 최소한 GT 범위의 10%
            im4 = ax_error.imshow(error.T, cmap='Reds', origin='lower',
                                 vmin=0, vmax=error_vmax)
            ax_error.set_title(f'Error Ch{ch+1}\nMAE: {mae_ch:.4f}', fontsize=10)
            ax_error.set_xticks([])
            ax_error.set_yticks([])

            # 각 행에 작은 컬러바 추가 (선택적)
            if ch == C_out - 1:  # 마지막 채널에만 컬러바 추가
                # GT/Pred용 컬러바
                cbar1 = plt.colorbar(im2, ax=[ax_gt, ax_pred], shrink=0.6, aspect=10)
                cbar1.set_label(f'Case {sample_idx + 1} Value', fontsize=10)

                # Error용 컬러바
                cbar2 = plt.colorbar(im4, ax=ax_error, shrink=0.6, aspect=10)
                cbar2.set_label('Error', fontsize=10)

        # 케이스 간 구분선
        if case_idx < cases_per_page - 1 and sample_idx + 1 < num_test_samples:
            separator_row = (case_idx + 1) * C_out - 1
            for col in range(4):
                axes[separator_row, col].axhline(y=-0.5, color='black',
                                               linewidth=2, clip_on=False)

    # 열 제목 추가
    axes[0, 0].text(0.5, 1.1, 'Input', transform=axes[0, 0].transAxes,
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[0, 1].text(0.5, 1.1, 'Ground Truth', transform=axes[0, 1].transAxes,
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[0, 2].text(0.5, 1.1, 'Prediction', transform=axes[0, 2].transAxes,
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[0, 3].text(0.5, 1.1, 'Absolute Error', transform=axes[0, 3].transAxes,
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 페이지 저장
    plt.savefig(os.path.join(save_result_path, f's2s_prediction_results_page{page_idx+1}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved page {page_idx + 1}")

# --- OLD VISUALIZATION SECTIONS (Disabled for video prediction) ---
# These sections are for snapshot-to-snapshot tasks and use different data structures
if False:
    # --- Compact Summary Visualization ---
    print("\nGenerating compact summary plot...")

    # 모든 샘플을 한 번에 보는 요약 플롯 (채널별로 가장 좋은/나쁜 예시)
    fig, axes = plt.subplots(C_out, 4, figsize=(16, 4 * C_out),
                            constrained_layout=True)
    if C_out == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Best and Worst Prediction Examples by Channel', fontsize=16, fontweight='bold')

    for ch in range(C_out):
        # 채널별 MAE 계산해서 최고/최악 찾기
        channel_maes = []
        for i in range(num_test_samples):
            pred_ch = predictions[i][0][ch]
            target_ch = test_targets_np[i][ch]
            mae = np.mean(np.abs(pred_ch - target_ch))
            channel_maes.append((mae, i))

        channel_maes.sort()
        best_idx = channel_maes[0][1]   # 가장 좋은 예측
        worst_idx = channel_maes[-1][1]  # 가장 나쁜 예측

        # Best case
        best_input = test_inputs_np[best_idx][min(ch, C_in-1)] if C_in > 0 else np.zeros_like(test_targets_np[best_idx][ch])
        best_target = test_targets_np[best_idx][ch]
        best_pred = predictions[best_idx][0][ch]
        best_error = np.abs(best_pred - best_target)
        best_mae = np.mean(best_error)

        # Worst case
        worst_input = test_inputs_np[worst_idx][min(ch, C_in-1)] if C_in > 0 else np.zeros_like(test_targets_np[worst_idx][ch])
        worst_target = test_targets_np[worst_idx][ch]
        worst_pred = predictions[worst_idx][0][ch]
        worst_error = np.abs(worst_pred - worst_target)
        worst_mae = np.mean(worst_error)

        # 컬러스케일을 두 케이스 모두 고려해서 설정
        all_values = np.concatenate([best_target.flatten(), best_pred.flatten(),
                                    worst_target.flatten(), worst_pred.flatten()])
        vmin, vmax = all_values.min(), all_values.max()

        # Best case 시각화 (위쪽 절반)
        input_vmin, input_vmax = best_input.min(), best_input.max()
        axes[ch, 0].imshow(best_input.T, cmap='RdBu_r', origin='lower', vmin=input_vmin, vmax=input_vmax)
        axes[ch, 0].set_title(f'Ch{ch+1} Best Input\n(Sample {best_idx+1})', fontsize=10)

        axes[ch, 1].imshow(best_target.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[ch, 1].set_title(f'Best GT\nMAE: {best_mae:.4f}', fontsize=10)

        axes[ch, 2].imshow(best_pred.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[ch, 2].set_title(f'Best Pred', fontsize=10)

        error_vmax = max(best_error.max(), worst_error.max())
        axes[ch, 3].imshow(best_error.T, cmap='Reds', origin='lower', vmin=0, vmax=error_vmax)
        axes[ch, 3].set_title(f'Best Error', fontsize=10)

        # 축 정리
        for ax in [axes[ch, 0], axes[ch, 1], axes[ch, 2], axes[ch, 3]]:
            ax.set_xticks([])
            ax.set_yticks([])

        axes[ch, 0].set_ylabel(f'Channel {ch+1}\n(Best: #{best_idx+1})', fontsize=11, fontweight='bold')

    # 열 제목
    axes[0, 0].text(0.5, 1.1, 'Input', transform=axes[0, 0].transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[0, 1].text(0.5, 1.1, 'Ground Truth', transform=axes[0, 1].transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[0, 2].text(0.5, 1.1, 'Prediction', transform=axes[0, 2].transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[0, 3].text(0.5, 1.1, 'Error', transform=axes[0, 3].transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.savefig(os.path.join(save_result_path, 's2s_summary_best_worst.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved summary plot")

    # --- Additional Error Analysis Visualization ---
    print("\nGenerating error analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle('Snapshot-to-Snapshot Error Analysis', fontsize=16, fontweight='bold')

    # 채널별 MAE/MSE 계산
    channel_mae = {f'Ch{i+1}': [] for i in range(C_out)}
    channel_mse = {f'Ch{i+1}': [] for i in range(C_out)}

    for i in range(num_test_samples):
        pred = predictions[i][0]
        target = test_targets_np[i]

        for ch in range(C_out):
            error = np.abs(pred[ch] - target[ch])
            channel_mae[f'Ch{ch+1}'].append(np.mean(error))
            channel_mse[f'Ch{ch+1}'].append(np.mean(error**2))

    # 1. 채널별 MAE 박스플롯
    ax = axes[0, 0]
    mae_data = [channel_mae[f'Ch{i+1}'] for i in range(C_out)]
    bp1 = ax.boxplot(mae_data, labels=[f'Ch{i+1}' for i in range(C_out)], patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_title('MAE Distribution by Channel')
    ax.set_ylabel('Mean Absolute Error')
    ax.grid(True, alpha=0.3)

    # 2. 채널별 MSE 박스플롯
    ax = axes[0, 1]
    mse_data = [channel_mse[f'Ch{i+1}'] for i in range(C_out)]
    bp2 = ax.boxplot(mse_data, labels=[f'Ch{i+1}' for i in range(C_out)], patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    ax.set_title('MSE Distribution by Channel')
    ax.set_ylabel('Mean Squared Error')
    ax.grid(True, alpha=0.3)

    # 3. 샘플별 전체 오차 트렌드
    ax = axes[1, 0]
    sample_mae = [np.mean([channel_mae[f'Ch{i+1}'][j] for i in range(C_out)])
                  for j in range(num_test_samples)]
    sample_mse = [np.mean([channel_mse[f'Ch{i+1}'][j] for i in range(C_out)])
                  for j in range(num_test_samples)]

    ax.plot(range(1, num_test_samples+1), sample_mae, 'o-', color='blue', label='MAE')
    ax2 = ax.twinx()
    ax2.plot(range(1, num_test_samples+1), sample_mse, 's-', color='red', label='MSE')

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('MAE', color='blue')
    ax2.set_ylabel('MSE', color='red')
    ax.set_title('Error Trends Across Test Samples')
    ax.grid(True, alpha=0.3)

    # 범례
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 4. 오차 히스토그램
    ax = axes[1, 1]
    all_errors = []
    for i in range(num_test_samples):
        pred = predictions[i][0]
        target = test_targets_np[i]
        errors = np.abs(pred - target).flatten()
        all_errors.extend(errors)

    ax.hist(all_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Error Distribution')
    ax.axvline(np.mean(all_errors), color='red', linestyle='--',
               label=f'Mean: {np.mean(all_errors):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_result_path, 's2s_error_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved error analysis plot")

    # --- Calculate Overall Metrics ---
    print("\nCalculating metrics...")
    all_mse = []
    all_mae = []
    channel_metrics = {f'Ch{i+1}': {'mae': [], 'mse': []} for i in range(C_out)}

    for i in range(num_test_samples):
        pred = predictions[i][0]
        target = test_targets_np[i]

        # 전체 오차
        mse = np.mean((pred - target) ** 2)
        mae = np.mean(np.abs(pred - target))
        all_mse.append(mse)
        all_mae.append(mae)

        # 채널별 오차
        for ch in range(C_out):
            ch_mse = np.mean((pred[ch] - target[ch]) ** 2)
            ch_mae = np.mean(np.abs(pred[ch] - target[ch]))
            channel_metrics[f'Ch{ch+1}']['mse'].append(ch_mse)
            channel_metrics[f'Ch{ch+1}']['mae'].append(ch_mae)

    # --- Save Test Results for Postprocessing ---
    print("\nSaving test results for postprocessing...")

    results_dict = {
        'model_type': 'edm_s2s_single',
        'test_inputs': test_inputs_np,
        'test_targets': test_targets_np,
        'predictions': np.array([pred[0] for pred in predictions]),  # Final predictions
        'intermediate_predictions': None,  # For EDM timesteps if needed
        'metrics': {
            'overall_mse': np.mean(all_mse),
            'overall_mae': np.mean(all_mae),
            'channel_metrics': channel_metrics
        },
        'model_config': {
            'model_name': args.model,
            'num_unets': num_unets,
            'max_cutoff': args.max_cutoff,
            'min_cutoff': args.min_cutoff,
            'C_in': C_in,
            'C_out': C_out
        }
    }

    # Add EDM-specific intermediate predictions if available
    if args.model == 'edm' and pred_timesteps[0] is not None:
        # Convert list of predictions to numpy array
        intermediate_preds = []
        for sample_preds in pred_timesteps:
            sample_intermediate = np.array([step_pred[0] for step_pred in sample_preds])
            intermediate_preds.append(sample_intermediate)
        results_dict['intermediate_predictions'] = np.array(intermediate_preds)

    # Save as numpy file
    np.save(os.path.join(save_result_path, 'test_results.npy'), results_dict)
    print(f"Test results saved to {os.path.join(save_result_path, 'test_results.npy')}")

    print(f"\n{'='*60}")
    print(f"Snapshot-to-Snapshot Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Model: {args.model.upper()} with singleUNet")
    print(f"Evaluated {num_test_samples} test samples")
    print(f"Input channels: {C_in}, Output channels: {C_out}")
    print(f"\nOverall Metrics:")
    print(f"  MSE: {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
    print(f"  MAE: {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")

    print(f"\nChannel-wise Metrics:")
    for ch in range(C_out):
        ch_name = f'Ch{ch+1}'
        ch_mse_mean = np.mean(channel_metrics[ch_name]['mse'])
        ch_mse_std = np.std(channel_metrics[ch_name]['mse'])
        ch_mae_mean = np.mean(channel_metrics[ch_name]['mae'])
        ch_mae_std = np.std(channel_metrics[ch_name]['mae'])

        print(f"  {ch_name}:")
        print(f"    MSE: {ch_mse_mean:.6f} ± {ch_mse_std:.6f}")
        print(f"    MAE: {ch_mae_mean:.6f} ± {ch_mae_std:.6f}")

    print(f"\nResults saved in: {save_result_path}")
    for i in range(num_pages):
        print(f"  - s2s_prediction_results_page{i+1}.png")
    print(f"  - s2s_error_analysis.png")
    print(f"  - test_results.npy (for postprocessing)")

# --- Energy Cascade Analysis for Predictions ---
print(f"\n{'='*60}")
print("Generating energy cascade analysis for predictions (GT vs Model)...")
print(f"{'='*60}")

model_predictions = {
    args.model: predictions_np  # (num_pred_steps, C, H, W)
}

# Call energy cascade analysis
energy_results = plot_energy_cascade_analysis(
    gt_data=gt_sequence,  # Use gt_sequence instead of test_targets_np for video prediction
    model_predictions=model_predictions,
    save_path=save_result_path,
    experiment_name=f'energy_cascade_{args.model}_video'
)

print(f"Energy cascade analysis completed!")
print(f"  - energy_cascade_{args.model}_video_comparison.png")
print(f"  - Analyzed {energy_results['n_samples_analyzed']} timesteps")
print(f"{'='*60}")

