from model.unet import UNet
from model.edm_single import EDM, FrequencyProgressiveUNet
#from model.edm_SR import EDM, FrequencyProgressiveUNet
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
from utils.spatialSR import create_lowres_from_highres, visualize_lowres_samples, save_lowres_data
from utils.plot import plot_q_sample_energy_cascade, plot_energy_cascade_analysis
import argparse


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='default learning rate')
parser.add_argument('--train_ratio', type=float, default=0.6, help='train_ratio')
parser.add_argument('--val_ratio', type=float, default=0.2, help='val_ratio')
parser.add_argument('--data_ratio', type=float, default=1.0, help='Ratio of total dataset to use (0.0-1.0). E.g., 0.6 uses 60%% of all data')
parser.add_argument('--lowres_grids', type=int, nargs='+', default=[8, 16], help='Grid sizes for low-res generation (e.g., 8 16 32)')
parser.add_argument('--u_ref', type=float, default=0.0499, help='Reference velocity for normalization')
parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
args = parser.parse_args()

epochs = args.epochs
dataset = args.dataset
model_name = args.model
num_unets = args.num_unets
batch_size = args.batch_size
learning_rate = args.learning_rate
train_ratio = args.train_ratio
val_ratio = train_ratio/3#args.val_ratio
test_ratio = 1 - train_ratio - val_ratio

save_result_path = f'test_SR/result_{model_name}_{dataset}_grid{"_".join(map(str, args.lowres_grids))}_{train_ratio*5000}'

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

# Dataset load and auto low-res generation
data_dir = f'/home/navier/Dataset/various_CFD/{dataset}/'
#data_dir = f'/mnt/hdd3/home/ddfe/jjw/Dataset/various_CFD/{dataset}/'

print(f"\n{'='*60}")
print(f"Auto Low-Resolution Generation")
print(f"{'='*60}")

# Load high-resolution data
highres_path = os.path.join(data_dir, f'{dataset}.npy')
print(f"Loading high-res data from: {highres_path}")
data_highres = np.load(highres_path).astype(np.float32)

# Generate low-resolution versions
lowres_result = create_lowres_from_highres(
    data_highres,
    target_grid_sizes=args.lowres_grids,
    u_ref=args.u_ref,
    normalize=True,
    remove_nan=True,
    verbose=True
)

# Visualize samples
visualize_lowres_samples(
    lowres_result,
    num_samples=3,
    channel_idx=0,
    save_path=os.path.join(save_result_path, 'lowres_comparison.png')
)

# Save low-res data to dataset directory
lowres_save_dir = os.path.join(data_dir, 'auto_lowres')
save_lowres_data(lowres_result, lowres_save_dir, dataset_name=dataset)

# Use the first low-res as condition, highres as target
first_grid = args.lowres_grids[0]
condition_key = f'lowres_{first_grid}x{first_grid}'
condition_data = lowres_result[condition_key]
target_data = lowres_result['highres']

print(f"\nUsing {condition_key} as condition (low-res)")
print(f"Using highres as target")
print(f"Condition shape: {condition_data.shape}")
print(f"Target shape: {target_data.shape}")

# Save condition and target temporarily
temp_condition_path = os.path.join(data_dir, 'X_temp.npy')
temp_target_path = os.path.join(data_dir, 'Y_temp.npy')
np.save(temp_condition_path, condition_data)
np.save(temp_target_path, target_data)

# Create dataloaders using existing function
train_loader, val_loader, test_loader, dataset = create_snapshot_dataloaders(
        data_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        data_ratio=args.data_ratio,
        condition='X_temp.npy',
        target='Y_temp.npy'
    )


batch_sample= next(iter(train_loader))


inputs_sample= batch_sample['input']
targets_sample = batch_sample['target']

print(targets_sample.shape)
_, C_in, _, _ = inputs_sample.shape
_, C_out, _, _ = targets_sample.shape


if args.model == 'edm':
    in_channels = C_in + C_out # condition frames + predicted state
    out_channels = C_out # condition frames + predicted state
    unet = FrequencyProgressiveUNet(
        in_channels=in_channels,  # 3 (condition) + 3 (state)
        out_channels=out_channels,  # target channels
        base_channels=64,
        time_emb_dim=256
    ).cuda()
    model = EDM(unet=unet,
                num_timesteps=num_timesteps,
                lr=learning_rate,
                max_freq_ratio=args.max_cutoff,
                min_freq_ratio=args.min_cutoff).to(device)
    print(f'Successfully initialized EDM model with singleUNet!\n')
elif args.model == 'ddim':
    in_channels = C_in + C_out # condition frames + predicted state
    out_channels = C_out # target channels
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
    
    print(f'Successfully initialized DDIM model with AdaGN UNet!\n')
elif args.model == 'cnn':
    in_channels = C_in  # only condition frames
    out_channels = C_out  # only condition frames
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
                
                im = ax.imshow(img_data, origin='lower', aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
                
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
        model.train(train_loader, val_loader,  epochs, f'{save_result_path}/best_model.pth', f'{save_result_path}/logs', use_tensorboard=args.tensorboard)
    elif args.model == 'ddim':
        model.train(train_loader, val_loader, epochs, f'{save_result_path}/best_model.pth', f'{save_result_path}/logs')
    else:
        os.makedirs(save_result_path, exist_ok=True)
        log_file = open(f'{save_result_path}/training_log.txt', 'w')

        # History for plotting
        history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': [],
                   'train_ssim': [], 'val_ssim': [], 'learning_rate': []}

        for epoch in range(epochs):
            model.train()
            train_losses = {'total': [], 'mse': [], 'ssim': []}

            for batch in tqdm(train_loader):
                cond = batch['input'].to(device)
                targets = batch['target'].to(device)

                optimizer.zero_grad()
                outputs = model(cond)

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
                for batch in tqdm(val_loader):
                    cond = batch['input'].to(device)
                    targets = batch['target'].to(device)

                    outputs = model(cond)

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
        import matplotlib.pyplot as plt
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

# --- Snapshot-to-Snapshot Evaluation ---
print("\nStarting snapshot-to-snapshot evaluation...")

num_test_samples = 15  # 평가할 샘플 수

print(f"Collecting {num_test_samples} test samples...")

# 테스트 샘플 수집
all_test_inputs = []
all_test_targets = []
samples_collected = 0

for batch in test_loader:
    batch_size = batch['input'].shape[0]
    samples_to_take = min(batch_size, num_test_samples - samples_collected)

    all_test_inputs.append(batch['input'][:samples_to_take].to(device))
    all_test_targets.append(batch['target'][:samples_to_take].to(device))

    samples_collected += samples_to_take
    if samples_collected >= num_test_samples:
        break

test_inputs = torch.cat(all_test_inputs, dim=0)[:num_test_samples]
test_targets = torch.cat(all_test_targets, dim=0)[:num_test_samples]

print(f"Test input shape: {test_inputs.shape}")   # (15, C_in, H, W)
print(f"Test target shape: {test_targets.shape}") # (15, C_out, H, W)

# 예측 수행 및 각 timestep별 비교 데이터 수집
predictions = []
gt_q_samples = []  # GT의 각 timestep별 q_sample
pred_timesteps = []  # 각 timestep별 예측 결과

with torch.no_grad():
    for i in range(num_test_samples):
        single_input = test_inputs[i:i+1]
        single_target = test_targets[i:i+1]

        if args.model == 'edm':
            # 전체 sampling 과정의 intermediate results 수집
            predicted_frames = model.sampling(1, device,
                                            (test_targets.shape[-2], test_targets.shape[-1]),
                                            cond=single_input)
            final_prediction = predicted_frames[0]
            
            # 각 timestep별 GT q_sample 생성 (t=0 to t=num_timesteps)
            gt_timestep_samples = []
            for t in range(model.num_timesteps + 1):
                gt_t = model.q_sample(single_target, t)  # scalar로 전달
                gt_timestep_samples.append(gt_t.cpu().numpy())
            
            gt_q_samples.append(gt_timestep_samples)
            pred_timesteps.append([p.cpu().numpy() for p in predicted_frames])
        elif args.model == 'ddim':
            # DDIM sampling
            final_prediction = model.sampling(single_input)
            gt_q_samples.append(None)
            pred_timesteps.append(None)
        else:
            final_prediction = model(single_input)
            gt_q_samples.append(None)
            pred_timesteps.append(None)

        predictions.append(final_prediction.cpu().numpy())

# numpy 변환
test_inputs_np = test_inputs.cpu().numpy()
test_targets_np = test_targets.cpu().numpy()

# --- EDM Timestep Comparison Visualization (Residual Prediction) ---
if args.model == 'edm':
    print("\nGenerating EDM residual timestep comparison plots...")

    # 3개 샘플에 대해 timestep별 비교 visualization
    num_comparison_samples = min(3, num_test_samples)

    for sample_idx in range(num_comparison_samples):
        # 5행: Input, Predicted Residual, Target Residual, Residual Error, Reconstructed vs GT
        fig, axes = plt.subplots(5, model.num_timesteps,
                                figsize=(3 * model.num_timesteps, 15),
                                squeeze=False, constrained_layout=True)

        fig.suptitle(f'EDM Residual Prediction - Sample {sample_idx + 1}', fontsize=16, fontweight='bold')

        # 첫 번째 채널만 시각화 (다채널인 경우)
        channel_idx = 0

        # t=0부터 t=num_timesteps-1까지 (training range와 일치)
        for col_idx, t_idx in enumerate(range(model.num_timesteps)):
            # Input: x_{t+1} (more filtered state)
            input_t_plus_1 = gt_q_samples[sample_idx][t_idx + 1][0, channel_idx]

            # Target: x_t (less filtered state)
            gt_t = gt_q_samples[sample_idx][t_idx][0, channel_idx]

            # Predicted output from sampling
            pred_t = pred_timesteps[sample_idx][t_idx][0, channel_idx]

            # Compute residuals
            target_residual = gt_t - input_t_plus_1
            # For predicted residual, we need to compute from consecutive predictions
            pred_t_plus_1 = pred_timesteps[sample_idx][t_idx + 1][0, channel_idx]
            pred_residual = pred_t - pred_t_plus_1

            # Residual error
            residual_error = np.abs(pred_residual - target_residual)

            # 컬러맵 범위 설정
            vmin_state = min(input_t_plus_1.min(), gt_t.min(), pred_t.min())
            vmax_state = max(input_t_plus_1.max(), gt_t.max(), pred_t.max())
            vmin_res = min(target_residual.min(), pred_residual.min())
            vmax_res = max(target_residual.max(), pred_residual.max())

            # 1행: Input x_{t+1}
            ax1 = axes[0, col_idx]
            im1 = ax1.imshow(input_t_plus_1, cmap='RdBu_r', origin='lower', aspect='auto',
                            vmin=vmin_state, vmax=vmax_state)
            ax1.set_title(f't={t_idx}\nInput x_{{{t_idx+1}}}', fontsize=9)
            ax1.set_xticks([])
            ax1.set_yticks([])
            if col_idx == 0:
                ax1.set_ylabel('Input\n(Filtered)', fontsize=11, fontweight='bold')

            # 2행: Target residual (x_t - x_{t+1})
            ax2 = axes[1, col_idx]
            im2 = ax2.imshow(target_residual, cmap='seismic', origin='lower', aspect='auto',
                            vmin=vmin_res, vmax=vmax_res)
            ax2.set_title(f'Target Δ', fontsize=9)
            ax2.set_xticks([])
            ax2.set_yticks([])
            if col_idx == 0:
                ax2.set_ylabel('Target\nResidual', fontsize=11, fontweight='bold')

            # 3행: Predicted residual
            ax3 = axes[2, col_idx]
            im3 = ax3.imshow(pred_residual, cmap='seismic', origin='lower', aspect='auto',
                            vmin=vmin_res, vmax=vmax_res)
            ax3.set_title(f'Pred Δ', fontsize=9)
            ax3.set_xticks([])
            ax3.set_yticks([])
            if col_idx == 0:
                ax3.set_ylabel('Predicted\nResidual', fontsize=11, fontweight='bold')

            # 4행: Residual error
            ax4 = axes[3, col_idx]
            im4 = ax4.imshow(residual_error, cmap='Reds', origin='lower', aspect='auto',
                            vmin=0, vmax=(vmax_res-vmin_res)*0.5)
            mae_res = np.mean(residual_error)
            ax4.set_title(f'MAE={mae_res:.4f}', fontsize=9)
            ax4.set_xticks([])
            ax4.set_yticks([])
            if col_idx == 0:
                ax4.set_ylabel('Residual\nError', fontsize=11, fontweight='bold')

            # 5행: Reconstructed vs GT
            ax5 = axes[4, col_idx]
            im5 = ax5.imshow(pred_t, cmap='RdBu_r', origin='lower', aspect='auto',
                            vmin=vmin_state, vmax=vmax_state)
            state_error = np.mean(np.abs(pred_t - gt_t))
            ax5.set_title(f'Recon MAE={state_error:.4f}', fontsize=9)
            ax5.set_xticks([])
            ax5.set_yticks([])
            if col_idx == 0:
                ax5.set_ylabel('Reconstructed\nvs GT', fontsize=11, fontweight='bold')

        # 컬러바 추가
        cbar_ax1 = fig.add_axes([0.92, 0.65, 0.01, 0.25])
        sm1 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin_state, vmax=vmax_state))
        sm1.set_array([])
        fig.colorbar(sm1, cax=cbar_ax1, label='State Value')

        cbar_ax2 = fig.add_axes([0.92, 0.35, 0.01, 0.25])
        sm2 = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=vmin_res, vmax=vmax_res))
        sm2.set_array([])
        fig.colorbar(sm2, cax=cbar_ax2, label='Residual')

        cbar_ax3 = fig.add_axes([0.92, 0.05, 0.01, 0.25])
        sm3 = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=(vmax_res-vmin_res)*0.5))
        sm3.set_array([])
        fig.colorbar(sm3, cax=cbar_ax3, label='Error')

        # 저장
        plt.savefig(os.path.join(save_result_path, f'edm_residual_timestep_comparison_sample{sample_idx+1}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved EDM residual timestep comparison for sample {sample_idx + 1}")

# --- Visualization: Snapshot-to-Snapshot Results ---
print("\nGenerating snapshot-to-snapshot visualization plots...")

cases_per_page = 5  # 페이지당 케이스 수
num_pages = 3       # 총 페이지 수

for page_idx in range(num_pages):
    # 각 케이스는 C_out개 행을 가짐 (채널별로), 4개 열 (Input, GT, Pred, Error)
    # 각 subplot 1:1 비율
    subplot_size = 4  # 각 subplot의 크기
    fig, axes = plt.subplots(cases_per_page * C_out, 4,
                            figsize=(subplot_size * 4, subplot_size * cases_per_page * C_out),
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
            im1 = ax_input.imshow(input_channel, cmap='RdBu_r', origin='lower', aspect='equal',
                                 vmin=input_vmin, vmax=input_vmax)
            ax_input.set_title(f'Input Ch{min(ch+1, C_in)}', fontsize=11)
            ax_input.set_xticks([])
            ax_input.set_yticks([])

            if ch == 0:  # 첫 번째 채널에만 케이스 번호 표시
                ax_input.set_ylabel(f'Case {sample_idx + 1}', fontsize=12, fontweight='bold')

            # 2. Ground Truth
            ax_gt = axes[row_idx, 1]
            im2 = ax_gt.imshow(gt_channel, cmap='RdBu_r', origin='lower', aspect='equal',
                              vmin=vmin, vmax=vmax)
            ax_gt.set_title(f'GT Ch{ch+1}', fontsize=11)
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])

            # 3. Prediction (GT와 동일한 컬러스케일)
            ax_pred = axes[row_idx, 2]
            pred_channel = pred_data[ch]
            im3 = ax_pred.imshow(pred_channel, cmap='RdBu_r', origin='lower', aspect='equal',
                                vmin=vmin, vmax=vmax)
            ax_pred.set_title(f'Pred Ch{ch+1}', fontsize=11)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            # 4. Error
            ax_error = axes[row_idx, 3]
            error = np.abs(pred_channel - gt_channel)
            mae_ch = np.mean(error)
            mse_ch = np.mean(error**2)
            
            # Calculate SSIM
            try:
                from skimage.metrics import structural_similarity as ssim
                ssim_ch = ssim(gt_channel, pred_channel, data_range=gt_channel.max() - gt_channel.min())
            except:
                ssim_ch = 0.0

            # 오차는 0부터 최대 오차까지
            error_vmax = max(error.max(), (vmax - vmin) * 0.1)  # 최소한 GT 범위의 10%
            im4 = ax_error.imshow(error, cmap='Reds', origin='lower', aspect='equal',
                                 vmin=0, vmax=error_vmax)
            ax_error.set_title(f'Error Ch{ch+1}\nMAE: {mae_ch:.4f}\nMSE: {mse_ch:.4f}\nSSIM: {ssim_ch:.4f}', fontsize=9)
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

# --- Contourf Version ---
print("\nGenerating contourf version of snapshot-to-snapshot plots...")

for page_idx in range(num_pages):
    # 각 subplot 1:1 비율
    subplot_size = 4  # 각 subplot의 크기
    fig, axes = plt.subplots(cases_per_page * C_out, 4,
                            figsize=(subplot_size * 4, subplot_size * cases_per_page * C_out),
                            squeeze=False, constrained_layout=True)

    fig.suptitle(f'Snapshot-to-Snapshot Results (Contourf) - Page {page_idx + 1}/{num_pages}',
                 fontsize=18, fontweight='bold')

    for case_idx in range(cases_per_page):
        sample_idx = page_idx * cases_per_page + case_idx
        if sample_idx >= num_test_samples:
            for ch in range(C_out):
                for col in range(4):
                    axes[case_idx * C_out + ch, col].axis('off')
            continue

        input_data = test_inputs_np[sample_idx]
        target_data = test_targets_np[sample_idx]
        pred_data = predictions[sample_idx][0]

        for ch in range(C_out):
            row_idx = case_idx * C_out + ch
            gt_channel = target_data[ch]
            vmin, vmax = gt_channel.min(), gt_channel.max()

            # 1. Input (imshow)
            ax_input = axes[row_idx, 0]
            if ch < C_in:
                input_channel = input_data[ch]
            else:
                input_channel = input_data[0] if C_in > 0 else np.zeros_like(gt_channel)

            input_vmin, input_vmax = input_channel.min(), input_channel.max()
            im1 = ax_input.imshow(input_channel, cmap='RdBu_r', origin='lower', aspect='equal',
                                 vmin=input_vmin, vmax=input_vmax)
            ax_input.set_title(f'Input Ch{min(ch+1, C_in)}', fontsize=11)
            ax_input.set_xticks([])
            ax_input.set_yticks([])
            ax_input.set_aspect('equal')

            if ch == 0:
                ax_input.set_ylabel(f'Case {sample_idx + 1}', fontsize=12, fontweight='bold')

            # 2. Ground Truth (contourf)
            ax_gt = axes[row_idx, 1]
            cf2 = ax_gt.contourf(gt_channel, levels=20, cmap='RdBu_r',
                                vmin=vmin, vmax=vmax)
            ax_gt.set_title(f'GT Ch{ch+1}', fontsize=11)
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            ax_gt.set_aspect('equal')

            # 3. Prediction (contourf)
            ax_pred = axes[row_idx, 2]
            pred_channel = pred_data[ch]
            cf3 = ax_pred.contourf(pred_channel, levels=20, cmap='RdBu_r',
                                  vmin=vmin, vmax=vmax)
            ax_pred.set_title(f'Pred Ch{ch+1}', fontsize=11)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            ax_pred.set_aspect('equal')

            # 4. Error (imshow)
            ax_error = axes[row_idx, 3]
            error = np.abs(pred_channel - gt_channel)
            mae_ch = np.mean(error)
            mse_ch = np.mean(error**2)
            
            # Calculate SSIM
            try:
                from skimage.metrics import structural_similarity as ssim
                ssim_ch = ssim(gt_channel, pred_channel, data_range=gt_channel.max() - gt_channel.min())
            except:
                ssim_ch = 0.0

            error_vmax = max(error.max(), (vmax - vmin) * 0.1)
            im4 = ax_error.imshow(error, cmap='Reds', origin='lower', aspect='equal',
                                 vmin=0, vmax=error_vmax)
            ax_error.set_title(f'Error Ch{ch+1}\nMAE: {mae_ch:.4f}\nMSE: {mse_ch:.4f}\nSSIM: {ssim_ch:.4f}', fontsize=9)
            ax_error.set_xticks([])
            ax_error.set_yticks([])

            if ch == C_out - 1:
                cbar1 = plt.colorbar(cf2, ax=[ax_gt, ax_pred], shrink=0.6, aspect=10)
                cbar1.set_label(f'Case {sample_idx + 1} Value', fontsize=10)

                cbar2 = plt.colorbar(im4, ax=ax_error, shrink=0.6, aspect=10)
                cbar2.set_label('Error', fontsize=10)

        if case_idx < cases_per_page - 1 and sample_idx + 1 < num_test_samples:
            separator_row = (case_idx + 1) * C_out - 1
            for col in range(4):
                axes[separator_row, col].axhline(y=-0.5, color='black',
                                               linewidth=2, clip_on=False)

    axes[0, 0].text(0.5, 1.1, 'Input', transform=axes[0, 0].transAxes,
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[0, 1].text(0.5, 1.1, 'Ground Truth', transform=axes[0, 1].transAxes,
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[0, 2].text(0.5, 1.1, 'Prediction', transform=axes[0, 2].transAxes,
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[0, 3].text(0.5, 1.1, 'Absolute Error', transform=axes[0, 3].transAxes,
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(save_result_path, f's2s_prediction_contourf_page{page_idx+1}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved contourf page {page_idx + 1}")

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
    axes[ch, 0].imshow(best_input, cmap='RdBu_r', origin='lower', aspect='equal', vmin=input_vmin, vmax=input_vmax)
    axes[ch, 0].set_title(f'Ch{ch+1} Best Input\n(Sample {best_idx+1})', fontsize=10)

    axes[ch, 1].imshow(best_target, cmap='RdBu_r', origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
    axes[ch, 1].set_title(f'Best GT\nMAE: {best_mae:.4f}', fontsize=10)

    axes[ch, 2].imshow(best_pred, cmap='RdBu_r', origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
    axes[ch, 2].set_title(f'Best Pred', fontsize=10)

    error_vmax = max(best_error.max(), worst_error.max())
    axes[ch, 3].imshow(best_error, cmap='Reds', origin='lower', aspect='equal', vmin=0, vmax=error_vmax)
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

# 채널별 MAE/MSE/SSIM 계산
channel_mae = {f'Ch{i+1}': [] for i in range(C_out)}
channel_mse = {f'Ch{i+1}': [] for i in range(C_out)}
channel_ssim = {f'Ch{i+1}': [] for i in range(C_out)}

for i in range(num_test_samples):
    pred = predictions[i][0]
    target = test_targets_np[i]

    for ch in range(C_out):
        error = np.abs(pred[ch] - target[ch])
        channel_mae[f'Ch{ch+1}'].append(np.mean(error))
        channel_mse[f'Ch{ch+1}'].append(np.mean(error**2))
        
        # Calculate SSIM
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_val = ssim(target[ch], pred[ch], data_range=target[ch].max() - target[ch].min())
        except:
            ssim_val = 0.0
        channel_ssim[f'Ch{ch+1}'].append(ssim_val)

# Calculate average values for each metric
mae_averages = [np.mean(channel_mae[f'Ch{i+1}']) for i in range(C_out)]
mse_averages = [np.mean(channel_mse[f'Ch{i+1}']) for i in range(C_out)]
ssim_averages = [np.mean(channel_ssim[f'Ch{i+1}']) for i in range(C_out)]

overall_mae_avg = np.mean(mae_averages)
overall_mse_avg = np.mean(mse_averages)
overall_ssim_avg = np.mean(ssim_averages)

# 1. 채널별 MAE 박스플롯
ax = axes[0, 0]
mae_data = [channel_mae[f'Ch{i+1}'] for i in range(C_out)]
bp1 = ax.boxplot(mae_data, labels=[f'Ch{i+1}' for i in range(C_out)], patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('lightblue')
ax.set_title(f'MAE Distribution by Channel\nOverall Avg: {overall_mae_avg:.6f}')
ax.set_ylabel('Mean Absolute Error')
ax.grid(True, alpha=0.3)
# Add average line
ax.axhline(y=overall_mae_avg, color='red', linestyle='--', linewidth=2, label=f'Avg: {overall_mae_avg:.6f}')
ax.legend(loc='upper right')

# 2. 채널별 MSE 박스플롯
ax = axes[0, 1]
mse_data = [channel_mse[f'Ch{i+1}'] for i in range(C_out)]
bp2 = ax.boxplot(mse_data, labels=[f'Ch{i+1}' for i in range(C_out)], patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('lightcoral')
ax.set_title(f'MSE Distribution by Channel\nOverall Avg: {overall_mse_avg:.6f}')
ax.set_ylabel('Mean Squared Error')
ax.grid(True, alpha=0.3)
# Add average line
ax.axhline(y=overall_mse_avg, color='red', linestyle='--', linewidth=2, label=f'Avg: {overall_mse_avg:.6f}')
ax.legend(loc='upper right')

# 3. 채널별 SSIM 박스플롯
ax = axes[1, 0]
ssim_data = [channel_ssim[f'Ch{i+1}'] for i in range(C_out)]
bp3 = ax.boxplot(ssim_data, labels=[f'Ch{i+1}' for i in range(C_out)], patch_artist=True)
for patch in bp3['boxes']:
    patch.set_facecolor('lightgreen')
ax.set_title(f'SSIM Distribution by Channel\nOverall Avg: {overall_ssim_avg:.6f}')
ax.set_ylabel('Structural Similarity Index')
ax.grid(True, alpha=0.3)
# Add average line
ax.axhline(y=overall_ssim_avg, color='red', linestyle='--', linewidth=2, label=f'Avg: {overall_ssim_avg:.6f}')
ax.legend(loc='upper right')

# 4. 오차 히스토그램 (MAE, MSE, SSIM combined)
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
all_ssim = []
channel_metrics = {f'Ch{i+1}': {'mae': [], 'mse': [], 'ssim': []} for i in range(C_out)}

for i in range(num_test_samples):
    pred = predictions[i][0]
    target = test_targets_np[i]

    # 전체 오차
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    all_mse.append(mse)
    all_mae.append(mae)
    
    # Overall SSIM (average over channels)
    sample_ssims = []
    for ch in range(C_out):
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_val = ssim(target[ch], pred[ch], data_range=target[ch].max() - target[ch].min())
        except:
            ssim_val = 0.0
        sample_ssims.append(ssim_val)
    all_ssim.append(np.mean(sample_ssims))

    # 채널별 오차
    for ch in range(C_out):
        ch_mse = np.mean((pred[ch] - target[ch]) ** 2)
        ch_mae = np.mean(np.abs(pred[ch] - target[ch]))
        channel_metrics[f'Ch{ch+1}']['mse'].append(ch_mse)
        channel_metrics[f'Ch{ch+1}']['mae'].append(ch_mae)
        channel_metrics[f'Ch{ch+1}']['ssim'].append(sample_ssims[ch])

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
        'overall_ssim': np.mean(all_ssim),
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
print(f"  SSIM: {np.mean(all_ssim):.6f} ± {np.std(all_ssim):.6f}")

print(f"\nChannel-wise Metrics:")
for ch in range(C_out):
    ch_name = f'Ch{ch+1}'
    ch_mse_mean = np.mean(channel_metrics[ch_name]['mse'])
    ch_mse_std = np.std(channel_metrics[ch_name]['mse'])
    ch_mae_mean = np.mean(channel_metrics[ch_name]['mae'])
    ch_mae_std = np.std(channel_metrics[ch_name]['mae'])
    ch_ssim_mean = np.mean(channel_metrics[ch_name]['ssim'])
    ch_ssim_std = np.std(channel_metrics[ch_name]['ssim'])

    print(f"  {ch_name}:")
    print(f"    MSE: {ch_mse_mean:.6f} ± {ch_mse_std:.6f}")
    print(f"    MAE: {ch_mae_mean:.6f} ± {ch_mae_std:.6f}")
    print(f"    SSIM: {ch_ssim_mean:.6f} ± {ch_ssim_std:.6f}")

print(f"\nResults saved in: {save_result_path}")
for i in range(num_pages):
    print(f"  - s2s_prediction_results_page{i+1}.png")
print(f"  - s2s_error_analysis.png")
print(f"  - test_results.npy (for postprocessing)")

# --- Energy Cascade Analysis for Predictions ---
print(f"\n{'='*60}")
print("Generating energy cascade analysis for predictions (GT vs Model)...")
print(f"{'='*60}")

# Prepare predictions dictionary (only include the models being compared)
model_predictions = {
    args.model: np.array([pred[0] for pred in predictions])  # (N_samples, C, H, W)
}

# Call energy cascade analysis
energy_results = plot_energy_cascade_analysis(
    gt_data=test_targets_np,
    model_predictions=model_predictions,
    save_path=save_result_path,
    experiment_name=f'energy_cascade_{args.model}'
)

print(f"Energy cascade analysis completed!")
print(f"  - energy_cascade_{args.model}_comparison.png")
print(f"  - Analyzed {energy_results['n_samples_analyzed']} samples")
print(f"{'='*60}")
