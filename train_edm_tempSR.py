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

from utils.tempSR import (
    create_temporal_dataset,
    TemporalDataset,
    quick_visualize,
    print_usage,
    get_version
)
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
parser.add_argument('--train_size', type=int, default=600, help='The number of training timesteps.')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'load'], help='Set mode to train or load.')
parser.add_argument('--dataset', type=str, default='JHS_tempSR', help='Select the datasets')
parser.add_argument('--num_unets', type=int, default=3, help='The number of U-Net models in the refinement chain.')
parser.add_argument('--max_cutoff', type=float, default=0.8, help='The maximum cutoff frequency')
parser.add_argument('--min_cutoff', type=float, default=0.02, help='The minimum cutoff frequency')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='default learning rate')
parser.add_argument('--temporal_stride', type=int, default=1, help='temporal_stride for SR, defined timesteps coarse')
parser.add_argument('--train_ratio', type=float, default=0.6, help='train_ratio')
parser.add_argument('--val_ratio', type=float, default=0.2, help='val_ratio')
parser.add_argument('--sequence_length', type=int, default=10, help='Summation total timesteps each sequenct, 2 Input + N betweent, default=10')
#parser.add_argument('--cond_frames', type=int, default=1, help='The number of condition frames (1, 2, 3, etc.)')
args = parser.parse_args()

epochs = args.epochs
train_size = args.train_size
dataset = args.dataset
model_name = args.model
num_unets = args.num_unets
temporal_stride=args.temporal_stride
batch_size = args.batch_size
learning_rate = args.learning_rate
train_ratio = args.train_ratio
val_ratio = args.val_ratio
test_ratio = 1 - train_ratio - val_ratio
sequence_length = args.sequence_length

save_result_path = f'test_tempSR/result_{model_name}_{dataset}'

os.makedirs(save_result_path, exist_ok=True)
with open(os.path.join(save_result_path, 'parameters.txt'), 'w') as f:
    for arg, value in vars(args).items():
        f.write(f'{arg}: {value}\n')


batch_size = 32
num_timesteps = num_unets + 1
###########################################################################


# initializing model
print(f'Initializing {args.model.upper()} model....')
torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)


## load dataset
full_dataset_np = np.load(f'/home/navier/Dataset/various_CFD/{dataset}/{dataset}.npy').astype(np.float32)
# full_dataset_np = full_dataset_np[:, 0, :, :].reshape(-1, 1, 192, 64)  # Add channel dimension
# Normalize the dataset per channel
# Assuming full_dataset_np is (N, C, H, W)
mean = np.mean(full_dataset_np, axis=(0, 2, 3), keepdims=True)
std = np.std(full_dataset_np, axis=(0, 2, 3), keepdims=True)
normalized_dataset_np = (full_dataset_np - mean) / (std + 1e-8)
normalized_dataset_np = (full_dataset_np)


train_loader, val_loader, test_loader = create_temporal_dataset(
        full_dataset_np,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        temporal_stride=temporal_stride,
        sequence_length=sequence_length
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

    print(f'Successfully initialized EDM model with {num_unets} U-Net!\n')
elif args.model == 'ddim':
    in_channels = C_in + C_out # condition frames + predicted state
    out_channels = C_out # target channels
    unet = FrequencyProgressiveUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=64,
        time_emb_dim=256
    ).cuda()
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
    print('Successfully initialized CNN model!\n')
else:
    print('You didnt select right model name..')



if args.model == 'edm':
# --- Monitoring q_sample ---
    print(f"\nMonitoring EDM q_sample for {model.num_timesteps} DDIM steps...")
    NUM_SAMPLES_TO_MONITOR = 5
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

    # Only visualize first channel
    ch_idx = 0

    fig, axes = plt.subplots(NUM_SAMPLES_TO_MONITOR, NUM_T_STEPS_TO_PLOT,
                             figsize=(3 * NUM_T_STEPS_TO_PLOT, 3.2 * NUM_SAMPLES_TO_MONITOR),
                             squeeze=False, constrained_layout=True)
    fig.suptitle('EDM Forward Process (q_sample) Visualization - Channel 1', fontsize=20)

    for i in range(NUM_SAMPLES_TO_MONITOR):
        for j, t_idx in enumerate(plot_t_indices):
            ax = axes[i, j]

            # Extract the first channel only
            img_data = all_snapshots_by_sample[i][j][0, ch_idx, :, :]

            # Determine color limits for first channel from the original targets
            vmin = train_targets[:, ch_idx, :, :].min()
            vmax = train_targets[:, ch_idx, :, :].max()

            im = ax.imshow(img_data.T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)

            # Add a colorbar to each subplot
            fig.colorbar(im, ax=ax, shrink=0.8)

            ax.set_title(f't = {t_idx}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(f'Sample {i+1}', rotation=0, size='large', labelpad=40)

    plot_filename = os.path.join(save_result_path, "q_sample_monitoring_result.pdf")
    #plot_filename = os.path.join(save_result_path, "q_sample_monitoring_result.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Monitoring plot saved to '{plot_filename}'")

    # --- GIF Animation for q_sample evolution ---
    print("\nGenerating GIF animation for q_sample evolution...")
    from PIL import Image

    # Select first sample for GIF
    sample_idx_for_gif = 0
    ch_idx_for_gif = 0

    # GT (t=0)
    gt_img = train_targets[sample_idx_for_gif, ch_idx_for_gif, :, :].cpu().numpy()

    # Color limits
    vmin = train_targets[:, ch_idx_for_gif, :, :].min().cpu().numpy()
    vmax = train_targets[:, ch_idx_for_gif, :, :].max().cpu().numpy()

    gif_frames = []
    for t_idx in tqdm(plot_t_indices, desc="Creating GIF frames"):
        # Get q_sample at this timestep
        q_sample_img = all_snapshots_by_sample[sample_idx_for_gif][t_idx][0, ch_idx_for_gif, :, :]

        # Create comparison figure
        fig_gif, axes_gif = plt.subplots(1, 2, figsize=(10, 4))

        # GT
        im1 = axes_gif[0].imshow(gt_img.T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes_gif[0].set_title('Ground Truth (t=0)', fontsize=14, fontweight='bold')
        axes_gif[0].set_xticks([])
        axes_gif[0].set_yticks([])
        fig_gif.colorbar(im1, ax=axes_gif[0], shrink=0.8)

        # q_sample at t
        im2 = axes_gif[1].imshow(q_sample_img.T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes_gif[1].set_title(f'q_sample at t={t_idx}', fontsize=14, fontweight='bold')
        axes_gif[1].set_xticks([])
        axes_gif[1].set_yticks([])
        fig_gif.colorbar(im2, ax=axes_gif[1], shrink=0.8)

        fig_gif.suptitle(f'EDM Forward Process Evolution - Sample {sample_idx_for_gif+1}, Channel 1',
                        fontsize=16, fontweight='bold')
        fig_gif.tight_layout()

        # Save to buffer and convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        pil_img = Image.open(buf)
        gif_frames.append(pil_img.copy())
        buf.close()
        plt.close(fig_gif)

    # Save GIF
    gif_filename = os.path.join(save_result_path, "q_sample_evolution.gif")
    gif_frames[0].save(
        gif_filename,
        save_all=True,
        append_images=gif_frames[1:],
        duration=300,  # milliseconds per frame
        loop=0
    )
    print(f"GIF animation saved to '{gif_filename}'")

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
                cond = batch['input'].to("cuda")
                targets = batch['target'].to("cuda")

                optimizer.zero_grad()
                #dummy_time = torch.zeros(inputs.size(0), device=device)
                outputs = model(cond)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                
            model.eval()
            for batch in tqdm(val_loader):
                cond = batch['input'].to("cuda")
                targets = batch['target'].to("cuda")

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




# --- Simplified Temporal Super-Resolution Evaluation ---
print("\nStarting temporal super-resolution evaluation...")

num_inter_frames = C_out
num_test_samples = 15  # 5장 x 3케이스 = 15개 샘플

print(f"Collecting {num_test_samples} test samples...")

# 테스트 샘플 수집 (전체 먼저 모은 후 random sampling)
all_test_inputs = []
all_test_targets = []
all_seq_indices = []

for batch in test_loader:
    all_test_inputs.append(batch['input'])
    all_test_targets.append(batch['target'])
    all_seq_indices.extend(batch['sequence_idx'].tolist())

# 전체 합치기
all_test_inputs = torch.cat(all_test_inputs, dim=0)
all_test_targets = torch.cat(all_test_targets, dim=0)

# Random sampling
total_available = len(all_test_inputs)
random_indices = torch.randperm(total_available)[:num_test_samples]

test_inputs = all_test_inputs[random_indices].to(device)
test_targets = all_test_targets[random_indices].to(device)

print(f"Test input shape: {test_inputs.shape}")
print(f"Test target shape: {test_targets.shape}")

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

# --- EDM Timestep Comparison Visualization ---
if args.model == 'edm':
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
        plt.savefig(os.path.join(save_result_path, f'edm_timestep_comparison_sample{sample_idx+1}.pdf'),
        #plt.savefig(os.path.join(save_result_path, f'edm_timestep_comparison_sample{sample_idx+1}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved EDM timestep comparison for sample {sample_idx + 1}")

elif args.model == 'ddim':
    print("\nSkipping timestep comparison for DDIM (only final results needed)")

# --- Visualization: Dynamic sequence visualization ---
print("\nGenerating temporal super-resolution visualization plots...")

# 동적으로 sequence_length에 따른 설정
num_target_frames = sequence_length - 2  # 중간 예측 프레임 개수
total_frames = sequence_length  # 전체 프레임 개수

cases_per_page = 3
num_pages = 5

for page_idx in range(num_pages):
    fig, axes = plt.subplots(9, total_frames, figsize=(4 * total_frames, 27), squeeze=False, constrained_layout=True)

    fig.suptitle(f'Temporal Super-Resolution Results - Page {page_idx + 1}/{num_pages}',
                 fontsize=18, fontweight='bold')

    for case_idx in range(cases_per_page):
        sample_idx = page_idx * cases_per_page + case_idx
        if sample_idx >= num_test_samples:
            break

        # 각 케이스는 3행씩 차지 (case_idx * 3)
        row_start = case_idx * 3

        # t1, t_last 입력 프레임 분리
        channels_per_frame = C_in // 2
        t1_frame = test_inputs_np[sample_idx, :channels_per_frame]
        tlast_frame = test_inputs_np[sample_idx, channels_per_frame:]

        # 채널 평균 (시각화용)
        if channels_per_frame > 1:
            t1_vis = np.mean(t1_frame, axis=0)
            tlast_vis = np.mean(tlast_frame, axis=0)
        else:
            t1_vis = t1_frame[0]
            tlast_vis = tlast_frame[0]

        # GT: 중간 프레임들
        gt_frames = test_targets_np[sample_idx]  # (num_target_frames, H, W)

        # Pred: 중간 프레임들
        pred_frames = predictions[sample_idx][0]  # (num_target_frames, H, W)

        # 전체 시퀀스 구성 (동적)
        full_gt_sequence = [t1_vis] + [gt_frames[i] for i in range(num_target_frames)] + [tlast_vis]
        full_pred_sequence = [None] + [pred_frames[i] for i in range(num_target_frames)] + [None]

        # 컬러맵 범위 (GT 기준으로만 설정)
        vmin = min([f.min() for f in full_gt_sequence if f is not None])
        vmax = max([f.max() for f in full_gt_sequence if f is not None])

        # 각 프레임 그리기
        for t_idx in range(total_frames):
            # Ground Truth (1행)
            ax_gt = axes[row_start, t_idx]
            im = ax_gt.imshow(full_gt_sequence[t_idx].T, cmap='RdBu_r',
                             origin='lower', vmin=vmin, vmax=vmax)

            if t_idx == 0:
                ax_gt.set_ylabel(f'Case {sample_idx + 1}\nGT', fontsize=10, fontweight='bold')
            ax_gt.set_title(f't{t_idx + 1}', fontsize=9)
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])

            # Prediction (2행)
            ax_pred = axes[row_start + 1, t_idx]
            if full_pred_sequence[t_idx] is not None:
                im = ax_pred.imshow(full_pred_sequence[t_idx].T, cmap='RdBu_r',
                                   origin='lower', vmin=vmin, vmax=vmax)
            else:
                # 입력 프레임은 빈칸 (회색)
                ax_pred.imshow(np.ones_like(full_gt_sequence[t_idx]).T * 0.5,
                              cmap='gray', vmin=0, vmax=1, alpha=0.3)
                ax_pred.text(0.5, 0.5, 'Input', transform=ax_pred.transAxes,
                           ha='center', va='center', fontsize=12, color='black')

            if t_idx == 0:
                ax_pred.set_ylabel('Pred', fontsize=10, fontweight='bold')
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            # Error (3행)
            ax_err = axes[row_start + 2, t_idx]
            if full_pred_sequence[t_idx] is not None:
                error = np.abs(full_pred_sequence[t_idx] - full_gt_sequence[t_idx])
                im = ax_err.imshow(error.T, cmap='Reds', origin='lower',
                                  vmin=0, vmax=(vmax-vmin)*0.3)
                mae = np.mean(error)
                ax_err.set_title(f'MAE={mae:.4f}', fontsize=8)
            else:
                # 입력 프레임은 오차 없음 (흰색)
                ax_err.imshow(np.zeros_like(full_gt_sequence[t_idx]).T,
                             cmap='Reds', vmin=0, vmax=1)
                ax_err.set_title('N/A', fontsize=8)

            if t_idx == 0:
                ax_err.set_ylabel('Error', fontsize=10, fontweight='bold')
            ax_err.set_xticks([])
            ax_err.set_yticks([])

        # 케이스 구분선
        if case_idx < cases_per_page - 1:
            for col in range(total_frames):
                axes[row_start + 2, col].axhline(y=axes[row_start + 2, col].get_ylim()[0] - 5,
                                                 color='black', linewidth=2, clip_on=False)

    # 빈 subplot 제거 (마지막 페이지에서 케이스가 3개 미만일 경우)
    for row in range(len(axes)):
        for col in range(len(axes[0])):
            if row >= cases_per_page * 3:
                axes[row, col].axis('off')

    # 컬러바 추가 (우측에)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Value')

    cbar_ax2 = fig.add_axes([0.94, 0.3, 0.01, 0.4])
    sm2 = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=(vmax-vmin)*0.3))
    sm2.set_array([])
    fig.colorbar(sm2, cax=cbar_ax2, label='Error')

    # 페이지 저장
    plt.savefig(os.path.join(save_result_path, f'temporal_sr_results_page{page_idx+1}.pdf'),
    #plt.savefig(os.path.join(save_result_path, f'temporal_sr_results_page{page_idx+1}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved page {page_idx + 1}")

# --- Calculate Overall Metrics ---
print("\nCalculating metrics...")
all_mse = []
all_mae = []

for i in range(num_test_samples):
    pred = predictions[i][0]
    target = test_targets_np[i]

    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))

    all_mse.append(mse)
    all_mae.append(mae)

# --- Save Test Results for Postprocessing ---
print("\nSaving test results for postprocessing...")

results_dict = {
    'model_type': 'edm_single',
    'test_inputs': test_inputs_np,
    'test_targets': test_targets_np,
    'predictions': np.array([pred[0] for pred in predictions]),  # Final predictions
    'intermediate_predictions': None,  # For EDM timesteps if needed
    'metrics': {
        'overall_mse': np.mean(all_mse),
        'overall_mae': np.mean(all_mae)
    },
    'model_config': {
        'model_name': args.model,
        'num_unets': num_unets,
        'max_cutoff': args.max_cutoff,
        'min_cutoff': args.min_cutoff,
        'C_in': C_in,
        'C_out': C_out,
        'sequence_length': sequence_length,
        'temporal_stride': temporal_stride
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
print(f"Temporal Super-Resolution Evaluation Complete!")
print(f"{'='*60}")
print(f"Model: {args.model.upper()}")
print(f"Evaluated {num_test_samples} test samples")
print(f"Overall MSE: {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
print(f"Overall MAE: {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
print(f"Results saved in: {save_result_path}")
for i in range(num_pages):
    print(f"  - temporal_sr_results_page{i+1}.pdf")
    #print(f"  - temporal_sr_results_page{i+1}.png")
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
