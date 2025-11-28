"""
Plotting utilities for energy cascade analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def compute_energy_spectrum(field):
    """
    Compute 1D radially averaged energy spectrum from 2D field

    Args:
        field: 2D numpy array (H, W)

    Returns:
        k: wavenumber array
        E_k: energy spectrum
    """
    H, W = field.shape

    # 2D FFT
    f_hat = np.fft.fft2(field)
    psd = np.abs(f_hat)**2 / (H * W)

    # Create wavenumber grid
    kx = np.fft.fftfreq(W, d=1.0)
    ky = np.fft.fftfreq(H, d=1.0)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    psd = np.fft.fftshift(psd)

    # Radial binning
    k_max = 0.5  # Nyquist frequency
    n_bins = min(H, W) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    E_k = np.zeros(len(k_centers))

    for i in range(len(k_centers)):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        if np.any(mask):
            E_k[i] = np.mean(psd[mask])

    # Remove trailing zeros
    nonzero = E_k > 1e-15
    if np.any(nonzero):
        last = np.where(nonzero)[0][-1] + 1
        return k_centers[:last], E_k[:last]

    return k_centers, E_k


def extract_field_from_batch(data, sample_idx=0, channel_idx=None):
    """
    Extract 2D field from batch data with various dimensions

    Args:
        data: Can be (B, C, H, W), (C, H, W), or (H, W)
        sample_idx: Which sample to extract from batch
        channel_idx: Which channel to extract (None = use middle channel)

    Returns:
        field: 2D numpy array (H, W)
    """
    if isinstance(data, dict):
        # Handle dictionary with 'target' or 'input' keys
        if 'target' in data:
            data = data['target']
        elif 'input' in data:
            data = data['input']

    # Convert to numpy if tensor
    if hasattr(data, 'cpu'):
        data = data.cpu().numpy()

    if data.ndim == 4:
        # (B, C, H, W)
        n_channels = data.shape[1]
        if channel_idx is None:
            channel_idx = n_channels // 2  # Use middle channel
        field = data[sample_idx, channel_idx]
    elif data.ndim == 3:
        # (C, H, W) or (B, H, W)
        if data.shape[0] <= 10:  # Likely channel dimension
            n_channels = data.shape[0]
            if channel_idx is None:
                channel_idx = n_channels // 2
            field = data[channel_idx]
        else:  # Likely batch dimension
            field = data[sample_idx]
    elif data.ndim == 2:
        # (H, W)
        field = data
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    return field


def plot_energy_cascade_analysis(
    gt_data,
    model_predictions,
    test_loader=None,
    save_path='./test',
    experiment_name='energy_cascade'
):
    """
    Create energy cascade analysis comparing GT with multiple model predictions

    Creates 4 plots:
    1. Average spectrum over all test samples
    2-4. Individual sample comparisons (3 random samples)

    Args:
        gt_data: Ground truth data, can be:
                 - numpy array (B, C, H, W) or (C, H, W) or (H, W)
                 - DataLoader (will iterate through)
        model_predictions: Dictionary with model names as keys and predictions as values
                          e.g., {'edm': edm_pred, 'ddim': ddim_pred, 'cnn': cnn_pred}
                          Can contain any subset of models
        test_loader: Optional DataLoader for GT data
        save_path: Directory to save plots
        experiment_name: Name for saved file
    """
    os.makedirs(save_path, exist_ok=True)

    # Get list of available models
    available_models = list(model_predictions.keys())
    print(f"Available models for comparison: {available_models}")

    # Collect all spectra for averaging
    gt_spectra = []
    model_spectra = {model: [] for model in available_models}

    all_gt_fields = []
    all_model_fields = {model: [] for model in available_models}

    # Handle different input types
    if test_loader is not None:
        # Use DataLoader
        print("Computing energy spectra from test_loader...")
        for batch in test_loader:
            batch_size = batch['target'].shape[0] if isinstance(batch, dict) else batch.shape[0]

            for i in range(batch_size):
                # Extract fields
                gt_field = extract_field_from_batch(batch, sample_idx=i)

                # For predictions, assume they're already numpy arrays matched to test set
                if len(all_gt_fields) < len(gt_data) if hasattr(gt_data, '__len__') else True:
                    idx = len(all_gt_fields)

                    # Compute GT spectrum
                    k_gt, E_gt = compute_energy_spectrum(gt_field)
                    gt_spectra.append(E_gt)
                    all_gt_fields.append(gt_field)

                    # Compute spectra for each available model
                    for model_name in available_models:
                        model_pred = model_predictions[model_name]
                        model_field = extract_field_from_batch(model_pred, sample_idx=idx)
                        k_model, E_model = compute_energy_spectrum(model_field)

                        model_spectra[model_name].append(E_model)
                        all_model_fields[model_name].append(model_field)
    else:
        # Use numpy arrays directly
        print("Computing energy spectra from numpy arrays...")
        n_samples = gt_data.shape[0] if gt_data.ndim >= 3 else 1

        for i in range(n_samples):
            gt_field = extract_field_from_batch(gt_data, sample_idx=i)

            # Compute GT spectrum
            k_gt, E_gt = compute_energy_spectrum(gt_field)
            gt_spectra.append(E_gt)
            all_gt_fields.append(gt_field)

            # Compute spectra for each available model
            for model_name in available_models:
                model_pred = model_predictions[model_name]
                model_field = extract_field_from_batch(model_pred, sample_idx=i)
                k_model, E_model = compute_energy_spectrum(model_field)

                model_spectra[model_name].append(E_model)
                all_model_fields[model_name].append(model_field)

    print(f"Collected {len(gt_spectra)} samples for analysis")

    # Average spectra
    # Find minimum length for consistent comparison
    all_spectra = gt_spectra[:]
    for model_name in available_models:
        all_spectra.extend(model_spectra[model_name])

    min_len = min([len(E) for E in all_spectra])

    gt_avg = np.mean([E[:min_len] for E in gt_spectra], axis=0)
    model_avg = {model: np.mean([E[:min_len] for E in model_spectra[model]], axis=0)
                 for model in available_models}
    k_avg = k_gt[:min_len]

    # Select 3 random samples for individual comparison
    n_samples = min(3, len(all_gt_fields))
    sample_indices = np.random.choice(len(all_gt_fields), n_samples, replace=False)

    # Create figure with 4 subplots (1 average + 3 individuals)
    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    colors = {
        'GT': '#000000',
        'edm': '#FF4444',
        'ddim': '#4444FF',
        'cnn': '#44AA44'
    }

    linestyles = {
        'edm': '--',
        'ddim': '-.',
        'cnn': ':'
    }

    # Plot 1: Average spectrum
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.loglog(k_avg[1:], gt_avg[1:], color=colors['GT'], linewidth=3,
               label='GT', alpha=0.9)

    # Plot each available model
    for model_name in available_models:
        ax0.loglog(k_avg[1:], model_avg[model_name][1:],
                   color=colors.get(model_name, '#888888'),
                   linewidth=2.5,
                   label=model_name.upper(),
                   alpha=0.8,
                   linestyle=linestyles.get(model_name, '-'))

    # Add reference slopes
    k_ref = np.logspace(np.log10(k_avg[1]), np.log10(k_avg[-1]*0.5), 50)
    ax0.loglog(k_ref, 0.1*k_ref**(-5/3), 'k:', alpha=0.4, linewidth=1.5, label=r'$k^{-5/3}$')
    ax0.loglog(k_ref, 0.01*k_ref**(-3), 'k-.', alpha=0.4, linewidth=1.5, label=r'$k^{-3}$')

    ax0.set_xlabel('Wavenumber k', fontsize=12, fontweight='bold')
    ax0.set_ylabel('E(k)', fontsize=12, fontweight='bold')
    ax0.set_title(f'Average Spectrum\n({len(gt_spectra)} samples)',
                  fontsize=13, fontweight='bold')
    ax0.legend(loc='best', fontsize=10, framealpha=0.9)
    ax0.grid(True, alpha=0.3, which='both')

    # Plots 2-4: Individual samples
    for plot_idx, sample_idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[0, plot_idx + 1])

        # Get spectra for this sample
        k_s, E_gt_s = compute_energy_spectrum(all_gt_fields[sample_idx])

        ax.loglog(k_s[1:], E_gt_s[1:], color=colors['GT'], linewidth=3,
                  label='GT', alpha=0.9)

        # Plot each available model
        model_E_s = {}
        for model_name in available_models:
            k_s, E_model_s = compute_energy_spectrum(all_model_fields[model_name][sample_idx])
            model_E_s[model_name] = E_model_s
            ax.loglog(k_s[1:], E_model_s[1:],
                      color=colors.get(model_name, '#888888'),
                      linewidth=2.5,
                      label=model_name.upper(),
                      alpha=0.8,
                      linestyle=linestyles.get(model_name, '-'))

        # Add reference slopes
        k_ref_s = np.logspace(np.log10(k_s[1]), np.log10(k_s[-1]*0.5), 50)
        ax.loglog(k_ref_s, 0.1*k_ref_s**(-5/3), 'k:', alpha=0.4, linewidth=1.5,
                  label=r'$k^{-5/3}$')

        ax.set_xlabel('Wavenumber k', fontsize=12, fontweight='bold')
        ax.set_ylabel('E(k)', fontsize=12, fontweight='bold')
        ax.set_title(f'Sample {sample_idx + 1}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')

        # Compute and display spectral slopes
        # Fit in log-log space for inertial range (middle 1/3 of spectrum)
        start_idx = len(k_s) // 3
        end_idx = 2 * len(k_s) // 3

        def compute_slope(k, E, start, end):
            log_k = np.log10(k[start:end])
            log_E = np.log10(E[start:end] + 1e-20)
            valid = np.isfinite(log_k) & np.isfinite(log_E)
            if np.sum(valid) >= 3:
                slope, _ = np.polyfit(log_k[valid], log_E[valid], 1)
                return slope
            return np.nan

        slope_gt = compute_slope(k_s, E_gt_s, start_idx, end_idx)

        # Add text box with slopes
        slope_text = f'Spectral slopes:\n'
        slope_text += f'GT: {slope_gt:.2f}\n'

        for model_name in available_models:
            slope_model = compute_slope(k_s, model_E_s[model_name], start_idx, end_idx)
            slope_text += f'{model_name.upper()}: {slope_model:.2f}\n'

        ax.text(0.98, 0.02, slope_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Create dynamic title
    model_names_str = ' vs '.join([m.upper() for m in available_models])
    plt.suptitle(f'Energy Cascade Analysis: GT vs {model_names_str}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    save_file = os.path.join(save_path, f'{experiment_name}_comparison.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Energy cascade analysis saved to: {save_file}")

    # Build return dictionary
    return_dict = {
        'average_spectra': {
            'k': k_avg,
            'GT': gt_avg
        },
        'sample_indices': sample_indices,
        'n_samples_analyzed': len(gt_spectra)
    }

    # Add model spectra to return dict
    for model_name in available_models:
        return_dict['average_spectra'][model_name.upper()] = model_avg[model_name]

    return return_dict


def plot_q_sample_energy_cascade(
    q_sample_snapshots,
    timesteps,
    save_path='./test',
    experiment_name='q_sample_energy_cascade',
    sample_idx=0,
    channel_idx=0
):
    """
    Plot energy spectra evolution across q_sample timesteps (Kolmogorov cascade analysis)

    Args:
        q_sample_snapshots: List of snapshots at different timesteps
                          Each element is (B, C, H, W) numpy array
        timesteps: List of timestep indices
        save_path: Directory to save plots
        experiment_name: Name for saved file
        sample_idx: Which sample to analyze from batch
        channel_idx: Which channel to analyze

    Returns:
        dict with spectra data
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"\nComputing energy spectra for q_sample forward process...")
    print(f"Analyzing sample {sample_idx}, channel {channel_idx}")

    # Compute spectra for each timestep
    all_k = []
    all_E = []

    for t_idx, snapshot in enumerate(q_sample_snapshots):
        field = extract_field_from_batch(snapshot, sample_idx=sample_idx, channel_idx=channel_idx)
        k, E = compute_energy_spectrum(field)
        all_k.append(k)
        all_E.append(E)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: All spectra overlaid
    ax0 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))

    for i, (t, k, E) in enumerate(zip(timesteps, all_k, all_E)):
        if len(k) > 1 and len(E) > 1:
            ax0.loglog(k[1:], E[1:], color=colors[i], linewidth=2.5,
                      label=f't={t}', alpha=0.7)

    # Add Kolmogorov reference
    if len(all_k) > 0 and len(all_k[0]) > 1:
        k_ref = np.logspace(np.log10(all_k[0][1]), np.log10(all_k[0][-1]*0.5), 50)
        E_ref = 0.1 * k_ref**(-5/3)
        ax0.loglog(k_ref, E_ref, 'k--', linewidth=2, alpha=0.5, label=r'$k^{-5/3}$ (Kolmogorov)')

    ax0.set_xlabel('Wavenumber k', fontsize=14, fontweight='bold')
    ax0.set_ylabel('Energy E(k)', fontsize=14, fontweight='bold')
    ax0.set_title('Energy Spectra Evolution (q_sample)', fontsize=15, fontweight='bold')
    ax0.legend(loc='best', fontsize=9, ncol=2)
    ax0.grid(True, alpha=0.3, which='both')

    # Right plot: Total energy vs timestep
    ax1 = axes[1]
    total_energies = [np.sum(E) for E in all_E]
    ax1.plot(timesteps, total_energies, 'o-', linewidth=2.5, markersize=8, color='darkblue')
    ax1.set_xlabel('Timestep t', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total Energy', fontsize=14, fontweight='bold')
    ax1.set_title('Energy Dissipation vs Timestep', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Add energy decay annotation
    if len(total_energies) > 1:
        decay_ratio = total_energies[-1] / total_energies[0]
        ax1.text(0.05, 0.95, f'Energy decay: {decay_ratio:.2e}\n(t=0 → t={timesteps[-1]})',
                transform=ax1.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'Forward Process (q_sample) Energy Cascade Analysis',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    save_file = os.path.join(save_path, f'{experiment_name}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"q_sample energy cascade plot saved to: {save_file}")

    return {
        'timesteps': timesteps,
        'spectra': {'k': all_k, 'E': all_E},
        'total_energies': total_energies
    }
