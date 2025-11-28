"""
Spatial Super Resolution Dataset Library

Usage:
    from utils.spatialSR import create_lowres_from_highres, visualize_lowres_samples

    # Create low-resolution versions from high-resolution data
    X_lowres = create_lowres_from_highres(
        data_highres,
        target_grid_size=(8, 8),
        u_ref=0.0499
    )
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def create_lowres_from_highres(
    data_highres,
    target_grid_sizes=[8, 16],
    u_ref=None,
    normalize=True,
    remove_nan=True,
    verbose=True
):
    """
    Create low-resolution versions from high-resolution data.

    Parameters:
    -----------
    data_highres : numpy.ndarray
        High-resolution data with shape (N, C, H, W) or (N, 1, H, W)
    target_grid_sizes : list of int or list of tuples
        Target grid sizes for downsampling.
        If int: (size, size), if tuple: (height, width)
        Example: [8, 16] or [(8, 8), (16, 16)]
    u_ref : float, optional
        Reference velocity for normalization. If None, no normalization.
    normalize : bool
        Whether to normalize by u_ref
    remove_nan : bool
        Whether to remove samples with NaN values
    verbose : bool
        Whether to print progress information

    Returns:
    --------
    dict with keys:
        - 'highres': normalized high-res data (if normalize=True)
        - 'lowres_X': low-res data for each grid size X
        - 'grid_sizes': list of actual grid sizes used
        - 'u_ref': normalization reference (if used)
        - 'nan_indices': indices of removed NaN samples (if remove_nan=True)
    """

    if verbose:
        print(f"{'='*60}")
        print(f"Creating Low-Resolution Data from High-Resolution")
        print(f"{'='*60}")
        print(f"Original shape: {data_highres.shape}")

    data = data_highres.copy()

    # Remove NaN samples
    nan_indices = []
    if remove_nan:
        nan_mask = np.isnan(data).any(axis=(1, 2, 3))
        nan_indices = np.where(nan_mask)[0]
        if len(nan_indices) > 0:
            if verbose:
                print(f"Found {len(nan_indices)} samples with NaN: {nan_indices.tolist()}")
            data = data[~nan_mask]
            if verbose:
                print(f"After removing NaN: {data.shape}")

    # Normalize
    if normalize and u_ref is not None:
        data = data / u_ref
        if verbose:
            print(f"Normalized by u_ref = {u_ref}")

    # Parse grid sizes
    parsed_grid_sizes = []
    for size in target_grid_sizes:
        if isinstance(size, int):
            parsed_grid_sizes.append((size, size))
        else:
            parsed_grid_sizes.append(tuple(size))

    if verbose:
        print(f"\nTarget grid sizes: {parsed_grid_sizes}")
        print(f"Original grid size: {data.shape[2:4]}")

    # Prepare output dictionary
    result = {
        'highres': data,
        'grid_sizes': parsed_grid_sizes,
        'nan_indices': nan_indices if remove_nan else None,
        'u_ref': u_ref if normalize else None
    }

    # Get original grid size
    N, C, H_orig, W_orig = data.shape

    # Create low-resolution versions for each target grid size
    for grid_h, grid_w in parsed_grid_sizes:
        if verbose:
            print(f"\nProcessing grid size ({grid_h}, {grid_w})...")

        lowres_data = np.zeros_like(data)

        for i in tqdm(range(N), desc=f"Downsampling to {grid_h}x{grid_w}", disable=not verbose):
            for c in range(C):
                hr = torch.from_numpy(data[i, c]).float()  # (H, W)

                # Downsample: H x W -> grid_h x grid_w (bicubic)
                lr = F.interpolate(
                    hr.unsqueeze(0).unsqueeze(0),
                    size=(grid_h, grid_w),
                    mode="bicubic",
                    align_corners=False
                ).squeeze(0).squeeze(0)

                # Upsample back: grid_h x grid_w -> H x W (nearest neighbor)
                lr_upsampled = cv2.resize(
                    lr.numpy(),
                    (W_orig, H_orig),
                    interpolation=cv2.INTER_NEAREST
                )

                lowres_data[i, c] = lr_upsampled

        # Store in result dictionary
        key_name = f'lowres_{grid_h}x{grid_w}'
        result[key_name] = lowres_data

        if verbose:
            print(f"  Created {key_name}: {lowres_data.shape}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Low-resolution data creation complete!")
        print(f"{'='*60}")

    return result


def visualize_lowres_samples(
    result_dict,
    num_samples=3,
    channel_idx=0,
    save_path=None,
    cmap='viridis',
    figsize=None
):
    """
    Visualize comparison between high-res and low-res samples.

    Parameters:
    -----------
    result_dict : dict
        Dictionary returned from create_lowres_from_highres()
    num_samples : int
        Number of samples to visualize
    channel_idx : int
        Channel index to visualize (for multi-channel data)
    save_path : str, optional
        Path to save the figure. If None, displays interactively.
    cmap : str
        Colormap for visualization
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None.
    """

    highres = result_dict['highres']
    grid_sizes = result_dict['grid_sizes']

    num_samples = min(num_samples, highres.shape[0])
    num_cols = 1 + len(grid_sizes)  # highres + lowres variants

    if figsize is None:
        figsize = (4 * num_cols, 4 * num_samples)

    fig, axes = plt.subplots(num_samples, num_cols, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # High-resolution
        axes[i, 0].imshow(highres[i, channel_idx], cmap=cmap)
        axes[i, 0].set_title(f'High-Res (Sample {i})')
        axes[i, 0].axis('off')

        # Low-resolution versions
        for j, (grid_h, grid_w) in enumerate(grid_sizes):
            key_name = f'lowres_{grid_h}x{grid_w}'
            lowres = result_dict[key_name]

            axes[i, j+1].imshow(lowres[i, channel_idx], cmap=cmap)
            axes[i, j+1].set_title(f'Low-Res {grid_h}x{grid_w} (Sample {i})')
            axes[i, j+1].axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def save_lowres_data(result_dict, save_dir, dataset_name='data'):
    """
    Save low-resolution data to disk.

    Parameters:
    -----------
    result_dict : dict
        Dictionary returned from create_lowres_from_highres()
    save_dir : str
        Directory to save the data
    dataset_name : str
        Base name for saved files
    """

    os.makedirs(save_dir, exist_ok=True)

    # Save high-res (normalized)
    highres_path = os.path.join(save_dir, f'{dataset_name}_highres.npy')
    np.save(highres_path, result_dict['highres'])
    print(f"Saved high-res data to {highres_path}")

    # Save each low-res version
    for grid_h, grid_w in result_dict['grid_sizes']:
        key_name = f'lowres_{grid_h}x{grid_w}'
        lowres_path = os.path.join(save_dir, f'{dataset_name}_{key_name}.npy')
        np.save(lowres_path, result_dict[key_name])
        print(f"Saved {key_name} data to {lowres_path}")

    # Save metadata
    metadata = {
        'grid_sizes': result_dict['grid_sizes'],
        'u_ref': result_dict['u_ref'],
        'nan_indices': result_dict['nan_indices'],
        'shape': result_dict['highres'].shape
    }
    metadata_path = os.path.join(save_dir, f'{dataset_name}_metadata.npy')
    np.save(metadata_path, metadata)
    print(f"Saved metadata to {metadata_path}")

    print(f"\n{'='*60}")
    print(f"All data saved to {save_dir}")
    print(f"{'='*60}")


def print_usage():
    """Print usage examples"""
    print("""
🔥 Spatial SR Dataset Library Usage

1️⃣ Create low-resolution from high-resolution:
   from utils.spatialSR import create_lowres_from_highres, save_lowres_data

   # Load high-res data
   data = np.load('data_highres.npy')

   # Create low-res versions
   result = create_lowres_from_highres(
       data,
       target_grid_sizes=[8, 16, 32],
       u_ref=0.0499,
       normalize=True
   )

   # Save
   save_lowres_data(result, save_dir='./processed_data', dataset_name='my_dataset')

2️⃣ Visualize results:
   from utils.spatialSR import visualize_lowres_samples

   visualize_lowres_samples(
       result,
       num_samples=3,
       save_path='comparison.png'
   )

3️⃣ Access data:
   highres = result['highres']
   lowres_8x8 = result['lowres_8x8']
   lowres_16x16 = result['lowres_16x16']

📚 Key Features:
   - Automatic NaN removal
   - Normalization by reference velocity
   - Multiple grid sizes in one call
   - Bicubic downsampling + Nearest neighbor upsampling
   - Built-in visualization
""")


if __name__ == "__main__":
    print_usage()
