from model.unet import UNet
from model.edm_single import EDM, EDM_AE, FrequencyProgressiveUNet
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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils.s2s import create_snapshot_dataloaders 
import argparse


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

############################ Grid-Aligned Interpolation Functions ############################

def rbf_interpolation_efficient(x_low, target_size=256, rbf_function='gaussian', epsilon=1.0):
    """
    Efficient RBF interpolation using local neighborhoods
    """
    batch_size, channels, low_h, low_w = x_low.shape
    device = x_low.device
    
    # Use simple bilinear as fallback for large scale differences
    scale_factor = target_size / low_h
    if scale_factor > 16:  # Too large scale factor for RBF
        return F.interpolate(x_low, size=target_size, mode='bilinear', align_corners=False)
    
    # More efficient: Use local RBF patches
    result = torch.zeros(batch_size, channels, target_size, target_size, device=device)
    
    # Define local patch size based on scale factor
    patch_radius = max(2, int(scale_factor + 1))
    
    for b in range(batch_size):
        for c in range(channels):
            result[b, c] = _rbf_interpolate_channel(
                x_low[b, c], target_size, rbf_function, epsilon, patch_radius
            )
    
    return result

def _rbf_interpolate_channel(channel_data, target_size, rbf_function, epsilon, patch_radius):
    """RBF interpolation for a single channel"""
    low_h, low_w = channel_data.shape
    device = channel_data.device
    scale_factor = target_size / low_h
    
    # Create output grid
    result = torch.zeros(target_size, target_size, device=device)
    
    # High-res coordinates
    hi_y, hi_x = torch.meshgrid(
        torch.arange(target_size, device=device, dtype=torch.float32),
        torch.arange(target_size, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Map high-res coordinates to low-res space
    low_y_mapped = hi_y / scale_factor
    low_x_mapped = hi_x / scale_factor
    
    # For each high-res pixel, find local neighborhood in low-res
    for i in range(target_size):
        for j in range(target_size):
            # Current high-res pixel position in low-res space
            ly = low_y_mapped[i, j]
            lx = low_x_mapped[i, j]
            
            # Find local neighborhood
            min_ly = max(0, int(ly - patch_radius))
            max_ly = min(low_h, int(ly + patch_radius + 1))
            min_lx = max(0, int(lx - patch_radius))
            max_lx = min(low_w, int(lx + patch_radius + 1))
            
            # Extract local values and coordinates
            local_values = []
            local_coords = []
            
            for ii in range(min_ly, max_ly):
                for jj in range(min_lx, max_lx):
                    local_values.append(channel_data[ii, jj].item())
                    local_coords.append([float(ii), float(jj)])
            
            if len(local_values) == 0:
                continue
                
            local_values = torch.tensor(local_values, device=device)
            local_coords = torch.tensor(local_coords, device=device)
            
            # Compute distances
            current_coord = torch.tensor([[ly, lx]], device=device)
            distances = torch.cdist(current_coord, local_coords, p=2).squeeze(0)
            
            # Apply RBF
            if rbf_function == 'gaussian':
                weights = torch.exp(-(epsilon * distances) ** 2)
            elif rbf_function == 'multiquadric':
                weights = torch.sqrt(1 + (epsilon * distances) ** 2)
            elif rbf_function == 'thin_plate':
                weights = torch.where(distances > 0, 
                                    distances ** 2 * torch.log(distances + 1e-10), 
                                    torch.zeros_like(distances))
            else:
                weights = torch.exp(-(distances) ** 2)  # Default gaussian
            
            # Normalize weights
            weight_sum = weights.sum()
            if weight_sum > 1e-10:
                weights = weights / weight_sum
                result[i, j] = (weights * local_values).sum()
            else:
                # Fallback to nearest neighbor
                nearest_idx = distances.argmin()
                result[i, j] = local_values[nearest_idx]
    
    return result

def rbf_interpolation_scipy(x_low, target_size=256, rbf_function='gaussian', epsilon=1.0):
    """
    SciPy-based RBF interpolation for better accuracy
    """
    try:
        from scipy.interpolate import RBFInterpolator
        import numpy as np
        
        batch_size, channels, low_h, low_w = x_low.shape
        device = x_low.device
        dtype = x_low.dtype
        
        # Convert kernel name
        kernel_map = {
            'gaussian': 'gaussian',
            'multiquadric': 'multiquadric', 
            'thin_plate': 'thin_plate_spline'
        }
        scipy_kernel = kernel_map.get(rbf_function, 'multiquadric')
        
        result = torch.zeros(batch_size, channels, target_size, target_size, device=device, dtype=dtype)
        
        for b in range(batch_size):
            for c in range(channels):
                # Get low-res data
                low_data = x_low[b, c].cpu().numpy()  # (low_h, low_w)
                
                # Create low-res coordinate grid
                y_low = np.arange(low_h)
                x_low_grid = np.arange(low_w)
                yy_low, xx_low = np.meshgrid(y_low, x_low_grid, indexing='ij')
                
                # Low-res positions (row, col format)
                positions = np.column_stack([yy_low.ravel(), xx_low.ravel()])
                values = low_data.ravel()
                
                # Create RBF interpolator
                rbf = RBFInterpolator(positions, values.reshape(-1, 1),
                                    kernel=scipy_kernel, epsilon=epsilon)
                
                # Create high-res coordinate grid
                y_high = np.linspace(0, low_h-1, target_size)
                x_high = np.linspace(0, low_w-1, target_size)
                yy_high, xx_high = np.meshgrid(y_high, x_high, indexing='ij')
                
                # High-res grid points (row, col format)
                grid_points = np.column_stack([yy_high.ravel(), xx_high.ravel()])
                
                # Interpolate
                pred_rbf = rbf(grid_points).reshape(target_size, target_size)
                
                # Convert back to tensor
                result[b, c] = torch.from_numpy(pred_rbf).to(device=device, dtype=dtype)
        
        return result
        
    except ImportError:
        print("SciPy not available, falling back to custom RBF implementation")
        return rbf_interpolation_efficient(x_low, target_size, rbf_function, epsilon)

def rbf_interpolation(x_low, target_size=256, rbf_function='gaussian', epsilon=1.0):
    """
    RBF interpolation with SciPy fallback to custom implementation
    """
    batch_size, channels, low_h, low_w = x_low.shape
    device = x_low.device
    scale_factor = target_size / low_h
    
    # Try SciPy implementation first
    try:
        return rbf_interpolation_scipy(x_low, target_size, rbf_function, epsilon)
    except:
        # Fallback to custom implementation
        if low_h <= 16 or scale_factor >= 8:
            return rbf_interpolation_efficient(x_low, target_size, rbf_function, epsilon)
        else:
            # For moderate scales, use hybrid approach
            intermediate_size = min(target_size, low_h * 4)
            x_bilinear = F.interpolate(x_low, size=intermediate_size, mode='bilinear', align_corners=False)
            
            if intermediate_size < target_size:
                x_rbf = rbf_interpolation_efficient(x_bilinear, target_size, rbf_function, epsilon)
                return x_rbf
            else:
                return x_bilinear

def grid_aligned_interpolation(x_low, target_size=256, method='fourier', rbf_function='gaussian', epsilon=1.0):
    """
    Grid-aligned interpolation from low resolution to high resolution
    
    Args:
        x_low: Low-resolution input tensor (B, C, H, W)
        target_size: Target resolution (default: 256)
        method: 'fourier', 'bilinear', or 'rbf'
        rbf_function: RBF function type ('gaussian', 'multiquadric', 'thin_plate') - only for method='rbf'
        epsilon: Shape parameter for RBF - only for method='rbf'
    
    Returns:
        Interpolated high-resolution tensor
    """
    current_size = x_low.shape[-1]
    
    if current_size >= target_size:
        return x_low
    
    if method == 'fourier':
        # Fourier domain interpolation (prevents aliasing)
        x_fft = torch.fft.rfft2(x_low)
        
        # Calculate padding for frequency domain
        h_pad = (target_size - current_size) // 2
        w_pad = (target_size//2 + 1) - (current_size//2 + 1)
        
        # Zero-padding in frequency domain
        x_fft_padded = F.pad(x_fft, (0, w_pad, h_pad, h_pad))
        
        # Inverse FFT to get high-resolution image
        x_interp = torch.fft.irfft2(x_fft_padded, s=(target_size, target_size))
        
        # Scale factor correction
        scale_factor = (target_size / current_size) ** 2
        x_interp = x_interp * scale_factor
        
    elif method == 'rbf':
        # RBF interpolation
        x_interp = rbf_interpolation(x_low, target_size, rbf_function, epsilon)
        
    else:  # bilinear
        x_interp = F.interpolate(x_low, size=target_size, 
                               mode='bilinear', align_corners=False)
    
    return x_interp

def create_ring_mask(h, w, inner_radius, outer_radius):
    """Create a ring mask for radial spectrum computation"""
    center_h, center_w = h // 2, w // 2
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    # Distance from center
    dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
    
    # Ring mask
    mask = (dist >= inner_radius) & (dist < outer_radius)
    return mask.float()

############################ TimestepEstimator Class ############################

class TimestepEstimator:
    """Estimates the optimal EDM timestep for interpolated low-resolution images"""

    def __init__(self, edm_model, device):
        self.edm = edm_model
        self.device = device
        self.precomputed_spectra = None
        self.resolution_timestep_lookup = {}  # Resolution -> best timestep mapping
        self._precompute_edm_spectra()
    
    def _precompute_edm_spectra(self):
        """Precompute frequency characteristics for each EDM timestep"""
        print("Precomputing EDM frequency spectra for timestep estimation...")
        
        # Use a reference high-frequency image (get output channels from final layer)
        target_channels = self.edm.unet.final[-1].out_channels  # Get actual output channels from UNet
        # Use flexible size - get from args or default to 256
        ref_size = getattr(args, 'target_size', 256) if 'args' in globals() else 256
        reference_image = torch.randn(1, target_channels, ref_size, ref_size).to(self.device)
        
        self.precomputed_spectra = {}
        
        for t in range(self.edm.num_timesteps):
            # Generate filtered image at timestep t
            t_tensor = torch.tensor([t], device=self.device).long()
            x_filtered = self.edm.q_sample(reference_image, t_tensor)
            
            # Compute frequency spectrum
            spectrum = torch.fft.rfft2(x_filtered)
            energy = torch.abs(spectrum)**2
            
            # Compute radial average (frequency band-wise energy)
            radial_spectrum = self.compute_radial_spectrum(energy)
            self.precomputed_spectra[t] = radial_spectrum.cpu()
            
        print(f"Precomputed spectra for {self.edm.num_timesteps} timesteps")
    
    def compute_radial_spectrum(self, energy_2d):
        """Convert 2D spectrum to 1D radial spectrum"""
        h, w = energy_2d.shape[-2:]
        center = (h//2, w//2)
        
        max_radius = min(center)
        radial_profile = torch.zeros(max_radius, device=energy_2d.device)
        
        for r in range(max_radius):
            mask = create_ring_mask(h, w, r, r+1).to(energy_2d.device)
            
            # Average energy in this ring
            if mask.sum() > 0:
                radial_profile[r] = (energy_2d * mask).sum() / mask.sum()
        
        return radial_profile
    
    def learn_resolution_mapping(self, train_loader, test_resolutions, target_size,
                                interpolation_method='rbf', rbf_function='gaussian', epsilon=1.0):
        """
        Learn the best timestep for each resolution by analyzing training data

        Args:
            train_loader: DataLoader with training samples
            test_resolutions: List of resolutions to learn (e.g., [8, 16, 32])
            target_size: Target high-resolution size
            interpolation_method: Interpolation method to use
        """
        print("\nLearning resolution-to-timestep mapping from training data...")

        resolution_timesteps = {res: [] for res in test_resolutions}

        # Sample from training data
        num_samples = min(20, len(train_loader.dataset))  # Use 20 samples (faster)
        sample_count = 0

        for batch in train_loader:
            if sample_count >= num_samples:
                break

            targets = batch['target'].to(self.device)
            batch_size = targets.shape[0]

            for i in range(min(batch_size, num_samples - sample_count)):
                target = targets[i:i+1]

                for res in test_resolutions:
                    # Downsample and interpolate
                    low_res = F.interpolate(target, size=res, mode='area')
                    x_interp = grid_aligned_interpolation(low_res, target_size=target_size,
                                                         method=interpolation_method,
                                                         rbf_function=rbf_function,
                                                         epsilon=epsilon)

                    # Find best matching timestep using SSIM
                    ssim_scores = {}
                    for t in range(self.edm.num_timesteps):
                        t_tensor = torch.tensor([t], device=self.device).long()
                        x_t = self.edm.q_sample(target, t_tensor)
                        ssim_val = self._compute_ssim(x_interp, x_t)
                        ssim_scores[t] = ssim_val

                    best_t = max(ssim_scores.keys(), key=ssim_scores.get)
                    resolution_timesteps[res].append(best_t)

                sample_count += 1
                if sample_count >= num_samples:
                    break

        # Compute average timestep for each resolution
        for res in test_resolutions:
            if resolution_timesteps[res]:
                avg_t = int(np.mean(resolution_timesteps[res]))
                std_t = np.std(resolution_timesteps[res])
                self.resolution_timestep_lookup[res] = avg_t
                print(f"  {res}×{res} → t={avg_t} (std={std_t:.2f})")

        return self.resolution_timestep_lookup

    def estimate_timestep(self, x_interpolated, input_resolution=None, method='lookup'):
        """
        Estimate the best EDM timestep for an interpolated image

        Args:
            x_interpolated: Interpolated high-resolution image (B, C, H, W)
            input_resolution: Original low-res size (for lookup method)
            method: 'lookup' (use learned mapping) or 'ssim' (compute on-the-fly)

        Returns:
            matched_timestep: Best matching EDM timestep
            confidence: Matching confidence
        """
        if method == 'lookup' and input_resolution in self.resolution_timestep_lookup:
            # Use pre-learned mapping
            best_t = self.resolution_timestep_lookup[input_resolution]
            confidence = 1.0  # High confidence from learned data
            return best_t, confidence
        elif method == 'ssim':
            # Fallback to SSIM-based estimation (requires GT - only for visualization)
            return self._estimate_by_ssim(x_interpolated)
        else:
            # Fallback to heuristic
            return self.resolution_to_timestep_heuristic(x_interpolated)

    def _estimate_by_hf_ratio(self, energy, radial_spectrum):
        """Estimate timestep based on high-frequency energy ratio"""
        # Calculate high-frequency energy ratio
        total_energy = radial_spectrum.sum()
        # High frequency = second half of spectrum
        hf_start = len(radial_spectrum) // 2
        hf_energy = radial_spectrum[hf_start:].sum()
        hf_ratio = (hf_energy / (total_energy + 1e-8)).item()

        # Compare with precomputed HF ratios
        hf_ratios = {}
        for t, ref_spectrum in self.precomputed_spectra.items():
            ref_total = ref_spectrum.sum()
            ref_hf_start = len(ref_spectrum) // 2
            ref_hf = ref_spectrum[ref_hf_start:].sum()
            ref_hf_ratio = (ref_hf / (ref_total + 1e-8)).item()
            hf_ratios[t] = ref_hf_ratio

        # Find closest match
        best_t = min(hf_ratios.keys(), key=lambda t: abs(hf_ratios[t] - hf_ratio))
        confidence = 1.0 - abs(hf_ratios[best_t] - hf_ratio)

        return best_t, confidence

    def _estimate_by_l2(self, radial_spectrum):
        """Estimate timestep based on L2 distance"""
        distances = {}
        for t, ref_spectrum in self.precomputed_spectra.items():
            min_len = min(len(radial_spectrum), len(ref_spectrum))
            if min_len > 0:
                spectrum1 = radial_spectrum[:min_len]
                spectrum2 = ref_spectrum[:min_len]

                # Normalize to unit norm for fair comparison
                spectrum1_norm = spectrum1 / (spectrum1.norm() + 1e-8)
                spectrum2_norm = spectrum2 / (spectrum2.norm() + 1e-8)

                distance = torch.norm(spectrum1_norm - spectrum2_norm, p=2).item()
                distances[t] = distance
            else:
                distances[t] = float('inf')

        best_t = min(distances.keys(), key=distances.get)
        # Convert distance to confidence (smaller distance = higher confidence)
        confidence = 1.0 / (1.0 + distances[best_t])

        return best_t, confidence

    def _estimate_by_cosine(self, radial_spectrum):
        """Estimate timestep based on cosine similarity (original method)"""
        similarities = {}
        for t, ref_spectrum in self.precomputed_spectra.items():
            min_len = min(len(radial_spectrum), len(ref_spectrum))

            if min_len > 0:
                spectrum1 = radial_spectrum[:min_len]
                spectrum2 = ref_spectrum[:min_len]

                spectrum1_norm = spectrum1 / (spectrum1.norm() + 1e-8)
                spectrum2_norm = spectrum2 / (spectrum2.norm() + 1e-8)

                similarity = F.cosine_similarity(spectrum1_norm, spectrum2_norm, dim=0)
                similarities[t] = similarity.item()
            else:
                similarities[t] = 0.0

        best_t = max(similarities, key=similarities.get)
        confidence = similarities[best_t]

        return best_t, confidence

    def _compute_ssim(self, img1, img2, window_size=11):
        """Compute SSIM between two images (simplified version)"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Simple average pooling as window
        kernel = torch.ones(1, 1, window_size, window_size, device=img1.device) / (window_size ** 2)

        mu1 = F.conv2d(img1, kernel, padding=window_size//2)
        mu2 = F.conv2d(img2, kernel, padding=window_size//2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size//2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    def _estimate_by_ssim(self, x_interpolated):
        """Estimate timestep based on SSIM (Structural Similarity) - find highest SSIM"""
        ssim_scores = {}
        for t in self.precomputed_spectra.keys():
            # Get filtered version at timestep t
            t_tensor = torch.tensor([t], device=x_interpolated.device).long()
            x_t = self.edm.q_sample(x_interpolated, t_tensor)

            # Compute SSIM (higher is better)
            ssim_val = self._compute_ssim(x_interpolated, x_t)
            ssim_scores[t] = ssim_val

        # Find timestep with highest SSIM (most similar structure)
        best_t = max(ssim_scores.keys(), key=ssim_scores.get)
        confidence = ssim_scores[best_t]

        return best_t, confidence

    def _estimate_hybrid(self, energy, radial_spectrum, x_interpolated):
        """Hybrid method combining multiple metrics"""
        # Get estimates from different methods
        t_hf, conf_hf = self._estimate_by_hf_ratio(energy, radial_spectrum)
        t_l2, conf_l2 = self._estimate_by_l2(radial_spectrum)
        t_cos, conf_cos = self._estimate_by_cosine(radial_spectrum)
        t_ssim, conf_ssim = self._estimate_by_ssim(x_interpolated)

        # Weighted voting based on confidence
        votes = {
            t_ssim: conf_ssim * 0.4,  # SSIM gets highest weight
            t_cos: conf_cos * 0.3,
            t_hf: conf_hf * 0.2,
            t_l2: conf_l2 * 0.1
        }

        # Aggregate votes
        timestep_scores = {}
        for t, weight in votes.items():
            timestep_scores[t] = timestep_scores.get(t, 0) + weight

        best_t = max(timestep_scores, key=timestep_scores.get)
        confidence = timestep_scores[best_t]

        return best_t, confidence
    
    def resolution_to_timestep_heuristic(self, x_interpolated):
        """Heuristic mapping from effective resolution to EDM timestep"""
        # Estimate effective resolution based on high-frequency content
        spectrum = torch.fft.rfft2(x_interpolated)
        energy = torch.abs(spectrum)**2
        
        # Find cutoff frequency (where energy drops significantly)
        radial_spectrum = self.compute_radial_spectrum(energy)
        total_energy = radial_spectrum.sum()
        
        # Find frequency where 90% of energy is contained
        cumulative_energy = torch.cumsum(radial_spectrum, dim=0)
        cutoff_idx = torch.where(cumulative_energy >= 0.9 * total_energy)[0]
        
        if len(cutoff_idx) > 0:
            effective_freq_ratio = cutoff_idx[0].float() / len(radial_spectrum)
        else:
            effective_freq_ratio = 1.0
        
        # Map frequency ratio to timestep
        # Lower frequency content → higher timestep (more degraded)
        timestep = int((1 - effective_freq_ratio) * (self.edm.num_timesteps - 1))
        timestep = max(0, min(timestep, self.edm.num_timesteps - 1))
        return timestep, 0.5  # Return tuple with confidence

############################ UniversalSuperResolution Class ############################

class UniversalSuperResolution:
    """Universal Super-Resolution using EDM progressive refinement"""
    
    def __init__(self, edm_model, device):
        self.edm = edm_model
        self.device = device
        self.timestep_estimator = TimestepEstimator(edm_model, device)
    
    def super_resolve(self, x_low, target_size=None, method='fourier', rbf_function='gaussian', epsilon=1.0):
        """
        Perform universal super-resolution from arbitrary resolution to target resolution
        
        Args:
            x_low: Low-resolution input (B, C, H_low, W_low)
            target_size: Target resolution (if None, use args.target_size or 256)
            method: Interpolation method ('fourier', 'bilinear', or 'rbf')
            rbf_function: RBF function type (only for method='rbf')
            epsilon: RBF shape parameter (only for method='rbf')
            
        Returns:
            x_high: Super-resolved high-resolution image
            refinement_history: List of intermediate results
        """
        if target_size is None:
            target_size = getattr(args, 'target_size', 256) if 'args' in globals() else 256
            
        input_res = x_low.shape[-1]
        print(f"Universal SR: {input_res}×{input_res} → {target_size}×{target_size} ({method})")

        # Step 1: Grid-aligned interpolation
        x_interp = grid_aligned_interpolation(x_low, target_size, method, rbf_function, epsilon)

        # Step 2: Estimate optimal starting timestep using learned lookup
        start_timestep, confidence = self.timestep_estimator.estimate_timestep(x_interp, input_resolution=input_res, method='lookup')
        print(f"Estimated timestep: {start_timestep} (confidence: {confidence:.3f})")
        
        # Step 3: Progressive refinement via EDM
        return self._progressive_refinement(x_interp, start_timestep)
    
    def _progressive_refinement(self, x_interp, start_timestep):
        """Progressive refinement using EDM denoising steps"""
        x_current = x_interp
        refinement_history = []
        condition_history = []
        timestep_embeddings = []
        frequency_analysis = []
        
        # Store initial state
        refinement_history.append(x_current.cpu().detach().numpy())
        
        # Analyze initial frequency content
        spectrum = torch.fft.rfft2(x_current)
        energy = torch.abs(spectrum)**2
        radial_spectrum = self.timestep_estimator.compute_radial_spectrum(energy)
        frequency_analysis.append({
            'timestep': start_timestep,
            'spectrum': spectrum.cpu().detach().numpy(),
            'radial_spectrum': radial_spectrum.cpu().detach().numpy(),
            'max_freq': radial_spectrum.argmax().item(),
            'total_energy': radial_spectrum.sum().item()
        })
        
        self.edm.unet.eval()
        
        with torch.no_grad():
            for t in range(start_timestep, 0, -1):
                # Current timestep
                t_tensor = torch.tensor([t], device=self.device).long()
                
                # Unsupervised denoising: input is current filtered state
                # UNet predicts less filtered version from more filtered input
                
                # Store current state for visualization
                condition_history.append(x_current.cpu().detach().numpy())
                
                # EDM denoising step (UNet takes only current state, no conditioning)
                x_predicted = self.edm.unet(x_current, t_tensor)
                
                # Store timestep for visualization
                timestep_embeddings.append(t_tensor.cpu().detach().numpy())
                
                # Update current state
                x_current = x_predicted
                
                # Store intermediate result
                refinement_history.append(x_current.cpu().detach().numpy())
                
                # Analyze frequency content after this step
                spectrum = torch.fft.rfft2(x_current)
                energy = torch.abs(spectrum)**2
                radial_spectrum = self.timestep_estimator.compute_radial_spectrum(energy)
                frequency_analysis.append({
                    'timestep': t-1,
                    'spectrum': spectrum.cpu().detach().numpy(),
                    'radial_spectrum': radial_spectrum.cpu().detach().numpy(),
                    'max_freq': radial_spectrum.argmax().item(),
                    'total_energy': radial_spectrum.sum().item()
                })
                
                print(f"  Refinement step {start_timestep - t + 1}/{start_timestep}: t={t} → t={t-1}")
        
        # Package all refinement data
        refinement_data = {
            'history': refinement_history,
            'conditions': condition_history,
            'timestep_embeddings': timestep_embeddings,
            'frequency_analysis': frequency_analysis,
            'start_timestep': start_timestep
        }
        
        return x_current, refinement_data

############################ Comprehensive Visualization Functions ############################

def visualize_complete_refinement_process(low_res, ground_truth, refinement_data, save_path, sample_id=0):
    """
    Comprehensive visualization of the entire Universal Super-Resolution process
    
    Args:
        low_res: Original low-resolution input
        ground_truth: High-resolution ground truth
        refinement_data: Complete refinement process data
        save_path: Directory to save visualizations
        sample_id: Sample identifier
    """
    history = refinement_data['history']
    conditions = refinement_data['conditions']
    frequency_analysis = refinement_data['frequency_analysis']
    start_timestep = refinement_data['start_timestep']
    
    num_steps = len(history)
    channel_idx = 0  # Visualize first channel
    
    # 1. Complete Refinement Process Overview
    fig, axes = plt.subplots(4, num_steps, figsize=(3 * num_steps, 16), squeeze=False, constrained_layout=True)
    fig.suptitle(f'Complete Universal Super-Resolution Process - Sample {sample_id}', fontsize=16, fontweight='bold')
    
    # Determine color scale from ground truth
    gt_data = ground_truth[0, channel_idx].cpu().numpy() if torch.is_tensor(ground_truth) else ground_truth[0, channel_idx]
    vmin, vmax = gt_data.min(), gt_data.max()
    
    for step in range(num_steps):
        timestep = start_timestep - step if step == 0 else start_timestep - step + 1
        
        # Row 1: Current state
        ax = axes[0, step]
        state_data = history[step][0, channel_idx]
        im = ax.imshow(state_data.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        if step == 0:
            ax.set_title(f'Initial\n(t={timestep})', fontsize=10)
        elif step == num_steps - 1:
            ax.set_title(f'Final\n(t={timestep})', fontsize=10)
        else:
            ax.set_title(f't={timestep}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if step == 0:
            ax.set_ylabel('Current State', fontsize=12, fontweight='bold')
        
        # Row 2: Condition (if available)
        ax = axes[1, step]
        if step < len(conditions):
            cond_data = conditions[step][0, channel_idx]
            im = ax.imshow(cond_data.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if step == 0:
            ax.set_ylabel('Condition', fontsize=12, fontweight='bold')
        
        # Row 3: Error from GT
        ax = axes[2, step]
        error = np.abs(state_data - gt_data)
        mae = np.mean(error)
        im_err = ax.imshow(error.T, cmap='Reds', origin='lower', vmin=0, vmax=(vmax-vmin)*0.3)
        ax.set_title(f'MAE: {mae:.4f}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if step == 0:
            ax.set_ylabel('Error from GT', fontsize=12, fontweight='bold')
        
        # Row 4: Frequency content
        ax = axes[3, step]
        if step < len(frequency_analysis):
            freq_data = frequency_analysis[step]
            radial_spectrum = freq_data['radial_spectrum']
            ax.plot(radial_spectrum, linewidth=2)
            ax.set_title(f'Max Freq: {freq_data["max_freq"]}', fontsize=9)
        ax.set_xticks([])
        if step == 0:
            ax.set_ylabel('Frequency\nSpectrum', fontsize=12, fontweight='bold')
    
    # Add ground truth comparison
    plt.figtext(0.02, 0.5, 'Ground Truth', rotation=90, fontsize=14, fontweight='bold', va='center')
    
    plt.savefig(os.path.join(save_path, f'complete_refinement_process_sample{sample_id}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Frequency Evolution Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f'Frequency Analysis During Refinement - Sample {sample_id}', fontsize=16, fontweight='bold')
    
    # Extract frequency data
    timesteps = [freq['timestep'] for freq in frequency_analysis]
    max_freqs = [freq['max_freq'] for freq in frequency_analysis]
    total_energies = [freq['total_energy'] for freq in frequency_analysis]
    
    # Plot 1: Frequency peak evolution
    axes[0, 0].plot(range(len(timesteps)), max_freqs, 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Refinement Step')
    axes[0, 0].set_ylabel('Peak Frequency Index')
    axes[0, 0].set_title('Peak Frequency Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Total energy evolution  
    axes[0, 1].plot(range(len(timesteps)), total_energies, 'o-', color='orange', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Refinement Step')
    axes[0, 1].set_ylabel('Total Frequency Energy')
    axes[0, 1].set_title('Total Energy Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Frequency spectra overlay
    colors = plt.cm.viridis(np.linspace(0, 1, len(frequency_analysis)))
    for i, (freq_data, color) in enumerate(zip(frequency_analysis, colors)):
        radial_spectrum = freq_data['radial_spectrum']
        label = f"t={freq_data['timestep']}" if i % 2 == 0 or i == len(frequency_analysis)-1 else ""
        axes[1, 0].plot(radial_spectrum, color=color, alpha=0.7, linewidth=1.5, label=label)
    axes[1, 0].set_xlabel('Frequency Index')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Frequency Spectra Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Quality metrics evolution
    maes = []
    mses = []
    for step_data in history:
        step_image = step_data[0, channel_idx]
        mae = np.mean(np.abs(step_image - gt_data))
        mse = np.mean((step_image - gt_data)**2)
        maes.append(mae)
        mses.append(mse)
    
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(range(len(maes)), maes, 'o-', color='blue', label='MAE', linewidth=2, markersize=6)
    line2 = ax2.plot(range(len(mses)), mses, 's-', color='red', label='MSE', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Refinement Step')
    ax1.set_ylabel('MAE', color='blue')
    ax2.set_ylabel('MSE', color='red')
    ax1.set_title('Quality Metrics Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.savefig(os.path.join(save_path, f'frequency_analysis_sample{sample_id}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Side-by-side comparison at key steps
    key_steps = [0, len(history)//2, -1]  # Initial, middle, final
    fig, axes = plt.subplots(3, len(key_steps), figsize=(5 * len(key_steps), 15), 
                           squeeze=False, constrained_layout=True)
    fig.suptitle(f'Key Refinement Steps Comparison - Sample {sample_id}', fontsize=16, fontweight='bold')
    
    for i, step_idx in enumerate(key_steps):
        step_data = history[step_idx][0, channel_idx]
        step_name = ['Initial', 'Middle', 'Final'][i]
        actual_timestep = start_timestep - step_idx if step_idx == 0 else start_timestep - step_idx
        
        # Image
        axes[0, i].imshow(step_data.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'{step_name} Step\n(t={actual_timestep})', fontsize=12, fontweight='bold')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Error map
        error = np.abs(step_data - gt_data)
        mae = np.mean(error)
        axes[1, i].imshow(error.T, cmap='Reds', origin='lower', vmin=0, vmax=(vmax-vmin)*0.3)
        axes[1, i].set_title(f'Error (MAE: {mae:.4f})', fontsize=12)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        
        # Frequency spectrum
        if step_idx < len(frequency_analysis):
            freq_data = frequency_analysis[step_idx]
            radial_spectrum = freq_data['radial_spectrum']
            axes[2, i].plot(radial_spectrum, linewidth=2)
            axes[2, i].set_title(f'Frequency Spectrum', fontsize=12)
            axes[2, i].set_xlabel('Frequency Index')
            axes[2, i].set_ylabel('Energy')
            axes[2, i].grid(True, alpha=0.3)
    
    # Row labels
    axes[0, 0].set_ylabel('Refined Image', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Error from GT', fontsize=14, fontweight='bold')
    axes[2, 0].set_ylabel('Frequency Spectrum', fontsize=14, fontweight='bold')
    
    plt.savefig(os.path.join(save_path, f'key_steps_comparison_sample{sample_id}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_refinement_animation(refinement_data, ground_truth, save_path, sample_id=0):
    """Create an animated GIF showing the refinement process"""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        history = refinement_data['history']
        start_timestep = refinement_data['start_timestep']
        channel_idx = 0
        
        # Setup figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        
        gt_data = ground_truth[0, channel_idx].cpu().numpy() if torch.is_tensor(ground_truth) else ground_truth[0, channel_idx]
        vmin, vmax = gt_data.min(), gt_data.max()
        
        # Initialize plots
        im1 = axes[0].imshow(history[0][0, channel_idx].T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title('Current State')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        im2 = axes[1].imshow(gt_data.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title('Ground Truth')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        error_init = np.abs(history[0][0, channel_idx] - gt_data)
        im3 = axes[2].imshow(error_init.T, cmap='Reds', origin='lower', vmin=0, vmax=(vmax-vmin)*0.3)
        axes[2].set_title('Error')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        
        def update(frame):
            step_data = history[frame][0, channel_idx]
            timestep = start_timestep - frame if frame == 0 else start_timestep - frame
            
            im1.set_array(step_data.T)
            axes[0].set_title(f'Current State (t={timestep})')
            
            error = np.abs(step_data - gt_data)
            mae = np.mean(error)
            im3.set_array(error.T)
            axes[2].set_title(f'Error (MAE: {mae:.4f})')
            
            return [im1, im3]
        
        ani = FuncAnimation(fig, update, frames=len(history), interval=800, blit=False, repeat=True)
        ani.save(os.path.join(save_path, f'refinement_animation_sample{sample_id}.gif'), 
                writer=PillowWriter(fps=1.5))
        plt.close()
        
        print(f"Animation saved for sample {sample_id}")
        
    except ImportError:
        print("Animation requires matplotlib with pillow. Skipping animation creation.")

############################ Training Functions ############################

def create_multiscale_batch(high_res_batch, target_size=256):
    """
    Create multi-scale training batch by random downsampling
    
    Args:
        high_res_batch: High-resolution ground truth (B, C, H, W)
        target_size: Target high resolution
        
    Returns:
        low_res_interpolated: Interpolated low-resolution images
        scale_factors: Applied scale factors
        original_low_res: Original low-resolution images
    """
    batch_size = high_res_batch.shape[0]
    
    # Random scale factors for diverse resolution training
    scale_factors = [2, 3, 4, 5, 6, 8, 10, 12, 16]  # Creates resolutions: 64, 42, 32, 25, 21, 16, 12, 10, 8
    
    low_res_interpolated = []
    original_low_res = []
    applied_scales = []
    
    for i in range(batch_size):
        scale = random.choice(scale_factors)
        
        # Downsample to create low-resolution version
        low_size = target_size // scale
        if low_size < 4:  # Minimum size constraint
            low_size = 4
            scale = target_size // low_size
        
        x_high = high_res_batch[i:i+1]
        
        # Create low-resolution version using area interpolation (anti-aliasing)
        x_low = F.interpolate(x_high, size=low_size, mode='area')
        
        # Interpolate back to high resolution
        x_low_interp = grid_aligned_interpolation(x_low, target_size)
        
        low_res_interpolated.append(x_low_interp)
        original_low_res.append(x_low)
        applied_scales.append(scale)
    
    return torch.cat(low_res_interpolated, dim=0), applied_scales, original_low_res

def validate_universal_sr(model, universal_sr, val_loader, device, num_samples=5):
    """Validate universal super-resolution capability"""
    model.unet.eval()
    
    validation_results = []
    test_resolutions = [16, 24, 32, 48, 64]  # Test various resolutions
    
    # Get validation batch
    val_batch = next(iter(val_loader))
    val_targets = val_batch['target'][:num_samples].to(device)
    
    with torch.no_grad():
        for res in test_resolutions:
            print(f"\nTesting resolution: {res}×{res}")
            
            # Create low-resolution test images
            x_low = F.interpolate(val_targets, size=res, mode='area')
            
            # Perform universal super-resolution
            x_restored, _ = universal_sr.super_resolve(x_low, target_size=256)
            
            # Calculate metrics
            mse = F.mse_loss(x_restored, val_targets).item()
            mae = F.l1_loss(x_restored, val_targets).item()
            
            validation_results.append({
                'resolution': res,
                'mse': mse,
                'mae': mae,
                'scale_factor': 256 // res
            })
            
            print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    return validation_results

############################ Hyperparameters ############################

# Create an argument parser
parser = argparse.ArgumentParser(description='Train Universal Super-Resolution EDM model.')
parser.add_argument('--model', type=str, default='edm', choices=['edm'], help='The model to train (only edm supported).')
parser.add_argument('--epochs', type=int, default=10, help='The number of training epochs.')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'load'], help='Set mode to train or load.')
parser.add_argument('--dataset', type=str, default='decaying_snapshot', help='Select the datasets')
parser.add_argument('--num_unets', type=int, default=4, help='The number of U-Net timesteps in the refinement chain.')
parser.add_argument('--max_cutoff', type=float, default=0.6, help='The maximum cutoff frequency')
parser.add_argument('--min_cutoff', type=float, default=0.005, help='The minimum cutoff frequency')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='default learning rate')
parser.add_argument('--train_ratio', type=float, default=0.6, help='train_ratio')
parser.add_argument('--val_ratio', type=float, default=0.2, help='val_ratio')
parser.add_argument('--interpolation_method', type=str, default='fourier', choices=['fourier', 'bilinear', 'rbf'], help='Interpolation method')
parser.add_argument('--rbf_function', type=str, default='gaussian', choices=['gaussian', 'multiquadric', 'thin_plate'], help='RBF function type (only for rbf interpolation)')
parser.add_argument('--rbf_epsilon', type=float, default=1.0, help='RBF shape parameter (only for rbf interpolation)')
parser.add_argument('--target_size', type=int, default=256, help='Target resolution for super-resolution')
parser.add_argument('--multiscale_validation', action='store_true', help='Enable multi-scale validation of Universal SR capability')
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
target_size = args.target_size

save_result_path = f'test/result_{model_name}_{dataset}_univSR'

os.makedirs(save_result_path, exist_ok=True)
with open(os.path.join(save_result_path, 'parameters.txt'), 'w') as f:
    for arg, value in vars(args).items():
        f.write(f'{arg}: {value}\n')

# Remove zero field (max timestep) from training
num_timesteps = num_unets  # No zero field timestep

############################ Model Initialization ############################

print(f'Initializing Universal Super-Resolution {args.model.upper()} model....')
torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Dataset load
#data_dir = f'/mnt/hdd3/home/ddfe/jjw/Dataset/various_CFD/{dataset}/'

data_dir = f'/home/navier/Dataset/various_CFD/{dataset}/'

train_loader, val_loader, test_loader, dataset = create_snapshot_dataloaders(
        data_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

batch_sample = next(iter(train_loader))
inputs_sample = batch_sample['input']
targets_sample = batch_sample['target']

print(f"Target shape: {targets_sample.shape}")
_, C_out, _, _ = targets_sample.shape

# EDM model for Universal Super-Resolution (Unsupervised Learning)
# Input: heavily filtered timestep contour
# Output: less filtered timestep contour
in_channels = C_out  # only target channels (unsupervised)
out_channels = C_out  # same as input
unet = FrequencyProgressiveUNet(
    in_channels=in_channels,
    out_channels=out_channels,
    base_channels=64,
    time_emb_dim=256
).cuda()

model = EDM_AE(unet=unet,
               num_timesteps=num_timesteps,
               lr=learning_rate,
               max_freq_ratio=args.max_cutoff,
               min_freq_ratio=args.min_cutoff).to(device)

# Universal Super-Resolution wrapper
universal_sr = UniversalSuperResolution(model, device)

print(f'Successfully initialized Universal Super-Resolution EDM model!\n')

############################ Monitoring and Training ############################

if args.model == 'edm':
    # Monitor q_sample for different resolutions
    print(f"\nMonitoring Universal SR capabilities...")
    
    # Test with sample from training data
    sample_batch = next(iter(train_loader))
    sample_target = sample_batch['target'][:1].to(device)
    
    # Test different resolutions (reduced for less output)
    test_resolutions = [16, 32, 64]
    
    fig, axes = plt.subplots(len(test_resolutions), 6, 
                           figsize=(24, 4 * len(test_resolutions)),
                           squeeze=False, constrained_layout=True)
    fig.suptitle('Universal Super-Resolution Complete Process Comparison', fontsize=16)
    
    for i, res in enumerate(test_resolutions):
        # Create low-resolution version
        x_low = F.interpolate(sample_target, size=res, mode='area')
        
        # Test all interpolation methods for comparison
        x_interp_fourier = grid_aligned_interpolation(x_low, target_size=target_size, method='fourier')
        x_interp_bilinear = grid_aligned_interpolation(x_low, target_size=target_size, method='bilinear') 
        x_interp_rbf = grid_aligned_interpolation(x_low, target_size=target_size, method='rbf',
                                                rbf_function=args.rbf_function, epsilon=args.rbf_epsilon)
        
        # Use the specified method for main processing
        if args.interpolation_method == 'fourier':
            x_interp = x_interp_fourier
        elif args.interpolation_method == 'bilinear':
            x_interp = x_interp_bilinear
        else:  # rbf
            x_interp = x_interp_rbf
        
        # Estimate timestep (use heuristic before learning)
        estimated_t, confidence = universal_sr.timestep_estimator.estimate_timestep(x_interp, input_resolution=res, method='heuristic')
        
        # Visualize complete process with original low-res
        # Column 0: Original GT
        axes[i, 0].imshow(sample_target[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
        axes[i, 0].set_title(f'Ground Truth\n{target_size}×{target_size}')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # Column 1: Original Low-Res (actual size)
        low_res_display = x_low[0, 0].cpu().numpy()
        axes[i, 1].imshow(low_res_display.T, cmap='RdBu_r', origin='lower', 
                         extent=[0, target_size, 0, target_size], interpolation='nearest')
        axes[i, 1].set_title(f'Original Low-Res\n{res}×{res} (actual size)')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        
        # Column 2: Fourier interpolation
        axes[i, 2].imshow(x_interp_fourier[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
        mae_fourier = F.l1_loss(x_interp_fourier, sample_target).item()
        axes[i, 2].set_title(f'Fourier Interp\nMAE: {mae_fourier:.4f}')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        
        # Column 3: Bilinear interpolation  
        axes[i, 3].imshow(x_interp_bilinear[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
        mae_bilinear = F.l1_loss(x_interp_bilinear, sample_target).item()
        axes[i, 3].set_title(f'Bilinear Interp\nMAE: {mae_bilinear:.4f}')
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])
        
        # Column 4: RBF interpolation
        axes[i, 4].imshow(x_interp_rbf[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
        mae_rbf = F.l1_loss(x_interp_rbf, sample_target).item()
        axes[i, 4].set_title(f'RBF Interp ({args.rbf_function})\nMAE: {mae_rbf:.4f}')
        axes[i, 4].set_xticks([])
        axes[i, 4].set_yticks([])
        
        # Column 5: Final EDM result (using selected method)
        final_result, _ = universal_sr.super_resolve(x_low, target_size=target_size, 
                                                   method=args.interpolation_method,
                                                   rbf_function=args.rbf_function, 
                                                   epsilon=args.rbf_epsilon)
        mae_final = F.l1_loss(final_result, sample_target).item()
        axes[i, 5].imshow(final_result[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
        axes[i, 5].set_title(f'Final EDM Result\nMAE: {mae_final:.4f}')
        axes[i, 5].set_xticks([])
        axes[i, 5].set_yticks([])
    
    plt.savefig(os.path.join(save_result_path, 'universal_sr_monitoring.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Universal SR monitoring plot saved")

# Start Training
if args.mode == 'train':
    print("\nStarting EDM_AE training for Universal Super-Resolution...")
    print("Note: Universal SR capability comes from frequency filtering learned during EDM_AE training")
        
    # Train EDM_AE (unsupervised autoencoder for frequency filtering)
    model.train_autoencoder(train_loader, val_loader, epochs, f'{save_result_path}/best_model.pth', f'{save_result_path}/logs')
    
    print("Training finished.")
else:
    print("\nSkipping training and loading model directly.")

############################ Evaluation ############################

# Load best model
print("\nLoading best model for Universal Super-Resolution evaluation...")
checkpoint_path = f'{save_result_path}/best_model.pth'

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.unet.load_state_dict(checkpoint['model_state_dict'])
    model.unet.eval()
    print("Model loaded successfully")
else:
    print("No checkpoint found, using current model state")

# Learn resolution-to-timestep mapping from training data (only for test resolutions)
universal_sr.timestep_estimator.learn_resolution_mapping(
    train_loader,
    test_resolutions=[8, 16],  # Only learn for test resolutions
    target_size=target_size,
    interpolation_method=args.interpolation_method,
    rbf_function=args.rbf_function,
    epsilon=args.rbf_epsilon
)

# Visualize training quality
print("\nGenerating training quality visualization...")
def visualize_training_quality(model, test_loader, device, save_path):
    """Visualize full sampling process from t=29 to t=0 vs GT"""
    import matplotlib.pyplot as plt

    # Get one test sample
    sample_batch = next(iter(test_loader))
    target = sample_batch['target'][:1].to(device)

    # Test starting from t=29 (most filtered)
    start_timesteps = [29, 20, 10]  # Test 3 different starting points

    fig, axes = plt.subplots(len(start_timesteps), 5, figsize=(20, 5*len(start_timesteps)))

    model.unet.eval()
    with torch.no_grad():
        for row, start_t in enumerate(start_timesteps):
            # Column 0: Starting point (heavily filtered)
            x_start = model.q_sample(target, torch.tensor([start_t], device=device))
            axes[row, 0].imshow(x_start[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            axes[row, 0].set_title(f'Start: x_{{{start_t}}}')
            axes[row, 0].axis('off')

            # Column 1: Full sampling process (t → 0)
            x_current = x_start.clone()
            for t in reversed(range(1, start_t + 1)):
                t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                x_current = model.unet(x_current, t_tensor)

            # Final result after sampling
            axes[row, 1].imshow(x_current[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            mae_sampled = F.l1_loss(x_current, target).item()
            axes[row, 1].set_title(f'Sampled to t=0\nMAE: {mae_sampled:.4f}')
            axes[row, 1].axis('off')

            # Column 2: Ground Truth (t=0)
            axes[row, 2].imshow(target[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            axes[row, 2].set_title('Ground Truth\n(t=0)')
            axes[row, 2].axis('off')

            # Column 3: Error map
            error_map = torch.abs(x_current - target)
            im = axes[row, 3].imshow(error_map[0, 0].cpu().numpy().T, cmap='hot', origin='lower')
            axes[row, 3].set_title(f'Abs Error\nMax: {error_map.max():.4f}')
            axes[row, 3].axis('off')
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046)

            # Column 4: Stats
            mse = F.mse_loss(x_current, target).item()
            psnr = -10 * torch.log10(F.mse_loss(x_current, target)).item()
            text_str = f"Starting t={start_t}\n"
            text_str += f"Steps: {start_t}\n\n"
            text_str += f"MSE: {mse:.6f}\n"
            text_str += f"MAE: {mae_sampled:.6f}\n"
            text_str += f"PSNR: {psnr:.2f} dB\n"
            axes[row, 4].text(0.1, 0.5, text_str, fontsize=11, family='monospace',
                            verticalalignment='center', transform=axes[row, 4].transAxes)
            axes[row, 4].axis('off')
            axes[row, 4].set_title('Metrics')

    plt.suptitle('Training Quality: Full Sampling Process', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training quality visualization saved to {save_path}")

visualize_training_quality(model, test_loader, device, f'{save_result_path}/training_quality.png')

# Visualize all timesteps to understand data distribution
print("\nGenerating timestep data visualization...")
def visualize_all_timesteps(model, test_loader, device, save_path):
    """Visualize what data looks like at each timestep"""
    import matplotlib.pyplot as plt

    # Get one test sample
    sample_batch = next(iter(test_loader))
    target = sample_batch['target'][:1].to(device)

    # Show all timesteps
    num_timesteps = model.num_timesteps
    cols = 10
    rows = (num_timesteps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()

    model.unet.eval()
    with torch.no_grad():
        for t in range(num_timesteps):
            # Get filtered version at timestep t
            x_t = model.q_sample(target, torch.tensor([t], device=device))

            # Calculate high-frequency energy
            spectrum = torch.fft.rfft2(x_t)
            energy = torch.abs(spectrum)**2
            total_energy = energy.sum().item()
            high_freq_ratio = (energy[:, :, energy.shape[-1]//2:].sum() / total_energy).item()

            axes[t].imshow(x_t[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            axes[t].set_title(f't={t}\nHF:{high_freq_ratio:.3f}', fontsize=9)
            axes[t].axis('off')

    # Hide unused subplots
    for i in range(num_timesteps, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Data at Each Timestep (Original → Filtered)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Timestep data visualization saved to {save_path}")

visualize_all_timesteps(model, test_loader, device, f'{save_result_path}/timestep_data_overview.png')

# Visualize interpolation vs actual timestep contour with different estimation methods
print("\nGenerating interpolation comparison visualization...")
def visualize_interpolation_comparison(model, universal_sr, test_loader, device, save_path):
    """Compare low-res, interpolated, and actual timestep contour with different estimation methods"""
    import matplotlib.pyplot as plt

    # Get one test sample
    sample_batch = next(iter(test_loader))
    target = sample_batch['target'][:1].to(device)

    # Test different resolutions
    test_res = [8, 16, 32, 64]

    fig, axes = plt.subplots(len(test_res), 4, figsize=(16, 4*len(test_res)))

    model.unet.eval()
    with torch.no_grad():
        for i, res in enumerate(test_res):
            # Create low-resolution version
            low_res = F.interpolate(target, size=res, mode='area')

            # Interpolate to high resolution
            x_interp = grid_aligned_interpolation(low_res, target_size=target.shape[-1],
                                                 method=args.interpolation_method,
                                                 rbf_function=args.rbf_function,
                                                 epsilon=args.rbf_epsilon)

            # Column 0: Low-resolution (upsampled for visualization)
            low_res_display = F.interpolate(low_res, size=target.shape[-1], mode='nearest')
            axes[i, 0].imshow(low_res_display[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            axes[i, 0].set_title(f'Low-res {res}×{res}')
            axes[i, 0].axis('off')

            # Column 1: Interpolated
            mae_interp = torch.abs(x_interp - target).mean().item()
            axes[i, 1].imshow(x_interp[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            axes[i, 1].set_title(f'Interpolated\nMAE: {mae_interp:.4f}')
            axes[i, 1].axis('off')

            # Column 2: Learned timestep contour (from lookup table)
            if res in universal_sr.timestep_estimator.resolution_timestep_lookup:
                learned_t = universal_sr.timestep_estimator.resolution_timestep_lookup[res]
            else:
                learned_t, _ = universal_sr.timestep_estimator.estimate_timestep(x_interp, input_resolution=res, method='lookup')

            x_timestep = model.q_sample(target, torch.tensor([learned_t], device=device))
            mae_timestep = torch.abs(x_timestep - target).mean().item()
            axes[i, 2].imshow(x_timestep[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            axes[i, 2].set_title(f'Learned timestep\nt={learned_t}\nMAE={mae_timestep:.4f}')
            axes[i, 2].axis('off')

            # Column 3: Ground truth
            axes[i, 3].imshow(target[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            axes[i, 3].set_title('Ground Truth')
            axes[i, 3].axis('off')

    # Add column headers
    axes[0, 0].set_ylabel('8×8', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('16×16', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('32×32', fontsize=12, fontweight='bold')
    axes[3, 0].set_ylabel('64×64', fontsize=12, fontweight='bold')

    plt.suptitle('Learned Resolution-to-Timestep Mapping', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Interpolation comparison visualization saved to {save_path}")

visualize_interpolation_comparison(model, universal_sr, test_loader, device,
                                  f'{save_result_path}/interpolation_vs_timestep.png')

# Visualize SSIM comparison process
print("\nGenerating SSIM comparison process visualization...")
def visualize_ssim_comparison_process(model, universal_sr, test_loader, device, save_path):
    """Visualize how SSIM scores vary across timesteps for different resolutions"""
    import matplotlib.pyplot as plt

    # Get one test sample
    sample_batch = next(iter(test_loader))
    target = sample_batch['target'][:1].to(device)

    # Test different resolutions
    test_res = [8, 16, 32, 64]

    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(len(test_res), 5, hspace=0.3, wspace=0.3)

    model.unet.eval()
    with torch.no_grad():
        for i, res in enumerate(test_res):
            # Create low-resolution version and interpolate
            low_res = F.interpolate(target, size=res, mode='area')
            x_interp = grid_aligned_interpolation(low_res, target_size=target.shape[-1],
                                                 method=args.interpolation_method,
                                                 rbf_function=args.rbf_function,
                                                 epsilon=args.rbf_epsilon)

            # Compute SSIM for all timesteps
            ssim_scores = {}
            for t in range(model.num_timesteps):
                t_tensor = torch.tensor([t], device=device).long()
                x_t = model.q_sample(target, t_tensor)  # Filter GT, not x_interp!
                ssim_val = universal_sr.timestep_estimator._compute_ssim(x_interp, x_t)
                ssim_scores[t] = ssim_val

            best_t = max(ssim_scores.keys(), key=ssim_scores.get)

            # Plot 1: SSIM curve
            ax1 = fig.add_subplot(gs[i, 0])
            timesteps = list(ssim_scores.keys())
            scores = list(ssim_scores.values())
            ax1.plot(timesteps, scores, 'b-', linewidth=2)
            ax1.axvline(best_t, color='r', linestyle='--', linewidth=2, label=f'Best t={best_t}')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('SSIM Score')
            ax1.set_title(f'{res}×{res}: SSIM vs Timestep')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot 2: Low-resolution image
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(low_res[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            ax2.set_title(f'Low-Res\n{res}×{res}')
            ax2.axis('off')

            # Plot 3: Interpolated image
            ax3 = fig.add_subplot(gs[i, 2])
            ax3.imshow(x_interp[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            ax3.set_title(f'Interpolated\n{res}×{res} → {target.shape[-1]}×{target.shape[-1]}')
            ax3.axis('off')

            # Plot 4: Best timestep contour (from GT)
            ax4 = fig.add_subplot(gs[i, 3])
            x_best = model.q_sample(target, torch.tensor([best_t], device=device))
            ax4.imshow(x_best[0, 0].cpu().numpy().T, cmap='RdBu_r', origin='lower')
            ax4.set_title(f'GT filtered at t={best_t}\nSSIM={ssim_scores[best_t]:.4f}')
            ax4.axis('off')

            # Plot 5: Show a few candidate timesteps
            ax5 = fig.add_subplot(gs[i, 4])
            # Get top 5 timesteps
            top_timesteps = sorted(ssim_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            text_str = "Top 5 timesteps:\n"
            for rank, (t, score) in enumerate(top_timesteps, 1):
                text_str += f"{rank}. t={t}: {score:.4f}\n"
            ax5.text(0.1, 0.5, text_str, fontsize=10, family='monospace',
                    verticalalignment='center', transform=ax5.transAxes)
            ax5.axis('off')
            ax5.set_title('Top Candidates')

    plt.suptitle('SSIM-based Timestep Estimation Process', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SSIM comparison process visualization saved to {save_path}")

visualize_ssim_comparison_process(model, universal_sr, test_loader, device,
                                 f'{save_result_path}/ssim_comparison_process.png')

# Universal Super-Resolution Evaluation
print("\nStarting Universal Super-Resolution evaluation...")

num_test_samples = 5  # Reduced sample count
test_resolutions = [8, 16]  # Test resolutions for SR

# Collect test samples (only targets for unsupervised approach)
all_test_targets = []
samples_collected = 0

for batch in test_loader:
    batch_size = batch['target'].shape[0]
    samples_to_take = min(batch_size, num_test_samples - samples_collected)
    
    all_test_targets.append(batch['target'][:samples_to_take].to(device))
    
    samples_collected += samples_to_take
    if samples_collected >= num_test_samples:
        break

test_targets = torch.cat(all_test_targets, dim=0)[:num_test_samples]

print(f"Test target shape: {test_targets.shape}")

# Perform Universal Super-Resolution evaluation
universal_sr_results = {}

for res in test_resolutions:
    print(f"\nTesting Universal SR: {res}×{res} → 256×256")
    
    resolution_results = []
    
    with torch.no_grad():
        for i in range(num_test_samples):
            target = test_targets[i:i+1]
            
            # Create low-resolution version
            low_res = F.interpolate(target, size=res, mode='area')
            
            # Perform universal super-resolution
            restored, refinement_data = universal_sr.super_resolve(
                low_res, target_size=target_size, method=args.interpolation_method,
                rbf_function=args.rbf_function, epsilon=args.rbf_epsilon
            )
            
            # Calculate metrics
            mse = F.mse_loss(restored, target).item()
            mae = F.l1_loss(restored, target).item()
            psnr = 20 * torch.log10(1.0 / torch.sqrt(F.mse_loss(restored, target))).item()
            
            resolution_results.append({
                'sample_idx': i,
                'mse': mse,
                'mae': mae,
                'psnr': psnr,
                'restored': restored.cpu().numpy(),
                'low_res': low_res.cpu().numpy(),
                'target': target.cpu().numpy(),
                'refinement_data': refinement_data
            })
    
    universal_sr_results[res] = resolution_results
    
    # Print summary for this resolution
    avg_mse = np.mean([r['mse'] for r in resolution_results])
    avg_mae = np.mean([r['mae'] for r in resolution_results])
    avg_psnr = np.mean([r['psnr'] for r in resolution_results])
    
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Average MAE: {avg_mae:.6f}")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")

# Save results
np.save(os.path.join(save_result_path, 'universal_sr_results.npy'), universal_sr_results)

# Generate comprehensive step-by-step visualizations
print("\nGenerating comprehensive step-by-step visualizations...")

# Create detailed visualizations for selected samples and resolutions
visualization_resolutions = test_resolutions[-2:] if len(test_resolutions) >= 2 else test_resolutions  # Last 2 resolutions
visualization_samples = [0, 1]  # 2개 샘플만 (총 4개 세트)

for res in visualization_resolutions:
    print(f"\nGenerating detailed visualizations for {res}×{res} resolution...")
    
    for sample_idx in visualization_samples:
        if sample_idx < len(universal_sr_results[res]):
            result = universal_sr_results[res][sample_idx]
            
            # Create comprehensive visualization
            print(f"  Creating visualization for sample {sample_idx}...")
            visualize_complete_refinement_process(
                low_res=result['low_res'],
                ground_truth=result['target'],
                refinement_data=result['refinement_data'],
                save_path=save_result_path,
                sample_id=f"{sample_idx}_{res}x{res}"
            )
            
            # Animation disabled to reduce file count
            # create_refinement_animation(
            #     refinement_data=result['refinement_data'],
            #     ground_truth=result['target'],
            #     save_path=save_result_path,
            #     sample_id=f"{sample_idx}_{res}x{res}"
            # )

print("Detailed visualizations completed!")

# Create comprehensive visualization
print("\nGenerating Universal Super-Resolution summary visualization...")

# Summary plot: performance vs resolution
fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
fig.suptitle('Universal Super-Resolution Performance Analysis', fontsize=16, fontweight='bold')

resolutions = sorted(universal_sr_results.keys())
avg_metrics = {
    'mse': [np.mean([r['mse'] for r in universal_sr_results[res]]) for res in resolutions],
    'mae': [np.mean([r['mae'] for r in universal_sr_results[res]]) for res in resolutions],
    'psnr': [np.mean([r['psnr'] for r in universal_sr_results[res]]) for res in resolutions]
}

# Plot 1: MSE vs Resolution
axes[0, 0].plot(resolutions, avg_metrics['mse'], 'o-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Input Resolution')
axes[0, 0].set_ylabel('Mean Squared Error')
axes[0, 0].set_title('MSE vs Input Resolution')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Plot 2: MAE vs Resolution
axes[0, 1].plot(resolutions, avg_metrics['mae'], 'o-', color='orange', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Input Resolution')
axes[0, 1].set_ylabel('Mean Absolute Error')
axes[0, 1].set_title('MAE vs Input Resolution')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: PSNR vs Resolution
axes[1, 0].plot(resolutions, avg_metrics['psnr'], 'o-', color='green', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Input Resolution')
axes[1, 0].set_ylabel('PSNR (dB)')
axes[1, 0].set_title('PSNR vs Input Resolution')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Scale factor vs Performance
scale_factors = [256 // res for res in resolutions]
axes[1, 1].plot(scale_factors, avg_metrics['psnr'], 'o-', color='red', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Super-Resolution Scale Factor')
axes[1, 1].set_ylabel('PSNR (dB)')
axes[1, 1].set_title('Performance vs Scale Factor')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xscale('log', base=2)

plt.savefig(os.path.join(save_result_path, 'universal_sr_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# Sample results visualization
print("Generating sample results visualization...")

selected_resolutions = test_resolutions  # Use actual test resolutions
num_samples_to_show = 3

fig, axes = plt.subplots(num_samples_to_show, len(selected_resolutions) * 4, 
                        figsize=(4 * len(selected_resolutions) * 4, 4 * num_samples_to_show),
                        squeeze=False, constrained_layout=True)

fig.suptitle('Universal Super-Resolution Sample Results', fontsize=18, fontweight='bold')

for sample_idx in range(num_samples_to_show):
    for res_idx, res in enumerate(selected_resolutions):
        results = universal_sr_results[res][sample_idx]
        
        low_res = results['low_res'][0, 0]  # First channel
        restored = results['restored'][0, 0]
        target = results['target'][0, 0]
        
        # Calculate display metrics
        mse = results['mse']
        mae = results['mae']
        psnr = results['psnr']
        
        col_base = res_idx * 4
        
        # Low-resolution input
        ax = axes[sample_idx, col_base]
        im = ax.imshow(low_res.T, cmap='RdBu_r', origin='lower')
        ax.set_title(f'{res}×{res}\nInput', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Restored high-resolution
        ax = axes[sample_idx, col_base + 1]
        im = ax.imshow(restored.T, cmap='RdBu_r', origin='lower')
        ax.set_title(f'Restored\n256×256', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Ground truth
        ax = axes[sample_idx, col_base + 2]
        im = ax.imshow(target.T, cmap='RdBu_r', origin='lower')
        ax.set_title(f'Ground Truth\n256×256', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Error map
        ax = axes[sample_idx, col_base + 3]
        error = np.abs(restored - target)
        im = ax.imshow(error.T, cmap='Reds', origin='lower')
        ax.set_title(f'Error\nPSNR: {psnr:.1f}dB', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if sample_idx == 0:
            axes[sample_idx, col_base + 1].text(0.5, 1.15, f'Scale: {256//res}×', 
                                              transform=axes[sample_idx, col_base + 1].transAxes,
                                              ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Row labels
    axes[sample_idx, 0].set_ylabel(f'Sample {sample_idx + 1}', fontsize=12, fontweight='bold')

plt.savefig(os.path.join(save_result_path, 'universal_sr_samples.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*80}")
print(f"Universal Super-Resolution Evaluation Complete!")
print(f"{'='*80}")
print(f"Model: Universal Super-Resolution EDM")
print(f"Evaluated {num_test_samples} test samples")
print(f"Input/Output channels: {in_channels}/{out_channels}")
print(f"Interpolation method: {args.interpolation_method}")

print(f"\nPerformance Summary:")
for res in sorted(universal_sr_results.keys()):
    results = universal_sr_results[res]
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    scale = target_size // res
    print(f"  {res}×{res} → {target_size}×{target_size} (Scale: {scale}×): PSNR = {avg_psnr:.2f}dB, MAE = {avg_mae:.6f}")

print(f"\nResults saved in: {save_result_path}")
print(f"\n📊 Performance Analysis:")
print(f"  - universal_sr_analysis.png (performance vs resolution analysis)")
print(f"  - universal_sr_samples.png (sample results comparison)")
print(f"\n🔍 Step-by-Step Visualizations:")
for res in visualization_resolutions:
    for sample_idx in visualization_samples:
        if sample_idx < len(universal_sr_results[res]):
            print(f"  - complete_refinement_process_sample{sample_idx}_{res}x{res}.png")
            print(f"  - frequency_analysis_sample{sample_idx}_{res}x{res}.png") 
            print(f"  - key_steps_comparison_sample{sample_idx}_{res}x{res}.png")
            
print(f"\n📁 Data Files:")
print(f"  - universal_sr_results.npy (complete results with refinement data)")
print(f"  - universal_sr_monitoring.png (initial capability test)")

print(f"\n🎯 Key Features Demonstrated:")
print(f"  ✓ Automatic timestep estimation based on frequency analysis")
print(f"  ✓ Progressive refinement through EDM denoising steps")
print(f"  ✓ Frequency-based interpolation for alias-free upsampling")
print(f"  ✓ Multi-scale training for improved generalization")
print(f"  ✓ Complete visualization of the refinement process")
print(f"  ✓ Quantitative analysis at each refinement step")
print(f"{'='*80}")
