"""
Hallucination Detection for Turbulent Flow Fields
==================================================

This module provides physics-informed hallucination detection for generated
turbulent flow fields. It evaluates whether generated samples exhibit
non-physical or out-of-distribution patterns.

Main Components:
1. Energy Spectrum Anomaly Detection (Kolmogorov -5/3 law)
2. Structure Function Analysis
3. Nearest Neighbor Distance (OOD detection)
4. Maximum Mean Discrepancy (MMD)
5. Unified Hallucination Score

Author: Claude Code
Date: 2025-11-14
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class TurbulenceHallucinationDetector:
    """
    Physics-informed hallucination detector for turbulent flow fields.

    This detector uses multiple metrics to identify non-physical patterns
    and out-of-distribution samples in generated turbulence fields.
    """

    def __init__(self, reference_samples, sample_shape=(128, 128)):
        """
        Initialize the detector with reference (training) data.

        Args:
            reference_samples: np.ndarray of shape (N, C, H, W) - real training data
            sample_shape: tuple - expected shape of individual samples (H, W)
        """
        self.reference_samples = reference_samples
        self.sample_shape = sample_shape
        self.reference_statistics = self._compute_reference_statistics()

    def _compute_reference_statistics(self):
        """Compute statistics from reference samples."""
        stats_dict = {}

        # Energy spectrum statistics
        spectra = []
        for sample in self.reference_samples:
            E_k = self._compute_energy_spectrum(sample[0])  # First channel
            spectra.append(E_k)

        spectra = np.array(spectra)
        stats_dict['energy_spectrum_mean'] = np.mean(spectra, axis=0)
        stats_dict['energy_spectrum_std'] = np.std(spectra, axis=0)

        # High-frequency energy ratio
        high_freq_ratios = []
        for E_k in spectra:
            ratio = np.sum(E_k[len(E_k)//2:]) / np.sum(E_k[:len(E_k)//2])
            high_freq_ratios.append(ratio)
        stats_dict['high_freq_ratio_mean'] = np.mean(high_freq_ratios)
        stats_dict['high_freq_ratio_std'] = np.std(high_freq_ratios)

        # Statistical moments
        stats_dict['value_mean'] = np.mean(self.reference_samples)
        stats_dict['value_std'] = np.std(self.reference_samples)
        stats_dict['value_skewness'] = stats.skew(self.reference_samples.flatten())
        stats_dict['value_kurtosis'] = stats.kurtosis(self.reference_samples.flatten())

        return stats_dict

    def _compute_energy_spectrum(self, field):
        """
        Compute radially-averaged energy spectrum.

        Args:
            field: 2D array (H, W)

        Returns:
            E_k: 1D array - energy spectrum E(k)
        """
        # 2D FFT
        field_hat = np.fft.fft2(field)

        # Energy (power spectral density)
        E_2d = np.abs(field_hat)**2

        # Radial averaging
        H, W = field.shape
        ky, kx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2)

        # Bin by wavenumber magnitude
        k_max = int(min(H, W) / 2)
        k_bins = np.arange(0, k_max)
        E_k = np.zeros(k_max)

        for i, k in enumerate(k_bins):
            mask = (k_mag >= k) & (k_mag < k + 1)
            if np.sum(mask) > 0:
                E_k[i] = np.mean(E_2d[mask])

        return E_k

    def _compute_structure_function(self, field, order=2, max_r=50):
        """
        Compute structure function S_p(r).

        S_p(r) = <|u(x+r) - u(x)|^p>

        Args:
            field: 2D array (H, W)
            order: int - order of structure function (typically 2 or 3)
            max_r: int - maximum separation distance

        Returns:
            S_p: 1D array - structure function values
        """
        H, W = field.shape
        S_p = np.zeros(max_r)
        counts = np.zeros(max_r)

        # Sample random points for efficiency
        n_samples = min(1000, H * W // 10)
        y_samples = np.random.randint(0, H - max_r, n_samples)
        x_samples = np.random.randint(0, W - max_r, n_samples)

        for r in range(1, max_r):
            increments = []
            for y, x in zip(y_samples, x_samples):
                # Horizontal increments
                if x + r < W:
                    inc = field[y, x + r] - field[y, x]
                    increments.append(inc)
                # Vertical increments
                if y + r < H:
                    inc = field[y + r, x] - field[y, x]
                    increments.append(inc)

            if len(increments) > 0:
                S_p[r] = np.mean(np.abs(increments)**order)
                counts[r] = len(increments)

        return S_p

    def spectral_hallucination_score(self, generated_sample):
        """
        Detect spectral anomalies using Kolmogorov theory.

        Expected: E(k) ∝ k^(-5/3) in inertial range

        Args:
            generated_sample: np.ndarray of shape (C, H, W)

        Returns:
            dict with scores and components
        """
        E_k = self._compute_energy_spectrum(generated_sample[0])

        # Define inertial range (typically k in [10, N/4])
        k_min = max(5, len(E_k) // 20)
        k_max = min(len(E_k) // 4, len(E_k) - 1)

        if k_max <= k_min:
            return {'total_score': 0.0, 'components': {}}

        k_inertial = np.arange(k_min, k_max)
        E_inertial = E_k[k_min:k_max]

        # Remove zeros/invalid values
        valid_mask = (E_inertial > 0) & np.isfinite(E_inertial)
        if np.sum(valid_mask) < 5:
            return {'total_score': 0.5, 'components': {'insufficient_data': True}}

        k_valid = k_inertial[valid_mask]
        E_valid = E_inertial[valid_mask]

        # Check 1: Monotonicity (energy should generally decrease)
        non_monotonic_ratio = np.sum(np.diff(E_valid) > 0) / len(E_valid)

        # Check 2: Scaling exponent (should be close to -5/3)
        log_k = np.log(k_valid)
        log_E = np.log(E_valid)

        if len(log_k) > 2:
            slope, intercept = np.polyfit(log_k, log_E, 1)
            scaling_deviation = abs(slope + 5/3) / (5/3)
        else:
            scaling_deviation = 0.0

        # Check 3: High-frequency energy excess
        high_freq_ratio = np.sum(E_k[len(E_k)//2:]) / (np.sum(E_k[:len(E_k)//2]) + 1e-10)
        ref_ratio = self.reference_statistics['high_freq_ratio_mean']
        ref_std = self.reference_statistics['high_freq_ratio_std']
        high_freq_anomaly = max(0, (high_freq_ratio - ref_ratio - 2*ref_std) / (ref_std + 1e-10))

        # Check 4: Overall spectrum deviation from reference
        E_k_ref_mean = self.reference_statistics['energy_spectrum_mean']
        E_k_ref_std = self.reference_statistics['energy_spectrum_std']

        # Compare in log space (more meaningful for power spectra)
        min_len = min(len(E_k), len(E_k_ref_mean))
        E_k_trunc = E_k[:min_len]
        E_ref_trunc = E_k_ref_mean[:min_len]
        E_ref_std_trunc = E_k_ref_std[:min_len]

        valid_ref = (E_ref_trunc > 0) & (E_k_trunc > 0)
        if np.sum(valid_ref) > 0:
            log_E_gen = np.log(E_k_trunc[valid_ref])
            log_E_ref = np.log(E_ref_trunc[valid_ref])
            log_E_ref_std = E_ref_std_trunc[valid_ref] / (E_ref_trunc[valid_ref] + 1e-10)

            spectrum_deviation = np.mean(np.abs(log_E_gen - log_E_ref) / (log_E_ref_std + 1e-10))
        else:
            spectrum_deviation = 0.0

        # Combined score (weighted)
        total_score = (
            0.2 * non_monotonic_ratio +
            0.3 * min(1.0, scaling_deviation) +
            0.2 * min(1.0, high_freq_anomaly) +
            0.3 * min(1.0, spectrum_deviation / 3.0)  # Normalize by 3-sigma
        )

        return {
            'total_score': float(total_score),
            'components': {
                'non_monotonic_ratio': float(non_monotonic_ratio),
                'scaling_deviation': float(scaling_deviation),
                'high_freq_anomaly': float(high_freq_anomaly),
                'spectrum_deviation': float(spectrum_deviation)
            }
        }

    def structure_function_hallucination_score(self, generated_sample):
        """
        Detect anomalies in structure function scaling.

        Expected: S_2(r) ∝ r^(2/3), S_3(r) ∝ r

        Args:
            generated_sample: np.ndarray of shape (C, H, W)

        Returns:
            float - anomaly score
        """
        max_r = min(50, min(generated_sample.shape[-2:]) // 4)

        S2 = self._compute_structure_function(generated_sample[0], order=2, max_r=max_r)

        # Focus on scales where we have good statistics
        r_min = 2
        r_max = max_r // 2
        r_range = np.arange(r_min, r_max)
        S2_range = S2[r_min:r_max]

        # Remove invalid values
        valid = (S2_range > 0) & np.isfinite(S2_range)
        if np.sum(valid) < 5:
            return 0.5

        r_valid = r_range[valid]
        S2_valid = S2_range[valid]

        # Expected: S_2(r) ∝ r^(2/3)
        log_r = np.log(r_valid)
        log_S2 = np.log(S2_valid)

        slope, _ = np.polyfit(log_r, log_S2, 1)
        expected_slope = 2.0 / 3.0

        S2_anomaly = abs(slope - expected_slope) / expected_slope

        return float(min(1.0, S2_anomaly))

    def nearest_neighbor_hallucination_score(self, generated_sample):
        """
        Compute distance to nearest neighbor in reference data.

        High distance → likely OOD/hallucination

        Args:
            generated_sample: np.ndarray of shape (C, H, W)

        Returns:
            float - normalized distance score
        """
        # Extract features
        gen_features = self._extract_features(generated_sample)

        # Compute features for reference samples (cache if expensive)
        if not hasattr(self, '_reference_features'):
            self._reference_features = np.array([
                self._extract_features(sample)
                for sample in self.reference_samples
            ])

        # Compute distances
        distances = cdist([gen_features], self._reference_features, metric='euclidean')[0]
        min_distance = np.min(distances)

        # Normalize by local density (k-nearest neighbors)
        k = min(10, len(distances))
        local_distances = np.sort(distances)[:k]
        sigma_local = np.mean(local_distances)

        normalized_score = min_distance / (sigma_local + 1e-10)

        return float(min(1.0, normalized_score / 3.0))  # Normalize to [0, 1]

    def _extract_features(self, sample):
        """
        Extract physics-informed features from a sample.

        Args:
            sample: np.ndarray of shape (C, H, W)

        Returns:
            features: 1D array
        """
        features = []
        field = sample[0]  # First channel

        # 1. Energy spectrum (log-binned)
        E_k = self._compute_energy_spectrum(field)
        k_bins = np.logspace(0, np.log10(len(E_k)), 15)
        E_binned, _ = np.histogram(np.arange(len(E_k)), bins=k_bins, weights=E_k)
        features.extend(np.log(E_binned + 1e-10))

        # 2. Statistical moments
        features.extend([
            np.mean(field),
            np.std(field),
            stats.skew(field.flatten()),
            stats.kurtosis(field.flatten())
        ])

        # 3. Gradient statistics
        grad_y, grad_x = np.gradient(field)
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)
        features.extend([
            np.mean(grad_mag),
            np.std(grad_mag),
            np.max(grad_mag)
        ])

        # 4. Multi-scale statistics (downsampled)
        for scale in [2, 4, 8]:
            if min(field.shape) > scale * 4:
                field_down = field[::scale, ::scale]
                features.extend([
                    np.mean(field_down),
                    np.std(field_down)
                ])

        return np.array(features)

    def statistical_divergence_score(self, generated_sample):
        """
        Compute statistical divergence from reference distribution.

        Args:
            generated_sample: np.ndarray of shape (C, H, W)

        Returns:
            float - divergence score
        """
        field = generated_sample[0]

        # Compute moments
        gen_mean = np.mean(field)
        gen_std = np.std(field)
        gen_skew = stats.skew(field.flatten())
        gen_kurt = stats.kurtosis(field.flatten())

        ref_mean = self.reference_statistics['value_mean']
        ref_std = self.reference_statistics['value_std']
        ref_skew = self.reference_statistics['value_skewness']
        ref_kurt = self.reference_statistics['value_kurtosis']

        # Normalized deviations
        mean_dev = abs(gen_mean - ref_mean) / (ref_std + 1e-10)
        std_dev = abs(gen_std - ref_std) / (ref_std + 1e-10)
        skew_dev = abs(gen_skew - ref_skew) / (abs(ref_skew) + 1.0)
        kurt_dev = abs(gen_kurt - ref_kurt) / (abs(ref_kurt) + 3.0)

        # Weighted combination
        divergence = (
            0.3 * mean_dev +
            0.3 * std_dev +
            0.2 * skew_dev +
            0.2 * kurt_dev
        )

        return float(min(1.0, divergence))

    def unified_hallucination_score(self, generated_sample, weights=None):
        """
        Compute unified hallucination score from all metrics.

        Args:
            generated_sample: np.ndarray of shape (C, H, W)
            weights: dict - optional custom weights for each component

        Returns:
            dict with overall score, classification, and component scores
        """
        if weights is None:
            weights = {
                'spectral': 0.35,
                'structure': 0.25,
                'nearest_neighbor': 0.20,
                'statistical': 0.20
            }

        # Compute all component scores
        spectral_result = self.spectral_hallucination_score(generated_sample)
        spectral_score = spectral_result['total_score']

        structure_score = self.structure_function_hallucination_score(generated_sample)
        nn_score = self.nearest_neighbor_hallucination_score(generated_sample)
        stat_score = self.statistical_divergence_score(generated_sample)

        # Weighted combination
        total_score = (
            weights['spectral'] * spectral_score +
            weights['structure'] * structure_score +
            weights['nearest_neighbor'] * nn_score +
            weights['statistical'] * stat_score
        )

        # Classification (threshold can be tuned)
        is_hallucination = total_score > 0.4

        # Confidence (how far from threshold)
        confidence = abs(total_score - 0.4) / 0.4

        return {
            'is_hallucination': bool(is_hallucination),
            'total_score': float(total_score),
            'confidence': float(min(1.0, confidence)),
            'component_scores': {
                'spectral': float(spectral_score),
                'structure': float(structure_score),
                'nearest_neighbor': float(nn_score),
                'statistical': float(stat_score)
            },
            'spectral_details': spectral_result['components']
        }

    def batch_evaluate(self, generated_samples):
        """
        Evaluate hallucination for a batch of samples.

        Args:
            generated_samples: np.ndarray of shape (N, C, H, W)

        Returns:
            list of result dicts
        """
        results = []
        for sample in generated_samples:
            result = self.unified_hallucination_score(sample)
            results.append(result)
        return results


def aggregate_hallucination_results(results):
    """
    Aggregate hallucination detection results across multiple samples.

    Args:
        results: list of result dicts from unified_hallucination_score

    Returns:
        dict with aggregate statistics
    """
    n_samples = len(results)

    hallucination_rate = np.mean([r['is_hallucination'] for r in results])

    avg_scores = {
        'total': np.mean([r['total_score'] for r in results]),
        'spectral': np.mean([r['component_scores']['spectral'] for r in results]),
        'structure': np.mean([r['component_scores']['structure'] for r in results]),
        'nearest_neighbor': np.mean([r['component_scores']['nearest_neighbor'] for r in results]),
        'statistical': np.mean([r['component_scores']['statistical'] for r in results])
    }

    std_scores = {
        'total': np.std([r['total_score'] for r in results]),
        'spectral': np.std([r['component_scores']['spectral'] for r in results]),
        'structure': np.std([r['component_scores']['structure'] for r in results]),
        'nearest_neighbor': np.std([r['component_scores']['nearest_neighbor'] for r in results]),
        'statistical': np.std([r['component_scores']['statistical'] for r in results])
    }

    return {
        'n_samples': n_samples,
        'hallucination_rate': hallucination_rate,
        'avg_scores': avg_scores,
        'std_scores': std_scores,
        'avg_confidence': np.mean([r['confidence'] for r in results])
    }
