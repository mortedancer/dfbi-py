#!/usr/bin/env python3
"""
Advanced Wavelet Analysis Example
================================

This example demonstrates advanced wavelet kernel usage in DFBI,
including complex wavelets, kernel banks, and phase analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dfbi import fingerprint
from dfbi.alphabet import EN34
from dfbi.kernels import kernel_vector
from dfbi.metrics import dist_l2_multi, dist_cosine
import yaml

def load_config():
    """Load configuration for wavelet analysis."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert decay lists to tuples for compatibility
    def convert_config_section(section):
        if isinstance(section, dict):
            for key, value in section.items():
                if key == 'decay' and isinstance(value, list):
                    section[key] = tuple(value)
                elif key == 'phase' and isinstance(value, list):
                    section[key] = tuple(value)
                elif isinstance(value, dict):
                    convert_config_section(value)
    
    for section in config.values():
        convert_config_section(section)
    
    return config

def visualize_kernels():
    """Visualize different wavelet kernels."""
    print("=== Wavelet Kernel Visualization ===")
    
    horizon = 20
    
    # Define different kernels
    kernels = [
        ('Exponential', 'exp', [0.7]),
        ('Gaussian', 'gauss', [5.0, 2.0]),
        ('Morlet', 'morlet', [3.0, 2.0, 1.0, 0.0]),
        ('Mexican Hat', 'mexican', [2.0, 1.0, 0.0]),
        ('Gabor', 'gabor', [2.0, 2.0])
    ]
    
    print("Kernel properties:")
    for name, kind, params in kernels:
        kernel = kernel_vector(kind, params, horizon)
        
        # Calculate properties
        if np.iscomplexobj(kernel):
            magnitude = np.abs(kernel)
            phase = np.angle(kernel)
            energy = np.sum(magnitude**2)
            print(f"{name:12s}: Complex, Energy={energy:.3f}, Max Phase={np.max(phase):.3f}")
        else:
            energy = np.sum(kernel**2)
            peak_pos = np.argmax(np.abs(kernel)) + 1
            print(f"{name:12s}: Real, Energy={energy:.3f}, Peak at d={peak_pos}")
    
    # Save kernel data for plotting (if matplotlib available)
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, kind, params) in enumerate(kernels):
            if i >= len(axes):
                break
                
            kernel = kernel_vector(kind, params, horizon)
            x = np.arange(1, horizon + 1)
            
            if np.iscomplexobj(kernel):
                axes[i].plot(x, np.real(kernel), 'b-', label='Real', linewidth=2)
                axes[i].plot(x, np.imag(kernel), 'r--', label='Imaginary', linewidth=2)
                axes[i].plot(x, np.abs(kernel), 'g:', label='Magnitude', linewidth=2)
                axes[i].legend()
            else:
                axes[i].plot(x, kernel, 'b-', linewidth=2)
            
            axes[i].set_title(f'{name} Kernel')
            axes[i].set_xlabel('Distance (d)')
            axes[i].set_ylabel('Weight')
            axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(kernels) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / 'kernel_visualization.png', dpi=150, bbox_inches='tight')
        print("Kernel visualization saved to 'kernel_visualization.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")

def morlet_parameter_analysis():
    """Analyze how Morlet wavelet parameters affect text analysis."""
    print("\n=== Morlet Parameter Analysis ===")
    
    config = load_config()
    
    # Sample text
    text = "The quick brown fox jumps over the lazy dog. " * 10
    
    # Different Morlet parameters
    omega_values = [1.0, 3.0, 5.0, 8.0]
    sigma_values = [0.5, 1.0, 2.0, 3.0]
    
    print("Morlet parameter effects:")
    print("Omega\tSigma\tComplexity\tSparsity")
    print("-" * 40)
    
    for omega in omega_values:
        for sigma in sigma_values:
            fp = fingerprint(
                text,
                alphabet=EN34,
                horizon=10,
                decay=('morlet', {'omega': omega, 'sigma': sigma}),
                normalize='global'
            )
            
            # Calculate metrics
            complexity = np.std(np.abs(fp))
            sparsity = 1 - np.count_nonzero(fp) / fp.size
            
            print(f"{omega:.1f}\t{sigma:.1f}\t{complexity:.4f}\t\t{sparsity:.3f}")

def kernel_bank_analysis():
    """Demonstrate kernel bank functionality."""
    print("\n=== Kernel Bank Analysis ===")
    
    config = load_config()
    
    text = "Advanced wavelet analysis using multiple kernel functions simultaneously."
    
    # Single kernels
    single_kernels = [
        "morlet:omega=3.0;sigma=1.0",
        "mexican:sigma=1.5",
        "gauss:mu=3.0;sigma=2.0"
    ]
    
    print("Single kernel analysis:")
    single_fps = []
    for kernel_spec in single_kernels:
        fp = fingerprint(
            text,
            alphabet=EN34,
            horizon=8,
            bank=kernel_spec,
            normalize='global'
        )
        single_fps.append(fp)
        print(f"Kernel {kernel_spec:25s}: Shape {fp.shape}, Sum {np.sum(np.abs(fp)):.4f}")
    
    # Combined kernel bank
    print("\nKernel bank analysis:")
    bank_spec = "morlet:omega=3.0;sigma=1.0 + mexican:sigma=1.5 + gauss:mu=3.0;sigma=2.0"
    
    # Without aggregation (returns 3D array)
    fp_bank = fingerprint(
        text,
        alphabet=EN34,
        horizon=8,
        bank=bank_spec,
        normalize='global'
    )
    print(f"Bank without aggregation: Shape {fp_bank.shape}")
    
    # With sum_abs aggregation
    fp_aggregated = fingerprint(
        text,
        alphabet=EN34,
        horizon=8,
        bank=bank_spec,
        aggregate='sum_abs',
        normalize='global'
    )
    print(f"Bank with sum_abs: Shape {fp_aggregated.shape}, Sum {np.sum(fp_aggregated):.4f}")

def phase_modulation_demo():
    """Demonstrate phase modulation effects."""
    print("\n=== Phase Modulation Demo ===")
    
    text = "Phase modulation adds another dimension to wavelet analysis."
    
    # Different phase configurations
    phase_configs = [
        None,  # No phase
        ('theta', 0.5),  # Constant phase
        ('entropy', 0.1, 3.1415),  # Entropy-driven phase
    ]
    
    print("Phase modulation effects:")
    for i, phase_config in enumerate(phase_configs):
        fp = fingerprint(
            text,
            alphabet=EN34,
            horizon=6,
            decay=('morlet', {'omega': 4.0, 'sigma': 1.5}),
            phase=phase_config,
            normalize='global'
        )
        
        phase_name = "None" if phase_config is None else str(phase_config[0])
        is_complex = np.iscomplexobj(fp)
        
        if is_complex:
            magnitude = np.abs(fp)
            phase_var = np.var(np.angle(fp[fp != 0]))
            print(f"Phase {phase_name:8s}: Complex, Magnitude sum={np.sum(magnitude):.4f}, Phase var={phase_var:.4f}")
        else:
            print(f"Phase {phase_name:8s}: Real, Sum={np.sum(fp):.4f}")

def complex_metric_comparison():
    """Compare different metrics for complex fingerprints."""
    print("\n=== Complex Metric Comparison ===")
    
    # Create two similar texts
    text1 = "Complex wavelet analysis provides rich spectral information."
    text2 = "Complex wavelet analysis offers detailed spectral data."
    
    # Create complex fingerprints
    fp1 = fingerprint(
        text1,
        alphabet=EN34,
        horizon=5,
        decay=('morlet', {'omega': 3.0, 'sigma': 1.0}),
        normalize='global'
    )
    
    fp2 = fingerprint(
        text2,
        alphabet=EN34,
        horizon=5,
        decay=('morlet', {'omega': 3.0, 'sigma': 1.0}),
        normalize='global'
    )
    
    print("Comparing complex fingerprints:")
    print(f"Fingerprint 1: Complex={np.iscomplexobj(fp1)}, Shape={fp1.shape}")
    print(f"Fingerprint 2: Complex={np.iscomplexobj(fp2)}, Shape={fp2.shape}")
    
    # Compare using different metrics
    if np.iscomplexobj(fp1) and np.iscomplexobj(fp2):
        # Multi-channel L2 (complex-aware)
        dist_multi = dist_l2_multi(fp1, fp2)
        print(f"L2 Multi-channel distance: {dist_multi:.6f}")
        
        # Cosine distance (magnitude-based)
        dist_cosine_val = dist_cosine(fp1, fp2)
        print(f"Cosine distance: {dist_cosine_val:.6f}")
        
        # Compare magnitudes only
        mag1, mag2 = np.abs(fp1), np.abs(fp2)
        dist_magnitude = np.linalg.norm(mag1 - mag2)
        print(f"Magnitude-only L2: {dist_magnitude:.6f}")
        
        # Phase difference analysis
        phase1, phase2 = np.angle(fp1), np.angle(fp2)
        phase_diff = np.mean(np.abs(phase1 - phase2))
        print(f"Average phase difference: {phase_diff:.6f}")

def wavelet_sensitivity_analysis():
    """Analyze sensitivity of wavelets to text modifications."""
    print("\n=== Wavelet Sensitivity Analysis ===")
    
    base_text = "The quick brown fox jumps over the lazy dog."
    
    # Text modifications
    modifications = [
        ("Original", base_text),
        ("Punctuation", "The quick brown fox jumps over the lazy dog"),  # Remove period
        ("Case change", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG."),
        ("Word order", "The brown quick fox jumps over the dog lazy."),
        ("Synonym", "The fast brown fox leaps over the lazy dog."),
    ]
    
    # Different wavelets
    wavelets = [
        ('Exponential', ('exp', 0.7)),
        ('Gaussian', ('gauss', {'mu': 2.0, 'sigma': 1.0})),
        ('Morlet', ('morlet', {'omega': 3.0, 'sigma': 1.0})),
    ]
    
    print("Sensitivity to text modifications:")
    print("Wavelet\t\tPunct\tCase\tOrder\tSynonym")
    print("-" * 50)
    
    for wavelet_name, decay_spec in wavelets:
        # Create fingerprint for original text
        fp_orig = fingerprint(
            base_text,
            alphabet=EN34,
            horizon=5,
            decay=decay_spec,
            normalize='global'
        )
        
        distances = []
        for mod_name, mod_text in modifications[1:]:  # Skip original
            fp_mod = fingerprint(
                mod_text,
                alphabet=EN34,
                horizon=5,
                decay=decay_spec,
                normalize='global'
            )
            
            # Use appropriate distance metric
            if np.iscomplexobj(fp_orig):
                distance = dist_l2_multi(fp_orig, fp_mod)
            else:
                distance = dist_cosine(fp_orig, fp_mod)
            
            distances.append(distance)
        
        print(f"{wavelet_name:12s}\t{distances[0]:.3f}\t{distances[1]:.3f}\t{distances[2]:.3f}\t{distances[3]:.3f}")

if __name__ == "__main__":
    visualize_kernels()
    morlet_parameter_analysis()
    kernel_bank_analysis()
    phase_modulation_demo()
    complex_metric_comparison()
    wavelet_sensitivity_analysis()
    print("\n✓ Wavelet analysis example completed!")