#!/usr/bin/env python3
"""
Basic Text Analysis Example
==========================

This example demonstrates the simplest usage of DFBI library
for text fingerprinting and basic comparison.
"""

import numpy as np
from dfbi import fingerprint
from dfbi.alphabet import RU41, EN34
from dfbi.metrics import dist_l2, dist_cosine

def basic_fingerprint_demo():
    """Demonstrate basic fingerprinting functionality."""
    print("=== Basic Fingerprint Demo ===")
    
    # Sample texts
    text1 = "This is a sample text for analysis."
    text2 = "This is another sample text for comparison."
    text3 = "Completely different content here."
    
    # Create fingerprints with default settings
    print("Creating fingerprints...")
    fp1 = fingerprint(text1, alphabet=EN34, horizon=3)
    fp2 = fingerprint(text2, alphabet=EN34, horizon=3)
    fp3 = fingerprint(text3, alphabet=EN34, horizon=3)
    
    print(f"Fingerprint shape: {fp1.shape}")
    print(f"Fingerprint type: {fp1.dtype}")
    
    # Calculate distances
    dist_12 = dist_l2(fp1, fp2)
    dist_13 = dist_l2(fp1, fp3)
    dist_23 = dist_l2(fp2, fp3)
    
    print(f"\nDistances (L2):")
    print(f"Text1 vs Text2: {dist_12:.4f}")
    print(f"Text1 vs Text3: {dist_13:.4f}")
    print(f"Text2 vs Text3: {dist_23:.4f}")
    
    # Calculate cosine similarities
    cos_12 = 1 - dist_cosine(fp1, fp2)
    cos_13 = 1 - dist_cosine(fp1, fp3)
    cos_23 = 1 - dist_cosine(fp2, fp3)
    
    print(f"\nSimilarities (Cosine):")
    print(f"Text1 vs Text2: {cos_12:.4f}")
    print(f"Text1 vs Text3: {cos_13:.4f}")
    print(f"Text2 vs Text3: {cos_23:.4f}")

def horizon_comparison():
    """Compare different horizon values."""
    print("\n=== Horizon Comparison ===")
    
    text = "The quick brown fox jumps over the lazy dog. " * 10
    
    horizons = [1, 3, 5, 10]
    
    for h in horizons:
        fp = fingerprint(text, alphabet=EN34, horizon=h)
        non_zero = np.count_nonzero(fp)
        total = fp.size
        density = non_zero / total
        
        print(f"Horizon {h:2d}: Shape {fp.shape}, Density {density:.3f}")

def decay_function_demo():
    """Demonstrate different decay functions."""
    print("\n=== Decay Function Demo ===")
    
    text = "Sample text for decay function demonstration. " * 5
    
    decay_functions = [
        ('exp', 0.7),
        ('gauss', {'mu': 2.0, 'sigma': 1.0}),
        ('inv', {'p': 1.0}),
    ]
    
    for decay_spec in decay_functions:
        fp = fingerprint(text, alphabet=EN34, horizon=5, decay=decay_spec)
        print(f"Decay {decay_spec}: Sum = {fp.sum():.4f}")

if __name__ == "__main__":
    basic_fingerprint_demo()
    horizon_comparison()
    decay_function_demo()
    print("\n✓ Basic analysis example completed!")