#!/usr/bin/env python3
"""
Authorship Attribution Example
=============================

This example demonstrates how to use DFBI for authorship attribution
using different authors' writing styles.
"""

import numpy as np
from pathlib import Path
from dfbi import fingerprint, batch_from_texts
from dfbi.alphabet import EN34
from dfbi.metrics import dist_l2, dist_cosine, dist_chi2
from sklearn.metrics import accuracy_score
import yaml

def load_config():
    """Load configuration for authorship analysis."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert decay lists to tuples for compatibility
    for section in config.values():
        if 'decay' in section and isinstance(section['decay'], list):
            section['decay'] = tuple(section['decay'])
    
    return config

def create_author_corpus():
    """Create a synthetic corpus of different authors."""
    authors = {
        "shakespeare": [
            "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
            "All the world's a stage, and all the men and women merely players. They have their exits and their entrances.",
            "What's in a name? That which we call a rose by any other name would smell as sweet."
        ],
        "hemingway": [
            "The sun also rises. It was a nice morning. The old man went fishing.",
            "He was an old man who fished alone in a skiff in the Gulf Stream.",
            "The fish was beautiful. It had been alive for a long time."
        ],
        "dickens": [
            "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
            "Please, sir, I want some more. The workhouse was a terrible place for children.",
            "A Christmas Carol tells the story of Ebenezer Scrooge, a miserly old man."
        ]
    }
    return authors

def authorship_classification():
    """Perform authorship classification using DFBI."""
    print("=== Authorship Attribution Demo ===")
    
    config = load_config()
    authors = create_author_corpus()
    
    # Create training fingerprints
    print("Creating author profiles...")
    author_profiles = {}
    
    for author, texts in authors.items():
        # Use first two texts for training
        train_texts = texts[:2]
        fingerprints = []
        
        for text in train_texts:
            fp = fingerprint(
                text,
                alphabet=EN34,
                **config['authorship']
            )
            fingerprints.append(fp)
        
        # Average the fingerprints to create author profile
        author_profiles[author] = np.mean(fingerprints, axis=0)
        print(f"Profile created for {author}")
    
    # Test on remaining texts
    print("\nTesting authorship attribution...")
    predictions = []
    true_labels = []
    
    for author, texts in authors.items():
        test_text = texts[2]  # Use third text for testing
        
        test_fp = fingerprint(
            test_text,
            alphabet=EN34,
            **config['authorship']
        )
        
        # Find closest author
        min_distance = float('inf')
        predicted_author = None
        
        for candidate_author, profile in author_profiles.items():
            distance = dist_cosine(test_fp, profile)
            if distance < min_distance:
                min_distance = distance
                predicted_author = candidate_author
        
        predictions.append(predicted_author)
        true_labels.append(author)
        
        print(f"Text: '{test_text[:50]}...'")
        print(f"True author: {author}, Predicted: {predicted_author}")
        print(f"Distance: {min_distance:.4f}")
        print()
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Classification accuracy: {accuracy:.2%}")

def style_analysis():
    """Analyze writing style differences using multiple metrics."""
    print("\n=== Style Analysis ===")
    
    config = load_config()
    authors = create_author_corpus()
    
    # Create all fingerprints
    all_fingerprints = {}
    for author, texts in authors.items():
        all_fingerprints[author] = []
        for i, text in enumerate(texts):
            fp = fingerprint(
                text,
                alphabet=EN34,
                **config['style_analysis']
            )
            all_fingerprints[author].append((f"{author}_{i+1}", fp))
    
    # Compare using different metrics
    metrics = {
        'L2': dist_l2,
        'Cosine': dist_cosine,
        'Chi2': dist_chi2
    }
    
    print("Inter-author distances:")
    for metric_name, metric_func in metrics.items():
        print(f"\n{metric_name} distances:")
        
        # Calculate average intra-author distance
        intra_distances = []
        for author, fps in all_fingerprints.items():
            for i in range(len(fps)):
                for j in range(i+1, len(fps)):
                    dist = metric_func(fps[i][1], fps[j][1])
                    intra_distances.append(dist)
        
        avg_intra = np.mean(intra_distances)
        
        # Calculate average inter-author distance
        inter_distances = []
        author_names = list(all_fingerprints.keys())
        for i in range(len(author_names)):
            for j in range(i+1, len(author_names)):
                author1, author2 = author_names[i], author_names[j]
                for fp1_name, fp1 in all_fingerprints[author1]:
                    for fp2_name, fp2 in all_fingerprints[author2]:
                        dist = metric_func(fp1, fp2)
                        inter_distances.append(dist)
        
        avg_inter = np.mean(inter_distances)
        separation = avg_inter / avg_intra if avg_intra > 0 else float('inf')
        
        print(f"  Average intra-author distance: {avg_intra:.4f}")
        print(f"  Average inter-author distance: {avg_inter:.4f}")
        print(f"  Separation ratio: {separation:.2f}")

def kernel_comparison():
    """Compare different kernels for authorship attribution."""
    print("\n=== Kernel Comparison ===")
    
    authors = create_author_corpus()
    test_text = authors['shakespeare'][0]
    
    kernels = [
        ('Exponential', ('exp', 0.7)),
        ('Gaussian', ('gauss', {'mu': 2.0, 'sigma': 1.0})),
        ('Morlet', ('morlet', {'omega': 3.0, 'sigma': 1.0})),
    ]
    
    print(f"Analyzing text: '{test_text[:50]}...'")
    print()
    
    for kernel_name, decay_spec in kernels:
        fp = fingerprint(
            test_text,
            alphabet=EN34,
            horizon=5,
            decay=decay_spec,
            normalize='global'
        )
        
        complexity = np.std(fp)
        sparsity = 1 - np.count_nonzero(fp) / fp.size
        
        print(f"{kernel_name:12s}: Complexity={complexity:.4f}, Sparsity={sparsity:.3f}")

if __name__ == "__main__":
    authorship_classification()
    style_analysis()
    kernel_comparison()
    print("\n✓ Authorship attribution example completed!")