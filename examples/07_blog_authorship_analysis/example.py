#!/usr/bin/env python3
"""
Advanced Blog Authorship Analysis
=================================

This example demonstrates sophisticated authorship analysis techniques using DFBI
for blog posts and literary texts with real data analysis algorithms.
"""

import numpy as np
from pathlib import Path
import yaml
from collections import defaultdict, Counter

from dfbi import fingerprint, batch_from_texts, window_scan
from dfbi.alphabet import EN34
from dfbi.metrics import dist_l2, dist_cosine, dist_chi2

def load_config():
    """Load configuration for blog authorship analysis."""
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

def load_authorship_data():
    """Load real authorship data from files."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "authorship_samples"
    authors = {}
    
    print(f"Looking for data in: {data_dir}")
    
    if not data_dir.exists():
        print("Data directory not found!")
        return {}
    
    for author_dir in data_dir.iterdir():
        if author_dir.is_dir():
            author_name = author_dir.name
            texts = []
            
            print(f"Processing author: {author_name}")
            
            for text_file in author_dir.glob("*.txt"):
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            texts.append(content)
                            print(f"  Loaded: {text_file.name} ({len(content)} chars)")
                except Exception as e:
                    print(f"Warning: Could not read {text_file}: {e}")
            
            if texts:
                authors[author_name] = texts
    
    return authors

def create_synthetic_blog_data():
    """Create synthetic blog data with different writing styles."""
    return {
        "tech_blogger": [
            "Machine learning algorithms have revolutionized data analysis. Deep neural networks process vast amounts of information efficiently. Python frameworks like TensorFlow enable rapid prototyping. Cloud computing platforms provide scalable infrastructure for AI applications.",
            "Software development practices continue evolving rapidly. Agile methodologies emphasize iterative development cycles. Version control systems like Git facilitate collaborative coding. Continuous integration pipelines automate testing and deployment processes.",
        ],
        "lifestyle_blogger": [
            "Morning routines set the tone for productive days. Meditation practices reduce stress and improve focus. Healthy breakfast choices fuel both body and mind. Regular exercise boosts energy levels throughout the day.",
            "Travel experiences broaden our perspectives on life. Local cuisines offer authentic cultural insights. Photography captures precious memories from adventures. Meeting new people creates lasting friendships and connections.",
        ],
        "academic_writer": [
            "Contemporary research methodologies require rigorous statistical analysis. Peer review processes ensure scholarly publication quality. Interdisciplinary approaches yield innovative research findings. Collaborative research networks facilitate knowledge exchange globally.",
            "Theoretical frameworks provide foundational understanding for empirical studies. Literature reviews synthesize existing knowledge comprehensively. Experimental designs control variables to establish causal relationships. Data interpretation requires careful consideration of limitations.",
        ]
    }

def authorship_analysis_demo():
    """Perform comprehensive authorship analysis."""
    print("=== Advanced Blog Authorship Analysis Demo ===")
    
    config = load_config()
    
    # Load real data first, fallback to synthetic
    authors = load_authorship_data()
    if not authors:
        print("No real data found, using synthetic blog data...")
        authors = create_synthetic_blog_data()
    
    print(f"\nLoaded data for {len(authors)} authors:")
    for author, texts in authors.items():
        print(f"  {author}: {len(texts)} texts")
    
    # Extract fingerprints using different configurations
    print("\n=== Style Signature Analysis ===")
    
    # Basic analysis
    basic_fps = {}
    for author, texts in authors.items():
        author_fps = []
        for text in texts:
            fp = fingerprint(
                text,
                alphabet=EN34,
                **config['basic_analysis']
            )
            author_fps.append(fp)
        basic_fps[author] = author_fps
        print(f"{author}: {len(author_fps)} fingerprints extracted")
    
    # Compute author centroids and distances
    centroids = {}
    for author, fps in basic_fps.items():
        if fps:
            centroid = np.mean(fps, axis=0)
            centroids[author] = centroid
    
    print("\nAuthor style distances (cosine):")
    authors_list = list(centroids.keys())
    for i, author1 in enumerate(authors_list):
        for j, author2 in enumerate(authors_list):
            if i < j:
                try:
                    dist = dist_cosine(centroids[author1], centroids[author2])
                    print(f"  {author1:15s} vs {author2:15s}: {dist:.4f}")
                except Exception as e:
                    print(f"  {author1:15s} vs {author2:15s}: Error - {e}")
    
    # Statistical feature analysis
    print("\n=== Statistical Feature Analysis ===")
    
    for author, texts in authors.items():
        print(f"\n{author}:")
        
        # Calculate statistical features
        word_lengths = []
        sentence_lengths = []
        punct_ratios = []
        
        for text in texts:
            words = text.split()
            sentences = text.split('.')
            
            if words:
                avg_word_len = np.mean([len(word.strip('.,!?;:"()')) for word in words])
                word_lengths.append(avg_word_len)
            
            if sentences:
                avg_sent_len = np.mean([len(sent.split()) for sent in sentences if sent.strip()])
                sentence_lengths.append(avg_sent_len)
            
            punct_ratio = sum(1 for c in text if c in '.,!?;:"()') / len(text) if text else 0
            punct_ratios.append(punct_ratio)
        
        print(f"  Average word length: {np.mean(word_lengths):.2f}")
        print(f"  Average sentence length: {np.mean(sentence_lengths):.2f}")
        print(f"  Punctuation ratio: {np.mean(punct_ratios):.4f}")
    
    # Multi-scale analysis
    print("\n=== Multi-Scale Analysis ===")
    
    scales = ['basic_analysis', 'multi_scale']
    for scale in scales:
        print(f"\nAnalyzing at {scale} scale...")
        
        try:
            # Extract fingerprints for this scale
            scale_fps = {}
            for author, texts in authors.items():
                author_fps = []
                for text in texts:
                    fp = fingerprint(
                        text,
                        alphabet=EN34,
                        **config[scale]
                    )
                    author_fps.append(fp)
                scale_fps[author] = author_fps
            
            # Simple nearest centroid classification
            all_fps = []
            all_labels = []
            
            for author, fps in scale_fps.items():
                for fp in fps:
                    all_fps.append(fp.flatten())
                    all_labels.append(author)
            
            if len(all_fps) > 0:
                X = np.array(all_fps)
                y = all_labels
                
                # Calculate per-author centroids
                author_centroids = {}
                unique_authors = list(set(y))
                
                for author in unique_authors:
                    author_indices = [i for i, label in enumerate(y) if label == author]
                    if author_indices:
                        author_centroids[author] = np.mean(X[author_indices], axis=0)
                
                # Classify each sample
                correct = 0
                total = len(y)
                
                for i, (sample, true_label) in enumerate(zip(X, y)):
                    min_dist = float('inf')
                    predicted_label = None
                    
                    for author, centroid in author_centroids.items():
                        dist = np.linalg.norm(sample - centroid)
                        if dist < min_dist:
                            min_dist = dist
                            predicted_label = author
                    
                    if predicted_label == true_label:
                        correct += 1
                
                accuracy = correct / total if total > 0 else 0
                print(f"  Nearest centroid accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"  Error in {scale} analysis: {e}")
    
    # Window-based analysis for longer texts
    print("\n=== Window-Based Analysis ===")
    
    for author, texts in authors.items():
        print(f"\n{author} window analysis:")
        
        for i, text in enumerate(texts):
            if len(text) > 500:  # Only analyze longer texts
                try:
                    windows = window_scan(text, win=200, step=100, alphabet=EN34, **config['basic_analysis'])
                    
                    if len(windows) > 1:
                        # Analyze consistency across windows
                        similarities = []
                        for j in range(len(windows) - 1):
                            sim = 1 - dist_cosine(windows[j], windows[j + 1])
                            similarities.append(sim)
                        
                        consistency = np.mean(similarities)
                        print(f"  Text {i+1}: {len(windows)} windows, consistency = {consistency:.4f}")
                    else:
                        print(f"  Text {i+1}: Too short for window analysis")
                except Exception as e:
                    print(f"  Text {i+1}: Error in window analysis - {e}")
            else:
                print(f"  Text {i+1}: Too short ({len(text)} chars)")

def main():
    """Main function to run the blog authorship analysis."""
    print("Advanced Blog Authorship Analysis")
    print("=" * 50)
    
    try:
        authorship_analysis_demo()
        
        print(f"\n{'='*60}")
        print("Analysis completed successfully!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    print("\n✓ Advanced blog authorship analysis completed!")