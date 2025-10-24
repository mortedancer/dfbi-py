#!/usr/bin/env python3
"""
Batch Processing Example
=======================

This example demonstrates efficient batch processing of multiple texts,
performance optimization, and large-scale text analysis workflows.
"""

import numpy as np
import time
from pathlib import Path
from dfbi import fingerprint, batch_from_texts, window_scan
from dfbi.alphabet import EN34, RU41
from dfbi.metrics import dist_l2, dist_cosine
import yaml
import json

def load_config():
    """Load configuration for batch processing."""
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

def create_sample_corpus():
    """Create a sample corpus for batch processing demonstration."""
    corpus = {
        'doc_001': "The first document contains information about machine learning algorithms and their applications in natural language processing.",
        'doc_002': "Second document discusses the importance of data preprocessing in machine learning pipelines and feature engineering techniques.",
        'doc_003': "This third document explores deep learning architectures, particularly neural networks and their use in computer vision tasks.",
        'doc_004': "Document four examines the role of artificial intelligence in modern software development and automation processes.",
        'doc_005': "The fifth document analyzes statistical methods used in data science and their practical implementations in research.",
        'doc_006': "Sixth document covers distributed computing systems and their applications in big data processing and analytics.",
        'doc_007': "This document discusses cloud computing platforms and their integration with machine learning frameworks and tools.",
        'doc_008': "Document eight explores cybersecurity measures in modern computing environments and threat detection systems.",
        'doc_009': "The ninth document examines database management systems and their optimization for large-scale data operations.",
        'doc_010': "Final document discusses software engineering best practices and methodologies for developing scalable applications."
    }
    return corpus

def batch_processing_demo():
    """Demonstrate basic batch processing functionality."""
    print("=== Batch Processing Demo ===")
    
    config = load_config()
    corpus = create_sample_corpus()
    
    print(f"Processing {len(corpus)} documents...")
    
    # Measure processing time
    start_time = time.time()
    
    # Batch process all documents
    fingerprints = batch_from_texts(
        corpus,
        alphabet=EN34,
        **config['batch_basic']
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Batch processing completed in {processing_time:.3f} seconds")
    print(f"Average time per document: {processing_time / len(corpus):.4f} seconds")
    
    # Analyze results
    total_chars = sum(len(text) for text in corpus.values())
    throughput = total_chars / processing_time / 1024  # KB/s
    
    print(f"Total characters processed: {total_chars:,}")
    print(f"Throughput: {throughput:.2f} KB/s")
    
    # Show fingerprint properties
    sample_fp = list(fingerprints.values())[0]
    print(f"Fingerprint shape: {sample_fp.shape}")
    print(f"Fingerprint dtype: {sample_fp.dtype}")
    
    return fingerprints

def performance_comparison():
    """Compare performance of different configurations."""
    print("\n=== Performance Comparison ===")
    
    config = load_config()
    corpus = create_sample_corpus()
    
    # Different configuration profiles
    profiles = ['fast', 'balanced', 'accurate']
    
    results = {}
    
    for profile in profiles:
        print(f"\nTesting {profile} profile...")
        
        start_time = time.time()
        fingerprints = batch_from_texts(
            corpus,
            alphabet=EN34,
            **config[f'batch_{profile}']
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        total_chars = sum(len(text) for text in corpus.values())
        throughput = total_chars / processing_time / 1024  # KB/s
        
        results[profile] = {
            'time': processing_time,
            'throughput': throughput,
            'fingerprints': fingerprints
        }
        
        print(f"  Time: {processing_time:.3f}s")
        print(f"  Throughput: {throughput:.2f} KB/s")
    
    # Compare accuracy using document similarity
    print("\nAccuracy comparison (document similarity analysis):")
    
    # Calculate average intra-similarity for each profile
    for profile in profiles:
        fps = list(results[profile]['fingerprints'].values())
        similarities = []
        
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                sim = 1 - dist_cosine(fps[i], fps[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        print(f"  {profile:8s}: Avg similarity = {avg_similarity:.4f} ± {std_similarity:.4f}")

def large_text_processing():
    """Demonstrate processing of large texts using window scanning."""
    print("\n=== Large Text Processing ===")
    
    config = load_config()
    
    # Create a large synthetic text
    base_text = "This is a sample sentence for large text processing demonstration. "
    large_text = base_text * 1000  # ~65KB text
    
    print(f"Processing large text ({len(large_text):,} characters)...")
    
    # Method 1: Process entire text at once
    start_time = time.time()
    fp_whole = fingerprint(
        large_text,
        alphabet=EN34,
        **config['large_text']
    )
    time_whole = time.time() - start_time
    
    print(f"Whole text processing: {time_whole:.3f}s")
    
    # Method 2: Window scanning
    window_size = 1000
    step_size = 500
    
    start_time = time.time()
    fp_windows = window_scan(
        large_text,
        win=window_size,
        step=step_size,
        alphabet=EN34,
        **config['large_text']
    )
    time_windows = time.time() - start_time
    
    print(f"Window scanning: {time_windows:.3f}s")
    print(f"Number of windows: {len(fp_windows)}")
    
    # Aggregate window results
    fp_aggregated = np.mean(fp_windows, axis=0)
    
    # Compare results
    similarity = 1 - dist_cosine(fp_whole, fp_aggregated)
    print(f"Similarity between methods: {similarity:.4f}")
    
    # Memory usage comparison
    whole_memory = fp_whole.nbytes
    windows_memory = sum(fp.nbytes for fp in fp_windows)
    
    print(f"Memory usage - Whole: {whole_memory:,} bytes")
    print(f"Memory usage - Windows: {windows_memory:,} bytes")

def parallel_processing_simulation():
    """Simulate parallel processing benefits."""
    print("\n=== Parallel Processing Simulation ===")
    
    config = load_config()
    corpus = create_sample_corpus()
    
    # Simulate sequential processing
    print("Sequential processing simulation:")
    start_time = time.time()
    
    sequential_results = {}
    for doc_id, text in corpus.items():
        fp = fingerprint(
            text,
            alphabet=EN34,
            **config['batch_balanced']
        )
        sequential_results[doc_id] = fp
    
    sequential_time = time.time() - start_time
    
    # Actual batch processing (simulates parallel benefits)
    print("Batch processing:")
    start_time = time.time()
    
    batch_results = batch_from_texts(
        corpus,
        alphabet=EN34,
        **config['batch_balanced']
    )
    
    batch_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.3f}s")
    print(f"Batch time: {batch_time:.3f}s")
    print(f"Speedup factor: {sequential_time / batch_time:.2f}x")

def similarity_matrix_analysis():
    """Create and analyze similarity matrix for document corpus."""
    print("\n=== Similarity Matrix Analysis ===")
    
    config = load_config()
    corpus = create_sample_corpus()
    
    # Create fingerprints
    fingerprints = batch_from_texts(
        corpus,
        alphabet=EN34,
        **config['batch_accurate']
    )
    
    # Create similarity matrix
    doc_ids = list(fingerprints.keys())
    n_docs = len(doc_ids)
    similarity_matrix = np.zeros((n_docs, n_docs))
    
    print("Computing similarity matrix...")
    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                fp1 = fingerprints[doc_ids[i]]
                fp2 = fingerprints[doc_ids[j]]
                similarity = 1 - dist_cosine(fp1, fp2)
                similarity_matrix[i, j] = similarity
    
    # Analyze similarity matrix
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Find most similar document pairs
    upper_triangle = np.triu(similarity_matrix, k=1)
    max_similarity_idx = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
    max_similarity = upper_triangle[max_similarity_idx]
    
    print(f"Most similar documents: {doc_ids[max_similarity_idx[0]]} and {doc_ids[max_similarity_idx[1]]}")
    print(f"Similarity score: {max_similarity:.4f}")
    
    # Calculate average similarity
    avg_similarity = np.mean(upper_triangle[upper_triangle > 0])
    print(f"Average document similarity: {avg_similarity:.4f}")
    
    # Save results
    results = {
        'document_ids': doc_ids,
        'similarity_matrix': similarity_matrix.tolist(),
        'statistics': {
            'max_similarity': float(max_similarity),
            'avg_similarity': float(avg_similarity),
            'most_similar_pair': [doc_ids[max_similarity_idx[0]], doc_ids[max_similarity_idx[1]]]
        }
    }
    
    output_path = Path(__file__).parent / 'similarity_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")

def memory_optimization_demo():
    """Demonstrate memory optimization techniques."""
    print("\n=== Memory Optimization Demo ===")
    
    config = load_config()
    
    # Create larger corpus for memory testing
    large_corpus = {}
    base_text = "Memory optimization is crucial for processing large document collections efficiently. "
    
    for i in range(50):  # 50 documents
        large_corpus[f'doc_{i:03d}'] = base_text * (i + 1)  # Varying sizes
    
    print(f"Created corpus with {len(large_corpus)} documents")
    
    # Method 1: Process all at once (high memory)
    print("\nMethod 1: Batch processing (high memory)")
    start_time = time.time()
    
    try:
        all_fps = batch_from_texts(
            large_corpus,
            alphabet=EN34,
            **config['memory_optimized']
        )
        batch_time = time.time() - start_time
        print(f"Batch processing completed in {batch_time:.3f}s")
        
        # Estimate memory usage
        sample_fp = list(all_fps.values())[0]
        total_memory = len(all_fps) * sample_fp.nbytes
        print(f"Estimated memory usage: {total_memory / 1024 / 1024:.2f} MB")
        
    except MemoryError:
        print("Batch processing failed due to memory constraints")
    
    # Method 2: Process incrementally (low memory)
    print("\nMethod 2: Incremental processing (low memory)")
    start_time = time.time()
    
    incremental_results = []
    for doc_id, text in large_corpus.items():
        fp = fingerprint(
            text,
            alphabet=EN34,
            **config['memory_optimized']
        )
        # Store only essential information
        incremental_results.append({
            'doc_id': doc_id,
            'fingerprint_sum': float(np.sum(np.abs(fp))),
            'fingerprint_shape': fp.shape
        })
    
    incremental_time = time.time() - start_time
    print(f"Incremental processing completed in {incremental_time:.3f}s")
    print(f"Memory-efficient storage: {len(incremental_results)} summaries")

if __name__ == "__main__":
    fingerprints = batch_processing_demo()
    performance_comparison()
    large_text_processing()
    parallel_processing_simulation()
    similarity_matrix_analysis()
    memory_optimization_demo()
    print("\n✓ Batch processing example completed!")