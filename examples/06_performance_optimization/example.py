#!/usr/bin/env python3
"""
Performance Optimization Example
===============================

This example demonstrates various performance optimization techniques,
benchmarking, and parameter tuning for DFBI library.
"""

import numpy as np
import time
import psutil
import os
from pathlib import Path
from dfbi import fingerprint, batch_from_texts
from dfbi.alphabet import EN34, RU41
from dfbi.metrics import dist_l2, dist_cosine, dist_chi2
import yaml
import json

def load_config():
    """Load configuration for performance optimization."""
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

def create_benchmark_corpus():
    """Create a benchmark corpus with varying text sizes."""
    corpus = {}
    
    # Small texts (100-500 chars)
    small_base = "Short text for performance testing. "
    for i in range(10):
        corpus[f'small_{i:02d}'] = small_base * (i + 3)
    
    # Medium texts (1K-5K chars)
    medium_base = "Medium length text for comprehensive performance analysis and benchmarking. "
    for i in range(10):
        corpus[f'medium_{i:02d}'] = medium_base * (15 + i * 5)
    
    # Large texts (10K+ chars)
    large_base = "Large text document for testing performance with substantial content and extensive analysis. "
    for i in range(5):
        corpus[f'large_{i:02d}'] = large_base * (100 + i * 50)
    
    return corpus

def measure_performance(func, *args, **kwargs):
    """Measure execution time and memory usage of a function."""
    process = psutil.Process(os.getpid())
    
    # Measure initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Measure execution time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Measure final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'result': result,
        'execution_time': end_time - start_time,
        'memory_used': final_memory - initial_memory,
        'peak_memory': final_memory
    }

def horizon_performance_analysis():
    """Analyze performance impact of different horizon values."""
    print("=== Horizon Performance Analysis ===")
    
    config = load_config()
    test_text = "Performance analysis of horizon parameter effects on processing speed and accuracy. " * 20
    
    horizons = [1, 2, 3, 5, 8, 10, 15, 20]
    results = []
    
    print("Horizon\tTime(ms)\tMemory(MB)\tDensity")
    print("-" * 45)
    
    for horizon in horizons:
        perf = measure_performance(
            fingerprint,
            test_text,
            alphabet=EN34,
            horizon=horizon,
            **config['horizon_test']
        )
        
        fp = perf['result']
        density = np.count_nonzero(fp) / fp.size
        
        results.append({
            'horizon': horizon,
            'time_ms': perf['execution_time'] * 1000,
            'memory_mb': perf['memory_used'],
            'density': density,
            'matrix_size': fp.size
        })
        
        print(f"{horizon:7d}\t{perf['execution_time']*1000:8.2f}\t{perf['memory_used']:10.2f}\t{density:7.3f}")
    
    # Find optimal horizon (balance of speed and information)
    efficiency_scores = []
    for r in results:
        # Higher density and lower time is better
        efficiency = r['density'] / (r['time_ms'] / 1000) if r['time_ms'] > 0 else 0
        efficiency_scores.append(efficiency)
    
    optimal_idx = np.argmax(efficiency_scores)
    optimal_horizon = results[optimal_idx]['horizon']
    
    print(f"\nOptimal horizon for efficiency: {optimal_horizon}")
    
    return results

def decay_function_benchmark():
    """Benchmark different decay functions."""
    print("\n=== Decay Function Benchmark ===")
    
    config = load_config()
    test_text = "Decay function performance comparison across different mathematical formulations. " * 15
    
    decay_functions = [
        ('Exponential', ('exp', 0.7)),
        ('Inverse', ('inv', {'p': 1.0})),
        ('Gaussian', ('gauss', {'mu': 2.0, 'sigma': 1.0})),
        ('Morlet', ('morlet', {'omega': 3.0, 'sigma': 1.0})),
        ('Mexican Hat', ('mexican', {'sigma': 1.5})),
    ]
    
    results = []
    
    print("Decay Function\tTime(ms)\tMemory(MB)\tComplex\tSum")
    print("-" * 60)
    
    for name, decay_spec in decay_functions:
        perf = measure_performance(
            fingerprint,
            test_text,
            alphabet=EN34,
            decay=decay_spec,
            **config['decay_test']
        )
        
        fp = perf['result']
        is_complex = np.iscomplexobj(fp)
        fp_sum = np.sum(np.abs(fp)) if is_complex else np.sum(fp)
        
        results.append({
            'name': name,
            'time_ms': perf['execution_time'] * 1000,
            'memory_mb': perf['memory_used'],
            'is_complex': is_complex,
            'sum': fp_sum
        })
        
        print(f"{name:14s}\t{perf['execution_time']*1000:8.2f}\t{perf['memory_used']:10.2f}\t{is_complex}\t{fp_sum:8.4f}")
    
    # Find fastest decay function
    fastest_idx = np.argmin([r['time_ms'] for r in results])
    fastest_decay = results[fastest_idx]['name']
    
    print(f"\nFastest decay function: {fastest_decay}")
    
    return results

def batch_size_optimization():
    """Optimize batch processing size."""
    print("\n=== Batch Size Optimization ===")
    
    config = load_config()
    corpus = create_benchmark_corpus()
    
    # Test different batch sizes
    batch_sizes = [1, 5, 10, 15, 20, 25]
    results = []
    
    print("Batch Size\tTime(s)\tThroughput(KB/s)\tMemory(MB)")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        # Select subset of corpus
        corpus_subset = dict(list(corpus.items())[:batch_size])
        total_chars = sum(len(text) for text in corpus_subset.values())
        
        perf = measure_performance(
            batch_from_texts,
            corpus_subset,
            alphabet=EN34,
            **config['batch_optimization']
        )
        
        throughput = (total_chars / 1024) / perf['execution_time'] if perf['execution_time'] > 0 else 0
        
        results.append({
            'batch_size': batch_size,
            'time_s': perf['execution_time'],
            'throughput_kbs': throughput,
            'memory_mb': perf['memory_used'],
            'chars_per_sec': total_chars / perf['execution_time'] if perf['execution_time'] > 0 else 0
        })
        
        print(f"{batch_size:10d}\t{perf['execution_time']:7.3f}\t{throughput:15.2f}\t{perf['memory_used']:10.2f}")
    
    # Find optimal batch size
    optimal_idx = np.argmax([r['throughput_kbs'] for r in results])
    optimal_batch_size = results[optimal_idx]['batch_size']
    
    print(f"\nOptimal batch size: {optimal_batch_size}")
    
    return results

def memory_usage_analysis():
    """Analyze memory usage patterns."""
    print("\n=== Memory Usage Analysis ===")
    
    config = load_config()
    
    # Test with different text sizes
    text_sizes = [100, 500, 1000, 5000, 10000, 20000]  # characters
    base_text = "Memory usage analysis with varying text sizes. "
    
    results = []
    
    print("Text Size\tTime(ms)\tMemory(MB)\tMatrix Size\tBytes/Char")
    print("-" * 60)
    
    for size in text_sizes:
        # Create text of specific size
        repeat_count = max(1, size // len(base_text))
        test_text = (base_text * repeat_count)[:size]
        
        perf = measure_performance(
            fingerprint,
            test_text,
            alphabet=EN34,
            **config['memory_analysis']
        )
        
        fp = perf['result']
        matrix_size = fp.nbytes
        bytes_per_char = matrix_size / len(test_text) if len(test_text) > 0 else 0
        
        results.append({
            'text_size': size,
            'time_ms': perf['execution_time'] * 1000,
            'memory_mb': perf['memory_used'],
            'matrix_bytes': matrix_size,
            'bytes_per_char': bytes_per_char
        })
        
        print(f"{size:9d}\t{perf['execution_time']*1000:8.2f}\t{perf['memory_used']:10.2f}\t{matrix_size:11d}\t{bytes_per_char:10.3f}")
    
    # Analyze memory scaling
    if len(results) > 1:
        size_ratio = results[-1]['text_size'] / results[0]['text_size']
        memory_ratio = results[-1]['memory_mb'] / results[0]['memory_mb'] if results[0]['memory_mb'] > 0 else 0
        
        print(f"\nMemory scaling: {memory_ratio:.2f}x for {size_ratio:.2f}x text size")
    
    return results

def algorithm_comparison():
    """Compare different algorithmic approaches."""
    print("\n=== Algorithm Comparison ===")
    
    config = load_config()
    test_text = "Algorithm comparison for different DFBI configurations and parameter combinations. " * 10
    
    algorithms = [
        ('Fast', config['fast_algorithm']),
        ('Balanced', config['balanced_algorithm']),
        ('Accurate', config['accurate_algorithm']),
        ('Complex', config['complex_algorithm']),
    ]
    
    results = []
    
    print("Algorithm\tTime(ms)\tMemory(MB)\tAccuracy Score")
    print("-" * 50)
    
    # Create reference fingerprint for accuracy comparison
    ref_fp = fingerprint(
        test_text,
        alphabet=EN34,
        **config['reference_algorithm']
    )
    
    for name, algo_config in algorithms:
        perf = measure_performance(
            fingerprint,
            test_text,
            alphabet=EN34,
            **algo_config
        )
        
        fp = perf['result']
        
        # Calculate accuracy as similarity to reference
        if np.iscomplexobj(fp) or np.iscomplexobj(ref_fp):
            accuracy = 1 - dist_cosine(fp, ref_fp)
        else:
            accuracy = 1 - dist_cosine(fp, ref_fp)
        
        results.append({
            'name': name,
            'time_ms': perf['execution_time'] * 1000,
            'memory_mb': perf['memory_used'],
            'accuracy': accuracy
        })
        
        print(f"{name:9s}\t{perf['execution_time']*1000:8.2f}\t{perf['memory_used']:10.2f}\t{accuracy:13.4f}")
    
    # Find best balanced algorithm
    efficiency_scores = []
    for r in results:
        # Balance accuracy and speed
        efficiency = r['accuracy'] / (r['time_ms'] / 1000) if r['time_ms'] > 0 else 0
        efficiency_scores.append(efficiency)
    
    best_idx = np.argmax(efficiency_scores)
    best_algorithm = results[best_idx]['name']
    
    print(f"\nBest balanced algorithm: {best_algorithm}")
    
    return results

def comprehensive_benchmark():
    """Run comprehensive benchmark suite."""
    print("\n=== Comprehensive Benchmark ===")
    
    config = load_config()
    corpus = create_benchmark_corpus()
    
    # Benchmark different aspects
    benchmark_results = {
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
        },
        'horizon_analysis': horizon_performance_analysis(),
        'decay_benchmark': decay_function_benchmark(),
        'batch_optimization': batch_size_optimization(),
        'memory_analysis': memory_usage_analysis(),
        'algorithm_comparison': algorithm_comparison()
    }
    
    # Save benchmark results
    output_path = Path(__file__).parent / 'benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    print(f"\nBenchmark results saved to {output_path}")
    
    # Generate performance recommendations
    print("\n=== Performance Recommendations ===")
    
    # Horizon recommendation
    horizon_results = benchmark_results['horizon_analysis']
    fast_horizons = [r for r in horizon_results if r['time_ms'] < 50]
    if fast_horizons:
        recommended_horizon = max(fast_horizons, key=lambda x: x['density'])['horizon']
        print(f"Recommended horizon for speed: {recommended_horizon}")
    
    # Decay function recommendation
    decay_results = benchmark_results['decay_benchmark']
    fastest_decay = min(decay_results, key=lambda x: x['time_ms'])['name']
    print(f"Fastest decay function: {fastest_decay}")
    
    # Batch size recommendation
    batch_results = benchmark_results['batch_optimization']
    optimal_batch = max(batch_results, key=lambda x: x['throughput_kbs'])['batch_size']
    print(f"Optimal batch size: {optimal_batch}")
    
    return benchmark_results

if __name__ == "__main__":
    print("Starting comprehensive performance analysis...")
    
    horizon_results = horizon_performance_analysis()
    decay_results = decay_function_benchmark()
    batch_results = batch_size_optimization()
    memory_results = memory_usage_analysis()
    algorithm_results = algorithm_comparison()
    
    # Run full benchmark suite
    full_results = comprehensive_benchmark()
    
    print("\n✓ Performance optimization example completed!")
    print("Check 'benchmark_results.json' for detailed results.")