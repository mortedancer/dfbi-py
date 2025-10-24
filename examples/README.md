# DFBI Examples

This directory contains comprehensive examples demonstrating all features and capabilities of the DFBI library. Each example is self-contained and includes configuration files for easy customization.

## Quick Start

### Windows
```powershell
.\run_examples.ps1
```

### Linux/macOS
```bash
./run_examples.sh
```

### Manual Execution
```bash
cd examples/01_basic_analysis
python example.py
```

## Examples Overview

### 1. Basic Analysis (`01_basic_analysis/`)
**Difficulty: Beginner**

Learn the fundamentals of DFBI text fingerprinting:
- Simple text fingerprinting
- Distance calculations
- Horizon parameter effects
- Basic decay functions

**Key Concepts:**
- `fingerprint()` function
- Distance metrics (L2, cosine)
- Horizon values
- Normalization strategies

**Files:**
- `example.py` - Main demonstration
- `config.yaml` - Configuration options

### 2. Authorship Attribution (`02_authorship_attribution/`)
**Difficulty: Intermediate**

Identify authors using writing style analysis:
- Author profile creation
- Classification algorithms
- Style analysis metrics
- Multi-author comparison

**Key Concepts:**
- Author profiling
- Classification accuracy
- Style separation metrics
- Kernel comparison for authorship

**Files:**
- `example.py` - Authorship classification demo
- `config.yaml` - Authorship-specific settings

### 3. Language Detection (`03_language_detection/`)
**Difficulty: Intermediate**

Automatic language identification:
- Multi-language fingerprinting
- Cross-alphabet analysis
- Language profile building
- Character frequency analysis

**Key Concepts:**
- Multi-alphabet support (RU41, EN34)
- Language-specific patterns
- Cross-language comparison
- Character-level analysis

**Files:**
- `example.py` - Language detection system
- `config.yaml` - Language-specific configurations

### 4. Wavelet Analysis (`04_wavelet_analysis/`)
**Difficulty: Advanced**

Advanced mathematical kernels and complex analysis:
- Wavelet kernel visualization
- Morlet parameter tuning
- Kernel bank combinations
- Phase modulation
- Complex number handling

**Key Concepts:**
- Morlet wavelets
- Mexican Hat kernels
- Kernel banks
- Phase analysis
- Complex metrics

**Files:**
- `example.py` - Advanced wavelet demonstrations
- `config.yaml` - Wavelet-specific parameters

### 5. Batch Processing (`05_batch_processing/`)
**Difficulty: Intermediate**

Efficient large-scale text processing:
- Batch fingerprinting
- Performance optimization
- Memory management
- Similarity matrices
- Window scanning

**Key Concepts:**
- `batch_from_texts()`
- `window_scan()`
- Memory optimization
- Throughput analysis
- Parallel processing simulation

**Files:**
- `example.py` - Batch processing workflows
- `config.yaml` - Performance configurations

### 6. Performance Optimization (`06_performance_optimization/`)
**Difficulty: Advanced**

Benchmarking and parameter tuning:
- Comprehensive benchmarking
- Parameter optimization
- Memory usage analysis
- Algorithm comparison
- Performance recommendations

**Key Concepts:**
- Benchmarking methodology
- Parameter sensitivity
- Memory profiling
- Performance metrics
- Optimization strategies

**Files:**
- `example.py` - Performance analysis suite
- `config.yaml` - Optimization parameters

## Configuration System

Each example includes a `config.yaml` file demonstrating different parameter combinations:

```yaml
# Example configuration structure
default_settings:
  alphabet: "en"
  horizon: 3
  normalize: "global"

profiles:
  fast:
    decay: "('exp', 1.0)"
  
  accurate:
    decay: "('morlet', {'omega': 3.0, 'sigma': 1.0})"
    
  complex:
    bank: "morlet:omega=3.0;sigma=1.0 + gauss:mu=2.0;sigma=1.5"
    aggregate: "sum_abs"
```

## Integration Examples

### Simple Pipeline Integration

```python
import yaml
from dfbi import fingerprint

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Apply configuration
def analyze_text(text, profile='default'):
    settings = config['profiles'][profile]
    return fingerprint(text, **settings)

# Usage
result = analyze_text("Your text here", profile='accurate')
```

### Batch Processing Pipeline

```python
from dfbi import batch_from_texts
import yaml

def process_document_collection(documents, config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return batch_from_texts(
        documents,
        **config['batch_processing']
    )

# Usage
docs = {'doc1': 'text1', 'doc2': 'text2'}
fingerprints = process_document_collection(docs)
```

## Data Directory

The `../data/` directory contains sample datasets organized by use case:

```
data/
├── sample_texts/
│   ├── english/          # English text samples
│   └── russian/          # Russian text samples
├── authorship_samples/
│   ├── shakespeare/      # Shakespeare excerpts
│   ├── hemingway/        # Hemingway excerpts
│   └── dickens/          # Dickens excerpts
└── benchmark_data/       # Performance testing data
```

## Running Examples

### Interactive Mode
Run the script without arguments for an interactive menu:

```bash
./run_examples.sh
```

### Specific Example
```bash
./run_examples.sh --example 1
./run_examples.sh --example wavelet
```

### All Examples
```bash
./run_examples.sh --all
```

### With Custom Configuration
```python
# Modify config.yaml in any example directory
cd examples/01_basic_analysis
# Edit config.yaml
python example.py
```

## Requirements

- Python 3.8+
- NumPy >= 1.22
- Pandas >= 1.4
- PyYAML (for configuration)
- Matplotlib (for visualizations)
- scikit-learn (for metrics)
- psutil (for performance monitoring)

Install requirements:
```bash
pip install numpy pandas pyyaml matplotlib scikit-learn psutil
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure DFBI package is installed
   ```bash
   cd dfbi_lib_0_1_6_wave_kernels
   pip install -e .
   ```

2. **Memory Issues**: Reduce horizon or use window scanning
   ```python
   # Use smaller horizon
   fingerprint(text, horizon=3)
   
   # Or use window scanning for large texts
   from dfbi import window_scan
   chunks = window_scan(large_text, win=1000, step=500)
   ```

3. **Slow Performance**: Use simpler decay functions
   ```python
   # Fast exponential decay
   fingerprint(text, decay=('exp', 1.0))
   ```

4. **Complex Number Issues**: Use appropriate metrics
   ```python
   from dfbi.metrics import dist_l2_multi, dist_cosine
   
   # For complex fingerprints
   distance = dist_l2_multi(fp1, fp2)
   # or
   distance = dist_cosine(fp1, fp2)
   ```

### Getting Help

1. Check example output for error messages
2. Verify configuration syntax in YAML files
3. Ensure all dependencies are installed
4. Run tests: `python -m pytest dfbi_lib_0_1_6_wave_kernels/tests/`

## Advanced Usage

### Custom Kernels
```python
# Define custom decay function
def custom_decay(d):
    return 1.0 / (d ** 1.5)

# Use with fingerprint
fp = fingerprint(text, decay=custom_decay)
```

### Pipeline Integration
```python
class DFBIAnalyzer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def analyze(self, text, profile='default'):
        return fingerprint(text, **self.config[profile])
    
    def compare(self, text1, text2, profile='default'):
        fp1 = self.analyze(text1, profile)
        fp2 = self.analyze(text2, profile)
        return dist_cosine(fp1, fp2)

# Usage
analyzer = DFBIAnalyzer('config.yaml')
similarity = analyzer.compare("text1", "text2", "accurate")
```

## Contributing

To add new examples:

1. Create new directory: `examples/XX_new_example/`
2. Add `example.py` with comprehensive demonstrations
3. Include `config.yaml` with relevant parameters
4. Update this README with example description
5. Add sample data to `../data/` if needed

## License

All examples are provided under the same MIT license as the DFBI library.
