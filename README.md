# DFBI Library - Complete Package with Examples

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DFBI (Deterministic Finite-horizon Bigram Interference)** is a powerful text analysis library that uses wavelet-inspired kernels for text fingerprinting, authorship analysis, and language detection.

## 🚀 Quick Start

### Installation
```bash
cd dfbi_lib_0_1_6_wave_kernels
pip install -e .
```

### Basic Usage
```python
from dfbi import fingerprint
from dfbi.alphabet import EN34

# Simple text fingerprinting
text = "Your text here"
matrix = fingerprint(text, alphabet=EN34, horizon=3)

# Advanced wavelet analysis
matrix = fingerprint(
    text,
    alphabet=EN34,
    horizon=5,
    decay=('morlet', {'omega': 3.2, 'sigma': 1.1}),
    normalize='row'
)
```

### CLI Usage
```bash
# Basic fingerprinting
dfbi-cli fingerprint sample.txt --alphabet en --horizon 3

# Advanced analysis
dfbi-cli fingerprint sample.txt \
  --decay "('morlet', {'omega': 3.2, 'sigma': 1.1})" \
  --mask letters --normalize row
```

## 📁 Project Structure

```
├── dfbi_lib_0_1_6_wave_kernels/    # Core DFBI library
│   ├── src/dfbi/                   # Source code
│   ├── tests/                      # Test suite
│   └── README.md                   # Library documentation
├── docs/                           # Comprehensive documentation
│   ├── README_EN.md               # English documentation
│   └── README_RU.md               # Russian documentation
├── examples/                       # Complete example suite
│   ├── 01_basic_analysis/         # Beginner examples
│   ├── 02_authorship_attribution/ # Author identification
│   ├── 03_language_detection/     # Multi-language analysis
│   ├── 04_wavelet_analysis/       # Advanced kernels
│   ├── 05_batch_processing/       # Large-scale processing
│   ├── 06_performance_optimization/ # Benchmarking
│   └── README.md                  # Examples guide
├── data/                          # Sample datasets
│   ├── sample_texts/              # Basic text samples
│   └── authorship_samples/        # Author-specific texts
├── run_examples.ps1               # Windows runner script
├── run_examples.sh                # Linux/macOS runner script
└── README.md                      # This file
```

## 🎯 Features

### 🌊 Advanced Wavelet Kernels
- **Morlet Wavelets**: Complex oscillatory analysis
- **Mexican Hat**: Edge detection and feature extraction
- **Gaussian**: Smooth decay for balanced analysis
- **Exponential**: Fast computation for real-time applications
- **Kernel Banks**: Combine multiple kernels for enhanced accuracy

### 📊 Sophisticated Metrics
- **L1/L2 Distance**: Standard Euclidean and Manhattan distances
- **Cosine Similarity**: Angle-based similarity for high-dimensional data
- **Chi-squared**: Statistical distance for probability distributions
- **Complex-aware L2**: Multi-channel distance for phase-sensitive analysis

### 🔧 Flexible Configuration
- **YAML Configuration**: Easy parameter management
- **Pipeline Integration**: Simple integration into existing workflows
- **Batch Processing**: Efficient handling of large document collections
- **Memory Optimization**: Techniques for processing large texts

## 🏃‍♂️ Running Examples

### Interactive Mode (Recommended)

**Windows:**
```powershell
.\run_examples.ps1
```

**Linux/macOS:**
```bash
./run_examples.sh
```

### Command Line Options

```bash
# Run specific example
./run_examples.sh --example 1
./run_examples.sh --example wavelet

# Run all examples
./run_examples.sh --all

# Run tests
./run_examples.sh --test

# Show help
./run_examples.sh --help
```

## 📚 Examples Overview

| Example | Difficulty | Description | Key Features |
|---------|------------|-------------|--------------|
| **01_basic_analysis** | Beginner | Simple fingerprinting and comparison | Basic API, distance metrics |
| **02_authorship_attribution** | Intermediate | Author identification techniques | Classification, style analysis |
| **03_language_detection** | Intermediate | Multi-language text analysis | Cross-alphabet support |
| **04_wavelet_analysis** | Advanced | Complex kernels and phase analysis | Morlet wavelets, kernel banks |
| **05_batch_processing** | Intermediate | Large-scale text processing | Performance optimization |
| **06_performance_optimization** | Advanced | Benchmarking and parameter tuning | Memory profiling, speed analysis |

## ⚙️ Configuration System

Each example includes flexible YAML configuration:

```yaml
# config.yaml example
default_settings:
  alphabet: "en"
  horizon: 3
  normalize: "global"

profiles:
  fast:
    decay: "('exp', 1.0)"
    
  accurate:
    decay: "('morlet', {'omega': 3.0, 'sigma': 1.0})"
    
  research:
    bank: "morlet:omega=3.0;sigma=1.0 + gauss:mu=2.0;sigma=1.5"
    aggregate: "sum_abs"
```

### Pipeline Integration

```python
import yaml
from dfbi import fingerprint

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Simple integration function
def analyze_with_config(text, profile='default'):
    settings = config['profiles'][profile]
    return fingerprint(text, **settings)

# Usage
result = analyze_with_config("Your text", profile='accurate')
```

## 🔬 Mathematical Foundation

### Decay Functions

DFBI implements several mathematical decay functions:

1. **Exponential**: `f(d) = exp(-λ(d-1))`
2. **Gaussian**: `f(d) = exp(-((d-μ)²)/(2σ²))`
3. **Morlet Wavelet**: `f(d) = exp(-0.5(t/σ)²) * exp(iωt)`
4. **Mexican Hat**: `f(d) = (1-u²) * exp(-0.5u²)` where `u = t/σ`

### Kernel Banks

Combine multiple kernels for enhanced analysis:

```python
# Multi-kernel analysis
matrix = fingerprint(
    text,
    bank="morlet:omega=3;sigma=1 + mexican:sigma=1.2 + gauss:mu=2;sigma=1.5",
    aggregate="sum_abs"
)
```

## 📈 Performance Characteristics

| Configuration | Speed | Accuracy | Memory | Use Case |
|---------------|-------|----------|--------|----------|
| **Fast** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Real-time processing |
| **Balanced** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General purpose |
| **Accurate** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Research applications |
| **Research** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Maximum precision |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- NumPy >= 1.22
- Pandas >= 1.4

### Full Installation
```bash
# Clone or download the project
cd dfbi_lib_0_1_6_wave_kernels
pip install -e .

# Install example dependencies
pip install pyyaml matplotlib scikit-learn psutil

# Verify installation
python -c "import dfbi; print('DFBI installed successfully')"
```

### Docker Setup (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -e dfbi_lib_0_1_6_wave_kernels/
RUN pip install pyyaml matplotlib scikit-learn psutil

CMD ["python", "examples/01_basic_analysis/example.py"]
```

## 🌍 Multi-Language Support

### English Analysis
```python
from dfbi.alphabet import EN34

fp = fingerprint(text, alphabet=EN34, horizon=3)
```

### Russian Analysis
```python
from dfbi.alphabet import RU41

fp = fingerprint(text, alphabet=RU41, horizon=3)
```

### Cross-Language Comparison
```python
# Unified analysis using common alphabet
fp_en = fingerprint(english_text, alphabet=EN34)
fp_ru = fingerprint(russian_text, alphabet=EN34)  # Normalized
similarity = 1 - dist_cosine(fp_en, fp_ru)
```

## 🔍 Use Cases

### 1. Authorship Attribution
```python
# Build author profiles
author_profiles = {}
for author, texts in training_data.items():
    fps = [fingerprint(text, **config) for text in texts]
    author_profiles[author] = np.mean(fps, axis=0)

# Classify unknown text
unknown_fp = fingerprint(unknown_text, **config)
distances = {author: dist_cosine(unknown_fp, profile) 
            for author, profile in author_profiles.items()}
predicted_author = min(distances, key=distances.get)
```

### 2. Document Similarity
```python
# Compare document collections
docs = {'doc1': 'text1', 'doc2': 'text2', 'doc3': 'text3'}
fingerprints = batch_from_texts(docs, **config)

# Create similarity matrix
similarity_matrix = np.zeros((len(docs), len(docs)))
doc_ids = list(docs.keys())
for i, doc1 in enumerate(doc_ids):
    for j, doc2 in enumerate(doc_ids):
        similarity_matrix[i,j] = 1 - dist_cosine(
            fingerprints[doc1], fingerprints[doc2]
        )
```

### 3. Language Detection
```python
# Build language models
language_profiles = {}
for lang, texts in language_data.items():
    alphabet = RU41 if lang == 'russian' else EN34
    fps = [fingerprint(text, alphabet=alphabet, **config) for text in texts]
    language_profiles[lang] = np.mean(fps, axis=0)

# Detect language
def detect_language(text):
    best_lang, best_score = None, float('inf')
    for lang, profile in language_profiles.items():
        alphabet = RU41 if lang == 'russian' else EN34
        fp = fingerprint(text, alphabet=alphabet, **config)
        score = dist_cosine(fp, profile)
        if score < best_score:
            best_score, best_lang = score, lang
    return best_lang, 1 - best_score
```

## 🧪 Testing

```bash
# Run all tests
cd dfbi_lib_0_1_6_wave_kernels
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_fingerprints.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=dfbi --cov-report=html
```

## 📖 Documentation

- **English**: [docs/README_EN.md](docs/README_EN.md)
- **Russian**: [docs/README_RU.md](docs/README_RU.md)
- **Examples Guide**: [examples/README.md](examples/README.md)
- **API Reference**: See docstrings in source code

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Wavelet theory foundations
- Text analysis research community
- Open source contributors

## 📞 Support

- **Issues**: Use GitHub Issues for bug reports
- **Examples**: Check the `examples/` directory
- **Documentation**: See `docs/` directory
- **Tests**: Run the test suite for verification

---

**Ready to explore advanced text analysis?** Start with the interactive examples:

```bash
./run_examples.sh
```

Choose your path:
- **Beginner**: Start with Basic Analysis
- **Researcher**: Jump to Wavelet Analysis  
- **Developer**: Explore Performance Optimization
- **Curious**: Try Language Detection

Happy analyzing! 🚀