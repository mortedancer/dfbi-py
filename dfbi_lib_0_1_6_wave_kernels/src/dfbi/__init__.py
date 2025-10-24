"""
DFBI - Digital Fingerprinting for Behavioral Identification
==========================================================

A Python library for text analysis using digital fingerprinting techniques.
"""

from .fingerprints import fingerprint, batch_from_texts, window_scan
from .alphabet import Alphabet, RU41, EN34, RUS_LETTERS
from .metrics import dist_l2, dist_l1, dist_chi2, dist_cosine, dist_l2_multi
from .decay import get_decay
from .kernels import kernel_vector, parse_kernel_flag
from .phase import phase_vector
from .utils import normalize_char

__version__ = "0.1.6"

__all__ = [
    # Core fingerprinting functions
    'fingerprint',
    'batch_from_texts', 
    'window_scan',
    
    # Alphabets
    'Alphabet',
    'RU41',
    'EN34', 
    'RUS_LETTERS',
    
    # Distance metrics
    'dist_l2',
    'dist_l1', 
    'dist_chi2',
    'dist_cosine',
    'dist_l2_multi',
    
    # Utility functions
    'get_decay',
    'kernel_vector',
    'parse_kernel_flag',
    'phase_vector',
    'normalize_char',
]