
"""
DFBI Fingerprinting Module
=========================

This module implements the core DFBI (Deterministic Finite-horizon Bigram Interference) 
fingerprinting algorithm for text analysis. DFBI creates numerical fingerprints of texts
using character pair statistics weighted by distance-dependent decay functions.

The algorithm works by:
1. Converting text to normalized character sequences
2. Extracting character pairs within a specified horizon
3. Weighting pairs by distance using decay functions (exponential, Gaussian, wavelets)
4. Aggregating into matrices that capture stylistic patterns

Key Features:
- Multiple decay functions (exponential, Gaussian, Morlet wavelet, Mexican hat)
- Kernel banks for multi-scale analysis
- Phase-sensitive complex analysis
- Flexible normalization and masking options
- Support for multiple alphabets and languages

Example:
    >>> from dfbi import fingerprint
    >>> from dfbi.alphabet import EN34
    >>> 
    >>> # Basic fingerprinting
    >>> text = "The quick brown fox jumps over the lazy dog"
    >>> fp = fingerprint(text, alphabet=EN34, horizon=3)
    >>> 
    >>> # Advanced wavelet analysis
    >>> fp = fingerprint(text, alphabet=EN34, horizon=5,
    ...                 decay=('morlet', {'omega': 3.0, 'sigma': 1.0}),
    ...                 normalize='row', mask='letters')
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .alphabet import Alphabet, RU41, RUS_LETTERS
from .utils import normalize_char
from .decay import get_decay
from .kernels import kernel_vector, parse_kernel_flag
from .phase import phase_vector

def iter_pairs_with_horizon(seq, h: int):
    """
    Generate character pairs within a specified horizon distance.
    
    This function iterates through a sequence and yields all character pairs
    (a, b, d) where 'a' and 'b' are characters separated by distance 'd',
    and d is within the specified horizon.
    
    Args:
        seq: Sequence of characters
        h: Maximum horizon distance
        
    Yields:
        Tuple[char, char, int]: (first_char, second_char, distance)
        
    Example:
        >>> list(iter_pairs_with_horizon("abc", 2))
        [('a', 'b', 1), ('a', 'c', 2), ('b', 'c', 1)]
    """
    n = len(seq)
    for i in range(n):
        a = seq[i]
        for d in range(1, h+1):
            j = i + d
            if j >= n: 
                break
            yield a, seq[j], d

def _mask_matrix(alpha: Alphabet, mode: Optional[str]) -> np.ndarray:
    """
    Create a masking matrix for selective character pair analysis.
    
    This function generates a binary mask matrix that can be used to focus
    analysis on specific types of character pairs (e.g., only letters,
    only punctuation, or all characters).
    
    Args:
        alpha: Alphabet object containing character symbols
        mode: Masking mode - 'letters', 'punct', 'none', or None
        
    Returns:
        np.ndarray: Binary mask matrix of shape (alphabet_size, alphabet_size)
        
    Raises:
        ValueError: If mode is not recognized
        
    Example:
        >>> from dfbi.alphabet import EN34
        >>> mask = _mask_matrix(EN34, 'letters')
        >>> # mask[i,j] = 1.0 only if both chars are letters
    """
    W = np.ones((len(alpha.symbols), len(alpha.symbols)), dtype=np.float64)
    if not mode or mode == "none":
        return W
    
    letters = set(RUS_LETTERS)  # Note: This works for both RU and EN alphabets
    
    for i, a in enumerate(alpha.symbols):
        for j, b in enumerate(alpha.symbols):
            if mode == "letters":
                # Only letter-letter pairs
                W[i, j] = 1.0 if (a in letters and b in letters) else 0.0
            elif mode == "punct":
                # Only pairs involving punctuation
                W[i, j] = 1.0 if (a not in letters or b not in letters) else 0.0
            else:
                raise ValueError(f"Unknown mask mode: {mode}. Use 'letters', 'punct', or 'none'")
    return W

def fingerprint(text: str,
                alphabet: Alphabet = RU41,
                horizon: int = 1,
                decay = ('exp', 0.7),
                kernel: Optional[str] = None,
                bank: Optional[str] = None,
                aggregate: Optional[str] = None,
                mask: Optional[str] = None,
                normalize: str = "global",
                transform: Optional[str] = None,
                return_tensor: bool = False,
                dtype=np.float64,
                phase=None):
    """
    Generate a DFBI fingerprint matrix for the given text.
    
    This is the main function for creating DFBI fingerprints. It analyzes character
    pair statistics within a specified horizon, applies distance-dependent weighting
    using decay functions, and returns a numerical matrix that captures the text's
    stylistic patterns.
    
    Args:
        text (str): Input text to analyze
        alphabet (Alphabet): Character alphabet to use (default: RU41 for Russian)
        horizon (int): Maximum distance for character pairs (default: 1)
        decay (tuple): Decay function specification, e.g., ('exp', 0.7) or 
                      ('morlet', {'omega': 3.0, 'sigma': 1.0}) (default: ('exp', 0.7))
        kernel (str, optional): Alternative kernel specification (deprecated, use decay)
        bank (str, optional): Kernel bank specification for multi-scale analysis,
                              e.g., "morlet:omega=3.0,sigma=1.0 + gauss:mu=2.0,sigma=1.5"
        aggregate (str, optional): Aggregation method for kernel banks ('sum_abs', etc.)
        mask (str, optional): Character masking mode ('letters', 'punct', 'none')
        normalize (str): Normalization method ('global', 'row') (default: 'global')
        transform (str, optional): Post-processing transform ('sqrt', 'log1p')
        return_tensor (bool): If True, return full 3D tensor instead of aggregated matrix
        dtype: NumPy data type for computations (default: np.float64)
        phase: Phase analysis specification for complex wavelets
        
    Returns:
        np.ndarray: Fingerprint matrix of shape (alphabet_size, alphabet_size) or
                   (horizon, alphabet_size, alphabet_size) if return_tensor=True
                   
    Raises:
        ValueError: If parameters are invalid or incompatible
        
    Examples:
        Basic usage:
        >>> from dfbi import fingerprint
        >>> from dfbi.alphabet import EN34
        >>> 
        >>> text = "The quick brown fox"
        >>> fp = fingerprint(text, alphabet=EN34, horizon=3)
        >>> print(fp.shape)  # (34, 34)
        
        Advanced wavelet analysis:
        >>> fp = fingerprint(text, alphabet=EN34, horizon=5,
        ...                 decay=('morlet', {'omega': 3.0, 'sigma': 1.0}),
        ...                 normalize='row', mask='letters')
        
        Multi-kernel analysis:
        >>> fp = fingerprint(text, alphabet=EN34, horizon=4,
        ...                 bank="morlet:omega=3.0,sigma=1.0 + gauss:mu=2.0,sigma=1.5",
        ...                 aggregate='sum_abs')
        
        Complex phase analysis:
        >>> fp = fingerprint(text, alphabet=EN34, horizon=6,
        ...                 decay=('morlet', {'omega': 4.0, 'sigma': 1.2}),
        ...                 phase=('entropy', 0.1, 3.14159))
    
    Mathematical Background:
        The DFBI algorithm computes weighted character pair statistics:
        
        M[i,j] = Σ_{d=1}^h w(d) * count(char_i, char_j, distance=d)
        
        where w(d) is the decay function weight at distance d.
        
        Common decay functions:
        - Exponential: w(d) = exp(-λ(d-1))
        - Gaussian: w(d) = exp(-((d-μ)²)/(2σ²))
        - Morlet: w(d) = exp(-0.5(t/σ)²) * exp(iωt) (complex)
        - Mexican Hat: w(d) = (1-u²) * exp(-0.5u²)
    """
    h = max(1, int(horizon))
    A = len(alphabet.symbols)
    if not text:
        if return_tensor:
            return np.zeros((h, A, A), dtype=dtype)
        return np.zeros((A, A), dtype=dtype)

    s = [normalize_char(c, alphabet) for c in text]
    T = np.zeros((h, A, A), dtype=dtype)
    for a,b,d in iter_pairs_with_horizon(s, h):
        ia = alphabet.index[a]; jb = alphabet.index[b]
        T[d-1, ia, jb] += 1.0
    if return_tensor:
        return T

    phase_vec = phase_vector(phase, s, alphabet, h) if phase is not None else None

    if bank:
        bank_specs = parse_kernel_flag(bank, h)
        Ms = []
        for kind, params in bank_specs:
            wv = kernel_vector(kind, params, h)
            if phase_vec is not None:
                wv = wv * phase_vec
            wv = wv[:, None, None]
            M = (wv * T).sum(axis=0)
            Ms.append(M)
        Mstack = np.stack(Ms, axis=0)
        if aggregate == "sum_abs":
            M = np.abs(Mstack).sum(axis=0)
        else:
            M = Mstack
    else:
        w = get_decay(decay)
        values = [w(d) for d in range(1, h+1)]
        is_complex = any(isinstance(v, complex) for v in values)
        weights = np.array(values, dtype=np.complex128 if is_complex else dtype)
        if phase_vec is not None:
            weights = weights * phase_vec
        weights = weights[:, None, None]
        M = (weights * T).sum(axis=0)

    def _apply_transform(X, mode):
        if not mode:
            return X
        if mode == "sqrt":
            if np.iscomplexobj(X):
                mag = np.sqrt(np.abs(X))
                phase = np.exp(1j * np.angle(X))
                return mag * phase
            return np.sign(X) * np.sqrt(np.abs(X))
        if mode == "log1p":
            if np.iscomplexobj(X):
                mag = np.log1p(np.abs(X))
                phase = np.exp(1j * np.angle(X))
                return mag * phase
            return np.sign(X) * np.log1p(np.abs(X))
        raise ValueError("Unsupported transform mode")

    def _apply_mask_and_norm(X):
        W = _mask_matrix(alphabet, mask) if mask else None
        if W is not None:
            X = X * W
        epsilon = np.array(1e-6, dtype=X.dtype)
        if W is not None:
            X = X + epsilon * W
        else:
            X = X + epsilon
        if X.shape[0] == X.shape[1]:
            if np.iscomplexobj(X):
                X = 0.5 * (X + X.conj().T)
            else:
                X = 0.5 * (X + X.T)
        X = _apply_transform(X, transform)
        for _ in range(2):
            rs = np.abs(X).sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            X = X / rs
            cs = np.abs(X).sum(axis=0, keepdims=True)
            cs[cs == 0] = 1.0
            X = X / cs
        if normalize == "global":
            if np.iscomplexobj(X):
                s = np.abs(X).sum()
            else:
                s = X.sum()
            X = X / s if s > 0 else X
        elif normalize == "row":
            if np.iscomplexobj(X):
                rs = np.abs(X).sum(axis=1, keepdims=True)
            else:
                rs = X.sum(axis=1, keepdims=True)
            rs[rs==0] = 1.0
            X = X / rs
        else:
            raise ValueError("normalize must be 'global' or 'row'")
        return X

    if M.ndim == 2:
        M = _apply_mask_and_norm(M)
    else:
        if aggregate != "sum_abs":
            M = np.stack([_apply_mask_and_norm(m) for m in M], axis=0)
        else:
            M = _apply_mask_and_norm(M)

    return M

def batch_from_texts(texts: Dict[str, str], **kwargs) -> Dict[str, np.ndarray]:
    return {name: fingerprint(txt, **kwargs) for name, txt in texts.items()}

def window_scan(text: str, win: int = 500, step: int = 250, **kwargs) -> List[np.ndarray]:
    chunks: List[np.ndarray] = []
    t = text.strip()
    if len(t) <= win:
        return [fingerprint(t, **kwargs)]
    for i in range(0, max(1, len(t) - win + 1), step):
        chunks.append(fingerprint(t[i:i+win], **kwargs))
    return chunks
