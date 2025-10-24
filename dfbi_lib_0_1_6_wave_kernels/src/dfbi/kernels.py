
import math
from typing import List, Tuple

def parse_kernel_flag(flag: str, h: int) -> List[Tuple[str, list]]:
    if not flag: return []
    # Split by '+' first to get individual kernels
    kernel_parts = [p.strip() for p in flag.split('+') if p.strip()]
    bank = []
    for kernel_spec in kernel_parts:
        # Split by ':' to separate name from parameters
        name, _, rest = kernel_spec.partition(':')
        name = name.strip()
        rest = rest.strip()
        params = {}
        if rest:
            # Split parameters by either ';' or ',' 
            param_parts = rest.replace(';', ',').split(',')
            for kv in param_parts:
                if not kv.strip(): continue
                if '=' not in kv:
                    continue
                k, v = kv.split('=', 1)
                params[k.strip()] = float(v.strip())
        if name == 'exp':
            lam = float(params.get('lambda', 0.7))
            bank.append(('exp', [lam]))
        elif name in ('inv','lin'):
            pw = float(params.get('p', 1.0))
            bank.append(('inv', [pw]))
        elif name == 'gauss':
            mu = float(params.get('mu', 2.0))
            sig= float(params.get('sigma', 1.0))
            bank.append(('gauss', [mu, sig]))
        elif name == 'gabor':
            om = float(params.get('omega', 1.0))
            sig= float(params.get('sigma', 1.0))
            bank.append(('gabor', [om, sig]))
        elif name in ('morlet','morl'):
            om = float(params.get('omega', 5.0))
            sig= float(params.get('sigma', 1.0))
            scale = float(params.get('scale', 1.0))
            shift = float(params.get('shift', 0.0))
            bank.append(('morlet', [om, sig, scale, shift]))
        elif name in ('mexican','mexh'):
            sig= float(params.get('sigma', 1.0))
            scale = float(params.get('scale', 1.0))
            shift = float(params.get('shift', 0.0))
            bank.append(('mexican', [sig, scale, shift]))
        else:
            raise ValueError(f"Unknown kernel name: {name}")
    return bank

def kernel_vector(kind: str, params: list, h: int):
    import numpy as np
    d = np.arange(1, h+1, dtype=np.float64)
    if kind == 'exp':
        lam = params[0]
        return np.exp(-lam*(d-1))
    if kind == 'inv':
        p = params[0]
        return 1.0/(d**p)
    if kind == 'gauss':
        mu, sig = params
        return np.exp(-((d-mu)**2)/(2.0*sig*sig))
    if kind == 'gabor':
        om, sig = params
        return np.cos(om*d) * np.exp(-((d-1.0)**2)/(2.0*sig*sig))
    if kind == 'morlet':
        om, sig, scale, shift = params
        sig = max(sig, 1e-9)
        scale = max(scale, 1e-9)
        t = ((d - 1.0) - shift) / scale
        envelope = np.exp(-0.5 * (t / sig) ** 2)
        return envelope * np.exp(1j * om * t)
    if kind == 'mexican':
        sig, scale, shift = params
        sig = max(sig, 1e-9)
        scale = max(scale, 1e-9)
        t = ((d - 1.0) - shift) / scale
        u = t / sig
        return (1.0 - u * u) * np.exp(-0.5 * u * u)
    raise ValueError(f"Unknown kernel kind: {kind}")
