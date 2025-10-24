import matplotlib.pyplot as plt
import numpy as np

def heatmap(M: np.ndarray, title: str = None):
    plt.figure(figsize=(6,6))
    plt.imshow(M, interpolation='nearest')
    if title: plt.title(title)
    plt.xlabel("next symbol index")
    plt.ylabel("symbol index")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
