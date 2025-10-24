
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_heatmap(df: pd.DataFrame, title: str, outpath: str = None):
    plt.figure()
    im = plt.imshow(df.values, aspect='auto')
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.index)), df.index)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close()

def plot_bars(labels, values, title: str, ylabel: str, outpath: str = None):
    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close()
