# Headless sweep for DFBI
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dfbi.bench import load_folder_corpus, run_grid

def plot_accuracy(df, out_png):
    plt.figure(figsize=(6,4))
    for m in sorted(df['metric'].unique()):
        d = df[df['metric']==m].sort_values('horizon')
        plt.plot(d['horizon'], d['loocv_acc']*100, label=m)
    plt.xlabel("Horizon h")
    plt.ylabel("LOOCV accuracy, %")
    plt.title("Accuracy vs horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()

def plot_throughput(df, out_png):
    plt.figure(figsize=(6,4))
    for m in sorted(df['metric'].unique()):
        d = df[df['metric']==m].sort_values('horizon')
        plt.plot(d['horizon'], d['throughput_MBps'], label=m)
    plt.xlabel("Horizon h")
    plt.ylabel("Throughput (MB/s)")
    plt.title("Throughput vs horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--alphabet", choices=["ru","en"], default="en")
    ap.add_argument("--horizons", default="1,2,3,4,5,6")
    ap.add_argument("--metrics", default="l1,l2,chi2")
    ap.add_argument("--masks", default="letters")
    ap.add_argument("--normalize", choices=["global","row"], default="global")
    ap.add_argument("--decay", default="('exp',0.7)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    corpus = load_folder_corpus(Path(args.corpus))
    if not corpus:
        raise SystemExit(f"No author/*.txt found under {args.corpus}")

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    metrics  = [x.strip() for x in args.metrics.split(",") if x.strip()]
    masks    = [x.strip() for x in args.masks.split(",") if x.strip()]

    res = run_grid(corpus, horizons=horizons, metrics=metrics, masks=masks,
                   normalize=args.normalize, decay=eval(args.decay))
    df = pd.DataFrame(res)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/"grid_results.csv", index=False)

    for mask in sorted(df['mask'].unique()):
        dfx = df[df['mask']==mask]
        plot_accuracy(dfx, out/f"acc_vs_h_{mask}.png")
        plot_throughput(dfx, out/f"thr_vs_h_{mask}.png")

if __name__ == "__main__":
    main()
