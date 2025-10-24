
import argparse, os, glob, pandas as pd
from .fingerprints import fingerprint, window_scan
from .metrics import dist_l1, dist_l2, dist_chi2, dist_l2_multi, dist_cosine

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def alpha_from_flag(flag):
    from .alphabet import RU41, EN34
    return RU41 if flag=="ru" else EN34

def cmd_fingerprint(args):
    text = read_text(args.input)
    M = fingerprint(text, alphabet=alpha_from_flag(args.alphabet),
                    horizon=args.horizon, decay=eval(args.decay),
                    kernel=args.kernel, bank=args.bank, aggregate=args.aggregate,
                    mask=args.mask, normalize=args.normalize,
                    phase=eval(args.phase))
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        out = os.path.join(args.dump, os.path.basename(args.input) + "_matrix.npy")
        import numpy as np; np.save(out, M); print(out)
    else:
        print(getattr(M, "shape", None))

def cmd_compare(args):
    paths = []
    for pattern in args.inputs: paths.extend(glob.glob(pattern))
    mats = []
    for p in paths:
        text = read_text(p)
        M = fingerprint(text, alphabet=alpha_from_flag(args.alphabet),
                        horizon=args.horizon, decay=eval(args.decay),
                        kernel=args.kernel, bank=args.bank, aggregate=args.aggregate,
                        mask=args.mask, normalize=args.normalize,
                        phase=eval(args.phase))
        mats.append((os.path.basename(p), M))
    metric = {"l1":dist_l1,"l2":dist_l2,"chi2":dist_chi2,"l2_multi":dist_l2_multi,"cosine":dist_cosine}[args.metric]
    print("name_i,name_j,distance")
    for i in range(len(mats)):
        for j in range(i+1, len(mats)):
            d = metric(mats[i][1], mats[j][1])
            print(f"{mats[i][0]},{mats[j][0]},{d:.6f}")

def cmd_bench(args):
    from .bench import load_folder_corpus, run_grid
    from pathlib import Path
    corpus = load_folder_corpus(Path(args.root))
    if not corpus: raise SystemExit(f"No authors with .txt files found under: {args.root}")
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    metrics  = [x.strip() for x in args.metrics.split(",") if x.strip()]
    masks    = [x.strip() for x in args.masks.split(",") if x.strip()]
    results = run_grid(corpus, horizons=horizons, metrics=metrics, masks=masks,
                       normalize=args.normalize, decay=eval(args.decay),
                       kernel=args.kernel, bank=args.bank, aggregate=args.aggregate,
                       phase=eval(args.phase))
    print("horizon,mask,metric,throughput_MBps,loocv_acc,kernel,bank,aggregate,phase")
    for r in results:
        print(f"{r['horizon']},{r['mask']},{r['metric']},{r['throughput_MBps']:.2f},{r['loocv_acc']*100:.2f},{r['kernel']},{r['bank']},{r['aggregate']},{r.get('phase','none')}")
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        pd.DataFrame(results).to_csv(args.csv, index=False, encoding="utf-8")

def main():
    p = argparse.ArgumentParser(prog="dfbi-cli", description="DFBI with wave kernels")
    sub = p.add_subparsers(required=True)

    p1 = sub.add_parser("fingerprint", help="Compute signature matrix")
    p1.add_argument("input")
    p1.add_argument("--horizon", type=int, default=1)
    p1.add_argument("--decay", default="('exp', 0.7)")
    p1.add_argument("--kernel", default="")
    p1.add_argument("--bank", default="")
    p1.add_argument("--aggregate", choices=["sum_abs","none",""], default="")
    p1.add_argument("--mask", choices=["none","letters","punct"], default="none")
    p1.add_argument("--normalize", choices=["global","row"], default="global")
    p1.add_argument("--phase", default="None")
    p1.add_argument("--dump")
    p1.add_argument("--alphabet", choices=["ru","en"], default="ru")
    p1.set_defaults(func=cmd_fingerprint)

    p2 = sub.add_parser("compare", help="Compare multiple texts")
    p2.add_argument("inputs", nargs="+")
    p2.add_argument("--horizon", type=int, default=1)
    p2.add_argument("--decay", default="('exp', 0.7)")
    p2.add_argument("--kernel", default="")
    p2.add_argument("--bank", default="")
    p2.add_argument("--aggregate", choices=["sum_abs","none",""], default="")
    p2.add_argument("--mask", choices=["none","letters","punct"], default="none")
    p2.add_argument("--normalize", choices=["global","row"], default="global")
    p2.add_argument("--metric", choices=["l1","l2","chi2","l2_multi","cosine"], default="l2")
    p2.add_argument("--alphabet", choices=["ru","en"], default="ru")
    p2.add_argument("--phase", default="None")
    p2.set_defaults(func=cmd_compare)

    p4 = sub.add_parser("bench", help="Grid benchmark")
    p4.add_argument("root")
    p4.add_argument("--horizons", default="1,2,3")
    p4.add_argument("--metrics", default="l1,l2,chi2,cosine")
    p4.add_argument("--masks", default="letters")
    p4.add_argument("--normalize", choices=["global","row"], default="global")
    p4.add_argument("--decay", default="('exp', 0.7)")
    p4.add_argument("--kernel", default="")
    p4.add_argument("--bank", default="")
    p4.add_argument("--aggregate", choices=["sum_abs","none",""], default="")
    p4.add_argument("--phase", default="None")
    p4.add_argument("--csv")
    p4.add_argument("--alphabet", choices=["ru","en"], default="ru")
    p4.set_defaults(func=cmd_bench)

    args = p.parse_args()
    args.func(args)
