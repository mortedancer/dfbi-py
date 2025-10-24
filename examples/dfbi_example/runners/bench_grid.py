
import argparse, os, yaml
import pandas as pd
from pathlib import Path
from dfbi.bench import load_folder_corpus, run_grid
from viz.plots import plot_bars

def _coerce(node):
    if isinstance(node, list) and node and isinstance(node[0], str):
        return tuple(node)
    return node

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--alphabet", choices=["ru","en"], default="en")
    ap.add_argument("--horizons", default="1,2,3,4,6")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outfig", default="out/")
    args = ap.parse_args()

    with open(args.configs, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    corpus = load_folder_corpus(Path(args.root))
    if not corpus: raise SystemExit("No corpus under: "+args.root)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    metrics  = cfg["grid_sweep"]["metrics"]
    masks    = cfg["grid_sweep"]["masks"]
    runs     = cfg["grid_sweep"]["runs"]

    all_rows=[]
    for r in runs:
        name=r["name"]; fp=r["fp"]
        for h in horizons:
            fp_kwargs = dict(fp)
            if fp_kwargs.get("horizon", 0) == 0:
                fp_kwargs["horizon"] = h
            if "decay" in fp_kwargs:
                fp_kwargs["decay"] = _coerce(fp_kwargs["decay"])
            res = run_grid(corpus, horizons=[fp_kwargs["horizon"]], metrics=metrics, masks=masks,
                           normalize=fp_kwargs.get("normalize","global"),
                           decay=fp_kwargs.get("decay",("exp",0.7)),
                           kernel=fp_kwargs.get("kernel",""),
                           bank=fp_kwargs.get("bank",""),
                           aggregate=fp_kwargs.get("aggregate",""))
            for row in res:
                row["run"]=name; all_rows.append(row)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    df.to_csv(args.csv, index=False, encoding="utf-8")

    best = df.groupby("run")["loocv_acc"].max().reset_index()
    plot_bars(best["run"].tolist(), (best["loocv_acc"]*100.0).tolist(),
              title="Best LOOCV accuracy by DFBI configuration", ylabel="Accuracy (%)",
              outpath=os.path.join(args.outfig, "best_accuracy_by_run.png"))
    print("Saved:", args.csv)

if __name__ == "__main__":
    main()
