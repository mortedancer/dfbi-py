
#!/usr/bin/env python3
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml

from dfbi import fingerprint
from dfbi.alphabet import EN34
from viz.plots import plot_heatmap, plot_bars

DATASET_PATH = Path.home()/".cache/kagglehub/datasets/rtatman/blog-authorship-corpus/versions/2/blogtext.csv"
CORPUS_ROOT = Path("corpus_blogs")
CONFIG_PATH = Path("configs/bench_presets.yaml")

@dataclass(frozen=True)
class TopicConfig:
    topics: Iterable[str]
    docs_per_topic: int
    min_characters: int

def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        cfg=yaml.safe_load(stream)
    if "baseline_exp" in cfg and "decay" in cfg["baseline_exp"]:
        d = cfg["baseline_exp"]["decay"]
        if isinstance(d, list) and d:
            cfg["baseline_exp"]["decay"]=tuple(d)
    return cfg

def ensure_local_corpus(dataset_path: Path, cfg: TopicConfig, rng_seed: int = 7) -> Dict[str, List[str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
    df = pd.read_csv(dataset_path)
    if "topic" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV expected 'topic' and 'text' columns")
    active = [t for t in cfg.topics if (df['topic']==t).any()]
    if not active: raise ValueError("No requested topics found")
    random.seed(rng_seed)
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    topic_texts=defaultdict(list)
    for t in active:
        matches = df[df['topic']==t]
        sel = matches.sample(n=min(cfg.docs_per_topic,len(matches)), random_state=rng_seed)
        k=0
        d = CORPUS_ROOT / t; d.mkdir(parents=True, exist_ok=True)
        for _,row in sel.iterrows():
            tx = str(row['text']).strip()
            if len(tx) < cfg.min_characters: continue
            (d / f"{t}_{k:05d}.txt").write_text(tx, encoding="utf-8")
            topic_texts[t].append(tx); k+=1
    topic_texts = {t:xs for t,xs in topic_texts.items() if len(xs)>=5}
    if len(topic_texts)<2:
        raise RuntimeError("Need >=2 topics with >=5 docs each.")
    return topic_texts

def compute_fps(topic_texts: Dict[str, List[str]], fp_kwargs: dict) -> Dict[str, List[np.ndarray]]:
    out=defaultdict(list)
    for t, texts in topic_texts.items():
        for tx in texts:
            try:
                out[t].append(fingerprint(tx, alphabet=EN34, **fp_kwargs))
            except Exception as e:
                print("[WARN] fingerprint failed:", e)
    return out

def cosine(a,b):
    a=a.reshape(-1); b=b.reshape(-1)
    num=float(np.dot(a,b)); na=float(np.sqrt(np.dot(a,a))+1e-12); nb=float(np.sqrt(np.dot(b,b))+1e-12)
    return 1.0 - (num/(na*nb))

def centroid_dist(fps: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    topics=list(fps.keys()); n=len(topics)
    C={t: np.mean(np.stack(xs,axis=0),axis=0) for t,xs in fps.items()}
    M=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            d=cosine(C[topics[i]], C[topics[j]]); M[i,j]=M[j,i]=d
    return pd.DataFrame(M, index=topics, columns=topics)

def within_var(fps: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    rows=[]
    for t,xs in fps.items():
        if len(xs)<=1: continue
        C=np.mean(np.stack(xs,axis=0),axis=0)
        dev=[cosine(C,x) for x in xs]
        rows.append((t,float(np.mean(dev)), float(np.std(dev))))
    return pd.DataFrame(rows, columns=["topic","mean_dev","std_dev"]).sort_values("mean_dev")

def loocv_acc(fps: Dict[str, List[np.ndarray]]) -> float:
    items=[]
    for t,xs in fps.items():
        for x in xs: items.append((t,x))
    if not items: return 0.0
    correct=0
    for true_t,x in items:
        best_t=None; best_d=1e9
        for t,xs in fps.items():
            if t==true_t and len(xs)>1:
                C=(np.sum(np.stack(xs,axis=0),axis=0)-x)/(len(xs)-1)
            else:
                C=np.mean(np.stack(xs,axis=0),axis=0)
            d=cosine(C,x)
            if d<best_d: best_d=d; best_t=t
        if best_t==true_t: correct+=1
    return correct/len(items)

def main():
    cfg=load_config()
    topic_cfg=TopicConfig(
        topics=("Arts","Education","Engineering","indUnk","Internet","Law","Non-Profit","Student","Technology","Communication-Media"),
        docs_per_topic=180, min_characters=400)
    topic_texts=ensure_local_corpus(DATASET_PATH, topic_cfg)

    configs={"baseline_exp":cfg["baseline_exp"], "single_gabor":cfg["single_gabor"], "multi_scale":cfg["multi_scale"]}
    Path("out").mkdir(parents=True, exist_ok=True)
    summary=[]
    for name, fp_kwargs in configs.items():
        fps=compute_fps(topic_texts, fp_kwargs)
        fps={t:xs for t,xs in fps.items() if xs}
        if len(fps)<2: continue
        D=centroid_dist(fps); D.to_csv(f"out/{name}_centroid_cosine.csv", index=True, encoding="utf-8")
        plot_heatmap(D, title=f"Centroid Cosine Distances — {name}", outpath=f"out/{name}_centroid_heatmap.png")
        var=within_var(fps); var.to_csv(f"out/{name}_within_var.csv", index=False, encoding="utf-8")
        acc=loocv_acc(fps); summary.append((name, acc))
        print(f"{name}: LOOCV={acc:.4f}")
    if summary:
        labels=[s[0] for s in summary]; vals=[s[1]*100 for s in summary]
        plot_bars(labels, vals, title="LOOCV Accuracy by DFBI config", ylabel="Accuracy (%)", outpath="out/loocv_by_config.png")
        pd.DataFrame(summary, columns=["config","loocv_acc"]).to_csv("out/summary_loocv.csv", index=False, encoding="utf-8")
    print("Done. See ./out")

if __name__ == "__main__":
    main()
