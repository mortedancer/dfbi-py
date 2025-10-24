
# DFBI Extended Examples v0.1.7

This bundle extends your complex Blog Authorship DFBI example with:
- Multi-config benchmarking and LOOCV accuracy
- Visualizations: centroid distance heatmaps, accuracy bars, variability
- Optional second dataset: 20 Newsgroups (topic attribution)
- Generic folder corpus runner

## Blog Authorship quick start
python -m dfbi.cli prep-blogs --out ./corpus_blogs --group-by topic --top-k 10 --per-class 500 --min-chars 300
python complex_blogs_dfbi_extended.py
python runners/bench_grid.py --root ./corpus_blogs --alphabet en --horizons 1,2,3,4,6 --configs configs/bench_presets.yaml --csv out/grid_results.csv --outfig out/
