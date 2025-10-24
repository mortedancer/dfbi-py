
# DFBI 0.1.6 - Wave Kernels

- Wavelet-inspired decays: `gauss`, `morlet`, `mexican` + kernel banks with optional `aggregate=sum_abs`
- Complex-aware metrics: `cosine`, `l2_multi` work with multi-channel or phased signatures
- CLI flags: `--kernel`, `--bank`, `--aggregate`, `--phase`

## CLI quick-start

```bash
# Russian corpus, Morlet decay + entropy-driven phase
dfbi-cli fingerprint data/ru_sample.txt \
  --alphabet ru \
  --horizon 5 \
  --decay "('morlet', {'omega': 3.2, 'sigma': 1.1})" \
  --phase "('entropy', 0.1, 3.1415)" \
  --mask letters \
  --dump out/ru

# English corpus, Mexican-hat kernel bank with constant phase
dfbi-cli fingerprint data/en_sample.txt \
  --alphabet en \
  --bank "mexican:sigma=1.4,scale=2.0" \
  --aggregate sum_abs \
  --phase "('theta', 0.5)" \
  --normalize row
```

Tip: combine `--bank` with multiple entries (e.g. `"morlet:omega=3;sigma=1;scale=1.0;shift=0.0; mexh:sigma=1.2"`) to probe several horizons in one pass.
