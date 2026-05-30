# sae-feature-death

Code for **"On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders"** (ICML 2026). [arxiv.org/abs/2605.31518](https://arxiv.org/abs/2605.31518)

## Measure outlier severity on your activations

```python
from metrics import compute_gamma, effective_rank, geometric_median

# acts: [N, d] numpy array
compute_gamma(acts)        # outlier severity (higher = more dead features)
effective_rank(acts)       # how many dimensions carry variance
geometric_median(acts)     # centering vector for the SAE's pre-encoder bias
```

## Synthetic demo

[`demo.ipynb`](demo.ipynb) walks through the death mechanism on toy data: generate high-gamma activations, measure death by pathway (ReLU vs TopK), and show how centering and PCA whitening can eliminate it.

## Reproduce key figures

```bash
python reproduction/plot_figures.py   # generates Figures 2a, 4b, S11
```

Precomputed statistics for 454 model-layers (19 models) and 14 trained SAE runs are in `reproduction/`.

## Citation

```bibtex
@inproceedings{simon2026on,
    title={On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders},
    author={Simon, Elana and Adams, Etowah and Zou, James},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://arxiv.org/abs/2605.31518}
}
```
