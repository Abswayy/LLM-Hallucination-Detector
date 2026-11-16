### main

| Method | TriviaQA AUROC [95% CI] | +ms | PubMedQA AUROC [95% CI] | +ms |
|---|---|---|---|---|
| **Answer-token probe (ours)** | **0.881** [0.852, 0.908] | 0.8 | **0.839** [0.800, 0.873] | 1.8 |
| **Prompt-only probe (ours)** | 0.821 [0.781, 0.858] | (6.8) | 0.830 [0.791, 0.864] | (57.6) |
| Token log-prob (mean) | 0.837 [0.802, 0.870] | 0.0 | 0.731 [0.685, 0.775] | 0.0 |
| Answer identity (control) | — | — | 0.753 [0.708, 0.794] | 0.0 |
| Self-consistency, 5 samples | 0.823 [0.787, 0.858] | 87.6 | 0.669 [0.624, 0.712] | 311.5 |
| Self-consistency, 10 samples | 0.845 [0.809, 0.878] | 149.7 | 0.698 [0.649, 0.741] | 588.9 |
| Semantic entropy, 5 samples | 0.817 [0.782, 0.854] | 169.2 | 0.671 [0.624, 0.715] | 327.3 |
| Semantic entropy, 10 samples | 0.833 [0.799, 0.867] | 337.0 | 0.701 [0.652, 0.745] | 610.9 |
| Semantic entropy, 10 samples, lenient | 0.835 [0.802, 0.869] | 342.6 | 0.702 [0.655, 0.747] | 611.7 |

### transfer

| Train → test | Prompt-only probe | Answer-token probe |
|---|---|---|
| TriviaQA → TriviaQA (in-domain) | 0.837 [0.794, 0.882] | 0.909 [0.875, 0.938] |
| PubMedQA → PubMedQA (in-domain) | 0.827 [0.751, 0.896] | 0.828 [0.747, 0.898] |
| TriviaQA → PubMedQA | 0.741 [0.653, 0.823] | 0.540 [0.448, 0.637] |
| PubMedQA → TriviaQA | 0.496 [0.429, 0.561] | 0.577 [0.515, 0.641] |

### sep

| Setting | Token | Accuracy probe | SEP | Δ (accuracy − SEP) [95% CI] |
|---|---|---|---|---|
| TriviaQA (in-domain) | prompt | 0.772 | 0.759 | +0.013 [-0.021, +0.052] |
| TriviaQA (in-domain) | answer | 0.821 | 0.784 | +0.037 [-0.003, +0.079] |
| PubMedQA (in-domain) | prompt | 0.793 | 0.752 | +0.041 [+0.007, +0.075] |
| PubMedQA (in-domain) | answer | 0.842 | 0.754 | +0.088 [+0.046, +0.130] |
| TriviaQA → PubMedQA | prompt | 0.533 | 0.599 | -0.066 [-0.123, -0.010] |
| TriviaQA → PubMedQA | answer | 0.743 | 0.642 | +0.101 [+0.063, +0.142] |
| PubMedQA → TriviaQA | prompt | 0.469 | 0.473 | -0.004 [-0.066, +0.059] |
| PubMedQA → TriviaQA | answer | 0.612 | 0.721 | -0.108 [-0.174, -0.040] |

### scale

| | TriviaQA 3B | TriviaQA 7B (4-bit) | PubMedQA 3B | PubMedQA 7B (4-bit) |
|---|---|---|---|---|
| Model accuracy | 42.6% | 53.2% | 61.6% | 62.1% |
| **Answer-token probe (ours)** | 0.881 | 0.889 | 0.839 | 0.870 |
| **Prompt-only probe (ours)** | 0.821 | 0.824 | 0.830 | 0.852 |
| Token log-prob (mean) | 0.837 | 0.846 | 0.731 | 0.519 |
| Self-consistency, 10 samples | 0.845 | 0.840 | 0.698 | 0.616 |
| Semantic entropy, 10 samples | 0.833 | 0.830 | 0.701 | 0.615 |
