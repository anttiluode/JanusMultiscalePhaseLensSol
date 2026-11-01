# Experiment 001 — Frozen complex component fitting, negative result

**Date:** 2026-11-01. **Outcome:** No performance advantage from learned complex phase fitting in this configuration.

Frozen before held-out evaluation: 32 synthetic training scenes, 8 distinct validation scenes, 24 distinct familiar test scenes, 12 distinct unrelated test scenes. Image size 64×64 grayscale. Additive Gaussian noise levels σ=0.12 and 0.22 with independent per-scene RNG draws. 17-channel exactly reconstructing complex directional frame. 24 complex prototype neighborhoods, 8×8 patches at stride 4. Prototype training uses 1,000 clean training patches, eight phase-normalized clustering iterations, seed 22.

Validation at σ=0.17 selected Gaussian blur σ=1.3 pixels, coefficient shrinkage threshold 0.06, and observation blend 0.75 for **all** dictionary arms. No tuning on the listed tests. These parameter searches use fixed small grids and are not optimized broad hyperparameter sweeps.

## Mean MSE (lower is better)

| Method | Familiar .12 | Familiar .22 | Unrelated .12 | Unrelated .22 |
|---|---:|---:|---:|---:|
| Raw observation | 0.01433 | 0.04843 | **0.01442** | 0.04846 |
| Gaussian blur | 0.00255 | 0.00424 | 0.06938 | 0.07112 |
| Familiar-trained Wiener, noise known | **0.00251** | **0.00397** | 0.06562 | 0.07549 |
| Complex coefficient shrink | 0.00650 | 0.01280 | 0.02517 | **0.03268** |
| Frozen vocabulary, complex gain | 0.01323 | 0.03585 | 0.02102 | 0.04183 |
| Same vocabulary, nonnegative real gain | 0.01112 | 0.03309 | 0.01885 | 0.03976 |
| Phase-scrambled prototype bank | 0.02028 | 0.03958 | 0.02299 | 0.04243 |

The complex arm has a mean **higher MSE** than the matched no-phase arm on all four evaluated slices. Its per-scene MSE is higher on all 24/24 familiar episodes at σ=.12 and all 24/24 at σ=.22; 11/12 and 12/12 unrelated episodes respectively. The phase-scrambled bank usually does worse, consistent with useful learned prototype structure, but it does not imply unconstrained phase is useful.

Familiar blur/known-noise Wiener succeed because synthetic familiar scenes consist of relatively smooth shapes and gradients. Their prior fails badly on unrelated high-frequency stripe/checker scenes. This is a deliberately hard shift, not a realistic image distribution claim.

## Inference cannot prove world knowledge merely by reducing loss

The tested vocabulary is a compact image coefficient patch dictionary, **not** a semantic world. It is valid to fit a novel scene by selecting the closest available atoms; that does not prove the new scene is understood, and there is no online learning here. A convincing world model would need calibrated uncertainty, truly held-out structural generalization, and temporally consistent behavior.

## Mathematical checks

Six unit tests pass: exact analysis/synthesis, genuinely complex orientation responses, identity overlap-add for untouched coefficient patches, phase-vs-real gain on a known exemplar, sinusoidal translation phase/shift-equivariance, and unrelated generator distinction.

## Reproducibility

From repository root:

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python benchmark.py --output results/main.json
```

Per-episode values: [`main.csv`](main.csv). Parameters and unrounded summaries: [`main.json`](main.json). Visual illustration: [`heldout_examples.png`](heldout_examples.png).

## Decision

**Do not claim phase-driven perceptual-world reconstruction from this experiment.** Next test should use orientation-appropriate phase/translation constraints and localized component compatibility, not free complex gains. No temporal memory or branching was tested.
