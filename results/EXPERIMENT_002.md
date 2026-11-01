# Experiment 002 — Geometric phase, not phase keys (2026-11-01)

**Status:** Completed locally; 13 unit tests pass. This stage does not use the frozen learned vocabulary from Experiment 001. It isolates a necessary geometric fact before adding temporal state.

## Experimental question

Under a small displacement d, a narrowband complex directional component has approximate local phase advance Δφ = −k·d, where k is its wavevector. Can temporal phase changes recover subpixel translation from two noisy observations, without seeing the ground-truth motion? Is the result destroyed by associating phase measurements with incorrect frequency directions?

The implementation in `phase_motion.py` uses the repository's exactly reconstructing complex directional frame (4 scales, 4 orientations, plus DC). For each non-DC channel, it measures the cross-product of two complex filtered images. It computes an energy-weighted frequency centroid from the **previous noisy observation** and finds the displacement maximizing cross-channel phase agreement. No learned weights, clean images, class labels, or true shifts enter the estimator.

## Setup and controls

- 18 independent held-out synthetic shape scenes, 10 unrelated high-frequency texture scenes, size 64×64.
- Exact circular Fourier translations uniformly in [−0.98, +0.98] pixels per axis.
- Independently sampled previous/current observation noise at σ=0.06, 0.12, 0.22.
- Same known displacement search range (±1.5 pixels) for **all** registration methods.
- Controls: classical normalized Fourier phase correlation (subpixel search), direct least-squares pixel registration, phase direction **scrambled**, zero-motion average, noisy raw image, fixed Gaussian smoothing (σ=0.7), and oracle true shift strictly as a ceiling.
- Image reconstruction for all estimated motion methods = mean of current observation and motion-aligned previous observation. No learned denoising is claimed.

## Motion error (mean pixels, lower better)

| Domain | Noise | Directional complex phase | Classical phase corr. | Pixel registration | Scrambled geometry |
|---|---:|---:|---:|---:|---:|
| Familiar shapes | .06 | **.067** | .100 | .042 | .899 |
| Familiar shapes | .12 | **.100** | .205 | .115 | .892 |
| Familiar shapes | .22 | **.270** | .667 | .514 | .951 |
| Unrelated textures | .06 | .076 | .022 | **.006** | 1.254 |
| Unrelated textures | .12 | .088 | .041 | **.009** | 1.194 |
| Unrelated textures | .22 | .163 | .106 | **.030** | 1.455 |

At σ=.12, paired bootstrap over *independent scenes* (5,000 draws), directional phase minus classical phase correlation: familiar mean −.105 pixel, 95% interval [−.171,−.046] (14/18 better); unrelated mean +.047, interval [.020,.079] (1/10 better). Exploratory uncertainty summaries only, not population claims.

## Reconstruction MSE (lower better)

| Domain | σ | Raw | Gaussian .7 | Directional phase + temporal mean | Classical phase + mean | Oracle motion + mean |
|---|---:|---:|---:|---:|---:|---:|
| Familiar | .12 | .01434 | **.00337** | .00720 | .00730 | .00717 |
| Familiar | .22 | .04827 | **.00918** | .02413 | .02479 | .02384 |
| Unrelated | .12 | .01449 | .03774 | .00757 | .00722 | **.00713** |
| Unrelated | .22 | .04850 | .04342 | .02501 | .02441 | **.02387** |

Gaussian smoothing is markedly better on familiar shapes; motion-aligned accumulation is better on unrelated high-frequency scenes. Pixel registration and classical phase correlation offer robust, well-established alternatives.

## Crucial non-equivalence

For a shift-invariant complex image filter, an image translation means **spatially translating the coefficient field**, not merely adding a constant angle to each component at fixed image coordinates:

`c_j,t(x) = c_j,t-1(x-d)`.

The simpler narrowband approximation `c_j,t(x) ≈ exp(-i k_j·d)c_j,t-1(x)` neglects motion of the amplitude envelope. This works approximately for local periodic signals, but not for arbitrary moving boundaries. Using the true displacement, phase-only reconstruction leaves MSE about .00044 on familiar scenes and .0181 on unrelated scenes; there is no sensor noise in this diagnostic. The tests verify both the exact translation identity and the phase-only failure.

This is **signal processing**, not a biological dendrite model or evidence that the old learned vocabulary works. Directional phase weighting is a variant of classical geometric information, not new physics.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pip install matplotlib  # plotting only
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python benchmark_motion.py --output results/motion_main.json
```

The benchmark produces `results/motion_main.json` (aggregate results), `results/motion_main.csv` (all per-pair measurements) and `results/motion_examples.png`. These generated files are available in the companion ZIP package; run the script to regenerate them.

## Next falsifiable step

Use moving image sequences with occlusion, uncertain motion, and abrupt scene replacement. Compare a persistent phase-component state to a **matched** motion-compensated pixel-state filter. An invertible linear transform followed by identical fusion cannot gain information merely from being complex. Test whether *learned structural constraints or reliable component-specific uncertainty* are responsible for any improvement, and keep the Experiment 001 negative result visible.
