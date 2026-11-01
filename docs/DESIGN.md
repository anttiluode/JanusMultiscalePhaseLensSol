# Design ledger — what the model means and does not mean

## Separating image phase, coordinate phase and state phase

The older Janus Multiscale Lens (`attenuation.py`) rotates five groups of sinusoidal coordinate features. Its `theta` controls **coordinate encoding**, not physical image phase. It also explicitly trains all 32 binary A/B mixtures. Its "cross-scale phase congruence" diagnostic averages all feature phases, rather than aligning local edge phase across matched scale/orientation.

Here, `ComplexDirectionalFrame.analyze(image)` obtains actual complex **image-derived** coefficients. A single sinusoid translated by `dx` has the analytic relationship `delta phase ≈ -kx*dx`. A generic image cannot be translated by setting all channels to the same angle, and the exact phase-shift relationship becomes only local/approximate for broadband image patches.

A frame of filters is a *representation*, not a learned world. This experiment trains a codebook of coefficient neighborhoods, then freezes it and tests reconstruction of unfamiliar images.

## Frame mathematics

For each Fourier frequency sample `k`, choose real, directed filter responses `H_j(k)` and renormalize them pointwise such that:

```
Sum_j H_j(k)^2 = 1
c_j = inverse_fft( H_j * fft(image) )
image = real inverse_fft( Sum_j H_j * fft(c_j) )
```

The frame is redundant and exact on the test grid. Directional masks need not be Hermitian, so coefficients can be complex-valued even for real images. There is no forced quadtree/HMT in v1 and no claim that these are canonical analytic Gabor filters.

## Vocabulary and inference

Every 8×8 overlap patch contains a local multiscale coefficient neighborhood. Training estimates normalization statistics and learns 24 complex unit-norm prototypes by alternating assignment and complex-phase-aligned centroid updates on clean patches.

At inference, with normalized patch `x` and fixed prototype `d` (unit norm):

```
atom_index = argmax_j |<d_j, x>|^2
alpha = <d_index, x>
fit = alpha * d_index
```

For the real-gain ablation: maximize the positive real projection and restrict `alpha ≥ 0`. Both use the same prototypes. Overlap-add coefficient patches with positive windows, then synthesize the real image. A scalar blend with raw observation is chosen on a separate validation set; it is not reoptimized for each test image.

This imposes a compact bottleneck: approximately 225 complex gains and 225 categorical component IDs for 64×64 images, rather than 64×64×5 independently adjustable angles. But overlap, soft observation blending, and the dictionary itself still permit substantial variation; the output is not proof of semantic world knowledge.

## What experiment 001 disproves

The free per-patch complex gain **does not improve generalization** on the chosen image/noise setting. It is consistently worse than its nonnegative real-gain sibling, and both trail Gaussian/Wiener on familiar scenes. A global prototype phase is not a physically constrained displacement; it can simply rotate coefficients into a misleading configuration. A low fitting residual alone would not prove recognition of a learned world.

Unexpectedly, a training-spectrum Wiener filter catastrophically fails on unrelated high-frequency scenes: this is not a failure of Wiener theory, only of applying a mismatched spectral prior. Raw and thresholded coefficients matter as distribution-shift controls.

## Candidate next experiment (preregister before running)

1. Replace one arbitrary phase per patch with **two real displacement coordinates** `(dx,dy)` and, if needed, a gain. For orientation `j`, use predicted phase `-(kx_j dx + ky_j dy)`; forbid independent phase rotations of incompatible orientations.
2. Infer sparse atom identities plus local displacements using a spatial continuity prior, but allow discontinuities at edges/occlusions.
3. Evaluate unseen translations, deformations, clutter, camera noise and **new scene structure**; measure reconstruction, phase-error, uncertainty, compute and ghosting.
4. Compare against matched-capacity real dictionary, coefficient shrinkage, phase-scrambled templates, and conventional registration/flow. Keep train/validation/test sequences disjoint.
5. Only then add **causal temporal memory** and long-term component splitting if it significantly improves held-out likelihood at matched computation.

## Biological boundary

This is not a simulator of apical tufts, migraine aura, cortical spreading depolarization, or consciousness. The complex coefficients are engineered signal-processing variables. Any connection to cortical local orientation representations is motivation for a hypothesis, not empirical evidence about neural physiology.
