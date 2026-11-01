# Follow-up: vocabulary capacity does not reverse the negative phase result

**Date 2026-11-01; exploratory follow-up, not preregistered.** The same deterministic benchmark was rerun with 12, 48, and 96 atoms, alongside the original 24. All four sizes were trained from the same 32 training scenes with 1,000 training coefficient patches. They each selected a fixed 0.75 blend on the same held-out eight-image validation split. Four held-out test slices (24 familiar and 12 unrelated images at each noise level) were unchanged.

| Atoms | Familiar σ=.12: complex / real | Familiar σ=.22: complex / real | Unrelated σ=.12: complex / real | Unrelated σ=.22: complex / real |
|---:|---:|---:|---:|---:|
| 12 | .01262 / **.01049** | .03449 / **.03117** | .01866 / **.01628** | .04080 / **.03741** |
| 24 | .01323 / **.01112** | .03585 / **.03309** | .02102 / **.01885** | .04183 / **.03976** |
| 48 | .01295 / **.01105** | .03495 / **.03222** | .01969 / **.01717** | .03993 / **.03672** |
| 96 | .01309 / **.01135** | .03510 / **.03257** | .02049 / **.01733** | .04106 / **.03690** |

**Observation:** The freely rotatable complex-gain version is worse than the matched real-gain version at *every* tested capacity and domain/noise slice (16/16). More prototypes are not sufficient to rescue this design on this test. Differences can reflect local maxima, normalization, and imperfect atom learning; this is not a universal impossibility result for complex wavelet dictionaries.

**Why the result is informative:** Both versions share exactly the same learned complex templates, inference patch locations, synthesis operator, validation split, and per-case blend. The only difference is that the first can rotate each patch's *entire prototype* by an arbitrary phase, while the second uses only nonnegative scalar gain. The phase freedom is not constrained by the local spatial-frequency direction; rotations can therefore modify an image without representing a coherent displacement.

The next candidate should constrain phase changes to a physically meaningful displacement `Δφ_{s,o} = -k_{s,o}·Δx` (with tolerance for broadband coefficients), test temporal alignment, and keep a no-phase and pure-motion-estimator control. Do not claim generalization to natural images until real-video tests are performed. The per-slice numeric data is in [`capacity_sweep.csv`](capacity_sweep.csv).
