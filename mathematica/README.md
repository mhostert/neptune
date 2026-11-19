# Mathematica / FeynCalc

[`BSM_trident.nb`](BSM_trident.nb) is the single symbolic source for the trident matrix elements in neptune. It computes

    nu_alpha + N -> nu_alpha + l1^- + l2^+ + X

with the W and Z contributions combined into the effective four-Fermi vertex

    L_eff = (G_F/sqrt 2) [nubar gamma^a (1 - gamma_5) nu] [lbar gamma_a (V_ijk - A_ijk gamma_5) l],

where the channel-dependent couplings (V_ijk, A_ijk) may include a Z' contribution.

## Structure

1. **Constants and definitions.** Kinematic invariants x1..x6, the momentum-conservation check, and the on-shell rules `onshelldG`.
2. **Leptonic tensors.** The gamma* nu -> nu l1 l2 tensor, its Ward identity, and the transverse and longitudinal projections sigmaT and sigmaL.
3. **Hadronic tensors.** The coherent (charge form factor), diffractive (F1, F2 of the nucleon), DIS (F1, F2 in xB), and electron-target tensors, with their flux functions hT and hL and Ward-identity checks.
4. **Export to file.** Writes the pieces as `CForm` to `../src/neptune/data/matrixelements/`:

| File | Content |
| --- | --- |
| `sigmaT_lep_DIAG_AV.dat`, `sigmaL_lep_DIAG_AV.dat` | Leptonic sigmaT, sigmaL split into V^2, A^2, and VA pieces |
| `sigmaT_lep_q0_DIAG_AV.dat` | sigmaT at Q^2 = 0 (strict EPA) |
| `hT_coh.dat`, `hL_coh.dat` | Coherent fluxes |
| `hT_dif.dat`, `hL_dif.dat` | Diffractive (elastic nucleon) fluxes |
| `hT_dis.dat`, `hL_dis.dat` | Inclusive DIS fluxes in F1, F2, xB |
| `hT_ele.dat`, `hL_ele.dat` | Point-like electron-target fluxes |

[`src/neptune/_tl_amplitude.py`](../src/neptune/_tl_amplitude.py) parses these files into vectorized NumPy functions, and [`src/neptune/integrands.py`](../src/neptune/integrands.py) assembles

    d sigma = (hT sigmaT + hL sigmaL) / (64 pi^2 x1 x2) d Phi .

## Regenerating

Open the notebook in Mathematica with FeynCalc 9.2 or later (tested with 9.2 and 10.x) and evaluate top to bottom. The first cell sets the working directory to the notebook location, so the relative export path resolves to the package data folder. Run `pytest` afterward. `tests/test_tl_amplitude.py`, `tests/test_couplings_cp.py`, and the regime tests catch convention drifts in the exported expressions.

The fully assembled V^2, A^2, and VA cross sections of the original C++ code (formerly in `clean_code/matrixelements/`) are superseded by the T/L pieces. They remain in the git history.
