# Changelog

## 0.2.0

### Physics fixes that change results

- **Phase space.** The elastic and DIS maps rejected $q\cdot p_{\ell_2} < 0$. This region is physical and contains the $\ell_2$ propagator peak, $1/(Q^2 + 2\,q\cdot p_{\ell_2})$. The cut broke the CP relations between channels and underestimated cross sections. On argon (`nitn=10`, `neval=10^5`):

  | Channel | $E_\nu$ | Coherent | Diffractive |
  | --- | --- | --- | --- |
  | $\nu_\mu \to \nu_\mu\mu^-\mu^+$ | 10 GeV | +0.2% | +13% |
  | $\nu_\mu \to \nu_\mu\mu^-\mu^+$ | 3 GeV | +0.2% | +22% |
  | $\nu_\mu \to \nu_\mu e^-e^+$ | 1 GeV | +18% | +45% |
  | $\nu_e \to \nu_e e^-e^+$ | 1 GeV | +25% | +60% |
  | $\nu_\mu \to \nu_e\mu^-e^+$ | 10 GeV | +17% | +26% |
  | $\nu_e \to \nu_\mu e^-\mu^+$ | 10 GeV | +0.1% | +13% |

  The lepton-phase-space integral now agrees with an independent sequential two-body generator to 0.3% or better at most fixed $(Q^2, p_\nu\cdot q)$ points we sampled. The worst point, an $e^+e^-$ point at small $Q^2$, agrees to 1.5%, within the combined MC error.
- **Antineutrinos.** The CC channel for $\bar\nu_\alpha$ is now $\ell_2^+ = \ell_\alpha^+$. Previously $\bar\nu_\mu \to e^-\mu^+$ received NC couplings, and the forbidden $\bar\nu_\mu \to \mu^-e^+$ received CC couplings. The $\bar\nu$ sign flip now acts on the full $VA$ term, including the axial $Z^\prime$ piece. Flavor-violating channels raise `ValueError`.
- **$Z^\prime$.** The $Z^\prime$ no longer contributes to flavor-changing channels ($\ell_1 \neq \ell_2$).
- **Molière screening.** The coefficients were inverted: $x/(x + 1/\beta_i^2)$ instead of $x/(x+\beta_i^2)$.
- **Neutrino–electron scattering.** The recoil cutoff was $T_e < E_\nu - m_e$ instead of $T_e^\text{max} = 2E_\nu^2/(m_e + 2E_\nu)$. The $Z^\prime$ quadrature was 36% low at 1 MeV.
- **Pauli blocking in DIS and the $\Delta$.** The blocking factor now uses the inelastic $|\mathbf q|^2 = Q^2 + \nu^2$.
- **`SumStructureFunctions`.** It now takes the union of the component domains, not the intersection.

### Generator

- `TridentGenerator` samples coherent, diffractive-proton, and diffractive-neutron events. It honors `seed` and `nuclear_model`, and normalizes the weights over whole Vegas passes.

### Housekeeping

- Removed the legacy collinear-photon DIS code, the EPA-era coupling helpers, and the unused form-factor and blocking functions.
- `mathematica/BSM_trident.nb` is the single symbolic source. The old assembled-cross-section exports in `clean_code/` and the duplicate `export_TL_pieces.wl` are gone.
- Several tests compared cross sections with `pytest.approx` and its default `abs=1e-12`, which made them vacuous. They now use `abs=0`.
