<!-- <p align="center">
  <img src="assets/logo.png" alt="neptune: neutrino trident event generator" width="640">
</p> -->

# neptune

`neptune` computes neutrino trident cross sections and generates weighted trident events on arbitrary nuclear targets. It also computes neutrino–electron elastic scattering. The package extends [DarkNews](https://github.com/mhostert/DarkNews-generator) and reads detector targets, exposures, and fluxes from it.

## Physics

### Neutrino trident production

We compute

$$
\nu_\alpha(p_\nu) + N(P_N) \rightarrow \nu_\alpha(p^\prime_\nu) + \ell_1^-(p_3) + \ell_2^+(p_4) + X
$$

for all lepton-flavor combinations $(\alpha, \ell_1, \ell_2)$, with Standard Model (SM) $W^\pm$ and $Z$ exchange and an optional $Z^\prime$. The squared matrix element is derived with FeynCalc in [`mathematica/BSM_trident.nb`](mathematica/BSM_trident.nb). We factorize it into a leptonic piece and a hadronic flux through a transverse/longitudinal (T/L) photon decomposition,

$$
d\sigma = \frac{h_T\,\sigma_T + h_L\,\sigma_L}{64\pi^2\, x_1 x_2}\, d\Phi ,
$$

where $\sigma_{T,L}$ are the leptonic $\gamma^*\nu$ cross sections, $h_{T,L}$ are the hadronic fluxes, $x_1 = Q^2$, $x_2 = p_\nu \cdot q$, and $d\Phi$ is the invariant phase space. The leptonic pieces are common to all regimes. Only $h_{T,L}$ change:

| Regime | Method | Hadronic input | Target scaling |
| --- | --- | --- | --- |
| Coherent | `sigma_coherent` | Nuclear form factor $F(Q^2)$, optional atomic screening | $Z^2$ |
| Diffractive | `sigma_diffractive` | Dipole nucleon form factors, Pauli blocking | $Z\sigma_p + (A-Z)\sigma_n$ |
| $\Delta(1232)$ | `sigma_resonance` | Lalakulich–Paschos vector form factors times a relativistic Breit–Wigner | $A/3$ (impulse approximation, EM isospin factor) |
| Deep-inelastic (DIS) | `sigma_dis` | $F_2$ from an LHAPDF set, $F_1 = F_2/2x$ | $Z F^p_i + (A-Z) F^n_i$ |
| Atomic electrons | `sigma_electron` | Point-like free electron | $Z$ |

**Antineutrinos.** `is_nubar=True` selects $\bar\nu_\alpha$. Within the effective vertex $[\bar\nu\gamma^a(1-\gamma_5)\nu][\bar\ell_1\gamma_a(V-A\gamma_5)\ell_2]$, the charged-current (CC) diagram requires $\ell_1 = \alpha$ for $\nu_\alpha$ and $\ell_2 = \alpha$ for $\bar\nu_\alpha$. The couplings $(V, A)$ are the same for both, and the antineutrino flips the sign of the $VA$ term, including its $Z^\prime$ pieces. At one-photon exchange this gives $\sigma(\bar\nu_\alpha \to \ell_1^-\ell_2^+) = \sigma(\nu_\alpha \to \ell_2^-\ell_1^+)$ by CP, which the test suite checks. Flavor-violating channels raise an error, and the $Z^\prime$ is flavor-diagonal, so it does not contribute when $\ell_1 \neq \ell_2$.

All methods return cross sections in cm² per target nucleus (or atom). `sigma_total()` returns the coherent and diffractive pieces by default and adds the others with `include_resonance=True`, `include_dis=True`, and `include_electron=True`.

The `mode` argument selects how the T/L pieces are combined:

- `mode='full'` (default) keeps $h_T\sigma_T + h_L\sigma_L$ with full $Q^2$ dependence.
- `mode='improved-epa'` drops $h_L\sigma_L$ and keeps $\sigma_T(Q^2, \dots)$.
- `mode='epa'` drops $h_L\sigma_L$ and evaluates $\sigma_T$ at $Q^2 = 0$, the strict equivalent-photon approximation (EPA).

### Nuclear effects

**Coherent form factor.** The default is the analytic Woods–Saxon form factor (`form_factor='woods-saxon'`). `form_factor='darknews'` uses the DarkNews per-nucleus parametrization, namely Fourier–Bessel coefficients where tabulated and a symmetrized Fermi profile otherwise. Any callable $F(Q^2)$ is also accepted.

**Atomic screening.** `atomic_screening='tsai'` (single-pole Thomas–Fermi) or `'moliere'` (three-exponential Molière fit) multiplies $F(Q^2)$ by $G_\text{at}(Q^2)$, with $G_\text{at}(0) = 0$ and $G_\text{at}(\infty) = 1$. The crossover sits at $Q^2 \simeq 1/a_\text{TF}^2$, with $a_\text{TF} = 0.8853\, a_0 Z^{-1/3}$. Screening matters for $e^+e^-$ pairs and is negligible for $\mu^+\mu^-$.

**Pauli blocking.** The diffractive, resonance, and DIS integrands accept `nuclear_model=` with `FreeNucleon()`, `FermiGas(p_F)` (Smith–Moniz), or `LocalFermiGas(density)` (local $p_F(r)$ from a Woods–Saxon density). The blocking factor is evaluated at $|\mathbf{q}|^2 = Q^2 + \nu^2$, with $\nu = Q^2/(2 M x_B)$. Diffractive scattering defaults to `FermiGas(p_F=0.221 GeV)` for every target, inherited from the original C++ code. Resonance and DIS default to free nucleons.

### Neutrino–electron elastic scattering

We compute $\nu_\alpha + e^- \to \nu_\alpha + e^-$ with CC and NC exchange for $\nu_e$ and $\bar\nu_e$, and NC only for $\nu_{\mu,\tau}$ and their antiparticles. The differential cross section is

$$
\frac{d\sigma}{dT_e} = \frac{2 m_e G_F^2}{\pi} \Bigl[ C_L^2 + C_R^2\bigl(1 - T_e/E_\nu\bigr)^2 - C_L C_R \frac{m_e T_e}{E_\nu^2} \Bigr],
\qquad 0 \le T_e \le \frac{2E_\nu^2}{m_e + 2E_\nu},
$$

where $C_L = (C_v + C_a)/2$ and $C_R = (C_v - C_a)/2$, swapped for antineutrinos. An optional $Z^\prime$ shifts the vector coupling,

$$
C_v \to C_v + \frac{Q_V Q_L\, g'^{2}}{2\sqrt{2}\, G_F \bigl(M_{Z^\prime}^2 + 2 m_e T_e\bigr)} ,
$$

where $Q_V$ and $Q_L$ are the $Z^\prime$ charges of the neutrino and the electron.

### Limitations

- The resonance region contains only the $\Delta(1232)$, cut at $W^2 \le (M_\Delta + 5\Gamma_\Delta)^2 \simeq 3.3$ GeV². DIS starts at $W^2 = 4$ GeV² by default, so $3.3 < W^2 < 4$ GeV² is not covered. Higher resonances and non-resonant backgrounds can be added through `resonance.SumStructureFunctions`.
- The electron target is a free, distinguishable electron. We neglect atomic binding and, for $\ell_1 = e$, the exchange diagrams with the target electron.
- `TridentGenerator` samples the coherent and diffractive regimes only.
- The default Vegas settings (`nitn=10`, `neval=10_000`) underestimate channels with $\ell_1 = e$ by up to 2%. We recommend `neval` $\gtrsim 10^5$ for percent-level results with light leptons.

## Installation

```bash
git clone https://github.com/mhostert/neptune.git
cd neptune
pip install -e .          # pulls DarkNews>=0.4.8 from PyPI
pip install -e ".[dev]"   # adds pytest, jupyter, and build tools
```

The distribution name is `neptune-trident` (`neptune` is taken on PyPI). The import name is `neptune`. DIS requires the LHAPDF Python bindings (e.g., from conda-forge) and a PDF grid:

```bash
lhapdf-management update --init
lhapdf-management install CT18NLO
```

Run the test suite with `pytest`.

## Quick start

### Trident cross sections

```python
import neptune as nep

# nu_mu + Ar-40 -> nu_mu + mu- mu+ + X at E_nu = 10 GeV
model = nep.TridentSMModel(nu_flavor='mu', l1_flavor='mu', l2_flavor='mu')
proc = nep.TridentProcess(model, Z=18, A=40, Enu=10.0)

mean, sd = proc.sigma_coherent()                     # cm^2 per Ar-40
out = proc.sigma_total(include_resonance=True)       # dict of (mean, sd)
```

A $Z^\prime$ enters through `nep.TridentBSMModel(mzprime=..., gV_prime=..., bsm_mode=...)`, where `bsm_mode` isolates the SM, interference, or pure $Z^\prime$ terms. A flux is folded in by passing `flux=callable` with `Emin` and `Emax` instead of `Enu`.

DIS uses the same interface with PDF-backed structure functions:

```python
sf = nep.make_lhapdf_structure_functions("CT18NLO", Z=18, A=40, mode="isospin")
proc = nep.TridentProcess(model, Z=18, A=40, Enu=10.0, dis_structure_functions=sf)
mean, sd = proc.sigma_dis()                          # Q^2 > 1 GeV^2, W^2 > 4 GeV^2
```

### Trident events

```python
gen = nep.TridentGenerator(model, Z=18, A=40, Enu=10.0, n_events=50_000, seed=0)
events = gen.generate()       # dict: x1..x6, m6, Enu, weight, mode
```

Weights are in cm² per target nucleus. The coherent weights sum to `sigma_coherent()`, and the diffractive weights (protons times $Z$, neutrons times $A-Z$) sum to `sigma_diffractive()`. A fixed `seed` gives reproducible samples.

### Neutrino–electron scattering

```python
sm = nep.NuElectronSMModel(nu_flavor='mu')
proc = nep.NuElectronProcess(sm, Enu=2.0, T_min=0.05)   # T_e > 50 MeV
sigma = proc.total_xsec()                                # cm^2

df = nep.NuElectronGenerator(sm, Enu=2.0, n_events=10_000).generate()
```

The generator returns a DarkNews-style `pandas.DataFrame` with lab-frame four-momenta.

## Event rates at an experiment

For an exposure $N_\text{POT}$, $N_\text{nuclei}^{(t)}$ fiducial nuclei of each target $t$, and a flux $\Phi(E_\nu)$ in $\nu/\text{cm}^2/\text{POT}/\text{GeV}$,

$$
N_\text{events} = N_\text{POT} \sum_t N_\text{nuclei}^{(t)} \int dE_\nu\, \Phi(E_\nu)\, \sigma^{(t)}(E_\nu) ,
$$

where $\sigma^{(t)}$ is the per-nucleus sum of the regimes above. `neptune` takes $N_\text{POT}$, $N_\text{nuclei}^{(t)}$, $\Phi$, and the target nuclei from `DarkNews.detector.Detector`.

## Examples

- [`Example_1_trident_events.ipynb`](examples/Example_1_trident_events.ipynb): cross sections, energy scans, DIS, $Z^\prime$ scans, event generation, and the three T/L modes.
- [`Example_2_nu_electron_scattering.ipynb`](examples/Example_2_nu_electron_scattering.ipynb): neutrino–electron cross sections, recoil spectra, and events.
- [`Example_3_experiments_trident_rates.ipynb`](examples/Example_3_experiments_trident_rates.ipynb): dimuon trident rates at every experiment registered in DarkNews (DUNE, MINERvA, NOvA, MicroBooNE, MiniBooNE, ICARUS, SBND, NuTeV, ND280, FASER$\nu$, and others).
- [`Example_4_SNS_dielectron_tridents.ipynb`](examples/Example_4_SNS_dielectron_tridents.ipynb): annual $e^+e^-$ trident yields at a stopped-pion source with argon and CH$_2$ detectors.
- [`Example_5_form_factors_and_nuclear_models.ipynb`](examples/Example_5_form_factors_and_nuclear_models.ipynb): coherent form factors, atomic screening, and Pauli-blocking models.

## References

- Czyz, Sheppey & Walecka, *Phys. Rev.* **177** (1969) 2311.
- Løvseth & Radomski, *Phys. Rev. D* **3** (1971) 2686.
- Vogel & Engel, *Phys. Rev. D* **39** (1989) 3378.
- de Gouvêa & Jenkins, *Phys. Rev. D* **74** (2006) 033004.
- Altmannshofer, Gori, Pospelov & Yavin, *Phys. Rev. Lett.* **113** (2014) 091801.
- Magill & Plestid, *Phys. Rev. D* **95** (2017) 073004.
- Ballett, Hostert, Pascoli, Perez Gonzalez, Tabrizi & Zukanovich Funchal, *JHEP* **01** (2019) 119.
- Altmannshofer, Gori, Hamer & Patel, *Phys. Rev. Lett.* **123** (2019) 031802.
- Lalakulich & Paschos, *Phys. Rev. D* **71** (2005) 074003 ($\Delta$ form factors).
- Tsai, *Rev. Mod. Phys.* **46** (1974) 815 (atomic screening).
- de Vries, de Jager & de Vries, *At. Data Nucl. Data Tables* **36** (1987) 495 (nuclear charge densities).
