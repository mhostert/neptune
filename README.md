<!-- <p align="center">
  <img src="assets/logo.png" alt="neptune — neutrino trident event generator" width="640">
</p> -->

# neptune

**N**eutrino **E**vent generator for **P**hysics with **T**ridents **U**sing **N**uclear **E**xchange

A Python package for computing neutrino trident and neutrino–electron elastic
cross sections, and for generating Monte Carlo events on arbitrary nuclear
targets. Built to mirror and extend
[DarkNews](https://github.com/mhostert/DarkNews-generator).

## Physics

### Trident

The $2 \to 4$ coherent and diffractive process

$$
\nu_\alpha(p_\nu) + N(P_N)  \rightarrow  \nu_\alpha(p^\prime_\nu)  +  \ell_1^-(p_3)  +  \ell_2^+(p_4)  +  N(P^\prime_N)
$$

for all lepton-flavor combinations $(\alpha, \ell_1, \ell_2)$, with Standard Model
$(W^\pm, Z)$ exchange and an optional BSM $Z^\prime$ contribution. Coherent
scattering on a nucleus uses the Woods–Saxon form factor and scales as $Z^2$;
diffractive scattering uses the proton dipole form factor with Pauli blocking
and scales as $Z$.

### Neutrino–electron elastic scattering

The $2 \to 2$ process

$$
\nu_\alpha(p_\nu)  +  e^-(p_e)  \rightarrow  \nu_\alpha(p^\prime_\nu)  +  e^-(p^\prime_e)  ,
$$

with full SM (CC + NC) treatment for $\nu_e / \bar\nu_e$ and NC-only for
$\nu_\mu, \nu_\tau$ and their antiparticles, plus an optional BSM $Z^\prime$
contribution that modifies the vector coupling as

$$
C_v  \to  C_v  +  \frac{Q_V Q_L g'^{ 2}}{2\sqrt{2} G_F \bigl(M_{Z^\prime}^2 + 2 m_e T_e\bigr)}  .
$$

## Installation

```bash
pip install -e /path/to/DarkNews-generator   # install DarkNews first
pip install -e .
```

## Quick start

### Trident

```python
import neptune as nep

# SM model for nu_mu + N -> nu_mu + mu+ mu- + N
model = nep.TridentSMModel(nu_flavor='mu', l1_flavor='mu', l2_flavor='mu')

# Argon-40 target at E_nu = 10 GeV
proc = nep.TridentProcess(model, Z=18, A=40, Enu=10.0)
mean, sd = proc.sigma_coherent()
print(f"sigma (coherent, Ar-40, 10 GeV) = {mean:.3e} +/- {sd:.3e} cm^2")

# Generate weighted MC events
gen = nep.TridentGenerator(model, Z=18, A=40, Enu=10.0, n_events=5_000)
events = gen.generate()
```

The coherent integration uses the equivalent-photon approximation with the
$A$-dependent boundary $Q_\text{max}^\text{coh} = \Lambda_{\rm QCD}/A^{1/3}$ between the
coherent and nucleon-elastic regimes. Pass `use_epa=False` to switch to the
full $8$-D coherent matrix element (currently *experimental*; see
[examples/Example_1_trident_events.ipynb](examples/Example_1_trident_events.ipynb)).

### Neutrino–electron elastic

```python
import neptune as nep

# SM nu_mu + e -> nu_mu + e
sm = nep.NuElectronSMModel(nu_flavor='mu')
proc = nep.NuElectronProcess(sm, Enu=2.0, T_min=0.05)   # 50 MeV recoil cut
print(f"sigma (nu_mu, 2 GeV, T_e > 50 MeV) = {proc.total_xsec():.3e} cm^2")

# Generate events with full lab-frame kinematics
gen = nep.NuElectronGenerator(sm, Enu=2.0, n_events=10_000)
df = gen.generate()                 # DarkNews-style pandas DataFrame
print(df[('Te', '')].mean())
```

The differential cross section is

$$
\frac{d\sigma}{dT_e}  =  \frac{2 m_e G_F^2}{\pi} \Bigl[ C_L^2  +  C_R^2\bigl(1 - T_e/E_\nu\bigr)^2  -  C_L C_R \frac{m_e T_e}{E_\nu^2} \Bigr]  ,
$$

with $C_L = (C_v + C_a)/2$ and $C_R = (C_v - C_a)/2$ (swapped for antineutrinos).

## Event rate at an experiment

For an experiment with exposure $N_\text{POT}$, fiducial $N_\text{nuclei}^{(t)}$ on each
target $t$, and flux $\Phi(E_\nu)$ in $\nu/\text{cm}^2/\text{POT}/\text{GeV}$,

$$
N_\text{events}  =  N_\text{POT}  \sum_t N_\text{nuclei}^{(t)} \int dE_\nu \Phi(E_\nu) \sigma^{(t)}(E_\nu)  ,
$$

with $\sigma^{(t)}(E_\nu) = \sigma^{(t)}_\text{coh}(E_\nu) + Z^{(t)} \sigma^{(t)}_\text{diff}(E_\nu)$
the per-nucleus total cross section. `neptune` reads $N_\text{nuclei}^{(t)}$, $N_\text{POT}$,
$\Phi$ and target nuclei straight from `DarkNews.detector.Detector`; see
[examples/Example_3_experiments_trident_rates.ipynb](examples/Example_3_experiments_trident_rates.ipynb)
for a survey across every experiment registered in DarkNews.

## Examples

The [`examples/`](examples/) folder contains:

- [`Example_1_trident_events.ipynb`](examples/Example_1_trident_events.ipynb)
  — trident cross sections, BSM $Z^\prime$ scans, and event generation.
- [`Example_2_nu_electron_scattering.ipynb`](examples/Example_2_nu_electron_scattering.ipynb)
  — neutrino–electron elastic cross sections, electron recoil spectra, and event generation.
- [`Example_3_experiments_trident_rates.ipynb`](examples/Example_3_experiments_trident_rates.ipynb)
  — predicted dimuon trident rates at every experiment shipped with DarkNews
  (DUNE, MINERvA, NOvA, MicroBooNE, MiniBooNE, ICARUS, SBND, NuTeV, ND280, FASER$\nu$, …).

## References

- Czyz, Sheppey & Walecka, *Phys. Rev.* **177** (1969) 2311.
- Lovseth & Radescu, *Phys. Rev. D* **3** (1971) 2706.
- Vogel & Engel, *Phys. Rev. D* **39** (1989) 3378.
- de Gouvêa & Jenkins, *Phys. Rev. D* **74** (2006) 033004.
- Altmannshofer, Gori, Pospelov & Yavin, *Phys. Rev. Lett.* **113** (2014) 091801.
- Magill & Plestid, *Phys. Rev. D* **95** (2017) 073004.
- Ballett, Hostert, Pascoli, Perez Gonzalez, Tabrizi & Zukanovich Funchal, *JHEP* **01** (2019) 119.
- Altmannshofer, Gori, Hamer & Patel, *Phys. Rev. Lett.* **123** (2019) 031802.
