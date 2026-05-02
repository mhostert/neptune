# Mathematica / FeynCalc

Symbolic squared-matrix-element calculation for the SM and BSM
neutrino trident process

    nu_alpha + N -> nu_alpha + l1^- + l2^+ + N

in the **coherent** (Woods-Saxon nuclear form factor) and **diffractive**
(nucleon dipole form factor) regimes. The W and Z contributions are
combined into the effective four-Fermi vertex

    L_eff = (Gf/sqrt 2) [nubar gamma^a (1 - gamma_5) nu]
                       [lbar gamma_a (V_ijk - A_ijk gamma_5) l],

with channel-dependent (V_ijk, A_ijk) augmented by an optional Z'
contribution.

## `BSM_trident.nb`

The notebook is the canonical symbolic source for everything neptune
computes on the trident side. It defines the kinematic invariants,
builds the leptonic and hadronic tensors (coherent and diffractive
regimes), checks the Ward identity, and exports the differential cross
section split into vector / axial / interference pieces. The polynomial
output is what neptune's Python integrators in
[`src/neptune/amplitudes.py`](../src/neptune/amplitudes.py) and
[`src/neptune/integrands.py`](../src/neptune/integrands.py) consume.

Open in Mathematica and re-evaluate top to bottom to regenerate the
exported expressions. FeynCalc 9.2+ is sufficient; tested with
FeynCalc 9.2 / 10.x.
