"""
neptune
=======
Neutrino Event generator for Physics with Tridents Using Nuclear Exchange.

A sibling package to `DarkNews <https://github.com/mhostert/DarkNews-generator>`_
for the SM and BSM neutrino trident process

    nu + N → nu + l⁻ + l⁺ + X        (coherent, diffractive, Δ resonance, DIS,
                                      and atomic-electron targets)

and for neutrino-electron elastic scattering

    nu + e⁻ → nu + e⁻                (SM and BSM Z')

Quick start
-----------
>>> import neptune as nep
>>> model = nep.TridentSMModel(nu_flavor='mu', l1_flavor='mu', l2_flavor='mu')
>>> proc = nep.TridentProcess(model, Z=18, A=40, Enu=10.0)
>>> mean, sd = proc.sigma_coherent()
>>> mean
5.3e-42  # cm^2 per Argon-40 nucleus

>>> sm = nep.NuElectronSMModel(nu_flavor='mu')
>>> nep.NuElectronProcess(sm, Enu=2.0).total_xsec()
3.2e-42  # cm^2

References
----------
Czyz et al., Phys.Rev. 177 (1969) 2311
Løvseth & Radomski, Phys. Rev. D 3 (1971) 2686
Vogel & Engel, Phys.Rev. D39 (1989) 3378
"""

from neptune.model import TridentSMModel, TridentBSMModel
from neptune.processes import TridentProcess
from neptune.MC import TridentGenerator
from neptune.nu_electron import (
    NuElectronSMModel,
    NuElectronBSMModel,
    NuElectronProcess,
    NuElectronGenerator,
)
from neptune.dis import (
    ConstantStructureFunctions,
    DISStructureFunctions,
    make_lhapdf_structure_functions,
)
from neptune.nuclear_tools import (
    get_nuclear_target,
    get_form_factor,
    find_nucleus_name,
    NuclearTarget,
)
from neptune import const

__version__ = "0.2.0"
__all__ = [
    # Trident
    "TridentSMModel",
    "TridentBSMModel",
    "TridentProcess",
    "TridentGenerator",
    "DISStructureFunctions",
    "ConstantStructureFunctions",
    "make_lhapdf_structure_functions",
    # Nu-electron
    "NuElectronSMModel",
    "NuElectronBSMModel",
    "NuElectronProcess",
    "NuElectronGenerator",
    # Nuclear targets / form factors / constants
    "get_nuclear_target",
    "get_form_factor",
    "find_nucleus_name",
    "NuclearTarget",
    "const",
]
