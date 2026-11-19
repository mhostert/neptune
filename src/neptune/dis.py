"""
PDF-backed electromagnetic structure functions for DIS trident.

``DISStructureFunctions`` (LHAPDF) and ``ConstantStructureFunctions`` (tests)
supply ``F1(x, Q2)`` and ``F2(x, Q2)`` to ``TridentProcess.sigma_dis``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional

import numpy as np


EQ2 = {1: 1.0 / 9.0, 2: 4.0 / 9.0, 3: 1.0 / 9.0, 4: 4.0 / 9.0, 5: 1.0 / 9.0}


def _maybe_call(value):
    return value() if callable(value) else value


def _maybe_scalar(value, *inputs):
    if all(np.ndim(x) == 0 for x in inputs):
        return float(np.asarray(value))
    return value


@dataclass
class ConstantStructureFunctions:
    """
    Constant test structure functions.

    This is useful for smoke tests and validation. Physics calculations should
    use ``DISStructureFunctions`` with a real PDF set.
    """

    F2_value: float = 1.0
    F1_value: Optional[float] = None
    x_min: float = 1e-6
    x_max: float = 1.0
    q2_min: float = 0.0
    q2_max: float = np.inf

    def F2(self, x, Q2):
        x_arr, q2_arr = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(Q2, dtype=float)
        )
        valid = (
            (x_arr >= self.x_min)
            & (x_arr <= self.x_max)
            & (q2_arr >= self.q2_min)
            & (q2_arr <= self.q2_max)
        )
        out = np.where(valid, self.F2_value, 0.0)
        return _maybe_scalar(out, x, Q2)

    def F1(self, x, Q2):
        x_arr = np.asarray(x, dtype=float)
        if self.F1_value is None:
            out = self.F2(x, Q2) / (2.0 * x_arr)
        else:
            out = np.zeros_like(np.broadcast_arrays(x_arr, np.asarray(Q2))[0])
            out[...] = self.F1_value
        return _maybe_scalar(out, x, Q2)


class DISStructureFunctions:
    """
    Inclusive electromagnetic ``F1`` and ``F2`` from an LHAPDF set.

    Parameters
    ----------
    pdf_set : str
        LHAPDF set name, for example ``"CT18NLO"`` or a nuclear set.
    member : int
        PDF member. ``0`` is usually the central member.
    Z, A : int
        Nuclear charge and mass number for isospin or nuclear combinations.
    mode : {"proton", "neutron", "free_nucleon", "isospin", "nuclear"}
        ``"isospin"`` combines a free-proton set as ``Z F2p + (A-Z) F2n``.
        ``"nuclear"`` applies the same combination to a bound-proton-in-A set.
        ``"free_nucleon"`` is an alias for ``"proton"``.
    callan_gross : bool
        If true, use ``F1 = F2/(2x)``.
    extrapolate : bool
        If false, return zero outside the PDF grid.
    """

    def __init__(
        self,
        pdf_set: str,
        member: int = 0,
        Z: int = 1,
        A: int = 1,
        mode: str = "isospin",
        callan_gross: bool = True,
        extrapolate: bool = False,
    ):
        lhapdf = importlib.import_module("lhapdf")
        self.pdf = lhapdf.mkPDF(pdf_set, member)
        self.pdf_set = pdf_set
        self.member = member
        self.Z = int(Z)
        self.A = int(A)
        self.mode = self._normalise_mode(mode)
        self.callan_gross = bool(callan_gross)
        self.extrapolate = bool(extrapolate)

        self.x_min = float(_maybe_call(getattr(self.pdf, "xMin", 0.0)))
        self.x_max = float(_maybe_call(getattr(self.pdf, "xMax", 1.0)))
        self.q2_min = float(_maybe_call(getattr(self.pdf, "q2Min", 0.0)))
        self.q2_max = float(_maybe_call(getattr(self.pdf, "q2Max", np.inf)))

    @staticmethod
    def _normalise_mode(mode: str):
        mode = mode.lower().replace("-", "_")
        aliases = {
            "free": "proton",
            "free_nucleon": "proton",
            "p": "proton",
            "n": "neutron",
            "bound": "nuclear",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"proton", "neutron", "isospin", "nuclear"}:
            raise ValueError(
                "mode must be one of 'proton', 'neutron', 'isospin', or 'nuclear'"
            )
        return mode

    @classmethod
    def from_lhapdf(cls, *args, **kwargs):
        """Explicit constructor alias for readability at call sites."""
        return cls(*args, **kwargs)

    def _domain_mask(self, x, Q2):
        x_arr, q2_arr = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(Q2, dtype=float)
        )
        if self.extrapolate:
            return np.ones(x_arr.shape, dtype=bool)
        return (
            (x_arr >= self.x_min)
            & (x_arr <= self.x_max)
            & (q2_arr >= self.q2_min)
            & (q2_arr <= self.q2_max)
        )

    def _xfx(self, flav: int, x, Q2, valid):
        x_arr, q2_arr = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(Q2, dtype=float)
        )
        out = np.zeros(x_arr.shape, dtype=float)
        flat_out = out.ravel()
        flat_x = x_arr.ravel()
        flat_q2 = q2_arr.ravel()
        for idx in np.flatnonzero(valid.ravel()):
            flat_out[idx] = self.pdf.xfxQ2(int(flav), float(flat_x[idx]), float(flat_q2[idx]))
        return out

    @staticmethod
    def _neutron_flavor(flav: int):
        if flav == 1:
            return 2
        if flav == -1:
            return -2
        if flav == 2:
            return 1
        if flav == -2:
            return -1
        return flav

    def _F2_proton(self, x, Q2, valid):
        out = np.zeros(np.broadcast_arrays(np.asarray(x), np.asarray(Q2))[0].shape)
        for q in (1, 2, 3, 4, 5):
            out += EQ2[q] * (self._xfx(q, x, Q2, valid) + self._xfx(-q, x, Q2, valid))
        return out

    def _F2_neutron(self, x, Q2, valid):
        out = np.zeros(np.broadcast_arrays(np.asarray(x), np.asarray(Q2))[0].shape)
        for q in (1, 2, 3, 4, 5):
            out += EQ2[q] * (
                self._xfx(self._neutron_flavor(q), x, Q2, valid)
                + self._xfx(self._neutron_flavor(-q), x, Q2, valid)
            )
        return out

    def F2(self, x, Q2):
        valid = self._domain_mask(x, Q2)
        if self.mode == "proton":
            out = self._F2_proton(x, Q2, valid)
        elif self.mode == "neutron":
            out = self._F2_neutron(x, Q2, valid)
        else:
            out = self.Z * self._F2_proton(x, Q2, valid)
            out += (self.A - self.Z) * self._F2_neutron(x, Q2, valid)
        out = np.where(valid, out, 0.0)
        return _maybe_scalar(out, x, Q2)

    def F1(self, x, Q2):
        if not self.callan_gross:
            raise NotImplementedError("Non-Callan-Gross F1 requires an F_L model")
        x_arr = np.asarray(x, dtype=float)
        out = np.where(x_arr > 0.0, self.F2(x, Q2) / (2.0 * x_arr), 0.0)
        return _maybe_scalar(out, x, Q2)


def make_lhapdf_structure_functions(
    pdf_set: str,
    member: int = 0,
    Z: int = 1,
    A: int = 1,
    mode: str = "isospin",
    callan_gross: bool = True,
    extrapolate: bool = False,
):
    """
    Construct ``DISStructureFunctions`` with a clear error if LHAPDF is absent.
    """
    try:
        return DISStructureFunctions(
            pdf_set,
            member=member,
            Z=Z,
            A=A,
            mode=mode,
            callan_gross=callan_gross,
            extrapolate=extrapolate,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "lhapdf":
            raise
        raise ImportError(
            "DIS PDF structure functions require the LHAPDF Python bindings. "
            "Install LHAPDF (for example with conda-forge or your system "
            "package manager) and use lhapdf-management to install the grid."
        ) from exc
