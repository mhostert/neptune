(* ::Package:: *)

(* ============================================================== *)
(*  T/L decomposition exports for the neptune Python package.     *)
(*                                                                *)
(*  Paste these cells into BSM_trident.nb after every section     *)
(*  has been evaluated (so that sigmaT/sigmaL, hTcoh/hLcoh,       *)
(*  hTdif/hLdif, and onshelldG are all defined).                  *)
(*                                                                *)
(*  Output directory (relative to BSM_trident.nb):                *)
(*      ../src/neptune/data/matrixelements/                       *)
(*                                                                *)
(*  Files produced:                                               *)
(*      sigmaT_lep_DIAG_AV.dat                                    *)
(*      sigmaL_lep_DIAG_AV.dat                                    *)
(*      sigmaT_lep_q0_DIAG_AV.dat   (sigmaT evaluated at q^2=0)   *)
(*      hT_coh.dat,  hL_coh.dat                                   *)
(*      hT_dif.dat,  hL_dif.dat                                   *)
(*                                                                *)
(*  Mirrors the existing pipeline (lines 6829-7290 of the .nb)    *)
(*  that produced AVcoh / V2coh / A2coh / D1 / D2 / D12 / D21     *)
(*  and the assembled dsigmaCOH_FULL_DIAG_AV.dat. The same        *)
(*  pipeline is now applied to sigmaT and sigmaL instead of       *)
(*  d\[Sigma]coh, so coherent and diffractive can share kinematics *)
(*  and only differ in the choice of hadronic flux function.      *)
(* ============================================================== *)


(* --------------------------------------------------------------
   Helper: V^2 / A^2 / VA + DIAG decomposition.
   Identical to what the existing AVcoh/V2coh/A2coh/D1/D2/D12/D21
   chain does to d\[Sigma]coh and d\[Sigma]dif, but generic so we
   can apply it to sigmaT or sigmaL.
   -------------------------------------------------------------- *)
ClearAll[buildDIAGAV];
buildDIAGAV[expr_] := Module[
  {eAV, eV2, eA2, eSEP, e11, e22, e12, e21},
  eAV = First[Coefficient[expr,
              {Subscript[V, ijk] Subscript[A, ijk]}] /. onshelldG // Simplify];
  eV2 = First[Coefficient[expr,
              {Subscript[V, ijk]^2}] /. onshelldG // Simplify];
  eA2 = First[Coefficient[expr,
              {Subscript[A, ijk]^2}] /. onshelldG // Simplify];
  eSEP = (A2 eA2 + V2 eV2 + VA eAV) // Simplify;
  e11 = First[Coefficient[eSEP, {DIAG1  DIAG1c}] /. onshelldG // Simplify];
  e22 = First[Coefficient[eSEP, {DIAG2  DIAG2c}] /. onshelldG // Simplify];
  e12 = First[Coefficient[eSEP, {DIAG1  DIAG2c}] /. onshelldG // Simplify];
  e21 = First[Coefficient[eSEP, {DIAG2  DIAG1c}] /. onshelldG // Simplify];
  Diag11 e11 + Diag22 e22 + Diag12 (e21 + e12) // Simplify
];


(* Common substitution rule (matches the existing exports). *)
exportSubs = {
  echarge -> 2 Sqrt[\[Pi] alphaQED],
  m1 -> ml1,
  m2 -> ml2
};

outDir = "../src/neptune/data/matrixelements/";


(* --------------------------------------------------------------
   1. Leptonic transverse cross section sigmaT_{gamma nu}.
   -------------------------------------------------------------- *)
sigmaTleptonicFULL = buildDIAGAV[Subscript[\[Sigma]T, \[Gamma]\[Nu]]];
Export[outDir <> "sigmaT_lep_DIAG_AV.dat",
       CForm[sigmaTleptonicFULL /. exportSubs]]


(* --------------------------------------------------------------
   2. Leptonic longitudinal cross section sigmaL_{gamma nu}.
   -------------------------------------------------------------- *)
sigmaLleptonicFULL = buildDIAGAV[Subscript[\[Sigma]L, \[Gamma]\[Nu]]];
Export[outDir <> "sigmaL_lep_DIAG_AV.dat",
       CForm[sigmaLleptonicFULL /. exportSubs]]


(* --------------------------------------------------------------
   3. Leptonic transverse cross section evaluated at q^2 = 0.
      Used by the strict EPA mode in the Python harness.
      The L_T projector (g_{ab} + q^2/(p1.q)^2 p1.p1) has a smooth
      q^2 -> 0 limit; sigmaT itself has no inverse photon propagator
      (that lives in the hadronic flux), so direct substitution
      x1 -> 0 is sufficient.  If sigmaTleptonicFULL retains a 1/x1
      after Simplify, replace `... /. x1 -> 0` by
      `Limit[..., x1 -> 0]`.
   -------------------------------------------------------------- *)
sigmaTleptonicQ0 = (sigmaTleptonicFULL /. x1 -> 0) // Simplify;
Export[outDir <> "sigmaT_lep_q0_DIAG_AV.dat",
       CForm[sigmaTleptonicQ0 /. exportSubs]]


(* --------------------------------------------------------------
   4-5. Coherent flux functions hT_coh, hL_coh.
        These are purely hadronic (no V/A/Diag), so no
        buildDIAGAV step is needed.
   -------------------------------------------------------------- *)
Export[outDir <> "hT_coh.dat",
       CForm[(hTcoh /. onshelldG // Simplify) /. exportSubs]]
Export[outDir <> "hL_coh.dat",
       CForm[(hLcoh /. onshelldG // Simplify) /. exportSubs]]


(* --------------------------------------------------------------
   6-7. Diffractive flux functions hT_dif, hL_dif.
        The notebook has these in terms of the Expression219check
        hadronic tensor. Replace F1/F2 by HH1/HH2 so the Python
        side can pass dipole form-factor combinations directly.
   -------------------------------------------------------------- *)
Export[outDir <> "hT_dif.dat",
       CForm[(hTdif /. onshelldG // Simplify) /. exportSubs]]
Export[outDir <> "hL_dif.dat",
       CForm[(hLdif /. onshelldG // Simplify) /. exportSubs]]


(* --------------------------------------------------------------
   Sanity check: re-assemble dSigma_coh from the exported pieces
   and compare to the existing dSigma_cohFULL.

      d\[Sigma]coh = (1 / (64 \[Pi]^2 x1 x2)) *
                     (hTcoh sigmaT + hLcoh sigmaL)

   The check should reduce to 0 after Simplify. If not, the most
   likely cause is a sign / prefactor convention drift between
   sigmaT/sigmaL and hT/hL, which can be tracked down by comparing
   to the direct definition at line 6427 of BSM_trident.nb.
   -------------------------------------------------------------- *)
checkCoh = Simplify[
  (hTcoh sigmaTleptonicFULL + hLcoh sigmaLleptonicFULL) /
    (64 \[Pi]^2 x1 x2)
  - d\[Sigma]cohFULL
];
Print["coherent T/L re-assembly check (expected 0): ", checkCoh];

checkDif = Simplify[
  (hTdif sigmaTleptonicFULL + hLdif sigmaLleptonicFULL) /
    (64 \[Pi]^2 x1 x2)
  - d\[Sigma]difFULL
];
Print["diffractive T/L re-assembly check (expected 0): ", checkDif];
