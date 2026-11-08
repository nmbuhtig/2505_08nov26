#!/usr/bin/env python3
# -*- coding: utf-8 -*-
   
import argparse
import json
import os
import re
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2


class Out:
    def __init__(self):
        self.lines = []
        self.data = {}

    def h(self, t):
        self.lines += ["", t, "-" * len(t)]

    def add(self, *s):
        self.lines += [str(x) for x in s]

    def kv(self, k, v):
        self.lines.append(f"  {k:<58s}: {v}")


def read_R_trial(folder, default=500):
    p = os.path.join(folder, "00_run_info.txt")
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8", errors="replace"):
            m = re.match(r"\s*R_trial\s*:\s*(\d+)", ln)
            if m:
                return int(m.group(1))
    return default


def noncentrality_for_power(dof, alpha, power):
    q = chi2.ppf(1 - alpha, dof)
    return brentq(lambda L: ncx2.sf(q, dof, L) - power, 1e-6, 1e5)


                                                                              
def part_identifiability(S, R0, o):
    o.h("1. Identifiability: well-determined combination (log parameters, full covariance)")
    o.add(f"  From summary.json 'ident' (noise from one MC run with R = {R0}).",
          "  The whitened Jacobian J (columns d/d ln sigma_h, d/d ln l_c) has singular",
          "  values s1 >= s2 and right singular vectors v1 (best) and v2 (worst).",
          "  sd along v1 = 1/s1, sd along v2 = 1/s2.  The best combination is",
          "  q = v1[0] ln sigma_h + v1[1] ln l_c; written as sigma_h^a * l_c (a = v1[0]/v1[1]),",
          "  its relative precision is sd(q)/|v1[1]|.", "")
    res = {}
    for dn, d in S["ident"].items():
        blk = {}
        for lab in ("both distances", "z1 only", "z2 only"):
            m = d[lab]["full_cov"]
            if not m.get("identifiable", True):
                blk[lab] = dict(identifiable=False, modes=m.get("modes_kept"))
                continue
            s1, s2 = m["sing_vals"][0], m["sing_vals"][-1]
            v1 = np.array(m["best_dir"])
            if v1[1] < 0:
                v1 = -v1
            a = v1[0] / v1[1]
            blk[lab] = dict(identifiable=True, modes=m["modes_kept"],
                            crb_sig_h_pct=100 * m["crb_rel_sig_h"], crb_l_c_pct=100 * m["crb_rel_l_c"],
                            corr=m["corr"], cond=m["cond"],
                            best_dir=v1.tolist(), sd_best=1 / s1, sd_worst=1 / s2,
                            exponent_a=a, rel_precision_combination_pct=100 * (1 / s1) / abs(v1[1]))
        res[dn] = blk
        b = blk["both distances"]
        o.add(f"  Design {dn}")
        if not b.get("identifiable", True):
            o.kv("    both distances", "not identifiable")
            continue
        o.kv("    modes kept", b["modes"])
        o.kv("    CRB sigma_h, l_c [%]", f"{b['crb_sig_h_pct']:.2f}, {b['crb_l_c_pct']:.2f}")
        o.kv("    correlation, condition number", f"{b['corr']:+.5f}, {b['cond']:.1f}")
        o.kv("    best direction (ln sigma_h, ln l_c)", f"({b['best_dir'][0]:+.3f}, {b['best_dir'][1]:+.3f})")
        o.kv("    sd along best direction", f"{b['sd_best']:.4f}  ({100 * b['sd_best']:.2f} %)")
        o.kv("    sd along worst direction", f"{b['sd_worst']:.3f}")
        o.kv("    combination sigma_h^a * l_c, a", f"{b['exponent_a']:.3f}")
        o.kv("    relative precision of sigma_h^a * l_c [%]", f"{b['rel_precision_combination_pct']:.2f}")
        for lab in ("z1 only", "z2 only"):
            m = blk[lab]
            if m.get("identifiable", True):
                o.kv(f"    {lab}: CRB sigma_h, l_c [%] (modes)",
                     f"{m['crb_sig_h_pct']:.2f}, {m['crb_l_c_pct']:.2f}  ({m['modes']})")
            else:
                o.kv(f"    {lab}", f"not identifiable ({m['modes']} mode)")
        o.add("")
    o.data["identifiability"] = res
    return res


def part_scaling(ident, R0, o):
    o.h("2. Ensemble size needed (CRB scales exactly as 1/sqrt(R) for fixed design)")
    o.add(f"  R_needed = R0 * (CRB(R0) / target)^2 with R0 = {R0}.", "")
    res = {}
    rows = []
    for dn, blk in ident.items():
        b = blk["both distances"]
        if not b.get("identifiable", True):
            continue
        r = {}
        for tgt in (20, 10, 5):
            r[f"sig_h_{tgt}pct"] = R0 * (b["crb_sig_h_pct"] / tgt) ** 2
            r[f"l_c_{tgt}pct"] = R0 * (b["crb_l_c_pct"] / tgt) ** 2
        res[dn] = r
        rows.append(f"  {dn:<14s} sigma_h: 20% {r['sig_h_20pct']:8.0f}   10% {r['sig_h_10pct']:8.0f}   "
                    f"5% {r['sig_h_5pct']:9.0f}")
        rows.append(f"  {'':<14s} l_c    : 20% {r['l_c_20pct']:8.0f}   10% {r['l_c_10pct']:8.0f}   "
                    f"5% {r['l_c_5pct']:9.0f}")
    o.add(*rows)
    o.data["R_needed"] = res


def part_faults(S, R0, o):
    o.h("3. Fault isolation: minimum detectable fault sizes")
    o.add("  Linear sensitivity model: a fault of size eps adds whitened residual",
          "  eps * v, of which the part outside the surface subspace has norm",
          "  eps * rho * |v|.  After re-fitting the surface parameters, chi2 is",
          "  non-central chi2(m - 2, lambda) with lambda = (eps rho |v|)^2.",
          "  Detection at the 99% null quantile with power P requires",
          "  lambda >= lambda_P, so eps_min = sqrt(lambda_P) / (rho |v|).",
          "  Units: relative (scale faults) or mm (decentre).  Values at R0; they",
          "  scale as 1/sqrt(R).  First-order estimates; large faults are nonlinear.", "")
    res = {}
    rho_all = S["fault"]["rho"]
    for dn, rr in rho_all.items():
        sc = S["fault"]["scenarios"][dn]
        m = int(sc["no fault"]["modes"])
        o.add(f"  Design {dn}  (modes m = {m}, residual dof = {m - 2})")
        if m <= 2:
            o.add("    no residual space: fault isolation impossible for this design", "")
            res[dn] = dict(modes=m, possible=False)
            continue
        lam50 = noncentrality_for_power(m - 2, 0.01, 0.5)
        lam90 = noncentrality_for_power(m - 2, 0.01, 0.9)
        o.kv("    lambda for 50% / 90% power", f"{lam50:.2f} / {lam90:.2f}")
        d = dict(modes=m, possible=True, lambda50=lam50, lambda90=lam90, faults={})
        for fn, r in rr.items():
            if "|v|" in fn:
                continue
            nv = rr[fn + " |v| (whitened)"]
            if r is None or isinstance(r, str) or r <= 0 or nv <= 0:
                o.kv(f"    {fn}", "first-order insensitive" if r is None else str(r))
                d["faults"][fn] = None
                continue
            e50 = np.sqrt(lam50) / (r * nv)
            e90 = np.sqrt(lam90) / (r * nv)
            unit = "mm" if "decentre" in fn else "rel."
            o.kv(f"    {fn}", f"rho={r:.3f} |v|={nv:8.2f}  eps_min 50%: {e50:.3g} {unit},"
                               f" 90%: {e90:.3g} {unit}")
            d["faults"][fn] = dict(rho=r, v=nv, eps50=e50, eps90=e90, unit=unit)
                                                  
        targets = {"z2 scale (block z2 only)": 0.03, "lambda (relative)": 0.02,
                   "w envelope (relative)": 0.05, "decentre/tilt both [1 mm]": 0.1}
        o.add("    R needed for 50% detection of:")
        d["R_needed_50pct"] = {}
        for fn, eps in targets.items():
            f = d["faults"].get(fn)
            if f:
                Rn = R0 * (f["eps50"] / eps) ** 2
                d["R_needed_50pct"][fn] = dict(eps=eps, R=Rn)
                o.kv(f"      {fn} = {eps:g}", f"R ~ {Rn:,.0f}")
                                
        names = [n for n in ("z scale (both blocks)", "lambda (relative)", "w envelope (relative)")
                 if d["faults"].get(n)]
        if len(names) >= 2:
            prods = [d["faults"][n]["rho"] * d["faults"][n]["v"] for n in names]
            o.kv("    rho*|v| for z (both), lambda, w", ", ".join(f"{x:.3f}" for x in prods))
            o.add("    (equal magnitudes suggest, but do not prove, one shared residual",
                  "     signature; the directions are compared in sect_extra_checks.py,",
                  "     check D, which reports the angles between the orthogonal parts)")
        res[dn] = d
        o.add("")
    o.data["faults"] = res

    o.h("3b. MC fault scenarios (nonlinear re-fit test), restated from summary.json")
    for dn, sc in S["fault"]["scenarios"].items():
        o.add(f"  Design {dn}")
        for sn, d in sc.items():
            flag = d.get("refit_flag", "n/a")
            o.kv(f"    {sn}", f"chi2 after re-fit {d['refit_chi2']:.3g} (dof {d['modes'] - 2}, "
                              f"p99 {d['refit_p99']:.2f}) -> {flag}; linear E_S {d['E_S']:.3g}, E_P {d['E_P']:.3g}")
        o.add("")


def part_bayes(S, o):
    o.h("4. Bayesian inference summary (restated from summary.json)")
    res = {}
    for k, d in S["bayes"].items():
        res[k] = {kk: d[kk] for kk in ("coverage95_LR", "coverage95", "bias_logsig", "bias_loglc",
                                       "sd_logsig", "sd_loglc", "median_sig_h_nm", "median_l_c_um")}
        o.kv(k, f"LR cov {d['coverage95_LR']:.3f}, Laplace cov {d['coverage95']:.3f}, "
                f"bias ({d['bias_logsig']:+.3f}, {d['bias_loglc']:+.3f}), "
                f"sd ({d['sd_logsig']:.3f}, {d['sd_loglc']:.3f}), "
                f"median ({d['median_sig_h_nm']:.1f} nm, {d['median_l_c_um']:.2f} um)")
    n = None
    for f in os.listdir(o.v3):
        if f.startswith("data_bayes_trials_"):
            n = len(np.atleast_2d(np.loadtxt(os.path.join(o.v3, f))))
            break
    if n:
        o.kv("trials", n)
        o.kv("binomial SE of a 0.95 coverage", f"{np.sqrt(0.95 * 0.05 / n):.3f}")
    o.data["bayes"] = res


def part_inverse(inv, o):
    o.h("5. Inverse design: metrics recomputed excluding dX = 0")
    res = {}
    for tn in ("T1_gaussian", "T2_fringed"):
        for xc in ("0", "10"):
            f = os.path.join(inv, f"data_validate_{tn}_xc{xc}mm.txt")
            if not os.path.exists(f):
                continue
            v = np.loadtxt(f)
            v = v[v[:, 0] > 0]
            r = v[:, 1] - v[:, 4]
            d = dict(chi2_per_dof=float(np.mean((r / v[:, 3]) ** 2)), rms_err=float(np.sqrt(np.mean(r ** 2))),
                     max_err=float(np.max(np.abs(r))), mean_SE=float(v[:, 3].mean()),
                     max_abs_imag=float(np.max(np.abs(v[:, 2]))), n_points=int(len(v)))
            res[f"validate_{tn}_xc{xc}mm"] = d
            o.kv(f"{tn}, pairs centred at {xc} mm",
                 f"chi2/dof {d['chi2_per_dof']:.2f}, RMS {d['rms_err']:.2e}, max {d['max_err']:.2e}, "
                 f"mean SE {d['mean_SE']:.2e}, max|Im| {d['max_abs_imag']:.2e} (n={d['n_points']})")
        f = os.path.join(inv, f"data_validate_{tn}_intensity.txt")
        if os.path.exists(f):
            v = np.loadtxt(f)
            w = float(np.sqrt(np.sum(v[:, 0] ** 2 * v[:, 1]) / np.sum(v[:, 1])) * 1e3)
            res[f"intensity_{tn}"] = dict(rms_width_mm=w)
            o.kv(f"{tn}, far-field intensity rms width", f"{w:.2f} mm")
    o.add("")
    for tn in ("T1_gaussian", "T2_fringed"):
        a = os.path.join(inv, f"data_knobs_{tn}_diffuser1_coherence.txt")
        b = os.path.join(inv, f"data_knobs_{tn}_diffuser2_coherence.txt")
        if os.path.exists(a) and os.path.exists(b):
            A, B = np.loadtxt(a), np.loadtxt(b)
            m = A[:, 0] > 0
            z = (A[m, 1] - B[m, 1]) / np.sqrt(A[m, 2] ** 2 + B[m, 2] ** 2)
            d = dict(max_abs_z=float(np.max(np.abs(z))), chi2_per_dof=float(np.mean(z ** 2)),
                     max_abs_diff=float(np.max(np.abs(A[m, 1] - B[m, 1]))))
            res[f"knob_coherence_vs_diffuser_{tn}"] = d
            o.kv(f"knob: coherence vs diffuser, envelope {tn}",
                 f"max|z| {d['max_abs_z']:.2f}, chi2/dof {d['chi2_per_dof']:.2f}, max|diff| {d['max_abs_diff']:.3f}")
    for dn in ("diffuser1", "diffuser2"):
        a = os.path.join(inv, f"data_knobs_T1_gaussian_{dn}_intensity.txt")
        b = os.path.join(inv, f"data_knobs_T2_fringed_{dn}_intensity.txt")
        if os.path.exists(a) and os.path.exists(b):
            A, B = np.loadtxt(a), np.loadtxt(b)
            Xt = 0.015
            sel = np.abs(A[:, 0]) <= 2 * Xt
            z = (A[sel, 1] - B[sel, 1]) / np.sqrt(A[sel, 2] ** 2 + B[sel, 2] ** 2)
            wa = float(np.sqrt(np.sum(A[:, 0] ** 2 * A[:, 1]) / np.sum(A[:, 1])) * 1e3)
            wb = float(np.sqrt(np.sum(B[:, 0] ** 2 * B[:, 1]) / np.sum(B[:, 1])) * 1e3)
            d = dict(max_abs_z=float(np.max(np.abs(z))), chi2_per_dof=float(np.mean(z ** 2)),
                     width_T1_mm=wa, width_T2_mm=wb)
            res[f"knob_intensity_vs_envelope_{dn}"] = d
            o.kv(f"knob: intensity vs envelope, {dn} (|x| <= 30 mm)",
                 f"max|z| {d['max_abs_z']:.2f}, chi2/dof {d['chi2_per_dof']:.2f}, widths {wa:.2f} / {wb:.2f} mm")
    o.add("")
    for xc in ("0", "10"):
        f = os.path.join(inv, f"data_limits_spot_xc{xc}mm.txt")
        if os.path.exists(f):
            v = np.loadtxt(f)
            v = v[v[:, 0] > 0]
            d = dict(chi2_per_dof_vs_exact=float(np.mean(((v[:, 1] - v[:, 3]) / v[:, 2]) ** 2)),
                     rms_vs_target=float(np.sqrt(np.mean((v[:, 1] - v[:, 4]) ** 2))))
            res[f"limits_spot_xc{xc}mm"] = d
            o.kv(f"weak diffuser (MC), pairs at {xc} mm",
                 f"chi2/dof vs exact {d['chi2_per_dof_vs_exact']:.2f}, RMS vs target {d['rms_vs_target']:.3e}")
    sj = os.path.join(inv, "summary.json")
    if os.path.exists(sj):
        L = json.load(open(sj)).get("limits", {}).get("scan", [])
        if L:
            o.add("", "  Diffuser-depth scan (exact formula, max error vs target, pairs on axis):")
            for r in L:
                o.kv(f"    sig_phi = {r['sig_phi']:.3f} rad (specular {r['specular_fraction']:.1e})",
                     f"{r['max_err_xc0']:.2e}")
            res["limits_scan"] = L
    o.data["inverse_design"] = res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3", default="results_v3")
    ap.add_argument("--inv", default="results_invdesign")
    ap.add_argument("--out", default="results_post")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    o = Out()
    o.v3 = a.v3
    o.add("=" * 78, "DERIVED NUMBERS FOR THE SECT REVISION (from existing result folders)", "=" * 78,
          f"  generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
          f"  v3 folder : {os.path.abspath(a.v3)}",
          f"  inverse-design folder: {os.path.abspath(a.inv)}")
    S = json.load(open(os.path.join(a.v3, "summary.json")))
    R0 = read_R_trial(a.v3)
    o.kv("R_trial used in v3 (ident, fault, bayes)", R0)
    ident = part_identifiability(S, R0, o)
    part_scaling(ident, R0, o)
    part_faults(S, R0, o)
    part_bayes(S, o)
    if os.path.isdir(a.inv):
        part_inverse(a.inv, o)
    else:
        o.add("", f"  (inverse-design folder '{a.inv}' not found; section 5 skipped)")
    with open(os.path.join(a.out, "derived_numbers.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(o.lines) + "\n")
    with open(os.path.join(a.out, "derived_numbers.json"), "w") as fh:
        json.dump(o.data, fh, indent=2, default=float)
    print("\n".join(o.lines))


if __name__ == "__main__":
    main()
