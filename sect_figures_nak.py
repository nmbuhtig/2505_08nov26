#!/usr/bin/env python3
# -*- coding: utf-8 -*-
   
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_MC, C_EXACT, C_OLD, C_ENV, C_ALT = "#1f4e79", "#c55a11", "#7f7f7f", "#548235", "#7030a0"
W2 = 7.0                                
OUT = "results_figures"
INDEX = []


def save(fig, stem, caption, sources):
    fig.savefig(os.path.join(OUT, stem + ".pdf"))
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=300)
    plt.close(fig)
    INDEX.append((stem, caption, sources))
    print(f"wrote {stem}.pdf / .png")


def panel_label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def load(path):
    return np.loadtxt(path)


                                                                              
def fig_factorization(v3):
    fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0))
    files = [("A_onaxis_2z", "z1", "a"), ("A_onaxis_2z", "z2", "b"),
             ("B_offaxis_2z", "z1", "c"), ("B_offaxis_2z", "z2", "d")]
    S = json.load(open(os.path.join(v3, "summary.json")))
    z = {"z1": S["params"]["z1"], "z2": S["params"]["z2"]}
    srcs = []
    for ax, (dn, blk, lab) in zip(axs.flat, files):
        f = os.path.join(v3, f"data_validation_{dn}_{blk}.txt")
        srcs.append(os.path.basename(f))
        d = load(f)
        X = d[:, 0] * 1e3
        ax.plot(X, d[:, 6], ":", color=C_ENV, label="illumination (VCZ) factor only")
        if d.shape[1] > 7:
            ax.plot(X, d[:, 7], "--", color=C_OLD, label="submitted Eq. (S69)")
        ax.plot(X, d[:, 5], "-", color=C_EXACT, label="exact factorization")
        ax.errorbar(X, d[:, 3], yerr=1.96 * d[:, 4], fmt="o", ms=2.5, color=C_MC, elinewidth=0.8,
                    label="phase-screen MC (95%)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(r"$|\gamma|$")
        if dn.startswith("A"):
            ax.set_xlabel(r"pinhole separation $b$ [mm]")
            ax.set_title(f"on-axis pairs, $z$ = {z[blk]} m")
        else:
            ax.set_xlabel(r"pair centre $x_0$ [mm] (separation 2 mm)")
            ax.set_title(f"off-axis pairs, $z$ = {z[blk]} m")
        panel_label(ax, lab)
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[0, 1].legend(h, l, loc="lower left")                                  
    axs[1, 0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_factorization",
         "Far-field degree of coherence for a Gaussian-correlated Kirchhoff surface "
         "(sigma_h = 50 nm, l_c = 8 um, lambda = 500 nm) under Gaussian illumination "
         "(w = 25 um). Points: independent phase-screen Monte Carlo (R = 2000; 95% bars). "
         "Solid: exact factorization. Dashed: submitted Eq. (S69). Dotted: illumination "
         "(VCZ) factor alone. (a,b) on-axis pairs (submitted geometry); (c,d) off-axis "
         "pairs: the coherence dip marks the specular/diffuse crossover, beyond which the "
         "coherence reduces to the illumination factor and carries no surface information.",
         srcs)


def fig_inverse(inv):
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.8))
    srcs = []
    names = {"T1_gaussian": "Gaussian", "T2_fringed": "fringed"}
    for i, tn in enumerate(("T1_gaussian", "T2_fringed")):
        for j, xc in enumerate(("0", "10")):
            f = os.path.join(inv, f"data_validate_{tn}_xc{xc}mm.txt")
            srcs.append(os.path.basename(f))
            d = load(f)
            ax = axs[i, j]
            ax.plot(d[:, 0] * 1e3, d[:, 4], "-", color=C_EXACT, label="target")
            ax.errorbar(d[:, 0] * 1e3, d[:, 1], yerr=1.96 * d[:, 3], fmt="o", ms=2.5, color=C_MC,
                        elinewidth=0.8, label="MC (95%)")
            ax.set_xlabel(r"separation $\Delta x$ [mm]")
            ax.set_ylabel(r"Re $\gamma$")
            ax.set_title(f"{names[tn]}, " + ("on axis" if xc == "0" else rf"$x_c$ = {xc} mm"))
            panel_label(ax, "abcdef"[3 * i + j])
        f = os.path.join(inv, f"data_validate_{tn}_intensity.txt")
        srcs.append(os.path.basename(f))
        d = load(f)
        ax = axs[i, 2]
        x = d[:, 0] * 1e3
        ax.fill_between(x, d[:, 1] - 1.96 * d[:, 2], d[:, 1] + 1.96 * d[:, 2], color=C_MC, alpha=0.25,
                        lw=0, label="MC 95% band")
        ax.plot(x, d[:, 1], "-", color=C_MC, lw=0.7, label="MC")
        ax.plot(x, d[:, 3], "-", color=C_EXACT, label="target")
        ax.set_xlabel(r"detector position $x$ [mm]")
        ax.set_ylabel(r"$I/I(0)$")
        ax.set_title("far-field intensity")
        ax.set_ylim(0, 1.45)
        panel_label(ax, "abcdef"[3 * i + 2])
    axs[0, 0].legend()
    axs[0, 2].legend(loc="upper left", fontsize=6.3, frameon=False)
    fig.tight_layout()
    save(fig, "fig_inverse_design",
         "End-to-end inverse design. Coherence targets (rows: Gaussian, rho = 1 mm; fringed, "
         "rho = 2.5 mm, period 2 mm) are realized by the designed illumination envelope "
         "(inverse VCZ; two spots for the fringed target), and the intensity target "
         "(Gaussian, 15 mm rms at z = 0.5 m) by the designed phase diffuser (sigma_phi = 2 pi, "
         "l_phi = 16.57 um). Points: independent phase-screen MC (R = 2000). Coherence is "
         "shown for pinhole pairs centred on axis and 10 mm off axis.", srcs)


def fig_knobs(inv):
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    srcs = []
    sty = {("T1_gaussian", "diffuser1"): ("-", C_MC), ("T1_gaussian", "diffuser2"): ("--", C_MC),
           ("T2_fringed", "diffuser1"): ("-", C_EXACT), ("T2_fringed", "diffuser2"): ("--", C_EXACT)}
    lab_e = {"T1_gaussian": "envelope T1", "T2_fringed": "envelope T2"}
    lab_d = {"diffuser1": r"$\ell_\phi$ = 16.6 $\mu$m", "diffuser2": r"$\ell_\phi$ = 9.9 $\mu$m"}
    for (tn, dn), (ls, c) in sty.items():
        f1 = os.path.join(inv, f"data_knobs_{tn}_{dn}_coherence.txt")
        f2 = os.path.join(inv, f"data_knobs_{tn}_{dn}_intensity.txt")
        srcs += [os.path.basename(f1), os.path.basename(f2)]
        a, b = load(f1), load(f2)
        axs[0].plot(a[:, 0] * 1e3, a[:, 1], ls, color=c, label=f"{lab_e[tn]}, {lab_d[dn]}")
        axs[1].plot(b[:, 0] * 1e3, b[:, 1], ls, color=c, lw=0.9, label=f"{lab_e[tn]}, {lab_d[dn]}")
    axs[0].set_xlabel(r"separation $\Delta x$ [mm]")
    axs[0].set_ylabel(r"Re $\gamma$")
    axs[0].set_title("coherence: set by the illumination")
    axs[1].set_xlabel(r"detector position $x$ [mm]")
    axs[1].set_ylabel(r"$I/I(0)$")
    axs[1].set_title("intensity: set by the surface")
    axs[0].legend(fontsize=6.5)
    panel_label(axs[0], "a")
    panel_label(axs[1], "b")
    fig.tight_layout()
    save(fig, "fig_knobs",
         "Separation of the two design knobs (phase-screen MC, R = 2000 per case). Changing "
         "the diffuser correlation length (solid vs dashed) leaves the coherence unchanged "
         "within noise (max |z| = 1.5, 1.7), while changing the illumination envelope "
         "(colours) leaves the intensity profile unchanged (max |z| = 2.7, 2.2); each knob "
         "alters its own quantity strongly.", srcs)


def fig_limits(inv):
    S = json.load(open(os.path.join(inv, "summary.json")))
    L = S["limits"]["scan"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4))
    sp = np.array([r["sig_phi"] for r in L])
    e0 = np.array([r["max_err_xc0"] for r in L])
    sf = np.array([r["specular_fraction"] for r in L])
    ax = axs[0]
    ax.semilogy(sp, e0, "o-", color=C_EXACT, ms=3, label="max error (on axis)")
    ax.semilogy(sp, sf, "s:", color=C_OLD, ms=3, label=r"specular fraction $e^{-\sigma_\phi^2}$")
    ax.axhline(0.01, ls="--", color="k", lw=0.7)
    ax.set_xlabel(r"diffuser phase rms $\sigma_\phi$ [rad]")
    ax.set_ylabel("error / fraction")
    ax.legend(fontsize=6.5)
    ax.set_ylim(1e-5, 2)
    ax.set_title("accuracy vs diffuser depth")
    panel_label(ax, "a")
    srcs = ["summary.json (limits.scan)"]
    for j, xc in enumerate(("0", "10")):
        f = os.path.join(inv, f"data_limits_spot_xc{xc}mm.txt")
        srcs.append(os.path.basename(f))
        d = load(f)
        ax = axs[j + 1]
        ax.plot(d[:, 0] * 1e3, d[:, 4], ":", color=C_ENV, label="design target")
        ax.plot(d[:, 0] * 1e3, d[:, 3], "-", color=C_EXACT, label="exact formula")
        ax.errorbar(d[:, 0] * 1e3, d[:, 1], yerr=1.96 * d[:, 2], fmt="o", ms=2, color=C_MC,
                    elinewidth=0.7, label="MC (95%)")
        ax.set_xlabel(r"separation $\Delta x$ [mm]")
        ax.set_ylabel(r"Re $\gamma$")
        ax.set_title(rf"$\sigma_\phi$ = 1 rad, pairs at {xc} mm")
        panel_label(ax, "bc"[j])
    axs[1].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_limits",
         "Limits of the design rule. (a) Maximum coherence error (exact formula, pairs on "
         "axis) versus diffuser phase rms at fixed l_phi; the dashed line marks 1%. (b,c) A "
         "shallow diffuser (sigma_phi = 1 rad, specular fraction 0.37): MC agrees with the "
         "exact formula, but the target is met only off axis, outside the specular lobe.", srcs)


def fig_inference(v3):
    S = json.load(open(os.path.join(v3, "summary.json")))
    p = S["params"]
    th = np.array([p["sig_h"], p["l_c"]])
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    srcs = []
    for pn, mk, c in (("centred", "o", C_MC), ("offset", "^", C_EXACT)):
        f = os.path.join(v3, f"data_bayes_trials_B_offaxis_2z_{pn}.txt")
        srcs.append(os.path.basename(f))
        d = load(f)
        ax.plot(d[:, 1] * 1e9, d[:, 2] * 1e6, mk, ms=3.5, mfc="none", color=c, label=f"MAP, {pn} prior")
    v1 = np.array(S["ident"]["B_offaxis_2z"]["both distances"]["full_cov"]["best_dir"])
    if v1[1] < 0:
        v1 = -v1
    a_exp = v1[0] / v1[1]
    allsig = []
    for pn in ("centred", "offset"):
        allsig += list(load(os.path.join(v3, f"data_bayes_trials_B_offaxis_2z_{pn}.txt"))[:, 1])
    sg = np.logspace(np.log10(min(allsig)) - 0.08, np.log10(max(allsig)) + 0.08, 50)
    lc = th[1] * (sg / th[0]) ** (-a_exp)
    ax.plot(sg * 1e9, lc * 1e6, "-", color=C_ENV, lw=1,
            label=rf"$\sigma_h^{ {a_exp:.2f}} \,\ell_c$ = true value")
    ax.plot(th[0] * 1e9, th[1] * 1e6, "k*", ms=9, label="truth")
    ax.plot(1.5 * th[0] * 1e9, 0.6 * th[1] * 1e6, "x", color=C_EXACT, ms=6, label="offset prior mean")
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([30, 40, 50, 60, 80, 100]))
    ax.yaxis.set_major_locator(FixedLocator([2, 3, 4, 6, 8, 10, 15, 20]))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\sigma_h$ [nm]")
    ax.set_ylabel(r"$\ell_c$ [$\mu$m]")
    ax.set_title("design B: MAP estimates (40 trials)")
    ax.legend(fontsize=6.3)
    panel_label(ax, "a")
    ax = axs[1]
    keys = list(S["bayes"].keys())
    lab = [k.replace("_onaxis_2z", "").replace("_offaxis_2z", "").replace("_scaled_2z", "").replace("|", ", ")
           for k in keys]
    xs = np.arange(len(keys))
    lr = [S["bayes"][k]["coverage95_LR"] for k in keys]
    lp = [S["bayes"][k]["coverage95"] for k in keys]
    ax.bar(xs - 0.18, lr, 0.36, color=C_MC, label="likelihood-ratio region")
    ax.bar(xs + 0.18, lp, 0.36, color=C_OLD, label="Laplace ellipse")
    n = len(load(os.path.join(v3, "data_bayes_trials_B_offaxis_2z_centred.txt")))
    se = np.sqrt(0.95 * 0.05 / n)
    ax.axhspan(0.95 - 2 * se, min(1.0, 0.95 + 2 * se), color=C_ENV, alpha=0.15, lw=0)
    ax.axhline(0.95, ls="--", color="k", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("95% coverage")
    ax.set_title("calibration (nominal 0.95)")
    ax.legend(fontsize=6.5, loc="lower left")
    panel_label(ax, "b")
    srcs.append("summary.json (bayes, ident)")
    fig.tight_layout()
    save(fig, "fig_inference",
         "Bayesian inference on independent phase-screen MC data (R = 500 per trial, 40 "
         "trials). (a) Design B MAP estimates scatter along the curve on which the "
         "well-determined combination sigma_h^a l_c is constant (a from the leading "
         "singular vector). (b) Empirical coverage of nominal 95% regions for designs A, B, "
         "C and two priors; the shaded band is +/- 2 binomial SE. Likelihood-ratio regions "
         "are calibrated; Laplace ellipses undercover because the posterior is curved.", srcs)


def fig_faults(v3):
    S = json.load(open(os.path.join(v3, "summary.json")))
    sc = S["fault"]["scenarios"]["B_offaxis_2z"]
    names = list(sc.keys())
    short = {"no fault": "no fault", "surface sig_h +10%, l_c +10%": "surface +10%",
             "surface sig_h -20%": r"surface $\sigma_h$ $-$20%", "z2 +3%": r"$z_2$ +3%",
             "lambda +2%": r"$\lambda$ +2%", "w +10%": r"$w$ +10%", "decentre 0.3 mm (both)": "decentre 0.3 mm"}
    xs = np.arange(len(names))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
    ref = [sc[n]["refit_chi2"] for n in names]
    ep = [max(sc[n]["E_P"], 1e-2) for n in names]
    p99 = sc[names[0]]["refit_p99"]
    p99l = sc[names[0]]["p99_E_P"]
    m = sc[names[0]]["modes"]
    for ax, vals, thr, title, lab in ((axs[0], ref, p99, "nonlinear test (re-fit surface first)", "a"),
                                      (axs[1], ep, p99l, r"linear projection $E_P$ at nominal $\theta^\ast$", "b")):
        cols = [C_EXACT if v > thr else C_MC for v in vals]
        ax.bar(xs, vals, 0.6, color=cols)
        ax.axhline(thr, ls="--", color="k", lw=0.8, label=f"99% null ({m - 2} dof)")
        ax.set_yscale("log")
        ax.set_ylim(5, 1.3 * max(max(ref), max(ep)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(xs)
        ax.set_xticklabels([short.get(n, n) for n in names], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(r"whitened $\chi^2$")
        ax.set_title(title)
        ax.legend(fontsize=6.5, loc="upper left")
        panel_label(ax, lab)
    fig.tight_layout()
    save(fig, "fig_faults",
         "Fault isolation, design B (phase-screen MC data, R = 500; whitened residuals, "
         f"{m} noise modes). (a) After re-fitting the surface parameters, surface changes are "
         "absorbed (no alarm), while envelope (w) and decentre faults are flagged; z2 +3% "
         "and lambda +2% are below the detection limits at this R. (b) The linear "
         "projection at the nominal parameters raises a false alarm for a 10% surface "
         "change, because of the curvature of the forward map, and misses the 0.3 mm "
         "decentre. Red bars exceed the 99% null.",
         ["summary.json (fault.scenarios)"])


def figs_extra(ex):
    srcs = []
    fa = [os.path.join(ex, f"data_extra_aperture_{b}.txt") for b in ("z1", "z2")]
    if all(os.path.exists(f) for f in fa):
        S = json.load(open(os.path.join(ex, "summary_extra.json")))
        fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5))
        for ax, f, zz, lab in zip(axs, fa, ("0.8", "2.5"), "ab"):
            d = load(f)
            ax.plot(d[:, 0] * 1e3, np.abs(d[:, 4]), ":", color=C_ENV, label=r"incoherent disk $|2J_1(v)/v|$")
            ax.plot(d[:, 0] * 1e3, d[:, 3], "--", color=C_OLD, label="submitted Eq. (S69)")
            ax.errorbar(d[:, 0] * 1e3, d[:, 1], yerr=1.96 * d[:, 2], fmt="o", ms=2.5, color=C_MC,
                        elinewidth=0.8, label="MC, same hard aperture")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel(r"pinhole separation $b$ [mm]")
            ax.set_ylabel(r"$|\gamma|$")
            ax.set_title(rf"$z$ = {zz} m, $R_A$ = 50 $\mu$m")
            panel_label(ax, lab)
        axs[0].legend(fontsize=6.5)
        fig.tight_layout()
        save(fig, "figS_aperture",
             "Like-for-like test of the submitted Eq. (S69): phase-screen MC with the same "
             "uniform circular aperture (R_A = 50 um) assumed by S69. S69 follows the "
             "coherence of an incoherent disk (dotted) because it weights the surface "
             "correlation factor as a source intensity.", [os.path.basename(f) for f in fa])
    ft = os.path.join(ex, "data_extra_tilt_shift.txt")
    if os.path.exists(ft):
        S = json.load(open(os.path.join(ex, "summary_extra.json")))
        d = load(ft)
        fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5))
        for bi, (ax, zz, lab) in enumerate(zip(axs, ("0.8", "2.5"), "ab")):
            s = d[:, 1] == bi
            ax.plot(d[s, 0] * 1e3, d[s, 5], ":", color="k", label="nominal model")
            ax.plot(d[s, 0] * 1e3, d[s, 3], "-", color=C_EXACT, label=r"model, decentre $-z\theta$")
            ax.plot(d[s, 0] * 1e3, d[s, 2], "o", ms=2.5, color=C_MC, label="MC, tilted illumination")
            ax.plot(d[s, 0] * 1e3, d[s, 4], "s", ms=2.5, mfc="none", color=C_ALT, label="MC, shifted spot")
            ax.set_xlabel(r"pair centre $x_0$ [mm]")
            ax.set_ylabel(r"$|\gamma|$")
            ax.set_title(f"design B, $z$ = {zz} m")
            panel_label(ax, lab)
        axs[0].legend(fontsize=6.3)
        fig.tight_layout()
        B, C = S["B_tilt"], S["C_shift"]
        save(fig, "figS_tilt_shift",
             f"Illumination tilt (theta = {B['theta'] * 1e3:g} mrad; a surface tilt alpha acts as "
             f"theta = 2 alpha) is exactly equivalent to decentring the pinholes by -z theta "
             f"(chi2 = {B['chi2_decentre']:.1f} for {B['modes']} modes, vs {B['chi2_nominal']:.0f} "
             f"for the nominal model). A lateral spot shift of {C['s0'] * 1e6:g} um leaves |gamma| "
             f"unchanged (chi2 = {C['chi2_nominal']:.1f} for {C['modes']} modes).", [os.path.basename(ft)])


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3", default="results_v3")
    ap.add_argument("--inv", default="results_invdesign")
    ap.add_argument("--extra", default="results_extra")
    ap.add_argument("--out", default="results_figures")
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    fig_factorization(a.v3)
    fig_inference(a.v3)
    fig_faults(a.v3)
    if os.path.isdir(a.inv):
        fig_inverse(a.inv)
        fig_knobs(a.inv)
        fig_limits(a.inv)
    if os.path.isdir(a.extra):
        figs_extra(a.extra)
    with open(os.path.join(OUT, "figures_index.txt"), "w", encoding="utf-8") as fh:
        fh.write("FIGURES FOR THE SECT REVISION (draft captions; edit before use)\n")
        fh.write("=" * 70 + "\n\n")
        for stem, cap, srcs in INDEX:
            fh.write(f"{stem}.pdf\n  sources: {', '.join(srcs)}\n  caption: {cap}\n\n")
    print("wrote figures_index.txt")


if __name__ == "__main__":
    main()
