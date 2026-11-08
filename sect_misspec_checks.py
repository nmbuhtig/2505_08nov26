#!/usr/bin/env python3
# -*- coding: utf-8 -*-
   
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
from scipy.stats import chi2, skew

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sect_mc_revision as M

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f} s] {m}", flush=True)


                                                                              
            
                                                                              
def cov_sqrt_general(N, dx, rho_fun, var):
    m = np.fft.fftfreq(N) * N * dx
    MX, MY = np.meshgrid(m, m)
    c = var * rho_fun(np.sqrt(MX ** 2 + MY ** 2))
    C = np.fft.fft2(c).real
    neg = float(np.sum(np.clip(C, None, 0) ** 2) / np.sum(C ** 2))
    return np.sqrt(np.clip(C, 0.0, None)), neg


def lognormal_transform(G, s):
                                                        
    mu = np.exp(s ** 2 / 2)
    sd = np.sqrt((np.exp(s ** 2) - 1) * np.exp(s ** 2))
    return (np.exp(s * G) - mu) / sd


def mc_fields_general(p, R, seed, A, sqC, transform=None, sig_h=None, batch=8, keep=0):
                                                                                 
    N, dx = p["N"], p["dx"]
    x = (np.arange(N) - N // 2) * dx
    rng = np.random.default_rng(seed)
    k2 = 2 * M.kk(p["lam"])
    u = np.empty((R, N), complex)
    kept = []
    for i0 in range(0, R, batch):
        nb = min(batch, R - i0)
        h = M.surfaces(rng, nb, N, sqC)
        if transform is not None:
            h = sig_h * transform(h)
        if keep and len(kept) * batch < keep:
            kept.append(h.copy())
        u[i0:i0 + nb] = (A * np.exp(1j * k2 * h)).sum(axis=1) * dx
    H = np.concatenate(kept)[:keep] if keep else None
    return x, u, H


def envelopes(p):
    N, dx, w = p["N"], p["dx"], p["w"]
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    nominal = np.exp(-r2 / (4 * w ** 2))
    q = 1.1
    ellip = np.exp(-(X ** 2 / (4 * (q * w) ** 2) + Y ** 2 / (4 * (w / q) ** 2)))
    trunc = nominal * (r2 <= (1.5 * w) ** 2)
                                                                         
                                                   
    rr = np.linspace(0, 10 * w, 20001)
    def rms_per_coord(a):
        I = np.exp(-(rr / a) ** 4)
        return np.sqrt(np.trapezoid(I * rr ** 3, rr) / np.trapezoid(I * rr, rr) / 2)            if hasattr(np, "trapezoid") else            np.sqrt(np.trapz(I * rr ** 3, rr) / np.trapz(I * rr, rr) / 2)
    lo, hi = 0.5 * w, 5 * w
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rms_per_coord(mid) < w:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    flat = np.exp(-0.5 * (np.sqrt(r2) / a) ** 4)                                   
    return {"nominal": nominal, "elliptical": ellip, "truncated": trunc, "flat-top": flat}, a


def fit_and_test(y, Sig, design, geo, th, rtol):
    W, _ = M.whitener(Sig, rtol)
    lt, H, phi = M.infer(y, W, design, geo, np.log(th), np.array([3.0, 3.0]))
    r = W @ (y - M.forward(np.exp(lt), design, geo))
    c = float(r @ r)
    m = W.shape[0]
    thr = float(chi2.ppf(0.99, m - 2))
    return dict(sig_h=float(np.exp(lt[0])), l_c=float(np.exp(lt[1])), chi2=c, modes=m,
                p99=thr, flag=bool(c > thr), p_value=float(chi2.sf(c, m - 2)))


                                                                              
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20, help="trials per truth model (Part 1)")
    ap.add_argument("--R", type=int, default=500, help="realizations per trial / scenario")
    ap.add_argument("--l_e", type=float, default=8e-6, help="exponential ACF length [m]")
    ap.add_argument("--skew_s", type=float, default=0.4, help="log-normal shape parameter")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results_misspec")
    a = ap.parse_args()
    if a.quick:
        a.trials, a.R = 3, 200
    os.makedirs(a.out, exist_ok=True)
    M.OUT = a.out
    p = dict(M.P0)
    th = np.array([p["sig_h"], p["l_c"]])
    N, dx = p["N"], p["dx"]
    design = M.make_designs(p)["B_offaxis_2z"]
    geo0 = M.nominal_geo(p)
    envs, a_flat = envelopes(p)
    S = {"params": dict(p), "args": vars(a)}

    rep = M.Report("MODEL-MISSPECIFICATION AND APERTURE-SHAPE CHECKS (design B)")
    rep.kv("command", " ".join(sys.argv))
    rep.kv("start", time.strftime("%Y-%m-%d %H:%M:%S"))
    rep.kv("python / numpy / platform", f"{sys.version.split()[0]} / {np.__version__} / {platform.platform()}")
    rep.kv("trials per truth (Part 1), R", f"{a.trials}, {a.R}")
    rep.kv("covariance cutoff rtol", f"{p['rtol']:g}")
    rep.add("", "  Inversion model in all cases: Gaussian ACF, Gaussian heights, Gaussian",
            "  envelope (w = 25 um), exact factorization; weak log-normal prior (s = 3)",
            "  centred at (50 nm, 8 um).  Test: chi2 at the MAP vs chi2(m - 2), 99% level.")

                                                                             
    sq_g, neg_g = cov_sqrt_general(N, dx, lambda r: np.exp(-r ** 2 / (2 * p["l_c"] ** 2)), p["sig_h"] ** 2)
    sq_e, neg_e = cov_sqrt_general(N, dx, lambda r: np.exp(-r / a.l_e), p["sig_h"] ** 2)
    sq_u, neg_u = cov_sqrt_general(N, dx, lambda r: np.exp(-r ** 2 / (2 * p["l_c"] ** 2)), 1.0)
    truths = {
        "G (matched control)": dict(sqC=sq_g, transform=None),
        f"E (exponential ACF, l_e = {a.l_e * 1e6:g} um)": dict(sqC=sq_e, transform=None),
        f"NG (skewed heights, s = {a.skew_s:g})": dict(sqC=sq_u, transform=lambda G: lognormal_transform(G, a.skew_s)),
    }
    rep.h("Part 1. Model misspecification")
    res1 = {}
    for ti, (tn, cfg) in enumerate(truths.items()):
        recs = []
        Hs = None
        for t in range(a.trials):
            log(f"part 1: {tn}, trial {t + 1}/{a.trials}")
            x, u, H = mc_fields_general(p, a.R, 7000 + 100 * ti + t, envs["nominal"], cfg["sqC"],
                                        cfg["transform"], p["sig_h"], keep=16 if t == 0 else 0)
            if H is not None:
                Hs = H
            y, Sig = M.design_data(x, u, design, p["lam"])
            recs.append(fit_and_test(y, Sig, design, geo0, th, p["rtol"]))
        sig = np.array([r["sig_h"] for r in recs])
        lc = np.array([r["l_c"] for r in recs])
        c2 = np.array([r["chi2"] for r in recs])
        m = recs[0]["modes"]
        d = dict(sample_sig_h_nm=float(Hs.std() * 1e9), sample_skewness=float(skew(Hs.ravel())),
                 median_sig_h_nm=float(np.median(sig) * 1e9),
                 bias_log_sig_h=float(np.mean(np.log(sig / p["sig_h"]))),
                 sd_log_sig_h=float(np.std(np.log(sig))),
                 median_l_c_um=float(np.median(lc) * 1e6),
                 mean_chi2=float(c2.mean()), dof=int(m - 2),
                 bias_log_combination=float(np.mean(1.63 * np.log(sig / p["sig_h"]) + np.log(lc / p["l_c"]))),
                 flag_rate=float(np.mean([r["flag"] for r in recs])),
                 modes=int(m))
        res1[tn] = d
        rep.add("", f"  Truth {tn}")
        rep.kv("    sample sigma_h [nm] / skewness (first trial)",
               f"{d['sample_sig_h_nm']:.2f} / {d['sample_skewness']:+.2f}")
        rep.kv("    median MAP sigma_h [nm] (true 50)", f"{d['median_sig_h_nm']:.2f}")
        rep.kv("    mean bias ln(sigma_h) / scatter sd", f"{d['bias_log_sig_h']:+.3f} / {d['sd_log_sig_h']:.3f}")
        rep.kv("    median MAP l_c [um] (Gaussian-model value)", f"{d['median_l_c_um']:.2f}")
        rep.kv("    mean bias ln(sigma_h^1.63 l_c) vs (50 nm, 8 um)", f"{d['bias_log_combination']:+.3f}")
        rep.kv("    mean chi2 at MAP / dof", f"{d['mean_chi2']:.1f} / {d['dof']}")
        rep.kv("    misspecification flagged (99% level)", f"{d['flag_rate']:.2f} of trials")
        M.save_table(f"data_misspec_part1_truth{ti}.txt",
                     ["trial", "sigma_h_MAP_m", "l_c_MAP_m", "chi2", "modes", "flag"],
                     [np.arange(a.trials), sig, lc, c2, [r["modes"] for r in recs],
                      [int(r["flag"]) for r in recs]])
    rep.add("", "  Notes: the exact factorization holds for any height autocorrelation, so",
            "  truth E misspecifies only the fitted parametric family; truth NG changes",
            "  the Kirchhoff factor chi itself.  Expected flag rate for the control: ~0.01.",
            "  sigma_h^1.63 l_c is the well-determined combination for design B (results_post);",
            "  for truth E its reference uses l_e in place of l_c (not the same quantity).",
            f"  Negative spectral weight clipped in generators: G {neg_g:.1e}, E {neg_e:.1e}, NG {neg_u:.1e}.")
    S["part1"] = res1

                                                                             
    rep.h("Part 2. Illumination / aperture shape (nominal Gaussian model)")
    rep.add(f"  flat-top: super-Gaussian intensity exp(-(r/a)^4), a = {a_flat * 1e6:.2f} um (same rms)",
            "  truncated: Gaussian clipped at r = 1.5 w; elliptical: w_x = 1.1 w, w_y = w/1.1")
    res2 = {}
    rows = []
    for si, (en, A) in enumerate(envs.items()):
        log(f"part 2: envelope {en}")
        x, u, _ = mc_fields_general(p, a.R, 8000 + si, A, sq_g)
        y, Sig = M.design_data(x, u, design, p["lam"])
        d = fit_and_test(y, Sig, design, geo0, th, p["rtol"])
        res2[en] = d
        rows.append([en, d["sig_h"] * 1e9, d["l_c"] * 1e6, d["chi2"], d["modes"] - 2, d["p99"],
                     "YES" if d["flag"] else "no"])
    rep.table(["envelope", "fit sig nm", "fit lc um", "chi2 after fit", "dof", "p99", "non-surface flag"],
              rows, ["{}", "{:.2f}", "{:.2f}", "{:.3g}", "{}", "{:.2f}", "{}"])
    S["part2"] = res2

                                                                             
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    names = list(res1)
    for i, tn in enumerate(names):
        d = np.loadtxt(os.path.join(a.out, f"data_misspec_part1_truth{i}.txt"), ndmin=2)
        axs[0].plot(d[:, 1] * 1e9, d[:, 2] * 1e6, "os^"[i], mfc="none", ms=4, label=tn)
    axs[0].plot(50, 8, "k*", ms=10, label="nominal (50 nm, 8 um)")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("MAP sigma_h [nm]"); axs[0].set_ylabel("MAP l_c [um] (Gaussian model)")
    axs[0].set_title("Part 1: estimates under misspecification"); axs[0].legend(fontsize=7)
    labels = [f"P1: {n.split(' ')[0]}" for n in names] + [f"P2: {e}" for e in res2]
    vals = [res1[n]["mean_chi2"] for n in names] + [res2[e]["chi2"] for e in res2]
    thr = [chi2.ppf(0.99, res1[n]["dof"]) for n in names] + [res2[e]["p99"] for e in res2]
    cols = ["#c55a11" if v > t else "#1f4e79" for v, t in zip(vals, thr)]
    xs = np.arange(len(vals))
    axs[1].bar(xs, vals, color=cols)
    axs[1].plot(xs, thr, "k_", ms=18, mew=2, label="99% null threshold")
    axs[1].set_yscale("log")
    axs[1].set_xticks(xs); axs[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axs[1].set_ylabel("chi2 after re-fit (Part 1: mean)")
    axs[1].set_title("goodness of fit"); axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_misspec.png"), dpi=150)
    fig.savefig(os.path.join(a.out, "fig_misspec.pdf"))
    plt.close(fig)

    rep.h("Run time")
    rep.kv("total [s]", f"{time.time() - T0:.1f}")
    rep.write("misspec_report.txt")
    json.dump(S, open(os.path.join(a.out, "summary_misspec.json"), "w"), indent=2, default=float)
    log("finished")


if __name__ == "__main__":
    main()
