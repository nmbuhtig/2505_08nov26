#!/usr/bin/env python3
# -*- coding: utf-8 -*-
   
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
from scipy.stats import chi2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sect_mc_revision as M

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f} s] {m}", flush=True)


def mc_fields_amp(p, Afun, R, seed, batch=16):
                                                                               
    N, dx = p["N"], p["dx"]
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    A = Afun(X, Y)
    sqC = M.cov_sqrt(N, dx, p["l_c"], p["sig_h"])
    rng = np.random.default_rng(seed)
    k2 = 2 * M.kk(p["lam"])
    u = np.empty((R, N), complex)
    for i0 in range(0, R, batch):
        nb = min(batch, R - i0)
        h = M.surfaces(rng, nb, N, sqC)
        u[i0:i0 + nb] = (A * np.exp(1j * k2 * h)).sum(axis=1) * dx
    return x, u


def chi2_white(y, f, Sig, rtol):
    W, _ = M.whitener(Sig, rtol)
    r = W @ (y - f)
    m = W.shape[0]
    c = float(r @ r)
    return c, m, float(chi2.sf(c, m))


def save_fig(fig, out, stem):
    fig.savefig(os.path.join(out, stem + ".png"), dpi=150)
    fig.savefig(os.path.join(out, stem + ".pdf"))
    plt.close(fig)
    log(f"wrote {stem}.png / .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--R", type=int, default=2000, help="realizations for check A")
    ap.add_argument("--R_bc", type=int, default=1000, help="realizations for checks B and C")
    ap.add_argument("--R_noise", type=int, default=500, help="realizations for the check-D covariance")
    ap.add_argument("--theta", type=float, default=0.4e-3, help="illumination tilt [rad]")
    ap.add_argument("--s0", type=float, default=20e-6, help="spot shift [m]")
    ap.add_argument("--R_A", type=float, default=50e-6, help="hard aperture radius [m]")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results_extra")
    a = ap.parse_args()
    if a.quick:
        a.R, a.R_bc, a.R_noise = 300, 200, 200
    os.makedirs(a.out, exist_ok=True)
    M.OUT = a.out
    p = dict(M.P0)
    th = np.array([p["sig_h"], p["l_c"]])
    designs = M.make_designs(p)
    geo0 = M.nominal_geo(p)
    rep = M.Report("EXTRA CHECKS: hard-aperture S69 test, tilt, spot shift, shared signature")
    rep.kv("command", " ".join(sys.argv))
    rep.kv("start", time.strftime("%Y-%m-%d %H:%M:%S"))
    rep.kv("python / numpy", f"{sys.version.split()[0]} / {np.__version__}")
    rep.kv("platform", platform.platform())
    rep.kv("R (A), R (B, C), R (D noise)", f"{a.R}, {a.R_bc}, {a.R_noise}")
    rep.kv("covariance cutoff rtol", f"{p['rtol']:g}")
    S = {}

                                                                          
    log("A: hard-aperture MC")
    RA = a.R_A
    xA, uA = mc_fields_amp(p, lambda X, Y: (X ** 2 + Y ** 2 <= RA ** 2).astype(float), a.R, seed=901)
    dA = designs["A_onaxis_2z"]
    rep.h(f"A. Submitted Eq. (S69) vs MC with the SAME hard aperture (R_A = {RA * 1e6:g} um)")
    rep.add("  Uniform amplitude inside a circular aperture of radius R_A (as assumed",
            "  by S69); Kirchhoff phase screen; design-A on-axis pairs; |gamma| from MC.")
    resA = {}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    nb = len(dA[0]["x1"])
    for bi, blk in enumerate(dA):
        g, psi = M.gamma_influence(xA, uA, blk["x1"], blk["x2"], blk["z"], p["lam"])
        se = np.sqrt(np.mean(psi ** 2, axis=0) / uA.shape[0])
        old = M.old_S69(blk["absc"], blk["z"], p["sig_h"], p["l_c"], p["lam"], R_A=RA)
                                                                                                 
        from scipy.special import j1
        v = 2 * np.pi * RA * blk["absc"] / (p["lam"] * blk["z"])
        vcz = np.where(v > 0, 2 * j1(v) / np.where(v > 0, v, 1), 1.0)
        d = dict(rms_MC_minus_S69=float(np.sqrt(np.mean((g - old) ** 2))),
                 max_MC_minus_S69=float(np.max(np.abs(g - old))),
                 mean_SE=float(se.mean()),
                 MC_min_gamma=float(g.min()), S69_min_gamma=float(old.min()))
        resA[blk["name"]] = d
        rep.kv(f"  z = {blk['z']} m: RMS |MC - S69|", f"{d['rms_MC_minus_S69']:.3f}")
        rep.kv(f"  z = {blk['z']} m: max |MC - S69|", f"{d['max_MC_minus_S69']:.3f}")
        rep.kv(f"  z = {blk['z']} m: mean SE (MC)", f"{d['mean_SE']:.2e}")
        rep.kv(f"  z = {blk['z']} m: min |gamma| MC / S69", f"{d['MC_min_gamma']:.3f} / {d['S69_min_gamma']:.3f}")
        M.save_table(f"data_extra_aperture_{blk['name']}.txt",
                     ["b_m", "gamma_MC", "SE", "gamma_S69", "aperture_VCZ_2J1v_over_v"],
                     [blk["absc"], g, se, old, vcz])
        ax = axs[bi]
        ax.errorbar(blk["absc"] * 1e3, g, yerr=1.96 * se, fmt="o", ms=3, label="MC, hard aperture")
        ax.plot(blk["absc"] * 1e3, old, "--", label="submitted Eq. (S69)")
        ax.plot(blk["absc"] * 1e3, np.abs(vcz), ":", label="|2J1(v)/v| (incoherent disk)")
        ax.set_xlabel("pinhole separation b [mm]")
        ax.set_ylabel(r"$|\gamma|$")
        ax.set_title(f"z = {blk['z']} m, R_A = {RA * 1e6:g} um")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, a.out, "fig_extra_aperture")
    rep.add("", "  The dotted curve is the VCZ coherence of a fully incoherent uniform",
            "  disk of the same radius; S69 behaves like it because it weights the",
            "  surface correlation factor as if it were a source-intensity profile.")
    S["A_hard_aperture"] = resA

                                                                             
    dB = designs["B_offaxis_2z"]
    w = p["w"]
    k = M.kk(p["lam"])
    gauss = lambda X, Y: np.exp(-(X ** 2 + Y ** 2) / (4 * w ** 2))
    log("B: tilted illumination MC")
    xt, ut = mc_fields_amp(p, lambda X, Y: gauss(X, Y) * np.exp(1j * k * a.theta * X), a.R_bc, seed=902)
    yt, St = M.design_data(xt, ut, dB, p["lam"])
    f_nom = M.forward(th, dB, geo0)
    geo_t = dict(geo0, dl=[-dB[0]["z"] * a.theta, -dB[1]["z"] * a.theta])
    f_dec = M.forward(th, dB, geo_t)
    c_nom, m_t, p_nom = chi2_white(yt, f_nom, St, p["rtol"])
    c_dec, _, p_dec = chi2_white(yt, f_dec, St, p["rtol"])
    rep.h(f"B. Illumination tilt theta = {a.theta * 1e3:g} mrad (surface tilt alpha = theta/2)")
    rep.add("  Exact identity: exp(i 2 k alpha x) = exp(i k theta x) with theta = 2 alpha.",
            f"  Predicted equivalent decentre: dl = -z theta = {-dB[0]['z'] * a.theta * 1e3:.3f} mm (z1),",
            f"  {-dB[1]['z'] * a.theta * 1e3:.3f} mm (z2).  Whitened chi2 of MC vs model:")
    rep.kv("  vs nominal model (no decentre)", f"chi2 = {c_nom:.1f}, modes = {m_t}, p = {p_nom:.2e}")
    rep.kv("  vs model with dl = -z theta", f"chi2 = {c_dec:.1f}, modes = {m_t}, p = {p_dec:.3f}")
    S["B_tilt"] = dict(theta=a.theta, chi2_nominal=c_nom, chi2_decentre=c_dec, modes=m_t,
                       p_nominal=p_nom, p_decentre=p_dec)

    log("C: shifted illumination spot MC")
    xs, us = mc_fields_amp(p, lambda X, Y: gauss(X - a.s0, Y), a.R_bc, seed=903)
    ys, Ss = M.design_data(xs, us, dB, p["lam"])
    c_s, m_s, p_s = chi2_white(ys, f_nom, Ss, p["rtol"])
    edge = (p["N"] * p["dx"] / 2 - a.s0)
    rep.h(f"C. Lateral spot shift s0 = {a.s0 * 1e6:g} um on a stationary surface")
    rep.add("  Prediction: |gamma| unchanged (deterministic phase only).",
            f"  Envelope amplitude at the nearest domain edge: {np.exp(-edge ** 2 / (4 * w ** 2)):.1e}")
    rep.kv("  vs nominal model", f"chi2 = {c_s:.1f}, modes = {m_s}, p = {p_s:.3f}")
    S["C_shift"] = dict(s0=a.s0, chi2_nominal=c_s, modes=m_s, p_nominal=p_s)

    nbB = len(dB[0]["x1"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for bi, blk in enumerate(dB):
        sl = slice(bi * nbB, (bi + 1) * nbB)
        ax = axs[bi]
        ax.plot(blk["absc"] * 1e3, yt[sl], "o", ms=3, label=f"MC, tilt {a.theta * 1e3:g} mrad")
        ax.plot(blk["absc"] * 1e3, f_dec[sl], "-", label="model, decentre -z theta")
        ax.plot(blk["absc"] * 1e3, ys[sl], "s", ms=3, mfc="none", label=f"MC, spot shift {a.s0 * 1e6:g} um")
        ax.plot(blk["absc"] * 1e3, f_nom[sl], "k:", label="nominal model")
        ax.set_xlabel("pair centre x0 [mm]")
        ax.set_ylabel(r"$|\gamma|$")
        ax.set_title(f"design B, z = {blk['z']} m")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, a.out, "fig_extra_tilt_shift")
    M.save_table("data_extra_tilt_shift.txt",
                 ["x0_m", "block", "gamma_MC_tilt", "model_decentre", "gamma_MC_shift", "model_nominal"],
                 [np.concatenate([dB[0]["absc"], dB[1]["absc"]]),
                  np.concatenate([np.zeros(nbB), np.ones(nbB)]), yt, f_dec, ys, f_nom])

                                                                          
    log("D: shared fault signature (noise covariance as in v3 fault stage)")
    x, u, _ = M.mc_fields(p, a.R_noise, seed=21)
    rep.h("D. Shared residual signature of z, lambda and w faults")
    rep.add("  cos = |<P_perp v_i, P_perp v_j>| / (|P_perp v_i| |P_perp v_j|) for the",
            "  whitened fault directions (1 = identical signature).  Also listed: the",
            "  cosine between P_perp v_z and P_perp v_w predicted to be exactly 1 by the",
            "  scaling symmetry (lambda z) <-> (w, l_c).")

    def geo_mod(**kw):
        g = dict(lam=geo0["lam"], w=geo0["w"], zs=list(geo0["zs"]), dl=list(geo0["dl"]))
        g.update(kw)
        return g

    faults = {"z (both)": lambda e: geo_mod(zs=[1 + e, 1 + e]),
              "lambda": lambda e: geo_mod(lam=geo0["lam"] * (1 + e)),
              "w": lambda e: geo_mod(w=geo0["w"] * (1 + e)),
              "z2 only": lambda e: geo_mod(zs=[1.0, 1 + e]),
              "decentre (both)": lambda e: geo_mod(dl=[e * 1e-3, e * 1e-3])}
    resD = {}
    for dn in ("B_offaxis_2z", "C_scaled_2z"):
        design = designs[dn]
        _, Sig = M.design_data(x, u, design, p["lam"])
        W, _ = M.whitener(Sig, p["rtol"])
        Js = W @ M.jac_log(lambda t: M.forward(t, design, geo0), th)
        Q, _ = np.linalg.qr(Js)
        Pp = np.eye(W.shape[0]) - Q @ Q.T
        vp = {}
        for fn, gf in faults.items():
            e = 1e-4
            v = W @ ((M.forward(th, design, gf(e)) - M.forward(th, design, gf(-e))) / (2 * e))
            vp[fn] = Pp @ v
        names = list(faults)
        cos = np.zeros((len(names), len(names)))
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                a_, b_ = vp[ni], vp[nj]
                cos[i, j] = abs(a_ @ b_) / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-300)
        resD[dn] = dict(names=names, cos=cos.tolist(), modes=int(W.shape[0]),
                        norms={n: float(np.linalg.norm(vp[n])) for n in names})
        rep.add("", f"  Design {dn} (modes {W.shape[0]}); |P_perp v|: " +
                ", ".join(f"{n} {np.linalg.norm(vp[n]):.3f}" for n in names))
        rep.table(["cos"] + names, [[ni] + [f"{cos[i, j]:.6f}" for j in range(len(names))]
                                    for i, ni in enumerate(names)], ["{}"] * (len(names) + 1))
    S["D_signature"] = resD
    rep.add("", "  Interpretation: z (both), lambda and w are expected to show cos = 1",
            "  among themselves (one signature, so not attributable to one cause);",
            "  z2-only and decentre are expected to differ (distinguishable).")
    rep.h("Run time")
    rep.kv("total [s]", f"{time.time() - T0:.1f}")
    rep.write("extra_checks_report.txt")
    json.dump(S, open(os.path.join(a.out, "summary_extra.json"), "w"), indent=2, default=float)
    log("finished")


if __name__ == "__main__":
    main()
