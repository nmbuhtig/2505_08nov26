#!/usr/bin/env python3
# -*- coding: utf-8 -*-
   
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.special import j0
from scipy.stats import chi2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

                                                                              
            
                                                                              
P0 = dict(
    lam=500e-9,                      
    sig_h=50e-9,                             
    l_c=8e-6,                                         
    w=25e-6,                                                  
    z1=0.8,                          
    z2=2.5,                          
    b_min=0.1e-3,                                       
    b_max=6e-3,                                        
    n_b=40,                                                        
    sep_off=2e-3,                                   
    x0_max=20e-3,                                       
    n_off=40,                                                         
    x0_max_C=7e-3,                                                         
                                                                        
    N=512,                                    
    dx=0.5e-6,                            
    rtol=1e-3,                                                                
                                                                            
                                                                           
                                                                          
                                                                              
                                                      
)

OUT = "results_v2"
T_START = time.time()


def log(msg):
    print(f"[{time.time() - T_START:7.1f} s] {msg}", flush=True)


def kk(lam):
    return 2 * np.pi / lam


                                                                              
                      
                                                                              
def make_designs(p):
    b = np.linspace(p["b_min"], p["b_max"], p["n_b"])
    x0 = np.linspace(0.0, p["x0_max"], p["n_off"])
    s = p["sep_off"]
    A = [dict(name="z1", z=p["z1"], x1=-b / 2, x2=b / 2, absc=b, absc_name="b [m]"),
         dict(name="z2", z=p["z2"], x1=-b / 2, x2=b / 2, absc=b, absc_name="b [m]")]
    B = [dict(name="z1", z=p["z1"], x1=x0 - s / 2, x2=x0 + s / 2, absc=x0,
              absc_name="pair centre x0 [m]"),
         dict(name="z2", z=p["z2"], x1=x0 - s / 2, x2=x0 + s / 2, absc=x0,
              absc_name="pair centre x0 [m]")]
                                                                               
                                                                              
                                                                             
    xc1 = np.linspace(0.0, p["x0_max_C"], p["n_off"])
    xc2 = xc1 * p["z2"] / p["z1"]
    C = [dict(name="z1", z=p["z1"], x1=xc1 - s / 2, x2=xc1 + s / 2, absc=xc1,
              absc_name="pair centre x0 [m]"),
         dict(name="z2", z=p["z2"], x1=xc2 - s / 2, x2=xc2 + s / 2, absc=xc2,
              absc_name="pair centre x0 [m]")]
    return {"A_onaxis_2z": A, "B_offaxis_2z": B, "C_scaled_2z": C}


def nominal_geo(p, nblocks=2):
                                                              
    return dict(lam=p["lam"], w=p["w"], zs=[1.0] * nblocks, dl=[0.0] * nblocks)


                                                                              
                                        
                                                                              
_SGRID = {}


def S_of_nu(nu, sig_h, l_c, w, lam, ns=3001):
    key = (round(w, 15), ns)
    if key not in _SGRID:
        _SGRID[key] = np.linspace(0.0, 12.0 * w, ns)
    s = _SGRID[key]
    g = (2 * kk(lam) * sig_h) ** 2
    chi = np.exp(-g * (1 - np.exp(-s ** 2 / (2 * l_c ** 2))))
    f = chi * np.exp(-s ** 2 / (8 * w ** 2)) * s
    nu = np.abs(np.atleast_1d(nu))
    return 2 * np.pi * simpson(f[None, :] * j0(2 * np.pi * nu[:, None] * s[None, :]),
                               x=s, axis=1)


def gamma_pairs(x1, x2, z, sig_h, l_c, w, lam):
    nu1 = np.asarray(x1, float) / (lam * z)
    nu2 = np.asarray(x2, float) / (lam * z)
    nub = 0.5 * (nu1 + nu2)
    n = len(nu1)
    S = S_of_nu(np.concatenate([nu1, nu2, nub]), sig_h, l_c, w, lam)
    env = np.exp(-2 * np.pi ** 2 * w ** 2 * (nu1 - nu2) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
                                                                            
                                                                           
        return env * S[2 * n:] / np.sqrt(S[:n] * S[n:2 * n])


def envelope_factor(x1, x2, z, w, lam):
    return np.exp(-2 * np.pi ** 2 * w ** 2 * ((np.asarray(x1) - np.asarray(x2)) / (lam * z)) ** 2)


def forward(theta, design, geo):
    sig_h, l_c = theta
    out = []
    for i, blk in enumerate(design):
        z = blk["z"] * geo["zs"][i]
        out.append(gamma_pairs(blk["x1"] + geo["dl"][i], blk["x2"] + geo["dl"][i],
                               z, sig_h, l_c, geo["w"], geo["lam"]))
    return np.concatenate(out)


def old_S69(b, z, sig_h, l_c, lam, R_A=50e-6, ns=4001):
                                                          
    r = np.linspace(0, R_A, ns)
    g = (2 * kk(lam) * sig_h) ** 2
    K = np.exp(-g * (1 - np.exp(-r ** 2 / (2 * l_c ** 2))))
    num = simpson(K[None, :] * j0(2 * np.pi * np.asarray(b)[:, None] * r[None, :] / (lam * z))
                  * r[None, :], x=r, axis=1)
    return num / simpson(K * r, x=r)


                                                                              
                          
                                                                              
def cov_sqrt(N, dx, l_c, sig_h):
    m = np.fft.fftfreq(N) * N * dx
    MX, MY = np.meshgrid(m, m)
    c = sig_h ** 2 * np.exp(-(MX ** 2 + MY ** 2) / (2 * l_c ** 2))
    return np.sqrt(np.clip(np.fft.fft2(c).real, 0.0, None))


def surfaces(rng, nb, N, sqC):
    return np.fft.ifft2(sqC * np.fft.fft2(rng.standard_normal((nb, N, N)))).real


def mc_fields(p, R, seed, batch=16, keep_heights=0):
                                                                             
    N, dx = p["N"], p["dx"]
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    A = np.exp(-(X ** 2 + Y ** 2) / (4 * p["w"] ** 2))
    sqC = cov_sqrt(N, dx, p["l_c"], p["sig_h"])
    rng = np.random.default_rng(seed)
    k2 = 2 * kk(p["lam"])
    u = np.empty((R, N), complex)
    kept = []
    for i0 in range(0, R, batch):
        nb = min(batch, R - i0)
        h = surfaces(rng, nb, N, sqC)
        if len(kept) * batch < keep_heights:
            kept.append(h.copy())
        u[i0:i0 + nb] = (A * np.exp(1j * k2 * h)).sum(axis=1) * dx
    H = np.concatenate(kept)[:keep_heights] if keep_heights else None
    return x, u, H


def farfield(x, u, nu):
    return u @ np.exp(-2j * np.pi * np.outer(nu, x)).T * (x[1] - x[0])


def gamma_influence(x, u, x1, x2, z, lam):
                                                                                   
    F1 = farfield(x, u, np.asarray(x1) / (lam * z))
    F2 = farfield(x, u, np.asarray(x2) / (lam * z))
    c = np.conj(F1) * F2
    G = c.mean(0)
    I1 = (np.abs(F1) ** 2).mean(0)
    I2 = (np.abs(F2) ** 2).mean(0)
    aG = np.abs(G)
    g = aG / np.sqrt(I1 * I2)
    psi = g * (np.real(np.conj(G) * (c - G)) / aG ** 2
               - 0.5 * (np.abs(F1) ** 2 - I1) / I1 - 0.5 * (np.abs(F2) ** 2 - I2) / I2)
    return g, psi


def design_data(x, u, design, lam_true, geo=None):
                                                                          
    geo = geo or dict(zs=[1.0] * len(design), dl=[0.0] * len(design))
    ys, ps = [], []
    for i, blk in enumerate(design):
        z = blk["z"] * geo["zs"][i]
        g, psi = gamma_influence(x, u, blk["x1"] + geo["dl"][i], blk["x2"] + geo["dl"][i],
                                 z, lam_true)
        ys.append(g)
        ps.append(psi)
    y = np.concatenate(ys)
    psi = np.concatenate(ps, axis=1)
    R = u.shape[0]
    return y, psi.T @ psi / R ** 2


def whitener(Sig, rtol):
    ev, V = np.linalg.eigh(0.5 * (Sig + Sig.T))
    keep = ev > rtol * ev.max()
    return (V[:, keep] / np.sqrt(ev[keep])).T, ev[::-1]


                                                                              
               
                                                                              
class Report:
    def __init__(self, title):
        self.lines = ["=" * 78, title, "=" * 78, ""]

    def h(self, t):
        self.lines += ["", t, "-" * len(t)]

    def add(self, *a):
        for s in a:
            self.lines.append(str(s))

    def kv(self, k, v, fmt="{}"):
        self.lines.append(f"  {k:<46s}: " + (fmt.format(v) if not isinstance(v, str) else v))

    def table(self, header, rows, fmts):
        cells = [[(f.format(v) if not isinstance(v, str) else v) for v, f in zip(r, fmts)]
                 for r in rows]
        w = [max([len(hh)] + [len(c[i]) for c in cells]) for i, hh in enumerate(header)]
        self.lines.append("  " + "  ".join(
            (hh.ljust(ww) if i == 0 else hh.rjust(ww)) for i, (hh, ww) in enumerate(zip(header, w))))
        for c in cells:
            self.lines.append("  " + "  ".join(
                (v.ljust(ww) if i == 0 else v.rjust(ww)) for i, (v, ww) in enumerate(zip(c, w))))

    def write(self, name):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")
        log(f"wrote {name}")


def save_fig(fig, stem):
                                                        
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=150)
    fig.savefig(os.path.join(OUT, stem + ".pdf"))
    log(f"wrote {stem}.png / .pdf")


def save_table(name, header, cols):
    arr = np.column_stack(cols)
    np.savetxt(os.path.join(OUT, name), arr, header="  ".join(header), fmt="%.10e")


def jac_log(fun, th, eps=1e-4):
    th = np.asarray(th, float)
    cols = []
    for j in range(len(th)):
        tp, tm = th.copy(), th.copy()
        tp[j] *= np.exp(eps)
        tm[j] *= np.exp(-eps)
        cols.append((fun(tp) - fun(tm)) / (2 * eps))
    return np.array(cols).T


def fisher_metrics(Jw):
                                                                                   
    U, sv, Vt = np.linalg.svd(Jw, full_matrices=False)
    if Jw.shape[0] < 2 or len(sv) < 2 or sv[-1] <= 1e-10 * sv[0]:
        nan = float("nan")
        return dict(sing_vals=sv.tolist(), cond=float("inf"), crb_rel_sig_h=nan,
                    crb_rel_l_c=nan, corr=nan, identifiable=False,
                    best_dir=Vt[0].tolist(), worst_dir=Vt[-1].tolist())
    C = np.linalg.inv(Jw.T @ Jw)
    crb = np.sqrt(np.diag(C))
    return dict(sing_vals=sv.tolist(), cond=float(sv[0] / sv[-1]),
                crb_rel_sig_h=float(crb[0]), crb_rel_l_c=float(crb[1]),
                corr=float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])), identifiable=True,
                best_dir=Vt[0].tolist(), worst_dir=Vt[-1].tolist())


                                                                              
                     
                                                                              
def stage_validate(p, R, n_calib, R_calib, S):
    rep = Report("01  VALIDATION: phase-screen MC vs exact Kirchhoff/Fraunhofer model")
    designs = make_designs(p)
    th = np.array([p["sig_h"], p["l_c"]])
    geo = nominal_geo(p)
    log(f"validate: MC with R={R}")
    t0 = time.time()
    x, u, H = mc_fields(p, R, seed=1, keep_heights=32)
    rep.kv("realizations R", R)
    rep.kv("MC wall time [s]", time.time() - t0, "{:.1f}")
    rep.kv("grid N x N, dx [um]", f"{p['N']} x {p['N']}, {p['dx'] * 1e6:.3f}")
    rep.kv("domain side [um]", p["N"] * p["dx"] * 1e6, "{:.1f}")
    res = {}

                                         
    rep.h("A. Surface statistics (first 32 realizations)")
    lags = np.arange(0, 40)
    acf = np.array([np.mean(H * np.roll(H, -L, axis=2)) for L in lags]) / np.mean(H ** 2)
    sl = lags * p["dx"]
    acf_th = np.exp(-sl ** 2 / (2 * p["l_c"] ** 2))
    k2 = 2 * kk(p["lam"])
    chi_mc = np.array([np.mean(np.exp(1j * k2 * (H - np.roll(H, -L, axis=2)))).real for L in lags])
    gph = (k2 * p["sig_h"]) ** 2
    chi_th = np.exp(-gph * (1 - acf_th))
    res["sig_h_sample_nm"] = float(H.std() * 1e9)
    res["acf_max_abs_err"] = float(np.max(np.abs(acf - acf_th)))
    res["chi_max_abs_err"] = float(np.max(np.abs(chi_mc - chi_th)))
    res["phase_variance_g"] = float(gph)
    rep.kv("target sigma_h [nm]", p["sig_h"] * 1e9, "{:.2f}")
    rep.kv("sample sigma_h [nm]", res["sig_h_sample_nm"], "{:.2f}")
    rep.kv("phase variance g=(2k sigma_h)^2", gph, "{:.4f}")
    rep.kv("coherent (specular) fraction exp(-g)", np.exp(-gph), "{:.4f}")
    rep.kv("max |ACF_sample - ACF_target|", res["acf_max_abs_err"], "{:.4f}")
    rep.kv("max |chi_MC - chi_Kirchhoff|", res["chi_max_abs_err"], "{:.4f}")
    save_table("data_surface_stats.txt",
               ["lag_m", "acf_sample", "acf_target", "chi_mc", "chi_kirchhoff"],
               [sl, acf, acf_th, chi_mc, chi_th])

                                                               
    rep.h("B. Angular intensity by 2-D FFT propagation (zero-padded)")
    N, dx = p["N"], p["dx"]
    xx = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xx, xx)
    A = np.exp(-(X ** 2 + Y ** 2) / (4 * p["w"] ** 2))
    Npad, nfft = 2048, min(200, R)
    rng = np.random.default_rng(7)
    sqC = cov_sqrt(N, dx, p["l_c"], p["sig_h"])
    Iacc = np.zeros(Npad)
    done = 0
    while done < nfft:
        nb = min(16, nfft - done)
        hh = surfaces(rng, nb, N, sqC)
        for i in range(nb):
            Uf = np.fft.fftshift(np.fft.fft2(A * np.exp(1j * k2 * hh[i]), s=(Npad, Npad)))
            Iacc += np.abs(Uf[Npad // 2, :]) ** 2
        done += nb
    nu_f = np.fft.fftshift(np.fft.fftfreq(Npad, dx))
    I_mc = Iacc / Iacc[Npad // 2]
    I_th = S_of_nu(nu_f, p["sig_h"], p["l_c"], p["w"], p["lam"])
    I_th /= I_th[Npad // 2]
    sel = np.abs(nu_f) < 6e4
    res["fft_realizations"] = nfft
    res["intensity_max_abs_err"] = float(np.max(np.abs(I_mc[sel] - I_th[sel])))
    rep.kv("FFT realizations", nfft)
    rep.kv("max |I_MC - S(nu)/S(0)| for |nu|<60/mm", res["intensity_max_abs_err"], "{:.4f}")
    save_table("data_angular_intensity.txt", ["nu_per_m", "I_mc_norm", "S_norm"],
               [nu_f[sel], I_mc[sel], I_th[sel]])

                                                 
    rep.h("C. |gamma| : MC vs exact model (and old Eq. S69 for design A)")
    rep.add("  chi2_diag/dof uses per-point SE only; chi2_white uses the full covariance",
            "  (truncated eigen-whitening).  Expected: chi2_diag/dof ~ 1, chi2_white ~ rank.")
    res["coherence"] = {}
    fig, axs = plt.subplots(len(designs), 2, figsize=(12, 4 * len(designs)))
    for di, (dname, design) in enumerate(designs.items()):
        y, Sig = design_data(x, u, design, p["lam"])
        f0 = forward(th, design, geo)
        W, ev = whitener(Sig, p["rtol"])
        se = np.sqrt(np.diag(Sig))
        rw = W @ (y - f0)
        d = dict(rank_kept=int(W.shape[0]),
                 chi2_white=float(rw @ rw),
                 chi2_diag_per_dof=float(np.mean(((y - f0) / se) ** 2)),
                 rms_abs_err=float(np.sqrt(np.mean((y - f0) ** 2))),
                 mean_SE=float(se.mean()),
                 cov_eig_rel=(ev[:8] / ev[0]).tolist())
        rep.add("", f"  Design {dname}")
        rep.kv("  RMS |MC - exact|", d["rms_abs_err"], "{:.3e}")
        rep.kv("  mean per-point SE", d["mean_SE"], "{:.3e}")
        rep.kv("  chi2_diag / dof", d["chi2_diag_per_dof"], "{:.3f}")
        rep.kv("  covariance modes kept (rtol)", f"{d['rank_kept']}  (rtol={p['rtol']:g})")
        rep.kv("  chi2_white (expect ~ modes kept)", d["chi2_white"], "{:.2f}")
        rep.kv("  leading cov eigenvalues / max", " ".join(f"{v:.1e}" for v in d["cov_eig_rel"]))
                                                                                  
        evs, Vs = np.linalg.eigh(0.5 * (Sig + Sig.T))
        evs, Vs = evs[::-1], Vs[:, ::-1]
        per_mode = (Vs.T @ (y - f0)) ** 2 / np.clip(evs, 1e-300, None)
        scan = []
        for rt in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            k = int((evs > rt * evs[0]).sum())
            c = float(per_mode[:k].sum())
            scan.append(dict(rtol=rt, modes=k, chi2=c, p_value=float(chi2.sf(c, k))))
        d["rtol_scan"] = scan
        rep.add("    chi2 vs eigenvalue cutoff (p-value: probability of a larger chi2):")
        rep.table(["    rtol", "modes", "chi2", "p-value"],
                  [[f"    {q['rtol']:.0e}", q["modes"], q["chi2"], q["p_value"]] for q in scan],
                  ["{}", "{}", "{:.2f}", "{:.3f}"])
        nb = len(design[0]["x1"])
        for bi, blk in enumerate(design):
            sl_ = slice(bi * nb, (bi + 1) * nb)
            env = envelope_factor(blk["x1"], blk["x2"], blk["z"], p["w"], p["lam"])
            cols = [blk["absc"], blk["x1"], blk["x2"], y[sl_], se[sl_], f0[sl_], env]
            hdr = ["abscissa_m", "x1_m", "x2_m", "gamma_MC", "SE_MC", "gamma_exact", "envelope_factor"]
            if dname.startswith("A"):
                old = old_S69(blk["absc"], blk["z"], p["sig_h"], p["l_c"], p["lam"])
                cols.append(old)
                hdr.append("gamma_oldS69")
                d[f"rms_abs_err_oldS69_{blk['name']}"] = float(np.sqrt(np.mean((y[sl_] - old) ** 2)))
                rep.kv(f"  RMS |MC - old S69|, {blk['name']}", d[f"rms_abs_err_oldS69_{blk['name']}"], "{:.3f}")
            save_table(f"data_validation_{dname}_{blk['name']}.txt", hdr, cols)
            ax = axs[di, bi]
            ax.errorbar(blk["absc"] * 1e3, y[sl_], yerr=1.96 * se[sl_], fmt="o", ms=3,
                        label="phase-screen MC (95%)")
            ax.plot(blk["absc"] * 1e3, f0[sl_], "-", lw=2, label="exact model")
            ax.plot(blk["absc"] * 1e3, env, ":", label="envelope factor only")
            if dname.startswith("A"):
                ax.plot(blk["absc"] * 1e3, cols[-1], "--", label="old Eq. (S69)")
            ax.set_title(f"{dname}, z = {blk['z']} m")
            ax.set_xlabel(blk["absc_name"].replace("[m]", "[mm]"))
            ax.set_ylabel("|gamma|")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
        res["coherence"][dname] = d
    fig.tight_layout()
    save_fig(fig, "fig_validation")
    plt.close(fig)

                                                                  
    rep.h(f"D. Noise-model calibration: {n_calib} independent MC runs, R={R_calib}")
    rep.add("  For each run: chi2 of (y - exact) with that run's own estimated covariance.",
            "  Calibrated noise model => mean chi2_white ~ modes kept; mean chi2_diag ~ n_data.")
    calib = {dn: dict(cw=[], cd=[], rk=[], res=[]) for dn in designs}
    for t in range(n_calib):
        log(f"validate: calibration run {t + 1}/{n_calib}")
        xc, uc, _ = mc_fields(p, R_calib, seed=5000 + t)
        for dn, design in designs.items():
            y, Sig = design_data(xc, uc, design, p["lam"])
            f0 = forward(th, design, geo)
            W, _ = whitener(Sig, p["rtol"])
            r = y - f0
            calib[dn]["cw"].append(float(np.sum((W @ r) ** 2)))
            calib[dn]["cd"].append(float(np.sum(r ** 2 / np.diag(Sig))))
            calib[dn]["rk"].append(int(W.shape[0]))
            calib[dn]["res"].append(r)
            calib[dn].setdefault("var", []).append(np.diag(Sig))
    res["calibration"] = {}
    rows = []
    for dn in designs:
        Rm = np.array(calib[dn]["res"])
                                                                           
                                                                       
        se_mean = np.sqrt(np.mean(np.array(calib[dn]["var"]), axis=0) / n_calib)
        tbias = np.abs(Rm.mean(0) / (se_mean + 1e-300))
        c = dict(mean_chi2_white=float(np.mean(calib[dn]["cw"])),
                 mean_modes=float(np.mean(calib[dn]["rk"])),
                 mean_chi2_diag=float(np.mean(calib[dn]["cd"])),
                 n_data=int(Rm.shape[1]),
                 max_abs_t_bias=float(tbias.max()))
        res["calibration"][dn] = c
        rows.append([dn, c["mean_chi2_white"], c["mean_modes"], c["mean_chi2_diag"],
                     c["n_data"], c["max_abs_t_bias"]])
    rep.table(["design", "chi2_white", "modes", "chi2_diag", "n_data", "max|t_bias|"],
              rows, ["{}", "{:.2f}", "{:.1f}", "{:.1f}", "{}", "{:.2f}"])
    rep.add("", "  max|t_bias|: largest |run-averaged residual| / its pooled SE over the",
            "  data points.  For 80 unbiased points the max is typically < 3.3;",
            "  clearly larger values indicate model or estimator bias.")
    rep.write("01_validation_report.txt")
    return res


                                                                              
                          
                                                                              
def stage_ident(p, R, S):
    rep = Report("02  IDENTIFIABILITY: Fisher information with MC-estimated noise")
    designs = make_designs(p)
    th = np.array([p["sig_h"], p["l_c"]])
    geo = nominal_geo(p)
    log(f"ident: MC with R={R}")
    x, u, _ = mc_fields(p, R, seed=11)
    rep.add("  Parameters in log space: relative CRB = sqrt(diag(F^-1)).",
            "  'corr' = correlation of (log sigma_h, log l_c) in F^-1.",
            "  'best/worst dir' = right singular vectors in (log sigma_h, log l_c).",
            f"  Noise from one MC run with R={R}; CRB scales as 1/sqrt(R).", "")
    res = {}
    for dn, design in designs.items():
        y, Sig = design_data(x, u, design, p["lam"])
        J = jac_log(lambda t: forward(t, design, geo), th)
        nb = len(design[0]["x1"])
        out = {}
        rows = []
        for label, sl_ in (("both distances", slice(0, 2 * nb)), ("z1 only", slice(0, nb)),
                           ("z2 only", slice(nb, 2 * nb))):
            W, ev = whitener(Sig[sl_, sl_], p["rtol"])
            m_full = fisher_metrics(W @ J[sl_])
            m_diag = fisher_metrics(J[sl_] / np.sqrt(np.diag(Sig[sl_, sl_]))[:, None])
            m_full["modes_kept"] = int(W.shape[0])
            out[label] = dict(full_cov=m_full, diag_noise_NOT_VALID=m_diag)
            if m_full["identifiable"]:
                rows.append([label, m_full["modes_kept"], f"{m_full['crb_rel_sig_h'] * 100:.2f}",
                             f"{m_full['crb_rel_l_c'] * 100:.2f}", f"{m_full['corr']:+.5f}",
                             f"{m_full['cond']:.1f}",
                             f"{m_diag['crb_rel_sig_h'] * 100:.2f}", f"{m_diag['crb_rel_l_c'] * 100:.2f}"])
            else:
                rows.append([label, m_full["modes_kept"], "not identifiable", "-", "-", "-",
                             f"{m_diag['crb_rel_sig_h'] * 100:.2f}", f"{m_diag['crb_rel_l_c'] * 100:.2f}"])
        rep.h(f"Design {dn}")
        rep.table(["data", "modes", "CRB sig_h %", "CRB l_c %", "corr", "cond",
                   "diagCRB sig%", "diagCRB lc%"], rows, ["{}"] * 8)
        b = out["both distances"]["full_cov"]
        rep.kv("best-determined direction (log sig, log lc)", str(np.round(b["best_dir"], 3)))
        rep.kv("worst-determined direction", str(np.round(b["worst_dir"], 3)))
        rep.add("  (diag-noise columns ignore cross-baseline correlation; shown only to",
                "   document that they overstate the information.)")
        res[dn] = out
    rep.write("02_identifiability_report.txt")
    return res


                                                                              
                             
                                                                              
def infer(y, W, design, geo, mu, s_prior):
    def phi(lt):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            f = forward(np.exp(lt), design, geo)
        if not np.all(np.isfinite(f)):
            return 1e30                                                     
        r = W @ (y - f)
        return 0.5 * r @ r + 0.5 * np.sum(((lt - mu) / s_prior) ** 2)

    best = None
    for s0 in (mu, mu + np.array([0.5, -0.5]), mu + np.array([-0.5, 0.5])):
        r = minimize(phi, s0, method="Nelder-Mead",
                     options=dict(xatol=1e-6, fatol=1e-8, maxiter=3000))
        if best is None or r.fun < best.fun:
            best = r
    lt = best.x
    h = 1e-3
    Hs = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
            Hs[i, j] = (phi(lt + ei + ej) - phi(lt + ei - ej)
                        - phi(lt - ei + ej) + phi(lt - ei - ej)) / (4 * h * h)
    return lt, Hs, phi


def stage_bayes(p, R, trials, S):
    rep = Report("03  BAYESIAN INFERENCE on independent phase-screen MC data")
    designs = make_designs(p)
    th = np.array([p["sig_h"], p["l_c"]])
    lt_true = np.log(th)
    geo = nominal_geo(p)
    priors = {"centred": (lt_true.copy(), np.array([0.8, 0.8])),
              "offset": (lt_true + np.log([1.5, 0.6]), np.array([1.0, 1.0]))}
    q95 = chi2.ppf(0.95, 2)
    rep.add(f"  Trials: {trials};  realizations per trial: {R}.",
            "  Data: MC |gamma| (both designs from the same surfaces in each trial).",
            "  Model: exact analytic formula.  Noise: per-trial delta-method covariance,",
            "  truncated eigen-whitening.  Priors: log-normal in (sigma_h, l_c):",
            "    centred: mean = truth,            s = 0.8 (both)",
            "    offset : mean = (1.5 sig*, 0.6 lc*), s = 1.0 (both)",
            "  Coverage (nominal 0.95; binomial SE ~ sqrt(.95*.05/trials)):",
            "    cov_LR  : truth inside the likelihood-ratio 95% region",
            "              2[Phi(theta*) - Phi(MAP)] <= chi2_2(0.95) = 5.99.",
            "              Follows a curved posterior; this is the primary measure.",
            "    cov_Lap : truth inside the Laplace (Gaussian) 95% ellipse in log space.",
            "              Expected to undercover when the posterior is curved.",
            f"  Covariance modes kept: eigenvalues > rtol x max, rtol = {p['rtol']:g}.")
    recs = {(dn, pn): [] for dn in designs for pn in priors}
    for t in range(trials):
        log(f"bayes: trial {t + 1}/{trials}")
        x, u, _ = mc_fields(p, R, seed=1000 + t)
        for dn, design in designs.items():
            y, Sig = design_data(x, u, design, p["lam"])
            W, _ = whitener(Sig, p["rtol"])
            for pn, (mu, sp) in priors.items():
                lt, Hs, phi = infer(y, W, design, geo, mu, sp)
                d = lt_true - lt
                lr_stat = float(2 * (phi(lt_true) - phi(lt)))
                covered_lr = bool(lr_stat <= q95)
                try:
                    C = np.linalg.inv(Hs)
                    pd_ok = bool(np.all(np.linalg.eigvalsh(Hs) > 0))
                    covered = bool(pd_ok and d @ Hs @ d <= q95)
                    hw = 1.96 * np.sqrt(np.abs(np.diag(C)))
                except np.linalg.LinAlgError:
                    covered, hw, pd_ok = False, np.array([np.nan, np.nan]), False
                recs[(dn, pn)].append(dict(trial=t, sig=np.exp(lt[0]), lc=np.exp(lt[1]),
                                           hw_sig=hw[0], hw_lc=hw[1], covered=covered,
                                           covered_lr=covered_lr, lr_stat=lr_stat,
                                           H_posdef=pd_ok, modes=int(W.shape[0]),
                                           y=y if t == 0 else None))
    res = {}
    rows = []
    for (dn, pn), rr in recs.items():
        sig = np.array([r["sig"] for r in rr])
        lc = np.array([r["lc"] for r in rr])
        cov = np.array([r["covered"] for r in rr])
        covlr = np.array([r["covered_lr"] for r in rr])
        hws = np.array([r["hw_sig"] for r in rr])
        hwl = np.array([r["hw_lc"] for r in rr])
        d = dict(coverage95_LR=float(covlr.mean()),
                 coverage95=float(cov.mean()),
                 coverage_binom_SE=float(np.sqrt(0.95 * 0.05 / len(rr))),
                 median_sig_h_nm=float(np.median(sig) * 1e9),
                 median_l_c_um=float(np.median(lc) * 1e6),
                 bias_logsig=float(np.mean(np.log(sig / th[0]))),
                 bias_loglc=float(np.mean(np.log(lc / th[1]))),
                 sd_logsig=float(np.std(np.log(sig))),
                 sd_loglc=float(np.std(np.log(lc))),
                 mean_halfwidth95_logsig=float(np.nanmean(hws)),
                 mean_halfwidth95_loglc=float(np.nanmean(hwl)),
                 frac_H_posdef=float(np.mean([r["H_posdef"] for r in rr])))
        res[f"{dn}|{pn}"] = d
        rows.append([dn, pn, d["coverage95_LR"], d["coverage95"], d["median_sig_h_nm"],
                     d["median_l_c_um"], d["bias_logsig"], d["bias_loglc"],
                     d["sd_logsig"], d["sd_loglc"], d["mean_halfwidth95_logsig"] / 1.96,
                     d["mean_halfwidth95_loglc"] / 1.96])
        save_table(f"data_bayes_trials_{dn}_{pn}.txt",
                   ["trial", "sigma_h_MAP_m", "l_c_MAP_m", "halfwidth95_logsig",
                    "halfwidth95_loglc", "covered95_laplace", "covered95_LR", "LR_stat",
                    "H_posdef", "modes_kept"],
                   [[r["trial"] for r in rr], sig, lc, hws, hwl, cov.astype(int),
                    covlr.astype(int), [r["lr_stat"] for r in rr],
                    [int(r["H_posdef"]) for r in rr], [r["modes"] for r in rr]])
    rep.h("Results")
    rep.table(["design", "prior", "cov_LR", "cov_Lap", "med sig nm", "med lc um",
               "bias log sig", "bias log lc", "sd log sig", "sd log lc",
               "Laplace sd sig", "Laplace sd lc"], rows,
              ["{}", "{}", "{:.3f}", "{:.3f}", "{:.2f}", "{:.2f}", "{:+.3f}", "{:+.3f}",
               "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}"])
    rep.add("", "  bias = mean(log MAP - log truth) over trials.",
            "  Calibration check: 'sd log' (scatter over trials) should be comparable",
            "  to 'Laplace sd' (mean posterior sd); cov_LR should be ~0.95.")
    rep.write("03_bayes_report.txt")

                                                                    
    nd = len(designs)
    fig, axs = plt.subplots(1, nd + 1, figsize=(5.3 * (nd + 1), 4.3))
    for k, dn in enumerate(designs):
        design = designs[dn]
        r0 = recs[(dn, "offset")][0]
        y0 = r0["y"]
        fm = forward(np.array([r0["sig"], r0["lc"]]), design, geo)
        ft = forward(th, design, geo)
        nb = len(design[0]["x1"])
        for bi, blk in enumerate(design):
            sl_ = slice(bi * nb, (bi + 1) * nb)
            axs[k].plot(blk["absc"] * 1e3, y0[sl_], "o", ms=3, label=f"MC data {blk['name']}")
            axs[k].plot(blk["absc"] * 1e3, ft[sl_], "k-", lw=1)
            axs[k].plot(blk["absc"] * 1e3, fm[sl_], "r--", lw=1)
        axs[k].plot([], [], "k-", label="truth")
        axs[k].plot([], [], "r--", label="MAP (offset prior)")
        axs[k].set_title(dn + ", trial 1")
        axs[k].set_xlabel(design[0]["absc_name"].replace("[m]", "[mm]"))
        axs[k].set_ylabel("|gamma|")
        axs[k].legend(fontsize=8)
    ax = axs[nd]
    marks = ("o", "s", "^", "v", "x", "+", "D", "*")
    for (dn, pn), mk in zip(recs, marks):
        rr = recs[(dn, pn)]
        ax.plot([r["sig"] * 1e9 for r in rr], [r["lc"] * 1e6 for r in rr], mk, ms=4,
                mfc="none", label=f"{dn}, {pn}")
    ax.plot(th[0] * 1e9, th[1] * 1e6, "k*", ms=14, label="truth")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sigma_h MAP [nm]")
    ax.set_ylabel("l_c MAP [um]")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig_bayes")
    plt.close(fig)
    return res


                                                                              
                          
                                                                              
def stage_fault(p, R, S):
    rep = Report("04  FAULT ISOLATION: residual decomposition with MC data")
    designs = make_designs(p)
    th = np.array([p["sig_h"], p["l_c"]])
    geo0 = nominal_geo(p)
    rep.add("  Surface subspace = span of whitened Jacobian wrt (log sigma_h, log l_c).",
            "  rho = |P_perp v| / |v| for each whitened fault direction v (0 = fault is",
            "  indistinguishable from a surface change, 1 = fully orthogonal).",
            "  MC scenarios: data from phase-screen MC with the fault present; model",
            "  = nominal exact formula; residual whitened with the nominal covariance.",
            "  Null: E_S ~ chi2(2), E_P ~ chi2(m-2), m = covariance modes kept.",
            "  The linear split is local: for faults large compared with the noise,",
            "  curvature of the forward map leaks surface faults into E_P.  The",
            "  nonlinear test (ii) re-fits the surface parameters first; a surface",
            "  fault is then absorbed and only non-surface faults leave chi2 excess.",
            "  With m <= 2 modes there are no residual degrees of freedom (n/a).")

    def geo_mod(**kw):
        g = dict(lam=geo0["lam"], w=geo0["w"], zs=list(geo0["zs"]), dl=list(geo0["dl"]))
        g.update(kw)
        return g

    fault_dirs = {
        "z2 scale (block z2 only)": lambda e: geo_mod(zs=[1.0, 1 + e]),
        "z scale (both blocks)": lambda e: geo_mod(zs=[1 + e, 1 + e]),
        "lambda (relative)": lambda e: geo_mod(lam=geo0["lam"] * (1 + e)),
        "w envelope (relative)": lambda e: geo_mod(w=geo0["w"] * (1 + e)),
        "decentre/tilt both [1 mm]": lambda e: geo_mod(dl=[e * 1e-3, e * 1e-3]),
        "decentre z2 only [1 mm]": lambda e: geo_mod(dl=[0.0, e * 1e-3]),
    }
    log(f"fault: reference MC with R={R}")
    x, u, _ = mc_fields(p, R, seed=21)
    ref = {}
    res = {"rho": {}, "scenarios": {}}
    for dn, design in designs.items():
        _, Sig = design_data(x, u, design, p["lam"])
        W, _ = whitener(Sig, p["rtol"])
        Js = W @ jac_log(lambda t: forward(t, design, geo0), th)
        Q, _ = np.linalg.qr(Js)
        PS = Q @ Q.T
        ref[dn] = (W, PS, forward(th, design, geo0))
        rr = {}
        for fn, gfun in fault_dirs.items():
            e = 1e-4
            v = W @ ((forward(th, design, gfun(e)) - forward(th, design, gfun(-e))) / (2 * e))
            nv = np.linalg.norm(v)
            ns = np.linalg.norm(Js)
                                                                                       
                                                                                       
            if W.shape[0] <= 2:
                rr[fn] = "no residual space"
            else:
                rr[fn] = float(np.linalg.norm(v - PS @ v) / nv) if nv > 1e-8 * ns else None
            rr[fn + " |v| (whitened)"] = float(nv)
        res["rho"][dn] = rr

    rep.h("A. Orthogonality rho of fault directions to the surface subspace")
    rows = []
    for fn in fault_dirs:
        row = [fn]
        for dn in designs:
            r_ = res["rho"][dn][fn]
            row.append("first-order insensitive" if r_ is None else
                       (r_ if isinstance(r_, str) else f"{r_:.4f}"))
            row.append(f"{res['rho'][dn][fn + ' |v| (whitened)']:.3g}")
        rows.append(row)
    hdr = ["fault direction"]
    for dn in designs:
        hdr += [f"rho {dn}", f"|v| {dn}"]
    rep.table(hdr, rows, ["{}"] * len(hdr))
    rep.add("", "  |v| = whitened first-order sensitivity per unit fault (relative for",
            "  scale faults, per mm for decentre).  'first-order insensitive' means the",
            "  fault enters only at second order (e.g. decentring a symmetric pinhole",
            "  pair, which is even in the offset).  'no residual space': only m <= 2",
            "  covariance modes are kept, so the 2-D surface subspace fills the whole",
            "  whitened data space and no fault can be separated from a surface change.")

    scen = {
        "no fault": (dict(), dict()),
        "surface sig_h +10%, l_c +10%": (dict(sig_h=p["sig_h"] * 1.1, l_c=p["l_c"] * 1.1), dict()),
        "surface sig_h -20%": (dict(sig_h=p["sig_h"] * 0.8), dict()),
        "z2 +3%": (dict(), dict(zs=[1.0, 1.03])),
        "lambda +2%": (dict(lam=p["lam"] * 1.02), dict()),
        "w +10%": (dict(w=p["w"] * 1.1), dict()),
        "decentre 0.3 mm (both)": (dict(), dict(dl=[0.3e-3, 0.3e-3])),
    }
    rep.h("B. MC fault scenarios (whitened residual energies)")
    for si, (sn, (pmod, gmod)) in enumerate(scen.items()):
        log(f"fault: scenario '{sn}'")
        pp = dict(p, **pmod)
        xs, us, _ = mc_fields(pp, R, seed=300 + si)
        for dn, design in designs.items():
            W, PS, f0 = ref[dn]
            g_true = dict(zs=gmod.get("zs", [1.0, 1.0]), dl=gmod.get("dl", [0.0, 0.0]))
            y, _ = design_data(xs, us, design, pp["lam"], g_true)
            r = W @ (y - f0)
            ES = float(r @ PS @ r)
            EP = max(float(r @ r - ES), 0.0)                      
            m = W.shape[0]
            d = dict(E_S=ES, E_P=EP, modes=m,
                     p99_E_S=float(chi2.ppf(0.99, 2)),
                     p99_E_P=float(chi2.ppf(0.99, max(m - 2, 1))),
                     surface_flag=bool(ES > chi2.ppf(0.99, 2)),
                     propagation_flag=bool(m > 2 and EP > chi2.ppf(0.99, max(m - 2, 1))))
                                                                              
                                                                     
            lt_fit, _, _ = infer(y, W, design, geo0, np.log(th), np.array([3.0, 3.0]))
            rf = W @ (y - forward(np.exp(lt_fit), design, geo0))
            chi_fit = float(rf @ rf)
            d.update(refit_sig_h_nm=float(np.exp(lt_fit[0]) * 1e9),
                     refit_l_c_um=float(np.exp(lt_fit[1]) * 1e6),
                     refit_chi2=chi_fit,
                     refit_p99=float(chi2.ppf(0.99, m - 2)) if m > 2 else float("nan"),
                     refit_flag=("n/a" if m <= 2 else
                                 ("YES" if chi_fit > chi2.ppf(0.99, m - 2) else "no")))
            res["scenarios"].setdefault(dn, {})[sn] = d
    for dn in designs:
        rep.add("", f"  Design {dn}   (flags at 99% null quantiles)")
        rows = []
        for sn, d in res["scenarios"][dn].items():
            rows.append([sn, d["E_S"], d["E_P"], d["modes"], d["p99_E_S"], d["p99_E_P"],
                         "YES" if d["surface_flag"] else "no",
                         "YES" if d["propagation_flag"] else ("n/a" if d["modes"] <= 2 else "no")])
        rep.add("  (i) Linear projection at the nominal parameters:")
        rep.table(["scenario", "E_S", "E_P", "modes", "E_S p99", "E_P p99",
                   "surf flag", "prop flag"], rows,
                  ["{}", "{:.3g}", "{:.3g}", "{}", "{:.2f}", "{:.2f}", "{}", "{}"])
        rows = []
        for sn, d in res["scenarios"][dn].items():
            rows.append([sn, d["refit_sig_h_nm"], d["refit_l_c_um"], d["refit_chi2"],
                         d["modes"] - 2, d["refit_p99"], d["refit_flag"]])
        rep.add("", "  (ii) Nonlinear test: re-fit (sigma_h, l_c), then chi2 of the residual:")
        rep.table(["scenario", "fit sig nm", "fit lc um", "chi2 after fit", "dof",
                   "p99", "non-surface flag"], rows,
                  ["{}", "{:.2f}", "{:.2f}", "{:.3g}", "{}", "{:.2f}", "{}"])
        names = list(res["scenarios"][dn])
        save_table(f"data_fault_scenarios_{dn}.txt",
                   ["scenario_index", "E_S", "E_P", "modes", "refit_sigma_h_nm",
                    "refit_l_c_um", "refit_chi2"],
                   [np.arange(len(names)),
                    [res["scenarios"][dn][n]["E_S"] for n in names],
                    [res["scenarios"][dn][n]["E_P"] for n in names],
                    [res["scenarios"][dn][n]["modes"] for n in names],
                    [res["scenarios"][dn][n]["refit_sig_h_nm"] for n in names],
                    [res["scenarios"][dn][n]["refit_l_c_um"] for n in names],
                    [res["scenarios"][dn][n]["refit_chi2"] for n in names]])
    rep.add("", "  scenario_index order in data files:")
    for i, n in enumerate(scen):
        rep.add(f"    {i}: {n}")
    rep.write("04_fault_report.txt")

    fig, axs = plt.subplots(1, len(designs), figsize=(14, 4.5))
    for ax, dn in zip(axs, designs):
        names = list(res["scenarios"][dn])
        xs_ = np.arange(len(names))
        ax.bar(xs_ - 0.2, [max(res["scenarios"][dn][n]["E_S"], 1e-3) for n in names], 0.4, label="E_S")
        ax.bar(xs_ + 0.2, [max(res["scenarios"][dn][n]["E_P"], 1e-3) for n in names], 0.4, label="E_P")
        m = res["scenarios"][dn][names[0]]["modes"]
        ax.axhline(chi2.ppf(0.99, max(m - 2, 1)), ls=":", c="k", label="E_P null 99%")
        ax.set_yscale("log")
        ax.set_xticks(xs_)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_title(dn)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_fault")
    plt.close(fig)
    return res


                                                                              
      
                                                                              
def write_run_info(args, p):
    rep = Report("00  RUN INFORMATION")
    rep.kv("command", " ".join(sys.argv))
    rep.kv("start time", time.strftime("%Y-%m-%d %H:%M:%S"))
    rep.kv("python", sys.version.split()[0])
    rep.kv("platform", platform.platform())
    rep.kv("numpy / scipy / matplotlib",
           f"{np.__version__} / {scipy.__version__} / {matplotlib.__version__}")
    rep.h("Arguments")
    for k, v in vars(args).items():
        rep.kv(k, str(v))
    rep.h("Physical and numerical parameters")
    for k, v in p.items():
        rep.kv(k, f"{v:g}" if isinstance(v, float) else str(v))
    g = (2 * kk(p["lam"]) * p["sig_h"]) ** 2
    rep.h("Derived")
    rep.kv("phase variance g = (2k sigma_h)^2", g, "{:.4f}")
    rep.kv("specular fraction exp(-g)", np.exp(-g), "{:.4f}")
    rep.kv("Fresnel number w^2/(lambda z1)", p["w"] ** 2 / (p["lam"] * p["z1"]), "{:.2e}")
    rep.kv("l_c / dx", p["l_c"] / p["dx"], "{:.1f}")
    rep.kv("domain / w", p["N"] * p["dx"] / p["w"], "{:.1f}")
    return rep


def write_summary(S):
    rep = Report("05  SUMMARY OF KEY NUMBERS")
    if "validate" in S:
        v = S["validate"]
        rep.h("Validation")
        for dn, d in v["coherence"].items():
            rep.kv(f"{dn}: RMS |MC-exact| / mean SE", f"{d['rms_abs_err']:.2e} / {d['mean_SE']:.2e}")
            for k in ("rms_abs_err_oldS69_z1", "rms_abs_err_oldS69_z2"):
                if k in d:
                    rep.kv(f"{dn}: {k}", d[k], "{:.3f}")
        for dn, c in v.get("calibration", {}).items():
            rep.kv(f"{dn}: calib chi2_white / modes", f"{c['mean_chi2_white']:.2f} / {c['mean_modes']:.1f}")
    if "ident" in S:
        rep.h("Identifiability (both distances, full covariance)")
        for dn, d in S["ident"].items():
            m = d["both distances"]["full_cov"]
            if not m.get("identifiable", True):
                rep.kv(f"{dn}", "not identifiable (fewer than 2 informative modes)")
                continue
            rep.kv(f"{dn}: modes kept", m["modes_kept"])
            rep.kv(f"{dn}: CRB sig_h, l_c [%]", f"{m['crb_rel_sig_h'] * 100:.2f}, {m['crb_rel_l_c'] * 100:.2f}")
            rep.kv(f"{dn}: corr, cond", f"{m['corr']:+.4f}, {m['cond']:.1f}")
    if "bayes" in S:
        rep.h("Bayesian coverage (nominal 0.95)")
        for k, d in S["bayes"].items():
            rep.kv(k, f"cov LR {d['coverage95_LR']:.3f}, cov Laplace {d['coverage95']:.3f}, "
                      f"bias log(sig,lc) ({d['bias_logsig']:+.3f},{d['bias_loglc']:+.3f})")
    if "fault" in S:
        rep.h("Fault orthogonality rho (None = first-order insensitive)")
        for dn, rr in S["fault"]["rho"].items():
            for fn, r in rr.items():
                if "|v|" in fn:
                    continue
                rep.kv(f"{dn}: {fn}", "None" if r is None else (r if isinstance(r, str) else f"{r:.4f}"))
    rep.write("05_summary.txt")


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["validate", "ident", "bayes", "fault", "all"])
    ap.add_argument("--R", type=int, default=2000, help="realizations for validation")
    ap.add_argument("--R_trial", type=int, default=500, help="realizations per trial / ident / fault")
    ap.add_argument("--trials", type=int, default=40, help="Bayesian trials")
    ap.add_argument("--n_calib", type=int, default=10, help="independent runs for noise calibration")
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    ap.add_argument("--out", default="results_v2")
    a = ap.parse_args()
    if a.quick:
        a.R, a.R_trial, a.trials, a.n_calib = 300, 200, 3, 3
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    p = dict(P0)
    info = write_run_info(a, p)
    sp = os.path.join(OUT, "summary.json")
    S = json.load(open(sp)) if os.path.exists(sp) else {}
    S["params"] = p
    stage_times = {}
    for st, fn in (("validate", lambda: stage_validate(p, a.R, a.n_calib, a.R_trial, S)),
                   ("ident", lambda: stage_ident(p, a.R_trial, S)),
                   ("bayes", lambda: stage_bayes(p, a.R_trial, a.trials, S)),
                   ("fault", lambda: stage_fault(p, a.R_trial, S))):
        if a.stage in (st, "all"):
            t0 = time.time()
            S[st] = fn()
            stage_times[st] = time.time() - t0
            json.dump(S, open(sp, "w"), indent=2, default=float)
    info.h("Stage wall times [s]")
    for k, v in stage_times.items():
        info.kv(k, v, "{:.1f}")
    info.kv("total [s]", time.time() - T_START, "{:.1f}")
    info.write("00_run_info.txt")
    write_summary(S)
    log("finished")


if __name__ == "__main__":
    main()
