#!/usr/bin/env python3
# -*- coding: utf-8 -*-
   
import argparse
import os
import time

import numpy as np
from scipy.stats import chi2

import sect_mc_revision as M


def phi_factory(y, W, design, geo, mu, sp):
    def phi(lt):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            f = M.forward(np.exp(lt), design, geo)
        if not np.all(np.isfinite(f)):
            return 1e30
        r = W @ (y - f)
        return 0.5 * r @ r + 0.5 * np.sum(((lt - mu) / sp) ** 2)
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--R_trial", type=int, default=500)
    ap.add_argument("--R_ref", type=int, default=2000)
    ap.add_argument("--designs", default="A_onaxis_2z,B_offaxis_2z")
    ap.add_argument("--out", default="results_diag")
    ap.add_argument("--rtol", type=float, default=None, help="override eigenvalue cutoff")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    M.OUT = a.out
    p = dict(M.P0)
    if a.rtol is not None:
        p["rtol"] = a.rtol
    th = np.array([p["sig_h"], p["l_c"]])
    lt_true = np.log(th)
    geo = M.nominal_geo(p)
    designs = {k: v for k, v in M.make_designs(p).items() if k in a.designs.split(",")}
    mu, sp = lt_true.copy(), np.array([0.8, 0.8])                         
    q = chi2.ppf(0.95, 2)
    rng = np.random.default_rng(12345)
    t0 = time.time()

                                                                  
    M.log(f"reference MC, R={a.R_ref}")
    xr, ur, _ = M.mc_fields(p, a.R_ref, seed=99999)
    ref = {}
    for dn, d in designs.items():
        _, S = M.design_data(xr, ur, d, p["lam"])
        S = S * a.R_ref / a.R_trial
        W, _ = M.whitener(S, p["rtol"])
        ev, V = np.linalg.eigh(0.5 * (S + S.T))
        Lh = V * np.sqrt(np.clip(ev, 0, None))                        
        ref[dn] = (S, W, Lh, M.forward(th, d, geo))

    cases = ("own_cov", "ref_cov", "gauss_ref", "noiseless")
    rec = {(dn, c): [] for dn in designs for c in cases}
    for t in range(a.trials):
        M.log(f"trial {t + 1}/{a.trials}")
        x, u, _ = M.mc_fields(p, a.R_trial, seed=1000 + t)                            
        for dn, d in designs.items():
            S_ref, W_ref, Lh, f0 = ref[dn]
            y_mc, S_own = M.design_data(x, u, d, p["lam"])
            W_own, _ = M.whitener(S_own, p["rtol"])
            data = {
                "own_cov": (y_mc, W_own),
                "ref_cov": (y_mc, W_ref),
                "gauss_ref": (f0 + Lh @ rng.standard_normal(Lh.shape[1]), W_ref),
                "noiseless": (f0.copy(), W_ref),
            }
            for c, (y, W) in data.items():
                if c == "noiseless" and t > 0:
                    continue
                lt, H, _ = M.infer(y, W, d, geo, mu, sp)
                phi = phi_factory(y, W, d, geo, mu, sp)
                dd = lt_true - lt
                lap = bool(np.all(np.linalg.eigvalsh(H) > 0) and dd @ H @ dd <= q)
                lr = bool(2 * (phi(lt_true) - phi(lt)) <= q)
                rw = W @ (y - M.forward(np.exp(lt), d, geo))
                rec[(dn, c)].append(dict(t=t, ls=lt[0], ll=lt[1], lap=lap, lr=lr,
                                         chi2min=float(rw @ rw), m=W.shape[0],
                                         lap_d2=float(dd @ H @ dd),
                                         lr_d2=float(2 * (phi(lt_true) - phi(lt)))))

    L = ["=" * 78, "BAYESIAN COVERAGE DIAGNOSTICS (centred prior, s = 0.8)", "=" * 78, "",
         f"  trials = {a.trials}, R_trial = {a.R_trial}, R_ref = {a.R_ref}, rtol = {p['rtol']:g}",
         "  Nominal coverage 0.95; binomial SE ~ %.3f" % np.sqrt(.95 * .05 / a.trials), "",
         "  bias = mean(log MAP - log truth); sd = std over trials.",
         "  chi2min/m: mean whitened chi2 at MAP divided by modes kept (~1 if calibrated).", ""]
    hdr = (f"  {'design':<14}{'case':<11}{'cov_Lap':>8}{'cov_LR':>8}{'bias_ls':>9}"
           f"{'bias_ll':>9}{'sd_ls':>8}{'sd_ll':>8}{'chi2/m':>8}")
    L.append(hdr)
    for (dn, c), rr in rec.items():
        ls = np.array([r["ls"] for r in rr]) - lt_true[0]
        ll = np.array([r["ll"] for r in rr]) - lt_true[1]
        L.append(f"  {dn:<14}{c:<11}{np.mean([r['lap'] for r in rr]):8.3f}"
                 f"{np.mean([r['lr'] for r in rr]):8.3f}{ls.mean():+9.3f}{ll.mean():+9.3f}"
                 f"{ls.std():8.3f}{ll.std():8.3f}"
                 f"{np.mean([r['chi2min'] / max(r['m'] - 2, 1) for r in rr]):8.2f}")
        np.savetxt(os.path.join(a.out, f"diag_trials_{dn}_{c}.txt"),
                   np.array([[r["t"], r["ls"], r["ll"], r["lap"], r["lr"], r["lap_d2"],
                              r["lr_d2"], r["chi2min"], r["m"]] for r in rr]),
                   header="trial logsig_MAP loglc_MAP covered_laplace covered_LR "
                          "laplace_d2 LR_d2 chi2min modes", fmt="%.6e")
    L += ["", "  (chi2/m uses m-2 degrees of freedom; noiseless row has a single trial.)",
          f"  wall time {time.time() - t0:.0f} s"]
    with open(os.path.join(a.out, "diagnostics_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
