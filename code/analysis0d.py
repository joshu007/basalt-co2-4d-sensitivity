"""Corridor-cell (0-D) analyses: fluid substitution, stiffness, f_inf-tau sweep,
Monte Carlo uncertainty, non-uniqueness (Vp vs Vs) and end-member sensitivity.

python analysis0d.py -> ../results/analysis0d.json, analysis0d.npz
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
import rp

OUT = Path(__file__).parents[1] / 'results'
VP0, VS0, R0 = rp.reference()
res, arr = {}, {}

# 1) Fluid substitution: uniform vs patchy, free CO2, no cement --------------
S = np.linspace(0, 1, 201)
kf, rf = rp.reuss_fluid(S, rp.K_CO2, rp.RHO_CO2)
vpu, vsu, _ = rp.velocities(rp.PHI_C, rp.KD_C, rp.MU_C, kf, rf)
vpp, vsp = rp.patchy_vp(S, rp.K_CO2, rp.RHO_CO2)
arr.update(S=S, dvp_uni=100 * (vpu / VP0 - 1), dvs_uni=100 * (vsu / VS0 - 1),
           dvp_pat=100 * (vpp / VP0 - 1), dvs_pat=100 * (vsp / VS0 - 1))
idx = {s: int(np.argmin(np.abs(S - s))) for s in (0.05, 0.2, 0.6, 1.0)}
res['fluid'] = {f'{int(100*s)}': dict(uniform=float(arr['dvp_uni'][i]), patchy=float(arr['dvp_pat'][i]),
                                       dvs_uniform=float(arr['dvs_uni'][i])) for s, i in idx.items()}
res['fluid']['kfl_5pct_GPa'] = float(kf[idx[0.05]] / 1e9)
res['reference'] = dict(vp=float(VP0), vs=float(VS0), rho=float(R0), vpvs=float(VP0 / VS0))

# 2) Frame stiffness factor ----------------------------------------------
fac = np.linspace(0.75, 1.5, 31)
d_free, d_dis = [], []
for a in fac:
    v0 = rp.reference(a)[0]
    d_free.append(100 * (rp.state(0, finf=0, tau=1, S=0.6, Kinj=rp.K_CO2, rinj=rp.RHO_CO2, stiff=a)[0] / v0 - 1))
    d_dis.append(100 * (rp.state(24, stiff=a, **rp.DISSOLVED)[0] / v0 - 1))
arr.update(fac=fac, stiff_free=np.array(d_free), stiff_dis=np.array(d_dis))
res['stiffness'] = {f'{a:.2f}': dict(free_static=float(np.interp(a, fac, d_free)),
                                     dissolved_24=float(np.interp(a, fac, d_dis)),
                                     vp_ref=float(rp.reference(a)[0]))
                    for a in (0.75, 1.0, 1.25, 1.5)}

# 3) f_inf - tau sweep at 24 months (dissolved fluid) ---------------------
finf = np.linspace(0, 0.5, 101)
tau = np.linspace(1, 36, 71)
F, T = np.meshgrid(finf, tau)
sc = dict(rp.DISSOLVED)
sc.pop('finf'); sc.pop('tau')
G = 100 * (rp.state(24, finf=F, tau=T, **sc)[0] / VP0 - 1)
arr.update(finf=finf, tau=tau, sweep_dvp24=G)
res['sweep'] = {f'{f:.2f}': float(100 * (rp.state(24, finf=f, tau=7.5, **sc)[0] / VP0 - 1))
                for f in (0.0, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5)}
# f_inf needed (tau 7.5 mo) to offset the fluid effect and to reach +5 %, +10 %
g = lambda f, target: 100 * (rp.state(24, finf=f, tau=7.5, **sc)[0] / VP0 - 1) - target
res['sweep_thresholds'] = {str(tg): float(brentq(g, 0, 0.5, args=(tg,))) for tg in (0, 5, 10)}

# 4) Monte Carlo ----------------------------------------------------------
rng = np.random.default_rng(20261001)
N = 20000
end = rng.uniform(1.75, 1.90, N)   # Vp/Vs of the massive end member


def mc(fi, ta, Sx, kinj, rinj, st, vpvs):
    out = np.empty(N); outs = np.empty(N)
    for i in range(N):
        kmax, mumax = rp.inverse_gassmann(rp.VP_MAS, rp.VP_MAS / vpvs[i], rp.PHI_MAS)
        f = rp.cement_fraction(24, fi[i], ta[i])
        phi, kd, mu = rp.frame(f, kd0=rp.KD_C * st[i], mu0=rp.MU_C * st[i], kmax=kmax, mumax=mumax)
        kf_, rf_ = rp.reuss_fluid(Sx[i], kinj, rinj)
        vp, vs, _ = rp.velocities(phi, kd, mu, kf_, rf_)
        v0, s0, _ = rp.reference(st[i])
        out[i] = 100 * (vp / v0 - 1); outs[i] = 100 * (vs / s0 - 1)
    return out, outs


st = rng.uniform(0.75, 1.5, N)
dis, dis_s = mc(rng.uniform(0.3, 0.5, N), rng.uniform(5, 10, N), np.ones(N), rp.K_DIS, rp.RHO_DIS, st, end)
fre, fre_s = mc(rng.uniform(0.02, 0.06, N), rng.uniform(5, 10, N), rng.uniform(0.4, 0.8, N), rp.K_CO2, rp.RHO_CO2,
                rng.uniform(0.75, 1.5, N), end)
wide, _ = mc(rng.uniform(0.0, 0.5, N), rng.uniform(5, 10, N), np.ones(N), rp.K_DIS, rp.RHO_DIS,
             rng.uniform(0.75, 1.5, N), end)
arr.update(mc_dis=dis, mc_free=fre, mc_wide=wide, mc_dis_s=dis_s, mc_free_s=fre_s)
q = lambda a: dict(p10=float(np.percentile(a, 10)), p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)),
                   min=float(a.min()), max=float(a.max()))
res['montecarlo'] = dict(N=N, seed=20261001, dissolved=q(dis), free=q(fre), dissolved_wide_finf=q(wide),
                         dissolved_vs=q(dis_s), free_vs=q(fre_s),
                         frac_wide_below_2pct=float(np.mean(np.abs(wide) < 2)))
# time evolution bands (P10-P90) on a reduced ensemble
tt = np.linspace(0, 24, 97)
M = 2000


def band(params, kinj, rinj):
    curves = []
    for i in range(M):
        fi, ta, Sx, a, vpvs = (p[i] for p in params)
        kmax, mumax = rp.inverse_gassmann(rp.VP_MAS, rp.VP_MAS / vpvs, rp.PHI_MAS)
        f = rp.cement_fraction(tt, fi, ta)
        phi, kd, mu = rp.frame(f, kd0=rp.KD_C * a, mu0=rp.MU_C * a, kmax=kmax, mumax=mumax)
        kf_, rf_ = rp.reuss_fluid(Sx, kinj, rinj)
        curves.append(100 * (rp.velocities(phi, kd, mu, kf_, rf_)[0] / rp.reference(a)[0] - 1))
    c = np.array(curves)
    return np.percentile(c, 10, axis=0), np.percentile(c, 50, axis=0), np.percentile(c, 90, axis=0)


r2 = np.random.default_rng(7)
pd_ = [r2.uniform(0.3, 0.5, M), r2.uniform(5, 10, M), np.ones(M), r2.uniform(0.75, 1.5, M), r2.uniform(1.75, 1.9, M)]
pf_ = [r2.uniform(0.02, 0.06, M), r2.uniform(5, 10, M), r2.uniform(0.4, 0.8, M), r2.uniform(0.75, 1.5, M), r2.uniform(1.75, 1.9, M)]
arr['tt'] = tt
arr['band_dis'] = np.array(band(pd_, rp.K_DIS, rp.RHO_DIS))
arr['band_free'] = np.array(band(pf_, rp.K_CO2, rp.RHO_CO2))
cd = 100 * (rp.state(tt, **rp.DISSOLVED)[0] / VP0 - 1)
cf = 100 * (rp.state(tt, **rp.FREE)[0] / VP0 - 1)
arr.update(central_dis=cd, central_free=cf,
           phi_dis=100 * rp.state(tt, **rp.DISSOLVED)[3], phi_free=100 * rp.state(tt, **rp.FREE)[3])
res['time'] = dict(dis_t0=float(cd[0]), free_t0=float(cf[0]), dis_24=float(cd[-1]), free_24=float(cf[-1]),
                   dis_cross_months=float(np.interp(0, cd, tt)),
                   dis_vs_24=float(100 * (rp.state(24, **rp.DISSOLVED)[1] / VS0 - 1)),
                   free_vs_24=float(100 * (rp.state(24, **rp.FREE)[1] / VS0 - 1)),
                   dis_phi_24=float(100 * rp.state(24, **rp.DISSOLVED)[3]),
                   free_phi_24=float(100 * rp.state(24, **rp.FREE)[3]),
                   dis_rho_24=float(100 * (rp.state(24, **rp.DISSOLVED)[2] / R0 - 1)),
                   frac_change_first12=float((cd[48] - cd[0]) / (cd[-1] - cd[0])))

# 5) Non-uniqueness: free-CO2 saturation x cemented fraction ---------------
Sg = np.linspace(0, 0.8, 161)
fg = np.linspace(0, 0.5, 201)
SS, FF = np.meshgrid(Sg, fg)
phi, kd, mu = rp.frame(FF)
kf_, rf_ = rp.reuss_fluid(SS, rp.K_CO2, rp.RHO_CO2)
vp, vs, rho = rp.velocities(phi, kd, mu, kf_, rf_)
arr.update(nu_S=Sg, nu_f=fg, nu_dvp=100 * (vp / VP0 - 1), nu_dvs=100 * (vs / VS0 - 1),
           nu_dai=100 * (vp * rho / (VP0 * R0) - 1))


def f_zero(s):
    def h(f):
        p, k, m = rp.frame(f)
        kf1, rf1 = rp.reuss_fluid(s, rp.K_CO2, rp.RHO_CO2)
        return rp.velocities(p, k, m, kf1, rf1)[0] - VP0
    return brentq(h, 0, 0.99)


pts = {}
for name, s in (('A', 0.0), ('B', 0.05), ('C', 0.6)):
    f0 = 0.0 if s == 0 else f_zero(s)
    p, k, m = rp.frame(f0)
    kf1, rf1 = rp.reuss_fluid(s, rp.K_CO2, rp.RHO_CO2)
    v, w, r = rp.velocities(p, k, m, kf1, rf1)
    pts[name] = dict(S=s, f=float(f0), dvp=float(100 * (v / VP0 - 1)), dvs=float(100 * (w / VS0 - 1)),
                     vpvs=float(v / w), drho=float(100 * (r / R0 - 1)))
res['nonuniqueness'] = pts

# 6) End-member sensitivity (dissolved, 24 months) -------------------------
em = {}
for vpvs in (1.75, 1.80, 1.85, 1.90):
    kmax, mumax = rp.inverse_gassmann(rp.VP_MAS, rp.VP_MAS / vpvs, rp.PHI_MAS)
    f = rp.cement_fraction(24, 0.4, 7.5)
    p, k, m = rp.frame(f, kmax=kmax, mumax=mumax)
    v, w, _ = rp.velocities(p, k, m, rp.K_DIS, rp.RHO_DIS)
    em[f'{vpvs:.2f}'] = dict(Kmax=float(kmax / 1e9), mumax=float(mumax / 1e9), dvp=float(100 * (v / VP0 - 1)),
                            dvs=float(100 * (w / VS0 - 1)))
res['endmember'] = em
# HS check of all frames
hs = {}
for name, (ph, k, m) in dict(corridor=(rp.PHI_C, rp.KD_C, rp.MU_C), massive=(rp.PHI_MAS, rp.KD_MAS, rp.MU_MAS),
                             vesicular=(rp.PHI_VES, rp.KD_VES, rp.MU_VES)).items():
    ku, mu_ = rp.hs_upper_dry(ph)
    hs[name] = dict(K=float(k / 1e9), mu=float(m / 1e9), K_HS=float(ku / 1e9), mu_HS=float(mu_ / 1e9),
                    inside=bool(k <= ku and m <= mu_))
res['hs_check'] = hs

# 7) Seismic resolution
lam = VP0 / 25
res['resolution'] = dict(wavelength_m=float(lam), tuning_m=float(lam / 4))

np.savez_compressed(OUT / 'analysis0d.npz', **arr)
(OUT / 'analysis0d.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
