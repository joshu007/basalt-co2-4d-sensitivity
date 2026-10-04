"""2-D synthetic basalt reservoir: time-lapse elastic and seismic response.

python model2d.py  -> ../results/model2d.npz, ../results/model2d.json
Geometry from the original talk code (charla_ccs/modelos.py, geology('basalto')).
All rock-physics parameters from rp.py (single, consistent set).
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / 'charla_ccs'))
import rp
from modelos import geology, synth  # geometry + convolutional modelling

OUT = Path(__file__).parents[1] / 'results'
OUT.mkdir(exist_ok=True)

# Facies: 0 overburden, 3 lower unit, 4 massive basalt, 5 vesicular lens,
# 6 fractured corridor.  Columns: phi, K0, Kdry, mu_dry, rho_grain (SI)
PROPS = {
    0: (0.15, 36e9, 13e9, 10e9, 2650.),
    3: (0.04, 45e9, 29e9, 20e9, 2750.),
    4: (rp.PHI_MAS, rp.K_GR, rp.KD_MAS, rp.MU_MAS, rp.RHO_GR),
    5: (rp.PHI_VES, rp.K_GR, rp.KD_VES, rp.MU_VES, rp.RHO_GR),
    6: (rp.PHI_C, rp.K_GR, rp.KD_C, rp.MU_C, rp.RHO_GR),
}
INVADED = (5, 6)
TIMES = np.array([0, 1, 6, 12, 24], float)   # months
FREQ, DT, NOISE = 25.0, 0.002, 0.01
SEEDS = [910, 911, 912, 913, 914]
BAND = (5, 65)


def facies_arrays(f):
    out = [np.zeros(f.shape) for _ in range(5)]
    for k, p in PROPS.items():
        for a, v in zip(out, p):
            a[f == k] = v
    return out


def elastic(f, sn, t, sc):
    """sn: normalized invasion (0-1); sc: scenario dict; t: months."""
    phi0, k0, kd0, mu0, r0 = facies_arrays(f)
    inv = np.isin(f, INVADED) & (sn > 0)
    cem = np.where(inv, rp.cement_fraction(t, sc['finf'], sc['tau']) * sn, 0.0)
    phi = phi0 * (1 - cem)
    kd = kd0 + cem * (rp.KD_MAX - kd0)
    mu = mu0 + cem * (rp.MU_MAX - mu0)
    S = np.where(inv, sc['S'] * sn, 0.0)
    kfl, rfl = rp.reuss_fluid(S, sc['Kinj'], sc['rinj'])
    ks = kd + (1 - kd / k0) ** 2 / (phi / kfl + (1 - phi) / k0 - kd / k0 ** 2)
    rho = (1 - phi) * r0 + phi * rfl
    return np.sqrt((ks + 4 / 3 * mu) / rho), np.sqrt(mu / rho), rho, phi, S, cem


def twt(vp, z):
    dz = np.diff(z)[:, None]
    return np.vstack([np.zeros(vp.shape[1]), np.cumsum(dz * (1 / vp[:-1] + 1 / vp[1:]), axis=0)])


def ricker(f, dt):
    tw = np.arange(-0.128, 0.128 + dt / 2, dt)
    a = (np.pi * f * tw) ** 2
    return (1 - 2 * a) * np.exp(-a)


def process(a):
    sos = butter(3, BAND, btype='bandpass', fs=1 / DT, output='sos')
    return sosfiltfilt(sos, a, axis=0)


def noise_metrics(clean, win):
    """RMSE, empirical SNR and gain before/after band-pass, 5 seeds."""
    res = {'raw': [], 'proc': []}
    mask = np.broadcast_to(win[:, None], clean.shape)
    peak = np.abs(clean[mask]).max()
    local = (np.abs(clean) > 0.05 * peak) & mask
    for seed in SEEDS:
        r = np.random.default_rng(seed)
        noisy = clean + r.normal(0, NOISE, clean.shape) - r.normal(0, NOISE, clean.shape)
        for key, pred in (('raw', noisy), ('proc', process(noisy))):
            e = pred - clean
            row = {}
            for reg, m in (('global', mask), ('local', local)):
                y, p = clean[m], pred[m]
                mse = np.mean(e[m] ** 2)
                row[reg] = dict(rmse=float(np.sqrt(mse)),
                                snr_db=float(10 * np.log10(np.mean(y ** 2) / mse)),
                                gain=float(np.dot(p, y) / np.dot(y, y)))
            res[key].append(row)
    agg = {}
    for key in res:
        agg[key] = {reg: {m: float(np.mean([r[reg][m] for r in res[key]])) for m in ('rmse', 'snr_db', 'gain')}
                    for reg in ('global', 'local')}
    rms = float(np.sqrt(np.mean(clean[mask] ** 2)))
    agg['rms_clean'] = rms
    agg['snr_expected_db'] = float(20 * np.log10(rms / (np.sqrt(2) * NOISE)))
    return agg


def main():
    x = np.arange(401) * 10.0
    z = np.arange(401) * 4.0
    t = np.arange(1001) * DT
    f, S = geology('basalto', x, z)
    sn = S / S.max()
    w = ricker(FREQ, DT)
    brine = dict(finf=0, tau=1, S=0, Kinj=rp.K_BR, rinj=rp.RHO_BR)
    vp0, vs0, r0, phi0, _, _ = elastic(f, sn, 0, brine)
    base = synth(vp0, r0, z, t, w)
    tw0 = twt(vp0, z)
    win = (t >= 0.18) & (t <= 0.70)
    store = dict(x_m=x, z_m=z, t_s=t, facies=f, sn=sn, vp0=vp0, vs0=vs0, rho0=r0, phi0=phi0, base=base)
    meta = dict(n_cells_corridor=int(((f == 6) & (sn > 0.05 / 0.6)).sum()),
                n_cells_total=int(f.size), times_months=TIMES.tolist(), freq_hz=FREQ,
                noise_sigma=NOISE, seeds=SEEDS, band_hz=BAND,
                facies_vp_brine={str(k): float(vp0[f == k].mean()) for k in PROPS},
                facies_vs_brine={str(k): float(vs0[f == k].mean()) for k in PROPS},
                scenarios={})
    corr = (f == 6) & (sn > 0.05 / 0.6)
    lens = (f == 5) & (sn > 0.05 / 0.6)
    for name, sc in (('dissolved', rp.DISSOLVED), ('free', rp.FREE)):
        md = {}
        for ti in TIMES:
            vp, vs, rho, phi, Sm, cem = elastic(f, sn, ti, sc)
            mon = synth(vp, rho, z, t, w)
            dt_shift = (twt(vp, z) - tw0)[-1]
            tag = f'{name}_{int(ti)}'
            store[tag + '_vp'] = vp; store[tag + '_vs'] = vs
            store[tag + '_rho'] = rho; store[tag + '_phi'] = phi
            store[tag + '_diff'] = mon - base
            d = {}
            for reg, m in (('corridor', corr), ('lens', lens)):
                d[reg] = dict(dvp_pct=float(np.mean(100 * (vp / vp0 - 1)[m])),
                              dvp_p10=float(np.percentile(100 * (vp / vp0 - 1)[m], 10)),
                              dvp_p90=float(np.percentile(100 * (vp / vp0 - 1)[m], 90)),
                              dvs_pct=float(np.mean(100 * (vs / vs0 - 1)[m])),
                              drho_pct=float(np.mean(100 * (rho / r0 - 1)[m])),
                              dai_pct=float(np.mean(100 * (vp * rho / (vp0 * r0) - 1)[m])),
                              phi_pct=float(np.mean(100 * phi[m])))
            d['max_abs_timeshift_ms'] = float(1000 * np.abs(dt_shift).max())
            d['timeshift_ms_at_max'] = float(1000 * dt_shift[np.abs(dt_shift).argmax()])
            d['x_max_timeshift_m'] = float(x[np.abs(dt_shift).argmax()])
            d['rms_diff_window'] = float(np.sqrt(np.mean((mon - base)[win] ** 2)))
            md[str(int(ti))] = d
            if ti == 24:  # noise test per target (alteration restricted)
                nt = {}
                for target, keep in (('lenses', (5,)), ('corridors', (6,)), ('combined', INVADED)):
                    sn_t = np.where(np.isin(f, keep), sn, 0)
                    v_, _, r_, *_ = elastic(f, sn_t, ti, sc)
                    nt[target] = noise_metrics(synth(v_, r_, z, t, w) - base, win)
                    store[f'{name}_24_{target}_clean'] = synth(v_, r_, z, t, w) - base
                d['noise_test'] = nt
                # SNR_expected vs sigma (theoretical power balance)
                d['snr_vs_sigma'] = {tg: {str(s): float(20 * np.log10(nt[tg]['rms_clean'] / (np.sqrt(2) * s)))
                                          for s in (0.002, 0.005, 0.01, 0.02, 0.04)} for tg in nt}
        meta['scenarios'][name] = md
    # Filter-band selection on development seeds (independent of test seeds)
    sel = {}
    for name in ('dissolved', 'free'):
        for target in ('lenses', 'corridors'):
            clean = store[f'{name}_24_{target}_clean']
            for band in ((3, 50), (5, 65), (8, 80)):
                sos = butter(3, band, btype='bandpass', fs=1 / DT, output='sos')
                m = np.broadcast_to(win[:, None], clean.shape)
                g = float(np.dot(sosfiltfilt(sos, clean, axis=0)[m], clean[m]) / np.dot(clean[m], clean[m]))
                e = []
                for seed in (110, 111, 112):
                    r = np.random.default_rng(seed)
                    p = sosfiltfilt(sos, clean + r.normal(0, NOISE, clean.shape) - r.normal(0, NOISE, clean.shape), axis=0)
                    e.append(np.sqrt(np.mean((p - clean)[m] ** 2)))
                d = sel.setdefault(str(band), {'rmse': [], 'gain_dev': []})
                d['rmse'].append(float(np.mean(e))); d['gain_dev'].append(abs(g - 1))
    meta['band_selection'] = {b: dict(mean_rmse=float(np.mean(v['rmse'])), max_gain_dev=float(max(v['gain_dev'])))
                              for b, v in sel.items()}
    meta['baseline_rms_window'] = float(np.sqrt(np.mean(base[win] ** 2)))
    meta['nrms_noise_pct'] = float(200 * np.sqrt(2) * NOISE / (2 * meta['baseline_rms_window']))
    # Synthetic wells: corridor (x=1650 m), lens-dominated, control
    wells = {'corridor': int(np.argmin(np.abs(x - 1650))),
             'control': int(np.argmin(sn.sum(axis=0)))}
    lens_score = np.where(np.any(f == 6, axis=0), -1, (sn * (f == 5)).sum(axis=0))
    wells['lens'] = int(lens_score.argmax())
    meta['wells_x_m'] = {k: float(x[v]) for k, v in wells.items()}
    store['wells_ix'] = np.array([wells['corridor'], wells['lens'], wells['control']])
    np.savez_compressed(OUT / 'model2d.npz', **store)
    (OUT / 'model2d.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ('n_cells_corridor', 'facies_vp_brine', 'wells_x_m')}, indent=1))
    for n in ('dissolved', 'free'):
        d = meta['scenarios'][n]['24']
        print(n, d['corridor'], d['lens'], 'dt', d['max_abs_timeshift_ms'], d['timeshift_ms_at_max'])
        for tg, v in d['noise_test'].items():
            print('  ', tg, 'SNRexp', round(v['snr_expected_db'], 2), 'raw', {k: round(x, 4) for k, x in v['raw']['global'].items()},
                  'proc', {k: round(x, 4) for k, x in v['proc']['global'].items()})


if __name__ == '__main__':
    main()
