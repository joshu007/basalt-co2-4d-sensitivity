"""Manuscript figures (English). python figures.py -> ../figures/*.png|pdf"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, str(Path(__file__).parent))
import rp

R = Path(__file__).parents[1] / 'results'
OUT = Path(__file__).parents[1] / 'figures'
OUT.mkdir(exist_ok=True)
M = np.load(R / 'model2d.npz'); MJ = json.loads((R / 'model2d.json').read_text())
A = np.load(R / 'analysis0d.npz'); AJ = json.loads((R / 'analysis0d.json').read_text())

DIS, FREE, GRN, PUR, INK, GRY = '#D55E00', '#0072B2', '#009E73', '#6B4C9A', '#1F2937', '#9CA3AF'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5,
                     'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#444', 'axes.linewidth': 0.7, 'lines.linewidth': 1.6,
                     'axes.unicode_minus': True, 'savefig.dpi': 600, 'figure.dpi': 150})
W1, W2 = 3.5, 7.16
x = M['x_m'] / 1000; z = M['z_m']; t = M['t_s']; F = M['facies']
FAC_COL = {0: '#E9D8A6', 3: '#9C8573', 4: '#4B5563', 5: '#2BB5AE', 6: '#C04FC6'}
FAC_NAME = {0: 'Overburden', 3: 'Lower unit', 4: 'Massive basalt', 5: 'Vesicular lens', 6: 'Fractured corridor'}


def panel(ax, s):
    ax.set_title(s, loc='left', fontweight='bold')


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def facies_img():
    img = np.ones(F.shape + (3,))
    for k, c in FAC_COL.items():
        img[F == k] = matplotlib.colors.to_rgb(c)
    return img


# Fig. 1 workflow ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(W2, 2.0)); ax.axis('off'); ax.set_xlim(0, 10.4); ax.set_ylim(0, 3)
boxes = [
    (0.1, 'Inputs', 'Grain and fluid moduli\nSAG-P2/P3 log medians\nCarbFix and Wallula\nfield constraints', '#E8EEF5'),
    (2.7, 'Rock physics', 'Cementation law $f(t)$\nPorosity loss and\nframe stiffening\nReuss/patchy + Gassmann', '#FCEBDD'),
    (5.3, 'Uncertainty', 'Stiffness, end member\n$f_\\infty$–$\\tau$ sweep\nMonte Carlo ($N$ = 20 000)\n$V_P$ vs $V_S$ ambiguity', '#E6F4EE'),
    (7.9, '2-D monitoring', 'Reservoir 4 × 1.6 km\nConvolutional 4-D\nNoise test, time shifts\nSynthetic well logs', '#EFE8F5'),
]
for x0, h, body, c in boxes:
    ax.add_patch(FancyBboxPatch((x0, 0.25), 2.25, 2.5, boxstyle='round,pad=0.04,rounding_size=0.12', fc=c, ec='#6B7280', lw=0.8))
    ax.text(x0 + 1.125, 2.45, h, ha='center', va='center', fontweight='bold', fontsize=8.5)
    ax.text(x0 + 1.125, 1.3, body, ha='center', va='center', fontsize=6.8, linespacing=1.45)
for x0 in (2.35, 4.95, 7.55):
    ax.annotate('', xy=(x0 + 0.33, 1.5), xytext=(x0 + 0.02, 1.5), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.2))
save(fig, 'fig01_workflow')

# Fig. 2 model geometry --------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), constrained_layout=True)
axs[0].imshow(facies_img(), extent=[0, 4, 1600, 0], aspect='auto', interpolation='nearest')
wells = MJ['wells_x_m']
for name, lab in (('corridor', 'W1'), ('lens', 'W2'), ('control', 'W3')):
    xx = wells[name] / 1000
    axs[0].axvline(min(xx, 3.985), color='white', ls=(0, (4, 3)), lw=0.9)
    axs[0].text(min(xx, 3.93), 60, lab, color=INK, fontsize=7, ha='center', fontweight='bold',
                bbox=dict(fc='white', ec='none', pad=1.2, alpha=0.9))
axs[0].set(xlabel='Distance (km)', ylabel='Depth (m)'); panel(axs[0], '(a) Facies')
axs[0].legend(handles=[Patch(color=FAC_COL[k], label=FAC_NAME[k]) for k in (0, 4, 5, 6, 3)], loc='lower left',
              fontsize=6, frameon=True, framealpha=0.9, ncol=2, handlelength=1, borderpad=0.4)
im = axs[1].imshow(np.ma.masked_where(M['sn'] <= 0, M['sn']), extent=[0, 4, 1600, 0], aspect='auto', cmap='magma_r',
                   vmin=0, vmax=1, interpolation='nearest')
axs[1].imshow(np.where(M['sn'][..., None] > 0, np.nan, facies_img() * 0 + 0.93), extent=[0, 4, 1600, 0], aspect='auto')
axs[1].set(xlabel='Distance (km)', yticklabels=[]); panel(axs[1], '(b) Normalized invasion $s_n$')
fig.colorbar(im, ax=axs[1], shrink=0.85, label='$s_n$ (–)')
save(fig, 'fig02_model')

# Fig. 3 fluid substitution + stiffness ----------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), constrained_layout=True)
S = A['S'] * 100
ax = axs[0]
ax.axhline(0, color=GRY, lw=0.7)
ax.plot(S, A['dvp_uni'], color=FREE, label='$V_P$, uniform (Reuss)')
ax.plot(S, A['dvp_pat'], color=FREE, ls='--', label='$V_P$, patchy (Hill)')
ax.plot(S, A['dvs_uni'], color=GRN, label='$V_S$ (both limits)')
for s in (5, 60):
    i = np.argmin(np.abs(S - s))
    ax.plot(s, A['dvp_uni'][i], 'o', ms=4.5, color=FREE, mec='white', mew=0.8, zorder=5)
    ax.annotate(f"{A['dvp_uni'][i]:.1f}%".replace('-', '−'), (s, A['dvp_uni'][i]), xytext=(4, -9), textcoords='offset points', fontsize=7)
ax.set(xlabel='Free-CO$_2$ saturation (%)', ylabel='Change relative to brine (%)', xlim=(0, 100), ylim=(-18.5, 2))
ax.legend(loc='center right', frameon=False); panel(ax, '(a) Fluid substitution (no cement)')
ax = axs[1]
ax.axhline(0, color=GRY, lw=0.7)
ax.plot(A['fac'], A['stiff_dis'], color=DIS, label='Dissolved, 24 months')
ax.plot(A['fac'], A['stiff_free'], color=FREE, label='Free CO$_2$ ($S$ = 0.6), no cement')
for a in (0.75, 1.0, 1.25, 1.5):
    for key, c in (('stiff_dis', DIS), ('stiff_free', FREE)):
        v = np.interp(a, A['fac'], A[key]); ax.plot(a, v, 'o', ms=4, color=c, mec='white', mew=0.8)
        ax.annotate(f'{v:+.1f}%'.replace('-', '−'), (a, v), xytext=(9 if a == 0.75 else 0, 5 if v > 0 else -10), textcoords='offset points', ha='center', fontsize=6.5)
ax.set(xlabel='Frame-stiffness factor on $K_{dry}$ and $\\mu_{dry}$ (–)', ylabel='$\\Delta V_P$ (%)', ylim=(-25, 21))
ax.legend(loc='center right', frameon=False); panel(ax, '(b) Sensitivity to frame stiffness')
save(fig, 'fig03_rockphysics')

# Fig. 4 time evolution --------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.5, 1]))
tt = A['tt']; ax = axs[0]
ax.axhline(0, color=GRY, lw=0.7, ls='--')
for key, c, lab in (('dis', DIS, 'Dissolved CO$_2$'), ('free', FREE, 'Free CO$_2$')):
    b = A['band_' + key]
    ax.fill_between(tt, b[0], b[2], color=c, alpha=0.18, lw=0)
    ax.plot(tt, A['central_' + key], color=c, label=lab + ' (central)')
ax.text(24.3, A['central_dis'][-1], f"{A['central_dis'][-1]:+.1f}%", color=INK, fontsize=7, va='center')
ax.text(24.3, A['central_free'][-1], f"{A['central_free'][-1]:+.1f}%".replace('-', '−'), color=INK, fontsize=7, va='center')
ax.set(xlabel='Time since injection (months)', ylabel='$\\Delta V_P$ relative to brine (%)', xlim=(0, 27), xticks=[0, 6, 12, 18, 24])
ax.legend(loc='center right', frameon=False, bbox_to_anchor=(1, 0.6)); panel(ax, '(a) Corridor $V_P$ with P10–P90 bands')
ax = axs[1]
ax.plot(tt, A['phi_dis'], color=DIS, label='Dissolved'); ax.plot(tt, A['phi_free'], color=FREE, label='Free')
ax.set(xlabel='Time since injection (months)', ylabel='Porosity (%)', xticks=[0, 6, 12, 18, 24], ylim=(2.5, 4.7))
ax.legend(frameon=False); panel(ax, '(b) Porosity')
save(fig, 'fig04_time')

# Fig. 5 f_inf - tau sweep -----------------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 2.8), constrained_layout=True)
G = A['sweep_dvp24']
im = ax.pcolormesh(A['finf'] * 100, A['tau'], G, cmap='RdBu_r', norm=TwoSlopeNorm(0, -17, 17), shading='auto', rasterized=True)
cs = ax.contour(A['finf'] * 100, A['tau'], G, levels=[0, 2, 5, 10, 15], colors=INK, linewidths=0.7)
ax.clabel(cs, fmt=lambda v: f'{v:+.0f}%', fontsize=6.5)
ax.plot(40, 7.5, '*', ms=9, color='white', mec=INK, mew=0.8); ax.annotate('Dissolved\nscenario', (40, 7.5), xytext=(-8, 8), textcoords='offset points', ha='right', fontsize=6.8)
ax.plot(4, 9, 'o', ms=5, color='white', mec=INK, mew=0.8); ax.annotate('Wallula-like\nfilling (~4%)', (4, 9), xytext=(6, -4), textcoords='offset points', fontsize=6.8, va='top',
                                                                     bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
ax.set(xlabel='$f_\\infty$: final cemented pore fraction (%)', ylabel='$\\tau$ (months)')
fig.colorbar(im, ax=ax, label='$\\Delta V_P$ at 24 months (%)')
save(fig, 'fig05_sweep')

# Fig. 6 non-uniqueness ---------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.8), constrained_layout=True)
Sg, fg = A['nu_S'] * 100, A['nu_f'] * 100
P = AJ['nonuniqueness']
for ax, key, lab, title in ((axs[0], 'nu_dvp', '$\\Delta V_P$ (%)', '(a) $\\Delta V_P$'), (axs[1], 'nu_dvs', '$\\Delta V_S$ (%)', '(b) $\\Delta V_S$')):
    Z = A[key]; lim = np.abs(Z).max()
    im = ax.pcolormesh(Sg, fg, Z, cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim), shading='auto', rasterized=True)
    cs = ax.contour(Sg, fg, Z, levels=[-10, -5, 5, 10, 20, 30], colors=INK, linewidths=0.5, alpha=0.7)
    ax.clabel(cs, fmt=lambda v: f'{v:+.0f}'.replace('-', '−'), fontsize=6)
    z0 = ax.contour(Sg, fg, A['nu_dvp'], levels=[0], colors='k', linewidths=1.6)
    for n, p in P.items():
        ax.plot(p['S'] * 100, p['f'] * 100, 'o', ms=6, color='#FFD166', mec='k', mew=0.8, zorder=6, clip_on=False)
        lab_ = n if key == 'nu_dvp' else f"{n} ({p['dvs']:+.1f}%)"
        ax.annotate(lab_, (p['S'] * 100, p['f'] * 100), xytext=(5, 4), textcoords='offset points', fontsize=7, fontweight='bold')
    ax.set(xlabel='Free-CO$_2$ saturation (%)', ylabel='Cemented pore fraction $f$ (%)')
    panel(ax, title); fig.colorbar(im, ax=ax, label=lab)
axs[0].text(30, 5, 'black line: $\\Delta V_P$ = 0', fontsize=6.8, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
save(fig, 'fig06_nonuniqueness')

# Fig. 7 spatial maps -----------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5), constrained_layout=True)
for ax, n, title in ((axs[0], 'dissolved', '(a) Dissolved CO$_2$, 24 months'), (axs[1], 'free', '(b) Free CO$_2$, 24 months')):
    dv = 100 * (M[f'{n}_24_vp'] / M['vp0'] - 1)
    im = ax.imshow(np.ma.masked_where(np.abs(dv) < 0.05, dv), extent=[0, 4, 1600, 0], aspect='auto', cmap='RdBu_r',
                   norm=TwoSlopeNorm(0, -16, 16), interpolation='nearest')
    edge = np.zeros(F.shape + (4,)); edge[F == 4] = (0.85, 0.85, 0.85, 1); edge[F == 0] = (0.95, 0.93, 0.88, 1); edge[F == 3] = (0.8, 0.77, 0.74, 1)
    ax.imshow(np.where((np.abs(dv) < 0.05)[..., None], edge, np.nan), extent=[0, 4, 1600, 0], aspect='auto', interpolation='nearest')
    ax.set(xlabel='Distance (km)', ylabel='Depth (m)' if n == 'dissolved' else None)
    panel(ax, title)
fig.colorbar(im, ax=axs, shrink=0.85, label='$\\Delta V_P$ (%)')
save(fig, 'fig07_maps')

# Fig. 8 seismic -------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6), constrained_layout=True, sharey=True)
amp = np.percentile(np.abs(M['base']), 99.5)
axs[0].imshow(M['base'], extent=[0, 4, t[-1] * 1000, 0], aspect='auto', cmap='seismic', vmin=-amp, vmax=amp)
dlim = max(np.abs(M['dissolved_24_diff']).max(), np.abs(M['free_24_diff']).max()) * 0.6
for ax, n, title in ((axs[1], 'dissolved', '(b) Difference, dissolved'), (axs[2], 'free', '(c) Difference, free')):
    im = ax.imshow(M[f'{n}_24_diff'], extent=[0, 4, t[-1] * 1000, 0], aspect='auto', cmap='seismic', vmin=-dlim, vmax=dlim)
    panel(ax, title)
panel(axs[0], '(a) Baseline')
for ax in axs:
    ax.set_ylim(800, 150); ax.set_xlabel('Distance (km)')
    ax.axvline(wells['corridor'] / 1000, color=INK, lw=0.6, ls=':')
axs[0].set_ylabel('TWT (ms)')
fig.colorbar(im, ax=axs, shrink=0.85, label='Amplitude (reflectivity units)')
save(fig, 'fig08_seismic')

# Fig. 9 noise ------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axs[0]
sig = [0.002, 0.005, 0.01, 0.02, 0.04]
for n, c in (('dissolved', DIS), ('free', FREE)):
    sv = MJ['scenarios'][n]['24']['snr_vs_sigma']
    for tg, ls, mk in (('corridors', '-', 'o'), ('lenses', '--', 's')):
        ax.plot(sig, [sv[tg][str(s)] for s in sig], ls=ls, marker=mk, ms=3.5, color=c,
                label=f"{'Dissolved' if n == 'dissolved' else 'Free'}, {tg}")
ax.axhline(0, color=GRY, lw=0.8); ax.axvline(0.01, color=GRY, lw=0.6, ls=':')
ax.set(xscale='log', xlabel='Noise std per survey $\\sigma$ (reflectivity units)', ylabel='Expected SNR of 4-D difference (dB)')
ax.set_xticks(sig); ax.set_xticklabels([str(s) for s in sig])
ax.legend(frameon=False, fontsize=6.3); panel(ax, '(a) Detectability vs noise')
ax = axs[1]
clean = M['free_24_combined_clean']
rng = np.random.default_rng(910)
noisy = clean + rng.normal(0, 0.01, clean.shape) - rng.normal(0, 0.01, clean.shape)
from scipy.signal import butter, sosfiltfilt
proc = sosfiltfilt(butter(3, (5, 65), btype='bandpass', fs=500, output='sos'), noisy, axis=0)
lim = np.abs(clean).max() * 0.5
im = ax.imshow(proc, extent=[0, 4, t[-1] * 1000, 0], aspect='auto', cmap='seismic', vmin=-lim, vmax=lim)
ax.set(ylim=(800, 150), xlabel='Distance (km)', ylabel='TWT (ms)'); panel(ax, '(b) Free CO$_2$: noisy difference after 5–65 Hz filter')
fig.colorbar(im, ax=ax, shrink=0.85, label='Amplitude (reflectivity units)')
save(fig, 'fig09_noise')

# Fig. 10 synthetic well -----------------------------------------------------------------
ix = int(M['wells_ix'][0])
fig, axs = plt.subplots(1, 5, figsize=(W2, 3.0), constrained_layout=True, sharey=True)
times = [0, 1, 12, 24]; cols = ['#1F2937', '#6B4C9A', '#CC79A7', DIS]
fields = [('vp', '$V_P$ (km s$^{-1}$)', 1e-3), ('vs', '$V_S$ (km s$^{-1}$)', 1e-3), ('rho', '$\\rho$ (g cm$^{-3}$)', 1e-3),
          ('phi', '$\\phi$ (%)', 100), ('ai', 'AI (km s$^{-1}$ g cm$^{-3}$)', 1e-6)]
zc = (F[:, ix] == 6)
z_top, z_bot = z[zc].min(), z[zc].max()
for ax, (fld, lab, sc) in zip(axs, fields):
    ax.axhspan(z_top, z_bot, color='#C04FC6', alpha=0.12, lw=0)
    for ti, c in zip(times, cols):
        if fld == 'ai':
            v = M[f'dissolved_{ti}_vp'][:, ix] * M[f'dissolved_{ti}_rho'][:, ix]
        else:
            v = M[f'dissolved_{ti}_{fld}'][:, ix]
        ax.plot(v * sc, z, color=c, lw=1.0, label=f'{ti} mo')
        sel = (z > 650) & (z < 1300)
        lo_, hi_ = ax.get_xlim() if ti else (np.inf, -np.inf)
        ax._rng = (min(getattr(ax, '_rng', (np.inf,))[0], (v * sc)[sel].min()), max(getattr(ax, '_rng', (0, -np.inf))[1], (v * sc)[sel].max()))
    pad = 0.08 * (ax._rng[1] - ax._rng[0]); ax.set_xlim(ax._rng[0] - pad, ax._rng[1] + pad)
    ax.set_xlabel(lab, fontsize=7)
axs[0].set(ylim=(1300, 650), ylabel='Depth (m)')
axs[0].legend(frameon=False, fontsize=6.3, loc='lower left')
axs[0].text(axs[0].get_xlim()[0], z_top - 8, ' corridor', fontsize=6.5, color='#7A2E7E')
save(fig, 'fig10_well')
print('figures written', sorted(p.name for p in OUT.glob('*.png')))
print('corridor depth interval at W1', z_top, z_bot)
