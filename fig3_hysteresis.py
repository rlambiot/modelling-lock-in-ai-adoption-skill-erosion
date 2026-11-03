"""Fig. 3: bistability and hysteresis (Sec. IV), c = 0.15.

(a) Stationary adoption against b for rho = 0.39: exact solution and simulations of 10^4 agents in
    which b is varied linearly at a rate of 10^-4 per time unit, up from x = 0.02 with all skills
    at 1 and down, in a separate run, from x = 0.98 with all skills at 0.
(b) Phase diagram: folds of the exact solution for mu = 0.02, folds measured in simulations by
    independent runs at fixed b (resolution 0.01), and the boundaries b = 1 and b = c + theta*(rho)
    of the bistable region for mu -> 0 (Eq. 8).
Simulations are cached in figures/abm_bistability.json; delete it to rerun them (of the order of ten minutes).
"""
import numpy as np, json, os, warnings
import matplotlib.pyplot as plt
from scipy import optimize
from coevo import *
from style import *
warnings.filterwarnings('ignore')

base = Params(c=0.15, mu=0.02, beta=0.8)
N = 10000
rng = np.random.default_rng(7)
xs = np.linspace(0.003, 0.997, 300)
EPS_ABM = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]


def sweep(p, b0, b1, rate, x0, s_init):
    """Simulation in which b changes linearly from b0 to b1 at the given rate."""
    T = abs(b1 - b0) / rate
    r = simulate(p, N=N, T=T, dt=0.05, x0=x0, s0=np.full(N, s_init), rng=rng, record_dt=2.0,
                 b_of_t=lambda t: b0 + (b1 - b0) * t / T)
    bs = b0 + (b1 - b0) * r['t'] / T
    return bs, r['x']


# ---- simulations: hysteresis loop for panel (a), fold scans for panel (b)
cache = 'figures/abm_bistability.json'
if os.path.exists(cache):
    abm = {(k if k == 'loop' else float(k)): v for k, v in json.load(open(cache)).items()}
else:
    abm = {}
    p = base.with_(eps=0.2)
    rate = 0.01 / 100
    bu, xu = sweep(p, 0.45, 1.0, rate, 0.02, 1.0)
    bd, xd = sweep(p, 1.0, 0.45, rate, 0.98, 0.0)
    abm['loop'] = dict(bu=bu.tolist(), xu=xu.tolist(), bd=bd.tolist(), xd=xd.tolist())
    for eps in EPS_ABM:
        p = base.with_(eps=eps)
        br = exact_branch(p, xs)
        fd, fu = branch_folds(xs, br[:, 0])
        res = {}
        for key, bc, x0, s0 in [('up', fu, 0.02, 1.0), ('down', fd, 0.98, 0.0)]:
            grid = np.round(np.arange(bc - 0.06, bc + 0.061, 0.01), 3)
            surv = []
            for b in grid:
                r = simulate(p.with_(b=b), N=N, T=400, dt=0.05, x0=x0, s0=np.full(N, s0), rng=rng, record_dt=10)
                surv.append(float(r['x'][-5:].mean()))
            res[key] = dict(b=grid.tolist(), x=surv)
        # fold estimate: midpoint between the last b at which the state survives and the first at which it jumps
        bu_, xu_ = np.array(res['up']['b']), np.array(res['up']['x'])
        bd_, xd_ = np.array(res['down']['b']), np.array(res['down']['x'])
        i = np.argmax(xu_ > 0.5); res['b_up'] = float(bu_[i] - 0.005)
        j = np.nonzero(xd_ < 0.5)[0]; res['b_down'] = float(bd_[j[-1]] + 0.005) if len(j) else float('nan')
        abm[eps] = res
    json.dump({str(k): v for k, v in abm.items()}, open(cache, 'w'))

# ---- theory: folds against rho for mu = 0.02, and the boundaries for mu -> 0
r_th = np.geomspace(0.04, 2.76, 60)
ex_f = []
for rho in r_th:
    f = branch_folds(xs, exact_branch(base.with_(eps=rho * (1 + base.mu) / 2), xs)[:, 0])
    ex_f.append(f if f else (np.nan, np.nan))
ex_f = np.array(ex_f)
r_h = np.geomspace(0.03, 10, 80)               # for mu = 0, rho = 2 eps / nu
b_h = np.array([base.c + holdout_threshold(base.with_(mu=0.0, eps=r / 2)) for r in r_h])

# ---- figure
fig, ax = plt.subplots(1, 2, figsize=(5.3, 2.5))
p = base.with_(eps=0.2)
bb = exact_branch(p, xs)[:, 0]
stable = np.gradient(bb) > 0
for mask, ls in [(stable, '-'), (~stable, '--')]:
    ax[0].plot(bb, np.where(mask, xs, np.nan), ls, color=BLUE, lw=1.8)
a = {k: np.array(v) for k, v in abm['loop'].items()}
ax[0].plot(a['bu'][::8], a['xu'][::8], 'o', ms=3, color=ORANGE, mfc='none', mew=0.9)
ax[0].plot(a['bd'][::8], a['xd'][::8], 's', ms=3, color=AQUA, mfc='none', mew=0.9)
fd, fu = branch_folds(xs, bb)
ax[0].annotate('', xy=(fd, 1.1), xytext=(fu, 1.1),
               arrowprops=dict(arrowstyle='<->', color=BLUE, lw=1.2, shrinkA=0, shrinkB=0))
ax[0].text(fu + 0.02, 1.1, f'bistable: {fu - fd:.2f}', color=INK2, fontsize=6.8, va='center')
ax[0].set_xlim(0.4, 1.15); ax[0].set_ylim(-0.02, 1.18)
ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax[0].set_xlabel('AI benefit $b$'); ax[0].set_ylabel('adoption $x$')
ax[0].set_title(f'(a) hysteresis, $\\rho={rho_of(0.2, base):.2f}$', loc='left')
# panel (b) carries the legend; its colours and markers are the same as in panel (a)

r_abm = rho_of(EPS_ABM, base)
ax[1].fill_between(r_th, ex_f[:, 0], ex_f[:, 1], color=BLUE, alpha=0.12, lw=0)
ax[1].plot(r_th, ex_f[:, 0], color=BLUE, label='exact, $\\mu=0.02$')
ax[1].plot(r_th, ex_f[:, 1], color=BLUE)
ax[1].plot(r_h, b_h, '--', color=INK, lw=1.2, label='$\\mu\\to0$, lone invader')
ax[1].axhline(1.0, ls='--', color=INK, lw=1.2)
ax[1].plot(r_abm, [abm[e]['b_up'] for e in EPS_ABM], 'o', color=ORANGE, ms=4, mfc='none', mew=1.1, label='simulations, $b\\uparrow$')
ax[1].plot(r_abm, [abm[e]['b_down'] for e in EPS_ABM], 's', color=AQUA, ms=4, mfc='none', mew=1.1, label='simulations, $b\\downarrow$')
ax[1].text(0.075, 0.69, 'bistable', color=BLUE, fontsize=7.5)
ax[1].text(0.045, 1.045, 'high adoption only', color=INK2, fontsize=7, va='center')
ax[1].text(0.045, 0.135, 'low adoption only', color=INK2, fontsize=7, va='center')
ax[1].set_xscale('log')
ax[1].set_xlabel('ratio of timescales $\\rho$'); ax[1].set_ylabel('AI benefit $b$')
ax[1].set_title('(b) phase diagram', loc='left')
ax[1].set_ylim(0.1, 1.1); ax[1].set_xlim(0.03, 10)
ax[1].legend(fontsize=6.2, loc='lower right', bbox_to_anchor=(1.0, 0.02), handlelength=1.6)
fig.tight_layout(w_pad=0.8)
fig.savefig('figures/fig_letter_hysteresis.pdf'); fig.savefig('figures/fig_letter_hysteresis.png')

# ---- numbers quoted in Sec. IV
print(f'rho = {rho_of(0.2, base):.3f}: exact folds b = {fd:.3f} (down) and {fu:.3f} (up), window {fu - fd:.3f}')
dev = []
for e in EPS_ABM:
    f = branch_folds(xs, exact_branch(base.with_(eps=e), xs)[:, 0])
    dev += [abs(abm[e]['b_down'] - f[0]), abs(abm[e]['b_up'] - f[1])]
    print(f'  rho = {float(rho_of(e, base)):.3f}: exact {f[0]:.3f}, {f[1]:.3f}; simulations {abm[e]["b_down"]:.3f}, {abm[e]["b_up"]:.3f}')
print(f'largest difference between exact and simulated folds: {max(dev):.3f}')
last = np.nonzero(np.isfinite(ex_f[:, 0]))[0][-1]
print(f'mu = 0.02: the bistable region closes between rho = {r_th[last]:.2f} and {r_th[min(last + 1, len(r_th) - 1)]:.2f}')
rc = optimize.brentq(lambda r: base.c + holdout_threshold(base.with_(mu=0.0, eps=r / 2)) - 1, 1, 10)
print(f'mu -> 0: c + theta* = 1 at rho = {rc:.2f}')
