"""Fig. 2: same adoption and mean skill, different distributions of skill (Sec. III).

c = 0.95 (a single stable state) and b tuned so that x = 0.4, hence S = 0.6, for rho = 0.039, 0.39
and 3.9. Simulations of 2 x 10^4 agents are cached in figures/abm_distributions.json; delete it to
rerun them (a few minutes). Prints the numbers quoted in Sec. III.
"""
import numpy as np, warnings, json, os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from coevo import *
from style import *
warnings.filterwarnings('ignore')

base = Params(c=0.95, mu=0.02, beta=0.8)
xt = 0.4                                      # adoption in every panel
N = 20000
TH = 0.2                                      # agents with skill below TH count as deskilled
EPS = [0.02, 0.2, 2.0]                        # rho = 0.039, 0.39, 3.9
EDGES = np.linspace(0, 1, 51)
rng = np.random.default_rng(3)

# ---- simulations: histograms pooled over 10 snapshots of the stationary state
cache = 'figures/abm_distributions.json'
C = json.load(open(cache)) if os.path.exists(cache) else {}
if 'panels' not in C:
    C['panels'] = {}
    for eps in EPS:
        p = base.with_(eps=eps)
        b, mH, mA = exact_branch(p, np.array([xt]))[0]
        q = p.with_(b=b)
        burn = max(150, 12 / eps)
        r = simulate(q, N=N, T=burn, dt=0.02, x0=xt, s0=np.full(N, 1 - xt), rng=rng, record_dt=5)
        samples = []
        sig, s = r['sigma'], r['s']
        for rep in range(10):
            r = simulate(q, N=N, T=20, dt=0.02, sigma0=sig, s0=s, rng=rng, record_dt=5)
            sig, s = r['sigma'], r['s']
            samples.append(s.copy())
        samples = np.concatenate(samples)
        C['panels'][str(eps)] = dict(hist=np.histogram(samples, bins=EDGES, density=True)[0].tolist(),
                                     x=float((sig == 1).mean()), S=float(samples.mean()),
                                     P=float((samples < TH).mean()))
    json.dump(C, open(cache, 'w'))

# ---- figure: histograms and exact densities, Eq. (7)
mids = (EDGES[1:] + EDGES[:-1]) / 2
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
for k, eps in enumerate(EPS):
    p = base.with_(eps=eps)
    b, mH, mA = exact_branch(p, np.array([xt]))[0]
    q = p.with_(b=b)
    co = rate_coeffs(xt, mH, mA, q)
    st = pdmp_stats(co, q, theta=TH)
    sg = np.linspace(1e-4, 1 - 1e-4, 2000)
    nH, nA = pdmp_pdf(sg, co, q, st)
    d = C['panels'][str(eps)]
    ax[k].bar(mids, d['hist'], width=EDGES[1] - EDGES[0], color=SEQ[0], edgecolor='white', lw=0.3)
    ax[k].plot(sg, nH + nA, color=BLUE, lw=1.6)
    ax[k].plot(sg, nA, color=ORANGE, lw=1.0, ls='--')
    ax[k].plot(sg, nH, color=AQUA, lw=1.0, ls='--')
    ax[k].set_ylim(0, {0: 5.5, 1: 2.2, 2: 8}[k])
    ax[k].set_xlabel('independent skill $s$')
    ax[k].set_title(f'({"abc"[k]}) $\\rho={float(rho_of(eps, base)):.2g}$', loc='left')
    ax[k].text(0.5 if k == 2 else 0.03, 0.97,
               f'$x={d["x"]:.3f}$, $S={d["S"]:.3f}$\n$P(s<{TH:g})={st["P_below"]:.2f}$',
               transform=ax[k].transAxes, va='top', ha='center' if k == 2 else 'left', fontsize=6.6, color=INK2)
    print(f'rho = {float(rho_of(eps, base)):.3f}: b = {b:.3f}; m_H = {mH:.3f}, m_A = {mA:.3f}, gap = {mH - mA:.3f}; '
          f'P(s < {TH}) exact {st["P_below"]:.3f}, simulations {d["P"]:.3f}; simulations x = {d["x"]:.3f}, S = {d["S"]:.3f}; '
          f'mean payoff of non-users minus AI users {mH - (b - base.c * xt):+.4f}')
ax[0].set_ylabel('density')
ax[2].legend(handles=[Patch(color=SEQ[0], label='simulations'), Line2D([], [], color=BLUE, label='exact, all'),
                      Line2D([], [], color=AQUA, ls='--', lw=1, label='exact, non-users'),
                      Line2D([], [], color=ORANGE, ls='--', lw=1, label='exact, AI users')],
             fontsize=6.5, loc='center', bbox_to_anchor=(0.5, 0.45))
fig.tight_layout(w_pad=0.8)
fig.savefig('figures/fig_letter_distributions.pdf'); fig.savefig('figures/fig_letter_distributions.png')
