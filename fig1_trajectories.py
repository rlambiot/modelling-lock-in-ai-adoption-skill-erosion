"""Fig. 1(b,c): skill of a single agent when behaviour changes often (rho = 0.1) and when it is
persistent (rho = 3). Switching is independent of skill, with rates chosen so that the agent uses
AI 40% of the time; panel (a) of Fig. 1 is drawn in TikZ in the manuscript."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import *

nu, mu, x_share = 1.0, 0.02, 0.4
K = nu * (1 + mu) / 2                       # sum of the two switching rates without selection
k_HA, k_AH = x_share * K, (1 - x_share) * K
s_mean = 1 - x_share
T = 80


def trajectory(eps, T, burn, rng):
    """Exact path of a single agent with constant switching rates: list of (t, s, behaviour) segments."""
    t, s, state, segs = -burn, s_mean, 'H', []
    while t < T:
        rate = k_HA if state == 'H' else k_AH
        tau = rng.exponential(1 / rate)
        target = 1.0 if state == 'H' else 0.0
        tt = np.linspace(0, tau, max(2, int(tau * 40)))
        ss = target + (s - target) * np.exp(-eps * tt)
        if t + tau > 0:
            keep = t + tt >= 0
            segs.append((t + tt[keep], ss[keep], state))
        t, s = t + tau, ss[-1]
        state = 'A' if state == 'H' else 'H'
    return segs


fig, ax = plt.subplots(2, 1, figsize=(3.7, 2.9), sharex=True)
for a, rho, seed, lab in [(ax[0], 0.1, 4, '(b) frequent switching, $\\rho=0.1$'),
                          (ax[1], 3.0, 2, '(c) persistent behaviour, $\\rho=3$')]:
    rng = np.random.default_rng(seed)
    for tt, ss, state in trajectory(rho * K, T, burn=300, rng=rng):
        a.plot(tt, ss, color=AQUA if state == 'H' else ORANGE, lw=1.6, solid_capstyle='butt')
    a.axhline(s_mean, color=INK2, lw=0.9, ls='--')
    a.set_xlim(0, T); a.set_ylim(-0.03, 1.03)
    a.set_yticks([0, 0.5, 1])
    a.set_ylabel('independent\nskill $s$')
    a.set_title(lab, loc='left')
ax[1].set_xlabel('time (units of $1/\\nu$)')
ax[0].legend(handles=[Line2D([], [], color=AQUA, lw=1.6, label='non-user'),
                      Line2D([], [], color=ORANGE, lw=1.6, label='AI user'),
                      Line2D([], [], color=INK2, lw=0.9, ls='--', label='mean skill')],
             loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=7, handlelength=1.6,
             columnspacing=1.0)
fig.tight_layout(h_pad=0.4)
fig.savefig('figures/fig_trajectories.pdf'); fig.savefig('figures/fig_trajectories.png')
