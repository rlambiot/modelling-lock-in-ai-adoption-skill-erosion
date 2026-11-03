"""Fig. 4: abrupt adoption and lock-in (Sec. V), c = 0.15 and rho = 0.39, simulations of 10^4 agents.

(a) b rises linearly from 0.3 to 1.15 at a rate r (over a time T_r = 0.85/r), stays at 1.15 for 300
    time units, falls back at the same rate and stays at 0.3, for r = 3e-4, 1e-3, 3e-3 and 1e-2.
(b) b is raised from 0.7 to 1 for a duration D and then restored to 0.7. The inset gives the fraction
    of 20 runs per duration that remain at high adoption, with binomial standard errors.
Simulations are cached in figures/abm_ramp.json; delete it to rerun them (about 10 minutes).
Prints the numbers quoted in Sec. V.
"""
import os, json, warnings
import numpy as np
import matplotlib.pyplot as plt
from coevo import *
from style import *
warnings.filterwarnings('ignore')

base = Params(c=0.15, mu=0.02, beta=0.8, eps=0.2)
N, DT = 10000, 0.05
B_LO, B_HI, HOLD, T_PRE = 0.3, 1.15, 300.0, 300.0
T_RAMPS = [2833, 850, 283, 85]
RATES = {2833: '3\\times10^{-4}', 850: '10^{-3}', 283: '3\\times10^{-3}', 85: '10^{-2}'}   # r = 0.85/T_r
DS = [10, 20, 28, 32, 34, 36, 40, 60, 80, 160]
DS_STATS = [26, 28, 30, 32, 33, 34, 35, 36, 37, 38]   # 20 runs each (seeds 300-319) for the inset
T_AFTER = 400.0
LOW = 0.2                      # agents with skill below LOW count as deskilled
cache = 'figures/abm_ramp.json'


def run_segments(segments, b0, seed):
    """Chain simulate() over segments (duration, b_of_t), after an equilibration at constant b0."""
    rng = np.random.default_rng(seed)
    r = simulate(base.with_(b=b0), N=N, T=T_PRE, dt=DT, x0=0.02, s0=np.ones(N), rng=rng, record_dt=T_PRE)
    sigma, s = r['sigma'], r['s']
    t0, ts, bs, xs_, lows = 0.0, [], [], [], []
    for T, f in segments:
        rec = min(2.0, max(0.25, T / 1000))
        r = simulate(base, N=N, T=T, dt=DT, sigma0=sigma, s0=s, rng=rng, record_dt=rec, b_of_t=f)
        ts += list(t0 + r['t']); bs += [f(t) for t in r['t']]; xs_ += list(r['x'])
        sigma, s = r['sigma'], r['s']
        lows.append(float(np.mean(s < LOW)))
        t0 += T
    return dict(t=ts, b=bs, x=xs_, low_end=lows, x_end=float(sigma.mean()))


def ramp_run(T_r, seed):
    up = lambda t: B_LO + (B_HI - B_LO) * t / T_r
    down = lambda t: B_HI - (B_HI - B_LO) * t / T_r
    segs = [(T_r, up), (HOLD, lambda t: B_HI), (T_r, down), (HOLD, lambda t: B_LO)]
    o = run_segments(segs, B_LO, seed)
    t, b, x = map(np.array, (o['t'], o['b'], o['x']))
    first_up = t < T_r + HOLD
    ku, kd = np.nonzero(x[first_up] > 0.5)[0], np.nonzero(x[~first_up] < 0.5)[0]
    o['b_up'] = float(b[first_up][ku[0]]) if len(ku) else np.nan
    o['b_down'] = float(b[~first_up][kd[0]]) if len(kd) else np.nan
    o['low_top'] = o['low_end'][1]          # fraction deskilled at the end of the high plateau
    return o


def boost_run(D, seed):
    return run_segments([(D, lambda t: 1.0), (T_AFTER, lambda t: 0.7)], 0.7, seed)


def boost_final(D, seed):
    """State at the end of a boost of duration D (adoption, mean skills of non-users and AI users),
    and adoption T_AFTER later."""
    rng = np.random.default_rng(seed)
    r = simulate(base.with_(b=0.7), N=N, T=T_PRE, dt=DT, x0=0.02, s0=np.ones(N), rng=rng, record_dt=T_PRE)
    r = simulate(base.with_(b=1.0), N=N, T=D, dt=DT, sigma0=r['sigma'], s0=r['s'], rng=rng, record_dt=D)
    sg, s = r['sigma'], r['s']
    mid = dict(x=float(sg.mean()), mH=float(s[sg == 0].mean()), mA=float(s[sg == 1].mean()))
    r = simulate(base.with_(b=0.7), N=N, T=T_AFTER, dt=DT, sigma0=sg, s0=s, rng=rng, record_dt=T_AFTER)
    return dict(mid=mid, x_final=float(r['sigma'].mean()))


if os.path.exists(cache):
    abm = json.load(open(cache))
else:
    abm = {'ramp': {}, 'boost': {}}
    for T_r in T_RAMPS:
        for seed in range(3):
            abm['ramp'][f'{T_r}_{seed}'] = ramp_run(T_r, seed)
    for D in DS:
        for seed in range(5):
            abm['boost'][f'{D}_{seed}'] = boost_run(D, 100 + seed)
    json.dump(abm, open(cache, 'w'))
if 'stats' not in abm:
    abm['stats'] = {f'{D}_{seed}': boost_final(D, seed) for D in DS_STATS for seed in range(300, 320)}
    json.dump(abm, open(cache, 'w'))

# ---- summary
xs = np.linspace(0.003, 0.997, 300)
bb = exact_branch(base, xs)[:, 0]
fd, fu = branch_folds(xs, bb)
lo0 = base.c + holdout_threshold(base)
print(f'exact folds {fd:.3f}, {fu:.3f}; thresholds for mu -> 0: {lo0:.3f} and 1')
for T_r in T_RAMPS:
    R = [abm['ramp'][f'{T_r}_{s}'] for s in range(3)]
    bu, bd = np.array([r['b_up'] for r in R]), np.array([r['b_down'] for r in R])
    print(f'T_r={T_r:5d}: up {bu.mean():.3f}+-{bu.std():.3f}, down {bd.mean():.3f}+-{bd.std():.3f}, '
          f'deskilled at the top {np.mean([r["low_top"] for r in R]):.3f}')
locked = {}
for D in DS:
    B = [abm['boost'][f'{D}_{s}'] for s in range(5)]
    locked[D] = np.mean([b['x_end'] > 0.5 for b in B])
    print(f'D={D:3d}: locked in {locked[D]:.1f}, final x {np.mean([b["x_end"] for b in B]):.3f}, '
          f'deskilled at the end {np.mean([b["low_end"][1] for b in B]):.3f}')
p_lock = [np.mean([abm['stats'][f'{D}_{s}']['x_final'] > 0.5 for s in range(300, 320)]) for D in DS_STATS]
print('fraction of 20 runs locked in:', dict(zip(DS_STATS, p_lock)))
R34 = [abm['stats'][f'34_{s}'] for s in range(300, 320)]
x34 = np.array([r['mid']['x'] for r in R34]); mA34 = np.array([r['mid']['mA'] for r in R34])
mH34 = np.array([r['mid']['mH'] for r in R34]); lk = np.array([r['x_final'] > 0.5 for r in R34])
# unstable stationary state at b = 0.7: the middle branch of the exact solution
br = exact_branch(base, xs)
mid = np.gradient(br[:, 0]) < 0
k = np.argmin(np.where(mid, np.abs(br[:, 0] - 0.7), np.inf))
xu, mHu, mAu = xs[k], br[k, 1], br[k, 2]
ret = (x34 > xu) & ~lk
print(f'unstable state at b = 0.7: x = {xu:.3f}, m_H = {mHu:.3f}, m_A = {mAu:.3f}')
print(f'D = 34, end of the increase: {np.sum(x34 > xu)} of 20 runs above x = {xu:.3f}, of which {ret.sum()} return; '
      f'mean skills of those {ret.sum()}: {mH34[ret].mean():.3f} and {mA34[ret].mean():.3f}; '
      f'of the {lk.sum()} locked in: {mH34[lk].mean():.3f} and {mA34[lk].mean():.3f}')
sel = (x34 > 0.63) & (x34 < 0.66)
print(f'  runs ending with 0.63 < x < 0.66: {sel.sum()}, of which {np.sum(sel & lk)} remain locked in')

# ---- figure
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.5))
a = ax[0]
stable = np.gradient(bb) > 0
for mask, ls in [(stable, '-'), (~stable, '--')]:
    a.plot(bb, np.where(mask, xs, np.nan), ls, color=GREY, lw=1.3)
for T_r, col in zip(T_RAMPS, SEQ):
    r = abm['ramp'][f'{T_r}_0']
    a.plot(r['b'], r['x'], color=col, lw=1.1, label=f'${RATES[T_r]}$')
for v in (lo0, 1.0):
    a.vlines(v, -0.03, 1.03, color=GREY, lw=0.7, ls=':')
a.text(lo0, 1.05, r'$c+\theta^*$', ha='center', fontsize=6.5, color=INK2)
a.text(1.0, 1.05, r'$b=1$', ha='center', fontsize=6.5, color=INK2)
a.set_xlim(B_LO, B_HI); a.set_ylim(-0.03, 1.38); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a.set_xlabel('AI benefit $b$'); a.set_ylabel('adoption $x$')
a.set_title('(a) ramps of $b$ up and down', loc='left')
a.legend(title=r'ramp rate $r$', fontsize=6.3, title_fontsize=6.3, loc='upper center', ncol=4,
         handlelength=1.0, columnspacing=0.7, borderaxespad=0.2)
a.set_ylim(-0.03, 1.42)

a = ax[1]
shown = [20, 34, 36, 80]
for D, col in zip(shown, SEQ):
    r = abm['boost'][f'{D}_0']
    t, x = np.array(r['t']), np.array(r['x'])
    a.plot(t, x, color=col, lw=1.1, label=f'{D}')
    k = np.argmin(np.abs(t - D))
    a.plot(t[k], x[k], 'o', color=col, ms=3.5, mec='white', mew=0.6)
a.set_xlim(0, 250); a.set_ylim(-0.03, 1.03)
a.set_xlabel(r'time (units of $1/\nu$)'); a.set_ylabel('adoption $x$')
a.set_title(r'(b) boosts of $b$ from $0.7$ to $1$', loc='left')
a.legend(title='boost duration $D$', fontsize=6.3, title_fontsize=6.3, loc='center right', handlelength=1.2)
ins = a.inset_axes([0.26, 0.2, 0.25, 0.18])
se = np.sqrt(np.array(p_lock) * (1 - np.array(p_lock)) / 20)      # binomial standard error
ins.errorbar(DS_STATS, p_lock, yerr=se, fmt='o-', color=BLUE, ms=2.2, lw=0.9, elinewidth=0.6, capsize=0)
ins.set_xticks([26, 32, 38]); ins.set_yticks([0, 1]); ins.set_ylim(-0.08, 1.08)
ins.tick_params(labelsize=5.5, length=2, pad=1)
ins.set_title('locked in', fontsize=5.8, pad=2)
ins.set_xlabel('$D$', fontsize=5.8, labelpad=0)

fig.tight_layout(w_pad=0.8)
fig.savefig('figures/fig_letter_ramp.pdf'); fig.savefig('figures/fig_letter_ramp.png')
