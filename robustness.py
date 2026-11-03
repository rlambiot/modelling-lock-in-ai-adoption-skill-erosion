"""Robustness checks of Appendix C: the Fermi rule, unequal rates of learning and erosion, and
imitation without noise. Prints the numbers quoted there.

Simulations are cached in figures/robustness.json and figures/abm_hard.json; delete them to rerun
the simulations (about an hour in total). Theory is recomputed, which takes a few minutes.
"""
import numpy as np, json, os, warnings
from scipy import optimize
from coevo import *
warnings.filterwarnings('ignore')

XS = np.linspace(0.004, 0.996, 220)
rng = np.random.default_rng(17)


# ---- theory -----------------------------------------------------------------
def linear_solve(x, p, guess):
    """Self-consistent state at fixed x for linear imitation, by quadrature (allows eps_A != eps)."""
    def F(v):
        b, mH, mA = v
        q = p.with_(b=b)
        a0, a1, g0, g1 = rate_coeffs(x, mH, mA, q)
        st = pdmp_general(lambda s: a0 - a1 * s, lambda s: g0 - g1 * s, q, n_grid=201)
        return [st['x'] - x, st['mH'] - mH, st['mA'] - mA]
    sol = optimize.fsolve(F, guess, xtol=1e-11)
    if np.max(np.abs(F(sol))) > 1e-8:
        return None, None
    q = p.with_(b=sol[0])
    a0, a1, g0, g1 = rate_coeffs(x, sol[1], sol[2], q)
    return sol, pdmp_general(lambda s: a0 - a1 * s, lambda s: g0 - g1 * s, q, n_grid=201)


def linear_branch(p):
    out, guess = [], np.array([p.b, 1 - XS[0] + 0.01, 1 - XS[0] - 0.01])
    for x in XS:
        sol, _ = linear_solve(x, p, guess)
        out.append(sol[0] if sol is not None else np.nan)
        if sol is not None:
            guess = sol
    return np.array(out)


def window(bx):
    """Folds (b_down, b_up) of a branch b(x); the solver can fail at the extreme ends of the grid."""
    ok = np.isfinite(bx)
    f = branch_folds(XS[ok], bx[ok])
    return f if f else (np.nan, np.nan)


def solve_at(x, p, rule):
    """Exact state at adoption x: returns b, Var(s)/[x(1-x)] and the skill gap m_H - m_A."""
    if rule == 'fermi':
        sol, st = fermi_solve(x, p, np.array([1.0, p.nu * 0.3]))
    else:
        sol, st = linear_solve(x, p, np.array([1.0, 0.7, 0.4]))
    if sol is None:
        return np.nan, np.nan, np.nan
    return sol[0], st['var'] / (x * (1 - x)), st['mH'] - st['mA']


# ---- simulations ----------------------------------------------------------------
def fold_scan(p, bd, bu, N=10000, T=400, half=0.08):
    """Folds measured by runs at fixed b, with resolution 0.01, around the predicted folds bd, bu."""
    res = {}
    for key, bc, x0, s0 in [('up', bu, 0.02, 1.0), ('down', bd, 0.98, 0.0)]:
        grid = np.round(np.arange(bc - half, bc + half + 1e-9, 0.01), 3)
        xs_ = []
        for b in grid:
            r = simulate(p.with_(b=b), N=N, T=T, dt=0.05, x0=x0, s0=np.full(N, s0), rng=rng, record_dt=10)
            xs_.append(float(r['x'][-5:].mean()))
        res[key] = (grid, np.array(xs_))
    g, x_ = res['up']; i = np.argmax(x_ > 0.5)
    b_up = float(g[i] - 0.005) if x_[i] > 0.5 else np.nan
    g, x_ = res['down']; j = np.nonzero(x_ < 0.5)[0]
    b_dn = float(g[j[-1]] + 0.005) if len(j) else np.nan
    return b_dn, b_up


def strat_abm(p, b, N=20000):
    """Simulated adoption, Var(s)/[x(1-x)] and skill gap, averaged over five snapshots."""
    q = p.with_(b=b)
    emin, emax = min(p.eps, p.eA), max(p.eps, p.eA)
    dt = min(0.02, 0.2 / emax)
    r = simulate(q, N=N, T=max(150, 15 / emin), dt=dt, x0=0.4, s0=np.full(N, 0.6), rng=rng, record_dt=50)
    sig, s = r['sigma'], r['s']
    out = []
    for _ in range(5):
        r = simulate(q, N=N, T=20, dt=dt, sigma0=sig, s0=s, rng=rng, record_dt=20)
        sig, s = r['sigma'], r['s']
        x = (sig == 1).mean()
        out.append([x, s.var() / (x * (1 - x)), s[sig == 0].mean() - s[sig == 1].mean()])
    return np.mean(out, axis=0).tolist()


cache = 'figures/robustness.json'
R = json.load(open(cache)) if os.path.exists(cache) else {}

# (a) Fermi rule, gamma = 4 (beta in the code), c = 0.15: bistable window against eps
pa = Params(c=0.15, mu=0.02, beta=4.0, rule='fermi', b=0.8)
eps_a = np.linspace(0.02, 1.6, 40)
if 'a' not in R:
    th = [window(fermi_branch(pa.with_(eps=e), XS)[0][:, 0]) for e in eps_a]
    abm = {}
    for e in [0.1, 0.3, 0.6]:
        bd, bu = window(fermi_branch(pa.with_(eps=e), XS)[0][:, 0])
        abm[e] = fold_scan(pa.with_(eps=e), bd, bu)
    R['a'] = dict(eps=eps_a.tolist(), th=np.array(th).tolist(), abm={str(k): v for k, v in abm.items()})
    json.dump(R, open(cache, 'w'))

# (b) Fermi rule, c = 1.2: skill gap at x = 0.4 against eps, for three strengths of selection
pb = Params(c=1.2, mu=0.02, rule='fermi')
eps_b = np.logspace(-2, 1, 36)
betas = [1.6, 4.0, 10.0]
if 'b' not in R:
    R['b'] = {}
    for bF in betas:
        th = [solve_at(0.4, pb.with_(beta=bF, eps=e), 'fermi') for e in eps_b]
        pts = []
        for e in [0.03, 0.1, 0.3, 1.0, 3.0]:
            b, _, _ = solve_at(0.4, pb.with_(beta=bF, eps=e), 'fermi')
            pts.append([e] + strat_abm(pb.with_(beta=bF, eps=e), b))
        R['b'][str(bF)] = dict(th=np.array(th).tolist(), abm=pts)
    json.dump(R, open(cache, 'w'))

# (c) unequal rates, eps_A = r eps, c = 0.15: bistable window against eps
pc = Params(c=0.15, mu=0.02, beta=0.8, b=0.8)
ratios = [0.5, 1.0, 2.0]
eps_c = np.linspace(0.02, 1.6, 34)
if 'c' not in R:
    R['c'] = {}
    for rr in ratios:
        th = [window(linear_branch(pc.with_(eps=e, eps_A=rr * e))) for e in eps_c]
        abm = []
        if rr != 1.0:
            for e in [0.2, 0.5]:
                bd, bu = window(linear_branch(pc.with_(eps=e, eps_A=rr * e)))
                abm.append([e, *fold_scan(pc.with_(eps=e, eps_A=rr * e), bd, bu)])
        R['c'][str(rr)] = dict(th=np.array(th).tolist(), abm=abm)
    json.dump(R, open(cache, 'w'))

# (d) unequal rates, c = 1.2: skill gap at x = 0.4 against eps
pd_ = Params(c=1.2, mu=0.02, beta=0.8)
if 'd' not in R:
    R['d'] = {}
    for rr in ratios:
        th = [solve_at(0.4, pd_.with_(eps=e, eps_A=rr * e), 'linear') for e in eps_b]
        pts = []
        for e in [0.03, 0.1, 0.3, 1.0, 3.0]:
            b, _, _ = solve_at(0.4, pd_.with_(eps=e, eps_A=rr * e), 'linear')
            pts.append([e] + strat_abm(pd_.with_(eps=e, eps_A=rr * e), b))
        R['d'][str(rr)] = dict(th=np.array(th).tolist(), abm=pts)
    json.dump(R, open(cache, 'w'))

# exact windows at the values of eps of the simulations (computed once and cached)
if 'th_at_abm' not in R:
    R['th_at_abm'] = dict(
        a={str(e): list(window(fermi_branch(pa.with_(eps=e), XS)[0][:, 0])) for e in [0.1, 0.3, 0.6]},
        c={str(rr): {str(e): list(window(linear_branch(pc.with_(eps=e, eps_A=rr * e)))) for e in [0.2, 0.5]}
           for rr in [0.5, 2.0]})
    json.dump(R, open(cache, 'w'))

rho = lambda e: float(rho_of(e, pa))
print('Fermi rule, gamma = 4, c = 0.15: bistable window')
th = np.array(R['a']['th'])
for r0 in [0.2, 1.2]:                       # interpolated between the computed values of eps
    e0 = r0 * pa.nu * (1 + pa.mu) / 2
    lo, hi = np.interp(e0, eps_a, th[:, 0]), np.interp(e0, eps_a, th[:, 1])
    print(f'  rho = {r0}: {lo:.3f} < b < {hi:.3f}, width {hi - lo:.3f}')
dev = []
for e, (bd, bu) in R['a']['abm'].items():
    td, tu = R['th_at_abm']['a'][e]
    dev += [abs(bd - td), abs(bu - tu)]
    print(f'  rho = {rho(float(e)):.2f}: exact {td:.3f}, {tu:.3f}; simulations {bd:.3f}, {bu:.3f}')
print(f'  largest difference: {max(dev):.3f}')
print('Fermi rule, c = 1.2: skill gap at x = 0.4 (exact; simulations)')
for bF in betas:
    pts = np.array(R['b'][str(bF)]['abm'])
    gaps = np.interp(np.log(pts[:, 0]), np.log(eps_b), np.array(R['b'][str(bF)]['th'])[:, 2])
    print(f'  gamma = {bF:g}: ' + ', '.join(f'rho = {rho(e):.2g}: {g:.3f}; {a:.3f}' for e, g, a in zip(pts[:, 0], gaps, pts[:, 3])))

print('Unequal rates eps_A = r eps, c = 0.15')
i = np.argmin(abs(eps_c - 0.2))
w_eq = branch_folds(np.linspace(0.003, 0.997, 300), exact_branch(pc.with_(eps=0.2), np.linspace(0.003, 0.997, 300))[:, 0])
print(f'  eps = 0.2, r = 1: window width {w_eq[1] - w_eq[0]:.3f}')
dev = []
for rr in ['0.5', '2.0']:
    td, tu = R['th_at_abm']['c'][rr]['0.2']
    print(f'  eps = 0.2, r = {rr}: window width {tu - td:.3f}')
    for e, bd, bu in R['c'][rr]['abm']:
        td, tu = R['th_at_abm']['c'][rr][str(e)]
        dev += [abs(bd - td), abs(bu - tu)]
        print(f'    eps = {e}: exact {td:.3f}, {tu:.3f}; simulations {bd:.3f}, {bu:.3f}')
print(f'  largest difference between exact and simulated folds: {max(dev):.3f}')
print('Unequal rates, c = 1.2: skill gap at x = 0.4 (exact; simulations)')
for rr in ['0.5', '2.0']:
    pts = np.array(R['d'][rr]['abm'])
    gaps = np.interp(np.log(pts[:, 0]), np.log(eps_b), np.array(R['d'][rr]['th'])[:, 2])
    print(f'  r = {rr}: ' + ', '.join(f'eps = {e:g}: {g:.3f}; {a:.3f}' for e, g, a in zip(pts[:, 0], gaps, pts[:, 3])))

# ---- imitation without noise: agents copy a peer only if the peer earns more
hb = Params(rule='hard', mu=0.02)
eps_of = lambda r: r * hb.nu * (1 + hb.mu) / 2
xh = np.linspace(0.0105, 0.9895, 300)      # the interior branch exists for F* in (0, 1)
print('Imitation without noise, c = 0.15, rho = 0.39')
f = branch_folds(xh, hard_branch(hb.with_(c=0.15, eps=eps_of(0.39)), xh))
print(f'  {f[0]:.3f} < b < {f[1]:.3f}, width {f[1] - f[0]:.3f} (linear imitation: {w_eq[1] - w_eq[0]:.3f})')
print('Imitation without noise, c = 0.95, x = 0.4')
theory = {}
for r in [0.039, 0.39, 3.9]:
    st, = hard_solve(0.4, hb.with_(c=0.95, eps=eps_of(r)))
    theory[r] = st
    print(f'  rho = {r:5.3f}: m_H = {st["mH"]:.3f}, m_A = {st["mA"]:.3f}, gap = {st["mH"] - st["mA"]:.3f}, '
          f'P(s < 0.2) = {st["below"](0.2):.3f}, fraction of non-users who earn more than AI users = {1 - st["FH"]:.3f}')
hcache = 'figures/abm_hard.json'
if os.path.exists(hcache):
    H = json.load(open(hcache))
else:
    hrng = np.random.default_rng(1)
    st = theory[0.39]
    p = hb.with_(c=0.95, eps=eps_of(0.39), b=st['b'])
    r = simulate(p, N=20000, T=300, dt=0.05, x0=0.4, rng=hrng)
    snaps = []
    for k in range(40):
        r = simulate(p, N=20000, T=5, dt=0.05, sigma0=r['sigma'], s0=r['s'], rng=hrng)
        A, s = r['sigma'] == 1, r['s']
        snaps.append([A.mean(), s[~A].mean(), s[A].mean(), s.var(), (s < 0.2).mean()])
    x, mH, mA, v, low = np.mean(snaps, 0)
    H = dict(dist=dict(x=x, mH=mH, mA=mA, Sigma=v / (x * (1 - x)), low=low, ret=mH - (p.b - p.c * x)))
    H['folds'] = []
    for b, x0 in [(0.79, 0.95), (0.84, 0.95), (1.00, 0.02), (1.04, 0.02)]:
        q = hb.with_(c=0.15, eps=eps_of(0.39), b=b)
        r = simulate(q, N=10000, T=2000, dt=0.05, x0=x0, rng=hrng)   # long runs: the population lingers near a fold
        H['folds'].append(dict(b=b, x0=x0, x=float(r['x'][-200:].mean())))
    json.dump(H, open(hcache, 'w'), indent=1)
d = H['dist']
print(f'  simulations, rho = 0.39: x = {d["x"]:.3f}, m_H = {d["mH"]:.3f}, m_A = {d["mA"]:.3f}, P(s < 0.2) = {d["low"]:.3f}')
for f in H['folds']:
    print(f'  c = 0.15, b = {f["b"]:.2f}, started at x = {f["x0"]}: stationary x = {f["x"]:.3f}')
