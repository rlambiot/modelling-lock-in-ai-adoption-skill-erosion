"""Numbers quoted in the text that the figure scripts do not print (theory only, a few minutes)."""
import numpy as np, warnings
from coevo import *
warnings.filterwarnings('ignore')

base = Params(c=0.15, mu=0.02, beta=0.8)
xs = np.linspace(0.003, 0.997, 300)

# Sec. II: the imitation probability stays within [0.04, 0.96] in all the states considered
import json
cases = [('ramps of Fig. 4: b = 1.15 at x -> 0', 1.15, 0.0)]          # (label, b - c x, eta)
for eps in [0.02, 0.2, 2.0]:
    b, mH, mA = exact_branch(Params(c=0.95, eps=eps), np.array([0.4]))[0]
    cases.append((f'Fig. 2, rho = {float(rho_of(eps, base)):.2g}', b - 0.95 * 0.4, 0.0))
for eta in [0.2, 0.4]:                                               # complementarity: the bistable window
    br = exact_branch(base.with_(eps=0.2, eta=eta), xs)
    fd, fu = branch_folds(xs, br[:, 0])
    sel = np.isfinite(br[:, 0]) & (br[:, 0] >= fd - 0.05) & (br[:, 0] <= fu + 0.05)
    cases += [(f'complementarity, eta = {eta}', br[k, 0] - base.c * xs[k], eta) for k in np.nonzero(sel)[0]]
    b, mH, mA = exact_branch(Params(c=0.95, eps=0.2, eta=eta), np.array([0.4]))[0]
    cases.append((f'complementarity, c = 0.95, eta = {eta}', b - 0.95 * 0.4, eta))
R = json.load(open('figures/robustness.json'))                       # linear imitation, unequal rates, c = 1.2
for rr, d in R['d'].items():
    cases += [(f'unequal rates, r = {rr}', b - 1.2 * 0.4, 0.0) for b in np.array(d['th'])[:, 0] if np.isfinite(b)]
worst = max(cases, key=lambda c: max(c[1] + c[2], 1 - c[1]))
dpi = max(worst[1] + worst[2], 1 - worst[1])                       # largest |pi_j - pi_i| over all skills
print(f'Sec. II: largest |pi_j - pi_i| = {dpi:.3f} ({worst[0]}), so beta |pi_j - pi_i| <= {base.beta * dpi:.3f}')

# Sec. IV: lone holdouts for mu -> 0 (Appendix B)
for rho in [0.39, 0.40]:
    th = holdout_threshold(base.with_(mu=0.0, eps=rho / 2))      # for mu = 0, rho = 2 eps / nu
    print(f'Sec. IV: rho = {rho}: theta* = {th:.4f} (beta = 0.8), {rho / (1 + rho):.4f} (weak selection)')

# Sec. IV: folds for mu = 0.02 against rho
for rho in [0.04, 2.0]:
    f = branch_folds(xs, exact_branch(base.with_(eps=rho * (1 + base.mu) / 2), xs)[:, 0])
    print(f'Sec. IV: rho = {rho}: threshold of return {f[0]:.3f}, threshold of adoption {f[1]:.3f}')

# Appendix B: for mu = 0, b(x) decreases monotonically from 1 to c + theta*
xm = np.linspace(0.002, 0.998, 400)
ok = []
for rho in [0.03, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    p = base.with_(mu=0.0, eps=rho / 2)
    bb = exact_branch(p, xm)[:, 0]
    mono = np.all(np.isfinite(bb)) and np.all(np.diff(bb) < 0)
    ok.append(mono)
    print(f'App. B: rho = {rho:4}: b(x) from {bb[0]:.4f} to {bb[-1]:.4f} (c + theta* = {p.c + holdout_threshold(p):.4f}), '
          f'monotonic: {mono}')
print(f'App. B: monotonic for all {len(ok)} values of rho: {all(ok)}')

# Sec. V: unstable stationary state at b = 0.7, rho = 0.39
br = exact_branch(base.with_(eps=0.2), xs)
mid = np.gradient(br[:, 0]) < 0
k = np.argmin(np.where(mid, np.abs(br[:, 0] - 0.7), np.inf))
print(f'Sec. V: unstable state at b = 0.7: x = {xs[k]:.3f}, m_H = {br[k, 1]:.3f}, m_A = {br[k, 2]:.3f}')

# Appendix B: complementarity
for eta in [0.0, 0.2, 0.4]:
    f = branch_folds(xs, exact_branch(base.with_(eps=0.2, eta=eta), xs)[:, 0])
    b, mH, mA = exact_branch(Params(c=0.95, eps=0.2, eta=eta), np.array([0.4]))[0]
    print(f'App. B: eta = {eta}: bistable window {f[1] - f[0]:.3f} (c = 0.15); skill gap at x = 0.4 {mH - mA:.3f} (c = 0.95)')
