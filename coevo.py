"""
Coevolution of AI adoption and independent skill: exact stationary solution and agent-based model.

Code accompanying R. Lambiotte, "Modelling lock-in of AI adoption through skill erosion".

Model
-----
Agent i has a behaviour sigma_i in {0 = H (works without AI), 1 = A (uses AI)} and an
independent skill s_i in [0, 1]; x is the fraction of AI users and time is measured in units of 1/nu.
  payoffs   pi_H = s_i,   pi_A = b - c x + eta s_i          (eta = 0 except for complementarity)
  skills    ds/dt = eps (1 - s) if H,   ds/dt = -eps_A s if A    (eps_A = eps unless set)
  revision  at rate nu: with probability mu, choose H or A at random; otherwise observe a random
            peer j and, if the behaviours differ, copy j with probability
              1/2 + (beta/2)(pi_j - pi_i)          rule='linear' (the article's main rule)
              1/(1 + exp(-beta (pi_j - pi_i)))     rule='fermi'  (beta is gamma in the article)
              1 if pi_j > pi_i, 0 otherwise         rule='hard'   (imitation without noise)

Theory
------
For linear imitation, the switching rates are linear in skill, k_HA(s) = a0 - a1 s and
k_AH(s) = g0 - g1 s. Zero flux gives the stationary densities
  n_H(s) = C (1-s)^(A-1) s^B e^(-lam s),   n_A(s) = C (1-s)^A s^(B-1) e^(-lam s),
with A = k_HA(1)/eps, B = k_AH(0)/eps_A and lam = a1/eps + g1/eps_A, whose self-consistency
gives the stationary states (exact_branch). pdmp_general solves the same zero-flux equation
by quadrature for arbitrary rates, e.g. the Fermi rule.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy import integrate, optimize
from scipy.special import expit, betaln, betainc


@dataclass
class Params:
    b: float = 0.6       # benefit of AI-assisted work
    c: float = 0.15      # crowding: loss of value of AI-assisted work per unit of adoption
    eta: float = 0.0     # complementarity: weight of the user's own skill in AI-assisted work
    nu: float = 1.0      # rate of behaviour revision (sets the time unit)
    eps: float = 0.2     # rate of skill adaptation (learning, and erosion unless eps_A is set)
    mu: float = 0.02     # exploration probability
    beta: float = 0.8    # strength of selection
    eps_A: float = None  # erosion rate under AI use (None: equal to eps)
    rule: str = 'linear' # imitation rule: 'linear', 'fermi' or 'hard'

    def with_(self, **kw):
        return replace(self, **kw)

    @property
    def eA(self):
        return self.eps if self.eps_A is None else self.eps_A


def rho_of(eps, p):
    """Ratio of timescales rho = 2 eps / [nu (1 + mu)] (Eq. 4 of the article)."""
    return np.asarray(eps) / (p.nu * (1 + p.mu) / 2)


# ---------------------------------------------------------------------------
# Exact stationary solution for linear imitation
# ---------------------------------------------------------------------------
def rate_coeffs(x, mH, mA, p):
    """Linear-in-s switching rates k_HA(s) = a0 - a1 s and k_AH(s) = g0 - g1 s (Eq. A1)."""
    PA = p.b - p.c * x + p.eta * mA
    a1 = p.nu * (1 - p.mu) * x * p.beta / 2
    a0 = p.nu * (p.mu / 2 + (1 - p.mu) * x * (0.5 + p.beta / 2 * PA))
    g1 = p.nu * (1 - p.mu) * (1 - x) * p.beta * p.eta / 2
    g0 = p.nu * (p.mu / 2 + (1 - p.mu) * (1 - x) * (0.5 + p.beta / 2 * (mH - p.b + p.c * x)))
    return a0, a1, g0, g1


def pdmp_moments(coeffs, p):
    """Adoption x, mean skills m_H and m_A, and the exponents (A, B, lam) of the stationary density."""
    a0, a1, g0, g1 = coeffs
    A = max((a0 - a1) / p.eps, 1e-9)       # guard: rates must stay positive
    B = max(g0 / p.eA, 1e-9)
    lam = a1 / p.eps + g1 / p.eA

    def f(u, moment):
        return np.exp(-lam * u) * (u ** moment)
    # n_H ∝ (1-s)^(A-1) s^B,  n_A ∝ (1-s)^A s^(B-1); the 'alg' weight handles the endpoint powers
    IH0 = integrate.quad(f, 0, 1, args=(0,), weight='alg', wvar=(B, A - 1), limit=200)[0]
    IA0 = integrate.quad(f, 0, 1, args=(0,), weight='alg', wvar=(B - 1, A), limit=200)[0]
    IH1 = integrate.quad(f, 0, 1, args=(1,), weight='alg', wvar=(B, A - 1), limit=200)[0]
    IA1 = integrate.quad(f, 0, 1, args=(1,), weight='alg', wvar=(B - 1, A), limit=200)[0]
    IH0, IH1 = IH0 / p.eps, IH1 / p.eps
    IA0, IA1 = IA0 / p.eA, IA1 / p.eA
    Z = IH0 + IA0
    return IA0 / Z, IH1 / IH0, IA1 / IA0, (A, B, lam)


def exact_branch(p, xs):
    """Stationary states parametrised by adoption: returns b(x), m_H(x), m_A(x) for each x in xs.

    Solves the self-consistency conditions for (b, m_H, m_A) at fixed x, so that the unstable
    (middle) branch is obtained as well. The folds are the extrema of b(x) (branch_folds).
    """
    out = []
    guess = np.array([p.b, 1 - xs[0] + 0.1, 1 - xs[0] - 0.1])
    for x in xs:
        def F(v):
            b, mH, mA = v
            q = p.with_(b=b)
            xn, mHn, mAn, _ = pdmp_moments(rate_coeffs(x, mH, mA, q), q)
            return [xn - x, mHn - mH, mAn - mA]
        sol = optimize.fsolve(F, guess, xtol=1e-11)
        ok = np.max(np.abs(F(sol))) < 1e-8
        out.append(sol if ok else [np.nan] * 3)
        if ok:
            guess = sol
    return np.array(out)


def branch_folds(xs, bs):
    """Lower and upper folds (b_down, b_up) of an S-shaped curve b(x); None if b(x) is monotonic."""
    d = np.diff(bs)
    idx = np.nonzero(np.sign(d[1:]) != np.sign(d[:-1]))[0] + 1
    if len(idx) < 2:
        return None
    return bs[idx[1]], bs[idx[0]]


def pdmp_stats(coeffs, p, theta=None):
    """Normalised statistics of the stationary density: x, mean, var, m_H, m_A, the normalisation
    (for pdmp_pdf) and, if theta is given, the fraction P_below of agents with skill below theta."""
    a0, a1, g0, g1 = coeffs
    eH, eA = p.eps, p.eA
    A = max((a0 - a1) / eH, 1e-9)
    B = max(g0 / eA, 1e-9)
    lam = a1 / eH + g1 / eA

    def I(moment, wv):
        return integrate.quad(lambda s: np.exp(-lam * s) * s ** moment, 0, 1.0,
                              weight='alg', wvar=wv, limit=400)[0]
    wH, wA = (B, A - 1), (B - 1, A)
    H0, A0 = I(0, wH) / eH, I(0, wA) / eA
    H1, A1 = I(1, wH) / eH, I(1, wA) / eA
    H2, A2 = I(2, wH) / eH, I(2, wA) / eA
    Z = H0 + A0
    out = dict(x=A0 / Z, mean=(H1 + A1) / Z, mH=H1 / H0, mA=A1 / A0, Z=Z, A=A, B=B, lam=lam)
    out['var'] = (H2 + A2) / Z - out['mean'] ** 2
    if theta is not None:
        th = min(max(theta, 1e-12), 1.0)
        fH = lambda s: np.exp(-lam * s) * (1 - s) ** (A - 1)
        fA = lambda s: np.exp(-lam * s) * (1 - s) ** A
        PH = integrate.quad(fH, 0, th, weight='alg', wvar=(B, 0), limit=400)[0] / eH
        PA = integrate.quad(fA, 0, th, weight='alg', wvar=(B - 1, 0), limit=400)[0] / eA
        out['P_below'] = (PH + PA) / Z
    return out


def pdmp_pdf(s, coeffs, p, stats=None):
    """Normalised densities (n_H, n_A) on a grid of skills s; their sum integrates to one."""
    st = pdmp_stats(coeffs, p) if stats is None else stats
    A, B, lam = st['A'], st['B'], st['lam']
    base = np.exp(-lam * s)
    nH = base * (1 - s) ** (A - 1) * s ** B / p.eps
    nA = base * (1 - s) ** A * s ** (B - 1) / p.eA
    return nH / st['Z'], nA / st['Z']


# ---------------------------------------------------------------------------
# Vanishing exploration: invasion of an all-AI population by lone holdouts (Appendix B)
# ---------------------------------------------------------------------------
def holdout_threshold(p):
    """Payoff of AI-assisted work theta* = b - c below which lone non-users invade an all-AI
    population, for mu -> 0 and linear imitation (Eq. B6).

    A holdout enters with skill 0 and switches back to AI at rate k_HA(s) = (nu/2)[1 + beta(theta - s)].
    The skill of a holdout, averaged over the time they hold out, follows (1-s)^(A-1) e^(-lam s),
    with A = k_HA(1)/eps and lam = nu beta/(2 eps); holdouts invade when this mean exceeds theta.
    The all-AI state is stable for b > c + theta*, the all-H state for b < 1.
    """
    lam = p.nu * p.beta / (2 * p.eps)

    def mean(theta):
        A = p.nu * (1 + p.beta * (theta - 1)) / (2 * p.eps)
        w = lambda k: integrate.quad(lambda s: s ** k * np.exp(-lam * s), 0, 1, weight='alg', wvar=(0, A - 1))[0]
        return w(1) / w(0)
    return optimize.brentq(lambda th: mean(th) - th, 1e-9, 1 - 1e-9)


# ---------------------------------------------------------------------------
# Agent-based model
# ---------------------------------------------------------------------------
def simulate(p, N=10000, T=200.0, dt=0.05, x0=0.5, s0=None, sigma0=None, rng=None, record_dt=1.0,
             b_of_t=None):
    """Well-mixed agent-based simulation (Appendix C).

    In each step of length dt, every agent revises with probability 1 - exp(-nu dt), using the
    payoffs at the start of the step, and skills then relax exactly over the step.
    b_of_t: optional function t -> b, for a benefit that changes over time.
    Returns the time series t, x (adoption), S (mean skill), mH, mA, and the final state sigma, s.
    """
    rng = np.random.default_rng() if rng is None else rng
    sigma = (rng.random(N) < x0).astype(np.int8) if sigma0 is None else sigma0.copy()
    s = np.full(N, 1 - x0) if s0 is None else np.array(s0, float).copy()
    prev = 1 - np.exp(-np.full(N, p.nu) * dt)
    decayH, decayA = np.exp(-p.eps * dt), np.exp(-p.eA * dt)
    nsteps = int(round(T / dt))
    rec_every = max(1, int(round(record_dt / dt)))
    ts, xs, Ss, mHs, mAs = [], [], [], [], []
    b = p.b
    for step in range(nsteps):
        t = step * dt
        if b_of_t is not None:
            b = b_of_t(t)
        x = sigma.sum() / N
        # revisions
        idx = np.nonzero(rng.random(N) < prev)[0]
        if len(idx):
            explore = rng.random(len(idx)) < p.mu
            ie = idx[explore]
            sigma_new = sigma.copy()
            sigma_new[ie] = (rng.random(len(ie)) < 0.5)
            ii = idx[~explore]
            if len(ii):
                jj = rng.integers(0, N, len(ii))
                diff = sigma[jj] != sigma[ii]
                ii, jj = ii[diff], jj[diff]
                pi_i = np.where(sigma[ii] == 1, b - p.c * x + p.eta * s[ii], s[ii])
                pi_j = np.where(sigma[jj] == 1, b - p.c * x + p.eta * s[jj], s[jj])
                if p.rule == 'fermi':
                    pr = expit(p.beta * (pi_j - pi_i))
                elif p.rule == 'hard':
                    pr = (pi_j > pi_i).astype(float)
                else:
                    pr = np.clip(0.5 + 0.5 * p.beta * (pi_j - pi_i), 0, 1)
                sw = rng.random(len(ii)) < pr
                sigma_new[ii[sw]] = sigma[jj[sw]]
            sigma = sigma_new
        # skills relax exactly over the step
        target = np.where(sigma == 1, 0.0, 1.0)
        s = target + (s - target) * np.where(sigma == 1, decayA, decayH)
        if step % rec_every == 0:
            H = sigma == 0
            ts.append(t); xs.append(x); Ss.append(s.mean())
            mHs.append(s[H].mean() if H.any() else np.nan)
            mAs.append(s[~H].mean() if (~H).any() else np.nan)
    return dict(t=np.array(ts), x=np.array(xs), S=np.array(Ss), mH=np.array(mHs), mA=np.array(mAs),
                sigma=sigma, s=s)


# ---------------------------------------------------------------------------
# Arbitrary switching rates, e.g. the Fermi rule: stationary density by quadrature (Appendix C)
# ---------------------------------------------------------------------------
def pdmp_general(kHA, kAH, p, extra=None, n_grid=4001, n_quad=120):
    """Stationary statistics of the zero-flux density (Eq. B1) for arbitrary rate functions k(s).

    Zero flux gives n_H = w/(eps (1-s)) and n_A = w/(eps_A s), with
    w = (1-s)^A s^B exp(phi(s)), A = k_HA(1)/eps, B = k_AH(0)/eps_A and phi smooth.
    Integrals use Gauss-Jacobi quadrature, which absorbs the endpoint powers exactly.
    `extra(s)` is an optional function whose mean over non-users is returned as extra_H.
    """
    from scipy.special import roots_jacobi
    eH, eA = p.eps, p.eA
    t = np.linspace(0, 1, n_grid)
    kH, kA = np.asarray(kHA(t), float), np.asarray(kAH(t), float)
    A, B = max(kH[-1] / eH, 1e-9), max(kA[0] / eA, 1e-9)
    h = np.empty(n_grid)
    ti = t[1:-1]
    h[1:-1] = (kA[1:-1] - kA[0]) / (eA * ti) - (kH[1:-1] - kH[-1]) / (eH * (1 - ti))
    h[0], h[-1] = 2 * h[1] - h[2], 2 * h[-2] - h[-3]
    phi = integrate.cumulative_trapezoid(h, t, initial=0.0)
    phi -= phi.max()

    def nodes(alpha, beta_):
        z, w = roots_jacobi(n_quad, alpha, beta_)        # weight (1-z)^alpha (1+z)^beta on [-1, 1]
        s = (1 + z) / 2
        return s, w * np.exp(np.interp(s, t, phi))
    sH, wH = nodes(A - 1, B)                             # n_H ∝ (1-s)^(A-1) s^B e^phi / eps
    sA, wA = nodes(A, B - 1)                             # n_A ∝ (1-s)^A s^(B-1) e^phi / eps_A
    wH, wA = wH / eH, wA / eA
    H0, A0 = wH.sum(), wA.sum()
    Z = H0 + A0
    out = dict(x=A0 / Z, mH=(wH * sH).sum() / H0, mA=(wA * sA).sum() / A0, A=A, B=B)
    out['mean'] = ((wH * sH).sum() + (wA * sA).sum()) / Z
    out['var'] = ((wH * sH ** 2).sum() + (wA * sA ** 2).sum()) / Z - out['mean'] ** 2
    if extra is not None:
        out['extra_H'] = (wH * extra(sH)).sum() / H0
    return out


def fermi_rates(x, b, kAH, p):
    """Fermi-rule rates (eta = 0): a non-user of skill s copies an AI user (payoff b - c x) with
    probability F(beta (b - c x - s)), so that k_AH does not depend on the AI user's own skill."""
    PA = p.b if b is None else b
    PA = PA - p.c * x
    kHA = lambda s: p.nu * (p.mu / 2 + (1 - p.mu) * x * expit(p.beta * (PA - s)))
    kA = lambda s: np.full_like(np.asarray(s, float), kAH)
    return kHA, kA, (lambda s: expit(p.beta * (s - PA)))


def fermi_solve(x, p, guess):
    """Self-consistent (b, k_AH) at fixed adoption x under the Fermi rule (eta = 0)."""
    assert p.eta == 0, 'Fermi theory implemented for eta = 0'

    def F(v):
        b, kAH = v
        kHA, kA, ex = fermi_rates(x, b, kAH, p)
        st = pdmp_general(kHA, kA, p, extra=ex)
        return [st['x'] - x, p.nu * (p.mu / 2 + (1 - p.mu) * (1 - x) * st['extra_H']) - kAH]
    sol = optimize.fsolve(F, guess, xtol=1e-11)
    if np.max(np.abs(F(sol))) > 1e-8:
        return None, None
    kHA, kA, _ = fermi_rates(x, sol[0], sol[1], p)
    return sol, pdmp_general(kHA, kA, p)


def fermi_branch(p, xs):
    """Stationary states b(x) under the Fermi rule, with the statistics at each x."""
    out, stats = [], []
    guess = np.array([p.b, p.nu * (p.mu / 2 + (1 - p.mu) * (1 - xs[0]) / 2)])
    for x in xs:
        sol, st = fermi_solve(x, p, guess)
        if sol is None:
            out.append([np.nan, np.nan]); stats.append(None)
            continue
        guess = sol
        out.append(sol); stats.append(st)
    return np.array(out), stats


# ---------------------------------------------------------------------------
# Imitation without noise (rule='hard', eta = 0): two Beta laws joined at theta = b - c x
# ---------------------------------------------------------------------------
def hard_stats(x, delta, p):
    """Stationary statistics at adoption x, with payoff of AI-assisted work theta = 1 - delta.

    Non-users below theta copy AI users at rate nu(1-mu)x; above theta they switch only by
    exploration. k_AH is fixed by the stationary adoption equation, which requires the fraction
    F_H of non-users below theta to equal F* (1/2 when mu -> 0, i.e. their median).
    Working with delta resolves thresholds within 1e-300 of s = 1.
    """
    nu, mu, eps = p.nu, p.mu, p.eps
    Fstar = 0.5 + mu * (x - 0.5) / (2 * (1 - mu) * x * (1 - x))
    Alo = nu * (mu / 2 + (1 - mu) * x) / eps
    Ahi = nu * mu / 2 / eps
    B = nu * (mu / 2 + (1 - mu) * (1 - x) * (1 - Fstar)) / eps
    d = min(max(delta, 1e-300), 1.0)
    lK = (Alo - Ahi) * np.log(d)

    def I(pa, pb, s0=None):
        """int s^pa (1-s)^pb w(s) ds, w = s^B (1-s)^A(s); over [0, s0] if s0 < theta is given."""
        a1, b1, b2 = B + pa + 1, Alo + pb + 1, Ahi + pb + 1
        if s0 is not None:
            return np.exp(betaln(a1, b1)) * betainc(a1, b1, s0)
        lo = np.exp(betaln(a1, b1)) * (1 - betainc(b1, a1, d))
        hi = np.exp(lK + betaln(a1, b2)) * betainc(b2, a1, d)
        return lo + hi, lo
    (NH, NHlo), (NA, _) = I(0, -1), I(-1, 0)
    MH, MA = I(1, -1)[0], I(0, 0)[0]
    MH2, MA2 = I(2, -1)[0], I(1, 0)[0]
    Z = NH + NA
    out = dict(x=NA / Z, FH=NHlo / NH, Fstar=Fstar, mH=MH / NH, mA=MA / NA, theta=1 - d, A_lo=Alo, A_hi=Ahi, B=B)
    out['mean'] = (MH + MA) / Z
    out['var'] = (MH2 + MA2) / Z - out['mean'] ** 2
    out['below'] = lambda s0: (I(0, -1, s0) + I(-1, 0, s0)) / Z      # fraction of agents with s < s0 < theta
    return out


def hard_solve(x, p):
    """Stationary state(s) at adoption x: the theta that reproduces x, and b = theta + c x."""
    g = lambda u: hard_stats(x, 10.0 ** (-u), p)['x'] - x
    us = np.concatenate([-np.log10(1 - np.linspace(1e-6, 0.9, 200)), np.linspace(1.0001, 300, 600)])
    vals = np.array([g(u) for u in us])
    out = []
    for i in np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]:
        u = optimize.brentq(g, us[i], us[i + 1], xtol=1e-12)
        st = hard_stats(x, 10.0 ** (-u), p)
        st['b'] = st['theta'] + p.c * x
        out.append(st)
    return out


def hard_branch(p, xs):
    """Stationary states b(x) under imitation without noise (nan where no unique solution)."""
    bs = []
    for x in xs:
        sol = hard_solve(x, p)
        bs.append(sol[0]['b'] if len(sol) == 1 else np.nan)
    return np.array(bs)
