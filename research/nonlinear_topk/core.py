"""BoTorch nonlinear utility plus exact one-step Gaussian SUM top-k EVOI.

GP/Laplace/probit are supplied by BoTorch (Meta, MIT), also used by
Eichelbeck, Voigt & Althoff, ICLR2026 (pinned MIT source in upstream/).
This adapter changes initialization, discrete query pool and decision utility.
Known Gaussian conditioning and decision theory; no scientific novelty claim.
"""
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from scipy.special import ndtr
import torch
from botorch.models.pairwise_gp import PairwiseGP
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.means import Mean


def inputs(values):
    x = np.asarray(values, dtype=np.float64)
    if (x.ndim != 2 or not 1 <= len(x) <= 100 or not 1 <= x.shape[1] <= 3
            or not np.isfinite(x).all() or np.any((x < 0) | (x > 1))):
        raise ValueError('1..100 profiles and 1..3 finite features in [0,1] required')
    return x.copy()


def topk(scores, k):
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or not np.isfinite(s).all() or type(k) is not int or not 1 <= k <= len(s):
        raise ValueError('finite score vector and integer 1 <= k <= n required')
    return tuple(map(int, np.argsort(-np.round(s, 12), kind='stable')[:k]))


def topksum(values, k):
    v = np.asarray(values, dtype=float)
    if v.ndim < 1 or not np.isfinite(v).all() or type(k) is not int or not 1 <= k <= v.shape[-1]:
        raise ValueError('invalid scores or k')
    return np.partition(v, v.shape[-1]-k, axis=-1)[..., -k:].sum(-1)


def valid_pairs(values, asked=()):
    x = inputs(values)
    excluded = {tuple(sorted(p)) for p in asked}
    if any(len(p) != 2 or p[0] == p[1] or min(p) < 0 or max(p) >= len(x)
           or any(type(i) not in (int, np.int64) for i in p) for p in excluded):
        raise ValueError('invalid previously asked pair')
    seen = {tuple(sorted((tuple(x[a]), tuple(x[b])))) for a, b in excluded}
    pairs = []
    for a, b in combinations(range(len(x)), 2):
        delta = x[a]-x[b]
        key = tuple(sorted((tuple(x[a]), tuple(x[b]))))
        if ((a, b) in excluded or key in seen
                or not (np.any(delta > 0) and np.any(delta < 0))):
            continue
        pairs.append((a, b))
        seen.add(key)
    return pairs


@dataclass(frozen=True)
class GaussianBelief:
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        m, c = np.asarray(self.mean, float), np.asarray(self.covariance, float)
        if (m.ndim != 1 or len(m) < 1 or c.shape != (len(m), len(m))
                or not np.isfinite(m).all() or not np.isfinite(c).all()
                or not np.allclose(c, c.T, atol=1e-9, rtol=1e-9)
                or np.linalg.eigvalsh(c).min() < -1e-7):
            raise ValueError('finite mean and symmetric positive semidefinite covariance required')
        m, c = m.copy(), ((c+c.T)/2).copy()
        m.setflags(write=False)
        c.setflags(write=False)
        object.__setattr__(self, 'mean', m)
        object.__setattr__(self, 'covariance', c)


@dataclass(frozen=True)
class Query:
    pair: tuple[int, int]
    gain: float
    probability_left: float
    eubo: float


def gaussian_queries(belief, k, pairs, *, noise_variance=2.0):
    """Exact first-moment VOI under THIS Gaussian and probit observation model.

    If D=F_a-F_b+noise, E[F 1(D>0)] = Phi(z)m + Cov(F,D)phi(z)/sd(D).
    TopKSum's positive homogeneity cancels branch normalizing constants.
    After an answer, the full model uses a fresh Laplace approximation; this
    one-step identity is not a guarantee for that entire approximate sequence.
    """
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError('positive finite noise variance required')
    m, c = belief.mean, belief.covariance
    baseline = float(topksum(m, k))
    if not pairs:
        return []
    if any(len(p) != 2 or any(type(i) not in (int, np.int64) for i in p)
           or p[0] == p[1] or min(p) < 0 or max(p) >= len(m) for p in pairs):
        raise ValueError('distinct valid pair indices required')
    a, b = np.asarray(pairs, int).T
    variances = np.maximum(c[a, a]+c[b, b]-2*c[a, b], 0.)
    sd = np.sqrt(variances+noise_variance)
    z = (m[a]-m[b])/sd
    prob = ndtr(z)
    pdf = np.exp(-z*z/2)/math.sqrt(2*math.pi)
    first = prob[:, None]*m + (c[:, a]-c[:, b]).T*(pdf/sd)[:, None]
    second = m-first
    gains = topksum(first, k)+topksum(second, k)-baseline
    if gains.min() < -1e-9:
        raise ArithmeticError('negative Gaussian VOI beyond roundoff')
    latent_sd = np.sqrt(np.maximum(variances, 1e-16))
    u = (m[a]-m[b])/latent_sd
    eubo = m[b]+(m[a]-m[b])*ndtr(u)+latent_sd*np.exp(-u*u/2)/math.sqrt(2*math.pi)
    return [Query(tuple(map(int, pair)), float(max(gain, 0.)), float(p), float(e))
            for pair, gain, p, e in zip(pairs, gains, prob, eubo)]


@dataclass(frozen=True)
class Config:
    residual_sd: float = .04
    lengthscale: float = .25
    balance_mean: float = 0.
    utility_scale: float = .05

    def __post_init__(self):
        if (not all(np.isfinite(v) for v in vars(self).values())
                or self.residual_sd < 0 or self.lengthscale <= 0
                or not 0 <= self.balance_mean <= 1 or self.utility_scale <= 0):
            raise ValueError('invalid fixed GP prior parameters')


class SkillMean(Mean):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        average = x.mean(-1)
        m = average+self.config.balance_mean*(x.min(-1).values-average)
        return (m-.75)/self.config.utility_scale


class SkillKernel(Kernel):
    """Moment-matched Dirichlet linear uncertainty plus nonlinear RBF residual.

    This is a Gaussian approximation, not bounded positive simplex weights.
    Consequently neither the prior nor posterior guarantees monotonicity.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x1, x2, diag=False, **params):
        del params
        d = x1.shape[-1]
        a, b = x1-x1.mean(-1, keepdim=True), x2-x2.mean(-1, keepdim=True)
        if diag:
            linear = (a*b).sum(-1)/(d*(d+1))
            squared = ((x1-x2)/self.config.lengthscale).square().sum(-1)
        else:
            linear = a @ b.transpose(-1, -2)/(d*(d+1))
            squared = ((x1.unsqueeze(-2)-x2.unsqueeze(-3))/self.config.lengthscale).square().sum(-1)
        return linear+self.config.residual_sd**2*torch.exp(-.5*squared)


def posterior(values, history=(), config=Config()):
    """Only actual binary answers enter likelihood; skip supplies no evidence."""
    x = inputs(values)
    comparisons = []
    for pair, choice in history:
        if (len(pair) != 2 or any(type(i) not in (int, np.int64) for i in pair)
                or pair[0] == pair[1] or min(pair) < 0 or max(pair) >= len(x)):
            raise ValueError('invalid observation pair')
        if choice is None:
            continue
        if type(choice) is not int or choice not in (0, 1):
            raise ValueError('choice must be 0, 1 or None')
        if np.array_equal(x[pair[0]], x[pair[1]]):
            raise ValueError('identical options cannot provide preference evidence')
        comparisons.append(pair if choice == 0 else pair[::-1])
    tx = torch.from_numpy(x)
    kernel = ScaleKernel(SkillKernel(config)).double()
    kernel.outputscale = 1/config.utility_scale**2
    model = PairwiseGP(None, None, covar_module=kernel, jitter=1e-8).double()
    model.mean_module = SkillMean(config)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if comparisons:
        used = sorted(set(i for pair in comparisons for i in pair))
        index = {old: new for new, old in enumerate(used)}
        tc = torch.tensor([[index[a], index[b]] for a, b in comparisons], dtype=torch.long)
        # BoTorch uses NumPy for its MAP starting point. Keep calls deterministic
        # and restore external state. This isolated adapter is single-threaded.
        state = np.random.get_state()
        try:
            np.random.seed(10850001)
            model.set_train_data(tx[used], tc)
        finally:
            np.random.set_state(state)
    model.eval()
    with torch.no_grad():
        p = model.posterior(tx)
        belief = GaussianBelief(p.mean.squeeze(-1).numpy(), p.distribution.covariance_matrix.numpy())
    return belief


def recommend(values, history=(), *, config=Config(), policy='topk', k=5):
    x = inputs(values)
    if type(k) is not int or k < 1:
        raise ValueError('positive integer k required')
    if policy not in ('topk', 'eubo'):
        raise ValueError('unknown policy')
    if len(history) > 3:
        raise ValueError('at most three comparison slots')
    k = min(k, len(x))
    belief = posterior(x, history, config)
    selected = topk(belief.mean, k)
    pairs = valid_pairs(x, [p for p, _ in history])
    query = None
    if len(history) < 3 and k < len(x):
        options = gaussian_queries(belief, k, pairs)
        if options:
            if policy == 'topk':
                query = max(options, key=lambda q: q.gain)
            elif policy == 'eubo':
                query = max(options, key=lambda q: q.eubo)
            else:
                raise ValueError('unknown policy')
    return selected, query, belief
