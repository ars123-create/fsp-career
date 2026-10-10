"""AGPL-3.0 PABBO adaptation, 2026-10-08. See NOTICE.md and LICENSE.txt.

Uses the published PABBO transformer and pretrained parameters. Changes:
consistent attention including empty context; deterministic prediction mean;
pool-aware acquisition residual; discrete, monotonic shortlist contract.
"""
from graphlib import TopologicalSorter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .smoke_upstream import UP, architecture, load_source

UpstreamModel = load_source().TransformerModel


class PabboModel(UpstreamModel):
    def __init__(self):
        super().__init__(**architecture())
        # Initial residual is exactly zero, preserving the transferred actor.
        self.pool_head = nn.Sequential(nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.pool_head[-1].weight)
        nn.init.zeros_(self.pool_head[-1].bias)

    def encode(self, query_src, c_src, eval_pos):
        """Explicit published cross-attention; no Torch fused-path substitution.

        For empty context, independent one-token attention replaces the upstream
        first-target-as-context artifact. Dropout in the pinned model is zero.
        """
        nctx, nout = query_src.shape[1], eval_pos.shape[1]
        emb = self.embed_input(query_src, c_src, eval_pos, nctx)
        out = emb
        for layer in self.encoder.layers:
            if layer.norm_first or not layer.self_attn.batch_first:
                raise RuntimeError('unexpected upstream transformer configuration')
            if nctx:
                attended = layer.self_attn(out, out[:, :nctx], out[:, :nctx], need_weights=False)[0]
            else:
                independent = out.reshape(-1, 1, self.d_model)
                attended = layer.self_attn(independent, independent, independent,
                                          need_weights=False)[0].reshape_as(out)
            out = layer.norm1(out+layer.dropout1(attended))
            out = layer.norm2(out+layer._ff_block(out))
        if self.encoder.norm is not None:
            out = self.encoder.norm(out)
        return out, nout, emb

    def outputs(self, context, choices, x, pairs=None, *, slot=0, budget=3, k=5):
        z = self.encode(context, choices, x)[0][:, -x.shape[1]:]
        raw = self.eval_bucket_decoder(z)
        mean = raw[..., 0]
        sd = F.softplus(raw[..., 1])+1e-4
        logits = None
        if pairs is not None and len(pairs):
            pair_z = z[:, pairs].flatten(start_dim=-2)
            clock = torch.full((*pair_z.shape[:-1], 1), (slot+1)/budget,
                               device=x.device, dtype=x.dtype)
            logits = self.af_mlp(torch.cat([pair_z, clock], -1)).squeeze(-1)
            k = min(k, mean.shape[-1])
            boundary = mean.topk(k, dim=-1).values[..., -1:].detach()
            a, b = pairs[:, 0], pairs[:, 1]
            context_features = torch.stack([
                mean[:, a]-boundary, mean[:, b]-boundary,
                sd[:, a], sd[:, b], mean[:, a]-mean[:, b],
                boundary.expand(-1, len(pairs))], -1).detach()
            logits = logits+self.pool_head(context_features).squeeze(-1)
        return mean, sd, logits


def published_model():
    model = PabboModel()
    checkpoint = torch.load(UP/'results/PABBO/PABBO_GP6D/ckpt.tar',
                            map_location='cpu', weights_only=True)
    incompatible = model.load_state_dict(checkpoint['model'], strict=False)
    expected = {'pool_head.0.weight', 'pool_head.0.bias', 'pool_head.2.weight', 'pool_head.2.bias'}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError('published checkpoint architecture drift')
    return model


def load_model(path=None):
    model = published_model()
    if path is not None:
        state = torch.load(Path(path), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    return model.eval()


def values_input(values):
    x = np.asarray(values, dtype=np.float64)
    if (x.ndim != 2 or not 1 <= len(x) <= 100 or not 1 <= x.shape[1] <= 3
            or not np.isfinite(x).all() or np.any((x < 0) | (x > 1))):
        raise ValueError('1..100 profiles and 1..3 finite normalized skills required')
    return x.copy()


def pair_key(left, right):
    return tuple(sorted((tuple(left), tuple(right))))


def history_input(history, dimensions):
    if not isinstance(history, (list, tuple)) or len(history) > 3:
        raise ValueError('at most three comparison slots required')
    cleaned, seen = [], set()
    for entry in history:
        if not isinstance(entry, dict) or set(entry) != {'left', 'right', 'choice'}:
            raise ValueError('comparison requires exactly left, right, choice')
        vectors = np.asarray([entry['left'], entry['right']], dtype=float)
        if vectors.shape != (2, dimensions):
            raise ValueError('comparison skill dimensions differ')
        values_input(vectors)
        left, right = map(tuple, vectors)
        choice = entry['choice']
        if choice is not None and (type(choice) is not int or choice not in (0, 1)):
            raise ValueError('choice must be 0, 1 or None')
        key = pair_key(left, right)
        if left == right or key in seen:
            raise ValueError('identical or repeated comparison')
        seen.add(key)
        if left > right:
            left, right = right, left
            choice = None if choice is None else 1-choice
        cleaned.append({'left': left, 'right': right, 'choice': choice})
    return cleaned


def encode_values(x):
    """Pinned 6D transfer: zero-fill unused normalized coordinates."""
    t = torch.as_tensor(np.array(x, copy=True), dtype=torch.float32)
    return F.pad(2*t-1, (0, 6-t.shape[-1]))


def tensors(x, history):
    actual = [h for h in history if h['choice'] is not None]
    if actual:
        left = encode_values([h['left'] for h in actual])
        right = encode_values([h['right'] for h in actual])
        context = torch.cat([left, right], -1).unsqueeze(0)
        choices = torch.tensor([h['choice'] for h in actual]).reshape(1, -1, 1)
    else:
        context, choices = torch.empty(1, 0, 12), torch.empty(1, 0, 1, dtype=torch.long)
    return context, choices, encode_values(x).unsqueeze(0)


def query_pool(x, history):
    seen = {pair_key(h['left'], h['right']) for h in history}
    pairs = []
    for a, b in combinations(range(len(x)), 2):
        key = pair_key(x[a], x[b])
        difference = x[a]-x[b]
        if key in seen or not (np.any(difference > 0) and np.any(difference < 0)):
            continue
        seen.add(key)
        pairs.append((a, b))
    return pairs


def guarded_topk(values, scores, k=5):
    """Priority topological traversal: no excluded strict dominator of a pick.

    This is an explicit decision rule, not a claim that the neural utility is
    monotonic or that greedy traversal maximizes its constrained sum.
    """
    x = values_input(values)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(x),) or not np.isfinite(scores).all():
        raise ValueError('finite score per profile required')
    if type(k) is not int or not 1 <= k <= len(x):
        raise ValueError('integer 1 <= k <= n required')
    graph = {i: np.flatnonzero(np.all(x >= x[i], axis=1)&np.any(x > x[i], axis=1)).tolist()
             for i in range(len(x))}
    sorter = TopologicalSorter(graph)
    sorter.prepare()
    ready, result = set(), []
    for _ in range(k):
        ready.update(sorter.get_ready())
        selected = min(ready, key=lambda i: (-round(float(scores[i]), 6), tuple(-x[i]), i))
        ready.remove(selected)
        result.append(selected)
        sorter.done(selected)
    return tuple(result)


def batched_guarded_topk(x, scores, k=5):
    """Same dominance traversal for synthetic training tensors, without gradients."""
    dominates = (x[:, :, None] >= x[:, None, :]).all(-1)&(x[:, :, None] > x[:, None, :]).any(-1)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    result = []
    for _ in range(k):
        blocked = (dominates&~selected[:, :, None]).any(1)
        choice = scores.masked_fill(selected|blocked, -torch.inf).argmax(1)
        selected.scatter_(1, choice[:, None], True)
        result.append(choice)
    return torch.stack(result, 1)


def recommend(model, values, history=(), *, k=5, strategy='learned', rng=None):
    x = values_input(values)
    history = history_input(history, x.shape[1])
    if type(k) is not int or k < 1 or k > 5:
        raise ValueError('integer k in 1..5 required')
    if strategy not in ('learned', 'random', 'none'):
        raise ValueError('unknown question strategy')
    k = min(k, len(x))
    # Semantic vector order makes query orientation and candidate order stable.
    order = np.array(sorted(range(len(x)), key=lambda i: tuple(x[i])))
    ordered = x[order]
    pairs = query_pool(ordered, history) if len(history) < 3 and len(x) > k else []
    context, choices, pending = tensors(ordered, history)
    tpairs = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    with torch.no_grad():
        mean, sd, logits = model.outputs(context, choices, pending, tpairs,
                                         slot=len(history), budget=3, k=k)
    scores = mean[0].numpy()
    if not np.isfinite(scores).all() or not torch.isfinite(sd).all():
        raise ArithmeticError('non-finite model prediction')
    selected = guarded_topk(ordered, scores, k)
    question = None
    if pairs and strategy != 'none':
        if not torch.isfinite(logits).all():
            raise ArithmeticError('non-finite acquisition')
        if strategy == 'random':
            if rng is None:
                raise ValueError('random strategy needs explicit rng')
            index = int(rng.integers(len(pairs)))
        else:
            index = int(logits[0].argmax())
        a, b = pairs[index]
        question = {'left': ordered[a].tolist(), 'right': ordered[b].tolist(),
                    'indices': [int(order[a]), int(order[b])]}
    original_scores = np.empty(len(x)); original_scores[order] = scores
    return {'selected': [int(order[i]) for i in selected], 'question': question,
            'latent_scores': original_scores.tolist(), 'slots_used': len(history),
            'answers': sum(h['choice'] is not None for h in history),
            'remaining': 3-len(history), 'model': 'PABBO top-5 experimental'}
