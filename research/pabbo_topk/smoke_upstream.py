"""Unmodified PABBO checkpoint/model minimal run, before any adaptation.

PABBO by Zhang, Huang, Kaski and Martinelli, ICLR 2025, AGPL-3.0.
Upstream and its license are preserved in upstream/. This harness, 2026-10-08,
is distributed under AGPL-3.0; see LICENSE.txt and NOTICE.md.
"""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[2]
UP = Path(__file__).parent/'upstream'


def load_source():
    spec = importlib.util.spec_from_file_location('fsp_pabbo_upstream', UP/'policies/transformer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def architecture(d=6):
    return dict(d_x=d, d_f=1, d_model=64, nhead=4, dropout=0., n_layers=6,
                dim_feedforward=128, emb_depth=3, transformer_encoder_layer_cls='efficient')


def run():
    torch.set_num_threads(1)
    torch.manual_seed(10881001)
    original = load_source()
    results = []
    for d in (1, 2, 6):
        path = UP/f'results/PABBO/PABBO_GP{d}D/ckpt.tar'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = original.TransformerModel(**architecture(d))
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        x = torch.rand(1, 18, d)*2-1
        context = torch.cat([x[:, :1], x[:, 1:2]], dim=-1)
        choices = torch.tensor([[[1]]])
        start = perf_counter()
        with torch.no_grad():
            acquisition = model(context, choices, x, t=1, T=3)[0]
            prediction = model(context, choices, x, acquire=False)[1]
        assert acquisition.shape == (1, 153, 1)
        assert prediction.shape == (1, 18, 1)
        assert torch.isfinite(acquisition).all() and torch.isfinite(prediction).all()
        assert acquisition.std().item() > 0
        results.append({'dimension': d, 'checkpoint_sha256': sha256(path.read_bytes()).hexdigest(),
                        'saved_step': checkpoint['step'], 'parameters': sum(p.numel() for p in model.parameters()),
                        'acquisition_shape': list(acquisition.shape), 'prediction_shape': list(prediction.shape),
                        'seconds': perf_counter()-start})
    return {'status': 'PASS', 'scope': 'unmodified published models and weights, not full paper reproduction',
            'torch': torch.__version__, 'seed': 10881001, 'checkpoints': results,
            'source_sha256': sha256((UP/'policies/transformer.py').read_bytes()).hexdigest()}


if __name__ == '__main__':
    payload = run()
    target = ROOT/'output/validation/pabbo-topk-20261008/upstream-smoke.json'
    with target.open('x') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
