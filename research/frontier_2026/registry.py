"""Explicit experiment configurations, not a count of novel algorithms."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from .pilot import CONFIGS as PILOT_CONFIGS
from .policies import Config as ParticleConfig, ParticleSession
from .components.monne.adapter import Config as MoNNEConfig, MoNNE
from .components.constrained_gp.adapter import Config as GPConfig, Session as GPSession
from .components.legacy_controls import METHODS as LEGACY, make_session as legacy_session
from .components.risk_decisions import Config as RiskConfig, Session as RiskSession
from .components.distillation.adapter import Config as DistillConfig, Session as DistillSession

PARTICLES = dict(PILOT_CONFIGS)
PARTICLES.update({
    'linear_kg_matched_frozen_quadrature': ParticleConfig(prior='linear', m=9, seed=202610072),
    'linear_kg_small_frozen_seed': ParticleConfig(prior='linear', m=4, seed=202610072),
    'linear_kg_fine_new_seed': ParticleConfig(prior='linear', m=9),
    'mixed_kg_finer_quadrature': ParticleConfig(m=6),
    'mixed_kg_temperature008': ParticleConfig(temperature=.08),
})

RISK = {"P08_kl_dro_nominal_query": "kl_dro", "P09_cvar_nominal_query": "cvar",
        "P10_bounded_error_mmr": "mmr", "P11_baseline_safe_nominal_query": "baseline_safe",
        "P17_regret_edgecut": "edgecut"}
MONNE = {"P03_monne_top5": "top5", "monne_ieubo": "ieubo"}
CONSTRAINED = {"P06_constrained_gp_kg": "top5", "constrained_gp_bald": "information",
               "constrained_gp_eubo": "eubo"}
GP_CONTROLS = {"gp_eubo_guard": "eubo", "gp_no_questions_guard": "none"}
DISTILLED = {"P20_learned_lookahead": "learned_lookahead", "distilled_random_lookahead": "random_lookahead",
             "distilled_bounded_teacher_pool8": "teacher", "distilled_matched_greedy": "greedy"}
METHODS = tuple(PARTICLES)+tuple(LEGACY)+tuple(GP_CONTROLS)+tuple(CONSTRAINED)+tuple(RISK)+tuple(MONNE)+tuple(DISTILLED)


def description(name):
    if name in PARTICLES:
        return {"component": "particle", "config": asdict(PARTICLES[name])}
    if name in LEGACY:
        return {"component": "frozen_legacy", "method": name,
                "source": "components/legacy_controls/verification.json"}
    if name in GP_CONTROLS:
        return {"component": "frozen_gp_with_guard", "acquisition": GP_CONTROLS[name]}
    if name in CONSTRAINED:
        return {"component": "finite_pool_rejection_snis_gp", "config": asdict(GPConfig()),
                "acquisition": CONSTRAINED[name], "exact_sun_inference": False}
    if name in RISK:
        return {"component": "risk_decisions", "config": asdict(RiskConfig()), "method": RISK[name],
                "risk_aligned_query": name not in {"P08_kl_dro_nominal_query", "P09_cvar_nominal_query",
                                                   "P11_baseline_safe_nominal_query"}}
    if name in MONNE:
        return {"component": "upstream_monne_adaptation", "config": asdict(MoNNEConfig()),
                "acquisition": MONNE[name], "posterior_calibrated": False}
    if name in DISTILLED:
        config = asdict(DistillConfig(mode=DISTILLED[name]))
        checkpoint = Path(config['checkpoint'])
        config['checkpoint'] = str(checkpoint.relative_to(Path(__file__).resolve().parents[2]))
        return {"component": "trained_value_distillation", "config": config,
                "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                "exact_global_teacher": False}
    raise ValueError(f"Unknown method: {name}")


def create(name, values):
    """Only observed features cross this constructor boundary."""
    if name in PARTICLES:
        session = ParticleSession(values, PARTICLES[name])
        return session, session.recommend
    if name in LEGACY:
        session = legacy_session(name, values)
        return session, session.recommend
    if name in GP_CONTROLS:
        session = legacy_session("gp_topk_guard", values)
        return session, lambda history: session.recommend(history, acquisition=GP_CONTROLS[name])
    if name in CONSTRAINED:
        session = GPSession(values, GPConfig())
        return session, lambda history: session.recommend(history, acquisition=CONSTRAINED[name])
    if name in RISK:
        session = RiskSession(values, RiskConfig())
        return session, lambda history: session.recommend(history, method=RISK[name])
    if name in MONNE:
        session = MoNNE(values, MoNNEConfig())
        return session, lambda history: session.recommend(history, acquisition=MONNE[name])
    if name in DISTILLED:
        session = DistillSession(values, DistillConfig(mode=DISTILLED[name]))
        return session, session.recommend
    raise ValueError(f"Unknown method: {name}")
