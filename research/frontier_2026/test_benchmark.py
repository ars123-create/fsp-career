"""Evaluator boundary and error retention, independent of optimizer quality."""
import numpy as np
import pytest

from .benchmark import run, summarize, SummaryAccumulator, seal_protocol, read_sealed_protocol
from .finite import tradeoff_pairs
from .simulation import make_world


@pytest.mark.parametrize('method', ['equal', 'linear_evoi_3', 'gp_topk_guard',
                                    'P01_mixed_kg', 'P06_constrained_gp_kg', 'P17_regret_edgecut'])
def test_common_contract(method):
    row = run(make_world(771028, 'linear', n=8), method)
    assert row['status'] == 'PASS', row
    assert len(row['regret_by_slot']) == 4
    assert row['slots'] <= 3
    assert row['constraint_status'] == 'PASS'


def test_only_features_enter_factory_and_errors_are_retained():
    world = make_world(771029, 'linear', n=8)
    seen = []
    def factory(name, values):
        assert name == 'fixture'
        assert isinstance(values, np.ndarray)
        assert np.array_equal(values, world.values)
        assert values is not world.values
        seen.append(True)
        raise ArithmeticError('deliberate fixture failure')
    row = run(world, 'fixture', factory=factory)
    assert seen == [True]
    assert row['status'] == 'FAIL'
    assert row['error_type'] == 'ArithmeticError'
    assert summarize([row], ['fixture'])['fixture']['failures'] == 1


@pytest.mark.parametrize('defect', ['repeat', 'budget', 'selection', 'query_type'])
def test_contract_violation_is_failure(defect):
    world = make_world(771030, 'linear', n=8)
    class Broken:
        pairs = tradeoff_pairs(world.values)
        def recommend(self, history):
            selected = (0, 1, 2, 3, 4)
            if defect == 'selection':
                selected = (0, 0, 1, 2, 3)
            question = 0 if defect == 'repeat' else len(history)
            if defect == 'query_type':
                question = True
            return selected, question
    def factory(name, values):
        session = Broken()
        return session, session.recommend
    result = run(world, 'fixture', factory=factory)
    assert result['status'] == 'FAIL'


def test_streaming_summary_matches_independent_batch_implementation():
    rows = [run(make_world(s, 'linear', n=8), m)
            for s in (71, 72) for m in ('equal','P01_mixed_kg')]
    rows.append({'method':'equal','seed':73,'status':'FAIL','total_seconds':.5})
    accumulator = SummaryAccumulator(['equal','P01_mixed_kg'])
    for row in rows:
        accumulator.add(row)
    assert accumulator.result() == summarize(rows, ['equal','P01_mixed_kg'])


def protocol_fixture(tmp_path):
    return {'args':{'out':str(tmp_path),'seed_start':71,'seeds':2,'n':18,'d':3,
                    'methods':['equal'],'families':['linear']}, 'seeds':[71,72], 'execution_workers':2}


def test_protocol_checksum_catches_changed_criteria(tmp_path):
    path = tmp_path/'protocol.json'
    p = protocol_fixture(tmp_path)
    seal_protocol(path, p)
    assert read_sealed_protocol(path) == p
    path.write_text(path.read_text().replace('"n": 18','"n": 19'))
    with pytest.raises(ValueError, match='checksum'):
        read_sealed_protocol(path)


@pytest.mark.parametrize('defect', ['duplicate','negative','count','empty','workers'])
def test_sealed_protocol_semantics_reject_invalid_seed_shards(tmp_path,defect):
    p = protocol_fixture(tmp_path)
    if defect == 'duplicate': p['seeds'] = [71,71]
    elif defect == 'negative': p['args']['seed_start'] = -1; p['seeds'] = [-1,0]
    elif defect == 'count': p['args']['seeds'] = 3
    elif defect == 'empty': p['args']['seeds'] = 0; p['seeds'] = []
    elif defect == 'workers': p['execution_workers'] = 3
    path = tmp_path/'protocol.json'
    seal_protocol(path,p)
    with pytest.raises(ValueError):
        read_sealed_protocol(path)
