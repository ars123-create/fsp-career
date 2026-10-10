"""Independent executable references, not checks against the generator itself."""
import ast
import contextlib
from fractions import Fraction
import io
import sqlite3
from urllib.parse import urlparse, parse_qs

import pytest
from hypothesis import given, settings, strategies as st
from validation.assessment_v2.bank import build_practical, correct, number, validate_values


def database(q):
    db = sqlite3.connect(':memory:')
    for t in q['tables']:
        db.execute(f"CREATE TABLE {t['name']} ({', '.join(t['columns'])})")
        db.executemany(f"INSERT INTO {t['name']} VALUES ({','.join('?' for _ in t['columns'])})",t['rows'])
    return db


def reference(q, grade):
    family = q['family']
    with contextlib.closing(database(q)) as db:
        if family.startswith('sql_'):
            cursor = db.execute(q['code'])
            return dict(zip([c[0] for c in cursor.description],cursor.fetchone()))
        if family.startswith('python_'):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                # Only fixed local generator source, never candidate input.
                exec(q['code'], {})
            return ast.literal_eval(output.getvalue())
        if family == 'api_retry_ledger':
            initial = int(q['prompt'].split(': ')[1].split('.')[0])
            db.executescript('CREATE TABLE ledger(key TEXT PRIMARY KEY); CREATE TABLE stock(value INTEGER);')
            db.execute('INSERT INTO stock VALUES(?)',(initial,)); db.commit()
            for _,key,amount,stage in q['tables'][0]['rows']:
                changed = db.execute('INSERT OR IGNORE INTO ledger VALUES(?)',(key,)).rowcount
                if changed: db.execute('UPDATE stock SET value=value-?',(amount,))
                db.rollback() if stage == 'before_commit' else db.commit()
            return {'stock':db.execute('SELECT value FROM stock').fetchone()[0],
                    'writes':db.execute('SELECT COUNT(*) FROM ledger').fetchone()[0]}
        if family == 'api_cursor_page':
            params=parse_qs(urlparse(q['code'][4:]).query)
            args=[]; condition=''
            if 'after_ts' in params:
                ts=int(params['after_ts'][0]);id_=int(params['after_id'][0])
                condition=' WHERE created_at < ? OR (created_at = ? AND id < ?)'
                args=[ts,ts,id_]
            args.append(int(params['limit'][0]))
            page=db.execute('SELECT id FROM events'+condition+' ORDER BY created_at DESC,id DESC LIMIT ?',args).fetchall()
            return {'first_id':page[0][0],'last_id':page[-1][0]}
        if family == 'analytics_standardization':
            rates={v:Fraction(conv,vis) for v,conv,vis in db.execute('SELECT variant,SUM(conversions),SUM(visitors) FROM experiment GROUP BY variant')}
            difference=(rates['B']-rates['A'])*100
            if grade=='Junior': adjusted=(rates['B']/rates['A']-1)*100
            else:
                segments={v:[] for v in ('A','B')}
                for v,conv,vis in db.execute('SELECT variant,conversions,visitors FROM experiment'):
                    segments[v].append(Fraction(conv,vis))
                adjusted=(sum(segments['B'])-sum(segments['A']))*50
            return {'raw_delta':difference,'adjusted':adjusted}
        if family == 'analytics_temporal_features':
            row=db.execute("""WITH valid AS (
                SELECT *,ROW_NUMBER() OVER (PARTITION BY name ORDER BY occurred_at DESC,id DESC) AS rn
                FROM features WHERE occurred_at<=60 AND available_at<=60 AND role='feature')
                SELECT COUNT(*),COALESCE(SUM(value),0) FROM valid WHERE rn=1""").fetchone()
            return dict(zip(('features','total'),row))
        if family == 'quality_versioned_events':
            order='version DESC,seq ASC' if grade=='Middle' else 'seq DESC'
            row=db.execute(f"""WITH ranked AS (
                SELECT *,ROW_NUMBER() OVER (PARTITION BY source,event_id ORDER BY {order}) AS rn FROM deliveries)
                SELECT COUNT(*),COALESCE(SUM(value),0) FROM ranked WHERE rn=1 AND op='upsert'""").fetchone()
            return dict(zip(('live','total'),row))
        if family == 'quality_contract':
            row=db.execute("""SELECT COUNT(*),COALESCE(SUM(CASE unit WHEN 's' THEN duration*1000 ELSE duration END),0)
                FROM measurements WHERE typeof(duration)='integer' AND duration>=0 AND
                ((version=1 AND unit='ms') OR (version=2 AND unit IN ('ms','s')))""").fetchone()
            return dict(zip(('accepted','milliseconds'),row))
    raise AssertionError(f'No independent reference: {family}')


def rounded_fraction(value, places):
    """Exact rational rounding, independent of Decimal/production comparator."""
    value=Fraction(value)*10**places
    sign=-1 if value<0 else 1
    x=abs(value)
    n=(2*x.numerator+x.denominator)//(2*x.denominator)
    return Fraction(sign*n,10**places)


@pytest.mark.parametrize('spec',['backend','data'])
@pytest.mark.parametrize('grade',['Junior','Middle'])
def test_200_seeds_per_form_against_execution(spec,grade):
    for seed in range(200):
        qs=build_practical(spec,grade,seed)
        assert len(qs)==6 and len({q['family'] for q in qs})==6
        assert sorted(sum(q['skill']==s for q in qs) for s in {q['skill'] for q in qs})==[2,2,2]
        for q in qs:
            independent=reference(q,grade)
            assert set(independent)==set(q['_correct'])
            for field in q['fields']:
                name=field['name']
                assert Fraction(q['_correct'][name])==rounded_fraction(independent[name],field['places']), (spec,grade,seed,q)
            assert correct(q,q['_correct'])
            wrong={k:str(Fraction(v)+1) if Fraction(v).denominator==1 else str(float(v)+1) for k,v in q['_correct'].items()}
            assert not correct(q,wrong)


@pytest.mark.parametrize('bad',['', 'NaN','Infinity','1e4','1/2','2**8','1_000','1 000','1; DROP TABLE x','1.1234567','1'*41,'１２',True,1,{},None])
def test_parser_rejects_expressions_nonfinite_and_wrong_types(bad):
    with pytest.raises(ValueError): number(bad,2)


@settings(max_examples=120,derandomize=True)
@given(st.integers(-10**6,10**6))
def test_equivalent_integer_representations(n):
    assert number(f'  {n}.0  ')==number(str(n))==number(f'{n},000')


def test_rounding_and_answer_fields():
    assert str(number(' -1,235 ',2))=='-1.24'
    assert str(number('1.225',2))=='1.23'
    with pytest.raises(ValueError):number('2.1')
    q=build_practical('backend','Middle',1)[0]
    for values in ({}, {'invented':'3'}, {**q['_correct'],'extra':'0'}):
        with pytest.raises(ValueError):validate_values(q,values)
    assert not correct(q,None)


def test_determinism_and_variation():
    for spec in ('backend','data'):
        a=build_practical(spec,'Middle',17)
        assert a==build_practical(spec,'Middle',17)
        assert a!=build_practical(spec,'Junior',17)
        signatures={q['family']:set() for q in a}
        for seed in range(100):
            for q in build_practical(spec,'Middle',seed):signatures[q['family']].add(tuple(q['_correct'].items()))
        assert all(len(x)>10 for x in signatures.values())
