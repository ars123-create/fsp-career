"""Demo operator aid using only public task statements; no answer-key imports.

Only the six known Junior backend families from the inspected application bank
are supported. SQL is evaluated in an isolated in-memory database; Python
answers are calculated from literal numbers, never executed as submitted code.
"""
import ast
import json
import re
import sqlite3
import sys

q = json.load(sys.stdin)
assert not any(k.startswith('_') or k.startswith('correct') for k in q)
family = q['family']
if family == 'python_iterator_state':
    values = ast.literal_eval(re.search(r'values = (\[.*\])', q['code'])[1])
    count = int(re.search(r'islice\(it, (\d+)\)', q['code'])[1])
    result = dict(prefix=sum(values[:count]), tail=sum(values[count:]))
elif family == 'python_exception_flow':
    values = ast.literal_eval(re.search(r'for value in (\[.*\]):', q['code'])[1])
    result = dict(total=sum(int(x) for x in values if x.isdigit()), visited=len(values))
elif q['skill'] == 'SQL':
    assert family in ('sql_group_report', 'sql_latest_records')
    con = sqlite3.connect(':memory:')
    for table in q['tables']:
        assert all(re.fullmatch(r'\w+', x) for x in [table['name'], *table['columns']])
        columns = ','.join('"'+c+'"' for c in table['columns'])
        con.execute(f'CREATE TABLE "{table["name"]}" ({columns})')
        marks = ','.join('?' for _ in table['columns'])
        con.executemany(f'INSERT INTO "{table["name"]}" VALUES ({marks})', table['rows'])
    assert q['code'].startswith('WITH ')
    cursor = con.execute(q['code'])
    result = dict(zip([c[0] for c in cursor.description], cursor.fetchone()))
elif family == 'api_retry_ledger':
    initial = int(re.search(r'Начальный запас: (\d+)', q['prompt'])[1])
    committed = {}
    for _, key, amount, stage in q['tables'][0]['rows']:
        if stage != 'before_commit':
            committed.setdefault(key, amount)
    result = dict(stock=initial-sum(committed.values()), writes=len(committed))
elif family == 'api_cursor_page':
    size = int(re.search(r'limit=(\d+)', q['code'])[1])
    rows = sorted(q['tables'][0]['rows'], key=lambda r: (r[1],r[0]), reverse=True)[:size]
    result = dict(first_id=rows[0][0], last_id=rows[-1][0])
else:
    raise ValueError(f'Unsupported public task family: {family}')
assert set(result) == {f['name'] for f in q['fields']}
print(json.dumps({k:str(v) for k,v in result.items()}))
