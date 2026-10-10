"""Minimal official stdlib examples, run before the assessment adapter.

Python docs examples: PSF-2.0 / 0BSD; https://docs.python.org/3.14/library/
"""
import json
import sqlite3
import sys
from decimal import Decimal, ROUND_DOWN

assert str(Decimal('7.325').quantize(Decimal('.01'), rounding=ROUND_DOWN)) == '7.32'
assert Decimal('0.1') * 3 == Decimal('0.3')
with sqlite3.connect(':memory:') as db:
    db.execute('CREATE TABLE movie(title, year, score)')
    db.executemany('INSERT INTO movie VALUES(?, ?, ?)', [
        ('Monty Python and the Holy Grail', 1975, 8.2),
        ('And Now for Something Completely Different', 1971, 7.5)])
    assert db.execute('SELECT score FROM movie ORDER BY year').fetchall() == [(7.5,), (8.2,)]
print(json.dumps({'status':'PASS','python':sys.version.split()[0], 'sqlite':sqlite3.sqlite_version,
                  'decimal_round_down':'7.32','sqlite_parameterized_insert_and_select':'PASS'}))
