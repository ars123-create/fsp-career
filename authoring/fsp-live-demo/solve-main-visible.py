"""Operator answers computed from public Junior statements, never answer keys."""
import ast,json,re,subprocess,sys
from pathlib import Path
q=json.load(sys.stdin)
assert not any(k.startswith('_') or k.startswith('correct') for k in q)
if q.get('answer_type')=='fields':
 p=subprocess.run([sys.executable,str(Path(__file__).with_name('solve-visible.py'))],input=json.dumps(q),text=True,capture_output=True,check=True)
 print(p.stdout);sys.exit()
f=q['family'];text=q['prompt'];code=q['code'];nums=list(map(int,re.findall(r'\d+',text)))
if f=='api_concurrency': answer=nums[0]+1
elif f=='api_idempotency':answer=nums[0]-nums[1]
elif f in ('python_collections','python_mutability'):
 # Only arithmetic/list operations from inspected public task programs.
 tree=ast.parse(code)
 forbidden=(ast.Import,ast.ImportFrom,ast.With,ast.FunctionDef,ast.ClassDef)
 assert not any(isinstance(n,forbidden) for n in ast.walk(tree))
 assert not any(isinstance(n,ast.Attribute) and n.attr.startswith('_') for n in ast.walk(tree))
 outputs=[]
 exec(compile(tree,'<public-statement>','exec'),{'__builtins__':{'sum':sum,'len':len,'set':set,'min':min,'max':max,'sorted':sorted,'print':lambda x:outputs.append(x)}})
 answer=outputs[-1]
elif f=='sql_join_cardinality':
 users,owners,extra=nums[:3];answer=owners+extra+(users-owners if 'LEFT JOIN' in code else 0)
elif f=='sql_null_distinct':
 values=ast.literal_eval(code.splitlines()[0].split(':',1)[1].strip())
 answer=len(set(v for v in values if v is not None)) if 'DISTINCT' in code else len(values) if 'COUNT(*)' in code else sum(v is not None for v in values)
else:raise ValueError(f)
print(json.dumps({'choice':q['choices'].index(str(answer)),'value':str(answer)}))
