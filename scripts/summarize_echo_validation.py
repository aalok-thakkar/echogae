#!/usr/bin/env python3
"""Summarise completed controlled runs; never treat nested fits as independent graphs."""
from pathlib import Path
from statistics import mean,stdev
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/validation';OUT=BASE/'results'
TOPICS=['gun','abortion','super_bowl','sxsw']
PUBLISHED=[.714,.626,.485,.465];NOTEBOOK=[.627835,.621917,.561706,.464753]
def summary(xs):
 xs=[x for x in xs if x is not None]
 return {'n':len(xs),'mean':mean(xs) if xs else None,'sd':stdev(xs) if len(xs)>1 else None,'min':min(xs) if xs else None,'max':max(xs) if xs else None}
report={'topics':{}}
for topic in TOPICS:
 path=OUT/f'{topic}.json'
 if not path.exists():continue
 d=json.loads(path.read_text());rows=d.get('factorial',[])
 r={'inputs':d['inputs'],'completed_factorial_cells':len(rows),'reference_partition':d['partitions']['42'],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 ref=[v for v in rows if v['partition_seed']==42]
 r['fixed_partition']={key:summary([v[key] for v in ref]) for key in ['user','community']}
 r['averaging_difference']=summary([v['community']-v['user'] for v in ref if v['user'] is not None])
 r['undefined_cells']=sum(v['user'] is None for v in rows)
 if len(rows)==100:
  r['finite_design_variation']={}
  for key in ['user','community']:
   mat=np.array([[next(v[key] for v in rows if v['fit_seed']==f and v['partition_seed']==p) for p in range(42,52)] for f in range(42,52)],dtype=float)
   valid=np.all(np.isfinite(mat),axis=0);A=mat[:,valid];mu=A.mean();fit=A.mean(axis=1,keepdims=True)-mu;part=A.mean(axis=0,keepdims=True)-mu;interaction=A-mu-fit-part
   parts={'fitting':float((fit**2).mean()),'partition':float((part**2).mean()),'interaction':float((interaction**2).mean())};total=float(((A-mu)**2).mean())
   assert abs(sum(parts.values())-total)<1e-12
   r['finite_design_variation'][key]={'valid_partition_seeds':[42+i for i,x in enumerate(valid) if x],'mean':float(mu),'total_mean_square':total,'fractions':{k:v/total for k,v in parts.items()}}
 r['architecture']={key:summary([v[key]-next(x[key] for x in ref if x['fit_seed']==v['fit_seed']) for v in d.get('architecture',[])]) for key in ['user','community']}
 r['structural']={}
 if 'structural' in d:
  for mode in ['closure','mixing']:
   rs=[v for v in d['structural']['runs'] if v['structure']['mode']==mode]
   r['structural'][mode]={'graphs':len(rs),'original':d['structural']['original_structure'],'graph_records':[{'seed':v['structure']['seed'],'structure':v['structure'],'mean_delta_user':mean(f['delta_user'] for f in v['fits']),'mean_delta_community':mean(f['delta_community'] for f in v['fits'])} for v in rs]}
   for key in ['user','community']:
    r['structural'][mode][f'delta_{key}']=summary([mean(f['delta_'+key] for f in v['fits']) for v in rs])
 report['topics'][topic]=r
if len(report['topics'])==4 and all(r['completed_factorial_cells']==100 for r in report['topics'].values()):
 report['group_contrasts']={'published':(PUBLISHED[0]+PUBLISHED[1]-PUBLISHED[2]-PUBLISHED[3])/2,'notebook_stored':(NOTEBOOK[0]+NOTEBOOK[1]-NOTEBOOK[2]-NOTEBOOK[3])/2}
 for key in ['user','community']:
  m=[report['topics'][t]['fixed_partition'][key]['mean'] for t in TOPICS];report['group_contrasts'][key]=(m[0]+m[1]-m[2]-m[3])/2
if 'group_contrasts' in report:
 report['contrast_grid']={}
 datasets={t:json.loads((OUT/f'{t}.json').read_text()) for t in TOPICS}
 paired=[]
 for f in range(42,52):
  for p in range(42,52):
   cells=[next(v for v in datasets[t]['factorial'] if v['fit_seed']==f and v['partition_seed']==p) for t in TOPICS]
   row={'fit_seed':f,'partition_seed':p}
   for key in ['user','community']:
    values=[v[key] for v in cells]
    row[key]=None if any(v is None for v in values) else (values[0]+values[1]-values[2]-values[3])/2
   paired.append(row)
 report['contrast_grid']={'cells':paired,'undefined':sum(v['user'] is None for v in paired),'opposite_signs':sum(v['user']*v['community']<0 for v in paired if v['user'] is not None)}
 for key in ['user','community']:
  vals=[v[key] for v in paired if v[key] is not None]
  report['contrast_grid'][key]={'all_defined':summary(vals),'negative_cells':sum(v<0 for v in vals),'fixed_partition':summary([v[key] for v in paired if v['partition_seed']==42])}
report['complete']=len(report['topics'])==4 and all(r['completed_factorial_cells']==100 and all(r['structural'].get(m,{}).get('graphs')==5 for m in ['closure','mixing']) for r in report['topics'].values())
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
for t,r in report['topics'].items():print(t,r['completed_factorial_cells'],r['fixed_partition'],r['averaging_difference'],flush=True)
print('Complete:',report['complete'])
