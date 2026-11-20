#!/usr/bin/env python3
"""Check the complete declared design, saved graph constraints, and score arithmetic."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np,networkx as nx
from run_echo_validation import BASE,ART,load_topic,partition,scores,structure
from sklearn.metrics.pairwise import euclidean_distances
report={'topics':{},'graph_checks':0,'fit_arrays':0,'primary_score_cells':0}
# The original local run used a byte-identical legacy-path copy. New users
# can run solely from the included controlled source; no patched vendor is required.
report['controlled_model_sha256']={filename:hashlib.sha256((BASE/'controlled/src'/filename).read_bytes()).hexdigest() for filename in ['EchoGAE.py','GAE.py']}
manifest=json.loads((BASE/'upstream/manifest.json').read_text())
for name,metadata in manifest['files'].items():
 assert hashlib.sha256((BASE/'upstream'/name).read_bytes()).hexdigest()==metadata['sha256']
report['pinned_source_files_checked']=len(manifest['files'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for t in ['gun','abortion','super_bowl','sxsw']:
 path=BASE/f'results/{t}.json';d=json.loads(path.read_text());G,X,meta=load_topic(t,BASE/'features');assert meta==d['inputs']
 assert len(d['factorial'])==100 and len(d['architecture'])==5 and len(d['domain_controls'])==2 and len(d['structural']['runs'])==10
 assert {(x['fit_seed'],x['partition_seed']) for x in d['factorial']}=={(f,p) for f in range(42,52) for p in range(42,52)}
 maxerror=0
 for f in range(42,52):
  z=np.load(ART/f'{t}_lcc_seed{f}_h20.npz')['z'];D=euclidean_distances(z)
  for p in range(42,52):
   v=next(v for v in d['factorial'] if (v['fit_seed'],v['partition_seed'])==(f,p));y=partition(G,p);s=scores(D,y)
   for key in ['user','community','non_singleton']:
    if s[key] is None:assert v[key] is None
    else: maxerror=max(maxerror,abs(v[key]-s[key]));assert abs(v[key]-s[key])<1e-7
   if s['user'] is not None:
    # Independent arithmetic from community means and sizes.
    weighted=float(np.dot(v['sizes'],v['community_scores'])/sum(v['sizes']));assert abs(weighted-v['user'])<2e-7
    assert abs(np.mean(v['community_scores'])-v['community'])<1e-12
   report['primary_score_cells']+=1
 y=partition(G,42)
 for mode in ['closure','mixing']:
  rows=[v for v in d['structural']['runs'] if v['structure']['mode']==mode];assert {v['structure']['seed'] for v in rows}==set(range(100,105))
  for row in rows:
   st=row['structure'];H=nx.Graph();H.add_nodes_from(G);H.add_edges_from(np.load(ART/f"{t}_{mode}{st['seed']}_graph.npz")['edges'].tolist())
   assert dict(G.degree())==dict(H.degree())
   if mode=='closure':assert all(Counter(y[w] for w in G[v])==Counter(y[w] for w in H[v]) for v in G)
   check=structure(H,y)
   for k,val in check.items():assert abs(val-st[k])<1e-12,(t,mode,k,val,st[k])
   assert {f['fit_seed'] for f in row['fits']}=={42,43,44}
   for f in row['fits']:
    orig=next(v for v in d['factorial'] if v['partition_seed']==42 and v['fit_seed']==f['fit_seed'])
    for key in ['user','community']:assert abs(f['delta_'+key]-(f[key]-orig[key]))<1e-12
   report['graph_checks']+=1
 arrays=[p for p in ART.glob(f'{t}_*_seed*_h*.npz') if any(tag in p.name for tag in ['_lcc_','_full_','_closure','_mixing'])];assert len(arrays)==47,(t,len(arrays))
 for p in arrays:
  a=np.load(p);assert np.isfinite(a['z']).all();assert len(str(a['key']).split(':'))==6
 report['fit_arrays']+=len(arrays)
 report['topics'][t]={'result_sha256':sha(path),'primary_score_max_abs_error':maxerror,'input_verified':True}
assert report['graph_checks']==40 and report['fit_arrays']==188 and report['primary_score_cells']==400
(BASE/'results/validation_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
