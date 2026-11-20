#!/usr/bin/env python3
"""Isolate the original NetworkX version's partitioning effect on fixed current fits."""
import json,time
import networkx as nx,numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from run_echo_validation import BASE,ART,load_topic,partition,scores,digest,write
assert nx.__version__=='3.1',nx.__version__
report={'networkx':nx.__version__,'scope':'Changes partitioning only; uses the same verified ordered inputs and primary fitted representations. Not a historical environment reconstruction.','topics':{}}
for t in ['sxsw','super_bowl','abortion','gun']:
 path=BASE/f'results/{t}.json'
 while not path.exists() or len(json.loads(path.read_text()).get('factorial',[]))!=100:time.sleep(10)
 d=json.loads(path.read_text());G,X,meta=load_topic(t,BASE/'features');assert meta==d['inputs']
 ys={p:partition(G,p) for p in range(42,52)}
 row={'ordered_input_verified':True,'partitions':{str(p):{'counts':np.bincount(y).tolist(),'sha256':digest(y.tobytes()),'matches_primary':digest(y.tobytes())==d['partitions'][str(p)]['sha256']} for p,y in ys.items()},'factorial':[]}
 for seed in range(42,52):
  z=np.load(ART/f'{t}_lcc_seed{seed}_h20.npz')['z'];D=euclidean_distances(z)
  for p,y in ys.items():row['factorial'].append({'fit_seed':seed,'partition_seed':p,**scores(D,y)})
 report['topics'][t]=row;write(BASE/'results/partition_version_check.json',report)
 print(t,row['partitions']['42'],flush=True)
