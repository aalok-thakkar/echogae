#!/usr/bin/env python3
"""Compare our prepared inputs against the unmodified pinned upstream loader."""
import sys,types,importlib,json
from pathlib import Path
import numpy as np,pandas as pd
from run_echo_validation import ROOT,BASE,load_topic,partition,write
package=types.ModuleType('loader_upstream');package.__path__=[str(BASE/'upstream/src')];sys.modules['loader_upstream']=package
native=importlib.import_module('loader_upstream.load_data').get_data
out={}
for topic in ['sxsw','super_bowl','abortion','gun']:
 fpath=BASE/f'features/{topic}.npz'
 if not fpath.exists():continue
 folder=BASE/'artifacts/loadercheck'/topic;folder.mkdir(parents=True,exist_ok=True)
 for name in ['graph.gml','tweets.feather','allsides.feather']:
  p=folder/name
  if not p.exists():p.symlink_to(ROOT/f'experiments/vendor/echo-chamber-score/data/{topic}/{name}')
 f=np.load(fpath);pd.DataFrame({'user_id':f['user_ids'].astype(str),'embeddings':list(f['embeddings'])}).to_feather(folder/'embeddings.feather')
 G,X,meta=load_topic(topic,BASE/'features');y=partition(G,42)
 H,e,labels,_,_=native(str(folder)+'/')
 HX=np.stack([e[v] for v in H]);same_edges={tuple(sorted(e)) for e in G.edges()}=={tuple(sorted(e)) for e in H.edges()}
 same_partition=np.array_equal(y[:,None]==y[None,:],labels[:,None]==labels[None,:])
 assert same_edges and np.array_equal(X,HX) and same_partition
 out[topic]={'nodes':len(G),'edges':G.number_of_edges(),'graph_matches':same_edges,'features_match_exactly':True,'partition_matches':bool(same_partition),'sizes':np.bincount(y).tolist()}
 print(topic,out[topic],flush=True)
write(BASE/'results/upstream_loader_check.json',out)
