#!/usr/bin/env python3
"""Record the exact training-edge graphs implied by the released split and saved inputs."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,networkx as nx,torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from run_echo_validation import BASE,load_topic,partition,structure,write
out={}
for topic in ['sxsw','super_bowl','abortion','gun']:
 path=BASE/f'results/{topic}.json'
 if not path.exists():continue
 d=json.loads(path.read_text());G,X,_=load_topic(topic,BASE/'features');y=partition(G,42)
 graphs={'original':G}
 for row in d.get('structural',{}).get('runs',[]):
  st=row['structure'];tag=f"{st['mode']}{st['seed']}";edges=np.load(BASE/f'artifacts/{topic}_{tag}_graph.npz')['edges']
  H=nx.Graph();H.add_nodes_from(G.nodes());H.add_edges_from(edges.tolist());graphs[tag]=H
 out[topic]={}
 for tag,H in graphs.items():
  records=[]
  for seed in range(42,45):
   torch.manual_seed(seed)
   data=Data(x=torch.zeros((len(H),1)),edge_index=torch.tensor(np.array(list(H.edges())).T,dtype=torch.int64))
   split=train_test_split_edges(data);edges=split.train_pos_edge_index.numpy().T
   T=nx.Graph();T.add_nodes_from(H.nodes());T.add_edges_from(edges.tolist())
   records.append({'fit_seed':seed,'training_edges':T.number_of_edges(),'training_graph_hash':hashlib.sha256(np.array(sorted(T.edges()),dtype=np.int64).tobytes()).hexdigest(),**structure(T,y)})
  out[topic][tag]=records
write(BASE/'results/training_graph_diagnostics.json',out)
print('Recorded training-graph diagnostics for',sum(len(d) for d in out.values()),'graph inputs.')
