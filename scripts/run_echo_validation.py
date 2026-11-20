#!/usr/bin/env python3
"""Controlled, resumable EchoGAE validation. See experiments/validation/protocol.md.

Never overwrites historical results. Uses fixed-step rewiring, complete RNG seeding,
explicit graph/feature/source hashes, and stores every fitted representation.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ['CUDA_VISIBLE_DEVICES']=''  # This protocol fixes model fitting to CPU.
import sys,json,time,random,hashlib,platform,subprocess,argparse,warnings
from pathlib import Path
from collections import Counter
import numpy as np,networkx as nx,torch
from sklearn.metrics.pairwise import euclidean_distances
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/validation';OUT=BASE/'results';ART=BASE/'artifacts'
sys.path.insert(0,str(ROOT/'experiments/validation/controlled'))
from src.EchoGAE import EchoGAE_algorithm
from src.echo_chamber_measure import EchoChamberMeasure
warnings.filterwarnings('ignore',category=UserWarning)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
TOPICS=['sxsw','super_bowl','abortion','gun']
SEEDS=list(range(42,52))
def digest(data):return hashlib.sha256(data).hexdigest()
def write(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load_topic(topic,feature_root,full=False):
 raw=ROOT/f'experiments/vendor/echo-chamber-score/data/{topic}/graph.gml'
 G=nx.read_gml(raw)
 if G.is_directed():G=G.to_undirected()
 fp=feature_root/f'{topic}.npz'
 if not fp.exists():raise FileNotFoundError(fp)
 f=np.load(fp,allow_pickle=True);features=dict(zip(f['user_ids'].astype(str),f['embeddings']))
 G=G.subgraph(features).copy()
 if not full:G=G.subgraph(max(nx.connected_components(G),key=len)).copy()
 users=list(G);X=np.stack([features[u] for u in users]);G=nx.convert_node_labels_to_integers(G)
 return G,X,{'raw_graph_sha256':digest(raw.read_bytes()),'feature_sha256':digest(fp.read_bytes()),'ordered_user_hash':digest('\n'.join(users).encode()),'n':len(G),'m':G.number_of_edges(),'self_loops':nx.number_of_selfloops(G),'domain':'full_feature_covered' if full else 'largest_component'}
def partition(G,seed,full=False):
 y=np.full(len(G),-1,dtype=int)
 H=G.subgraph(max(nx.connected_components(G),key=len)).copy() if full else G
 for i,c in enumerate(nx.community.louvain_communities(H,resolution=.05,seed=seed)):
  y[list(c)]=i
 for v in np.where(y<0)[0]:y[v]=y.max()+1
 return y

def scores(D,y):
 labels,counts=np.unique(y,return_counts=True)
 if len(labels)<2:return {'user':None,'community':None,'non_singleton':None,'communities':1,'sizes':counts.tolist(),'reason':'one_community_no_between_distance'}
 avg=np.stack([D[:,y==c].mean(axis=1) for c in labels],axis=1)
 index=np.searchsorted(labels,y);a=avg[np.arange(len(y)),index].copy();avg[np.arange(len(y)),index]=np.inf;b=avg.min(axis=1)
 den=np.maximum(a,b)
 valid=den>0
 s=np.full(len(y),np.nan,dtype=den.dtype);s[valid]=(den[valid]+b[valid]-a[valid])/(2*den[valid])
 # Match undefined native ratios explicitly; never replace 0/0 with zero or 0.5.
 means=[float(s[y==c].mean()) if np.all(valid[y==c]) else None for c in labels]
 non_singleton=[m for m,n in zip(means,counts) if n>1]
 result={'user':float(s.mean()) if np.all(valid) else None,'community':float(np.mean(means)) if all(m is not None for m in means) else None,'non_singleton':float(np.mean(non_singleton)) if non_singleton and all(m is not None for m in non_singleton) else None,'communities':len(labels),'sizes':counts.tolist(),'community_scores':means}
 if not np.all(valid):result.update({'reason':'zero_distance_denominator','zero_denominator_users':int((~valid).sum())})
 return result

def fit(G,X,topic,tag,seed,width=20):
 fp=ART/f'{topic}_{tag}_seed{seed}_h{width}.npz'
 # Edge order affects the released random split, so cache that order too.
 gh=digest(np.array(list(G.edges()),dtype=np.int64).tobytes())
 xh=digest(X.tobytes());source=digest((ROOT/'experiments/validation/controlled/src/EchoGAE.py').read_bytes()+(ROOT/'experiments/validation/controlled/src/GAE.py').read_bytes())
 key=f'{gh}:{xh}:{source}:{seed}:{width}:300'
 if fp.exists():
  f=np.load(fp)
  if str(f['key'])==key:return f['z']
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 t=time.time();z=EchoGAE_algorithm(G,{i:x for i,x in enumerate(X)},show_progress=False,epochs=300,hidden_channels=width,out_channels=width//2,seed=seed)
 fp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(fp,z=z,key=np.array(key))
 print(topic,tag,'seed',seed,'width',width,'fit_seconds',round(time.time()-t,2),flush=True)
 return z

def mixing(G,y):
 edges=[(u,v) for u,v in G.edges() if u!=v]
 return sum(y[u]!=y[v] for u,v in edges)/len(edges)
def structure(G,y):
 return {'triangles':sum(nx.triangles(G).values())//3,'clustering':nx.average_clustering(G),'cross_community_fraction':mixing(G,y),'components':nx.number_connected_components(G),'largest_component':len(max(nx.connected_components(G),key=len))}

def rewire(G,y,seed,mode):
 H=G.copy();edges=[(u,v) for u,v in H.edges() if u!=v];adj={v:set(H[v])-{v} for v in H};rng=np.random.default_rng(seed)
 accepted=0;proposed=0;goal=10*len(edges) if mode=='mixing' else None
 budget=100000 if mode=='closure' else 100*goal
 for proposed in range(1,budget+1):
  i,j=rng.integers(len(edges),size=2)
  if i==j:continue
  a,b=edges[i];c,d=edges[j]
  if rng.random()<.5:c,d=d,c
  if len({a,b,c,d})<4 or d in adj[a] or b in adj[c]:continue
  if mode=='closure':
   if y[a]!=y[c] or y[b]!=y[d]:continue
   before=len(adj[a]&adj[b])+len(adj[c]&adj[d])
   after=len((adj[a]-{b})&(adj[d]-{c}))+len((adj[c]-{d})&(adj[b]-{a}))
   if after>=before:continue
  adj[a].remove(b);adj[b].remove(a);adj[c].remove(d);adj[d].remove(c)
  adj[a].add(d);adj[d].add(a);adj[c].add(b);adj[b].add(c)
  H.remove_edge(a,b);H.remove_edge(c,d);H.add_edge(a,d);H.add_edge(c,b);edges[i]=(a,d);edges[j]=(c,b);accepted+=1
  if goal is not None and accepted>=goal:break
 assert all(G.degree(v)==H.degree(v) for v in G)
 preserved=all(Counter(y[w] for w in G[v])==Counter(y[w] for w in H[v]) for v in G)
 if mode=='closure':assert preserved
 return H,{'seed':seed,'mode':mode,'proposals':proposed,'accepted':accepted,'per_vertex_degree_preserved':True,'per_vertex_community_degree_preserved':preserved,**structure(H,y)}

def run_topic(topic,features,phases):
 G,X,meta=load_topic(topic,features);ys={p:partition(G,p) for p in SEEDS}
 result={'topic':topic,'inputs':meta,'config':{'epochs':300,'lr':.01,'hidden':20,'output':10,'partition_seeds':SEEDS,'fit_seeds':SEEDS,'fully_seeded':True},'partitions':{str(p):{'counts':np.bincount(y).tolist(),'sha256':digest(y.tobytes())} for p,y in ys.items()}}
 dest=OUT/f'{topic}.json'
 if dest.exists():
  old=json.loads(dest.read_text())
  if old.get('inputs')==meta:result=old
 if 'factorial' in phases:
  result['factorial']=[]
  for seed in SEEDS:
   z=fit(G,X,topic,'lcc',seed);D=euclidean_distances(z)
   if seed==42 and len(np.unique(ys[42]))>1:
    native=EchoChamberMeasure(z,ys[42]).echo_chamber_index();ours=scores(D,ys[42])['user'];assert abs(float(native)-ours)<1e-6
    result['native_score_check']={'native':float(native),'efficient':ours,'absolute_difference':abs(float(native)-ours)}
   for p,y in ys.items():result['factorial'].append({'fit_seed':seed,'partition_seed':p,**scores(D,y)})
   write(dest,result)
 if 'controls' in phases:
  result['architecture']=[]
  for seed in range(42,47):
   z=fit(G,X,topic,'lcc',seed,100);result['architecture'].append({'fit_seed':seed,**scores(euclidean_distances(z),ys[42])});write(dest,result)
  FG,FX,fmeta=load_topic(topic,features,True);fy=partition(FG,42,True);result['domain_controls']=[]
  for width in [20,100]:
   z=fit(FG,FX,topic,'full',42,width);result['domain_controls'].append({'hidden':width,'inputs':fmeta,**scores(euclidean_distances(z),fy)})
  write(dest,result)
 if 'structural' in phases:
  y=ys[42];baseline={seed:scores(euclidean_distances(fit(G,X,topic,'lcc',seed)),y) for seed in range(42,45)}
  result['structural']={'original_structure':structure(G,y),'baseline':baseline,'runs':[]}
  for mode in ['closure','mixing']:
   for graph_seed in range(100,105):
    H,stats=rewire(G,y,graph_seed,mode);tag=f'{mode}{graph_seed}'
    np.savez_compressed(ART/f'{topic}_{tag}_graph.npz',edges=np.array(list(H.edges())),labels=y)
    row={'structure':stats,'fits':[]}
    for seed in range(42,45):
     z=fit(H,X,topic,tag,seed);sc=scores(euclidean_distances(z),y);row['fits'].append({'fit_seed':seed,**sc,'delta_user':sc['user']-baseline[seed]['user'],'delta_community':sc['community']-baseline[seed]['community']})
    result['structural']['runs'].append(row);write(dest,result)
 print('COMPLETE',topic,phases,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--topics',nargs='+',default=TOPICS);ap.add_argument('--phases',nargs='+',default=['factorial','controls','structural']);ap.add_argument('--features',type=Path,default=BASE/'features');ap.add_argument('--wait-features',action='store_true');args=ap.parse_args()
 env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'networkx':nx.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),'device':'cpu','script_sha256':digest(Path(__file__).read_bytes())}
 write(OUT/'environment.json',env)
 for t in args.topics:
  while args.wait_features and not (args.features/f'{t}.npz').exists():
   print('Waiting for regenerated features:',t,flush=True);time.sleep(30)
  run_topic(t,args.features,args.phases)
