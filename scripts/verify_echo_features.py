#!/usr/bin/env python3
"""Regenerate text features at a local immutable model snapshot, without altering legacy caches."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,hashlib,time,argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/validation/upstream'))
from src.tweet_preprocessing import preprocess_tweet_for_bert
from sentence_transformers import SentenceTransformer
REV='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
SNAP=Path(os.environ.get('ECHO_MODEL_SNAPSHOT', str(Path.home()/'.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots'/REV)))
OUT=ROOT/'experiments/validation/features';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
device='mps' if torch.backends.mps.is_available() else 'cpu'
print('Feature device:',device,flush=True)
model=SentenceTransformer(str(SNAP),device=device,local_files_only=True)
manifest={'device':device,'model_revision':REV,'model_files':{str(p.relative_to(SNAP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SNAP.rglob('*') if p.is_file()},'batch_size':128,'users_per_encoding_block':100,'torch_threads':2,'preprocessing_sha256':hashlib.sha256((ROOT/'experiments/validation/upstream/src/tweet_preprocessing.py').read_bytes()).hexdigest(),'topics':{}}
for topic in ['sxsw','super_bowl','abortion','gun']:
 start=time.time();df=pd.read_feather(ROOT/f'experiments/vendor/echo-chamber-score/data/{topic}/tweets.feather')
 cache_path=ROOT/f'experiments/vendor/echo-chamber-score/data/{topic}/sbert_embeddings.npz'
 cached={}
 if cache_path.exists():
  f=np.load(cache_path,allow_pickle=True);cached=dict(zip(f['user_ids'].astype(str),f['embeddings']))
 rows=[]
 for _,r in df.iterrows():
  texts=[' '.join(t) for tw in r['tweets'] if len(t:=preprocess_tweet_for_bert(tw))>1]
  if texts:rows.append((str(r['user_id']),texts))
 print(topic,'users',len(rows),'tweets',sum(len(t) for _,t in rows),flush=True)
 ids=[];embs=[]
 for start_idx in range(0,len(rows),100):
  block=rows[start_idx:start_idx+100];flat=[t for _,texts in block for t in texts]
  enc=model.encode(flat,batch_size=128,show_progress_bar=False,convert_to_numpy=True)
  offset=0
  for uid,texts in block:
   ids.append(uid);embs.append(enc[offset:offset+len(texts)].mean(axis=0));offset+=len(texts)
  if start_idx%500==0: print(topic,len(ids),'/',len(rows),'seconds',round(time.time()-start),flush=True)
 arr=np.stack(embs);np.savez_compressed(OUT/f'{topic}.npz',user_ids=np.array(ids),embeddings=arr)
 common=[(i,cached[u]) for i,u in enumerate(ids) if u in cached]
 diffs=np.array([np.max(np.abs(arr[i]-v)) for i,v in common])
 cos=np.array([np.dot(arr[i],v)/(np.linalg.norm(arr[i])*np.linalg.norm(v)) for i,v in common])
 manifest['topics'][topic]={'users':len(ids),'encoded_tweets':sum(len(t) for _,t in rows),'raw_tweets_sha256':hashlib.sha256((ROOT/f'experiments/vendor/echo-chamber-score/data/{topic}/tweets.feather').read_bytes()).hexdigest(),'cached_users':len(cached),'same_user_set':set(ids)==set(cached),'max_abs_difference':float(diffs.max()) if len(diffs) else None,'median_max_abs_difference':float(np.median(diffs)) if len(diffs) else None,'minimum_cosine_similarity':float(cos.min()) if len(cos) else None,'seconds':time.time()-start,'feature_sha256':hashlib.sha256((OUT/f'{topic}.npz').read_bytes()).hexdigest()}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(topic,manifest['topics'][topic],flush=True)
