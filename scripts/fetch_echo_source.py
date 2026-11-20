#!/usr/bin/env python3
"""Fetch selected public upstream files at an immutable commit, without data."""
import urllib.request,json,hashlib,subprocess
def fetch(url):
 return subprocess.check_output(["curl","-fsSL","--max-time","60",url])
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
COMMIT='062633017c9ea886b15d099f8370bbef447ecb5c'
BASE='https://raw.githubusercontent.com/faalatawi/echo-chamber-score/'+COMMIT+'/'
OUT=ROOT/'experiments/validation/upstream'
files=['src/EchoGAE.py','src/GAE.py','src/load_data.py','src/echo_chamber_measure.py','src/tweet_preprocessing.py','expt_1_echo_chamber_score.ipynb','environment.yml','LICENSE','README.md']
manifest={'repository':'https://github.com/faalatawi/echo-chamber-score','commit':COMMIT,'files':{}}
for rel in files:
 data=fetch(BASE+rel);p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 local=ROOT/'experiments/vendor/echo-chamber-score'/rel
 manifest['files'][rel]={'sha256':hashlib.sha256(data).hexdigest(),'matches_local':local.read_bytes()==data}
 print(rel,manifest['files'][rel]['matches_local'],flush=True)
tree=json.loads(fetch('https://api.github.com/repos/faalatawi/echo-chamber-score/git/trees/'+COMMIT+'?recursive=1'))
manifest['data_git_blob_matches']={}
for r in tree['tree']:
 rel=r['path'];p=ROOT/'experiments/vendor/echo-chamber-score'/rel
 if rel.startswith('data/') and p.is_file() and r['type']=='blob':
  data=p.read_bytes();h=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  manifest['data_git_blob_matches'][rel]={'git_blob':r['sha'],'matches_local':h==r['sha'],'sha256':hashlib.sha256(data).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
