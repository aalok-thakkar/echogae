#!/usr/bin/env python3
"""Preserve feature provenance and run checks once the controlled experiment has finished."""
from pathlib import Path
import json,hashlib,re,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/validation'
# Wait only for the declared graph/fit design, never synthesize absent observations.
while True:
 ready=True
 for t in ['gun','abortion','super_bowl','sxsw']:
  p=BASE/f'results/{t}.json'
  if not p.exists():ready=False;break
  d=json.loads(p.read_text())
  if len(d.get('structural',{}).get('runs',[]))!=10:ready=False;break
 if ready:break
 time.sleep(20)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((BASE/'features/manifest.json').read_text())
counts={t:int(n) for t,n in re.findall(r'^(\w+) users \d+ tweets (\d+)$',(BASE/'feature_regeneration.log').read_text(),re.M)}
assert len(counts)==4 and len(m['topics'])==4
m.update({'batch_size':128,'users_per_encoding_block':100,'torch_threads':2,'preprocessing_sha256':sha(BASE/'upstream/src/tweet_preprocessing.py'),'manifest_note':'Encoding counts and input hashes completed from the saved run log and checked raw files after extraction.'})
for t,v in m['topics'].items():
 v['encoded_tweets']=counts[t];v['raw_tweets_sha256']=sha(ROOT/f'experiments/vendor/echo-chamber-score/data/{t}/tweets.feather')
 assert v['feature_sha256']==sha(BASE/f'features/{t}.npz')
(BASE/'results/feature_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
p=BASE/'results/environment.json';e=json.loads(p.read_text());e['hardware']={'cpu':'Apple M3 Max','memory_bytes':38654705664,'feature_device':'Apple MPS GPU','model_fitting_device':'CPU','historical_environment':False};e['current_analysis_script_sha256']=sha(ROOT/'scripts/run_echo_validation.py');e['source_path_note']='Running process initially imported the byte-identical seed-controlled model from the legacy vendor path; current runner imports the included controlled snapshot.';p.write_text(json.dumps(e,indent=2)+'\n')
py=ROOT/'experiments/build/venv_clean/bin/python'
for script in ['check_echo_loader.py','check_echo_training_graphs.py','check_echo_validation.py']:
 print('CHECK',script,flush=True);subprocess.run([str(py),str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
print('SUMMARISE',flush=True);subprocess.run([sys.executable,str(ROOT/'scripts/summarize_echo_validation.py')],cwd=ROOT,check=True)
print('Validation and provenance finalization complete.',flush=True)
