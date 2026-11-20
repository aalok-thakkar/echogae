#!/usr/bin/env python3
"""Independent clean-environment execution and unmodified-upstream equivalence."""
import sys,json,random,types,importlib
from pathlib import Path
import numpy as np,torch
from run_echo_validation import ROOT,BASE,load_topic,partition,fit,scores,write
from sklearn.metrics.pairwise import euclidean_distances
G,X,meta=load_topic('sxsw',BASE/'features');y=partition(G,42)
z=fit(G,X,'sxsw','cleancheck',42)
reference=np.load(BASE/'artifacts/sxsw_lcc_seed42_h20.npz')['z']
assert np.max(np.abs(z-reference))<1e-6
package=types.ModuleType('untouched_upstream');package.__path__=[str(BASE/'upstream/src')];sys.modules['untouched_upstream']=package
native=importlib.import_module('untouched_upstream.EchoGAE').EchoGAE_algorithm
random.seed(42);np.random.seed(42);torch.manual_seed(42)
z_native=native(G,{i:x for i,x in enumerate(X)},show_progress=False,epochs=300,hidden_channels=20,out_channels=10)
assert np.max(np.abs(z_native-z))<1e-6
out={'isolated_environment':str(Path(sys.prefix).relative_to(ROOT)),'python':sys.version,'max_embedding_difference_from_primary':float(np.max(np.abs(z-reference))),'max_embedding_difference_from_unmodified_upstream':float(np.max(np.abs(z_native-z))),'scores':scores(euclidean_distances(z),y),'inputs':meta,'scope':'Seed42 SXSW 300-epoch reference case, not a rerun of every experimental cell.'}
write(BASE/'results/clean_environment_check.json',out)
print(json.dumps(out,indent=2))
