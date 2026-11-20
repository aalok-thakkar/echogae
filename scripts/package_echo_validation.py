#!/usr/bin/env python3
"""Create a local anonymous review artifact, without publishing or including user arrays."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/validation';out=ROOT/'paper/build/icwsm_artifact.zip'
r=json.loads((BASE/'results/validation_checks.json').read_text());assert r['fit_arrays']==188 and r['graph_checks']==40
paths=[]
for folder in ['controlled','upstream','results']:
 paths.extend(p for p in (BASE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
paths.extend(BASE/n for n in ['README.md','protocol.md','requirements.in','requirements.lock','seed_parameterization.patch'])
scripts=['fetch_echo_source.py','verify_echo_features.py','run_echo_validation.py','check_echo_loader.py','check_echo_clean_environment.py','check_echo_training_graphs.py','check_echo_validation.py','check_echo_partition_version.py','summarize_echo_validation.py','plot_echo_validation.py','plot_echo_design.py','write_echo_validation_text.py','build_validation_paper.sh']
paths.extend(ROOT/'scripts'/n for n in scripts)
paths.extend(ROOT/'paper'/n for n in ['main.tex','validated_methods.tex','validated_structure.tex','appendix.tex','checklist.tex','references.bib','aaai2026.sty','aaai2026.bst','figures/message_passing_limit.pdf'])
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
readme='''# Anonymous review artifact: Whose Echo Chamber?\n\nThis local package contains the corrected study's source, aggregate results, source/feature manifests, environment lock, and manuscript source. It excludes raw tweets, user-level embeddings, fitted arrays, machine logs, author identities, and earlier exploratory outputs. The original upstream authors remain credited in their source and MIT license. New audit code is supplied for inspection; final public licensing remains an author release decision.\n\nRead `experiments/validation/README.md` for exact reconstruction commands and `protocol.md` for the design and explicitly marked secondary extensions. Existing aggregate results suffice to regenerate charts and manuscript numbers; they do not substitute for running models when reproducing experiments.\n\nFor the manuscript, install NumPy and Matplotlib and make TeX Live's `pdflatex`/`bibtex` and `rg` available, then run `bash scripts/build_validation_paper.sh`. Training requires the pinned raw data and immutable text-model snapshot described in the validation README. The feature-manifest file records the completed original extraction; rerunning extraction writes a new local manifest under the ignored `features/` directory.\n\nThe clean-environment comparison covers one full SXSW reference fit and unmodified upstream execution. It is not claimed to reproduce every run twice or recover the historical publication environment. Structural ranges describe five graph draws, each averaging three matched fitting deltas; they are not confidence intervals.\n\n`artifact_manifest.json` hashes every included input file. This package has not been uploaded or published.\n'''
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
 z.writestr('README.md',readme);z.writestr('artifact_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(out,len(paths),'files',out.stat().st_size,'bytes')
