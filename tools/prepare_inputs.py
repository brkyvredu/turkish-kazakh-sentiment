#!/usr/bin/env python3
"""Release-only input adapter. Download pinned assets only with explicit permission.

This does not train, split, sample, tune, infer, or inspect other user directories.
It installs exact experiment scripts into a NEW reproduction workspace, obtains
pinned upstream inputs, and emits a step01-compatible input inventory. The row
counts in that inventory are explicitly the historical pinned audit's reference
counts, not a newly fetched Hugging Face datasets metadata claim.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(path:Path):return json.loads(path.read_text(encoding='utf-8'))
def atomic(path:Path,data:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)

def install_scripts(work:Path)->None:
    if work==ROOT:raise ValueError('Use a separate reproduction workspace, not the public repository root.')
    if any((work/d).exists() for d in ['_revision_data','_revision_linear','_revision_xlmr','_revision_test']):
        if not (work/'REPRODUCTION_WORKSPACE.json').is_file():
            raise ValueError('Refusing to modify an existing research workspace. Use a NEW directory.')
    work.mkdir(parents=True,exist_ok=True)
    for d in load(ROOT/'metadata/frozen_scripts.json')['scripts']:
        src=ROOT/d['path'];dest=work/d['path']
        if sha(src)!=d['sha256']:raise ValueError('Release script checksum mismatch: '+d['path'])
        if dest.exists():
            if sha(dest)!=d['sha256']:raise ValueError('Existing workspace script differs: '+d['path'])
        else:dest.write_bytes(src.read_bytes())
    marker=work/'REPRODUCTION_WORKSPACE.json'
    if not marker.exists():atomic(marker,{'purpose':'New reproduction; NOT the original paper run','script_manifest_sha256':sha(ROOT/'metadata/frozen_scripts.json')})

def validate_file(path:Path,record:dict)->None:
    if not path.is_file():raise ValueError('Pinned input missing: '+path.name)
    if 'bytes' in record and path.stat().st_size!=record['bytes']:raise ValueError('Pinned input byte size mismatch: '+path.name)
    if 'sha256' in record and sha(path)!=record['sha256']:raise ValueError('Pinned input checksum mismatch: '+path.name)

def selected_inventory(work:Path,assets:dict)->dict:
    hub=work/'hf_home/hub';inventories=[];metadata=[]
    for lang,d in assets['datasets'].items():
        rr=hub/('datasets--'+d['repo_id'].replace('/','--'))
        rel=f"snapshots/{d['revision']}/{d['filename']}";p=rr/rel;validate_file(p,d)
        inventories.append({'kind':'hf_dataset','repo_id':d['repo_id'],'path':str(rr),'files':[{'relative_path':rel,'bytes':p.stat().st_size}]})
        metadata.append({'path':str(rr/'snapshots'/d['revision']/'historical_reference_counts'),
            'split_counts':{'train':{'num_examples':d['raw_rows']}},
            'reference_count_origin':'historical shared data audit for these identical pinned CSV bytes; not fresh datasets metadata'})
    for alias,d in assets['models'].items():
        if alias=='qwen3_8b':continue
        rr=hub/('models--'+d['repo_id'].replace('/','--'));files=[]
        for name,desc in d['files'].items():
            p=rr/'snapshots'/d['revision']/name
            if p.is_file():files.append({'relative_path':str(p.relative_to(rr)),'bytes':p.stat().st_size})
        if files:inventories.append({'kind':'hf_model','repo_id':d['repo_id'],'path':str(rr),'files':files})
    return {'schema_version':'1.0.0','project':str(work),'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'producer':'tools/prepare_inputs.py; new release input adapter, not historical step00 output',
        'scope':'exact selected pinned assets only; no search of user caches',
        'inventories':inventories,'dataset_metadata':metadata,'asset_lock_sha256':sha(ROOT/'metadata/upstream_assets.json')}

def obtain(work:Path,assets:dict,scope:str,allow:bool)->None:
    # Imports occur after the caller's explicit authorization; no installation.
    from huggingface_hub import hf_hub_download
    hub=work/'hf_home/hub';hub.mkdir(parents=True,exist_ok=True)
    for lang,d in assets['datasets'].items():
        p=Path(hf_hub_download(repo_id=d['repo_id'],repo_type='dataset',filename=d['filename'],revision=d['revision'],cache_dir=str(hub),local_files_only=not allow))
        validate_file(p,d)
        print('[VERIFIED DATASET]',lang,d['revision'],flush=True)
    if scope=='supervised':
        for alias,d in assets['models'].items():
            if alias=='qwen3_8b':continue
            for name,desc in d['files'].items():
                p=Path(hf_hub_download(repo_id=d['repo_id'],filename=name,revision=d['revision'],cache_dir=str(hub),local_files_only=not allow));validate_file(p,desc)
            # Original step03's XLM-R resolver reads refs/main. This isolated
            # workspace reference is locked to the paper revision, never latest.
            ref=hub/('models--'+d['repo_id'].replace('/','--'))/'refs/main'
            ref.parent.mkdir(parents=True,exist_ok=True)
            if ref.exists() and ref.read_text().strip()!=d['revision']:raise ValueError('Conflicting isolated cache ref; no overwrite.')
            if not ref.exists():ref.write_text(d['revision']+'\n')
            print('[VERIFIED MODEL]',alias,d['revision'],flush=True)

def pin_qwen(work:Path,assets:dict,run_dir:Path,allow:bool)->None:
    # Run only AFTER original step10 --stage prepare. No corpus is opened here.
    if not run_dir.resolve().is_relative_to(work):raise ValueError('Qwen run must be inside the new workspace.')
    if not (run_dir/'run_identity.json').is_file():raise ValueError('Run step10 --stage prepare first.')
    d=assets['models']['qwen3_8b'];existing=run_dir/'model_manifest.local.json'
    if existing.exists():
        m=load(existing)
        if m['repo_id']!=d['repo_id'] or m['revision']!=d['revision']:raise ValueError('A different Qwen snapshot is already pinned; refusing overwrite.')
        for name,desc in m['files'].items():validate_file(Path(m['snapshot'])/name,desc)
        print('[REUSE] Original Qwen manifest verified.');return
    from huggingface_hub import snapshot_download
    snap=Path(snapshot_download(repo_id=d['repo_id'],revision=d['revision'],cache_dir=str(work/'hf_home/hub'),allow_patterns=d['allow_patterns'],local_files_only=not allow,max_workers=2))
    files=sorted(p for p in snap.iterdir() if p.is_file() and p.suffix in ('.safetensors','.json','.model','.txt'))
    if not any(p.suffix=='.safetensors' for p in files) or not (snap/'config.json').is_file() or not (snap/'tokenizer_config.json').is_file():raise ValueError('Incomplete Qwen snapshot.')
    manifest={'repo_id':d['repo_id'],'revision':d['revision'],'snapshot':str(snap),
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files},
        'download_completed_at':datetime.now(timezone.utc).isoformat(),'license_model_card':'apache-2.0',
        'model_selected_before_additional_LLM_test_scores':True,
        'release_reproduction_note':'Uses the original paper pin, not upstream main. This is a new reproduction manifest; per-file checksums calculated locally.'}
    atomic(existing,manifest);print('[QWEN PINNED]',d['revision'])

def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',required=True);p.add_argument('--assets',choices=('datasets','supervised','qwen'),default='supervised')
    p.add_argument('--qwen-run',help='Existing step10 run directory, inside the workspace')
    p.add_argument('--allow-download',action='store_true',help='Explicit permission for upstream downloads; otherwise cache-only')
    a=p.parse_args(argv);work=Path(a.workspace).expanduser().resolve();assets=load(ROOT/'metadata/upstream_assets.json')
    install_scripts(work)
    if a.assets=='qwen':
        if not a.qwen_run:p.error('--qwen-run is required for qwen')
        pin_qwen(work,assets,Path(a.qwen_run).expanduser().resolve(),a.allow_download)
    else:
        obtain(work,assets,a.assets,a.allow_download)
        out=work/'_release_inputs/preflight_report.json'
        inventory=selected_inventory(work,assets)
        if out.exists():
            old=load(out)
            # Re-running input preparation must not silently change a file that
            # an existing step01 audit hashes. Write a new inventory if different.
            old.pop('created_at_utc',None);new=dict(inventory);new.pop('created_at_utc',None)
            if old!=new:out=out.with_name('preflight_report_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
            else:print('[REUSE INVENTORY]',out);return 0
        atomic(out,inventory);print('[INPUT INVENTORY]',out)
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError) as e:print('[STOP]',str(e),file=sys.stderr);raise SystemExit(1)
