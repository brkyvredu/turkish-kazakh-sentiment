#!/usr/bin/env python3
"""Run ONE unchanged experiment stage in a newly initialized workspace.
Default is a dry command preview; --execute is required. Never runs later stages
automatically. Paths are taken only from this workspace's saved stage outputs.
"""
from __future__ import annotations
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STAGES=('audit','split','smoke','features','linear','recovery','pilot','xlmr','test','statistics','llm-prepare','qwen-pin','llm-pilot','llm-run','llm-analyze')
GPU_STAGES={'smoke','features','pilot','xlmr','test','llm-pilot','llm-run'}

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def command(stage,work,state,gpu,threads,allow_download=False):
    def out(k):
        if k not in state or not state[k].get('output'):raise ValueError('Missing stage output: '+k)
        p=Path(state[k]['output']).resolve()
        if not p.is_relative_to(work) or not p.is_dir():raise ValueError('Stage output is outside/missing from workspace: '+k)
        return str(p)
    common=['--project',str(work)];cmd=[]
    def base(name):return [sys.executable,str(work/name),*common]
    def need_args(*keys):
        names={'linear':'recovery','xlmr':'xlmr','pilot':'pilot','split':'split','features':'features','smoke':'smoke','test':'test'}
        a=[]
        for key in keys:a.extend(['--'+key,out(names[key])])
        return a
    if stage=='audit':
        cmd=base('01_revision_data_gate.py')+['--preflight',str(work/'_release_inputs/preflight_report.json')]
        if stage in state and state[stage].get('output'):cmd+=['--resume',out(stage)]
    elif stage=='split':
        cmd=base('02_revision_split_gate.py')+['--audit',out('audit')]
        if stage in state and state[stage].get('output'):cmd+=['--resume',out(stage)]
    elif stage=='smoke':cmd=base('03_revision_model_smoke.py')+need_args('split')
    elif stage=='features':cmd=base('04_revision_build_features.py')+need_args('split','smoke')
    elif stage=='linear':cmd=base('05_revision_train_linear.py')+need_args('split','features','smoke')
    elif stage=='recovery':cmd=base('05_revision_train_linear_recovery.py')+need_args('split','features','smoke')+['--resume-from',out('linear')]
    elif stage=='pilot':cmd=base('06_revision_xlmr_pilot.py')+need_args('split','features','smoke','linear')+['--microbatch','4','--precisions','fp32','bf16']
    elif stage=='xlmr':cmd=base('07_revision_train_xlmr.py')+need_args('split','features','smoke','linear','pilot')
    elif stage=='test':cmd=base('08_revision_evaluate_test.py')+need_args('split','features','smoke','linear','pilot','xlmr')
    elif stage=='statistics':cmd=base('09_revision_analyse_predictions.py')+need_args('split','test')
    elif stage=='qwen-pin':
        cmd=[sys.executable,str(ROOT/'tools/prepare_inputs.py'),'--workspace',str(work),'--assets','qwen','--qwen-run',out('llm-prepare')]
        if allow_download:cmd+=['--allow-download']
    else:
        llm_stage={'llm-prepare':'prepare','llm-pilot':'pilot','llm-run':'run','llm-analyze':'analyze'}[stage]
        cmd=base('10_revision_llm_baseline.py')+['--stage',llm_stage]+need_args('split','test')
    if stage in GPU_STAGES:
        if gpu is None:raise ValueError('Pass an explicitly authorized --gpu for this stage.')
        cmd+=['--gpu',str(gpu)]
    if stage in {'features','linear','recovery','pilot','xlmr','test','statistics'}:cmd+=['--threads',str(threads)]
    return cmd

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True);p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--gpu');p.add_argument('--threads',type=int,default=4);p.add_argument('--allow-download',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv);work=Path(a.workspace).expanduser().resolve()
    if a.threads<1:p.error('--threads must be positive')
    if not (work/'REPRODUCTION_WORKSPACE.json').is_file():raise ValueError('Initialize a NEW workspace with tools/prepare_inputs.py first.')
    sf=work/'REPRODUCTION_STAGE_PATHS.json';state=load(sf) if sf.is_file() else {}
    cmd=command(a.stage,work,state,a.gpu,a.threads,a.allow_download)
    print('[COMMAND]',shlex.join(cmd),flush=True)
    if not a.execute:print('[DRY RUN] No stage was started.');return 0
    # Do not broaden inherited device masks; original scripts enforce them.
    env=dict(os.environ);env['HF_HOME']=str(work/'hf_home');env['HF_HUB_CACHE']=str(work/'hf_home/hub');env['TOKENIZERS_PARALLELISM']='false'
    proc=subprocess.Popen(cmd,cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
    output=None
    try:
        for line in proc.stdout:
            print(line,end='',flush=True)
            if line.startswith('[OUTPUT] '):output=line[len('[OUTPUT] '):].strip()
        rc=proc.wait()
    except KeyboardInterrupt:
        # The terminal delivers SIGINT to the original stage as well. Allow it to
        # save its controlled-stop state; do not kill it mid-checkpoint.
        print('\n[WAIT] Waiting for the original stage to finish its controlled stop.',flush=True)
        rc=proc.wait()
    if a.stage=='qwen-pin' and rc==0:output=state['llm-prepare']['output']
    if output:
        q=Path(output).resolve()
        if not q.is_relative_to(work):raise ValueError('Refusing a reported output outside the workspace.')
        state[a.stage]={'output':str(q),'returncode':rc,'command':cmd}
        tmp=sf.with_name(sf.name+'.partial');tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,sf)
    return rc
if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError) as e:print('[STOP]',str(e),file=sys.stderr);raise SystemExit(1)
