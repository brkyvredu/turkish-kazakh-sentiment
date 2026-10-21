#!/usr/bin/env python3
"""SAUCIS step07: approved 90-job XLM-R full training, source-validation selection.

FP32, microbatch4/effective32, max_length256, 3 epochs, LR1e-5/2e-5,
seeds42--46; main/binary/domain only. No test labels/predictions, no download,
no automatic numerical-setting changes. Pilot models are NEVER training inputs.

A parent process verifies the existing immutable artifacts and prepares aligned
train/validation arrays. Exactly one child at a time sees the explicitly chosen
physical GPU. Two reusable resume slots belong to THIS run only; all committed
epoch model weights/validation reports and completed candidates are retained.
The previous committed resume slot remains valid while a new slot is written.
Only the two disposable resume slots are rotated; legacy/input files are read-only.

A stopped active job resumes from its last committed optimizer boundary. Models,
AdamW, scheduler, Python/NumPy/CPU/CUDA RNG, epoch permutation identity and cursor
are restored. GPU changes are recorded, not claimed bit-identical. Final-test
evaluation is intentionally a separate later stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

VERSION = '1.0.0'
PILOT_HELPER = ('06_revision_xlmr_pilot.py', '6c8a833b1c3d5175516f966b6004f2511920576b335df470ed5bba60a970a8da')
SEEDS = (42, 43, 44, 45, 46)
FAMILIES = ('main', 'binary', 'domain')
LEARNING_RATES = (1e-5, 2e-5)
SETTINGS = {
    'model': 'FacebookAI/xlm-roberta-base', 'revision': 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089',
    'precision': 'fp32', 'microbatch': 4, 'effective_batch': 32,
    'max_length': 256, 'padding': 'fixed_256', 'epochs': 3,
    'learning_rates': list(LEARNING_RATES), 'seeds': list(SEEDS), 'families': list(FAMILIES),
    'optimizer': {'name': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01,
                  'parameter_groups': 'all_trainable_parameters', 'foreach': False, 'fused': False},
    'warmup_ratio': .1, 'warmup_rounding': 'ceil', 'scheduler': 'linear_warmup_then_linear_decay',
    'max_grad_norm': 1., 'class_weights': 'N_train/(K*class_count_train)',
    'loss_normalization': 'sum_weighted_CE_per_effective_batch/sum_label_weights_per_effective_batch',
    'attention': 'eager', 'TF32': False, 'deterministic_algorithms': True,
    'model_init_seed': 'split_seed, same initialization across both learning rates',
    'training_order': 'PCG64 permutation keyed by seed/context/epoch, independent of model RNG and LR',
    'validation_every': 'epoch', 'validation_microbatch': 4,
    'selection': 'source_macro_f1 or equal_language_mean_macro_f1 as fixed in each recipe',
    'tie_break': 'smaller_learning_rate_then_earlier_epoch_on_exact_score_tie',
    'refit_train_plus_validation': False, 'pilot_checkpoint_used': False,
    'final_test_read': False,
}
_STOP = False


class XLMTrainingError(RuntimeError):
    """Only aggregate, safe diagnostics; exception tracebacks remain PRIVATE."""


class TrainingStopped(XLMTrainingError):
    pass


def need(ok, message):
    if not ok:
        raise XLMTrainingError(message)


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(x):
    return hashlib.sha256(x.encode('utf-8')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, writer, binary=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.partial_{os.getpid()}_{time.time_ns()}')
    with tmp.open('wb' if binary else 'w', **({} if binary else {'encoding': 'utf-8'})) as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # Parent directory fsync makes the rename durable on the intended Linux filesystem.
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, x):
    atomic(path, lambda f: f.write(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n'), False)


def safe(root, relative):
    r = Path(relative)
    need(not r.is_absolute() and '..' not in r.parts, 'Unsafe relative artifact path.')
    p = Path(root) / r
    need(p.resolve().is_relative_to(Path(root).resolve()), 'Artifact escapes this run directory.')
    return p


def descriptor(path):
    p = Path(path)
    return {'bytes': p.stat().st_size, 'sha256': sha(p)}


class Verifier:
    """Re-hash once per invocation/stat identity; never trust persisted mtime alone."""
    def __init__(self):
        self.cache = {}

    def check(self, path, info):
        p = Path(path)
        need(p.is_file(), 'Committed artifact is missing; no automatic deletion/retraining.')
        st = p.stat()
        key = (str(p.resolve()), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        need(st.st_size == info['bytes'], 'Committed artifact size differs.')
        if key not in self.cache:
            self.cache[key] = sha(p)
        need(self.cache[key] == info['sha256'], 'Committed artifact checksum differs; stop and investigate.')


def commit_bundle(folder, signature, names):
    files = {n: descriptor(safe(folder, n)) for n in names}
    need(files, 'Cannot commit an empty bundle.')
    write_json(Path(folder) / 'complete.json', {'signature': signature, 'files': files})


def bundle_ready(folder, signature, verifier=None):
    path = Path(folder) / 'complete.json'
    if not path.exists():
        return False
    d = read_json(path)
    need(d['signature'] == signature and d['files'], 'Completed bundle identity differs.')
    v = verifier or Verifier()
    for n, info in d['files'].items():
        v.check(safe(folder, n), info)
    return True


@contextmanager
def lock(path):
    import fcntl
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise XLMTrainingError('Another writer owns this run. Do not run concurrent commands in the same output.') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def stop_handler(signum, frame):
    global _STOP
    _STOP = True


def stopping(stop_file=None):
    return _STOP or bool(stop_file and Path(stop_file).exists())


def check_stop(stop_file=None):
    if stopping(stop_file):
        raise TrainingStopped('Stop requested. Committed work is retained; resume with the same command.')


def load_pilot(project):
    name, expected = PILOT_HELPER
    path = Path(project) / name
    need(path.is_file() and sha(path) == expected, f'Required reviewed helper missing/changed: {name}.')
    alias = '_saucis_full_xlmr_pilot'
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    need(spec is not None and spec.loader is not None, 'Cannot import the pinned pilot helper.')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def configure_cpu(threads):
    need(threads > 0, 'Positive thread count required.')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(threads)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def audit_pilot(P, L, folder, base, neural_packages, asset):
    folder = Path(folder)
    r = read_json(folder / 'xlmr_pilot_report.json')
    need(r['state'] == 'XLMR_PILOT_PASSED', 'A complete passing XLM-R pilot is required.')
    ident = r['identity']
    need(read_json(folder / 'identity.json') == ident, 'Pilot identity file/report mismatch.')
    for k in ('protocol_id', 'corpus_sha256', 'split_report_sha256', 'feature_report_sha256', 'smoke_report_sha256'):
        need(ident[k] == base[k], 'Pilot/input identity mismatch.')
    need(ident['code_sha256'] == PILOT_HELPER[1] and ident['packages'] == neural_packages, 'Pilot code/environment changed.')
    need(ident['microbatch'] == 4 and ident['asset_sha256'] == asset['file_manifest_sha256'], 'Pilot microbatch/model differs.')
    need(not r['final_test_predictions_created'], 'Pilot unexpectedly reports final-test use.')
    expected = {(c, p) for c in ('main:joint', 'binary:joint') for p in ('fp32', 'bf16')}
    need({(s['context'], s['precision']) for s in r['stages']} == expected and len(r['stages']) == 4, 'Incomplete pilot coverage.')
    for s in r['stages']:
        need(s['state'] == 'PASS' and s['replay']['same_device_exact_replay'], 'Pilot replay did not pass.')
        for key in ('model', 'optimizer', 'scheduler', 'rng'):
            need(s['reference']['hashes'][key] == s['replay']['hashes'][key], 'Pilot state hashes differ.')
        need(s['reference']['validation_logits_sha256'] == s['replay']['validation_logits_sha256'], 'Pilot logits differ.')
        scope = folder / (L.safe_name(s['context']) + '_' + s['precision'])
        plan = read_json(scope / 'pilot_plan.local.json')
        need(plan['pilot_identity'] == ident, 'Pilot scope identity differs.')
        for mode in ('reference', 'replay'):
            need(P.phase_done(scope, mode, plan['signature']), 'Missing/corrupt committed pilot phase.')
            need(read_json(scope / (mode + '_report.json')) == s[mode], 'Pilot aggregate/phase report differs.')
    return r


def inspect_environment(project, split, features, smoke, linear, pilot, hub_cache):
    P = load_pilot(project)
    L, F, S = (P.load_helper(project, a) for a in ('linear', 'features', 'smoke'))
    base, fr, contexts = L.inspect_inputs(split, features, smoke)
    nbase, stages = F.inspect_inputs(split, smoke)
    la = P.audit_linear_directory(linear, base)
    snapshot, asset = F.verify_asset(stages['xlmr'], F.hub_root(hub_cache), 'xlmr_tokens')
    need(asset['revision'] == SETTINGS['revision'], 'Unapproved XLM-R snapshot; never substitute a different revision.')
    bankdir = L.safe_path(features, fr['banks']['xlmr_tokens']['bank_directory'])
    bank = F.FeatureBank(bankdir)
    need(bank.spec['kind'] == 'xlmr_token_ids' and bank.spec['alias'] == 'xlmr_tokens', 'Unexpected token bank.')
    need(bank.spec['protocol_id'] == base['protocol_id'] and bank.spec['corpus_sha256'] == base['corpus_sha256'], 'Token corpus identity mismatch.')
    need(bank.spec['max_length'] == 256 and not bank.spec['labels_read'] and not bank.spec['fit_on_corpus'], 'Token preparation settings changed.')
    need(bank.spec['revision'] == asset['revision'] and bank.spec['asset_sha256'] == asset['file_manifest_sha256'], 'Token/model asset mismatch.')
    need(digest(canonical(bank.spec)) == fr['banks']['xlmr_tokens']['spec_sha256'], 'Token-bank manifest fingerprint differs.')
    pr = audit_pilot(P, L, pilot, base, nbase['packages'], asset)
    need(pr['identity']['linear_report_sha256'] == la['report_sha256'], 'Pilot used a different linear training report.')
    cfg = read_json(snapshot / 'config.json')
    pad = cfg.get('pad_token_id')
    need(isinstance(pad, int) and pad >= 0 and cfg.get('model_type') == 'xlm-roberta', 'Unexpected pinned model config.')
    return P, L, F, S, base, fr, contexts, asset, nbase['packages'], pr, pad, bank


def job_key(seed, context_id, lr):
    need(lr in LEARNING_RATES, 'Unapproved learning rate.')
    return f'seed_{seed}/' + re_safe(context_id) + ('/lr_1e-5' if lr == 1e-5 else '/lr_2e-5')


def re_safe(text):
    import re
    return re.sub('[^a-zA-Z0-9_.-]', '_', text) + '_' + digest(text)[:8]


def make_jobs(P, L, split, contexts, root_spec):
    rows = P.future_jobs(L, split, contexts)
    need({r['seed'] for r in rows} == set(SEEDS), 'Five approved seeds required.')
    root_sig = digest(canonical(root_spec))
    out = []
    for row in rows:
        seed, context_id, lr = row['seed'], row['context_id'], row['learning_rate']
        ctx = contexts[seed][context_id]
        tr = P.set_metadata(L, split, seed, ctx['train_set'])
        va = P.set_metadata(L, split, seed, ctx['validation_set'])
        j = dict(row, context=ctx, train_manifest=tr, validation_manifest=va,
                 root_spec_sha256=root_sig, key=job_key(seed, context_id, lr))
        j['signature'] = digest(canonical(j))
        out.append(j)
    need(len(out) == 90 and len({j['key'] for j in out}) == 90, 'Invalid/duplicate 90-job plan.')
    return out


def context_folder(root, seed, context):
    return Path(root) / 'development' / f'seed_{seed}' / re_safe(context)


def expected_data_spec(job, token_spec_sha, pad):
    return {'schema': VERSION, 'root_spec_sha256': job['root_spec_sha256'], 'seed': job['seed'],
            'context': job['context'], 'train': job['train_manifest'], 'validation': job['validation_manifest'],
            'token_spec_sha256': token_spec_sha, 'max_length': 256, 'pad_token_id': pad,
            'test_read': False, 'model_text_changed': False}


def prepare_data(P, L, bank, split, root, job, pad, verifier):
    import numpy as np
    folder = context_folder(root, job['seed'], job['context_id'])
    spec = expected_data_spec(job, digest(canonical(bank.spec)), pad)
    sig = digest(canonical(spec))
    if bundle_ready(folder, sig, verifier):
        need(read_json(folder / 'data_spec.json') == spec, 'Saved development data spec differs.')
        return folder, read_json(folder / 'data_report.json')
    check_stop()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'data_spec.json').exists():
        need(read_json(folder / 'data_spec.json') == spec, 'Partial development data uses different inputs.')
    tr = L.load_development(split, job['seed'], job['context']['train_set'])
    va = L.load_development(split, job['seed'], job['context']['validation_set'])
    L.validate_pair(tr, va, job['context'])
    need(len(tr.targets) == job['train_rows'] and len(va.targets) == job['validation_rows'], 'Development count differs.')
    names = ['data_spec.json', 'data_report.json']
    profiles = {}
    for label, part in (('train', tr), ('validation', va)):
        check_stop()
        data = P.aligned_tokens(bank, part, np.arange(len(part.targets), dtype=np.int64))
        ids, mask = P.padded(data['token_ids'], pad, 256)
        arr = {'input_ids': ids.astype(np.int32), 'attention_mask': mask.astype(np.uint8),
               'labels': part.targets.astype(np.int64), 'row_ids': part.row_ids,
               'sample_ids': part.sample_ids.astype('S64'), 'group_ids': part.group_ids.astype('S64'),
               'languages': part.languages, 'domains': part.domains, 'token_lengths': data['token_lengths']}
        for key, value in arr.items():
            name = f'{label}_{key}.PRIVATE.npy'
            atomic(folder / name, lambda f, a=value: np.save(f, a, allow_pickle=False))
            names.append(name)
        profiles[label] = {'rows': len(part.targets), 'class_counts': dict(sorted(Counter(map(int, part.targets)).items())),
                           'languages': dict(Counter(map(str, part.languages))), 'truncated': int((data['token_lengths'] > 256).sum())}
        del data, ids, mask, arr
    labels = L.task_labels(tr.task)
    counts = Counter(map(int, tr.targets))
    weights = [len(tr.targets) / (len(labels) * counts[i]) for i in labels]
    r = {'state': 'DEVELOPMENT_READY', 'signature': sig, 'task': tr.task, 'num_labels': len(labels),
         'class_weights': weights, 'class_weights_computed_from': 'train_only',
         'context_id': job['context_id'], 'seed': job['seed'], 'profiles': profiles,
         'validation_languages': job['context']['validation_languages'], 'final_test_read': False}
    write_json(folder / 'data_spec.json', spec)
    write_json(folder / 'data_report.json', r)
    commit_bundle(folder, sig, names)
    need(bundle_ready(folder, sig, verifier), 'Development commit verification failed.')
    del tr, va
    gc.collect()
    return folder, r


def load_data(folder, signature):
    import numpy as np
    folder = Path(folder)
    need(bundle_ready(folder, signature), 'Missing committed development arrays.')
    r = read_json(folder / 'data_report.json')
    need(r['signature'] == signature and not r['final_test_read'], 'Development report differs.')
    out = {'report': r, 'signature': signature}
    for label in ('train', 'validation'):
        names = ('input_ids', 'attention_mask', 'labels', 'row_ids', 'sample_ids', 'group_ids', 'languages', 'domains', 'token_lengths')
        part = {k: np.load(folder / f'{label}_{k}.PRIVATE.npy', mmap_mode='r', allow_pickle=False) for k in names}
        n = r['profiles'][label]['rows']
        need(part['input_ids'].shape == (n, 256) and part['attention_mask'].shape == (n, 256), 'Padded-array shape differs.')
        need(all(len(a) == n for a in part.values()) and len(np.unique(part['row_ids'])) == n, 'Invalid development array identities.')
        need(set(map(int, part['labels'])) == set(range(r['num_labels'])), 'Task class coverage differs.')
        need(set(map(str, part['languages'])) == set(r['validation_languages']), 'Unauthorized source-language data.')
        need(np.all(part['attention_mask'] <= 1) and np.all(part['attention_mask'].sum(1) > 0), 'Invalid attention masks.')
        out[label] = part
    need(set(out['train']['group_ids']).isdisjoint(set(out['validation']['group_ids'])), 'Development text groups overlap.')
    need(set(out['train']['row_ids']).isdisjoint(set(out['validation']['row_ids'])), 'Development rows overlap.')
    counts = Counter(map(int, out['train']['labels']))
    expected = [len(out['train']['labels']) / (r['num_labels'] * counts[i]) for i in range(r['num_labels'])]
    need(np.allclose(r['class_weights'], expected, rtol=0, atol=1e-12), 'Class weights not training-frequency weights.')
    return out


def permutation(n, seed, context, epoch):
    import numpy as np
    # Independent of LR, validation calls and the global RNG used for dropout.
    key = int(digest(f'xlmr-order-v1:{seed}:{context}:{epoch}')[:16], 16)
    return np.random.Generator(np.random.PCG64(key)).permutation(n).astype(np.int64)


def array_sha(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def schedule_scale(step, total, warmup):
    need(total > 0 and 0 <= warmup <= total and step >= 0, 'Invalid LR schedule inputs.')
    if step < warmup:
        return float(step) / max(1, warmup)
    return max(0., float(total - step) / max(1, total - warmup))


def make_optimizer(model, lr, total_steps):
    import torch
    need(lr in LEARNING_RATES, 'Unapproved learning rate.')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .999), eps=1e-8,
                           weight_decay=.01, foreach=False, fused=False)
    warmup = math.ceil(.1 * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: schedule_scale(step, total_steps, warmup))
    return opt, sched


def tensor_bytes(x):
    import numpy as np
    import torch
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, np.ndarray):
        return x.nbytes
    if isinstance(x, dict):
        return sum(tensor_bytes(v) for v in x.values())
    if isinstance(x, (tuple, list)):
        return sum(tensor_bytes(v) for v in x)
    return 0


def check_space(folder, payload):
    # No scope reduction or deletion. The previous committed pointer is not touched.
    required = math.ceil(tensor_bytes(payload) * 1.08) + 32 * 1024 * 1024
    free = shutil.disk_usage(folder).free
    need(free >= required, f'Insufficient space for atomic checkpoint: need approximately {required // 2**20} MiB free, found {free // 2**20} MiB. Free space then resume; no files deleted.')


class ResumeStore:
    """Two bounded, replaceable run-local working slots and an atomic committed pointer.

    Completed epoch weights are stored separately and never overwritten. Switching
    to another candidate is allowed only after the old candidate is complete.
    """
    def __init__(self, root, run_signature, P):
        self.root = Path(root)
        self.folder = self.root / 'resume'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.run_signature = run_signature
        self.P = P

    def pointer(self):
        p = self.folder / 'latest.json'
        if not p.exists():
            return None
        d = read_json(p)
        need(d['run_signature'] == self.run_signature and d['slot'] in (0, 1), 'Resume pointer belongs to another run.')
        need(d['file'] == f'working_slot_{d["slot"]}.PRIVATE.pt', 'Unexpected resume slot name.')
        return d

    def verify(self, d):
        Verifier().check(safe(self.folder, d['file']), d)

    def save(self, state, job, reason, gpu):
        import torch
        old = self.pointer()
        progress_key = {k: state['progress'][k] for k in ('epoch', 'cursor', 'global_step')}
        if old and old['job_signature'] == job['signature'] and old['progress'] == progress_key:
            self.verify(old)
            return old
        if old and old['job_signature'] != job['signature']:
            need((safe(self.root / 'jobs', old['job_key']) / 'complete.json').is_file(), 'An unfinished different job owns the resume cache; finish/resume it first.')
        slot = 0 if old is None else 1 - old['slot']
        payload = dict(state, run_signature=self.run_signature, job_signature=job['signature'], job_key=job['key'])
        cpu = self.P.cpu_tree(payload)
        check_space(self.folder, cpu)
        started = time.monotonic()
        path = self.folder / f'working_slot_{slot}.PRIVATE.pt'
        atomic(path, lambda f: torch.save(cpu, f))
        d = dict(descriptor(path), file=path.name, slot=slot,
                 generation=(old['generation'] + 1 if old else 1),
                 run_signature=self.run_signature, job_signature=job['signature'], job_key=job['key'],
                 progress=progress_key, gpu_uuid=gpu['uuid'], saved_at=now(),
                 save_seconds=time.monotonic() - started, reason=reason)
        write_json(self.folder / 'latest.json', d)
        # Lightweight provenance history; weights from old working slots need not persist.
        hp = self.folder / 'checkpoint_history.json'
        history = read_json(hp) if hp.exists() else []
        history = [h for h in history if h['generation'] < d['generation']]
        history.append(d)
        write_json(hp, history)
        del cpu
        return d

    def load(self, job):
        import torch
        d = self.pointer()
        if d is None or d['job_signature'] != job['signature']:
            return None
        self.verify(d)
        # Never use this for arbitrary checkpoints from outside this run.
        state = torch.load(safe(self.folder, d['file']), map_location='cpu', weights_only=False)
        need(state['run_signature'] == self.run_signature and state['job_signature'] == job['signature'] and state['job_key'] == job['key'], 'Resume state identity mismatch.')
        need({k: state['progress'][k] for k in ('epoch', 'cursor', 'global_step')} == d['progress'], 'Resume cursor differs from committed pointer.')
        return state, d


def weights_ready(folder, signature, verifier=None):
    p = Path(folder) / 'weights.ready.json'
    if not p.exists():
        return False
    d = read_json(p)
    need(d['signature'] == signature, 'Epoch model signature differs.')
    (verifier or Verifier()).check(Path(folder) / 'weights.PRIVATE.pt', d)
    return True


def save_epoch_weights(P, folder, signature, model):
    import torch
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if weights_ready(folder, signature):
        return read_json(folder / 'weights.ready.json')
    payload = {'signature': signature, 'model': P.cpu_tree(model.state_dict())}
    check_space(folder, payload)
    path = folder / 'weights.PRIVATE.pt'
    atomic(path, lambda f: torch.save(payload, f))
    meta = dict(descriptor(path), signature=signature, state='EPOCH_WEIGHTS_READY')
    write_json(folder / 'weights.ready.json', meta)
    return meta


def validation_epoch(P, L, model, data, context, folder, signature, device, stop_file=None):
    """Resumable validation shards; fixed microbatch4, no final-test data.

    Restore training RNG even if validation is interrupted. The model is held fixed
    at an epoch-end training checkpoint while all validation shards are computed.
    """
    import numpy as np
    import torch
    folder = Path(folder) / 'validation'
    folder.mkdir(parents=True, exist_ok=True)
    v = data['validation']
    identity = {'epoch_signature': signature, 'data_signature': data['signature'],
                'rows': len(v['labels']), 'microbatch': 4, 'shard_rows': 512,
                'selection_rule': context['selection_metric'], 'source_languages': context['validation_languages']}
    sig = digest(canonical(identity))
    if bundle_ready(folder, sig):
        r = read_json(folder / 'validation_report.json')
        need(r['signature'] == sig, 'Saved validation identity differs.')
        return r
    if (folder / 'identity.json').exists():
        need(read_json(folder / 'identity.json') == identity, 'Partial validation inputs differ.')
    else:
        write_json(folder / 'identity.json', identity)
    before_rng = P.rng_state(torch)
    descriptors = []
    logits_parts = []
    seconds = 0.
    try:
        for number, lo in enumerate(range(0, len(v['labels']), 512)):
            check_stop(stop_file)
            hi = min(lo + 512, len(v['labels']))
            name = f'shard_{number:05d}.PRIVATE.npz'
            pointer = folder / f'shard_{number:05d}.json'
            if pointer.exists():
                meta = read_json(pointer)
                need(meta['signature'] == sig and meta['lo'] == lo and meta['hi'] == hi, 'Validation shard identity differs.')
                Verifier().check(folder / name, meta)
                with np.load(folder / name, allow_pickle=False) as f:
                    need(set(f.files) == {'row_ids', 'sample_ids', 'group_ids', 'y_true', 'logits'}, 'Unexpected validation shard fields.')
                    for field, key in (('row_ids', 'row_ids'), ('sample_ids', 'sample_ids'), ('group_ids', 'group_ids'), ('y_true', 'labels')):
                        need(np.array_equal(f[field], v[key][lo:hi]), 'Validation shard membership/labels differ.')
                    logits = f['logits'].copy()
            else:
                logits, elapsed = P.forward_probe(model, np.array(v['input_ids'][lo:hi], copy=True),
                                                np.array(v['attention_mask'][lo:hi], copy=True), 4, 'fp32', device)
                check_stop(stop_file)  # Uncommitted active shard can be recomputed; epoch training is already saved.
                arr = dict(row_ids=v['row_ids'][lo:hi], sample_ids=v['sample_ids'][lo:hi],
                           group_ids=v['group_ids'][lo:hi], y_true=v['labels'][lo:hi], logits=logits)
                atomic(folder / name, lambda f: np.savez(f, **arr))
                meta = dict(descriptor(folder / name), signature=sig, lo=lo, hi=hi, file=name,
                            forward_seconds=elapsed, created_at=now())
                write_json(pointer, meta)
            need(logits.shape == (hi - lo, data['report']['num_labels']) and np.isfinite(logits).all(), 'Nonfinite/wrong-shape validation logits.')
            logits_parts.append(logits)
            descriptors.append(meta)
            seconds += meta['forward_seconds']
            if number % 10 == 0 or hi == len(v['labels']):
                print(f'[VALIDATE] {context["context_id"]}: {hi:,}/{len(v["labels"]):,}', flush=True)
        logits = np.concatenate(logits_parts)
        pred = logits.argmax(1).astype(np.int64)
        val = SimpleNamespace(task=data['report']['task'], targets=v['labels'], languages=v['languages'])
        metrics = L.validation_metrics(val, pred, context)
        r = {'state': 'VALIDATION_READY', 'signature': sig, 'validation': metrics,
             'rows': len(pred), 'shards': descriptors, 'forward_seconds': seconds,
             'final_test_read': False, 'prediction_semantics': 'argmax FP32 logits; no probability calibration'}
        write_json(folder / 'validation_report.json', r)
        names = ['identity.json', 'validation_report.json']
        for number, meta in enumerate(descriptors):
            names += [meta['file'], f'shard_{number:05d}.json']
        commit_bundle(folder, sig, names)
        return r
    finally:
        P.restore_rng(torch, before_rng)


def epoch_signature(job, epoch, data_signature):
    return digest(canonical({'job_signature': job['signature'], 'epoch': epoch,
                             'data_signature': data_signature, 'settings': SETTINGS}))


def check_epoch(folder, job, epoch, data_signature, verifier=None):
    folder = Path(folder)
    sig = epoch_signature(job, epoch, data_signature)
    if not bundle_ready(folder, sig, verifier):
        return None
    need(weights_ready(folder, sig, verifier), 'Completed epoch weights missing.')
    r = read_json(folder / 'epoch_report.json')
    need(r['signature'] == sig and r['epoch'] == epoch and r['job_signature'] == job['signature'], 'Epoch report identity differs.')
    need(math.isfinite(r['validation']['selection_score']) and not r['final_test_read'], 'Invalid epoch validation report.')
    vr = read_json(folder / 'validation' / 'validation_report.json')
    need(bundle_ready(folder / 'validation', vr['signature'], verifier), 'Completed epoch validation payload missing.')
    need(r['validation'] == vr['validation'], 'Epoch/validation metrics differ.')
    return r


def candidate_best(epochs):
    need(len(epochs) == 3 and {r['epoch'] for r in epochs} == {1, 2, 3}, 'Selection requires all three epochs.')
    need(all(math.isfinite(r['validation']['selection_score']) for r in epochs), 'Nonfinite candidate epoch score.')
    return min(epochs, key=lambda r: (-r['validation']['selection_score'], r['epoch']))


def select_context(candidates):
    need(len(candidates) == 2 and {r['learning_rate'] for r in candidates} == set(LEARNING_RATES), 'Context selection requires both learning rates.')
    need(all(r['state'] == 'XLMR_CANDIDATE_READY' and len(r['epochs']) == 3 for r in candidates), 'Incomplete XLM candidate.')
    seed = candidates[0]['seed']
    context = candidates[0]['context_id']
    need(all(r['seed'] == seed and r['context_id'] == context for r in candidates), 'Mixed contexts in model selection.')
    choices = [(r, e) for r in candidates for e in r['epochs']]
    r, e = min(choices, key=lambda pair: (-pair[1]['validation']['selection_score'], pair[0]['learning_rate'], pair[1]['epoch']))
    return {'state': 'VALIDATION_SELECTED', 'seed': seed, 'context_id': context, 'task': r['task'],
            'learning_rate': r['learning_rate'], 'epoch': e['epoch'],
            'selection_score': e['validation']['selection_score'], 'validation': e['validation'],
            'selection_rule': r['selection_rule'], 'validation_languages': r['validation_languages'],
            'selected_job_key': r['key'], 'selected_model_relative_path': f'jobs/{r["key"]}/epoch_{e["epoch"]}/weights.PRIVATE.pt',
            'selected_model_sha256': e['weights']['sha256'], 'selected_epoch_signature': e['signature'],
            'tie_break': SETTINGS['tie_break'], 'both_learning_rates_completed': True,
            'refit_train_plus_validation': False, 'final_test_used_for_selection': False,
            'evaluation_recipe_ids': r['evaluation_recipe_ids']}


def initial_progress():
    return {'epoch': 0, 'cursor': 0, 'global_step': 0, 'order_sha256': None, 'updates': []}


def assert_progress(p, n, job):
    steps_epoch = math.ceil(n / 32)
    need(isinstance(p['epoch'], int) and 0 <= p['epoch'] <= 3, 'Invalid checkpoint epoch.')
    need(0 <= p['cursor'] <= n and (p['cursor'] == n or p['cursor'] % 32 == 0), 'Checkpoint is not at an optimizer boundary.')
    need(p['epoch'] < 3 or p['cursor'] == 0, 'Final progress cursor must be zero.')
    expected = p['epoch'] * steps_epoch + math.ceil(p['cursor'] / 32)
    need(expected == p['global_step'] and len(p['updates']) == p['global_step'], 'Checkpoint update/cursor mismatch.')
    if p['epoch'] < 3:
        perm = permutation(n, job['seed'], job['context_id'], p['epoch'])
        need(p['order_sha256'] in (None, array_sha(perm)), 'Checkpoint epoch-order mismatch.')
    for i, u in enumerate(p['updates']):
        need(u['global_step'] == i + 1 and u['rows'] > 0 and math.isfinite(u['loss']), 'Invalid committed optimizer trace.')


def engine(P, L, model_factory, data, root, job, device, gpu, checkpoint_every=500, stop_file=None, after_step=None):
    """Same engine is used by real CUDA jobs and independent-process CPU regression tests."""
    import numpy as np
    import torch
    root = Path(root)
    folder = safe(root / 'jobs', job['key'])
    folder.mkdir(parents=True, exist_ok=True)
    dsignature = data['signature']
    if (folder / 'job_spec.json').exists():
        need(read_json(folder / 'job_spec.json') == job, 'Existing candidate spec differs.')
    else:
        write_json(folder / 'job_spec.json', job)
    n = len(data['train']['labels'])
    need(n == job['train_rows'] and 3 * math.ceil(n / 32) == job['optimizer_steps'], 'Training-step plan differs.')
    store = ResumeStore(root, job['root_spec_sha256'], P)
    check_stop(stop_file)
    model = model_factory()
    need(all(p.requires_grad for p in model.parameters()), 'Frozen model parameter in full fine-tuning.')
    opt, sched = make_optimizer(model, job['learning_rate'], job['optimizer_steps'])
    p = initial_progress()
    init_sha = P.tree_sha(model.state_dict())
    resume = store.load(job)
    previous_device = None
    if resume:
        state, pointer = resume
        need(state['data_signature'] == dsignature and state['settings'] == SETTINGS, 'Resume training/data settings differ.')
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        p, init_sha = state['progress'], state['initial_model_sha256']
        assert_progress(p, n, job)
        need(sched.last_epoch == p['global_step'], 'Scheduler position differs from global optimizer step.')
        expected_lr = job['learning_rate'] * schedule_scale(p['global_step'], job['optimizer_steps'], math.ceil(.1 * job['optimizer_steps']))
        need(all(abs(g['lr'] - expected_lr) < 1e-15 for g in opt.param_groups), 'Restored optimizer learning rate differs from the approved schedule.')
        P.restore_rng(torch, state['rng'])
        previous_device = pointer['gpu_uuid']
        print(f'[RESUME] {job["key"]}: epoch={p["epoch"] + 1} cursor={p["cursor"]} step={p["global_step"]}', flush=True)
        del state
    else:
        # Noncompleted old progress must never silently restart from the base model.
        old_epoch = list(folder.glob('epoch_*/weights.ready.json'))
        need(not old_epoch, 'Epoch artifacts exist but no matching committed resume state; investigate instead of restarting.')
    write_json(folder / 'initialization.json', {'initial_model_sha256': init_sha, 'seed': job['seed'],
                                              'pretrained_revision': SETTINGS['revision'], 'pilot_checkpoint_used': False})
    inv_id = f'{os.getpid()}_{time.time_ns()}'
    invocation = {'started_at': now(), 'gpu': gpu, 'device': device, 'job_signature': job['signature'],
                  'resumed': resume is not None, 'start_global_step': p['global_step'],
                  'previous_checkpoint_gpu_uuid': previous_device,
                  'cross_gpu_resume': bool(previous_device and previous_device != gpu['uuid']),
                  'cross_gpu_bitwise_equivalence_claimed': False}
    write_json(folder / 'executions' / f'{inv_id}.json', invocation)
    if invocation['cross_gpu_resume']:
        print('[DEVICE CHANGE] Optimizer state restored on requested GPU; cross-GPU bitwise identity is not claimed.', flush=True)
    def snapshot(reason):
        state = {'settings': SETTINGS, 'data_signature': dsignature, 'initial_model_sha256': init_sha,
                 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                 'rng': P.rng_state(torch), 'progress': p}
        rec = store.save(state, job, reason, gpu)
        print(f'[CHECKPOINT] {job["key"]} step={p["global_step"]} epoch={p["epoch"] + 1} reason={reason}', flush=True)
        return rec
    try:
        while p['epoch'] < 3:
            epoch0 = p['epoch']
            order = permutation(n, job['seed'], job['context_id'], epoch0)
            p['order_sha256'] = array_sha(order)
            while p['cursor'] < n:
                if stopping(stop_file):
                    snapshot('requested_stop')
                    raise TrainingStopped('Stopped at optimizer boundary; checkpoint committed.')
                hi = min(p['cursor'] + 32, n)
                ix = order[p['cursor']:hi]
                tr = data['train']
                d = P.training_step(model, opt, sched, tr['input_ids'][ix], tr['attention_mask'][ix],
                                    tr['labels'][ix], data['report']['class_weights'], 4, 'fp32', device)
                p['cursor'] = hi
                p['global_step'] += 1
                d.update(global_step=p['global_step'], epoch=epoch0 + 1, cursor=hi, gpu_uuid=gpu['uuid'])
                p['updates'].append(d)
                need(sched.last_epoch == p['global_step'], 'Scheduler did not advance exactly once per optimizer update.')
                if p['global_step'] % 100 == 0 or hi == n:
                    print(f'[STEP] {job["key"]} epoch={epoch0 + 1}/3 rows={hi:,}/{n:,} loss={d["loss"]:.6f} lr={d["learning_rate_used"]:.9g}', flush=True)
                    write_json(folder / 'status.json', {'state': 'TRAINING', 'epoch': epoch0 + 1, 'cursor': hi,
                               'global_step': p['global_step'], 'total_steps': job['optimizer_steps'], 'last_loss': d['loss']})
                if after_step:
                    after_step(p, job)
                if stopping(stop_file):
                    snapshot('requested_stop')
                    raise TrainingStopped('Stopped at optimizer boundary; checkpoint committed.')
                if p['global_step'] % checkpoint_every == 0 and hi < n:
                    snapshot('periodic')
            # Always commit the epoch-end optimizer state BEFORE validation/weight export.
            snapshot('epoch_training_finished')
            ef = folder / f'epoch_{epoch0 + 1}'
            es = epoch_signature(job, epoch0 + 1, dsignature)
            existing = check_epoch(ef, job, epoch0 + 1, dsignature)
            if existing:
                print(f'[REUSE] {job["key"]} epoch={epoch0 + 1} validation/weights verified', flush=True)
            else:
                wm = save_epoch_weights(P, ef, es, model)
                check_stop(stop_file)
                write_json(folder / 'status.json', {'state': 'VALIDATING', 'epoch': epoch0 + 1, 'global_step': p['global_step']})
                vr = validation_epoch(P, L, model, data, job['context'], ef, es, device, stop_file)
                steps = [u for u in p['updates'] if u['epoch'] == epoch0 + 1]
                need(sum(u['rows'] for u in steps) == n, 'Epoch did not visit every training row once.')
                er = {'state': 'EPOCH_READY', 'signature': es, 'job_signature': job['signature'],
                      'epoch': epoch0 + 1, 'learning_rate': job['learning_rate'], 'seed': job['seed'],
                      'context_id': job['context_id'], 'global_step': p['global_step'],
                      'epoch_order_sha256': p['order_sha256'], 'weights': wm, 'validation': vr['validation'],
                      'train_rows': n, 'validation_rows': len(data['validation']['labels']),
                      'optimizer_steps_in_epoch': len(steps),
                      'mean_effective_batch_weighted_loss': sum(u['loss'] for u in steps) / len(steps),
                      'training_compute_seconds': sum(u['seconds'] for u in steps),
                      'validation_forward_seconds': vr['forward_seconds'], 'final_test_read': False}
                write_json(ef / 'epoch_report.json', er)
                commit_bundle(ef, es, ['weights.PRIVATE.pt', 'weights.ready.json', 'epoch_report.json',
                                      'validation/complete.json', 'validation/validation_report.json'])
                print(f'[EPOCH] {job["key"]} {epoch0 + 1}/3 source_validation_macro_f1={vr["validation"]["selection_score"]:.6f}', flush=True)
            p['epoch'] += 1
            p['cursor'] = 0
            p['order_sha256'] = None
        assert_progress(p, n, job)
        epochs = [check_epoch(folder / f'epoch_{i}', job, i, dsignature) for i in (1, 2, 3)]
        need(all(epochs), 'Missing candidate epoch outputs.')
        best = candidate_best(epochs)
        # Persist a reproducible update trace, without example texts or IDs.
        columns = ['global_step', 'epoch', 'cursor', 'rows', 'microsteps', 'loss', 'gradient_norm_before_clip',
                   'learning_rate_used', 'seconds', 'gpu_uuid', 'backbone_gradients_checked']
        write_csv(folder / 'optimizer_updates.csv', p['updates'], columns)
        r = {'state': 'XLMR_CANDIDATE_READY', 'key': job['key'], 'signature': job['signature'],
             'seed': job['seed'], 'context_id': job['context_id'], 'task': data['report']['task'],
             'learning_rate': job['learning_rate'], 'epochs': epochs, 'best_epoch': best['epoch'],
             'best_validation_score': best['validation']['selection_score'],
             'selection_rule': job['selection_rule'], 'validation_languages': job['validation_languages'],
             'evaluation_recipe_ids': job['context']['evaluation_recipe_ids'],
             'initial_model_sha256': init_sha, 'optimizer_steps': p['global_step'],
             'data_signature': dsignature, 'class_weights': data['report']['class_weights'],
             'train_rows': n, 'validation_rows': len(data['validation']['labels']),
             'settings': SETTINGS, 'final_test_read': False, 'refit_train_plus_validation': False,
             'finished_at': now()}
        write_json(folder / 'candidate_report.json', r)
        files = ['job_spec.json', 'candidate_report.json', 'optimizer_updates.csv', 'initialization.json']
        files += [f'epoch_{i}/complete.json' for i in (1, 2, 3)]
        commit_bundle(folder, job['signature'], files)
        write_json(folder / 'status.json', {'state': 'XLMR_CANDIDATE_READY', 'best_epoch': best['epoch'],
                                           'source_validation_macro_f1': r['best_validation_score']})
        return r
    finally:
        invocation.update(finished_at=now(), end_global_step=p['global_step'])
        if device.startswith('cuda'):
            invocation.update(peak_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20)
        write_json(folder / 'executions' / f'{inv_id}.json', invocation)
        del model, opt, sched
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


def verify_candidate(root, job, verifier=None):
    folder = safe(Path(root) / 'jobs', job['key'])
    v = verifier or Verifier()
    if not bundle_ready(folder, job['signature'], v):
        return None
    r = read_json(folder / 'candidate_report.json')
    need(r['signature'] == job['signature'] and r['state'] == 'XLMR_CANDIDATE_READY', 'Candidate identity differs.')
    need(r['settings'] == SETTINGS and not r['final_test_read'], 'Candidate scientific settings differ.')
    epochs = [check_epoch(folder / f'epoch_{e}', job, e, r['data_signature'], v) for e in (1, 2, 3)]
    need(all(epochs) and epochs == r['epochs'], 'Candidate epoch records differ.')
    best = candidate_best(epochs)
    need(r['best_epoch'] == best['epoch'] and r['best_validation_score'] == best['validation']['selection_score'], 'Candidate epoch selection mismatch.')
    need(r['optimizer_steps'] == job['optimizer_steps'], 'Candidate total optimizer steps differ.')
    return r


def write_csv(path, rows, columns):
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
    atomic(path, lambda f: f.write(buf.getvalue()), False)


def collect_report(root, root_spec, jobs, requested, verifier=None):
    v = verifier or Verifier()
    candidates, epochs, selections, pending = [], [], [], []
    grouped = {}
    for job in jobs:
        r = verify_candidate(root, job, v)
        if r is None:
            status = safe(Path(root)/'jobs', job['key'])/'status.json'
            partial_epochs = []
            dp = context_folder(root, job['seed'], job['context_id'])/'data_report.json'
            # Report already committed epochs even when the three-epoch candidate is unfinished.
            if dp.exists():
                data_signature = read_json(dp)['signature']
                for number in (1, 2, 3):
                    er = check_epoch(safe(Path(root)/'jobs', job['key'])/f'epoch_{number}', job,
                                     number, data_signature, v)
                    if er:
                        partial_epochs.append(number)
                        row = {k: er[k] for k in ('epoch','learning_rate','seed','context_id','global_step','train_rows','validation_rows','training_compute_seconds','validation_forward_seconds')}
                        row['selection_score'] = er['validation']['selection_score']
                        for lang, metrics in er['validation']['by_language'].items():
                            row[f'{lang}_macro_f1'] = metrics['macro_f1']
                        epochs.append(row)
            pending.append({'key': job['key'], 'verified_completed_epochs': partial_epochs,
                            'status': read_json(status) if status.exists() else {'state':'NOT_STARTED'}})
            continue
        candidates.append(r)
        grouped.setdefault((r['seed'], r['context_id']), []).append(r)
        for e in r['epochs']:
            row = {k:e[k] for k in ('epoch','learning_rate','seed','context_id','global_step','train_rows','validation_rows','training_compute_seconds','validation_forward_seconds')}
            row['selection_score'] = e['validation']['selection_score']
            for lang, metrics in e['validation']['by_language'].items():
                row[f'{lang}_macro_f1'] = metrics['macro_f1']
            epochs.append(row)
    for (seed, context), pair in sorted(grouped.items()):
        if len(pair) == 2:
            need(pair[0]['initial_model_sha256'] == pair[1]['initial_model_sha256'], 'Learning-rate trials started from different model initialization.')
            selected = select_context(pair)
            path = Path(root)/'selections'/f'seed_{seed}'/f'{re_safe(context)}.json'
            if path.exists():
                need(read_json(path) == selected, 'Existing context selection differs.')
            else:
                write_json(path, selected)
            selections.append(selected)
    finished = {r['key'] for r in candidates}
    request_done = set(requested) <= finished
    state = ('XLMR_MODELS_READY' if len(candidates)==90 and len(selections)==45 else
             'REQUESTED_XLMR_JOBS_READY' if request_done else 'XLMR_TRAINING_IN_PROGRESS')
    return {'schema_version': VERSION, 'state': state, 'protocol_id': root_spec['protocol_id'],
            'root_spec_sha256': digest(canonical(root_spec)), 'created_at': now(),
            'planned_jobs':90, 'completed_jobs':len(candidates), 'planned_selections':45,
            'selected_models':len(selections), 'completed_epoch_evaluations':len(epochs),
            'requested_jobs':len(requested), 'requested_scope_ready':request_done,
            'settings':SETTINGS, 'candidates':candidates, 'epoch_rows':epochs,
            'selections':selections, 'pending_jobs':pending,
            'final_test_predictions_created':False, 'refit_train_plus_validation':False,
            'model_selection_scope':'authorized validation only',
            'verification_scope':'committed job/epoch weights, validation shards and report identities'}, finished


def package(root, report, quiet=False):
    root = Path(root)
    write_json(root/'xlmr_training_report.json', report)
    write_csv(root/'all_epoch_validation_results.csv',report.get('epoch_rows',[]),
              ['seed','context_id','learning_rate','epoch','global_step','train_rows','validation_rows','selection_score',
               'tr_macro_f1','kz_macro_f1','training_compute_seconds','validation_forward_seconds'])
    write_csv(root/'selected_validation_results.csv',report.get('selections',[]),
              ['seed','context_id','task','learning_rate','epoch','selection_score','selection_rule','selected_model_sha256'])
    lines=['SAUCIS — TAM XLM-R EGITIMI / VALIDATION SECIMI','='*66,
           'Durum: '+report['state'], 'Protocol: '+report.get('protocol_id','unknown'),
           f"Tamamlanan LR adayi: {report.get('completed_jobs',0)}/90; secilmis model: {report.get('selected_models',0)}/45",
           f"Tamamlanan epoch validation: {report.get('completed_epoch_evaluations',0)}/270",
           'FP32; microbatch4; effectivebatch32; max_length256; 3epoch; 2LR; 5seed.',
           'Ana/binary/domain; low-resource XLM ve LLM bu kapsama eklenmedi.',
           'Secim kaynak validation; joint esit dil Macro-F1. Esitlik: kucuk LR, erken epoch.',
           'FINAL TEST OKUNMADI. Pilot checkpointi kullanilmadi. Train+validation refit YOK.',
           'Bitmis epoch/model dosyalari korunur; yalnizca bu kosunun 2 gecici resume slotu donusumlu guncellenir.',
           'Ham metin/kimlik/token/tahmin/PRIVATE checkpoint ZIPe dahil DEGIL.']
    if report.get('message'):
        lines.append('Bilgi: '+report['message'])
    atomic(root/'xlmr_training_summary.txt',lambda f:f.write('\n'.join(lines)+'\n'),False)
    names=['root_spec.json','training_plan.json','xlmr_training_report.json','xlmr_training_summary.txt',
           'all_epoch_validation_results.csv','selected_validation_results.csv']
    # All permitted source-report members are constructed here; never recursively zip the run folder.
    def writer(f):
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
            for name in names:
                if (root/name).exists():z.write(root/name,name)
            hp=root/'resume'/'checkpoint_history.json'
            if hp.exists():z.write(hp,'checkpoint_history.json')
            for ex in sorted((root/'jobs').glob('seed_*/*/lr_*/executions/*.json')):
                z.write(ex,'executions/'+str(ex.relative_to(root/'jobs')))
    atomic(root/'shareable_xlmr_training.zip',writer)
    if not quiet:
        print('\n'.join(lines),flush=True)
        print('[SHARE]',root/'shareable_xlmr_training.zip',flush=True)


def child(planfile):
    plan=read_json(planfile);root=Path(plan['root']);job=plan['job']
    folder=safe(root/'jobs',job['key']);folder.mkdir(parents=True,exist_ok=True)
    configure_cpu(plan['threads'])
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    try:
        P=load_pilot(plan['project']);L=P.load_helper(plan['project'],'linear');F=P.load_helper(plan['project'],'features')
        need(sha(__file__)==plan['code_sha256'],'Training runner source changed while running.')
        need(digest(canonical(read_json(root/'root_spec.json')))==job['root_spec_sha256'],'Child root identity differs.')
        import numpy as np
        import torch
        from transformers import AutoModelForSequenceClassification
        need(F.versions()==plan['neural_packages'],'Neural package versions changed.')
        need(torch.cuda.is_available() and torch.cuda.device_count()==1,'Child must see exactly one requested CUDA GPU.')
        torch.cuda.set_device(0);torch.set_num_threads(plan['threads'])
        actual=str(getattr(torch.cuda.get_device_properties(0),'uuid',''))
        need(actual.lower().removeprefix('gpu-')==plan['gpu']['uuid'].lower().removeprefix('gpu-'),'Physical GPU differs from command.')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        snapshot,asset=F.verify_asset(plan['xlmr_stage'],Path(plan['hub_root']),'xlmr_tokens')
        need(asset['file_manifest_sha256']==plan['asset_sha256'] and asset['revision']==SETTINGS['revision'],'Pinned pretrained asset changed.')
        data=load_data(plan['data_folder'],plan['data_signature'])
        check_stop(plan.get('stop_file'))
        def factory():
            random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
            names=['negative','neutral','positive'] if data['report']['num_labels']==3 else ['negative','positive']
            model=AutoModelForSequenceClassification.from_pretrained(str(snapshot),num_labels=len(names),
                id2label=dict(enumerate(names)),label2id={v:i for i,v in enumerate(names)},
                local_files_only=True,trust_remote_code=False,attn_implementation='eager')
            need(model.config.model_type=='xlm-roberta' and all(p.requires_grad for p in model.parameters()),'Unexpected/frozen base model.')
            model.float().to('cuda:0')
            # Same pinned config/head construction for each LR; never load a pilot or fine-tuned model here.
            write_json(folder/'model_config.PRIVATE.json',json.loads(model.config.to_json_string()))
            return model
        torch.cuda.reset_peak_memory_stats(0)
        gpu=dict(plan['gpu'],actual_cuda_uuid=actual,compute_capability=list(torch.cuda.get_device_capability(0)),torch_cuda=torch.version.cuda)
        engine(P,L,factory,data,root,job,'cuda:0',gpu,plan['checkpoint_every_steps'],plan.get('stop_file'))
        return 0
    except BaseException as exc:
        interrupted=isinstance(exc,(TrainingStopped,KeyboardInterrupt)) or _STOP
        state='INTERRUPTED' if interrupted else ('OOM_BLOCKED' if 'OutOfMemory' in type(exc).__name__ else 'XLMR_TRAINING_BLOCKED')
        safe_types=(XLMTrainingError,)
        try:safe_types+=(P.PilotError,L.TrainingError,F.FeatureError)
        except (NameError,UnboundLocalError):pass
        message=str(exc) if isinstance(exc,safe_types) else 'Local runtime error. PRIVATE traceback retained; no automatic setting changes.'
        write_json(folder/'status.json',{'state':state,'error_type':type(exc).__name__,'message':message})
        atomic(folder/f'error_{time.time_ns()}.PRIVATE.log',lambda f:f.write(traceback.format_exc()),False)
        return 130 if interrupted else 1


def launch(planfile,gpu,stop_file=None):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1';env['TOKENIZERS_PARALLELISM']='false'
    log=Path(planfile).parent/f'console_{time.time_ns()}.PRIVATE.log'
    with log.open('a',encoding='utf-8') as f:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--child-plan',str(planfile)],
            env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        offset=0;buffer='';sent=False
        # Stream only our safe progress tags, never arbitrary library warnings or tracebacks.
        tags=('[STEP]','[EPOCH]','[CHECKPOINT]','[RESUME]','[VALIDATE]','[REUSE]','[DEVICE CHANGE]')
        while True:
            if stopping(stop_file) and not sent and proc.poll() is None:
                proc.send_signal(signal.SIGTERM);sent=True
                print('[STOP REQUESTED] Waiting for optimizer boundary / committed checkpoint; do not send repeated interrupts.',flush=True)
            with log.open('r',encoding='utf-8',errors='replace') as reader:
                reader.seek(offset);chunk=reader.read();offset=reader.tell()
            buffer+=chunk
            chunks=buffer.split('\n');buffer=chunks.pop()
            for line in chunks:
                if line.startswith(tags):print(line,flush=True)
            if proc.poll() is not None:break
            time.sleep(.4)
    if proc.returncode==130 or sent:
        raise TrainingStopped('Child stopped; last committed optimizer boundary and validation shards are retained.')
    need(proc.returncode==0,'XLM-R child did not complete. See safe status in shareable report before sharing private logs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='.')
    for name in ('split','features','smoke','linear','pilot'):parser.add_argument('--'+name)
    parser.add_argument('--gpu',default='0',help='Authorized physical nvidia-smi index or complete UUID; no auto-switch.')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    parser.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    parser.add_argument('--checkpoint-every-steps',type=int,default=500,help='Working recovery checkpoint frequency; also saved at epoch ends and graceful stop.')
    parser.add_argument('--stop-file',help='Optional sentinel file; its existence requests a graceful stop. Remove explicitly to resume.')
    parser.add_argument('--hub-cache')
    parser.add_argument('--plan-only',action='store_true',help='Verify inputs and write plan only, without model training.')
    parser.add_argument('--child-plan',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.child_plan:return child(args.child_plan)
    need(all(getattr(args,n) for n in ('split','features','smoke','linear','pilot')),'Provide split/features/smoke/linear/pilot.')
    need(args.threads>0 and args.checkpoint_every_steps>0,'Positive threads and checkpoint frequency required.')
    need(len(set(args.seeds))==len(args.seeds) and len(set(args.families))==len(args.families),'Duplicate scope selection.')
    configure_cpu(args.threads)
    project=Path(args.project).expanduser().resolve()
    def resolve(v):
        p=Path(v).expanduser();return (p if p.is_absolute() else project/p).resolve()
    split,features,smoke,linear,pilot=map(resolve,(args.split,args.features,args.smoke,args.linear,args.pilot))
    stop_file=str(resolve(args.stop_file)) if args.stop_file else None
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    P,L,F,S,base,fr,contexts,asset,npkgs,pr,pad,bank=inspect_environment(project,split,features,smoke,linear,pilot,args.hub_cache)
    code=sha(__file__)
    root_spec=dict(schema_version=VERSION,protocol_id=base['protocol_id'],corpus_sha256=base['corpus_sha256'],
        runner_sha256=code,helper_sha256={k:v[1] for k,v in P.HELPERS.items()}|{'pilot':PILOT_HELPER[1]},
        split_report_sha256=base['split_report_sha256'],feature_report_sha256=base['feature_report_sha256'],
        smoke_report_sha256=base['smoke_report_sha256'],linear_report_sha256=sha(linear/'linear_training_report.json'),
        pilot_report_sha256=sha(pilot/'xlmr_pilot_report.json'),neural_packages=npkgs,linear_packages=base['packages'],
        xlmr_asset_sha256=asset['file_manifest_sha256'],token_spec_sha256=fr['banks']['xlmr_tokens']['spec_sha256'],
        settings=SETTINGS,threads=args.threads,
        retention='immutable weights/predictions for all 3 epochs of all 90 candidates; 2 rotating run-local full-state working slots',
        gpu_identity_in_scientific_key=False)
    need(pr['identity']['linear_report_sha256']==root_spec['linear_report_sha256'],'Linear evidence changed since pilot.')
    jobs=make_jobs(P,L,split,contexts,root_spec)
    pilot_plan=read_json(pilot/'future_xlmr_jobs.json')
    need(pilot_plan['count']==90 and len(pilot_plan['jobs'])==90, 'Pilot future-job scope differs.')
    previous={(int(j['seed']),j['context_id'],float(j['learning_rate'])):j for j in pilot_plan['jobs']}
    need(len(previous)==90, 'Pilot plan contains duplicated jobs.')
    for j in jobs:
        old=previous.get((j['seed'],j['context_id'],j['learning_rate']))
        need(old is not None and all(j.get(k)==value for k,value in old.items()),
             'Approved full training matrix differs from the reviewed pilot job plan.')
    root=project/'_revision_xlmr'/f'xlmr_{base["protocol_id"]}_{code[:10]}'
    root.mkdir(parents=True,exist_ok=True)
    print('[OUTPUT]',root,flush=True)
    requested=[j for j in jobs if j['seed'] in args.seeds and j['context_id'].split(':')[0] in args.families]
    need(requested,'No requested XLM jobs.')
    gpu=None if args.plan_only else S.safe_gpu_info(args.gpu)
    if gpu:print('[GPU]',gpu['index'],gpu['uuid'],gpu['name'],flush=True)
    status={'state':'INITIALIZING','protocol_id':base['protocol_id'],'completed_jobs':0,'selected_models':0,'completed_epoch_evaluations':0}
    rc=0;verifier=Verifier()
    with lock(root/'.lock'):
        try:
            if (root/'root_spec.json').exists():need(read_json(root/'root_spec.json')==root_spec,'Training run identity changed; do not bypass or overwrite.')
            else:write_json(root/'root_spec.json',root_spec)
            write_json(root/'training_plan.json',{'root_spec':root_spec,'count':90,'jobs':jobs,'approved_by_user':True,'final_test_evaluation_in_this_runner':False})
            status,finished=collect_report(root,root_spec,jobs,[j['key'] for j in requested],verifier)
            if args.plan_only:
                status['state']='XLMR_PLAN_READY' if status['state']!='XLMR_MODELS_READY' else status['state']
                return 0
            store=ResumeStore(root,digest(canonical(root_spec)),P)
            current=store.pointer()
            # Never overwrite the only resume state of an unfinished job from a different filter.
            if current and current['job_key'] not in finished:
                need(current['job_key'] in {j['key'] for j in requested},'An unfinished job outside requested filters owns the resume cache; resume its original scope first.')
                requested.sort(key=lambda j:j['key']!=current['job_key'])
            for index,job in enumerate(requested,1):
                check_stop(stop_file)
                if job['key'] in finished:
                    print('[REUSE]',job['key'],'verified complete; no training',flush=True);continue
                print(f'[JOB {index}/{len(requested)}] {job["key"]} n_train={job["train_rows"]} n_val={job["validation_rows"]} steps={job["optimizer_steps"]}',flush=True)
                df,dr=prepare_data(P,L,bank,split,root,job,pad,verifier)
                folder=safe(root/'jobs',job['key']);folder.mkdir(parents=True,exist_ok=True)
                plan={'root':str(root),'project':str(project),'job':job,'gpu':gpu,'threads':args.threads,
                      'data_folder':str(df),'data_signature':dr['signature'],'code_sha256':code,
                      'neural_packages':npkgs,'hub_root':str(F.hub_root(args.hub_cache)),
                      'xlmr_stage':{s['stage']:s for s in read_json(smoke/'model_smoke_report.json')['stages']}['xlmr'],
                      'asset_sha256':asset['file_manifest_sha256'],'checkpoint_every_steps':args.checkpoint_every_steps,
                      'stop_file':stop_file}
                planfile=folder/'launch_plan.local.json';write_json(planfile,plan)
                launch(planfile,gpu,stop_file)
                need(verify_candidate(root,job,verifier) is not None,'Child returned without complete candidate.')
                status,finished=collect_report(root,root_spec,jobs,[j['key'] for j in requested],verifier)
                package(root,status,quiet=True)
                print(f'[PROGRESS] candidates={status["completed_jobs"]}/90 selections={status["selected_models"]}/45',flush=True)
            status,finished=collect_report(root,root_spec,jobs,[j['key'] for j in requested],verifier)
        except BaseException as exc:
            try:status,_=collect_report(root,root_spec,jobs,[j['key'] for j in requested],verifier)
            except BaseException:pass
            status['state']='INTERRUPTED' if _STOP or isinstance(exc,(TrainingStopped,KeyboardInterrupt)) else 'XLMR_TRAINING_BLOCKED'
            safe_types=(XLMTrainingError,P.PilotError,L.TrainingError,F.FeatureError)
            status['error_type']=type(exc).__name__
            status['message']=str(exc) if isinstance(exc,safe_types) else 'Local runtime error; completed files retained. See PRIVATE log locally.'
            atomic(root/f'error_{time.time_ns()}.PRIVATE.log',lambda f:f.write(traceback.format_exc()),False)
            rc=130 if status['state']=='INTERRUPTED' else 1
        finally:
            status['finished_at']=now();package(root,status)
    return rc


if __name__=='__main__':
    try:sys.exit(main())
    except XLMTrainingError as exc:
        print('[BLOCKED]',str(exc),file=sys.stderr);sys.exit(1)
