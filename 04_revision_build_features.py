#!/usr/bin/env python3
"""SAUCIS revision: resumable, label-free full-corpus feature banks.

Consumes immutable step-02 corpus and successful step-03 smoke report. No fitting,
model download, final-test predictions, old-cache import, or split modification.
One explicitly selected physical GPU; separate child for each frozen encoder.
XLM-R output is tokenizer data ONLY, not a trained model or sentence embeddings.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import sqlite3
import subprocess
import sys
import time
import zipfile

VERSION = '1.0.0'
FROZEN = {
    'minilm': ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 384, 128),
    'labse': ('sentence-transformers/LaBSE', 768, 256),
    'bge_m3': ('BAAI/bge-m3', 1024, 8192),
}
ALIASES = tuple(FROZEN) + ('xlmr_tokens',)
PACKAGES = ('numpy', 'torch', 'transformers', 'sentence-transformers', 'tokenizers',
            'safetensors', 'huggingface-hub')
_STOP = False

class FeatureError(RuntimeError): pass
class Stopped(FeatureError): pass

def need(ok, message):
    if not ok: raise FeatureError(message)

def now(): return datetime.now(timezone.utc).isoformat()
def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()
def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.partial_{os.getpid()}_{time.time_ns()}')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def versions():
    out = {'python': platform.python_version()}
    for key in PACKAGES:
        try: out[key] = metadata.version(key)
        except metadata.PackageNotFoundError: out[key] = None
    return out

@contextmanager
def locked(path):
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e: raise FeatureError('Feature output is in use by another process.') from e
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)

@contextmanager
def ro_db(path):
    path = Path(path).resolve(); need(path.is_file(), 'Required local corpus is missing.')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    try: yield db
    finally: db.close()

@dataclass(frozen=True)
class Row:
    row_id: int
    sample_id: str
    text: str
    language: str
    domain: str


def corpus_shards(path, shard_size):
    """Intentionally never reads label, rating, native_id or any test membership."""
    need(shard_size > 0, 'Invalid shard size.')
    with ro_db(path) as db:
        cur = db.execute('SELECT row_id,sample_id,model_text,language,domain FROM records ORDER BY row_id')
        expected = 1
        while True:
            raw = cur.fetchmany(shard_size)
            if not raw: break
            rows = [Row(*r) for r in raw]
            need([r.row_id for r in rows] == list(range(expected, expected + len(rows))),
                 'Corpus row IDs are not contiguous; cannot use positional cache indexing.')
            need(all(isinstance(r.text, str) and len(r.text) >= 3 for r in rows), 'Unexpected input text schema.')
            need(all(len(r.sample_id) == 64 for r in rows), 'Unexpected sample ID format.')
            expected += len(rows)
            yield rows


def row_fingerprint(rows):
    return digest(canonical([[r.row_id, r.sample_id, digest(r.text), r.language, r.domain] for r in rows]))

def inspect_inputs(split, smoke):
    protocol = load_json(split / 'protocol.json')
    sr = load_json(split / 'split_report.json')
    need(sr['protocol_id'] == protocol['protocol_id'], 'Split protocol mismatch.')
    need(sr['state'] == 'SPLIT_MANIFESTS_READY' and sr['blocked_checks'] == 0, 'Split gate not ready.')
    need(load_json(split / 'status.json')['report_sha256'] == file_sha(split / 'split_report.json'),
         'Split report checksum mismatch.')
    cp = split / 'corpus.local.sqlite3'; stamp = load_json(split / 'corpus.complete.json')
    need(stamp['sha256'] == file_sha(cp), 'Corpus checksum differs from its completion record.')
    with ro_db(cp) as db:
        need(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'Corpus integrity error.')
        n, lo, hi, unique_ids = db.execute('SELECT COUNT(*),MIN(row_id),MAX(row_id),COUNT(DISTINCT sample_id) FROM records').fetchone()
        need(n > 0 and lo == 1 and hi == n and unique_ids == n, 'Invalid corpus row/sample IDs.')
    report = load_json(smoke / 'model_smoke_report.json')
    need(report['protocol_id'] == protocol['protocol_id'], 'Smoke and split protocol differ.')
    need(report['state'] == 'MODEL_SMOKE_PASSED', 'Complete successful step-03 smoke required.')
    stages = {s['stage']: s for s in report['stages']}
    need(len(stages) == 6 and all(stages.get(k, {}).get('state') == 'PASS'
         for k in ('cpu', 'cuda', *FROZEN, 'xlmr')), 'Missing successful smoke stages.')
    current = versions()
    for key in ('python', *PACKAGES):
        need(current[key] == report['packages'].get(key),
             f'Environment changed for {key}; do not silently mix feature runs. Review or repeat smoke.')
    return dict(protocol_id=protocol['protocol_id'], corpus_sha256=stamp['sha256'], rows=n,
                smoke_report_sha256=file_sha(smoke / 'model_smoke_report.json'), packages=current), stages


def gpu_info(selector):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    need(result.returncode == 0, 'nvidia-smi query failed.')
    cards = []
    for line in result.stdout.splitlines():
        p = [t.strip() for t in line.split(',')]
        need(len(p) == 5, 'Unexpected GPU query schema.')
        cards.append(dict(index=int(p[0]), uuid=p[1], name=p[2], total_mib=int(p[3]), free_mib=int(p[4])))
    matches = [g for g in cards if str(g['index']) == str(selector) or g['uuid'] == selector]
    need(len(matches) == 1, 'Requested physical GPU index/UUID not found.')
    g = matches[0]
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        mask = [v.strip() for v in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if v.strip()]
        need(str(g['index']) in mask or any(v.startswith('GPU-') and g['uuid'].startswith(v) for v in mask),
             'Requested GPU is excluded by the inherited mask; it is not widened.')
    return g


def hub_root(explicit=None):
    if explicit: return Path(explicit).expanduser().resolve()
    if os.environ.get('HF_HUB_CACHE'): return Path(os.environ['HF_HUB_CACHE']).expanduser().resolve()
    return Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))).expanduser().resolve() / 'hub'


def verify_asset(stage, root, alias):
    asset = stage['asset']; revision = asset['revision']
    need(len(revision) == 40 and all(c in '0123456789abcdef' for c in revision), 'Invalid pinned revision.')
    repo = FROZEN[alias][0] if alias in FROZEN else 'FacebookAI/xlm-roberta-base'
    snapshot = root / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
    need(snapshot.is_dir(), 'Pinned model is not cached; this step never downloads or repins.')
    entries = asset['files']
    need(asset['file_manifest_sha256'] == digest(canonical(entries)), 'Asset manifest hash mismatch.')
    # Exact relevant file set prevents an added second weight format from changing the loader silently.
    actual = {str(p.relative_to(snapshot)) for p in snapshot.rglob('*')
              if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt')}
    need(actual == set(entries), 'Pinned snapshot file set differs from smoke; review before continuing.')
    for name, item in entries.items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe asset manifest path.')
        p = snapshot / name
        need(p.is_file() and p.stat().st_size == item['bytes'] and file_sha(p) == item['sha256'],
             'Pinned model/tokenizer file checksum mismatch.')
    if alias in FROZEN:
        need(stage['dimension'] == FROZEN[alias][1] and stage['max_seq_length'] == FROZEN[alias][2],
             'Encoder differs from approved native dimensions/length.')
    return snapshot, asset


def make_spec(base, alias, asset, shard_size):
    frozen = alias in FROZEN
    return dict(schema_version=VERSION, builder_sha256=file_sha(__file__), protocol_id=base['protocol_id'],
        corpus_sha256=base['corpus_sha256'], rows=base['rows'], packages=base['packages'], alias=alias,
        asset_sha256=asset['file_manifest_sha256'], revision=asset['revision'],
        kind='frozen_embeddings' if frozen else 'xlmr_token_ids',
        dimension=FROZEN[alias][1] if frozen else None,
        max_length=FROZEN[alias][2] if frozen else 256,
        normalize_embeddings=bool(frozen), dtype='float32' if frozen else 'int32',
        prompt=None, fit_on_corpus=False, labels_read=False, shard_size=shard_size,
        batch_identity_policy='execution-only; batches/GPU recorded per shard, no bitwise cross-GPU claim',
        text_policy='existing model_text; no additional NFC/NFKC/casefold')


def validate_payload(arrays, spec, expected_n):
    import numpy as np
    required = {'row_ids', 'sample_ids', 'text_sha256', 'token_lengths'}
    required |= {'embeddings'} if spec['kind'] == 'frozen_embeddings' else {'input_ids', 'offsets'}
    need(set(arrays) == required, 'Wrong bank payload fields.')
    ids = arrays['row_ids']; lengths = arrays['token_lengths']
    need(ids.shape == (expected_n,) and ids.dtype == np.int64, 'Invalid row ID vector.')
    need(len(np.unique(ids)) == expected_n, 'Duplicate row IDs.')
    for key in ('sample_ids', 'text_sha256'):
        need(arrays[key].shape == (expected_n,) and arrays[key].dtype == np.dtype('S64'), 'Invalid sample/hash vector.')
    need(lengths.shape == (expected_n,) and lengths.dtype == np.int32 and np.all(lengths > 0), 'Invalid token lengths.')
    if spec['kind'] == 'frozen_embeddings':
        E = arrays['embeddings']
        need(E.shape == (expected_n, spec['dimension']) and E.dtype == np.float32, 'Wrong embedding shape/dtype.')
        need(np.isfinite(E).all(), 'Nonfinite embeddings.')
        need(np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=2e-4), 'Embeddings not L2 normalized.')
    else:
        flat, offsets = arrays['input_ids'], arrays['offsets']
        need(flat.ndim == 1 and flat.dtype == np.int32 and np.all(flat >= 0), 'Invalid token IDs.')
        need(offsets.shape == (expected_n + 1,) and offsets.dtype == np.int64 and offsets[0] == 0
             and offsets[-1] == len(flat), 'Invalid token offsets.')
        sizes = np.diff(offsets)
        need(np.all(sizes > 0) and np.all(sizes <= spec['max_length']), 'Token sequence length exceeds fixed limit.')
        need(np.array_equal(sizes, np.minimum(lengths, spec['max_length'])), 'Raw/capped token counts disagree.')


def atomic_npz(path, arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.partial_{os.getpid()}_{time.time_ns()}')
    with tmp.open('wb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load_payload(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def completed_shard(folder, number, rows, spec, with_arrays=False):
    folder = Path(folder); marker = folder / f'shard_{number:06d}.json'
    if not marker.exists(): return None
    m = load_json(marker); data = folder / f'shard_{number:06d}.PRIVATE.npz'
    need(m['spec_sha256'] == digest(canonical(spec)), 'Completed shard configuration changed.')
    need(m['input_sha256'] == row_fingerprint(rows), 'Completed shard text/order/sample identity mismatch.')
    need(m['rows'] == len(rows) and m['first_row_id'] == rows[0].row_id
         and m['last_row_id'] == rows[-1].row_id, 'Shard coverage mismatch.')
    need(data.is_file() and data.stat().st_size == m['bytes'] and file_sha(data) == m['sha256'],
         'Completed shard corrupt or missing; no silent recompute.')
    arrays = load_payload(data); validate_payload(arrays, spec, len(rows))
    need(arrays['row_ids'].tolist() == [r.row_id for r in rows], 'Saved row order mismatch.')
    need(arrays['sample_ids'].tolist() == [r.sample_id.encode() for r in rows], 'Saved sample IDs mismatch.')
    need(arrays['text_sha256'].tolist() == [digest(r.text).encode() for r in rows], 'Saved text hashes mismatch.')
    need(m['token_length_histograms'] == summary_hist(rows, arrays['token_lengths'], spec['max_length']), 'Shard profile differs from saved token lengths.')
    return (m, arrays) if with_arrays else m


def summary_hist(rows, lengths, max_length):
    counts = defaultdict(Counter)
    for r, length in zip(rows, lengths):
        n = int(length)
        for scope in ('all', 'language:' + r.language, 'domain:' + r.language + ':' + r.domain): counts[scope][n] += 1
    return {scope: {str(n): v for n, v in sorted(c.items())} for scope, c in counts.items()}


def quantile_hist(hist, q):
    n = sum(hist.values()); need(n > 0, 'Empty length histogram.')
    target = max(1, math.ceil(q * n)); running = 0
    for k, count in sorted(hist.items()):
        running += count
        if running >= target: return k
    raise FeatureError('Invalid histogram.')


def aggregate_profile(shards, max_length):
    scopes = defaultdict(Counter)
    for m in shards:
        for scope, h in m['token_length_histograms'].items():
            scopes[scope].update({int(k): int(v) for k, v in h.items()})
    out = {}
    for scope, hist in sorted(scopes.items()):
        n = sum(hist.values()); truncated = sum(v for k, v in hist.items() if k > max_length)
        out[scope] = dict(rows=n, mean_tokens=sum(k*v for k,v in hist.items())/n,
            median_tokens=quantile_hist(hist, .5), p95_tokens=quantile_hist(hist, .95),
            p99_tokens=quantile_hist(hist, .99), max_tokens=max(hist),
            truncated_rows=truncated, truncated_fraction=truncated/n)
    return out


def stop_requested(stop_file=None):
    return _STOP or (stop_file is not None and Path(stop_file).exists())
def check_stop(stop_file=None):
    if stop_requested(stop_file): raise Stopped('Stop requested; completed shards retained, current unfinished shard may be retried.')
def handle_stop(signum, frame):
    global _STOP
    _STOP = True

def parent_stop(signum, frame):
    raise KeyboardInterrupt('Parent stop requested.')


def fill_payload(rows, lengths, encoded, spec):
    import numpy as np
    out = dict(row_ids=np.array([r.row_id for r in rows], dtype=np.int64),
               sample_ids=np.array([r.sample_id.encode() for r in rows], dtype='S64'),
               text_sha256=np.array([digest(r.text).encode() for r in rows], dtype='S64'),
               token_lengths=np.asarray(lengths, dtype=np.int32))
    if spec['kind'] == 'frozen_embeddings': out['embeddings'] = np.asarray(encoded, dtype=np.float32)
    else:
        out['offsets'] = np.concatenate(([0], np.cumsum([len(v) for v in encoded], dtype=np.int64))).astype(np.int64)
        out['input_ids'] = np.concatenate([np.asarray(v, dtype=np.int32) for v in encoded])
    validate_payload(out, spec, len(rows))
    return out


class LocalProcessor:
    """Lazy local loaders; never fit and never ask for labels."""
    def __init__(self, spec, snapshot, gpu, batch_size, token_budget, threads):
        self.spec, self.snapshot, self.gpu = spec, snapshot, gpu
        self.batch_size, self.token_budget, self.threads = batch_size, token_budget, threads
        self.model = self.tokenizer = self.torch = None; self.lower = False

    def load(self):
        if self.tokenizer is not None: return
        from transformers import AutoTokenizer
        if self.spec['kind'] == 'xlmr_token_ids':
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), local_files_only=True,
                trust_remote_code=False, use_fast=True, token=False)
            return
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch; torch.set_num_threads(self.threads)
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Selected GPU CUDA runtime unavailable.')
        torch.cuda.set_device(0); prop = torch.cuda.get_device_properties(0)
        uuid = str(getattr(prop, 'uuid', ''))
        need(uuid and uuid.replace('GPU-', '') == self.gpu['uuid'].replace('GPU-', ''), 'Runtime GPU identity mismatch.')
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.model = SentenceTransformer(str(self.snapshot), device='cuda:0', local_files_only=True,
                                        trust_remote_code=False, token=False)
        self.model.eval(); self.model.float()
        for p in self.model.parameters(): p.requires_grad_(False)
        need(all(p.device.type == 'cuda' and p.device.index == 0 for p in self.model.parameters()), 'Wrong encoder device.')
        need(self.model.max_seq_length == self.spec['max_length'], 'Native encoder length changed.')
        need(self.model.get_sentence_embedding_dimension() == self.spec['dimension'], 'Encoder dimension changed.')
        need(getattr(self.model, 'default_prompt_name', None) is None, 'Unexpected prompt configuration.')
        first = self.model._first_module()
        need(hasattr(first, 'tokenizer'), 'Unsupported non-tokenizer encoder module.')
        self.tokenizer = first.tokenizer; self.lower = bool(getattr(first, 'do_lower_case', False))
        torch.cuda.reset_peak_memory_stats(0)

    def process(self, rows, stop_file=None):
        import numpy as np
        self.load(); check_stop(stop_file); started = time.monotonic()
        texts = [r.text for r in rows]
        count_texts = [(t.strip().lower() if self.lower else t.strip()) for t in texts]
        lengths, clipped = [], []
        for start in range(0, len(texts), 32):
            check_stop(stop_file)
            part = count_texts[start:start+32]
            full = self.tokenizer(part, truncation=False, padding=False, add_special_tokens=True)
            lengths.extend(len(v) for v in full['input_ids'])
            if self.spec['kind'] == 'xlmr_token_ids':
                cut = self.tokenizer(part, truncation=True, max_length=self.spec['max_length'],
                                     padding=False, add_special_tokens=True)
                clipped.extend(cut['input_ids'])
        if self.spec['kind'] == 'xlmr_token_ids':
            return fill_payload(rows, lengths, clipped, self.spec), dict(elapsed_seconds=time.monotonic()-started,
                device='cpu', native_do_lower_case=False, final_test_labels_accessed=False)
        E = np.empty((len(rows), self.spec['dimension']), dtype=np.float32)
        order = sorted(range(len(rows)), key=lambda i: (-min(lengths[i], self.spec['max_length']), i))
        actual_batches = Counter(); oom_retries = 0; position = 0
        while position < len(order):
            check_stop(stop_file)
            longest = min(lengths[order[position]], self.spec['max_length'])
            count = min(self.batch_size, len(order)-position, max(1, self.token_budget//max(1,longest)))
            while True:
                idx = order[position:position+count]
                try:
                    with self.torch.inference_mode():
                        result = self.model.encode([texts[i] for i in idx], batch_size=count,
                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                    break
                except self.torch.cuda.OutOfMemoryError:
                    oom_retries += 1
                    gc.collect(); self.torch.cuda.empty_cache()
                    need(count > 1, 'One text does not fit GPU at native FP32 settings; no truncation/precision/GPU change was made.')
                    count = max(1, count//2)
            E[idx] = result; actual_batches[count] += 1; position += count
        if rows[0].row_id == 1:
            check_stop(stop_file)
            with self.torch.inference_mode():
                repeat = self.model.encode(texts[:4], batch_size=1, convert_to_numpy=True,
                                          normalize_embeddings=True, show_progress_bar=False)
            need(np.allclose(E[:4], repeat, atol=2e-5, rtol=2e-4), 'Feature order/batch repeat probe differs beyond tolerance.')
        self.torch.cuda.synchronize()
        return fill_payload(rows, lengths, E, self.spec), dict(elapsed_seconds=time.monotonic()-started,
            gpu=self.gpu, requested_batch_size=self.batch_size, token_budget=self.token_budget,
            actual_batch_histogram=dict(actual_batches), oom_retries=oom_retries,
            peak_allocated_mib=self.torch.cuda.max_memory_allocated(0)/2**20,
            peak_reserved_mib=self.torch.cuda.max_memory_reserved(0)/2**20,
            native_do_lower_case=self.lower, dtype='float32', final_test_labels_accessed=False)


def build_bank(corpus, folder, spec, processor, stop_file=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    spec_sha = digest(canonical(spec)); spec_path = folder / 'bank_spec.json'
    with locked(folder / '.lock'):
        if spec_path.exists(): need(load_json(spec_path) == spec, 'Bank specification mismatch.')
        else: atomic_json(spec_path, spec)
        all_shards = []; reused = 0
        for number, rows in enumerate(corpus_shards(corpus, spec['shard_size'])):
            check_stop(stop_file)
            m = completed_shard(folder, number, rows, spec)
            if m is None:
                started = time.monotonic(); arrays, execution = processor.process(rows, stop_file)
                validate_payload(arrays, spec, len(rows)); check_stop(stop_file)
                data = folder / f'shard_{number:06d}.PRIVATE.npz'
                atomic_npz(data, arrays)
                m = dict(spec_sha256=spec_sha, input_sha256=row_fingerprint(rows),
                    first_row_id=rows[0].row_id, last_row_id=rows[-1].row_id, rows=len(rows),
                    file=data.name, bytes=data.stat().st_size, sha256=file_sha(data), created_at=now(),
                    token_length_histograms=summary_hist(rows, arrays['token_lengths'], spec['max_length']),
                    execution=execution, write_and_compute_seconds=time.monotonic()-started)
                atomic_json(folder / f'shard_{number:06d}.json', m)
                # Re-read the committed pair before accepting this shard.
                completed_shard(folder, number, rows, spec)
            else: reused += 1
            all_shards.append(m)
            if number % 10 == 0 or rows[-1].row_id == spec['rows']:
                print(f"[FEATURES] {spec['alias']}: {rows[-1].row_id:,}/{spec['rows']:,}; reused_shards={reused}", flush=True)
        need(sum(m['rows'] for m in all_shards) == spec['rows'], 'Full bank coverage is incomplete.')
        result = dict(state='FEATURE_BANK_READY', alias=spec['alias'], spec_sha256=spec_sha,
            rows=spec['rows'], dimension=spec['dimension'], max_length=spec['max_length'],
            precision=spec['dtype'], revision=spec['revision'], shards=len(all_shards),
            reused_shards_this_run=reused, feature_bytes=sum(m['bytes'] for m in all_shards),
            profiles=aggregate_profile(all_shards, spec['max_length']),
            recorded_gpu_uuids=sorted({m['execution'].get('gpu',{}).get('uuid') for m in all_shards if m['execution'].get('gpu')}),
            oom_retries_recorded=sum(m['execution'].get('oom_retries',0) for m in all_shards),
            labels_used=False, model_training=False, final_test_predictions=False,
            shard_files=[dict(file=m['file'], sha256=m['sha256'], rows=m['rows'],
                first_row_id=m['first_row_id'], last_row_id=m['last_row_id'],
                metadata_file=f'shard_{i:06d}.json', metadata_sha256=file_sha(folder/f'shard_{i:06d}.json'))
                for i,m in enumerate(all_shards)])
        atomic_json(folder/'bank_report.json', result)
        atomic_json(folder/'bank.complete.json', dict(spec_sha256=spec_sha,
            report_sha256=file_sha(folder/'bank_report.json'), rows=spec['rows']))
        return result


class FeatureBank:
    """Reader for future classifiers. Exact caller-requested order, local identity fields.

    read_rows returns a dictionary: embeddings OR ragged token_ids; token_lengths;
    truncated; row_ids; sample_ids; text_sha256. Caller must separately obtain labels
    from the authorized training/validation/test membership and assert ID matching.
    """
    def __init__(self, folder):
        self.folder = Path(folder); self.spec = load_json(self.folder/'bank_spec.json')
        done = load_json(self.folder/'bank.complete.json')
        need(done['spec_sha256'] == digest(canonical(self.spec)), 'Reader spec checksum mismatch.')
        need(done['report_sha256'] == file_sha(self.folder/'bank_report.json'), 'Reader report checksum mismatch.')
        self.report = load_json(self.folder/'bank_report.json'); self.checked = set()
        need(self.report['state']=='FEATURE_BANK_READY' and self.report['rows']==self.spec['rows'], 'Bank not ready.')

    def read_rows(self, row_ids):
        import numpy as np
        raw_ids = np.asarray(row_ids)
        need(raw_ids.dtype.kind in ('i','u'), 'Requested row IDs must be integers, not floats or strings.')
        ids = raw_ids.astype(np.int64)
        need(ids.ndim == 1 and len(ids)>0 and len(np.unique(ids))==len(ids), 'Empty/duplicate requested row IDs.')
        need(np.all((ids>=1)&(ids<=self.spec['rows'])), 'Requested row outside corpus.')
        out = dict(row_ids=ids.copy(), sample_ids=np.empty(len(ids),dtype='S64'),
                   text_sha256=np.empty(len(ids),dtype='S64'), token_lengths=np.empty(len(ids),dtype=np.int32))
        frozen = self.spec['kind']=='frozen_embeddings'
        if frozen: out['embeddings']=np.empty((len(ids),self.spec['dimension']),dtype=np.float32)
        else: out['token_ids']=[None]*len(ids)
        by_shard=defaultdict(list)
        for j,rid in enumerate(ids): by_shard[(int(rid)-1)//self.spec['shard_size']].append(j)
        for number, positions in by_shard.items():
            desc=self.report['shard_files'][number]; data=self.folder/desc['file']; meta=self.folder/desc['metadata_file']
            if number not in self.checked:
                need(file_sha(data)==desc['sha256'] and file_sha(meta)==desc['metadata_sha256'], 'Reader found corrupt shard.')
                self.checked.add(number)
            arrays=load_payload(data); validate_payload(arrays,self.spec,desc['rows'])
            loc=ids[positions]-desc['first_row_id']
            need(np.array_equal(arrays['row_ids'][loc],ids[positions]), 'Reader alignment error.')
            for key in ('sample_ids','text_sha256','token_lengths'): out[key][positions]=arrays[key][loc]
            if frozen: out['embeddings'][positions]=arrays['embeddings'][loc]
            else:
                for pos,local in zip(positions,loc):
                    lo,hi=arrays['offsets'][local:local+2]
                    out['token_ids'][pos]=arrays['input_ids'][lo:hi].copy()
        out['truncated']=out['token_lengths']>self.spec['max_length']
        return out


def package_report(out, report):
    atomic_json(out/'feature_build_report.json',report)
    lines=['SAUCIS — TAM VERI OZELLIK URETIMI', '='*58, 'Durum: '+report['state'],
           'Protocol: '+report['protocol_id'],
           'Frozen encoder ciktilari/token sayimlari; etiketle fit ve final-test skoru YOK.']
    for alias, r in report['banks'].items():
        lines.append(f"{alias}: {r['state']}; rows={r.get('rows')}; dimension={r.get('dimension')}; max_length={r.get('max_length')}")
    lines += ['PRIVATE NPZ ve ornek kimlikleri paylasim ZIPine eklenmez.',
              'TF-IDF her egitim butcesinde yeniden fit edilecek. XLM-R fine-tuning henuz yapilmadi.']
    (out/'feature_build_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    final=out/'shareable_feature_build.zip'; tmp=out/'shareable_feature_build.zip.partial'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('feature_build_report.json','feature_build_summary.txt'): z.write(out/name,name)
    os.replace(tmp,final)
    print('\n'.join(lines),flush=True); print('[SHARE]',final,flush=True)


def child(args):
    signal.signal(signal.SIGTERM,handle_stop); signal.signal(signal.SIGINT,handle_stop)
    p=Path(args.child_plan); plan=load_json(p); spec=plan['spec']; folder=Path(plan['folder'])
    try:
        need(file_sha(plan['corpus'])==spec['corpus_sha256'],'Corpus changed before worker start.')
        processor=LocalProcessor(spec,Path(plan['snapshot']),plan.get('gpu'),args.batch_size,args.token_budget,args.threads)
        result=build_bank(Path(plan['corpus']),folder,spec,processor,args.stop_file)
        need(file_sha(plan['corpus'])==spec['corpus_sha256'],'Corpus changed while encoding.')
        atomic_json(folder/'worker_result.json',result); return 0
    except BaseException as exc:
        status='INTERRUPTED' if isinstance(exc,(Stopped,KeyboardInterrupt)) else 'BLOCKED'
        atomic_json(folder/'worker_result.json',dict(state=status,error_type=type(exc).__name__,
           message=str(exc) if isinstance(exc,FeatureError) else 'Runtime failure; inspect local PRIVATE log.'))
        import traceback; traceback.print_exc()
        return 130 if status=='INTERRUPTED' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='.')
    parser.add_argument('--split',required=True)
    parser.add_argument('--smoke',required=True)
    parser.add_argument('--gpu',help='Physical nvidia-smi index or full GPU UUID; explicit authorization only.')
    parser.add_argument('--models',nargs='+',choices=ALIASES,default=list(ALIASES))
    parser.add_argument('--batch-size',type=int,default=16,help='Frozen inference only; never an XLM-R training batch.')
    parser.add_argument('--token-budget',type=int,default=8192,help='Maximum nominal padded tokens per inference batch (minimum one text).')
    parser.add_argument('--shard-size',type=int,default=512)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--hub-cache')
    parser.add_argument('--stop-file',help='Optional local sentinel; remove/rename it yourself before resuming.')
    parser.add_argument('--child-plan',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.child_plan: return child(args)
    signal.signal(signal.SIGTERM, parent_stop)
    need(all(v>0 for v in (args.batch_size,args.token_budget,args.shard_size,args.threads)), 'Settings must be positive.')
    need(len(args.models)==len(set(args.models)), 'Duplicate model selection.')
    if any(m in FROZEN for m in args.models) and args.gpu is None: parser.error('--gpu required for frozen encoders.')
    project=Path(args.project).expanduser().resolve()
    def resolve(p):
        p=Path(p).expanduser(); return (p if p.is_absolute() else project/p).resolve()
    split,smoke=resolve(args.split),resolve(args.smoke)
    if args.stop_file: args.stop_file=str(resolve(args.stop_file))
    base,stages=inspect_inputs(split,smoke)
    code=file_sha(__file__)[:10]; out=project/'_revision_features'/f"features_{base['protocol_id']}_{code}"
    out.mkdir(parents=True,exist_ok=True)
    print('[OUTPUT]',out,flush=True)
    report=dict(schema_version=VERSION,created_at=now(),state='RUNNING',protocol_id=base['protocol_id'],
        input_rows=base['rows'],corpus_sha256=base['corpus_sha256'],builder_sha256=file_sha(__file__),
        smoke_report_sha256=base['smoke_report_sha256'],packages=base['packages'],requested_models=args.models,
        banks={},final_test_predictions_created=False,models_fitted=False,
        limitations=['Frozen features include unlabeled test texts; no corpus fitting or test performance.',
                     'Not full XLM-R training or training-batch sizing. Native limits not changed.',
                     'Per-shard hardware/batch recorded; cross-GPU bitwise identity is not asserted.'])
    rc=0
    with locked(out/'.lock'):
        try:
            check_stop(args.stop_file)
            gpu=gpu_info(args.gpu) if any(m in FROZEN for m in args.models) else None
            if gpu: print('[GPU]',gpu['index'],gpu['uuid'],gpu['name'],flush=True)
            hub=hub_root(args.hub_cache)
            for alias in args.models:
                check_stop(args.stop_file)
                stage=stages['xlmr' if alias=='xlmr_tokens' else alias]
                snapshot,asset=verify_asset(stage,hub,alias)
                spec=make_spec(base,alias,asset,args.shard_size)
                folder=out/(alias+'_'+digest(canonical(spec))[:12]);folder.mkdir(exist_ok=True)
                atomic_json(folder/'worker_plan.local.json',dict(spec=spec,snapshot=str(snapshot),folder=str(folder),
                    corpus=str(split/'corpus.local.sqlite3'),gpu=gpu if alias in FROZEN else None))
                env=dict(os.environ)
                env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='42',
                    OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),
                    CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu['uuid'] if alias in FROZEN else '')
                cmd=[sys.executable,str(Path(__file__).resolve()),'--project',str(project),'--split',str(split),'--smoke',str(smoke),
                     '--child-plan',str(folder/'worker_plan.local.json'),'--batch-size',str(args.batch_size),
                     '--token-budget',str(args.token_budget),'--threads',str(args.threads)]
                if args.stop_file: cmd+=['--stop-file',args.stop_file]
                print('[BUILD]',alias,'(local pinned snapshot; label-free inference)',flush=True)
                log=folder/f'run_{time.time_ns()}.PRIVATE.log'
                # Child output is streamed locally; share package contains only aggregate reports.
                proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                      bufsize=1,start_new_session=True)
                try:
                    with log.open('w',encoding='utf-8') as f:
                        for line in proc.stdout:
                            f.write(line);f.flush();print(line,end='',flush=True)
                    exit_code=proc.wait()
                except KeyboardInterrupt:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise
                result_path=folder/'worker_result.json'
                need(result_path.exists(),'Worker stopped before report; completed shards still remain.')
                result=load_json(result_path)
                if exit_code!=0 or result['state']!='FEATURE_BANK_READY':
                    report['banks'][alias]=result
                    if exit_code==130: raise Stopped('Worker stopped; run same command to continue.')
                    raise FeatureError('Feature worker failed; inspect its local PRIVATE log.')
                public={k:v for k,v in result.items() if k!='shard_files'}
                public['bank_directory']=str(folder.relative_to(out));report['banks'][alias]=public
            report['state']='FEATURE_BANKS_READY' if set(args.models)==set(ALIASES) else 'REQUESTED_FEATURE_BANKS_READY'
        except BaseException as exc:
            report['state']='INTERRUPTED' if isinstance(exc,(Stopped,KeyboardInterrupt)) else 'FEATURE_BUILD_BLOCKED'
            report['error_type']=type(exc).__name__
            report['message']=str(exc) if isinstance(exc,FeatureError) else 'Local runtime failure; no source/split/legacy edits.'
            rc=130 if report['state']=='INTERRUPTED' else 1
            import traceback
            (out/f'error_{time.time_ns()}.PRIVATE.log').write_text(traceback.format_exc(),encoding='utf-8')
        finally: package_report(out,report)
    return rc

if __name__=='__main__': sys.exit(main())
