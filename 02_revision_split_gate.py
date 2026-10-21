#!/usr/bin/env python3
"""SAUCIS revision: offline, group-disjoint split and budget manifests.

No training, downloads, pip installation, GPU allocation or modification of legacy
files. Python >=3.10, standard library only. Read pinned CSVs from step 01's
source_manifest.local.json and verify their SHA-256 against its public report.
NFKC -> casefold -> whitespace collapse is used ONLY for grouping. Model text
receives whitespace collapse only; raw text and all accepted labels are retained.

Output is LOCAL PRIVATE data. Share ONLY shareable_split_audit.zip.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import platform
import re
import sqlite3
import sys
import time
import unicodedata
import zipfile

VERSION = '1.0.0'
SEEDS = (42, 43, 44, 45, 46)
LABELS = ('negative', 'neutral', 'positive')
LANGS = ('tr', 'kz')
SPECS = {
    'tr': dict(repo='maydogan/Turkish_SentimentAnalysis_TRSAv1', text='review',
               label='score', native_id='id', columns=('id', 'score', 'review')),
    'kz': dict(repo='issai/kazsandra', text='text', label='label',
               native_id='custom_id', columns=('custom_id', 'text', 'label', 'domain')),
}
DOMAINS = {s.casefold(): s for s in ('Appstore', 'Market', 'Mapping', 'Bookstore')}
DEFAULT_POLICY = {
    'seeds': list(SEEDS),
    'group_key': 'sha256(collapse_ws(NFKC(raw_text).casefold()))',
    'model_text': 'collapse_ws(raw_text); no added Unicode normalization or casefold',
    'eligibility': 'recognized_label and len(collapse_ws(raw_text)) >= 3',
    'duplicate_policy': 'retain records and original labels; keep group in one partition',
    'main_split': {'train': 0.70, 'validation': 0.10, 'test': 0.20},
    'split_strata': 'language + reported domain + mapped three-class label',
    'allocator': 'group_histogram_greedy_v1; repeated groups then exact singleton quota fill',
    'main_train_cap_per_class_per_language': 12000,
    'low_total_labels_per_class_per_language': [100, 500, 1000, 3000, 5000],
    'low_train_fraction': 0.80,
    'low_pools': '80/20 group-disjoint roles within the non-test development pool',
    'low_nesting': 'hash-ranked records per language/class/role, independent of budget',
    'domain_train_per_class': 500,
    'domain_validation_per_class': 80,
    'domain_sources': ['TRSAv1', 'KZ_Market', 'KZ_Appstore'],
    'sampling_unit_within_role': 'record; budgets may select part of a group but cannot move it across roles',
    'sampling_failure': 'REVIEW_REQUIRED; no quota shrink, replacement, seed retry or resplitting',
    'binary_labels': {'negative': 0, 'positive': 1},
    'extreme_test': 'Kazakh original rating 1 or 5; reuse binary-trained models, no extreme-only fit',
    'selection': 'source-only validation; joint = mean of the two language macro-F1 scores',
    'cross_protocol_roles': 'main and low-resource train/validation roles may differ; test is identical per seed',
}


class GateError(RuntimeError):
    """Safe error message, containing no raw text or original IDs."""


class BudgetError(GateError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def collapse_ws(text):
    return re.sub(r'\s+', ' ', text).strip()


def normalize_key(text):
    # Deliberately the SAME order as the approved step-01 diagnostic.
    return collapse_ws(unicodedata.normalize('NFKC', text).casefold())


def group_id(text):
    # Do NOT include language, domain or label in this hash.
    return digest(normalize_key(text))


def rank_hash(seed, purpose, identity):
    return digest(canonical([int(seed), str(purpose), str(identity)]))


def read_json(path, limit=40 * 1024 * 1024):
    p = Path(path)
    if not p.is_file() or p.stat().st_size > limit:
        raise GateError('Required JSON is missing or exceeds the size limit.')
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(path, obj):
    p = Path(path)
    tmp = p.with_name(p.name + '.partial')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def write_csv(path, rows, fields):
    p = Path(path); tmp = p.with_name(p.name + '.partial')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


@contextmanager
def ro_db(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        yield db
    finally:
        db.close()


@contextmanager
def lock_output(out):
    # Linux/JupyterLab: advisory locking is released automatically on process exit.
    import fcntl
    with (out / '.split_gate.lock').open('a+') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GateError('Another split builder is using this output directory.')
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def explicit_label(lang, value):
    s = value.strip()
    if lang == 'tr':
        mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        if s.casefold() not in mapping:
            raise GateError('Unrecognized Turkish label; numeric schemes are not inferred.')
        return mapping[s.casefold()], None
    if lang == 'kz':
        if s not in {'1', '2', '3', '4', '5'}:
            raise GateError('Kazakh input must contain original integer ratings 1..5.')
        r = int(s)
        return (0 if r <= 2 else 1 if r == 3 else 2), r
    raise GateError('Unsupported language.')


def source_checks(audit):
    report = read_json(audit / 'data_audit_report.json')
    local = read_json(audit / 'source_manifest.local.json')
    status = read_json(audit / 'status.json')
    if report.get('schema_version') != '1.0.0' or report.get('state') != 'RAW_AUDIT_COMPLETE':
        raise GateError('Step 01 must have a complete compatible raw-data audit.')
    if status.get('report_sha256') != file_sha(audit / 'data_audit_report.json'):
        raise GateError('Step-01 report checksum does not match its completion record.')
    if any(c['status'] in {'BLOCKED', 'REVIEW'} for c in report['checks']):
        raise GateError('Step-01 REVIEW/BLOCKED checks require resolution before splitting.')
    for lang, spec in SPECS.items():
        public = report['sources'][lang]; s = local[lang]
        for k in ('repo_id', 'revision', 'sha256'):
            if public[k] != s[k]:
                raise GateError('Local source manifest differs from the audited source identity.')
        if s['repo_id'] != spec['repo'] or not re.fullmatch('[0-9a-f]{40}', s['revision']):
            raise GateError('Unexpected repository/revision; no fallback source selected.')
        if not Path(s['path']).is_file() or file_sha(s['path']) != s['sha256']:
            raise GateError(f'{lang}: audited CSV missing/changed; do not redownload silently.')
    return report, local


def snapshot_expected(report):
    return {r['language']: r for r in report['tables']['corpus_flow']}


def build_corpus(out, sources, report):
    final = out / 'corpus.local.sqlite3'
    done = out / 'corpus.complete.json'
    if done.exists():
        stamp = read_json(done)
        if not final.exists() or file_sha(final) != stamp['sha256']:
            raise GateError('Completed corpus checksum mismatch; refusing silent reuse/rebuild.')
        print('[REUSE] Verified local corpus.', flush=True)
        return stamp
    # An interrupted temporary DB belongs to this builder; only that file is replaced.
    tmp = out / ('corpus.local.sqlite3.partial_' + str(time.time_ns()))
    db = sqlite3.connect(tmp)
    expected = snapshot_expected(report)
    try:
        db.executescript('''PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
            CREATE TABLE records (
              row_id INTEGER PRIMARY KEY, sample_id TEXT NOT NULL UNIQUE,
              language TEXT NOT NULL, domain TEXT NOT NULL, label INTEGER NOT NULL,
              rating INTEGER, source_row INTEGER NOT NULL, native_id TEXT NOT NULL,
              raw_label TEXT NOT NULL, raw_text TEXT NOT NULL, model_text TEXT NOT NULL,
              group_id TEXT NOT NULL, nfc_hash TEXT NOT NULL, nfkc_hash TEXT NOT NULL,
              nfc_casefold_hash TEXT NOT NULL, exact_hash TEXT NOT NULL,
              UNIQUE(language,native_id));
            CREATE TABLE source_counts(language TEXT PRIMARY KEY,raw_rows INTEGER,
              eligible_rows INTEGER,short_rows INTEGER);''')
        rid = 0; counts = []
        db.execute('BEGIN')
        for lang in LANGS:
            spec = SPECS[lang]; source = sources[lang]
            raw_n = accepted = short_n = 0
            with Path(source['path']).open('r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f, strict=True)
                if set(reader.fieldnames or []) != set(spec['columns']) or len(reader.fieldnames) != len(spec['columns']):
                    raise GateError('Raw CSV columns differ from the audited schema.')
                batch = []
                for raw_n, r in enumerate(reader, 1):
                    if None in r or any(r.get(c) is None for c in spec['columns']):
                        raise GateError('Malformed CSV record; no record skipped silently.')
                    lab, rating = explicit_label(lang, r[spec['label']])
                    raw = r[spec['text']]; text = collapse_ws(raw)
                    native = r[spec['native_id']]
                    if not native.strip(): raise GateError('Missing original identifier.')
                    if lang == 'tr': domain = 'NOT_PROVIDED'
                    else:
                        domain = DOMAINS.get(r['domain'].strip().casefold())
                        if domain is None: raise GateError('Missing/unrecognized Kazakh domain.')
                    if len(text) < 3:
                        short_n += 1; continue
                    rid += 1; accepted += 1
                    sid = digest(canonical([source['repo_id'], source['revision'], lang, native]))
                    nfc = unicodedata.normalize('NFC', raw)
                    nfkc = unicodedata.normalize('NFKC', raw)
                    batch.append((rid, sid, lang, domain, lab, rating, raw_n, native,
                        r[spec['label']], raw, text, group_id(raw), digest(collapse_ws(nfc)),
                        digest(collapse_ws(nfkc)), digest(collapse_ws(nfc.casefold())), digest(raw)))
                    if len(batch) >= 5000:
                        db.executemany('INSERT INTO records VALUES (' + ','.join('?' * 16) + ')', batch); batch.clear()
                    if raw_n % 50000 == 0: print(f'[READ] {lang}: {raw_n:,}', flush=True)
                if batch: db.executemany('INSERT INTO records VALUES (' + ','.join('?' * 16) + ')', batch)
            if raw_n != expected[lang]['raw_rows'] or accepted != expected[lang]['eligible_before_any_dedup_or_cap']:
                raise GateError('Record counts differ from step 01; investigate before splitting.')
            if file_sha(source['path']) != source['sha256']:
                raise GateError('Raw source changed while building the corpus.')
            db.execute('INSERT INTO source_counts VALUES (?,?,?,?)', (lang, raw_n, accepted, short_n))
            counts.append(dict(language=lang, raw_rows=raw_n, eligible_rows=accepted, short_rows=short_n))
        db.commit()
        db.executescript('''CREATE INDEX rec_group ON records(group_id);
            CREATE INDEX rec_scope ON records(language,domain,label);
            CREATE TABLE groups AS SELECT group_id,COUNT(*) n_rows,COUNT(DISTINCT label) n_labels,
              COUNT(DISTINCT language) n_languages,COUNT(DISTINCT domain) n_domains,
              COUNT(DISTINCT nfc_hash) n_nfc,COUNT(DISTINCT nfkc_hash) n_nfkc,
              COUNT(DISTINCT nfc_casefold_hash) n_nfc_casefold FROM records GROUP BY group_id;
            CREATE UNIQUE INDEX group_key ON groups(group_id);''')
        observed = db.execute('SELECT COUNT(*) FROM groups').fetchone()[0]
        audited = next(r['groups'] for r in report['tables']['text_group_audit']
                       if r['scope'] == 'all' and r['normalization'] == 'nfkc_casefold_ws_diagnostic')
        if observed != audited: raise GateError('Approved grouping count differs from the raw audit.')
        db.commit()
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok': raise GateError('Corpus integrity check failed.')
    except BaseException:
        db.rollback(); raise
    finally:
        db.close()
    os.replace(tmp, final)
    stamp = dict(sha256=file_sha(final), eligible_rows=rid, groups=observed,
                 source_counts=counts, raw_records_deleted=False,
                 ineligible_rows_excluded_from_modeling=True, labels_changed=False)
    write_json(done, stamp)
    return stamp


def rounding_quotas(n, proportions):
    raw = [n * x for x in proportions]
    result = [int(math.floor(x)) for x in raw]
    for j in sorted(range(len(raw)), key=lambda j: (-(raw[j] - result[j]), j))[:n - sum(result)]:
        result[j] += 1
    return result


def grouped_allocate(hist, proportions, seed, purpose):
    """Custom deterministic stratified allocation; NOT StratifiedGroupKFold.

    hist: group -> Counter of (language,domain,label) record counts.
    Multi-record groups are indivisible and processed by descending largest
    stratum-relative group mass; hash breaks ties. Assignment minimizes increase
    in sum((allocated-target)^2/max(target,1)). Singleton groups fill integer
    residual quotas exactly when possible. No fitted models or scores are used.
    """
    if not hist or not math.isclose(sum(proportions), 1.0, abs_tol=1e-12) or min(proportions) <= 0:
        raise GateError('Empty groups or invalid split proportions.')
    totals = Counter()
    for h in hist.values(): totals.update(h)
    strata = sorted(totals)
    targets = {s: rounding_quotas(totals[s], proportions) for s in strata}
    current = {s: [0] * len(proportions) for s in strata}
    multi = []; singles = defaultdict(list); assign = {}
    for gid, h in hist.items():
        n = sum(h.values())
        if n == 1: singles[next(iter(h))].append(gid)
        else: multi.append(gid)
    multi.sort(key=lambda g: (-max(n / totals[s] for s, n in hist[g].items()),
                             -sum(hist[g].values()), rank_hash(seed, purpose, g)))
    for gid in multi:
        h = hist[gid]
        costs = []
        for j in range(len(proportions)):
            delta = sum((2*n*(current[s][j]-targets[s][j]) + n*n) / max(targets[s][j], 1)
                        for s, n in h.items())
            costs.append((delta, rank_hash(seed, purpose + ':tie', gid + ':' + str(j)), j))
        j = min(costs)[2]; assign[gid] = j
        for s, n in h.items(): current[s][j] += n
    for s in strata:
        ids = sorted(singles[s], key=lambda g: rank_hash(seed, purpose + ':single', g))
        deficits = [targets[s][j] - current[s][j] for j in range(len(proportions))]
        if min(deficits) >= 0:
            if sum(deficits) != len(ids): raise GateError('Internal quota accounting failure.')
            offset = 0
            for j, n in enumerate(deficits):
                for gid in ids[offset:offset+n]: assign[gid] = j
                current[s][j] += n; offset += n
        else:
            # Large groups can make exact proportions impossible; never split/drop them.
            for gid in ids:
                j = min(range(len(proportions)), key=lambda k:
                        ((2*(current[s][k]-targets[s][k]) + 1)/max(targets[s][k],1), k))
                assign[gid] = j; current[s][j] += 1
    if len(assign) != len(hist): raise GateError('Some groups were not assigned.')
    metrics = []
    for s in strata:
        for j in range(len(proportions)):
            metrics.append(dict(language=s[0], domain=s[1], label=LABELS[s[2]], partition_index=j,
                                target_rows=targets[s][j], actual_rows=current[s][j],
                                deviation_rows=current[s][j]-targets[s][j], stratum_total=totals[s]))
    return assign, metrics


def corpus_arrays(path):
    with ro_db(path) as db:
        # Immutable arrays: rid,sample_id,language,domain,label,rating,group_id.
        rows = list(db.execute('SELECT row_id,sample_id,language,domain,label,rating,group_id FROM records ORDER BY row_id'))
    hist = defaultdict(Counter)
    for r in rows: hist[r[6]][(r[2], r[3], r[4])] += 1
    return rows, dict(hist)


def choose_rows(rows, n, seed, purpose, exact=True):
    """Random-in-order deterministic record subsampling WITHIN a preassigned role."""
    if n < 0: raise GateError('Negative sample budget.')
    if exact and len(rows) < n:
        raise BudgetError(f'Budget unavailable: need {n}, have {len(rows)}. No reduction applied.')
    n = min(n, len(rows))
    ordered = sorted(rows, key=lambda r: rank_hash(seed, purpose, r[1]))
    return ordered[:n]


def split_normalization_report(corpus, out):
    with ro_db(corpus) as db:
        summary = dict(zip(['groups','repeated_groups','conflicting_label_groups','cross_domain_groups','largest_group_rows'],
                    db.execute('SELECT COUNT(*),SUM(n_rows>1),SUM(n_labels>1),SUM(n_domains>1),MAX(n_rows) FROM groups').fetchone()))
        categories = Counter()
        for nnfc, nnfkc, ncf, nrows in db.execute('SELECT n_nfc,n_nfkc,n_nfc_casefold,n_rows FROM groups WHERE n_nfc>1'):
            category = ('either_transformation_suffices' if nnfkc == 1 and ncf == 1 else
                        'casefold_sufficient' if ncf == 1 else
                        'compatibility_normalization_sufficient' if nnfkc == 1 else
                        'combined_operations_required')
            categories[category] += 1
        summary['merged_nfc_group_categories'] = dict(categories)
        summary['nfc_groups'] = db.execute('SELECT COUNT(DISTINCT nfc_hash) FROM records').fetchone()[0]
        summary['row_retention'] = 'All eligible records retained in corpus and base split.'
        summary['interpretation'] = 'Matching-key grouping, not semantic equivalence, human label validation or exhaustive near-deduplication.'
        # PRIVATE spot-check file; deliberately omitted from the shareable ZIP.
        p = out / 'normalization_examples.PRIVATE.jsonl'
        with p.open('w', encoding='utf-8') as f:
            gids = [r[0] for r in db.execute('SELECT group_id FROM groups WHERE n_nfc>1 ORDER BY n_rows DESC,group_id LIMIT 20')]
            for gid in gids:
                examples = list(db.execute('SELECT language,domain,label,raw_text FROM records WHERE group_id=? GROUP BY raw_text,language,domain,label LIMIT 5', (gid,)))
                f.write(json.dumps({'group_id': gid, 'examples': [dict(language=r[0],domain=r[1],label=LABELS[r[2]],raw_text=r[3]) for r in examples]}, ensure_ascii=False)+'\n')
    write_json(out / 'normalization_audit.json', summary)
    return summary


def build_seed(out, corpus, rows, hist, seed, policy):
    seed_dir = out / f'seed_{seed}'
    seed_dir.mkdir(exist_ok=True)
    final = seed_dir / 'memberships.local.sqlite3'
    complete = seed_dir / 'complete.json'
    if complete.exists():
        stamp = read_json(complete)
        if not final.exists() or file_sha(final) != stamp['db_sha256']:
            raise GateError(f'Seed {seed}: completed membership checksum mismatch.')
        if file_sha(seed_dir/'seed_report.json') != stamp['report_sha256']:
            raise GateError(f'Seed {seed}: completed report checksum mismatch.')
        if file_sha(seed_dir/'experiments.json') != stamp['recipes_sha256']:
            raise GateError(f'Seed {seed}: experiment manifest checksum mismatch.')
        for name, expected_sha in stamp['table_checksums'].items():
            if file_sha(seed_dir/name) != expected_sha:
                raise GateError(f'Seed {seed}: completed diagnostic table checksum mismatch.')
        print(f'[REUSE] seed={seed}: verified completed split and budget manifests.', flush=True)
        return read_json(seed_dir / 'seed_report.json')
    print(f'[SPLIT] seed={seed}: global language/domain/class groups...', flush=True)
    main_names = ('train', 'validation', 'test')
    main, main_counts = grouped_allocate(hist, [policy['main_split'][x] for x in main_names], seed, 'main')
    dev_hist = {g: h for g, h in hist.items() if main[g] != 2}
    low, low_counts = grouped_allocate(dev_hist, [policy['low_train_fraction'], 1-policy['low_train_fraction']], seed, 'low_dev')
    if set(low) != set(dev_hist): raise GateError('Low-resource development groups differ from non-test pool.')
    for r in main_counts: r['partition'] = main_names[r.pop('partition_index')]
    for r in low_counts: r['partition'] = ('train','validation')[r.pop('partition_index')]
    buckets = defaultdict(list)
    for r in rows:
        buckets[('main', main[r[6]], r[2], r[4])].append(r)
        if main[r[6]] != 2: buckets[('low', low[r[6]], r[2], r[4])].append(r)
    tmp = seed_dir / ('memberships.local.sqlite3.partial_' + str(time.time_ns()))
    db = sqlite3.connect(tmp)
    checks = []; setstats = []; budgets = []; errors = []; recipes = []; sets_memory = {}
    def check(name, ok, detail=None):
        checks.append(dict(check=name, status='PASS' if ok else 'BLOCKED', details=detail))
        if not ok: errors.append(name)
    def add_set(name, records, role, task='three_class', expected=None):
        if name in sets_memory: raise GateError('Duplicate subset name.')
        ids = [r[0] for r in records]
        if len(ids) != len(set(ids)): raise GateError('Duplicate sample membership detected.')
        records = sorted(records, key=lambda r: r[0])
        targets = [(r[0], r[4] if task=='three_class' else 0 if r[4]==0 else 1) for r in records]
        if task != 'three_class' and any(r[4] not in (0,2) for r in records):
            raise GateError('Neutral record entered a binary subset.')
        h = hashlib.sha256()
        for r, (_, lab) in zip(records, targets): h.update((r[1]+':'+str(lab)+'\n').encode())
        labels = Counter(LABELS[r[4]] for r in records)
        perlang = Counter(r[2] for r in records)
        groups = {r[6] for r in records}
        stat = dict(set_id=name,role=role,task=task,rows=len(records),groups=len(groups),
                    label_counts=dict(labels),language_counts=dict(perlang),manifest_sha256=h.hexdigest())
        db.execute('INSERT INTO sets VALUES (?,?,?,?,?)',(name,role,task,len(records),canonical(stat)))
        db.executemany('INSERT INTO members VALUES (?,?,?)', ((name,rid,lab) for rid,lab in targets))
        sets_memory[name] = records; setstats.append(stat)
        if expected is not None: check(name+':exact_budget', len(records)==expected, dict(expected=expected,actual=len(records)))
        return name
    def add_recipe(name,train,val,test,context,source_langs,joint=False,reuse=None):
        # Validate the ACTUAL selected records, including cross-domain groups.
        a,b,c = (sets_memory[x] for x in (train,val,test))
        ga,gb,gc = ({r[6] for r in part} for part in (a,b,c))
        overlap = dict(train_validation=len(ga&gb),train_test=len(ga&gc),validation_test=len(gb&gc))
        check(name+':group_disjoint',not any(overlap.values()),overlap)
        expected_classes = {0,1,2} if name.split(':')[0] not in ('binary','extreme') else {0,2}
        check(name+':all_classes', all({r[4] for r in part}==expected_classes for part in (a,b,c)))
        check(name+':source_only_selection',set(r[2] for r in b)==set(source_langs))
        check(name+':training_languages',set(r[2] for r in a)==set(source_langs))
        recipes.append(dict(experiment_id=name,training_context_id=context,train_set=train,
            validation_set=val,test_set=test,reuse_fitted_model_from=reuse,
            validation_languages=list(source_langs),
            selection_metric='equal_language_mean_macro_f1' if joint else 'source_macro_f1',
            test_labels_must_not_select_models=True))
    try:
        db.executescript('''PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
            CREATE TABLE group_roles(group_id TEXT PRIMARY KEY,main_role TEXT NOT NULL,low_role TEXT NOT NULL);
            CREATE TABLE sets(set_id TEXT PRIMARY KEY,role TEXT NOT NULL,task TEXT NOT NULL,n_rows INTEGER NOT NULL,metadata_json TEXT NOT NULL);
            CREATE TABLE members(set_id TEXT NOT NULL,row_id INTEGER NOT NULL,target INTEGER NOT NULL,
                PRIMARY KEY(set_id,row_id),FOREIGN KEY(set_id) REFERENCES sets(set_id));''')
        db.execute('BEGIN')
        db.executemany('INSERT INTO group_roles VALUES (?,?,?)',
            ((g, main_names[main[g]], 'test' if main[g]==2 else ('train','validation')[low[g]]) for g in sorted(main)))
        check('all_groups_assigned',len(main)==len(hist))
        check('low_outer_test_unchanged',all(g not in low for g in main if main[g]==2))
        # Main task: cap only training, independently per language/class. Keep ALL validation/test rows.
        for lang in LANGS:
            parts = {role: [] for role in main_names}
            for label in (0,1,2):
                pool=buckets[('main',0,lang,label)]; cap=policy['main_train_cap_per_class_per_language']
                selected=choose_rows(pool,cap,seed,f'main_cap:{lang}:{label}',exact=False)
                budgets.append(dict(family='main',language=lang,label=LABELS[label],role='train',
                                    requested=cap,actual=len(selected),available=len(pool),mode='UPPER_CAP'))
                parts['train'].extend(selected)
                parts['validation'].extend(buckets[('main',1,lang,label)])
                parts['test'].extend(buckets[('main',2,lang,label)])
            for role in main_names: add_set(f'main:{lang}:{role}',parts[role],role)
        for role in ('train','validation'):
            add_set(f'main:joint:{role}',sets_memory[f'main:tr:{role}']+sets_memory[f'main:kz:{role}'],role)
        for source in ('tr','kz','joint'):
            for target in LANGS:
                add_recipe(f'main:{source}_to_{target}',f'main:{source}:train',f'main:{source}:validation',f'main:{target}:test',
                           f'main:{source}',LANGS if source=='joint' else (source,),source=='joint')
        # Binary task is the same selected main training records with neutrals removed.
        for source in ('tr','kz','joint'):
            for role in (main_names if source!='joint' else ('train','validation')):
                add_set(f'binary:{source}:{role}',[r for r in sets_memory[f'main:{source}:{role}'] if r[4]!=1],role,'binary')
        extreme = [r for r in sets_memory['binary:kz:test'] if r[5] in (1,5)]
        add_set('extreme:kz:test',extreme,'test','binary')
        for source in ('tr','kz','joint'):
            for target in LANGS:
                add_recipe(f'binary:{source}_to_{target}',f'binary:{source}:train',f'binary:{source}:validation',f'binary:{target}:test',
                           f'binary:{source}',LANGS if source=='joint' else (source,),source=='joint')
            add_recipe(f'extreme:{source}_to_kz',f'binary:{source}:train',f'binary:{source}:validation','extreme:kz:test',
                       f'binary:{source}',LANGS if source=='joint' else (source,),source=='joint',f'binary:{source}_to_kz')
        # Rank once per LOW role/class/lang; changing n never reshuffles existing prefixes.
        low_ranks={k: sorted(v,key=lambda r:rank_hash(seed,f'low:{k[1]}:{k[2]}:{k[3]}',r[1]))
                   for k,v in buckets.items() if k[0]=='low'}
        previous={}
        for n in policy['low_total_labels_per_class_per_language']:
            ntrain=n*4//5; nval=n-ntrain; current={}
            feasible=True
            for lang in LANGS:
                for role,role_idx,want in [('train',0,ntrain),('validation',1,nval)]:
                    picked=[]
                    for label in (0,1,2):
                        pool=low_ranks.get(('low',role_idx,lang,label),[])
                        ok=len(pool)>=want
                        budgets.append(dict(family=f'low_n{n}',language=lang,label=LABELS[label],role=role,
                                            requested=want,actual=want if ok else None,available=len(pool),mode='EXACT'))
                        if not ok:
                            check(f'low_n{n}:{lang}:{role}:{label}:capacity',False,dict(requested=want,available=len(pool)))
                            feasible=False
                        picked.extend(pool[:want] if ok else [])
                    current[(lang,role)]=picked
            if not feasible: continue  # NO incomplete-budget subsets are emitted.
            for (lang,role),picked in current.items():
                name=f'low_n{n}:{lang}:{role}'
                add_set(name,picked,role,expected=3*(ntrain if role=='train' else nval))
                old=previous.get((lang,role),set()); new={r[0] for r in picked}
                check(name+':nested',old<=new)
                previous[(lang,role)]=new
            for role in ('train','validation'):
                add_set(f'low_n{n}:joint:{role}',current[('tr',role)]+current[('kz',role)],role)
            for source in ('tr','kz','joint'):
                for target in LANGS:
                    add_recipe(f'low:{n}:{source}_to_{target}',f'low_n{n}:{source}:train',f'low_n{n}:{source}:validation',f'main:{target}:test',
                               f'low:{n}:{source}',LANGS if source=='joint' else (source,),source=='joint')
        # Domain subset groups inherit main roles; training and validation counts are matched.
        scope = {'TRSAv1':('tr',None),'KZ_Market':('kz','Market'),'KZ_Appstore':('kz','Appstore')}
        domain_picks={}; feasible=True
        for name,(lang,domain) in scope.items():
            for role,role_idx,want in [('train',0,policy['domain_train_per_class']),('validation',1,policy['domain_validation_per_class'])]:
                picked=[]
                for label in (0,1,2):
                    pool=[r for r in buckets[('main',role_idx,lang,label)] if domain is None or r[3]==domain]
                    ok=len(pool)>=want
                    budgets.append(dict(family='domain:'+name,language=lang,label=LABELS[label],role=role,
                                        requested=want,actual=want if ok else None,available=len(pool),mode='EXACT'))
                    if not ok:
                        check(f'domain:{name}:{role}:{label}:capacity',False,dict(requested=want,available=len(pool)));feasible=False
                    else:picked.extend(choose_rows(pool,want,seed,f'domain:{name}:{role}:{label}'))
                domain_picks[(name,role)]=picked
            domain_picks[(name,'test')]=[r for r in sets_memory[f'main:{lang}:test'] if domain is None or r[3]==domain]
        if feasible:
            for (name,role),picked in domain_picks.items():
                expected=3*(policy['domain_train_per_class'] if role=='train' else policy['domain_validation_per_class']) if role!='test' else None
                add_set(f'domain:{name}:{role}',picked,role,expected=expected)
            for source,(lang,_) in scope.items():
                for target in scope:
                    add_recipe(f'domain:{source}_to_{target}',f'domain:{source}:train',f'domain:{source}:validation',f'domain:{target}:test',
                               f'domain:{source}',(lang,))
        db.commit()
        # Independent SQL validation, not only the in-memory allocator's own checks.
        db.execute('ATTACH DATABASE ? AS corpus', (corpus.resolve().as_uri()+'?mode=ro',))
        n_orphan=db.execute('SELECT COUNT(*) FROM members m LEFT JOIN corpus.records r ON m.row_id=r.row_id WHERE r.row_id IS NULL').fetchone()[0]
        check('membership_ids_exist',n_orphan==0,n_orphan)
        covered=db.execute('SELECT COUNT(*) FROM corpus.records r JOIN group_roles g ON r.group_id=g.group_id').fetchone()[0]
        check('every_eligible_record_has_base_partition',covered==len(rows),dict(expected=len(rows),actual=covered))
        low_test_conflict=db.execute("SELECT COUNT(*) FROM group_roles WHERE (main_role='test') != (low_role='test')").fetchone()[0]
        check('sql_low_main_test_identity',low_test_conflict==0,low_test_conflict)
        wrong_label=db.execute('''SELECT COUNT(*) FROM members m JOIN sets s USING(set_id) JOIN corpus.records r USING(row_id)
          WHERE (s.task='three_class' AND m.target!=r.label)
          OR (s.task='binary' AND (r.label=1 OR m.target!=CASE r.label WHEN 0 THEN 0 ELSE 1 END))''').fetchone()[0]
        check('sql_all_task_label_mappings',wrong_label==0,wrong_label)
        wrong_role=db.execute('''SELECT COUNT(*) FROM members m JOIN sets s USING(set_id)
          JOIN corpus.records r USING(row_id) JOIN group_roles g USING(group_id)
          WHERE s.role != CASE WHEN substr(s.set_id,1,5)='low_n' THEN g.low_role ELSE g.main_role END''').fetchone()[0]
        check('sql_memberships_respect_assigned_roles',wrong_role==0,wrong_role)
        for stat in setstats:
            h=hashlib.sha256(); ncheck=0
            for sid,target in db.execute('''SELECT r.sample_id,m.target FROM members m
                JOIN corpus.records r USING(row_id) WHERE set_id=? ORDER BY r.row_id''',(stat['set_id'],)):
                h.update((sid+':'+str(target)+'\n').encode()); ncheck+=1
            check(stat['set_id']+':persisted_checksum',
                  ncheck==stat['rows'] and h.hexdigest()==stat['manifest_sha256'])
        check('sqlite_integrity',db.execute('PRAGMA quick_check').fetchone()[0]=='ok')
        db.execute('DETACH DATABASE corpus')
    except BaseException:
        db.rollback(); raise
    finally:db.close()
    os.replace(tmp,final)
    status='SPLIT_MANIFESTS_READY' if not errors else 'REVIEW_REQUIRED'
    result=dict(seed=seed,state=status,training_allowed=False,checks=checks,budgets=budgets,
                sets=setstats,main_strata=main_counts,low_pool_strata=low_counts,
                experiments=len(recipes),errors=errors,
                note='Main and low development roles differ by design; source/test fit isolation must also pass model-pipeline tests.')
    write_json(seed_dir/'seed_report.json',result)
    write_json(seed_dir/'experiments.json',recipes)
    write_csv(seed_dir/'main_strata.csv',main_counts,list(main_counts[0]))
    write_csv(seed_dir/'low_pool_strata.csv',low_counts,list(low_counts[0]))
    write_csv(seed_dir/'budgets.csv',budgets,list(budgets[0]))
    write_json(complete,dict(seed=seed,state=status,db_sha256=file_sha(final),
                            report_sha256=file_sha(seed_dir/'seed_report.json'),recipes_sha256=file_sha(seed_dir/'experiments.json'),
                            table_checksums={name:file_sha(seed_dir/name) for name in ('main_strata.csv','low_pool_strata.csv','budgets.csv')}))
    print(f'[SEED {seed}] {status}; sets={len(setstats)}; recipes={len(recipes)}; blocked_checks={len(errors)}',flush=True)
    return result


def iter_set(output,seed,set_id):
    """Read private records for a named subset. Never include these in shareable reports.

    Downstream model pipelines can use this helper instead of re-sampling. The
    returned target is task-specific; label is always the original 3-class mapping.
    Call only after reviewing SPLIT_MANIFESTS_READY and completing pipeline tests.
    """
    output=Path(output)
    with ro_db(output/f'seed_{int(seed)}'/'memberships.local.sqlite3') as db:
        if not db.execute('SELECT 1 FROM sets WHERE set_id=?',(set_id,)).fetchone():raise GateError('Unknown set_id.')
        db.execute('ATTACH DATABASE ? AS corpus',( (output/'corpus.local.sqlite3').resolve().as_uri()+'?mode=ro',))
        db.row_factory=sqlite3.Row
        cursor=db.execute('''SELECT r.*,m.target FROM members m JOIN corpus.records r USING(row_id)
                             WHERE m.set_id=? ORDER BY r.row_id''',(set_id,))
        for row in cursor: yield dict(row)


def package_report(out,results,normalization,identity):
    state='SPLIT_MANIFESTS_READY' if all(x['state']=='SPLIT_MANIFESTS_READY' for x in results) else 'REVIEW_REQUIRED'
    totalchecks=sum(len(x['checks']) for x in results)
    report=dict(schema_version=VERSION,state=state,training_allowed=False,seeds=[x['seed'] for x in results],
                protocol_id=identity['protocol_id'],policy=identity['policy'],normalization=normalization,
                source_fingerprints=identity['source_fingerprints'],checks_count=totalchecks,
                blocked_checks=sum(c['status']=='BLOCKED' for x in results for c in x['checks']),
                seed_states={str(x['seed']):x['state'] for x in results},
                runtime=dict(python=platform.python_version(),unicode=unicodedata.unidata_version,sqlite=sqlite3.sqlite_version),
                limitations=['No model training, vocabulary-fit isolation test, human annotation or CUDA test performed.',
                 'Group key excludes language/domain/label; exact semantic equivalence and exhaustive near-duplicate removal are not claimed.',
                 'Record budgets are exact within disjoint role pools; a budget may include only some records from a group.',
                 'Group stratification uses metadata labels solely for partitioning, not for tuning/model selection.',
                 'Split ratios are approximate when groups make exact counts infeasible.',
                 'Private corpus includes original texts/IDs; never publish it automatically.'])
    write_json(out/'split_report.json',report)
    lines=['SAUCIS REVIZYONU — GRUP BAGIMSIZ SPLIT VE BUTCE DENETIMI','='*64,
           f'Durum: {state}',f'Protocol: {identity["protocol_id"]}',
           'Gruplama: NFKC -> casefold -> bosluk; yalnizca bolunme anahtari.',
           'Ham/model metni casefold edilmedi; etiketler degistirilmedi.',
           'Ayni grubun kayitlari global olarak tek bolumde; domainler ayri bolunmedi.',
           'Model egitimi, indirme, GPU kullanimi veya legacy dosya degisikligi YOK.','']
    for x in results:
        lines += [f'Seed {x["seed"]}: {x["state"]}; {len(x["sets"])} subset; {x["experiments"]} evaluation recipes']
        for s in x['sets']:
            if s['set_id'].startswith('main:') and ':joint:' not in s['set_id']:
                lines.append(f'  {s["set_id"]}: rows={s["rows"]:,}, groups={s["groups"]:,}, labels={s["label_counts"]}')
        if x['errors']:lines.append('  BLOCKED: '+', '.join(x['errors']))
    lines += ['',f'Kontroller: {totalchecks}; BLOCKED: {report["blocked_checks"]}',
              'Bu asama egitim izni DEGIL; model pipeline ve CUDA smoke kontrolleri sirada.',
              'Kesinti: ayni komut dogrulanmis biten seedleri atlar; yarim seed yeniden olusturulur.',
              'PAYLAS: shareable_split_audit.zip. PRIVATE/local SQLite dosyalarini paylasma.']
    (out/'split_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    partial=out/'shareable_split_audit.zip.partial'
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('split_report.json','split_summary.txt','normalization_audit.json','protocol.json'):
            z.write(out/name,name)
        for x in results:
            prefix=f'seed_{x["seed"]}'
            for name in ('seed_report.json','experiments.json','main_strata.csv','low_pool_strata.csv','budgets.csv'):
                z.write(out/prefix/name,prefix+'/'+name)
    os.replace(partial,out/'shareable_split_audit.zip')
    write_json(out/'status.json',dict(state=state,training_allowed=False,completed_at_utc=now(),
                                    report_sha256=file_sha(out/'split_report.json')))
    print('\n'.join(lines),flush=True)
    print('[SHARE]',out/'shareable_split_audit.zip',flush=True)
    return state


def run(project,audit,resume=None,policy=None):
    start=time.monotonic()
    policy=json.loads(canonical(DEFAULT_POLICY if policy is None else policy))
    report,sources=source_checks(audit)
    public={lang:{k:s[k] for k in ('repo_id','revision','sha256')} for lang,s in sources.items()}
    basis=dict(code_version=VERSION,code_sha256=file_sha(Path(__file__)),policy=policy,
               source_fingerprints=public,unicode_version=unicodedata.unidata_version,
               audit_report_sha256=file_sha(audit/'data_audit_report.json'))
    identity={**basis,'protocol_id':digest(canonical(basis))[:20]}
    if resume:
        out=resume.resolve()
        if read_json(out/'identity.json')!=identity:raise GateError('Resume protocol/code/source mismatch.')
    else:
        out=project/'_revision_data'/('splits_'+identity['protocol_id'])
        out.mkdir(parents=True,exist_ok=True)
    with lock_output(out):
        if (out/'identity.json').exists():
            if read_json(out/'identity.json')!=identity:raise GateError('Output identity mismatch.')
        else:write_json(out/'identity.json',identity)
        write_json(out/'protocol.json',dict(protocol_id=identity['protocol_id'],policy=policy,source_fingerprints=public,
                                          code_sha256=basis['code_sha256'],unicode_version=unicodedata.unidata_version))
        write_json(out/'status.json',dict(state='RUNNING',training_allowed=False,started_at_utc=now()))
        print('[OUTPUT]',out,flush=True)
        try:
            build_corpus(out,sources,report)
            normalization=split_normalization_report(out/'corpus.local.sqlite3',out)
            rows,hist=corpus_arrays(out/'corpus.local.sqlite3')
            results=[]
            for seed in policy['seeds']:results.append(build_seed(out,out/'corpus.local.sqlite3',rows,hist,seed,policy))
            state=package_report(out,results,normalization,identity)
            print(f'[ELAPSED] {time.monotonic()-start:.1f} seconds',flush=True)
            return out,state
        except BaseException as exc:
            safe=str(exc) if isinstance(exc,GateError) else 'Interrupted or failed; no raw content printed. Use the same command to resume.'
            write_json(out/'status.json',dict(state='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                                            training_allowed=False,error_type=type(exc).__name__,safe_message=safe))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project',default='.')
    p.add_argument('--audit',required=True,help='LOCAL step-01 output directory (not shareable ZIP).')
    p.add_argument('--resume',default=None,help='Optional exact prior split output; identity must match.')
    args=p.parse_args();csv.field_size_limit(min(sys.maxsize,100_000_000))
    project=Path(args.project).expanduser().resolve();audit=Path(args.audit).expanduser().resolve()
    try:
        _,state=run(project,audit,Path(args.resume).expanduser() if args.resume else None)
        return 0 if state=='SPLIT_MANIFESTS_READY' else 2
    except KeyboardInterrupt:
        print('[STOP] Controlled interruption; rerun the same command to resume.',file=sys.stderr);return 130
    except GateError as exc:
        print('[STOP]',exc,file=sys.stderr);return 2
    except Exception as exc:
        print('[STOP]',type(exc).__name__,'See status.json; raw data not printed.',file=sys.stderr);return 3


if __name__=='__main__':
    raise SystemExit(main())
