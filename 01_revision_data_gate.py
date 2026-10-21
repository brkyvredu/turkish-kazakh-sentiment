#!/usr/bin/env python3
"""Offline full-row data audit for the approved SAUCIS revision.

Reads ONLY the two raw CSV snapshots identified by a preflight report.  Does
not install/download packages, import torch, train, split the data, overwrite
project files, or read token values.  Outputs go to a NEW audit directory.

No review text or native record ID is exported. The local SQLite ledger stores
hashes and descriptive measurements; it is NOT a training dataset. Counts under
different text normalizations are diagnostics, not automatic cleaning decisions.

Usage:
  python 01_revision_data_gate.py --project . --preflight PATH/preflight_report.json
  python 01_revision_data_gate.py --project . --preflight PATH/preflight_report.json \
      --resume _revision_audit/data_gate_TIMESTAMP

Python >= 3.10, standard library only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import time
import unicodedata
import zipfile

VERSION = '1.0.0'
LABEL_NAMES = {0: 'negative', 1: 'neutral', 2: 'positive'}
SPECS = {
    'tr': {
        'repo': 'maydogan/Turkish_SentimentAnalysis_TRSAv1',
        'suffix': 'TRSAv1.csv', 'text': 'review', 'label': 'score', 'id': 'id',
        'domain': None, 'columns': ['id', 'score', 'review'],
    },
    'kz': {
        'repo': 'issai/kazsandra', 'suffix': 'full/full.csv',
        'text': 'text', 'label': 'label', 'id': 'custom_id', 'domain': 'domain',
        'columns': ['custom_id', 'text', 'label', 'domain'],
    },
}
KNOWN_DOMAINS = {'appstore': 'Appstore', 'market': 'Market',
                 'mapping': 'Mapping', 'bookstore': 'Bookstore'}
MIN_CHARS = 3  # existing study threshold; audit view only, no source deletion
NORMALIZATIONS = {
    'exact': 'Raw CSV cell, without modification.',
    'nfc_ws': 'Unicode NFC + collapse whitespace + strip; case/accents/punctuation retained.',
    'nfkc_casefold_ws_diagnostic':
        'Unicode NFKC + casefold + collapse whitespace; diagnostic ONLY, not approved filtering.',
}


class AuditError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode('utf-8')).digest()


def json_write(path: Path, obj) -> None:
    """Atomic write; used only inside the new dedicated output directory."""
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else ['status']
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def collapse_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def group_keys(text: str) -> tuple[bytes, bytes, bytes]:
    return (digest(text), digest(collapse_ws(unicodedata.normalize('NFC', text))),
            digest(collapse_ws(unicodedata.normalize('NFKC', text).casefold())))


def parse_label(language: str, value: str) -> tuple[int | None, int | None, str]:
    """Explicit schemes; NEVER infer 0/1/2 vs 1..5 from overlapping numbers."""
    s = value.strip()
    if language == 'tr':
        found = {'negative': 0, 'neutral': 1, 'positive': 2}.get(s.casefold())
        return found, None, s if found is not None else 'UNRECOGNIZED'
    if language == 'kz':
        if s not in {'1', '2', '3', '4', '5'}:
            return None, None, 'UNRECOGNIZED'
        rating = int(s)
        return (0 if rating <= 2 else 1 if rating == 3 else 2), rating, s
    raise AuditError('Unsupported language scheme.')


def parse_domain(language: str, value: str | None) -> tuple[str, bool]:
    if language == 'tr':
        return 'NOT_PROVIDED', False  # do not fabricate a Turkish item-level domain
    if value is None or not value.strip():
        return 'MISSING', True
    key = value.strip().casefold()
    if key in KNOWN_DOMAINS:
        return KNOWN_DOMAINS[key], False
    return 'UNRECOGNIZED_' + hashlib.sha256(value.encode('utf-8')).hexdigest()[:12], True


@functools.lru_cache(maxsize=8192)
def script_character(ch: str) -> int:
    if not ch.isalpha():
        return 0
    name = unicodedata.name(ch, '')
    if 'LATIN' in name:
        return 1
    if 'CYRILLIC' in name:
        return 2
    return 3


def text_features(text: str) -> tuple[int, int, int, int, int, str]:
    lat = cyr = other = 0
    for ch in text:
        kind = script_character(ch)
        lat += kind == 1
        cyr += kind == 2
        other += kind == 3
    parts = []
    if lat:
        parts.append('latin')
    if cyr:
        parts.append('cyrillic')
    if other:
        parts.append('other')
    tag = '+'.join(parts) if parts else 'no_letters'
    return len(text), len(text.split()), lat, cyr, other, tag


def source_selection(preflight: dict) -> dict[str, dict]:
    """Exact report-backed paths; no search/download/latest-version fallback."""
    found = {}
    for language, spec in SPECS.items():
        candidates = []
        for inv in preflight.get('inventories', []):
            if inv.get('kind') != 'hf_dataset' or inv.get('repo_id') != spec['repo']:
                continue
            for item in inv.get('files', []):
                rel = item.get('relative_path', '')
                match = re.fullmatch(r'snapshots/([0-9a-f]{40})/' + re.escape(spec['suffix']), rel)
                if match:
                    candidates.append({'repo_id': spec['repo'], 'revision': match.group(1),
                        'path': str(Path(inv['path']) / rel), 'reported_bytes': item['bytes']})
        unique = {x['path']: x for x in candidates}
        if len(unique) != 1:
            raise AuditError(f'{language}: expected one pinned raw CSV, found {len(unique)}; no automatic selection.')
        item = next(iter(unique.values()))
        p = Path(item['path'])
        if not p.is_file():
            raise AuditError(f'{language}: pinned CSV is no longer readable; no fallback/download attempted.')
        if p.stat().st_size != item['reported_bytes']:
            raise AuditError(f'{language}: file size changed since preflight; regenerate inventory or investigate.')
        item['sha256'] = sha_file(p)
        item['bytes'] = p.stat().st_size
        item['language'] = language
        item['metadata_count'] = None
        for meta in preflight.get('dataset_metadata', []):
            if item['revision'] in meta.get('path', ''):
                item['metadata_count'] = meta.get('split_counts', {}).get('train', {}).get('num_examples')
        found[language] = item
    return found


def connect_ledger(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-65536')
    db.execute('''CREATE TABLE IF NOT EXISTS records (
        language TEXT NOT NULL, source_row INTEGER NOT NULL,
        original_id_hash BLOB, exact_hash BLOB NOT NULL,
        group_hash BLOB NOT NULL, loose_hash BLOB NOT NULL,
        label INTEGER, rating INTEGER, raw_label TEXT NOT NULL,
        domain TEXT NOT NULL, bad_domain INTEGER NOT NULL,
        raw_chars INTEGER NOT NULL, chars INTEGER NOT NULL, words INTEGER NOT NULL,
        latin INTEGER NOT NULL, cyrillic INTEGER NOT NULL, other_letters INTEGER NOT NULL,
        script_tag TEXT NOT NULL, eligible INTEGER NOT NULL,
        literal_placeholder INTEGER NOT NULL,
        PRIMARY KEY(language,source_row))''')
    db.execute('''CREATE TABLE IF NOT EXISTS completed_sources (
        language TEXT PRIMARY KEY, sha256 TEXT NOT NULL, row_count INTEGER NOT NULL,
        stats_json TEXT NOT NULL)''')
    db.commit()
    return db


def ingest(db: sqlite3.Connection, language: str, source: dict) -> dict:
    previous = db.execute('SELECT sha256,stats_json FROM completed_sources WHERE language=?', (language,)).fetchone()
    if previous:
        if previous[0] != source['sha256']:
            raise AuditError(f'{language}: source checksum changed; cannot resume this audit.')
        print(f'[REUSE] {language}: completed full-row audit ledger reused.', flush=True)
        return json.loads(previous[1])
    spec = SPECS[language]
    p = Path(source['path'])
    unknown_labels = Counter()
    invalid_label_count = bad_domain_count = missing_id_count = 0
    rows = 0
    begin_stat = p.stat()
    db.execute('BEGIN IMMEDIATE')
    try:
        if db.execute('SELECT COUNT(*) FROM records WHERE language=?', (language,)).fetchone()[0]:
            raise AuditError(f'{language}: incomplete committed ledger detected; use a new audit directory.')
        batch = []
        with p.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter=',', strict=True)
            header = reader.fieldnames or []
            if len(header) != len(set(header)) or set(header) != set(spec['columns']):
                raise AuditError(f'{language}: raw CSV schema differs from the report-backed expected schema.')
            for rows, row in enumerate(reader, 1):
                if None in row or any(row.get(c) is None for c in spec['columns']):
                    raise AuditError(f'{language}: malformed CSV record {rows}; row not skipped.')
                text_raw = row[spec['text']]
                clean = collapse_ws(text_raw)  # existing whitespace operation, not a new deletion policy
                label, rating, safe_raw_label = parse_label(language, row[spec['label']])
                if label is None:
                    invalid_label_count += 1
                    code = digest(row[spec['label']]).hex()[:12]
                    unknown_labels[code] += 1
                domain, bad_domain = parse_domain(language, row[spec['domain']] if spec['domain'] else None)
                bad_domain_count += bad_domain
                native_id = row[spec['id']]
                id_hash = digest(native_id) if native_id.strip() else None
                missing_id_count += id_hash is None
                exact_hash, group_hash, loose_hash = group_keys(text_raw)
                chars, words, lat, cyr, other, tag = text_features(clean)
                eligible = int(label is not None and chars >= MIN_CHARS)
                placeholder = int(clean.casefold() in {'nan', 'null', 'none'})
                batch.append((language, rows, id_hash, exact_hash, group_hash, loose_hash,
                    label, rating, safe_raw_label, domain, int(bad_domain), len(text_raw),
                    chars, words, lat, cyr, other, tag, eligible, placeholder))
                if len(batch) >= 5000:
                    db.executemany('INSERT INTO records VALUES (' + ','.join('?' for _ in range(20)) + ')', batch)
                    batch.clear()
                if rows % 50000 == 0:
                    print(f'[READ] {language}: {rows:,} rows', flush=True)
        if batch:
            db.executemany('INSERT INTO records VALUES (' + ','.join('?' for _ in range(20)) + ')', batch)
        if not rows:
            raise AuditError(f'{language}: source CSV is empty.')
        after_stat = p.stat()
        if begin_stat.st_size != after_stat.st_size or begin_stat.st_mtime_ns != after_stat.st_mtime_ns or sha_file(p) != source['sha256']:
            raise AuditError(f'{language}: source changed during audit; transaction cancelled.')
        stats = {'rows_read': rows, 'columns': header,
                 'invalid_label_count': invalid_label_count,
                 'unrecognized_label_hashes': dict(unknown_labels),
                 'unrecognized_or_missing_domain_count': bad_domain_count,
                 'missing_original_id_count': missing_id_count}
        db.execute('INSERT INTO completed_sources VALUES (?,?,?,?)',
                   (language, source['sha256'], rows, json.dumps(stats)))
        db.commit()
        return stats
    except BaseException:
        db.rollback()
        raise


def one(db, query, args=()):
    return db.execute(query, args).fetchone()[0]


def group_statistics(db, column: str, scope: str | None) -> dict:
    if column not in {'exact_hash', 'group_hash', 'loose_hash'}:
        raise AuditError('Invalid grouping column.')
    where = 'eligible=1' + (' AND language=?' if scope else '')
    args = (scope,) if scope else ()
    query = f'''SELECT COUNT(*), COALESCE(SUM(n),0),
        COALESCE(SUM(CASE WHEN n>1 THEN 1 ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN labels>1 THEN 1 ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN labels>1 THEN n ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN languages>1 THEN 1 ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN languages>1 THEN n ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN domains>1 THEN 1 ELSE 0 END),0),
        COALESCE(MAX(n),0)
        FROM (SELECT COUNT(*) n, COUNT(DISTINCT label) labels,
              COUNT(DISTINCT language) languages, COUNT(DISTINCT domain) domains
              FROM records WHERE {where} GROUP BY {column})'''
    vals = db.execute(query, args).fetchone()
    names = ['groups', 'rows', 'repeated_groups', 'rows_in_repeated_groups',
             'conflicting_label_groups', 'rows_in_conflicting_label_groups',
             'cross_language_groups', 'rows_in_cross_language_groups',
             'cross_domain_groups', 'largest_group_rows']
    return {'scope': scope or 'all', 'normalization': {
        'exact_hash': 'exact', 'group_hash': 'nfc_ws',
        'loose_hash': 'nfkc_casefold_ws_diagnostic'}[column], **dict(zip(names, vals))}


def quantile_sorted(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    x = (len(values)-1)*fraction
    lo = int(math.floor(x)); hi = int(math.ceil(x))
    return float(values[lo] + (values[hi]-values[lo])*(x-lo))


def distribution(values: list[int | float]) -> dict:
    if not values:
        return {'n': 0, 'mean': None, 'min': None, 'p50': None, 'p90': None, 'p95': None, 'p99': None, 'max': None}
    values.sort()
    return {'n': len(values), 'mean': statistics.fmean(values), 'min': float(values[0]),
            'p50': quantile_sorted(values, .50), 'p90': quantile_sorted(values, .90),
            'p95': quantile_sorted(values, .95), 'p99': quantile_sorted(values, .99), 'max': float(values[-1])}


def analyses(db: sqlite3.Connection) -> dict:
    tables = {}
    db.execute('CREATE INDEX IF NOT EXISTS idx_eligible_lang ON records(eligible,language,label,domain)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_group ON records(group_hash)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_exact ON records(exact_hash)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_loose ON records(loose_hash)')
    db.commit()
    flow = []; ids = []; labels = []; domains = []; capacity = []
    for lang in SPECS:
        flow.append({'language': lang,
            'raw_rows': one(db, 'SELECT COUNT(*) FROM records WHERE language=?', (lang,)),
            'invalid_label_rows': one(db, 'SELECT COUNT(*) FROM records WHERE language=? AND label IS NULL', (lang,)),
            'empty_text_rows': one(db, 'SELECT COUNT(*) FROM records WHERE language=? AND chars=0', (lang,)),
            'short_nonempty_text_rows': one(db, 'SELECT COUNT(*) FROM records WHERE language=? AND chars>0 AND chars<?', (lang,MIN_CHARS)),
            'eligible_before_any_dedup_or_cap': one(db,'SELECT COUNT(*) FROM records WHERE language=? AND eligible=1',(lang,)),
            'literal_placeholder_text_rows_not_automatically_removed': one(db,'SELECT COUNT(*) FROM records WHERE language=? AND literal_placeholder=1',(lang,)),
            'candidate_unique_normalized_text_label_pairs': one(db,
                'SELECT COUNT(*) FROM (SELECT group_hash,label FROM records WHERE language=? AND eligible=1 GROUP BY group_hash,label)',(lang,)),
            'candidate_unique_normalized_text_label_domain_tuples': one(db,
                'SELECT COUNT(*) FROM (SELECT group_hash,label,domain FROM records WHERE language=? AND eligible=1 GROUP BY group_hash,label,domain)',(lang,))})
        id_rows = db.execute('''SELECT COUNT(*),COALESCE(SUM(n),0),COALESCE(SUM(CASE WHEN texts>1 OR labels>1 OR domains>1 THEN 1 ELSE 0 END),0)
            FROM (SELECT COUNT(*) n,COUNT(DISTINCT exact_hash) texts,COUNT(DISTINCT raw_label) labels,
            COUNT(DISTINCT domain) domains FROM records WHERE language=? AND original_id_hash IS NOT NULL
            GROUP BY original_id_hash HAVING COUNT(*)>1)''',(lang,)).fetchone()
        ids.append({'language': lang, 'duplicate_native_id_groups': id_rows[0],
                    'rows_in_duplicate_id_groups': id_rows[1], 'duplicate_id_groups_with_different_content': id_rows[2]})
        for raw,label,n in db.execute('SELECT raw_label,label,COUNT(*) FROM records WHERE language=? GROUP BY raw_label,label ORDER BY raw_label',(lang,)):
            labels.append({'language':lang,'raw_label':raw,'mapped_label':LABEL_NAMES.get(label,'UNRECOGNIZED'),'raw_rows':n})
        for dom,label,n,n_eligible in db.execute('SELECT domain,label,COUNT(*),SUM(eligible) FROM records WHERE language=? GROUP BY domain,label ORDER BY domain,label',(lang,)):
            domains.append({'language':lang,'domain':dom,'label':LABEL_NAMES.get(label,'UNRECOGNIZED'),
                            'raw_rows':n,'eligible_rows_before_dedup':n_eligible})
        scopes = ['ALL'] + [x[0] for x in db.execute('SELECT DISTINCT domain FROM records WHERE language=? ORDER BY domain',(lang,))]
        for scope in scopes:
            cond='language=? AND eligible=1'; params=[lang]
            if scope!='ALL': cond+=' AND domain=?';params.append(scope)
            for label in (0,1,2):
                n=one(db,f'SELECT COUNT(*) FROM records WHERE {cond} AND label=?',(*params,label))
                ng=one(db,f'SELECT COUNT(DISTINCT group_hash) FROM records WHERE {cond} AND label=?',(*params,label))
                # Diagnostic lower-risk pool; not an automatic singleton-only selection.
                ns=one(db,f'''SELECT COUNT(*) FROM records WHERE {cond} AND label=? AND
                    group_hash IN (SELECT group_hash FROM records WHERE eligible=1 GROUP BY group_hash HAVING COUNT(*)=1)''',(*params,label))
                capacity.append({'language':lang,'domain_scope':scope,'label':LABEL_NAMES[label],
                    'eligible_rows_before_split':n,'groups_containing_class':ng,'rows_in_globally_singleton_groups':ns,
                    'split_budget_feasibility':'NOT_YET_VERIFIED; no split, no resampling, no budget reduction'})
    tables['corpus_flow']=flow;tables['id_audit']=ids;tables['label_counts']=labels
    tables['domain_label_counts']=domains;tables['budget_capacity']=capacity
    tables['domain_rating_counts']=[
        {'language':lang,'domain':dom,'raw_label':raw,'rating':rating,
         'mapped_label':LABEL_NAMES.get(label,'UNRECOGNIZED'),'raw_rows':n,'eligible_rows':ne,
         'eligible_groups':ng}
        for lang,dom,raw,rating,label,n,ne,ng in db.execute('''
            SELECT language,domain,raw_label,rating,label,COUNT(*),SUM(eligible),
            COUNT(DISTINCT CASE WHEN eligible=1 THEN group_hash END)
            FROM records GROUP BY language,domain,raw_label,rating,label
            ORDER BY language,domain,rating,raw_label''')]

    groups=[group_statistics(db,c,s) for c in ('exact_hash','group_hash','loose_hash') for s in ('tr','kz',None)]
    tables['text_group_audit']=groups
    profile=[]; script_counts=[]
    for lang in SPECS:
        scopes=[('ALL','ALL')]
        scopes += [(d,'ALL') for d, in db.execute('SELECT DISTINCT domain FROM records WHERE language=? AND eligible=1 ORDER BY domain',(lang,))]
        scopes += [('ALL',str(c)) for c in (0,1,2)]
        for dom,lab in scopes:
            cond='language=? AND eligible=1';params=[lang]
            if dom!='ALL':cond+=' AND domain=?';params.append(dom)
            if lab!='ALL':cond+=' AND label=?';params.append(int(lab))
            vals=db.execute(f'SELECT chars,words,latin,cyrillic,other_letters FROM records WHERE {cond}',params).fetchall()
            metrics={'characters_after_whitespace_cleanup':[r[0] for r in vals],
                     'whitespace_separated_words_not_model_tokens':[r[1] for r in vals],
                     'latin_fraction_of_letters':[r[2]/sum(r[2:]) for r in vals if sum(r[2:])],
                     'cyrillic_fraction_of_letters':[r[3]/sum(r[2:]) for r in vals if sum(r[2:])]}
            for metric,v in metrics.items():
                profile.append({'language':lang,'domain_scope':dom,'class_scope':LABEL_NAMES[int(lab)] if lab!='ALL' else 'ALL',
                                'metric':metric,**distribution(v)})
            if lab=='ALL':
                for tag,n in db.execute(f'SELECT script_tag,COUNT(*) FROM records WHERE {cond} GROUP BY script_tag ORDER BY script_tag',params):
                    script_counts.append({'language':lang,'domain_scope':dom,'script_tag':tag,'rows':n})
    tables['length_script_profiles']=profile;tables['script_counts']=script_counts
    return tables


def bounded_json(path: Path, limit: int=1024*1024):
    with path.open('rb') as f:
        content=f.read(limit+1)
    if len(content)>limit: raise AuditError('Model metadata exceeds audit size limit.')
    return json.loads(content.decode('utf-8-sig'))


def inspect_model_metadata(preflight: dict) -> list[dict]:
    out=[]
    allowed={'sentence_bert_config.json': ('max_seq_length','do_lower_case'),
             'config.json': ('model_type','architectures','hidden_size','max_position_embeddings','torch_dtype','dtype','vocab_size'),
             '1_Pooling/config.json': ('word_embedding_dimension','pooling_mode_cls_token','pooling_mode_mean_tokens','pooling_mode_max_tokens','pooling_mode_mean_sqrt_len_tokens'),
             '2_Dense/config.json': ('in_features','out_features','bias','activation_function')}
    for inv in preflight.get('inventories',[]):
        if inv.get('kind')!='hf_model':continue
        root=Path(inv['path']);refs={};snaps={};errors=[]
        for item in inv.get('files',[]):
            rel=item.get('relative_path','')
            if rel.startswith('refs/'):
                try:
                    p=root/rel
                    if p.stat().st_size>200:raise AuditError('Unexpected ref file size.')
                    v=p.read_text(encoding='utf-8').strip()
                    if not re.fullmatch('[0-9a-f]{40}',v):raise AuditError('Unexpected ref identifier.')
                    refs[rel]=v
                except (OSError,UnicodeError,AuditError):errors.append('Could not validate ref '+rel)
            match=re.fullmatch(r'snapshots/([0-9a-f]{40})/(.+)',rel)
            if not match:continue  # exclude .no_exist entries; they are NOT cached files
            revision,name=match.groups();snap=snaps.setdefault(revision,{'revision':revision,'weights':[],'metadata':{}})
            if name.endswith(('.bin','.safetensors')):
                p=root/rel
                snap['weights'].append({'relative_file':name,'bytes':item['bytes'],'readable_file_exists':p.is_file(),
                                        'content_or_tensor_values_verified':False})
            if name in allowed or name=='modules.json':
                try:
                    data=bounded_json(root/rel)
                    if name=='modules.json':
                        if not isinstance(data,list):raise AuditError('Unexpected modules structure.')
                        data=[{k:v for k,v in x.items() if k in {'idx','name','path','type'}} for x in data]
                    else:
                        data={k:data[k] for k in allowed[name] if k in data}
                    snap['metadata'][name]=data
                except (OSError,UnicodeError,ValueError,AuditError):errors.append('Metadata not readable: '+rel)
        out.append({'repo_id':inv.get('repo_id'),'refs':refs,'snapshots':list(snaps.values()),'errors':errors,
                    'note':'Metadata only; no model loaded and no revision merged or silently selected.'})
    return out


def inspect_legacy_overlap(project: Path) -> dict:
    """Full scan of old train/test identifiers as text hashes, without using predictions."""
    paths={'train': project/'data/processed/train.csv', 'test':project/'data/processed/test.csv'}
    if not all(p.is_file() for p in paths.values()):
        return {'status':'not_checked','reason':'Old train.csv/test.csv not both present at expected project path.'}
    sets={};counts={};sha={}
    for split,p in paths.items():
        before=sha_file(p);sha[split]=before
        sets[split]={norm:{} for norm in ('exact','nfc_ws')}; counts[split]=0
        with p.open('r',encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f,strict=True)
            if not {'text','language'}.issubset(reader.fieldnames or []):
                return {'status':'not_checked','reason':'Old CSV schema does not contain text/language.'}
            for row in reader:
                if None in row or row.get('text') is None or row.get('language') not in {'tr','kz'}:
                    raise AuditError('Malformed legacy CSV; no raw row content printed.')
                lang=row['language']; a,b,_=group_keys(row['text'])
                for norm,h in [('exact',a),('nfc_ws',b)]:
                    sets[split][norm].setdefault(lang,Counter())[h]+=1
                counts[split]+=1
        if sha_file(p)!=before:raise AuditError('Legacy CSV changed during audit.')
    overlaps=[]
    for norm in ('exact','nfc_ws'):
        for src in ('tr','kz'):
            for tgt in ('tr','kz'):
                train=sets['train'][norm].get(src,Counter());test=sets['test'][norm].get(tgt,Counter())
                common=train.keys() & test.keys()
                overlaps.append({'normalization':norm,'train_language':src,'test_language':tgt,
                                  'common_groups':len(common),'affected_test_rows':sum(test[g] for g in common)})
    return {'status':'checked','rows':counts,'checksums':sha,'overlaps':overlaps,
            'note':'Old outcome direction is not inferred from overlap; old files were not changed.'}


def build_checks(sources,ingested,tables,models) -> list[dict]:
    checks=[]
    def add(name,status,details):checks.append({'check':name,'status':status,'details':details})
    for lang in SPECS:
        s=ingested[lang];expected=sources[lang]['metadata_count']
        add(lang+'_csv_rows_vs_cached_metadata','PASS' if expected==s['rows_read'] else 'REVIEW',
            {'csv_rows':s['rows_read'],'metadata_rows':expected})
        add(lang+'_explicit_label_mapping','PASS' if not s['invalid_label_count'] else 'BLOCKED',s['invalid_label_count'])
        if lang=='kz':add('kz_domain_values','PASS' if not s['unrecognized_or_missing_domain_count'] else 'BLOCKED',s['unrecognized_or_missing_domain_count'])
        add(lang+'_native_id_completeness','PASS' if not s['missing_original_id_count'] else 'REVIEW',s['missing_original_id_count'])
    for row in tables['id_audit']:
        add(row['language']+'_original_id_reuse','PASS' if not row['duplicate_native_id_groups'] else 'REVIEW',row)
    for row in tables['corpus_flow']:
        add(row['language']+'_nonempty_eligible_data','PASS' if row['eligible_before_any_dedup_or_cap'] else 'BLOCKED',row['eligible_before_any_dedup_or_cap'])
        if row['literal_placeholder_text_rows_not_automatically_removed']:
            add(row['language']+'_literal_placeholder_strings','REVIEW',row['literal_placeholder_text_rows_not_automatically_removed'])
    if any(m['errors'] for m in models):add('cached_model_metadata','REVIEW','At least one metadata read failed; see model manifest.')
    add('train_validation_test_separation','NOT_RUN','This stage does not allocate splits.')
    add('per_seed_budget_feasibility','NOT_RUN','Only unsplit capacities computed; exact grouped quotas require split stage.')
    add('human_label_validation','NOT_RUN','Rating mapping audit is NOT independent annotation validation.')
    add('cuda_runtime_and_model_loading','NOT_RUN','No torch import, CUDA allocation or model loading in this stage.')
    return checks


def run(project: Path, preflight_path: Path, resume: Path | None=None) -> Path:
    start=time.monotonic()
    preflight=bounded_json(preflight_path,20*1024*1024)
    if preflight.get('schema_version')!='1.0.0':raise AuditError('Unsupported preflight schema; explicit review required.')
    expected_project=Path(preflight.get('project','')).expanduser().resolve()
    if project!=expected_project:
        raise AuditError('Project path differs from preflight. Use the original project or regenerate preflight.')
    # Get checksums BEFORE creating an output directory; do not silently use another snapshot.
    sources=source_selection(preflight)
    script_sha=sha_file(Path(__file__))
    identity={'auditor_version':VERSION,'auditor_sha256':script_sha,
              'preflight_sha256':sha_file(preflight_path),
              'source_checksums':{k:v['sha256'] for k,v in sources.items()}}
    if resume:
        out=resume.resolve()
        if not (out/'identity.json').is_file():raise AuditError('Resume directory is not a recognized data audit.')
        if bounded_json(out/'identity.json')!=identity:raise AuditError('Resume input/code hashes differ; do not mix audits.')
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        out=project/'_revision_audit'/('data_gate_'+stamp)
        out.mkdir(parents=True,exist_ok=False)
        json_write(out/'identity.json',identity)
    (out/'tables').mkdir(exist_ok=True)
    json_write(out/'status.json',{'state':'RUNNING','training_allowed':False})
    print('[OUTPUT]',out,flush=True)
    db=connect_ledger(out/'local_hash_ledger.sqlite3')
    try:
        ingested={}
        for lang,source in sources.items():
            print(f'[SOURCE] {lang} revision={source["revision"]}',flush=True)
            ingested[lang]=ingest(db,lang,source)
        print('[ANALYSE] Groups, label/domain counts, lengths and scripts...',flush=True)
        tables=analyses(db)
        models=inspect_model_metadata(preflight)
        legacy=inspect_legacy_overlap(project)
        checks=build_checks(sources,ingested,tables,models)
        blocked=any(c['status']=='BLOCKED' for c in checks)
        state='DATA_REVIEW_REQUIRED' if blocked else 'RAW_AUDIT_COMPLETE'
        json_write(out/'source_manifest.local.json',sources)
        source_public={lang:{k:v for k,v in s.items() if k!='path'} for lang,s in sources.items()}
        report={'schema_version':VERSION,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'state':state,'training_allowed':False,
                'sources':source_public,'ingestion':ingested,'checks':checks,'tables':tables,
                'legacy_overlap':legacy,'model_cache_manifest':models,'normalizations':NORMALIZATIONS,
                'audit_view_min_chars':MIN_CHARS,
                'source_rows_removed_or_relabeled':False,
                'split_or_budget_manifests_created':False,
                'elapsed_seconds':time.monotonic()-start,
                'limitations':[
                    'Full row counts/schema/rating conversion checks are not human label validation.',
                    'Text grouping views do not apply deduplication or casefolding to a training corpus.',
                    'No selected model, checkpoint, tensor content or cache-row alignment was verified.',
                    'Unsplit class capacities do not guarantee grouped train/validation budgets.',
                    'Script measurement is not language identification; words are whitespace tokens only.',
                    'No probability/performance scores were used to select preprocessing or models.'
                ]}
        for name,rows in tables.items():csv_write(out/'tables'/f'{name}.csv',rows)
        json_write(out/'data_audit_report.json',report)
        json_write(out/'checks.json',checks)
        json_write(out/'model_cache_manifest.json',models)
        lines=['SAUCIS REVIZYONU — TAM SATIR VERI DENETIMI', '='*62,
               f'Durum: {state}',
               'Indirme, paket kurulumu, model egitimi, GPU kullanimi veya eski dosya degisikligi yok.',
               'Asagidaki filtre/grup sayilari denetim gorunumudur; yeni split uretilmedi.', '']
        for lang in SPECS:
            f=next(x for x in tables['corpus_flow'] if x['language']==lang)
            lines += [f'{lang.upper()}: CSV={ingested[lang]["rows_read"]:,}; metadata={sources[lang]["metadata_count"]}; '
                      f'etiketi taninan ve >=3 karakter={f["eligible_before_any_dedup_or_cap"]:,}',
                      f'  revision={sources[lang]["revision"]}',f'  SHA256={sources[lang]["sha256"]}']
        lines += ['', 'GRUP DENETIMI (NFC + bosluk; buyuk/kucuk harf degismez):']
        for g in tables['text_group_audit']:
            if g['normalization']=='nfc_ws':
                lines += [f'  {g["scope"]}: grup={g["groups"]}; tekrar grubu={g["repeated_groups"]}; '
                          f'celiskili etiket grubu={g["conflicting_label_groups"]}; dillerarasi grup={g["cross_language_groups"]}']
        lines += ['', 'KONTROLLER:']
        for c in checks:lines.append(f'  {c["status"]}: {c["check"]}')
        lines += ['', 'NOT: Tamamlandi = veri okuma/denetim tamamlandi; egitim ve split onayi DEGIL.',
                  'model_cache_manifest: snapshot/config metadata; agirliklar yuklenmedi.',
                  'Paylasilacak: shareable_data_audit.zip (ham yorum ve ozgun kayit IDsi icermez).',
                  'local_hash_ledger.sqlite3 ve source_manifest.local.json yerel kalabilir.']
        (out/'data_audit_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        names=['data_audit_report.json','data_audit_summary.txt','checks.json','model_cache_manifest.json']
        with zipfile.ZipFile(out/'shareable_data_audit.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in names:z.write(out/name,name)
            for p in sorted((out/'tables').glob('*.csv')):z.write(p,'tables/'+p.name)
        json_write(out/'status.json',{'state':state,'training_allowed':False,'report_complete':True,
                                     'report_sha256':sha_file(out/'data_audit_report.json')})
        print('\n'.join(lines),flush=True)
        print('\n[SHARE]',out/'shareable_data_audit.zip',flush=True)
        return out
    except BaseException as exc:
        # Do not echo exception strings that might include a raw malformed CSV value.
        json_write(out/'status.json',{'state':'FAILED','training_allowed':False,'exception_type':type(exc).__name__,
            'safe_message':str(exc) if isinstance(exc,AuditError) else 'Audit interrupted/failed; no raw content included. Resume or inspect locally.'})
        raise
    finally:
        db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project',default='.')
    p.add_argument('--preflight',required=True,help='Exact preflight_report.json from step 00.')
    p.add_argument('--resume',default=None,help='Existing data_gate output; input/code hashes must match.')
    args=p.parse_args()
    csv.field_size_limit(min(sys.maxsize,100_000_000))
    try:
        out=run(Path(args.project).expanduser().resolve(),Path(args.preflight).expanduser().resolve(),
                Path(args.resume).expanduser() if args.resume else None)
        state=bounded_json(out/'status.json')['state']
        return 2 if state=='DATA_REVIEW_REQUIRED' else 0
    except KeyboardInterrupt:
        print('\n[STOP] Audit interrupted. Completed source transactions can be reused with --resume.',file=sys.stderr)
        return 130
    except AuditError as exc:
        print('[STOP]',exc,file=sys.stderr)
        return 2
    except Exception as exc:
        print(f'[STOP] {type(exc).__name__}; no raw text printed. See status.json in the output directory.',file=sys.stderr)
        return 3


if __name__=='__main__':
    raise SystemExit(main())
