#!/usr/bin/env python3
"""Update software citation metadata and refresh the file manifest.

Reads and writes UTF-8 with LF line endings. No network operations are performed.
The existing MIT license and experimental files are preserved.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = 'https://github.com/brkyvredu/turkish-kazakh-sentiment'
EXCLUDED_DIRECTORIES = {'.git', '__pycache__', '.pytest_cache', '_reproduction'}


def write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(text, encoding='utf-8', newline='\n')
    os.replace(temporary, path)


def refresh_hashes(root: Path) -> None:
    root = Path(root)
    paths = sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.name != 'SHA256SUMS.txt'
        and not any(part in EXCLUDED_DIRECTORIES for part in p.relative_to(root).parts)
        and not p.name.endswith('.partial')
    )
    text = ''.join(
        hashlib.sha256(p.read_bytes()).hexdigest() + '  '
        + p.relative_to(root).as_posix() + '\n' for p in paths
    )
    write(root / 'SHA256SUMS.txt', text)


def repository_url(value: str) -> str:
    value = value.rstrip('/')
    if value.endswith('.git'):
        value = value[:-4]
    if not re.fullmatch(r'https://github\.com/[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9_.-]+', value):
        raise ValueError('Expected an HTTPS GitHub repository URL without credentials or query parameters.')
    if any(x in value for x in ('OWNER', 'YOUR_', 'REPOSITORY')):
        raise ValueError('Expected an actual repository URL.')
    return value


def set_cff_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(r'^' + re.escape(key) + r':[^\n]*$', re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise ValueError(f'Duplicate CFF field: {key}')
    line = key + ': ' + json.dumps(value, ensure_ascii=False)
    if matches:
        return pattern.sub(lambda _: line, text)
    return text.rstrip() + '\n' + line + '\n'


def finalize(root: Path, repo: str, license_name: str = 'MIT',
             doi: str | None = None, date: str | None = None) -> dict:
    root = Path(root)
    repo = repository_url(repo)
    if license_name != 'MIT':
        raise ValueError('This release uses MIT; the metadata utility does not change its license.')
    if doi and not re.fullmatch(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', doi):
        raise ValueError('Invalid DOI syntax.')
    if date:
        datetime.date.fromisoformat(date)
    license_text = (root / 'LICENSE').read_text(encoding='utf-8')
    if not license_text.startswith('MIT License\n') or 'Permission is hereby granted' not in license_text:
        raise ValueError('An existing MIT LICENSE is required; no license file was changed.')
    meta = json.loads((root / '.zenodo.json').read_text(encoding='utf-8'))
    if str(meta.get('license', '')).lower() != 'mit':
        raise ValueError('Zenodo metadata and LICENSE disagree.')
    citation = (root / 'CITATION.cff').read_text(encoding='utf-8')
    citation = set_cff_scalar(citation, 'license', 'MIT')
    citation = set_cff_scalar(citation, 'repository-code', repo)
    if doi:
        citation = set_cff_scalar(citation, 'doi', doi)
    if date:
        citation = set_cff_scalar(citation, 'date-released', date)
        meta['publication_date'] = date
    identifiers = [
        item for item in meta.get('related_identifiers', [])
        if not (item.get('relation') == 'isSupplementTo'
                and str(item.get('identifier', '')).startswith('https://github.com/'))
    ]
    identifiers.append({'identifier': repo, 'relation': 'isSupplementTo', 'scheme': 'url'})
    meta['related_identifiers'] = identifiers
    write(root / '.zenodo.json', json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    write(root / 'CITATION.cff', citation)
    refresh_hashes(root)
    return meta


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', default=DEFAULT_REPOSITORY)
    parser.add_argument('--license', default='MIT', choices=['MIT'])
    parser.add_argument('--doi')
    parser.add_argument('--release-date')
    args = parser.parse_args(argv)
    metadata = finalize(ROOT, args.repository, args.license, args.doi, args.release_date)
    print('Metadata and checksums updated for version', metadata['version'])


if __name__ == '__main__':
    main()
