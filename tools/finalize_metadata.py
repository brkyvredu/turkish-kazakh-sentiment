#!/usr/bin/env python3
"""LOCAL-ONLY owner metadata/license selection; never calls GitHub or Zenodo.
--license MIT is an explicit choice by the person running this command. The
packaged option is not active until chosen. Frozen code/results are never changed.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def write(p,text):
    tmp=p.with_name(p.name+'.partial');tmp.write_text(text,encoding='utf-8');os.replace(tmp,p)
def refresh_hashes(root):
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt' and not any(x in p.relative_to(root).parts for x in ('.git','__pycache__','.pytest_cache','_reproduction')))
    write(root/'SHA256SUMS.txt',''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix()+'\n' for p in files))
def cff(meta,repo,doi=None,date=None):
    q=lambda x:json.dumps(x,ensure_ascii=False)
    lines=['cff-version: 1.2.0','title: '+q(meta['title']),'message: '+q('Please cite this software version and its version-specific Zenodo record when available.'),'type: software','version: '+q(meta['version']),'license: '+q('MIT'),'repository-code: '+q(repo),'authors:','  - family-names: Yavru','    given-names: Ismail Burak','    affiliation: Bursa Uludag University','    orcid: "https://orcid.org/0000-0002-7364-6166"','keywords:']
    lines+=['  - '+q(k) for k in meta['keywords']]
    if doi:lines+=['doi: '+q(doi)]
    if date:lines+=['date-released: '+q(date)]
    return '\n'.join(lines)+'\n'
def finalize(root,repo,license_name,doi=None,date=None):
    root=Path(root);repo=repo.rstrip('/')
    if repo.endswith('.git'):repo=repo[:-4]
    if not re.fullmatch(r'https://github\.com/[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9_.-]+',repo):raise ValueError('Provide the real HTTPS GitHub repository URL, without credentials/query parameters.')
    if any(x in repo for x in ('OWNER','YOUR_','REPOSITORY')):raise ValueError('Replace URL placeholders with the actual repository.')
    if license_name!='MIT':raise ValueError('This helper offers MIT only; other owner choices must be reflected manually and consistently.')
    if doi and not re.fullmatch(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+',doi):raise ValueError('Invalid DOI syntax. Omit until Zenodo actually assigns one.')
    if date:datetime.date.fromisoformat(date)
    template=root/'license_templates/MIT.txt';text=template.read_text(encoding='utf-8')
    active=root/'LICENSE'
    if active.exists() and active.read_text(encoding='utf-8')!=text:raise ValueError('Existing different LICENSE; no overwrite.')
    meta=json.loads((root/'metadata/zenodo.template.json').read_text());meta['license']='mit'
    meta['related_identifiers']=[{'identifier':repo,'relation':'isSupplementTo','scheme':'url'}]
    # DOI is a citation field, not a fabricated Zenodo deposit identifier.
    if date:meta['publication_date']=date
    write(active,text);write(root/'.zenodo.json',json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    write(root/'CITATION.cff',cff(meta,repo,doi,date))
    write(root/'metadata/release_owner_choices.json',json.dumps({'repository':repo,'license':'MIT','version':meta['version'],'software_doi':doi,'release_date':date,'external_action_performed':False},indent=2)+'\n')
    # A renamed historical warning should not remain as an active contradiction.
    warning=root/'LICENSE_CHOICE_REQUIRED.md'
    if warning.exists():write(warning,'# License choice recorded\n\nThe owner selected MIT using the local metadata helper. The active `LICENSE` and\n`.zenodo.json` are authoritative. Upstream inputs are not redistributed or relicensed.\n')
    refresh_hashes(root)
    return meta

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',required=True);p.add_argument('--license',required=True,choices=['MIT']);p.add_argument('--doi');p.add_argument('--release-date')
    a=p.parse_args(argv);m=finalize(ROOT,a.repository,a.license,a.doi,a.release_date)
    print('Local metadata prepared for',m['version']);print('No repository, release, DOI or upload was created.');print('Next: python tools/audit_release.py')
if __name__=='__main__':main()
