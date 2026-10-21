from pathlib import Path
import hashlib,importlib.util,json,shutil,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
def mod(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
A=mod('audit_release');P=mod('prepare_inputs');F=mod('finalize_metadata');S=mod('run_stage')

def test_frozen_script_hashes_and_syntax():
 d=json.loads((ROOT/'metadata/frozen_scripts.json').read_text())
 assert len(d['scripts'])==12
 for row in d['scripts']:
  p=ROOT/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];compile(p.read_bytes(),row['path'],'exec')

def test_confusion_macro_with_abstention():
 m=A.metrics([[1,0,0,0],[0,0,0,1],[0,0,1,0]])
 assert m['n']==3 and m['macro_f1']==pytest.approx(2/3) and m['accuracy']==pytest.approx(2/3)

def test_holm_reference():assert A.holm([.01,.04,.03,.002])==pytest.approx([.03,.06,.06,.008])
@pytest.mark.parametrize('bad',[-1,float('nan'),1.2])
def test_holm_invalid(bad):
 with pytest.raises(ValueError):A.holm([bad])

def test_pinned_file_failure(tmp_path):
 p=tmp_path/'a';p.write_bytes(b'abc')
 with pytest.raises(ValueError):P.validate_file(p,{'bytes':3,'sha256':'0'*64})
 P.validate_file(p,{'bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()})

def test_workspace_preserves_original_bytes(tmp_path):
 P.install_scripts(tmp_path);P.install_scripts(tmp_path)
 for d in json.loads((ROOT/'metadata/frozen_scripts.json').read_text())['scripts']:assert (tmp_path/d['path']).read_bytes()==(ROOT/d['path']).read_bytes()

def test_cannot_install_over_research_workspace(tmp_path):
 (tmp_path/'_revision_test').mkdir()
 with pytest.raises(ValueError):P.install_scripts(tmp_path)

def test_reject_edited_workspace_script(tmp_path):
 P.install_scripts(tmp_path);(tmp_path/'09_revision_analyse_predictions.py').write_text('edited')
 with pytest.raises(ValueError):P.install_scripts(tmp_path)

def test_inventory_exact_selected_sources(tmp_path):
 assets={'datasets':{},'models':{}}
 for lang in ('tr','kz'):
  rev=('a' if lang=='tr' else 'b')*40;repo=f'test/{lang}';f='data.csv';b=b'id,text\n1,synthetic\n'
  p=tmp_path/'hf_home/hub'/('datasets--'+repo.replace('/','--'))/'snapshots'/rev/f;p.parent.mkdir(parents=True);p.write_bytes(b)
  assets['datasets'][lang]={'repo_id':repo,'revision':rev,'filename':f,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'raw_rows':1}
 inv=P.selected_inventory(tmp_path,assets)
 assert len(inv['inventories'])==2 and all(x['split_counts']['train']['num_examples']==1 for x in inv['dataset_metadata'])
 assert 'historical' in inv['dataset_metadata'][0]['reference_count_origin']

def minimal_release(tmp_path):
 for rel in ['metadata/zenodo.template.json','license_templates/MIT.txt','CITATION.cff','LICENSE_CHOICE_REQUIRED.md']:
  d=tmp_path/rel;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/rel).read_bytes())
 return tmp_path

def test_explicit_metadata_choice(tmp_path):
 root=minimal_release(tmp_path);code=root/'frozen.py';code.write_text('untouched');before=code.read_bytes()
 F.finalize(root,'https://github.com/synthetic-owner/synthetic-repo','MIT')
 assert code.read_bytes()==before
 z=json.loads((root/'.zenodo.json').read_text());assert z['license']=='mit'
 assert 'doi' not in z and 'doi:' not in (root/'CITATION.cff').read_text()
 assert (root/'LICENSE').read_bytes()==(ROOT/'license_templates/MIT.txt').read_bytes()
 assert 'license: "MIT"' in (root/'CITATION.cff').read_text()
 for line in (root/'SHA256SUMS.txt').read_text().splitlines():
  h,p=line.split('  ',1);assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h

@pytest.mark.parametrize('url',['https://token@github.com/u/r','http://github.com/u/r','https://github.com/OWNER/REPOSITORY','https://github.com/u/r?token=secret'])
def test_metadata_reject_invalid_url(tmp_path,url):
 root=minimal_release(tmp_path)
 with pytest.raises(ValueError):F.finalize(root,url,'MIT')
 assert not (root/'LICENSE').exists()

def test_existing_license_not_replaced(tmp_path):
 root=minimal_release(tmp_path);(root/'LICENSE').write_text('another choice')
 with pytest.raises(ValueError):F.finalize(root,'https://github.com/synthetic-owner/repo','MIT')
 assert (root/'LICENSE').read_text()=='another choice'

def state_fixture(tmp_path):
 state={}
 for k in S.STAGES:
  p=tmp_path/k;p.mkdir();state[k]={'output':str(p)}
 return state
@pytest.mark.parametrize('stage',S.STAGES)
def test_stage_command_routes_only_one_script(tmp_path,stage):
 state=state_fixture(tmp_path);cmd=S.command(stage,tmp_path,state,'1',4,True)
 assert cmd[0]==sys.executable
 assert all('--stage' not in x for x in cmd[:2])
 assert not any(x in cmd for x in ('shell=True','rm','pip'))
 if stage in S.GPU_STAGES:assert cmd[cmd.index('--gpu')+1]=='1'
 if stage=='qwen-pin':assert '--allow-download' in cmd and 'prepare_inputs.py' in cmd[1]

def test_no_implicit_gpu(tmp_path):
 state=state_fixture(tmp_path)
 with pytest.raises(ValueError):S.command('xlmr',tmp_path,state,None,4)

def test_stage_path_escape_rejected(tmp_path):
 state=state_fixture(tmp_path);state['split']['output']=str(tmp_path.parent)
 with pytest.raises(ValueError):S.command('features',tmp_path,state,'1',4)

def test_no_active_license_chosen_in_delivered_candidate():
 # This assertion applies to the unfinalized candidate; skip after owner choice.
 if (ROOT/'metadata/release_owner_choices.json').exists():pytest.skip('Owner has explicitly selected a license.')
 assert not (ROOT/'LICENSE').exists() and not (ROOT/'.zenodo.json').exists()
