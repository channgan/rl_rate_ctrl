from pathlib import Path
import hashlib,json,shutil

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def required_disk(remaining):
    assert type(remaining) is int and 0<=remaining<=490
    return remaining*256*1024**2+30*1024**3
def require_disk(root,remaining):
    assert shutil.disk_usage(root).free>=required_disk(remaining),'insufficient disk: preserve evidence and stop before capture'
def require_grant(root):
    root=Path(root);g=read(root/'launch_grant.json');p=read(root/'protocol.json');l=read(root/'ledger.json')
    assert g['authorized'] is True and g['protocol_sha256']==sha(root/'protocol.json') and g['native_limit']==23030000 and g['blocks']==1
    assert p['authorized'] and l['authorized'] and p['native_limit']==l['native_limit']==23030000
    assert p['effective_step_unit']=='10ms_env_step' and p['new_updates']==700 and not p['automatic_development']
    assert not l['stopped'] and not l['reservations'] and not l['attempts'] and l['formal_optimizer_updates']==40
    assert sha(p['resume_checkpoint'])==p['resume_checkpoint_sha256']
    for f,h in p['prepared_runtime_sha256'].items():assert sha(f)==h
    source=Path(p['source_v49']);dev=read(source/'development/ledger.json')
    assert len(dev['attempts'])==54 and all(x['status']=='audited' for x in dev['attempts']) and not dev['reservations'] and not dev['stopped']
    assert g['complete_v49_review_acknowledged'] is True and g['zero_native_admission_passed'] is True

