import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('snapshot_materializer',Path(__file__).resolve().parents[1]/'scripts/materialize_snapshot.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_render_requires_new_destination_and_preserves_source(tmp_path):
    token='@'+'DRL_ROOT_LINUX'+'@'
    src=tmp_path/'source';src.mkdir();(src/'config.json').write_text('{"root":"'+token+'"}')
    out=tmp_path/'rendered';assert module.render(src,out,{'DRL_ROOT_LINUX':'/srv/rl'})==1
    assert token in (src/'config.json').read_text()
    assert '/srv/rl' in (out/'config.json').read_text()
    with pytest.raises(ValueError):module.render(src,out,{})
    with pytest.raises(ValueError):module.render(src,src/'nested',{})

def test_unresolved_tokens_fail_before_writing(tmp_path):
    src=tmp_path/'source';src.mkdir();(src/'a.py').write_text("root='"+'@'+'LINUX_HOME'+'@'+"'")
    out=tmp_path/'rendered'
    with pytest.raises(ValueError):module.render(src,out,{})
    assert not out.exists()
