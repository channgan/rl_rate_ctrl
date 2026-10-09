"""Render local path configuration into a NEW offline checkout; never launch training."""
from pathlib import Path
import argparse,json,re

def render(source, destination, mapping):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Use a new destination outside the source checkout')
    for key,value in mapping.items():
        if not re.fullmatch(r'[A-Z_]+',key) or not isinstance(value,str) or '\\' in value:
            raise ValueError('Use uppercase token names and forward-slash path values')
    files=[]
    for path in source.rglob('*'):
        if not path.is_file() or any(part in {'.git','__pycache__','.pytest_cache'} for part in path.relative_to(source).parts):continue
        text=path.read_text(encoding='utf-8')
        for key,value in mapping.items():text=text.replace('@'+key+'@',value)
        unresolved=re.findall(r'@[A-Z_]+@',text)
        if unresolved:raise ValueError(f'Unresolved path tokens in {path.name}: {sorted(set(unresolved))}')
        files.append((path.relative_to(source),text))
    destination.mkdir()
    for relative,text in files:
        out=destination/relative;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding='utf-8')
    return len(files)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--paths',type=Path,required=True)
    args=parser.parse_args();print(json.dumps({'rendered_files':render(args.source,args.out,json.loads(args.paths.read_text())),'training_started':False}))
