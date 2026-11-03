#!/usr/bin/env python3
"""Write or verify SHA-256 checksums of the final package, including reference data/logs; excluding local run results."""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IGNORED = {'results', '.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.DS_Store'}
MANIFEST = 'MANIFEST.sha256'


def files(root):
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and p.relative_to(root).as_posix() != MANIFEST
                  and not any(part in IGNORED for part in p.relative_to(root).parts))


def digest(path):
    with path.open('rb') as fp:
        return hashlib.file_digest(fp,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='replace the manifest after finalizing all files')
    parser.add_argument('--root',type=Path,default=ROOT,help='package root (default: this script directory)')
    args=parser.parse_args()
    root=args.root.resolve()
    actual={p.relative_to(root).as_posix():digest(p) for p in files(root)}
    manifest=root/MANIFEST
    if args.write:
        if any('\n' in name or '\r' in name for name in actual):
            parser.error('Newlines in filenames cannot be represented in this manifest.')
        manifest.write_text(''.join(f'{hash_value}  {name}\n' for name,hash_value in actual.items()))
        print(f'Wrote checksums for {len(actual)} files.')
        return 0
    if not manifest.is_file():
        print('No MANIFEST.sha256 found. Finalize the package, then run manifest.py --write.')
        return 1
    expected={}
    try:
        for line in manifest.read_text().splitlines():
            hash_value,name=line.split('  ',1)
            if len(hash_value)!=64 or any(c not in '0123456789abcdef' for c in hash_value) or name in expected:
                raise ValueError('invalid or duplicate entry')
            expected[name]=hash_value
    except ValueError as exc:
        print('Invalid manifest: '+str(exc))
        return 1
    problems=[]
    for name,hash_value in expected.items():
        if name not in actual: problems.append('MISSING '+name)
        elif actual[name]!=hash_value: problems.append('CHANGED '+name)
    problems += ['UNLISTED '+name for name in actual if name not in expected]
    if problems:
        print('\n'.join(problems))
        return 1
    print(f'Verified all {len(actual)} files; no missing, changed or unlisted files.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
