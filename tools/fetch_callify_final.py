"""Download the immutable release in parallel resumable ranges and verify it."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-archive', required=True)
    parser.add_argument('--bytes', type=int, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--download-dir', type=Path, required=True)
    parser.add_argument('--staging', type=Path, required=True)
    options = parser.parse_args()
    options.download_dir.mkdir(parents=True, exist_ok=True)
    parts = options.download_dir / 'parts'
    parts.mkdir(exist_ok=True)
    block = 16 * 1024 * 1024
    total = (options.bytes + block - 1) // block

    def fetch(index):
        start = index * block
        length = min(block, options.bytes - start)
        path = parts / f'{index:04d}.bin'
        for attempt in range(3):
            present = path.stat().st_size if path.exists() else 0
            if present == length:
                return index
            if present > length:
                raise ValueError('Download range is larger than the pinned archive range')
            program = (
                'import sys; f=open(' + repr(options.remote_archive) + ',"rb"); '
                'f.seek(' + str(start + present) + '); n=' + str(length - present) + '\n'
                'while n:\n b=f.read(min(n,1048576))\n if not b: raise EOFError()\n '
                'sys.stdout.buffer.write(b); n-=len(b)\n'
            )
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '18127',
                       'root@connect.bjb1.seetacloud.com',
                       '/root/miniconda3/envs/fontify/bin/python -c ' + shlex.quote(program)]
            with path.open('ab') as stream:
                outcome = subprocess.run(command, stdout=stream)
            if outcome.returncode == 0 and path.stat().st_size == length:
                print(json.dumps(dict(completed_range=index + 1, ranges=total)), flush=True)
                return index
        raise RuntimeError(f'Download range {index} failed three attempts')

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, range(total)))
    archive = options.download_dir / 'callify_final.tar.gz'
    with archive.open('wb') as stream:
        for index in range(total):
            with (parts / f'{index:04d}.bin').open('rb') as source:
                shutil.copyfileobj(source, stream)
    if archive.stat().st_size != options.bytes or digest(archive) != options.sha256:
        raise ValueError('Release archive size/hash mismatch')
    options.staging.mkdir(parents=True, exist_ok=False)
    destination = options.staging.resolve()
    with tarfile.open(archive) as package:
        regular_names = set()
        for member in package.getmembers():
            if not (destination / member.name).resolve().is_relative_to(destination):
                raise ValueError('Archive path escapes the staging directory')
            if member.issym():
                raise ValueError('Unexpected symbolic link in the release archive')
            if member.islnk():
                if member.linkname not in regular_names or not (
                    destination / member.linkname
                ).resolve().is_relative_to(destination):
                    raise ValueError('Archive hard link has no prior in-package file')
            elif member.isfile():
                regular_names.add(member.name)
        package.extractall(destination)
    manifest = json.loads((destination / '文件校验_SHA256.json').read_text())
    for name, expected in manifest.items():
        path = destination / name
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('Release payload mismatch: ' + name)
    from PIL import Image
    pngs = list(destination.rglob('*.png'))
    for path in pngs:
        with Image.open(path) as picture:
            picture.load()
    links = 0
    for path in destination.rglob('*.html'):
        for url in re.findall(r'(?:href|src)="([^"]+)"', path.read_text()):
            if not (path.parent / url).exists():
                raise ValueError('Broken HTML reference: ' + str(path) + ' -> ' + url)
            links += 1
    report = dict(status='verified', archive_bytes=options.bytes, archive_sha256=options.sha256,
                  payload_files=len(manifest), decoded_pngs=len(pngs), valid_html_links=links,
                  staging=str(destination))
    (options.download_dir / 'local_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
