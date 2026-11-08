"""Finish the release source snapshot and create a pinned delivery archive."""

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    release = args.release.resolve()
    root = Path(__file__).resolve().parents[1]
    status = json.loads((release / 'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('Evaluation must finish before packaging')
    snapshot = release / '05_协议与复现/source_code/tools'
    snapshot.mkdir(exist_ok=True)
    for path in (root / 'tools').glob('*.py'):
        shutil.copy2(path, snapshot / path.name)
    protocol = json.loads((release / '05_协议与复现/protocol.json').read_text())
    assert protocol['final_sha256'] == '6973168c7c396a9d153d2dc7368ead203952f38839ec43e320166efa65b55340'
    assert protocol['validation_268']['count'] == 268
    assert protocol['validation_268']['self_references'] == 0
    for name, count in [('historical_240', 240), ('validation_268', 268)]:
        from PIL import Image

        frozen = release / '03_固定验证集' / name
        cases = json.loads((frozen / 'cases.json').read_text())
        assert len(cases) == count
        for role in ('query_content', 'reference_content'):
            (frozen / role).mkdir(exist_ok=True)
        for case in cases:
            cid = case['case_id']
            with Image.open(frozen / 'source' / f'{cid}.png') as source:
                assert source.size == (448, 896)
                source.crop((0, 448, 448, 896)).save(frozen / 'query_content' / f'{cid}.png')
                source.crop((0, 0, 448, 448)).save(frozen / 'reference_content' / f'{cid}.png')
        for model in ('baseline', 'final', 'comparison'):
            assert len(list((release / '02_推理图片' / name / model).glob('*.png'))) == count
        for case in cases:
            assert digest(release / case['pack']) == case['pack_sha256']
    protocol['image_roles'] = dict(query_content='target computer glyph',
                                   reference_content='reference computer glyph',
                                   reference='reference style glyph', gt='target style ground truth',
                                   source='reference/target computer glyphs stacked vertically',
                                   visible_target='reference style glyph above a hidden lower target')
    (release / '05_协议与复现/protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    usage = release / 'README_使用说明.md'
    text = usage.read_text()
    note = '\n单字输入：query_content/是目标电脑字，reference_content/是参考电脑字，reference/是参考书法字，gt/是目标GT；文件名与cases.json的case_id一一对应。\n'
    if note not in text:
        usage.write_text(text + note)
    entries = {}
    for path in sorted(release.rglob('*')):
        if path.is_file() and path.name not in ('文件校验_SHA256.json', 'status.json'):
            entries[str(path.relative_to(release))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    (release / '文件校验_SHA256.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    status.update(files=len(entries), payload_bytes=sum(x['bytes'] for x in entries.values()),
                  packaged_at=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'))
    (release / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
    archive = release.parent / 'callify_final_release.tar.gz'
    seen = {}
    duplicate_bytes = 0
    with tarfile.open(archive, 'w:gz', compresslevel=1) as package:
        for path in sorted(release.rglob('*')):
            if path.is_file():
                name = str(path.relative_to(release))
                key = (digest(path), path.stat().st_size)
                info = package.gettarinfo(str(path), arcname=name)
                if key in seen:
                    info.type, info.linkname, info.size = tarfile.LNKTYPE, seen[key], 0
                    package.addfile(info)
                    duplicate_bytes += key[1]
                else:
                    with path.open('rb') as stream:
                        package.addfile(info, stream)
                    seen[key] = name
    receipt = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=digest(archive),
                   payload_files=len(entries), payload_bytes=status['payload_bytes'],
                   identical_content_bytes_deduplicated=duplicate_bytes)
    (release.parent / 'callify_final_delivery.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
