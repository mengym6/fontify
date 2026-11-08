"""Periodically record the server experiment state, then exit after completion."""

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def snapshot(root, controller_pid, result_dir='experiment'):
    result = root / result_dir
    state = json.loads((result / 'status.json').read_text())
    training = Path(state.get('training_output', root / 'models/runb_fix_pixel'))
    log = training / 'log.txt'
    rows = []
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A training log write can be in progress during a check.
                pass
    try:
        os.kill(controller_pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    record = {
        'checked_at': datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),
        'experiment_status': state['status'], 'controller_pid': controller_pid,
        'controller_alive': alive,
        'completed_epochs': state.get('base_epochs', 0) + len(rows),
        'total_epochs': state.get('total_epochs', 51),
        'interval_seconds': 600,
    }
    if rows:
        record['latest_S'] = rows[-1]['test_S']
        record['best_S'] = min(row['test_S'] for row in rows)
    console = result / 'train_console.log'
    if console.exists():
        text = console.read_text(errors='replace')
        steps = re.findall(r'\[([0-9:.]+)\] Epoch: \[(\d+)\]\s+\[\s*(\d+)/(\d+)\]', text)
        if steps:
            stamp, epoch, step, total = steps[-1]
            record['training_progress'] = dict(time=stamp, epoch=int(epoch),
                                               batch=int(step), batches=int(total))
    terminal = state['status'] in ('complete', 'failed')
    record['attention_required'] = state['status'] == 'failed' or (not alive and not terminal)
    if record['attention_required']:
        record['error'] = state.get('error', 'Controller exited before experiment completion')
    if state['status'] == 'complete':
        record['metrics'] = state.get('metrics')
        record['results'] = str(result / 'index.html')
    temporary = result / 'monitor_status.tmp'
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temporary.replace(result / 'monitor_status.json')
    with (result / 'monitor_log.jsonl').open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return terminal or record['attention_required']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--result-dir', default='experiment')
    args = parser.parse_args()
    while True:
        terminal = snapshot(args.root, args.controller_pid, args.result_dir)
        if args.once or terminal:
            return
        time.sleep(600)


if __name__ == '__main__':
    main()
