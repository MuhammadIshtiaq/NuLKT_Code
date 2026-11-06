"""Resumable, private independent full-pipeline fits from corrected source.

Usage: python run_prespecified_repeats.py --source CORRECTED_ROOT --output EXPERIMENT_ROOT
All mutations stay below --output. No test result is used to change this schedule.
Source/data are copied, never modified. Interrupted individual fits are restarted;
completed fit artifacts are skipped. Do not run two coordinators on one output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

SCHEDULE = [(7, 17, 27), (37, 47, 57), (67, 77, 87)]

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, p)
def done(p):
    return p.is_file() and json.loads(p.read_text(encoding='utf-8')).get('status') == 'complete'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise RuntimeError('Source and output must be independent directories')
    output.mkdir(parents=True, exist_ok=True)
    manifest = source / 'data/processed/en_es/manifest.json'
    plan = {'protocol': 'round2-independent-full-pipeline-v1',
            'neural_seeds': SCHEDULE, 'lightgbm_seeds': SCHEDULE,
            'catboost_seeds': [s[:2] for s in SCHEDULE],
            'learner_group_cv_seed': [s[0] for s in SCHEDULE],
            'regimes': ['benchmark', 'deployment'],
            'same_architecture_and_training_budget': '100 epochs ceiling, patience 15, original corrected settings',
            'eligible_combiners': 'all original seven rules including boosted-only and neural-only',
            'data_manifest_sha256': sha(manifest),
            'test_policy': 'Previously inspected official TEST; post hoc, scored only after DEV rule freeze. No TEST-based selection.',
            'failure_policy': 'stop at failed phase; retain logs and resume explicitly, no seed substitution',
            'estimated_total_hours': [24, 48],
            'estimate_basis': 'Historical corrected seed42 fit 2.05 h; 9 neural fits plus 30 DEV and 30 refit boosted members across two regimes. Wall time depends on early stopping and hardware.'}
    plan_path = output / 'prespecified_schedule.json'
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if {k:v for k,v in old.items() if k != 'frozen_utc'} != json.loads(json.dumps(plan)):
            raise RuntimeError('Prespecified plan differs from existing immutable schedule')
    else:
        dump(plan_path, dict(plan, frozen_utc=now()))
    ledger = output / 'coordinator_events.jsonl'
    def event(stage, status, **extra):
        row = dict(utc=now(), stage=stage, status=status, **extra)
        with ledger.open('a', encoding='utf-8') as stream: stream.write(json.dumps(row) + '\n')
        dump(output / 'current_status.json', row)
        print(json.dumps(row), flush=True)
    def run(root, stage, command, completion=None, existence=False):
        if completion and (completion.exists() if existence else done(completion)):
            event(stage, 'already_complete')
            return
        event(stage, 'running', command=command)
        started = time.monotonic()
        log = root / 'logs' / (stage.replace('/', '_') + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, *command], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        seconds = time.monotonic() - started
        if result.returncode or (completion and not (completion.exists() if existence else done(completion))):
            event(stage, 'failed', returncode=result.returncode, seconds=seconds, log=str(log))
            raise RuntimeError(f'{stage} failed; see {log}')
        event(stage, 'complete', seconds=seconds)
    for repetition, seeds in enumerate(SCHEDULE, 1):
        root = output / f'repeat_{repetition:02d}'
        if not (root / 'copy_manifest.json').is_file():
            event(f'repeat_{repetition:02d}/prepare', 'running')
            for folder in ('src', 'scripts', 'configs', 'tests', 'data/processed'):
                shutil.copytree(source / folder, root / folder, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
            shutil.copy2(source / 'offline_safe_preflight.json', root / 'offline_safe_preflight.json')
            boost = root / 'scripts/run_safe_boosted.py'
            text = boost.read_text(encoding='utf-8')
            text = text.replace('LGB_SEEDS = (42, 123, 2024)', f'LGB_SEEDS = {seeds}')
            text = text.replace('CAT_SEEDS = (42, 123)', f'CAT_SEEDS = {seeds[:2]}')
            boost.write_text(text, encoding='utf-8')
            pred = root / 'scripts/predict_safe_neural_test.py'
            pred.write_text(pred.read_text(encoding='utf-8').replace('SEEDS = (42, 123, 2024)', f'SEEDS = {seeds}'), encoding='utf-8')
            stack = root / 'scripts/evaluate_safe_stack.py'
            stack.write_text(stack.read_text(encoding='utf-8').replace('n_splits=5, seed=42)', f'n_splits=5, seed={seeds[0]})'), encoding='utf-8')
            dump(root / 'copy_manifest.json', {'created_utc': now(), 'source_manifest_sha256': sha(manifest),
                 'copied_manifest_sha256': sha(root / 'data/processed/en_es/manifest.json'),
                 'seeds': seeds, 'code_sha256': {str(p.relative_to(root)):sha(p) for d in ('src','scripts','configs') for p in (root/d).rglob('*') if p.is_file()}})
        if args.prepare_only: continue
        prefix = f'repeat_{repetition:02d}'
        run(root, f'{prefix}/preflight', ['scripts/run_safe_neural.py', '--preflight'])
        neural = root / 'results/raw_runs/offline_safe_nulkt'
        for seed in seeds:
            run(root, f'{prefix}/neural_{seed}', ['scripts/run_safe_neural.py','--seed',str(seed)], neural/f'seed_{seed}/safe_fit_provenance.json')
        # Freeze both regimes before the first TEST inference for this repetition.
        for regime in ('benchmark','deployment'):
            run(root, f'{prefix}/{regime}_boosted_dev', ['scripts/run_safe_boosted.py','--regime',regime,'--phase','dev'])
            run(root, f'{prefix}/{regime}_select_dev', ['scripts/evaluate_safe_stack.py','--select-dev','--regime',regime], root/f'results/offline_safe_evaluation/{regime}/dev_selection.json', True)
        for seed in seeds:
            run(root, f'{prefix}/neural_{seed}_posthoc_test', ['scripts/predict_safe_neural_test.py','--seed',str(seed)], neural/f'seed_{seed}/safe_test_inference_provenance.json')
        for regime in ('benchmark','deployment'):
            run(root, f'{prefix}/{regime}_boosted_refit', ['scripts/run_safe_boosted.py','--regime',regime,'--phase','refit'])
            run(root, f'{prefix}/{regime}_boosted_assemble', ['scripts/run_safe_boosted.py','--regime',regime,'--phase','assemble'], root/f'results/raw_runs/offline_safe_gbdt_{regime}/safe_fit_provenance.json')
            run(root, f'{prefix}/{regime}_evaluate_posthoc', ['scripts/evaluate_safe_stack.py','--evaluate-test','--regime',regime,'--bootstraps','200'], root/f'results/offline_safe_evaluation/{regime}/test_evaluation_posthoc.json')
        event(prefix, 'complete')
    event('all_repetitions', 'prepared' if args.prepare_only else 'complete')

if __name__ == '__main__': main()
