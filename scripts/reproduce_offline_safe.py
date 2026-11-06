"""One supported entrypoint for the corrected official offline protocol."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="Parse local raw SLAM track and build safe features")
    action.add_argument("--preflight", action="store_true", help="Verify the safe dataset and influence tests")
    action.add_argument("--fit-boosted", choices=("benchmark", "deployment"))
    action.add_argument("--fit-neural-seed", type=int)
    action.add_argument("--select-stack-dev", choices=("benchmark", "deployment"))
    action.add_argument("--predict-neural-test-seed", type=int, choices=(42, 123, 2024))
    action.add_argument("--evaluate-stack-test", choices=("benchmark", "deployment"))
    ap.add_argument("--boosted-phase", choices=("status", "dev", "refit", "assemble", "all"), default="dev")
    args = ap.parse_args()
    if args.boosted_phase != "dev" and args.fit_boosted is None:
        ap.error("--boosted-phase requires --fit-boosted")
    if args.prepare:
        from src.data.preprocess import process_corpus
        # The generic parser is used only as an intermediate. Its output is
        # immediately rewritten by the split-frozen builder before any fit.
        process_corpus(root=ROOT, force=True, unsafe_intermediate=True)
        run("scripts/build_offline_safe_data.py", "--build")
        run("scripts/preflight_offline_safe.py")
        run("-m", "pytest", "-q", "tests/test_safe_protocol.py")
        run("scripts/run_safe_neural.py", "--preflight")
    elif args.preflight:
        run("scripts/preflight_offline_safe.py")
        run("-m", "pytest", "-q", "tests/test_safe_protocol.py")
        run("scripts/run_safe_neural.py", "--preflight")
    elif args.fit_boosted:
        run("scripts/preflight_offline_safe.py")
        run("scripts/run_safe_boosted.py", "--regime", args.fit_boosted,
            "--phase", args.boosted_phase)
    elif args.fit_neural_seed is not None:
        run("scripts/preflight_offline_safe.py")
        run("scripts/run_safe_neural.py", "--seed", str(args.fit_neural_seed))
    elif args.select_stack_dev:
        run("scripts/preflight_offline_safe.py")
        run("scripts/evaluate_safe_stack.py", "--select-dev", "--regime", args.select_stack_dev)
    elif args.predict_neural_test_seed is not None:
        run("scripts/predict_safe_neural_test.py", "--seed", str(args.predict_neural_test_seed))
    elif args.evaluate_stack_test:
        run("scripts/evaluate_safe_stack.py", "--evaluate-test", "--regime", args.evaluate_stack_test)


if __name__ == "__main__":
    main()
