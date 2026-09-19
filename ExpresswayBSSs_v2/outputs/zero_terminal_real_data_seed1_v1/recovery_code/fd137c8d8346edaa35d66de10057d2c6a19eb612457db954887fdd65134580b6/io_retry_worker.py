"""Execute an immutable worker with only its JSON writer wrapped for retries."""
import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source", type=Path, required=True)
    args, worker_args = parser.parse_known_args()
    frozen = args.frozen_source.resolve()
    spec = importlib.util.spec_from_file_location("atomic_io_patch", Path(__file__).with_name("atomic_io.py"))
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    # Import the original experiment package before the worker imports names
    # from it. No frozen file or mathematical-model function is changed.
    sys.path.insert(0, str(frozen))
    import src.experiment_control as control
    control.atomic_json = patch.retry_atomic_writer(control.atomic_json)
    sys.argv = [str(frozen / "experiment_worker.py"), *worker_args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
