"""Import only the isolated, versioned engine used by this experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engine"
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT))


def enable_atomic_retries():
    from src.atomic_io import retry_atomic_writer
    from src import experiment_control, rolling_runner
    if getattr(experiment_control.atomic_json, "_rl_retry", False):
        return experiment_control.atomic_json
    writer = retry_atomic_writer(experiment_control.atomic_json)
    writer._rl_retry = True
    experiment_control.atomic_json = writer
    rolling_runner.atomic_json = writer
    return writer
