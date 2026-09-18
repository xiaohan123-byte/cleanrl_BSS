"""Bounded retries for Windows metadata writes, without repeating computation."""
from functools import wraps
import time


def retry_atomic_writer(writer, *, attempts=12, sleep=time.sleep):
    """Retry an idempotent JSON write only for Windows access/sharing errors.

    Callers already hold exclusive output locks. A retry rewrites the same
    temporary JSON and replaces its destination; it never appends a trajectory
    record or repeats a solver call. All other errors remain immediate failures.
    """
    @wraps(writer)
    def write(path, value):
        for attempt in range(attempts):
            try:
                return writer(path, value)
            except PermissionError as error:
                if getattr(error, "winerror", None) not in (5, 32, 33) or attempt + 1 == attempts:
                    raise
                sleep(min(.05 * 2**attempt, .5))
    return write
