from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator


@contextmanager
def data_path(*relative: str) -> Iterator[Path]:
    """Yield a filesystem path for bundled data regardless of install layout."""
    resource = resources.files('catsim').joinpath('data', *relative)
    with resources.as_file(resource) as resolved_path:
        yield Path(resolved_path)
