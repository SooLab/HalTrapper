from typing import Hashable
from pathlib import Path
import os
import json
import time
import random

from typing_extensions import Union, IO, Optional, Any
from collections.abc import Iterable

PathObj = Union[str, os.PathLike[str]]


def safe_open(
    file: PathObj,
    mode="r",
    buffering=-1,
    encoding="utf-8",
    retry=10,
    retry_interval_min=0.2,
    retry_interval_max=1.0,
    *args,
    **kwargs,
) -> IO[Any]:
    file_path = Path(file)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    last_exception = None

    for _ in range(retry):
        try:
            current_path = file_path
            current_mode = mode

            # Support mode='X': auto rename file when conflict
            if mode == "X" or (mode.startswith("X") and "b" in mode):
                base = file_path.stem
                suffix = file_path.suffix
                parent = file_path.parent
                counter = 1

                while current_path.exists():
                    new_name = f"{base}_{counter}{suffix}"
                    current_path = parent / new_name
                    counter += 1
                current_mode = mode.replace("X", "x")

            # actual open
            if "b" in current_mode:
                return open(
                    current_path, current_mode, buffering, None, *args, **kwargs
                )
            else:
                return open(
                    current_path, current_mode, buffering, encoding, *args, **kwargs
                )

        except Exception as e:
            last_exception = e
            time.sleep(random.uniform(retry_interval_min, retry_interval_max))

    # All trials fail
    assert last_exception is not None
    raise last_exception


def safe_close(file: IO[Any]):
    if file is None:  # type: ignore
        return
    if not file.closed:
        file.close()


def load_structured_file(file_path: PathObj) -> Any:
    file_path = Path(file_path)
    ext = file_path.suffix.lower().lstrip(".")

    try:
        with safe_open(file_path, "r", encoding="utf-8") as file:
            if ext == "json":
                return json.load(file)
            elif ext == "jsonl":
                return [json.loads(line) for line in file]
            elif ext in ("yaml", "yml"):
                import yaml

                return yaml.safe_load(file)
            else:
                raise ValueError(f"Unsupported file extension: .{ext}")
    except Exception as e:
        raise type(e)(f"Failed to read the file: {file_path}.") from e


StructuredObj = Union[
    Any,
    tuple["StructuredObj", ...],
    list["StructuredObj"],
    dict[Hashable, "StructuredObj"],
]


def save_structured_file(
    data: StructuredObj,
    file_path: PathObj,
    mode="x",
    ext: Optional[str] = None,
    **kwargs,
) -> Path:
    """
    Note:
        If mode='X' is used to automatically rename the file when it already exists,
        the actual file save path may differ from the provided 'file_path'.
        Always use the returned value as the accurate path to the saved file!
    """
    file_path = Path(file_path)
    if ext is None:
        ext = file_path.suffix.lower().lstrip(".")
    default_kwargs: dict[str, Any]

    try:
        with safe_open(file_path, mode, encoding="utf-8") as file:
            if ext == "json":
                default_kwargs = {"ensure_ascii": False, "indent": 4}
                default_kwargs.update(kwargs)
                json.dump(data, file, **default_kwargs)
            elif ext == "jsonl":
                if not isinstance(data, Iterable):
                    raise ValueError("JSONL format requires an iterable of objects.")
                default_kwargs = {"ensure_ascii": False}
                default_kwargs.update(kwargs)
                for item in data:
                    file.write(json.dumps(item, **default_kwargs) + "\n")
            elif ext in ("yaml", "yml"):
                import yaml

                default_kwargs = {
                    "allow_unicode": True,
                    "sort_keys": False,
                    "default_flow_style": False,
                }
                default_kwargs.update(kwargs)
                yaml.safe_dump(data, file, **default_kwargs)
            else:
                raise ValueError(f"Unsupported file extension: .{ext}")
            return Path(file.name)
    except Exception as e:
        raise RuntimeError(f"Failed to save the file: {file_path}") from e
