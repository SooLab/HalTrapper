import os
from ._utils._colors import print_warning, print_note

from typing import List, Union
from pathlib import Path

_PATH_TABLE = {
    "llava-v1.5-7b": "liuhaotian/llava-v1.5-7b",
    "llava-v1.5-13b": "liuhaotian/llava-v1.5-13b",
    "llava-v1.6-vicuna-7b": "liuhaotian/llava-v1.6-vicuna-7b",
    "llava-v1.6-vicuna-13b": "liuhaotian/llava-v1.6-vicuna-7b",
    "llava-v1.6-34b": "liuhaotian/llava-v1.6-34b",
    "Qwen-VL-Chat": "Qwen/Qwen-VL-Chat",
    "Qwen2-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Janus-Pro-7B": "deepseek-ai/Janus-Pro-7B",
    "mpt-7b": "mosaicml/mpt-7b",
    "siglip-so400m-patch14-384": "google/siglip-so400m-patch14-384",
    # You should set up these paths.
    # Please download COCO dataset from https://cocodataset.org/
    "COCO path": "path/to/COCO/val2014",
    # `git clone https://github.com/junyangwang0410/AMBER.git`, then set up the AMBER first
    "AMBER path": "path/to/AMBER",
    # `git clone https://github.com/Vision-CAIR/MiniGPT-4.git`, then set up the MiniGPT-4 first
    "MiniGPT4 repo root": "path/to/MiniGPT-4",
}


def get_path_from_table(name: str) -> Path:
    if name not in _PATH_TABLE:
        raise KeyError(f"'{name}' not found in path table.")
    path = _PATH_TABLE[name]
    print_note(f"Get '{name}' from path {path}")
    return Path(path)


# _current_file_path = os.path.abspath(__file__) if "__file__" in globals() else None

# _ROOT_PATHS: List[Union[str, None]] = [
#     os.getcwd(),
#     os.path.dirname(_current_file_path) if _current_file_path else None,
#     os.path.expanduser("~"),
# ]


# def get_path_from_table(name: str) -> Path:
#     if name not in _PATH_TABLE:
#         raise KeyError(f"Model '{name}' not found in path table.")

#     path = _PATH_TABLE[name]

#     if os.path.isabs(path):
#         print_note(f"Get '{name}' from path {path}")
#         return Path(path)

#     checked_paths = []
#     for root in _ROOT_PATHS:
#         if root is None:
#             continue
#         full_path = os.path.join(root, path)
#         try:
#             checked_paths.append(full_path)
#             if os.path.exists(full_path):
#                 print_note(f"Get {name!r} from path {full_path!r}")
#                 return Path(full_path)
#         except Exception as e:
#             print_warning(
#                 f"Unexpected error occurred while checking path {full_path!r}: {e}"
#             )

#     raise FileNotFoundError(f"Path for {name!r} not found. Checked {checked_paths}")
