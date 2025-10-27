import argparse
from pathlib import Path
from . import load_structured_file, get_eval_benchmark_from_args

from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument("log_file_path", type=str)
args, remain_args = parser.parse_known_args()
log_file_path = Path(args.log_file_path)

bench, args = get_eval_benchmark_from_args(remain_args)

if bench is None:
    raise ValueError("No benchmark specified.")

log_list = load_structured_file(log_file_path)
assert isinstance(log_list, list)

bench.get_score(log_list, log_file_path)
