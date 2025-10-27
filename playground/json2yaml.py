import argparse
from pathlib import Path
from ._utils._path import save_structured_file, load_structured_file

parser = argparse.ArgumentParser(description="utility: json2yaml")
parser.add_argument("load_path", type=str)
parser.add_argument("save_path", type=str, nargs="?")

args = parser.parse_args()

load_path = Path(args.load_path)
save_path = args.save_path

if save_path is None:
    save_path = load_path.with_suffix(".yaml")
else:
    save_path = Path(save_path)

data = load_structured_file(load_path)
save_structured_file(data, save_path)
print(f"File saved to `{save_path}`")
