from models_modified import (
    LlavaModified,
    JanusProModified,
    Qwen2VLModified,
    MiniGPT4Modified,
    QwenVLModified,
)
from parsers import AmberParser, ChairParser
from methods_utils.cache_table import ContextCDCandidates
from playground._utils._path import save_structured_file
from playground import get_eval_benchmark_from_args

from playground._utils._colors import *
from playground._utils._seed import seed_everything
from datetime import datetime
import argparse

from playground.chair.chair import (
    CHAIR,
)  # Do not remove, or the CHAIR parser cannot be loaded

from typing import Optional

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava")
    parser.add_argument("--method", type=str, default="haltrapper")
    parser.add_argument("--seed", type=int, default=42)

    # Output path
    parser.add_argument("--log-file-path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    # Transformers' parameters
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true")  # do_sample
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    # General CD algorithms' parameters
    # HalTrapper only has two parameters for cd algorithm, `cd_alpha` and `cd_beta`
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)

    # HalTrapper
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--repeat_mode", type=str, default="continuous")
    parser.add_argument("--ee_threshold", type=float, default=None)
    parser.add_argument("--ig_strategy", type=str, default="cos_sim")
    parser.add_argument("--ig_threshold", type=float, default=None)
    parser.add_argument("--candidates", type=int, default=None)  # Number of candidates
    parser.add_argument("--sep", type=str, default=None)

    # VCD
    parser.add_argument("--noise_step", type=int, default=500)

    # PAI, layer parameter for pai is hardcoded :(
    parser.add_argument("--pai_alpha", type=float, default=0.5)

    # Benchmark's information is in the `remain_args`
    args, remain_args = parser.parse_known_args()

    seed_everything(args.seed)

    method: str = args.method.lower()

    if method == "haltrapper" or method == "ours":
        method = "haltrapper"

    model_name: str = args.model.lower()
    assert model_name in [
        "llava",
        "llava13",
        "januspro",
        "qwen2vl",
        "minigpt4",
        "qwenvl",
    ]
    assert method in ["baseline", "vcd", "icd", "pai", "code", "haltrapper"]

    print_note(f"Using model {model_name}.")
    print_note(f"Using method {method}.")

    if model_name == "llava":
        model = LlavaModified()
    elif model_name == "llava13":
        model = LlavaModified(size="13b")
    elif model_name == "januspro":
        model = JanusProModified()
    elif model_name == "qwen2vl":
        model = Qwen2VLModified()
    elif model_name == "minigpt4":
        model = MiniGPT4Modified()
    elif model_name == "qwenvl":
        model = QwenVLModified()
    else:
        raise

    # Select hyperparameter for HalTrapper
    ee_threshold: Optional[float] = args.ee_threshold
    ig_threshold: Optional[float] = args.ig_threshold
    candidates: Optional[int] = args.candidates
    sep: Optional[str] = args.sep
    if ee_threshold is None:
        ee_threshold = 0.0 if model_name in ["minigpt4", "qwenvl"] else 1.0
    if ig_threshold is None:
        ig_threshold = 0.85 if model_name == "qwenvl" else 0.75
    if candidates is None:
        candidates = 5 if model_name in ["qwen2vl", "qwenvl", "januspro"] else 10
    if sep is None:
        sep = ", " if model_name == "minigpt4" else " "

    # `ee_threshold` should take negative due to history reasons
    ee_threshold = -ee_threshold

    kwargs = {
        "temperature": args.temperature if args.sample else 0.0,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.sample,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "top_p": args.top_p,
    }

    if method == "haltrapper":
        method_kwargs = {
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "repeat": args.repeat,
            "repeat_mode": args.repeat_mode,
            "sep": sep,
            "cd_type": "contrastive",
            "candidates_number": candidates,
            "ee_threshold": ee_threshold,
            "ig_threshold": ig_threshold,
            "ig_strategy": args.ig_strategy,
        }
    elif method == "vcd":
        method_kwargs = {
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "noise_step": args.noise_step,
            "cd_type": "contrastive",
        }
    elif method == "icd":
        method_kwargs = {
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "cd_type": "contrastive",
        }
    elif method == "pai":
        method_kwargs = {"pai_alpha": args.pai_alpha}
    elif method == "code":
        method_kwargs = {
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "cd_type": "code",
        }
    else:
        method_kwargs = {}

    benchmark, remain_args = get_eval_benchmark_from_args(remain_args)
    assert benchmark is not None
    if len(remain_args) > 0:
        print_warning(f"Unknown arguments: {remain_args}. They will be ignored.")

    if benchmark.name == "amber":
        print_note("Using AMBER parser...")
        parser = AmberParser()
    else:
        print_note("Using CHAIR parser...")
        parser = ChairParser()

    model.parser = parser
    model.ct = ContextCDCandidates(model, parser)

    timestr = datetime.now().astimezone().isoformat()

    if args.log_file_path is not None:
        log_file_path: str = args.log_file_path
        if not log_file_path.endswith(".jsonl"):
            log_file_path += ".jsonl"
    else:
        log_file_path = f"./results/{model.name}-{benchmark.name}-{method}.jsonl"

    model.set_log_file_path(log_file_path, "w" if args.overwrite else "X")

    # Re-get the actual `log_file_path` (may be renamed due to conflict)
    log_file_path = model.log_file_path

    save_structured_file(
        {
            "args": sys.argv,
            "model": model.name,
            "benchmark": benchmark.name,
            "method": method,
            "start_time": datetime.now().astimezone().isoformat(),
            "kwargs": {**kwargs, **method_kwargs},
        },
        log_file_path.parent / (log_file_path.stem + "-config.yaml"),
        "w",
    )

    model.eval(
        benchmark,
        method=method,
        **kwargs,
        **method_kwargs,
    )
