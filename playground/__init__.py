from .models import LM
from .benchmarks import BenchBase
import argparse
import inspect
import importlib

from typing import Optional, Any

from ._utils._colors import *
from ._utils._cuda import *
from ._utils._image import *
from ._utils._path import *
from ._utils._seed import *


def load_global_args(
    unknown_args_list: Optional[list[str]] = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Load general args.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--shuffle", action="store_true", help="Enable shuffling of benchmark."
    )
    parser.add_argument(
        "--n-samples",
        type=float,
        default=None,
        help="Number of samples to evaluate. Can be an integer (e.g., 100) or a float between 0 and 1 indicating a percentage (e.g., 0.1 means 10%% of the dataset).",
    )
    parser.add_argument(
        "--raise-error", action="store_true", help="Raise error in interactive mode."
    )
    args, unknown_args_list = parser.parse_known_args(unknown_args_list)
    return args, unknown_args_list


def load_model_from_args(
    unknown_args_list: Optional[list[str]] = None,
) -> tuple[LM, list[str]]:
    # 1. parse model name
    parser = argparse.ArgumentParser(description="Load a model.")
    parser.add_argument("model", type=str, help="model's command line name")

    args, unknown_args_list = parser.parse_known_args(unknown_args_list)

    model_name: str = args.model.lower()

    if ":" in model_name:
        module_path, model_name = model_name.split(":")
        importlib.import_module(module_path)

    try:
        registry_item = LM.registry[model_name]
    except KeyError as e:
        raise ValueError(
            f"Unsupported model {model_name}, should be in {list(LM.registry.keys())}."
        ) from e
    print_note(f"Loading model {model_name}")

    # 2. parse model arguments
    parser = argparse.ArgumentParser(
        description=f"Get arguments for model {model_name}."
    )
    method_signature = inspect.signature(registry_item.__init__)
    for i, (name, param) in enumerate(method_signature.parameters.items()):
        if i == 0:
            continue  # jump `self`
        parse_kwargs = {}
        if param.default is not inspect.Parameter.empty:
            parse_kwargs["default"] = param.default
        else:
            parse_kwargs["required"] = True
        if param.annotation is not inspect.Parameter.empty:
            parse_kwargs["type"] = param.annotation
        parser.add_argument(f'--{name.replace("_", "-")}', **parse_kwargs)

    args, unknown_args_list = parser.parse_known_args(unknown_args_list)

    kwargs = {}

    for key, value in args.__dict__.items():
        print_note(
            f'Model {model_name} is using {key}={value}, using --{key.replace("_", "-")} <value> to specify.'
        )
        kwargs[key] = value

    model = registry_item(**kwargs)
    return model, unknown_args_list


def get_generation_params_from_args(
    unknown_args_list: Optional[list[str]] = None,
) -> tuple[dict[str, Any], list[str]]:
    parser = argparse.ArgumentParser(description="Get generation parameters from args.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true")
    args, unknown_args_list = parser.parse_known_args(unknown_args_list)

    do_sample: bool = args.sample
    temperature: Optional[float] = args.temperature

    if temperature is None:
        if do_sample:
            temperature = 1.0
            print_note(
                "Temperature is not specified, using default temperature 1.0 for sampling."
            )
        else:
            temperature = 0.0

    if temperature == 0.0 and do_sample:
        print_warning(
            "Temperature is 0.0 but sampling is enabled, this will lead to deterministic outputs. "
            + "If you want to sample, please set temperature to a value greater than 0."
        )

    kwargs = dict(
        temperature=temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )

    print_note(f"Got parameters {kwargs}.")

    return kwargs, unknown_args_list


def get_eval_benchmark_from_args(
    unknown_args_list: Optional[list[str]] = None,
) -> tuple[Optional[BenchBase], list[str]]:
    # 1. parse benchmark name
    parser = argparse.ArgumentParser(description="Load a benchmark.")
    parser.add_argument("--eval", type=str, default=None, help="benchmark name")

    args, unknown_args_list = parser.parse_known_args(unknown_args_list)

    if args.eval is None:
        return None, unknown_args_list

    bench_name: str = args.eval.lower()

    try:
        registry_item = BenchBase.registry[bench_name]
    except KeyError as e:
        raise ValueError(
            f"Unsupported benchmark {bench_name}, should be in {list(BenchBase.registry.keys())}."
        ) from e
    print_note(f"Loading benchmark {bench_name}")

    # 2. parse benchmark arguments
    parser = argparse.ArgumentParser(
        description=f"Get arguments for benchmark {bench_name}."
    )
    method_signature = inspect.signature(registry_item.__init__)
    for i, (name, param) in enumerate(method_signature.parameters.items()):
        if i == 0:
            continue  # jump `self`
        parse_kwargs = {}
        if param.default is not inspect.Parameter.empty:
            parse_kwargs["default"] = param.default
        else:
            parse_kwargs["required"] = True
        if param.annotation is not inspect.Parameter.empty:
            parse_kwargs["type"] = param.annotation
        parser.add_argument(f'--{name.replace("_", "-")}', **parse_kwargs)

    args, unknown_args_list = parser.parse_known_args(unknown_args_list)

    kwargs = {}

    for key, value in args.__dict__.items():
        print_note(
            f'Benchmark {bench_name} is using {key}={value}, using --{key.replace("_", "-")} <value> to specify.'
        )
        kwargs[key] = value

    benchmark = registry_item(**kwargs)
    return benchmark, unknown_args_list
