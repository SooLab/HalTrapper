from . import (
    load_global_args,
    load_model_from_args,
    get_generation_params_from_args,
    get_eval_benchmark_from_args,
    seed_everything,
)
from ._utils._colors import print_warning

args, remain_args = load_global_args()

if args.seed is not None:
    seed_everything(args.seed)
shuffle = args.shuffle
raise_error = args.raise_error
n_samples = args.n_samples

model, remain_args = load_model_from_args(remain_args)
kwargs, remain_args = get_generation_params_from_args(remain_args)
benchmark, remain_args = get_eval_benchmark_from_args(remain_args)

if remain_args:
    print_warning(f"Cli args {remain_args} are not used.")

if benchmark is None:
    model.interact(**kwargs, raise_error=raise_error)
    if shuffle:
        print_warning("Shuffling is not supported in interactive mode.")
    if n_samples:
        print_warning("Sampling is not supported in interactive mode.")
else:
    model.eval(benchmark, shuffle=shuffle, n_samples=n_samples, **kwargs)
    if raise_error:
        print_warning("raise_error is not supported in eval mode, it will be ignored.")
