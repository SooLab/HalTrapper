import sys
from functools import partial


def _is_colorable():
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True
    except NameError:
        pass

    return sys.stdout.isatty()


def _color(code: str) -> str:
    return code if _is_colorable() else ""


_RESTORE = _color("\033[0m")

COLORS = {
    "red": _color("\033[91m"),
    "green": _color("\033[92m"),
    "yellow": _color("\033[93m"),
    "blue": _color("\033[94m"),
    "magenta": _color("\033[95m"),
    "cyan": _color("\033[96m"),
    "white": _color("\033[97m"),
    "grey": _color("\033[90m"),
}


def print_colored(label: str, main, color: str, sep: str = ":\n") -> None:
    color_code = COLORS.get(color, "")
    print(f"{color_code}{label}{sep}{_RESTORE}{main}")


print_with_label = partial(print_colored, color="yellow")
print_with_sublabel = partial(print_colored, color="green")


def print_line(n=None, color: str = "red") -> None:
    color_code = COLORS.get(color, "")
    line = "=" * 80 if n is None else f"{n:=>80d}"
    print(f"{color_code}{line}{_RESTORE}")


def print_tagged(main, label: str, color: str) -> None:
    color_code = COLORS.get(color, "")
    print(f"{color_code}[{label}]{_RESTORE} {main}")


print_note = partial(print_tagged, label="NOTICE", color="blue")
print_warning = partial(print_tagged, label="WARNING", color="yellow")
print_error = partial(print_tagged, label="ERROR", color="red")
print_success = partial(print_tagged, label="SUCCESS", color="green")
print_info = partial(print_tagged, label="INFO", color="cyan")
