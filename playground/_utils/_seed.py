from transformers import set_seed

from ._colors import print_note

from typing import Optional


def seed_everything(seed: Optional[int], **kwargs) -> None:
    if seed is not None:
        print_note(f"Using random seed {seed}")
        set_seed(seed, **kwargs)
