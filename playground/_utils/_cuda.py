import os
import torch
from ._colors import *


def assert_cuda_set_device() -> None:
    if not torch.cuda.is_available():
        print_warning("CUDA is not available.")
        return

    device_count = torch.cuda.device_count()
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")

    if device_count > 1:
        if not cuda_visible_devices:
            raise RuntimeError(
                "Detected multiple GPUs, but CUDA_VISIBLE_DEVICES is not set. Please set CUDA_VISIBLE_DEVICES to specify which GPUs to use."
            )
        else:
            print_note(
                f"Multi-GPU detected. CUDA_VISIBLE_DEVICES={cuda_visible_devices}"
            )
