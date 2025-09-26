from typing import Union

import torch
from cccv.util.device import DEFAULT_DEVICE


def get_device(device: str) -> Union[torch.device, str]:
    """
    Get device from string

    :param device: device string
    """
    device = device.lower()

    if device.startswith("auto"):
        return DEFAULT_DEVICE
    elif device.startswith("cpu"):
        return torch.device("cpu")
    elif device.startswith("cuda"):
        return torch.device("cuda")
    elif device.startswith("mps"):
        return torch.device("mps")
    elif device.startswith("directml"):
        import torch_directml

        return torch_directml.device()
    elif device.startswith("xpu"):
        return torch.device("xpu")
    else:
        return DEFAULT_DEVICE
