# Derived and modified from: https://github.com/IVY-LVLM/CODE/blob/master/models/contrastive_decoding/cd_utils.py

import torch
from torch import nn


def code_cd(model_kwargs, next_token_logits, next_token_logits_cd):
    cd_alpha = (
        model_kwargs.get("cd_alpha")
        if model_kwargs.get("cd_alpha") is not None
        else 0.2
    )
    cd_beta = (
        model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.25
    )

    p_v = nn.functional.softmax(next_token_logits, dim=-1)
    p_d = nn.functional.softmax(next_token_logits_cd, dim=-1)

    kl_d = 0.5 * ((torch.log2(torch.abs(p_v - p_d) ** cd_alpha + 1)) * (p_v + p_d)).sum(
        dim=-1
    ).unsqueeze(-1)

    kld_alpha = 1 - kl_d

    cutoff = kl_d * p_v.max(dim=-1, keepdim=True).values

    ##############################
    diffs = (1 + kld_alpha) * next_token_logits - kld_alpha * next_token_logits_cd
    cd_logits = diffs.masked_fill(p_v < cutoff, -float("inf"))

    return cd_logits
