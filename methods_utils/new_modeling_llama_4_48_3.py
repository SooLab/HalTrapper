# Derived and modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

import torch
from transformers.models.llama.modeling_llama import (
    Cache,
    LlamaForCausalLM,
    KwargsForCausalLM,
    CausalLMOutputWithPast,
)

from typing import Optional, Union, List, Tuple
from transformers.models.llama.modeling_llama import Unpack
from transformers import __version__ as transformers_version

assert transformers_version == "4.48.3"


def new_LlamaForCausalLM_forward(
    self: LlamaForCausalLM,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    # HalTrapper: new arg
    input_scaling: Optional[torch.Tensor] = None,
    # Just to avoid ValueError
    inputs_embeds_cd: Optional[torch.FloatTensor] = None,
    cd_beta: Optional[torch.FloatTensor] = None,
    cd_alpha: Optional[torch.FloatTensor] = None,
    use_cd: Optional[bool] = None,
    position_ids_cd: Optional[torch.LongTensor] = None,
    attention_mask_cd: Optional[torch.Tensor] = None,
    input_scaling_cd: Optional[torch.Tensor] = None,
    cd_type: Optional[str] = None,
    **kwargs: Unpack[KwargsForCausalLM],
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
