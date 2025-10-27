# Derived and modified from: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/language_model/llava_llama.py

from llava.model.language_model.llava_llama import *
from .new_llava_arch import new_prepare_inputs_labels_for_multimodal


def new_LlavaLlamaForCausalLM_forward(
    self: LlavaLlamaForCausalLM,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,
    image_sizes: Optional[List[List[int]]] = None,
    return_dict: Optional[bool] = None,
    # HalTrapper: new arg
    input_scaling: Optional[torch.Tensor] = None,
    # Just to avoid ValueError
    inputs_embeds_cd: Optional[torch.FloatTensor] = None,
    cd_beta: Optional[torch.FloatTensor] = None,
    cd_alpha: Optional[torch.FloatTensor] = None,
    cd_alpha_aug: Optional[torch.FloatTensor] = None,
    cd_gamma: Optional[torch.FloatTensor] = None,
    use_cd: Optional[bool] = None,
    position_ids_cd: Optional[torch.LongTensor] = None,
    attention_mask_cd: Optional[torch.Tensor] = None,
    input_scaling_cd: Optional[torch.Tensor] = None,
    cd_type: Optional[str] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    if inputs_embeds is None:
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_sizes,
        )

    return super(LlavaLlamaForCausalLM, self).forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        input_scaling=input_scaling,  # HalTrapper: new arg
    )


def new_LlavaLlamaForCausalLM_generate(
    self: LlavaLlamaForCausalLM,
    input_ids: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    images_cd: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    input_ids_cd: Optional[torch.Tensor] = None,
    input_scaling: Optional[torch.Tensor] = None,
    input_scaling_cd: Optional[torch.Tensor] = None,
    cd_type: Optional[str] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    with torch.no_grad():
        position_ids = kwargs.pop("position_ids", None)
        position_ids_cd = kwargs.pop("position_ids_cd", None)
        attention_mask = kwargs.pop("attention_mask", None)
        attention_mask_cd = kwargs.pop("attention_mask_cd", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if "inputs_embeds_cd" in kwargs:
            raise NotImplementedError("`inputs_embeds_cd` is not supported")

        # HalTrapper: Modification Here
        use_cd = (input_ids_cd is not None or images_cd is not None) and (
            images is not None
        )
        if use_cd and input_ids_cd is None and input_ids is not None:
            input_ids_cd = input_ids.clone()
        if use_cd and images_cd is None and images is not None:
            images_cd = images.clone()
        # HalTrapper: Modification End

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                input_scaling,
            ) = new_prepare_inputs_labels_for_multimodal(
                self,
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                input_scaling=input_scaling,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        # HalTrapper: Modification Here
        if use_cd:
            (
                input_ids_cd,
                position_ids_cd,
                attention_mask_cd,
                _,
                inputs_embeds_cd,
                _,
                input_scaling_cd,
            ) = new_prepare_inputs_labels_for_multimodal(
                self,
                input_ids_cd,
                position_ids_cd,
                attention_mask_cd,
                None,
                None,
                images_cd,
                image_sizes=image_sizes,
                input_scaling=input_scaling_cd,
            )
        else:
            inputs_embeds_cd = None
            input_scaling_cd = None
        # HalTrapper: Modification End

        return super(LlavaLlamaForCausalLM, self).generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            position_ids_cd=position_ids_cd,
            attention_mask_cd=attention_mask_cd,
            inputs_embeds=inputs_embeds,
            inputs_embeds_cd=inputs_embeds_cd,
            input_scaling=input_scaling,
            input_scaling_cd=input_scaling_cd,
            use_cd=use_cd,
            cd_type=cd_type,
            **kwargs,
        )
