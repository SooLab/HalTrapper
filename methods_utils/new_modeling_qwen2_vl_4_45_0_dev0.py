# Derived and modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py

from transformers.models.qwen2_vl.modeling_qwen2_vl import *  # type: ignore
from transformers import __version__ as transformers_version

assert transformers_version.startswith("4.45.0")


def new_Qwen2VLForConditionalGeneration_forward(
    self: Qwen2VLForConditionalGeneration,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    # HalTrapper: new arg
    input_scaling: Optional[torch.Tensor] = None,
    # Just to avoid ValueError
    input_ids_cd: Optional[torch.LongTensor] = None,
    inputs_embeds_cd: Optional[torch.FloatTensor] = None,
    cd_beta: Optional[torch.FloatTensor] = None,
    cd_alpha: Optional[torch.FloatTensor] = None,
    use_cd: Optional[bool] = None,
    position_ids_cd: Optional[torch.LongTensor] = None,
    attention_mask_cd: Optional[torch.Tensor] = None,
    input_scaling_cd: Optional[torch.Tensor] = None,
    cd_type: Optional[str] = None,
    pixel_values_cd: Optional[torch.Tensor] = None,
    image_grid_thw_cd: Optional[torch.LongTensor] = None,
    pixel_values_videos_cd: Optional[torch.FloatTensor] = None,
    video_grid_thw_cd: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
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

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(
                inputs_embeds.device
            )
            image_mask = input_ids == self.config.image_token_id
            if self.training:
                inputs_embeds = inputs_embeds.clone()
            inputs_embeds[image_mask] = image_embeds
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(
                inputs_embeds.device
            )
            video_mask = input_ids == self.config.video_token_id
            inputs_embeds[video_mask] = video_embeds
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )
