# Derived and modified from Tongyi Qianwen Materials

from methods_utils.vcd_add_noise import add_diffusion_noise
from methods_utils.qwen_vl_chat.modeling_qwen import *
from methods_utils.qwen_vl_chat.modeling_qwen import (
    _SENTINEL,
    _ERROR_STREAM_IN_CHAT,
    _ERROR_BAD_CHAT_FORMAT,
)


def new_chat(
    self: QWenLMHeadModel,
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: Optional[HistoryType],
    system: str = "You are a helpful assistant.",
    append_history: bool = True,
    stream: Optional[bool] = _SENTINEL,
    stop_words_ids: Optional[List[List[int]]] = None,
    generation_config: Optional[GenerationConfig] = None,
    # HalTrapper: New args
    method=None,
    cd_type=None,
    noise_step=None,
    hallu_objs=None,
    repeat=1,
    repeat_mode="continuous",
    pai_alpha=None,
    caption=None,
    append=None,
    update_input_ids_hook=None,
    sep=" ",
    **kwargs,
) -> Tuple[str, Any]:
    # HalTrapper: Modified Here
    from methods_utils.new_qwen_generation_utils import make_context

    generation_config = (
        generation_config if generation_config is not None else self.generation_config
    )

    assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
    assert generation_config.chat_format == "chatml", _ERROR_BAD_CHAT_FORMAT
    if history is None:
        history = []
    if stop_words_ids is None:
        stop_words_ids = []

    max_window_size = kwargs.get("max_window_size", None)
    if max_window_size is None:
        max_window_size = generation_config.max_window_size
    raw_text, context_tokens = make_context(
        tokenizer,
        query,
        history=history,
        system=system,
        max_window_size=max_window_size,
        chat_format=generation_config.chat_format,
        append=append,  # HalTrapper: New arg
    )

    # HalTrapper: Modified Here
    input_ids = torch.tensor([context_tokens]).to(self.device)

    if method is None:
        method = "baseline"
    if method == "haltrapper" and (hallu_objs is None or len(hallu_objs) == 0):
        method = "baseline"

    if method == "pai":
        raise NotImplementedError()

    use_vcd = False
    if method not in ["baseline", "vcd", "pai"]:
        assert append is None
        raw_text_cd, context_tokens_cd = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
            method=method,
            hallu_objs=hallu_objs,
            repeat=repeat,
            repeat_mode=repeat_mode,
            caption=caption,
            sep=sep,  # HalTrapper: New arg
        )
        input_ids_cd = torch.tensor([context_tokens_cd]).to(self.device)
    elif method == "vcd":
        use_vcd = True
        input_ids_cd = input_ids.clone()
    else:
        raw_text_cd = None
        context_tokens_cd = None
        input_ids_cd = None
    use_cd = input_ids_cd is not None
    # HalTrapper: Modification Ends

    # print(tokenizer.convert_ids_to_tokens(input_ids_cd[0]))

    stop_words_ids.extend(get_stop_words_ids(generation_config.chat_format, tokenizer))

    assert update_input_ids_hook is not None
    update_input_ids_hook(input_ids)

    outputs = self.generate(
        input_ids,
        input_ids_cd=input_ids_cd,  # HalTrapper: New arg
        stop_words_ids=stop_words_ids,
        # return_dict_in_generate=False,  # HalTrapper: Modified
        generation_config=generation_config,
        # CCD： New Args
        use_cd=use_cd,
        cd_type=cd_type,
        use_vcd=use_vcd,
        noise_step=noise_step,
        **kwargs,
    )

    # HalTrapper: Modification Here
    if "return_dict_in_generate" in kwargs.keys() and kwargs["return_dict_in_generate"]:
        output_ids = outputs["sequences"]
    else:
        output_ids = outputs
        outputs = None
    # HalTrapper: Modification Ends

    response = decode_tokens(
        output_ids[0],
        tokenizer,
        raw_text_len=len(raw_text),
        context_length=len(context_tokens),
        chat_format=generation_config.chat_format,
        verbose=False,
        errors="replace",
    )

    if append_history:
        history.append((query, response))

    return response, outputs


def new_QWenLMHeadModel_forward(
    self: QWenLMHeadModel,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    # VCD args
    use_vcd: bool = False,
    noise_step: int = 500,
    # Just to avoid ValueError
    input_ids_cd: Optional[torch.LongTensor] = None,
    cd_beta: Optional[torch.FloatTensor] = None,
    cd_alpha: Optional[torch.FloatTensor] = None,
    use_cd: Optional[bool] = None,
    position_ids_cd: Optional[torch.LongTensor] = None,
    attention_mask_cd: Optional[torch.Tensor] = None,
    input_scaling_cd: Optional[torch.Tensor] = None,
    cd_type: Optional[str] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        use_vcd=use_vcd,  # HalTrapper: New arg
        noise_step=noise_step,  # HalTrapper: New arg
    )
    hidden_states = transformer_outputs[0]

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


def new_QWenModel_forward(
    self: QWenModel,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_vcd: bool = False,  # HalTrapper: New arg
    noise_step: int = 500,  # HalTrapper: New arg
):
    if past_key_values is None and torch.any(
        input_ids == self.config.visual["image_start_id"]
    ):
        bos_pos = torch.where(input_ids == self.config.visual["image_start_id"])
        eos_pos = torch.where(input_ids == self.config.visual["image_start_id"] + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        # HalTrapper: Modified Here
        # img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        # HalTrapper: FIXME: Hardcoded image token length = 256
        img_pos = torch.stack((bos_pos[0], bos_pos[1], bos_pos[1] + 257), dim=1)
        images = []
        for i, a, b in img_pos:
            image = input_ids[i][a + 1 : b - 1].tolist()
            image = image[: image.index(self.config.visual["image_start_id"] + 2)]
            images.append(bytes(image).decode("utf-8"))

        images = self.visual.encode(images)
        # HalTrapper: Modified Here
        if use_vcd:
            images = add_diffusion_noise(images, noise_step)
        assert images.shape[0] == len(images)
        fake_images = None
    elif self.training:
        fake_images = torch.zeros(1, 3, 224, 224).to(
            dtype=self.visual.conv1.weight.dtype, device=self.visual.conv1.weight.device
        )
        images = self.visual(fake_images)
    else:
        fake_images = None
        images = None

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
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            input_shape[-1] + past_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    encoder_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    if batch_size <= 0:
        raise ValueError("batch_size has to be defined and > 0")
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_length
    )

    hidden_states = inputs_embeds

    kv_seq_len = hidden_states.size()[1]
    if past_key_values[0] is not None:
        # past key values[0][0] shape: bs * seq_len * head_num * dim
        kv_seq_len += past_key_values[0][0].shape[1]
    if (
        self.use_dynamic_ntk
        and kv_seq_len == hidden_states.size()[1]
        and not self.training
    ):
        context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
    else:
        ntk_alpha = self.rotary_emb._ntk_alpha_cached

    rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
    for idx in range(len(rotary_pos_emb)):
        rotary_pos_emb[idx] = rotary_pos_emb[idx].to(hidden_states.device)

    hidden_states = self.drop(hidden_states).clone()
    if fake_images is not None:
        hidden_states = hidden_states + images.mean() * 0
    elif images is not None:
        for idx, (i, a, b) in enumerate(img_pos):
            hidden_states[i][a + 1 : b] = images[idx]
    output_shape = input_shape + (hidden_states.size(-1),)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                rotary_pos_emb,
                self.registered_causal_mask,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                rotary_pos_emb=rotary_pos_emb,
                registered_causal_mask=self.registered_causal_mask,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1],
            )

    hidden_states = self.ln_f(hidden_states)
    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, presents, all_hidden_states] if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
