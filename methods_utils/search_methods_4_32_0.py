# Derived and modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py

from transformers.generation.utils import *

from transformers import __version__ as transformers_version

assert transformers_version == "4.32.0"


def new_greedy_search(
    self: GenerationMixin,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    input_ids_cd: Optional[torch.LongTensor] = None,  # HalTrapper: New arg
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(input_ids.device)
        if eos_token_id is not None
        else None
    )
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )

    # HalTrapper: Modify here
    # copy model_kwargs for cd only for the first forward process
    use_cd = model_kwargs.get("use_cd")
    use_vcd = model_kwargs.get("use_vcd")

    if use_cd:
        if input_ids_cd is None:
            input_ids_cd = input_ids.clone()
        assert "input_scaling" not in model_kwargs.keys()
        assert "input_scaling_cd" not in model_kwargs.keys()
        cd_type = model_kwargs["cd_type"]
        model_kwargs_cd = model_kwargs.copy()
        # del model_kwargs['input_scaling_cd']
        # model_kwargs_cd['input_scaling'] = model_kwargs_cd['input_scaling_cd']
        # del model_kwargs_cd['input_scaling_cd']
        if "inputs_embeds_cd" in model_kwargs.keys():
            del model_kwargs["inputs_embeds_cd"]
            model_kwargs_cd["inputs_embeds"] = model_kwargs_cd["inputs_embeds_cd"]
            del model_kwargs_cd["inputs_embeds_cd"]
        if "attention_mask_cd" in model_kwargs.keys():
            del model_kwargs["attention_mask_cd"]
            model_kwargs_cd["attention_mask"] = model_kwargs_cd["attention_mask_cd"]
            del model_kwargs_cd["attention_mask_cd"]
        if "position_ids_cd" in model_kwargs.keys():
            del model_kwargs["position_ids_cd"]
            model_kwargs_cd["position_ids"] = model_kwargs_cd["position_ids_cd"]
            del model_kwargs_cd["position_ids_cd"]

        # HalTrapper: FIXME: Hardcode Here
        if "inputs_embeds" in model_kwargs_cd.keys():
            model_kwargs_cd["attention_mask"] = torch.ones(
                *model_kwargs_cd["inputs_embeds"].shape[:2],
                dtype=torch.int64,
                device=model_kwargs["attention_mask"].device,
            )
        else:
            model_kwargs_cd["attention_mask"] = torch.ones(
                *input_ids_cd.shape[:2],
                dtype=torch.int64,
                device=model_kwargs["attention_mask"].device,
            )
    else:
        cd_type = None

    this_peer_finished = False  # used by synced_gpus only
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            # input_scaling=model_kwargs['input_scaling'],  # HalTrapper: new argument
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # HalTrapper: Modified here
        if use_cd:
            # cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation(
                input_ids_cd, **model_kwargs_cd
            )

            outputs_cd = self(
                **model_inputs_cd,
                # input_scaling=model_kwargs_cd['input_scaling'],
                use_vcd=use_vcd,  # HalTrapper: new args
                return_dict=True,
                output_attentions=(
                    output_attentions
                    if output_attentions is not None
                    else self.generation_config.output_attentions
                ),
                output_hidden_states=(
                    output_hidden_states
                    if output_hidden_states is not None
                    else self.generation_config.output_hidden_states
                ),
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            if cd_type == "code":
                from methods_utils.code_dynamic_cd import code_cd

                cd_logits = code_cd(
                    model_kwargs, next_token_logits, next_token_logits_cd
                )

            else:
                # cd_comments: pre-process logits from contrastive inputs
                cd_alpha = (
                    model_kwargs.get("cd_alpha")
                    if model_kwargs.get("cd_alpha") is not None
                    else 0.5
                )
                cd_beta = (
                    model_kwargs.get("cd_beta")
                    if model_kwargs.get("cd_beta") is not None
                    else 0.1
                )

                # version 1  set cutoff for Adaptive Plausibility Constraints
                # probs = nn.functional.softmax(next_token_logits, dim=-1)
                # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values

                # version 2 set cutoff for Adaptive Plausibility Constraints
                cutoff = (
                    torch.log(torch.tensor(cd_beta))
                    + next_token_logits.max(dim=-1, keepdim=True).values
                )

                if cd_type == "contrastive":
                    diffs = (
                        1 + cd_alpha
                    ) * next_token_logits - cd_alpha * next_token_logits_cd

                elif cd_type == "augmentive":
                    diffs = next_token_logits + cd_alpha * next_token_logits_cd

                else:
                    raise ValueError(f"Unknown cd_type={cd_type}.")

                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            next_tokens_scores = logits_processor(input_ids, cd_logits)
        else:
            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError(
                    "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                )
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        # HalTrapper: Modified Here
        if use_cd:
            input_ids_cd = torch.cat([input_ids_cd, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # HalTrapper: Modified here
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids


def new_sample(
    self: GenerationMixin,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    input_ids_cd: Optional[torch.LongTensor] = None,  # HalTrapper: New arg
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = (
        logits_warper if logits_warper is not None else LogitsProcessorList()
    )
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(input_ids.device)
        if eos_token_id is not None
        else None
    )
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )

    # HalTrapper: Modify here
    # copy model_kwargs for cd only for the first forward process
    use_cd = model_kwargs.get("use_cd")
    use_vcd = model_kwargs.get("use_vcd")

    if use_cd:
        if input_ids_cd is None:
            input_ids_cd = input_ids.clone()
        assert "input_scaling" not in model_kwargs.keys()
        assert "input_scaling_cd" not in model_kwargs.keys()
        cd_type = model_kwargs["cd_type"]
        model_kwargs_cd = model_kwargs.copy()
        # del model_kwargs['input_scaling_cd']
        # model_kwargs_cd['input_scaling'] = model_kwargs_cd['input_scaling_cd']
        # del model_kwargs_cd['input_scaling_cd']
        if "inputs_embeds_cd" in model_kwargs.keys():
            del model_kwargs["inputs_embeds_cd"]
            model_kwargs_cd["inputs_embeds"] = model_kwargs_cd["inputs_embeds_cd"]
            del model_kwargs_cd["inputs_embeds_cd"]
        if "attention_mask_cd" in model_kwargs.keys():
            del model_kwargs["attention_mask_cd"]
            model_kwargs_cd["attention_mask"] = model_kwargs_cd["attention_mask_cd"]
            del model_kwargs_cd["attention_mask_cd"]
        if "position_ids_cd" in model_kwargs.keys():
            del model_kwargs["position_ids_cd"]
            model_kwargs_cd["position_ids"] = model_kwargs_cd["position_ids_cd"]
            del model_kwargs_cd["position_ids_cd"]

        # HalTrapper: FIXME: Hardcode Here
        if "inputs_embeds" in model_kwargs_cd.keys():
            model_kwargs_cd["attention_mask"] = torch.ones(
                *model_kwargs_cd["inputs_embeds"].shape[:2],
                dtype=torch.int64,
                device=model_kwargs["attention_mask"].device,
            )
        else:
            model_kwargs_cd["attention_mask"] = torch.ones(
                *input_ids_cd.shape[:2],
                dtype=torch.int64,
                device=model_kwargs["attention_mask"].device,
            )
    else:
        cd_type = None

    this_peer_finished = False  # used by synced_gpus only
    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            # input_scaling=model_kwargs['input_scaling'],  # HalTrapper: new argument
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution

        # HalTrapper: Modified here
        if use_cd:
            # cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation(
                input_ids_cd, **model_kwargs_cd
            )

            outputs_cd = self(
                **model_inputs_cd,
                # input_scaling=model_kwargs_cd['input_scaling'],
                use_vcd=use_vcd,  # HalTrapper: new args
                return_dict=True,
                output_attentions=(
                    output_attentions
                    if output_attentions is not None
                    else self.generation_config.output_attentions
                ),
                output_hidden_states=(
                    output_hidden_states
                    if output_hidden_states is not None
                    else self.generation_config.output_hidden_states
                ),
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            if cd_type == "code":
                from methods_utils.code_dynamic_cd import code_cd

                cd_logits = code_cd(
                    model_kwargs, next_token_logits, next_token_logits_cd
                )

            else:
                # cd_comments: pre-process logits from contrastive inputs
                cd_alpha = (
                    model_kwargs.get("cd_alpha")
                    if model_kwargs.get("cd_alpha") is not None
                    else 0.5
                )
                cd_beta = (
                    model_kwargs.get("cd_beta")
                    if model_kwargs.get("cd_beta") is not None
                    else 0.1
                )

                # version 1  set cutoff for Adaptive Plausibility Constraints
                # probs = nn.functional.softmax(next_token_logits, dim=-1)
                # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values

                # version 2 set cutoff for Adaptive Plausibility Constraints
                cutoff = (
                    torch.log(torch.tensor(cd_beta))
                    + next_token_logits.max(dim=-1, keepdim=True).values
                )

                if cd_type == "contrastive":
                    diffs = (
                        1 + cd_alpha
                    ) * next_token_logits - cd_alpha * next_token_logits_cd

                elif cd_type == "augmentive":
                    diffs = next_token_logits + cd_alpha * next_token_logits_cd

                else:
                    raise ValueError(f"Unknown cd_type={cd_type}.")

                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            next_tokens_scores = logits_processor(input_ids, cd_logits)
        else:
            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

        next_token_scores = logits_warper(input_ids, next_tokens_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError(
                    "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                )
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # HalTrapper: Modified Here
        if use_cd:
            input_ids_cd = torch.cat([input_ids_cd, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # HalTrapper: Modified here
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids


def new_beam_search(
    self: GenerationMixin,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    input_ids_cd: Optional[torch.LongTensor] = None,  # HalTrapper: New arg
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn(
            "You don't have defined any stopping_criteria, this will likely loop forever",
            UserWarning,
        )
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size))
        if (return_dict_in_generate and output_scores)
        else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    # HalTrapper: Modify here
    # copy model_kwargs for cd only for the first forward process
    use_cd = model_kwargs.get("use_cd")
    use_vcd = model_kwargs.get("use_vcd")

    if use_cd:
        if input_ids_cd is None:
            input_ids_cd = input_ids.clone()
        assert "input_scaling" not in model_kwargs.keys()
        assert "input_scaling_cd" not in model_kwargs.keys()
        cd_type = model_kwargs["cd_type"]
        model_kwargs_cd = model_kwargs.copy()
        # del model_kwargs['input_scaling_cd']
        # model_kwargs_cd['input_scaling'] = model_kwargs_cd['input_scaling_cd']
        # del model_kwargs_cd['input_scaling_cd']
        if "inputs_embeds_cd" in model_kwargs.keys():
            del model_kwargs["inputs_embeds_cd"]
            model_kwargs_cd["inputs_embeds"] = model_kwargs_cd["inputs_embeds_cd"]
            del model_kwargs_cd["inputs_embeds_cd"]
        if "attention_mask_cd" in model_kwargs.keys():
            del model_kwargs["attention_mask_cd"]
            model_kwargs_cd["attention_mask"] = model_kwargs_cd["attention_mask_cd"]
            del model_kwargs_cd["attention_mask_cd"]
        if "position_ids_cd" in model_kwargs.keys():
            del model_kwargs["position_ids_cd"]
            model_kwargs_cd["position_ids"] = model_kwargs_cd["position_ids_cd"]
            del model_kwargs_cd["position_ids_cd"]

        # HalTrapper: FIXME: Hardcode Here
        if "inputs_embeds" in model_kwargs_cd.keys():
            model_kwargs_cd["attention_mask"] = torch.ones(
                *model_kwargs_cd["inputs_embeds"].shape[:2],
                dtype=torch.int64,
                device=model_kwargs["attention_mask"].device,
            )
        else:
            model_kwargs_cd["attention_mask"] = torch.ones(
                *input_ids_cd.shape[:2],
                dtype=torch.int64,
                device=model_kwargs["attention_mask"].device,
            )
    else:
        cd_type = None

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            # input_scaling=model_kwargs['input_scaling'],  # HalTrapper: new argument
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # HalTrapper: Modified here
        if use_cd:
            # cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation(
                input_ids_cd, **model_kwargs_cd
            )
            outputs_cd = self(
                **model_inputs_cd,
                # input_scaling=model_kwargs_cd['input_scaling'],
                use_vcd=use_vcd,  # HalTrapper: new args
                return_dict=True,
                output_attentions=(
                    output_attentions
                    if output_attentions is not None
                    else self.generation_config.output_attentions
                ),
                output_hidden_states=(
                    output_hidden_states
                    if output_hidden_states is not None
                    else self.generation_config.output_hidden_states
                ),
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            if cd_type == "code":
                from methods_utils.code_dynamic_cd import code_cd

                cd_logits = code_cd(
                    model_kwargs, next_token_logits, next_token_logits_cd
                )

            else:
                # cd_comments: pre-process logits from contrastive inputs
                cd_alpha = (
                    model_kwargs.get("cd_alpha")
                    if model_kwargs.get("cd_alpha") is not None
                    else 0.5
                )
                cd_beta = (
                    model_kwargs.get("cd_beta")
                    if model_kwargs.get("cd_beta") is not None
                    else 0.1
                )

                # version 1  set cutoff for Adaptive Plausibility Constraints
                # probs = nn.functional.softmax(next_token_logits, dim=-1)
                # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values

                # version 2 set cutoff for Adaptive Plausibility Constraints
                cutoff = (
                    torch.log(torch.tensor(cd_beta))
                    + next_token_logits.max(dim=-1, keepdim=True).values
                )

                if cd_type == "contrastive":
                    diffs = (
                        1 + cd_alpha
                    ) * next_token_logits - cd_alpha * next_token_logits_cd

                elif cd_type == "augmentive":
                    diffs = next_token_logits + cd_alpha * next_token_logits_cd

                else:
                    raise ValueError(f"Unknown cd_type={cd_type}.")

                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            next_token_scores = nn.functional.log_softmax(
                cd_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
        else:
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores,
            max(2, 1 + n_eos_tokens) * num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        # HalTrapper: Modified Here
        if use_cd:
            input_ids_cd = torch.cat(
                [input_ids_cd[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        # HalTrapper: Modified here
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if model_kwargs_cd["past_key_values"] is not None:
                model_kwargs_cd["past_key_values"] = self._reorder_cache(
                    model_kwargs_cd["past_key_values"], beam_idx
                )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(
                (
                    beam_indices[beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices))
                )
            )

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]
