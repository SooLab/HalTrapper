# Derived and modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py

from transformers.generation.utils import *  # type: ignore
from .prepare_cd import prepare_kwargs_for_cd

from transformers import __version__ as transformers_version

assert transformers_version == "4.48.3"


def new_sample(
    self: GenerationMixin,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
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
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__
    if isinstance(model_kwargs.get("past_key_values"), StaticCache):
        if self.device.type == "cuda":
            logger.warning_once("Using `torch.compile`.")
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

    # HalTrapper: Modify here
    # copy model_kwargs for cd only for the first forward process
    (
        input_ids,
        input_ids_cd,
        model_kwargs,
        model_kwargs_cd,
        use_cd,
        cd_type,
    ) = prepare_kwargs_for_cd(input_ids, model_kwargs)

    is_prefill = True
    is_prefill_cd = is_prefill
    while self._has_unfinished_sequences(
        this_peer_finished,
        synced_gpus,
        device=input_ids.device,
        cur_len=cur_len,
        max_length=max_length,
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        if is_prefill:
            outputs = self(
                **model_inputs,
                return_dict=True,
            )
            is_prefill = False
        else:
            outputs = model_forward(
                **model_inputs,
                return_dict=True,
            )

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # HalTrapper: Modified here
        if use_cd:
            # cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation(
                input_ids_cd, **model_kwargs_cd
            )

            model_inputs_cd.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs_cd.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            if is_prefill_cd:
                outputs = self(
                    **model_inputs_cd,
                    return_dict=True,
                )
                is_prefill_cd = False
            else:
                outputs = model_forward(
                    **model_inputs_cd,
                    return_dict=True,
                )

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits_cd = outputs.logits[:, -1, :].clone().float()
            next_token_logits_cd = next_token_logits_cd.to(input_ids_cd.device)

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

            next_token_scores = logits_processor(input_ids, cd_logits)
        else:
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
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

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        # HalTrapper: Modified here
        if use_cd:
            input_ids_cd = torch.cat([input_ids_cd, next_tokens[:, None]], dim=-1)

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def new_beam_search(
    self: GenerationMixin,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    raise NotImplementedError()
    # init values
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    sequential = generation_config.low_memory
    do_sample = generation_config.do_sample

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
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

    this_peer_finished = False

    # record the prompt length of decoder
    decoder_prompt_len = input_ids.shape[-1]

    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        # if sequential is True, split the input to batches of batch_size and run sequentially
        if sequential:
            if any(
                model_name in self.__class__.__name__.lower()
                for model_name in [
                    "fsmt",
                    "reformer",
                    "ctrl",
                    "gpt_bigcode",
                    "transo_xl",
                    "xlnet",
                    "cpm",
                    "jamba",
                ]
            ):
                raise RuntimeError(
                    f"Currently generation for {self.__class__.__name__} is not supported "
                    f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                )

            inputs_per_sub_batches = _split_model_inputs(
                model_inputs,
                split_size=batch_size,
                full_batch_size=batch_beam_size,
                config=self.config.get_text_config(),
            )
            outputs_per_sub_batch = [
                self(**inputs_per_sub_batch, return_dict=True)
                for inputs_per_sub_batch in inputs_per_sub_batches
            ]

            outputs = stack_model_outputs(
                outputs_per_sub_batch, self.config.get_text_config()
            )

        else:  # Unchanged original behavior
            outputs = self(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        # .float() is needed to retain precision for later logits manipulations
        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores_processed)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_logits:
                raw_logits += (next_token_logits,)
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

        # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
        # non eos token per beam.
        n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
        n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
            next_token_scores, _indices = torch.sort(
                next_token_scores, descending=True, dim=1
            )
            next_tokens = torch.gather(next_tokens, -1, _indices)
        else:
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
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
            decoder_prompt_len=decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
        # (that way the memory peak does not include outputs.logits)
        del outputs

        if model_kwargs.get("past_key_values", None) is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
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

        if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
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
        decoder_prompt_len=decoder_prompt_len,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return sequence_outputs["sequences"]
