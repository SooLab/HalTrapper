# Derived and modified from: https://github.com/LALBJ/PAI/blob/master/CFG.py


import torch
import torch.nn.functional as F
from transformers import (
    LogitsProcessor,
)


class CFGLogits(LogitsProcessor):
    def __init__(
        self,
        guidance_scale,
        uncond,
        model,
        image=None,
        input_type="inputs_ids",
        start_layer=0,
        end_layer=32,
    ):
        self.guidance_scale = guidance_scale
        self.uncond = uncond
        self.model = model
        self.image = image
        self.out = None
        self.input_type = input_type
        self.start_layer = start_layer
        self.end_layer = end_layer

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores
        # for i in range(self.start_layer, self.end_layer):
        #     self.model.model.layers[i].self_attn.use_cfg = True

        if self.out is None:
            if self.input_type == "inputs_ids":
                self.out = self.model(self.uncond, use_cache=True)
            elif self.input_type == "inputs_embeds":
                self.out = self.model(inputs_embeds=self.uncond, use_cache=True)
            else:
                print("Neither input_ids nor inputs_embeds is provided.")
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )
        # for i in range(self.start_layer, self.end_layer):
        #     self.model.model.layers[i].self_attn.use_cfg = False

        unconditional_logits = F.log_softmax(self.out.logits[:, -1, :], dim=-1)

        cutoff = torch.log(torch.tensor(0.1)) + scores.max(dim=-1, keepdim=True).values
        out = (
            self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        )
        cd_logits = out.masked_fill(scores < cutoff, -float("inf"))
        return cd_logits


def init_cfg_processor(
    tokenizer, llm_base_model, questions, gamma=1.1, beam=1, start_layer=0, end_layer=32
):
    # if self.model_name == "minigpt4":
    #     chunks = [q.split("<Img><ImageHere></Img>") for q in questions]
    # elif self.model_name == "llava-1.5":
    chunks = [q.split("<image>") for q in questions]
    # elif self.model_name == "shikra":
    #     split_token = (
    #         "<im_start>"
    #         + DEFAULT_IMAGE_PATCH_TOKEN * SHIKRA_IMAGE_TOKEN_LENGTH
    #         + "<im_end>"
    #     )
    #     chunks = [q.split(split_token) for q in questions]
    # else:
    #     raise ValueError(f"Unknown model: {self.model_name}")
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    token_before = tokenizer(
        chunk_before,
        return_tensors="pt",
        padding="longest",
        add_special_tokens=False,
    ).input_ids.to("cuda")
    token_after = tokenizer(
        chunk_after,
        return_tensors="pt",
        padding="longest",
        add_special_tokens=False,
    ).input_ids.to("cuda")

    batch_size = len(questions)
    bos = (
        torch.ones(
            [batch_size, 1], dtype=token_before.dtype, device=token_before.device
        )
        * tokenizer.bos_token_id
    )
    neg_promt = torch.cat([bos, token_before, token_after], dim=1)
    neg_promt = neg_promt.repeat(beam, 1)
    logits_processor = CFGLogits(
        gamma,
        neg_promt.to("cuda"),
        llm_base_model,
        start_layer=start_layer,
        end_layer=end_layer,
    )

    return logits_processor
