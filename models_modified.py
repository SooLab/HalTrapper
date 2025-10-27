import types
import torch

from playground.models import LLaVA, JanusPro, Qwen2VL, QwenVL, LM
from playground.path_table import get_path_from_table
from parsers import BaseParser

from playground._utils._colors import *

from methods_utils.cache_table import ContextCDCandidates
from methods_utils.graber import Graber
from methods_utils.vcd_add_noise import add_diffusion_noise

from tqdm import tqdm

import random
import os

from abc import ABC
from typing import TYPE_CHECKING, Dict, Tuple, Optional

if TYPE_CHECKING:
    from playground._utils._path import PathObj


GRABER = Graber()


class CTMixin(ABC):
    ct: ContextCDCandidates
    parser: BaseParser

    def get_candidates_from_cache(
        self,
        image: "PathObj",
        number: int,
        ee_threshold: float,
        ig_threshold: float,
        ig_strategy: str,
        show_progress: bool = False,
        random_state: Optional[int] = None,
    ) -> list[str]:
        datadict = self.ct.get_candidates(image, show_progress)
        _, caption_objs, _, _ = self.parser.extract_nouns(datadict["caption"])

        appeared = set(caption_objs)
        appeared |= set(datadict["metric1"]["scores"].keys())
        appeared |= set(datadict["metric2"]["scores"].keys())

        related = set()
        for word in appeared:
            related.add(word)
            for subword in self.parser.SAFE_WORDS[word]:
                related.add(subword)

        related_caption = set()
        for word in caption_objs:
            related_caption.add(word)
            for subword in self.parser.SAFE_WORDS[word]:
                related_caption.add(subword)

        unrelated = set(self.parser.PARSER_WORDS) - related
        unrelated = sorted(unrelated)  # Ensure the reproducibility

        candidates = {}

        for obj, score in datadict["metric2"]["scores"].items():
            score_strategy = score[ig_strategy]
            if score_strategy > ig_threshold:
                candidates[obj] = score_strategy

        if show_progress:
            tqdm.write(f"IG: {list(candidates.keys())}")

        EE_list = []

        for obj, score in datadict["metric1"]["scores"].items():
            if score < ee_threshold and obj not in candidates.keys():
                EE_list.append(obj)
                candidates[obj] = -1.0

        if show_progress:
            tqdm.write(f"EE: {EE_list}")

        candidates = sorted(candidates, key=candidates.get, reverse=True)  # type:ignore

        if random_state is not None:
            random_generator = random.Random(random_state)
        else:
            random_generator = random
            print_warning(
                "`random_state` is set to `None` when generating candidates, which may lead to unreproducable results."
            )

        if number >= 1:  # Added no add random candidates
            if len(candidates) > number:
                candidates = candidates[:number]
            else:
                candidates = candidates + random_generator.sample(
                    unrelated, min(number - len(candidates), len(unrelated))
                )

        candidates = sorted(candidates)

        if show_progress:
            tqdm.write(f"Final candidates: {candidates}")

        return candidates

    def preprocess_method(
        self,
        image,
        method,
        candidates_number,
        ee_threshold,
        ig_threshold,
        ig_strategy,
        question_id: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        GRABER.clear()
        if method == "haltrapper":
            assert image is not None
            assert candidates_number is not None
            assert ee_threshold is not None
            assert ig_threshold is not None
            candidates = {}
            for obj in self.get_candidates_from_cache(
                image,
                candidates_number,
                ee_threshold,
                ig_threshold,
                ig_strategy,
                random_state=question_id,
                show_progress=False,
            ):
                candidates[obj] = 0.0
            hallu_objs = candidates
        else:
            hallu_objs = None

        if method == "code":
            assert image is not None
            caption = self.ct.get_candidates(image, show_progress=False)["caption"]
        else:
            caption = None

        return hallu_objs, caption

    def close_cache_table(self):
        self.ct.close()


class LlavaModified(LLaVA, CTMixin):
    def __init__(self, size="7b") -> None:
        from transformers.generation.utils import (
            GenerationMixin,
        )
        from transformers.models.llama.modeling_llama import (
            LlamaModel,
            LlamaSdpaAttention,
            LlamaAttention,
        )
        from transformers.models.llama import LlamaForCausalLM

        from methods_utils.search_methods_4_37_2 import (
            new_greedy_search,
            new_sample,
            new_beam_search,
        )
        from methods_utils.new_llava_llama import (
            new_LlavaLlamaForCausalLM_generate,
            new_LlavaLlamaForCausalLM_forward,
        )
        from methods_utils.new_modeling_llama_4_37_2 import (
            new_LlamaForCausalLM_forward,
            new_LlamaModel_forward,
            new_LlamaSdpaAttention_forward,
            new_LlamaAttention_forward,
        )

        from llava import LlavaLlamaForCausalLM

        # CD Algorithm
        GenerationMixin.greedy_search = new_greedy_search
        GenerationMixin.sample = new_sample
        GenerationMixin.beam_search = new_beam_search
        # Pre-processing
        LlavaLlamaForCausalLM.generate = new_LlavaLlamaForCausalLM_generate
        # Attention Scaling
        LlamaSdpaAttention.forward = new_LlamaSdpaAttention_forward
        LlamaAttention.forward = new_LlamaAttention_forward
        # No substantial change, only adds and passes CD args
        LlavaLlamaForCausalLM.forward = new_LlavaLlamaForCausalLM_forward
        LlamaForCausalLM.forward = new_LlamaForCausalLM_forward
        LlamaModel.forward = new_LlamaModel_forward

        super().__init__("1.5", size)

    def new_eval_model_pretrained(
        self, args, disable_conv_mode_warning=False, **kwargs
    ):
        # Copied and modified from LLaVA: llava/eval/run_llava.py

        import torch

        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IMAGE_PLACEHOLDER,
        )
        from llava.conversation import conv_templates
        from llava.utils import disable_torch_init
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
        )

        import re

        # Model
        disable_torch_init()

        qs = args.query
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        if args.image_file is not None:
            if IMAGE_PLACEHOLDER in qs:
                if self.model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if self.model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if (
            conv_mode == "llava_v0"
            and args.conv_mode is None
            and not disable_conv_mode_warning
        ):
            print_warning(
                "The auto inferred conversation mode 'llava_v0' is currently being used for the LLaVA model. However, this is uncommon. This warning may appear because your model name does not match certain expected keywords. Using the incorrect conversation mode could result in performance decrease. Therefore, it is recommended to do a double-check. To disable this warning, you can pass `disable_conv_mode_warning=True` to this function."
            )

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            if not disable_conv_mode_warning:
                print_warning(
                    "The auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}. To disable this warning, you can pass `disable_conv_mode_warning=True` to this function.".format(
                        conv_mode, args.conv_mode, args.conv_mode
                    )
                )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if args.image_file is None:
            images_tensor = None
            image_sizes = None
        else:
            image_files = self.image_parser(args)
            images = self.load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images, self.image_processor, self.model.config
            ).to(self.model.device, dtype=torch.float16)

        # HalTrapper: Modification Here
        if args.append is not None:
            input_ids_origin = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            GRABER["input_ids_offset"] = len(input_ids_origin[0])

            prompt += " " + args.append
        # HalTrapper: Modification ends

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        # HalTrapper: Modification here
        GRABER["input_ids"] = input_ids
        if args.append is None:
            GRABER["input_ids_offset"] = len(input_ids[0])

        method = args.method

        if (
            method == "haltrapper"
            and len(args.hallu_objs) != 0
            and args.image_file is not None
        ):
            # tqdm.write(str(args.hallu_objs))
            hallu_objs = list(args.hallu_objs.keys())
            hallu_scores = list(args.hallu_objs.values())
            if args.repeat_mode == "continuous":
                hallu_objs = [x for x in hallu_objs for _ in range(args.repeat)]
                hallu_scores = [x for x in hallu_scores for _ in range(args.repeat)]
            elif args.repeat_mode == "cross":
                hallu_objs = hallu_objs * args.repeat
                hallu_scores = hallu_scores * args.repeat
            else:
                raise ValueError(f"repeat_mode must in 'continuous' or 'cross'.")

            context_ids = []
            context_scaling = []
            for obj, score in zip(hallu_objs, hallu_scores):
                cur_context_ids = torch.tensor(
                    self.tokenizer.encode(obj), dtype=input_ids.dtype
                )[1:]
                context_ids.append(cur_context_ids)
                context_scaling.append(torch.ones_like(cur_context_ids) * score)

            context_ids = (
                torch.cat(context_ids, dim=0).to(input_ids.dtype).to(input_ids.device)
            )
            context_scaling = (
                torch.cat(context_scaling, dim=0).to(torch.float16).to(input_ids.device)
            )

            input_ids_cd = input_ids.clone()
            input_scaling_cd = torch.zeros_like(input_ids, dtype=torch.float16)

            image_token_at = torch.where(input_ids_cd == IMAGE_TOKEN_INDEX)[1]

            # FIXME: Here assumes batch size == 0
            input_ids_cd = torch.cat(
                (
                    input_ids_cd[0][: image_token_at + 1],
                    context_ids,
                    input_ids_cd[0][image_token_at + 1 :],
                ),
                dim=0,
            ).unsqueeze(0)
            input_scaling_cd = torch.cat(
                (
                    input_scaling_cd[0][: image_token_at + 1],
                    context_scaling,
                    input_scaling_cd[0][image_token_at + 1 :],
                ),
                dim=0,
            ).unsqueeze(0)

            if torch.all(input_scaling_cd.eq(0)):
                input_scaling_cd = None  # To save memory

            input_scaling = None
            images_tensor_cd = None

        elif method == "vcd" and args.image_file is not None:
            from methods_utils.vcd_add_noise import add_diffusion_noise

            images_tensor_cd = add_diffusion_noise(images_tensor, args.noise_step)

            input_ids_cd = None
            input_scaling = None
            input_scaling_cd = None

        elif method == "icd" and args.image_file is not None:
            qs_cd = (
                "You are a confused objects detector to provide a fuzzy overview or impression of the image. "
                + args.query
            )
            if args.image_file is not None:
                if IMAGE_PLACEHOLDER in qs:
                    if self.model.config.mm_use_im_start_end:
                        qs_cd = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs_cd)
                    else:
                        qs_cd = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs_cd)
                else:
                    if self.model.config.mm_use_im_start_end:
                        qs_cd = image_token_se + "\n" + qs_cd
                    else:
                        qs_cd = DEFAULT_IMAGE_TOKEN + "\n" + qs_cd

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                pass
            else:
                args.conv_mode = conv_mode

            conv_cd = conv_templates[args.conv_mode].copy()
            conv_cd.append_message(conv.roles[0], qs_cd)
            conv_cd.append_message(conv.roles[1], None)
            prompt_cd = conv_cd.get_prompt()

            input_ids_cd = (
                tokenizer_image_token(
                    prompt_cd, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            images_tensor_cd = None
            input_scaling = None
            input_scaling_cd = None

        elif method == "pai" and args.image_file is not None:
            from transformers.generation.logits_process import LogitsProcessorList
            from methods_utils.pai_cfg import init_cfg_processor

            input_scaling = torch.zeros_like(input_ids, dtype=torch.float16)

            image_token_at = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]

            # FIXME: Here assumes batch size == 0
            input_scaling[0][image_token_at] = args.pai_alpha

            kwargs["logits_processor"] = LogitsProcessorList(
                [
                    init_cfg_processor(
                        self.tokenizer, self.model, [prompt], 1.1, 1, 2, 32
                    )
                ]
            )  # FIXME: Hardcode Here

            input_scaling_cd = None
            input_ids_cd = None
            images_tensor_cd = None

        elif method == "code" and args.image_file is not None:
            caption = args.caption
            qs_cd = args.query
            if args.image_file is not None:
                if IMAGE_PLACEHOLDER in qs:
                    if self.model.config.mm_use_im_start_end:
                        qs_cd = re.sub(IMAGE_PLACEHOLDER, caption, qs_cd)
                    else:
                        qs_cd = re.sub(IMAGE_PLACEHOLDER, caption, qs_cd)
                else:
                    if self.model.config.mm_use_im_start_end:
                        qs_cd = caption + "\n" + qs_cd
                    else:
                        qs_cd = caption + "\n" + qs_cd

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                pass
            else:
                args.conv_mode = conv_mode

            conv_cd = conv_templates[args.conv_mode].copy()
            conv_cd.append_message(conv.roles[0], qs_cd)
            conv_cd.append_message(conv.roles[1], None)
            prompt_cd = conv_cd.get_prompt()

            input_ids_cd = (
                tokenizer_image_token(
                    prompt_cd, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            images_tensor_cd = None
            input_scaling = None
            input_scaling_cd = None

        else:
            input_ids_cd = None
            images_tensor_cd = None
            input_scaling = None
            input_scaling_cd = None
        # HalTrapper: Modification ends

        GRABER["grid_hs"] = 24
        GRABER["grid_ws"] = 24

        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                input_ids_cd=input_ids_cd,
                images=images_tensor,
                images_cd=images_tensor_cd,
                image_sizes=image_sizes,
                input_scaling=input_scaling,
                input_scaling_cd=input_scaling_cd,
                cd_type=args.cd_type,
                **kwargs,
            )

        if (
            "return_dict_in_generate" in kwargs.keys()
            and kwargs["return_dict_in_generate"]
        ):
            output_ids = output["sequences"]  # type:ignore
            response = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()
        else:
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[
                0
            ].strip()
            output = None

        return response, output

    def submit(
        self,
        prompt,
        image=None,
        question_id=None,
        method=None,
        cd_type=None,
        noise_step=None,
        repeat=1,
        repeat_mode="continuous",
        pai_alpha=None,
        append=None,
        sep=" ",
        candidates_number=None,
        ee_threshold=None,
        ig_threshold=None,
        ig_strategy="cos_sim",
        **kwargs,
    ):
        hallu_objs, caption = self.preprocess_method(
            image,
            method,
            candidates_number,
            ee_threshold,
            ig_threshold,
            ig_strategy,
            question_id,
        )

        if sep != " ":
            raise NotImplementedError()
        if method is None:
            method = "baseline"
        if hallu_objs is None and method == "haltrapper":
            hallu_objs = []
        args = type(
            "Args",
            (),
            {
                "model_path": self.model_path,
                "model_base": None,
                "model_name": self.get_model_name_from_path(self.model_path),
                "query": prompt,
                "conv_mode": None,
                "image_file": image,
                "sep": ",",
                "noise_step": noise_step,
                "hallu_objs": hallu_objs,
                "repeat": repeat,
                "repeat_mode": repeat_mode,
                "pai_alpha": pai_alpha,
                "method": method,
                "caption": caption,
                "cd_type": cd_type,
                "append": append,
            },
        )()
        response, output = self.new_eval_model_pretrained(args, **kwargs)

        model_logs = {}
        if hallu_objs:
            model_logs["candidates"] = hallu_objs
        if caption:
            model_logs["caption"] = caption

        return response, output, model_logs if model_logs else None


class JanusProModified(JanusPro, CTMixin):
    def __init__(self) -> None:
        from methods_utils.new_modeling_llama_4_48_3 import new_LlamaForCausalLM_forward
        from methods_utils.search_methods_4_48_3 import new_sample, new_beam_search
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        from transformers.generation.utils import GenerationMixin

        # CD algorithm
        GenerationMixin._sample = new_sample  # type: ignore
        GenerationMixin._beam_search = new_beam_search  # type: ignore

        LlamaForCausalLM.forward = new_LlamaForCausalLM_forward

        super().__init__()

    def submit(
        self,
        prompt,
        image=None,
        question_id=None,
        method=None,
        cd_type=None,
        noise_step=None,
        repeat=1,
        repeat_mode="continuous",
        pai_alpha=None,
        append=None,
        sep=" ",
        candidates_number=None,
        ee_threshold=None,
        ig_threshold=None,
        ig_strategy="cos_sim",
        **kwargs,
    ):
        # 1. preprocess
        hallu_objs, caption = self.preprocess_method(
            image,
            method,
            candidates_number,
            ee_threshold,
            ig_threshold,
            ig_strategy,
            question_id,
        )

        if method is None:
            method = "baseline"
        if hallu_objs is None and method == "haltrapper":
            hallu_objs = {}

        # 2. Get conversation

        if image is None:
            method = "baseline"

        use_vcd = False
        if (
            method == "haltrapper"
            and hallu_objs is not None
            and len(hallu_objs) != 0
            and image is not None
        ):
            # tqdm.write(str(hallu_objs))
            hallu_scores = list(hallu_objs.values())
            hallu_objs = list(hallu_objs.keys())
            if repeat_mode == "continuous":
                hallu_objs = [x for x in hallu_objs for _ in range(repeat)]
                hallu_scores = [x for x in hallu_scores for _ in range(repeat)]
            elif repeat_mode == "cross":
                hallu_objs = hallu_objs * repeat
                hallu_scores = hallu_scores * repeat
            else:
                raise ValueError(f"repeat_mode must in 'continuous' or 'cross'.")
            for s in hallu_scores:
                assert s == 0  # else not implemented
            conversation_cd = [
                {
                    "role": "User",
                    "content": f"<image_placeholder> {sep.join(hallu_objs)}\n{prompt}",
                    "images": [image],
                },
                {"role": "Assistant", "content": ""},
            ]

        elif method == "vcd" and image is not None:
            use_vcd = True
            conversation_cd = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image],
                },
                {"role": "Assistant", "content": "" if append is None else append},
            ]
        elif method == "icd" and image is not None:
            conversation_cd = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>\nYou are a confused objects detector to provide a fuzzy overview or impression of the image. {prompt}",
                    "images": [image],
                },
                {"role": "Assistant", "content": ""},
            ]

        elif method == "pai" and image is not None:
            raise NotImplementedError()
            conversation_cd = None
        elif method == "code" and image is not None:
            assert caption is not None
            caption = caption.strip()
            conversation_cd = [
                {
                    "role": "User",
                    "content": f"{caption}\n{prompt}",
                },
                {"role": "Assistant", "content": ""},
            ]
        else:
            conversation_cd = None

        # TODO: PAI
        input_scaling = None
        input_scaling_cd = None

        import torch
        from janus.utils.io import load_pil_images

        if image is None:
            conversation = [
                {
                    "role": "User",
                    "content": prompt,
                },
                {"role": "Assistant", "content": "" if append is None else append},
            ]
        else:
            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image],
                },
                {"role": "Assistant", "content": "" if append is None else append},
            ]

        # 3. submit

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.model.device)

        if append is not None:
            prepare_inputs["input_ids"] = prepare_inputs["input_ids"][:, :-1]
            prepare_inputs["images_seq_mask"] = prepare_inputs["images_seq_mask"][
                :, :-1
            ]

        GRABER["input_ids"] = prepare_inputs["input_ids"]
        if image is None:
            GRABER["image_start_pos"] = None
            GRABER["image_end_pos"] = None
        else:
            GRABER["image_start_pos"] = (
                int(torch.where(prepare_inputs["input_ids"][0] == 100016)[0]) + 1
            )
            GRABER["image_end_pos"] = int(
                torch.where(prepare_inputs["input_ids"][0] == 100593)[0]
            )
            assert GRABER["image_end_pos"] - GRABER["image_start_pos"] == 576
        if append is None:
            GRABER["input_ids_offset"] = len(prepare_inputs["input_ids"][0])
        else:
            if image is None:
                conversation_origin = [
                    {
                        "role": "User",
                        "content": prompt,
                    },
                    {"role": "Assistant", "content": ""},
                ]
            else:
                conversation_origin = [
                    {
                        "role": "User",
                        "content": f"<image_placeholder>\n{prompt}",
                        "images": [image],
                    },
                    {"role": "Assistant", "content": ""},
                ]
            pil_images_origin = load_pil_images(conversation)
            prepare_inputs_origin = self.processor(
                conversations=conversation_origin,
                images=pil_images_origin,
                force_batchify=True,
            ).to(self.model.device)
            GRABER["input_ids_offset"] = len(prepare_inputs_origin["input_ids"][0])

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        if conversation_cd is not None:
            pil_images_cd = load_pil_images(conversation_cd)
            prepare_inputs_cd = self.processor(
                conversations=conversation_cd, images=pil_images_cd, force_batchify=True
            ).to(self.model.device)

            # run image encoder to get the image embeddings
            inputs_embeds_cd = self.model.prepare_inputs_embeds(**prepare_inputs_cd)
            use_cd = True
        else:
            inputs_embeds_cd = None
            use_cd = False

        if use_vcd:
            assert inputs_embeds_cd is not None
            image_start_pos = GRABER["image_start_pos"]
            image_end_pos = GRABER["image_end_pos"]
            inputs_embeds_cd[:, image_start_pos:image_end_pos, :] = add_diffusion_noise(
                inputs_embeds_cd[:, image_start_pos:image_end_pos, :],
                noise_step=noise_step,
            )

        GRABER["grid_hs"] = 24
        GRABER["grid_ws"] = 24

        # run the model to get the response
        output = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            inputs_embeds_cd=inputs_embeds_cd,
            input_scaling=input_scaling,
            input_scaling_cd=input_scaling_cd,
            cd_type=cd_type,
            use_cd=use_cd,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **kwargs,
        )

        if not isinstance(output, torch.Tensor):
            output_ids = output["sequences"]
            generated_ids = output_ids
        else:
            generated_ids = output
            output = None

        response: str = self.tokenizer.decode(
            generated_ids[0].cpu().tolist(), skip_special_tokens=True
        )
        response = response.lstrip()

        # 4. add logs and return
        model_logs = {}
        if hallu_objs:
            model_logs["candidates"] = hallu_objs
        if caption:
            model_logs["caption"] = caption

        return response, output, model_logs if model_logs else None


class Qwen2VLModified(Qwen2VL, CTMixin):
    def __init__(self) -> None:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLForConditionalGeneration,
        )
        from transformers.generation.utils import GenerationMixin

        from methods_utils.new_modeling_qwen2_vl_4_45_0_dev0 import (
            new_Qwen2VLForConditionalGeneration_forward,
        )
        from methods_utils.search_methods_4_45_0_dev0 import new_sample, new_beam_search

        import qwen_vl_utils

        qwen_vl_utils.vision_process.MAX_PIXELS = 409_920

        Qwen2VLForConditionalGeneration.forward = (
            new_Qwen2VLForConditionalGeneration_forward
        )
        GenerationMixin._sample = new_sample  # type: ignore
        GenerationMixin._beam_search = new_beam_search  # type: ignore

        super().__init__()

    def submit(
        self,
        prompt,
        image=None,
        question_id=None,
        method=None,
        cd_type=None,
        noise_step=None,
        repeat=1,
        repeat_mode="continuous",
        pai_alpha=None,
        append=None,
        sep=" ",
        candidates_number=None,
        ee_threshold=None,
        ig_threshold=None,
        ig_strategy="cos_sim",
        **kwargs,
    ):
        from qwen_vl_utils import process_vision_info

        # 0. get image or video kwargs.
        if image is not None and isinstance(image, str):
            if image.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv")):
                content_main = {
                    "type": "video",
                    "video": image,
                    "max_pixels": 360 * 420,
                    "fps": 1 / 3,
                }
            else:
                content_main = {
                    "type": "image",
                    "image": image,
                }
        else:
            content_main = {}

        # 1. preprocess
        hallu_objs, caption = self.preprocess_method(
            image,
            method,
            candidates_number,
            ee_threshold,
            ig_threshold,
            ig_strategy,
            question_id,
        )

        if method is None:
            method = "baseline"
        if hallu_objs is None and method == "haltrapper":
            hallu_objs = {}

        # 2. Get message

        if image is None:
            method = "baseline"

        use_vcd = False
        use_ccd = False
        if (
            method == "haltrapper"
            and hallu_objs is not None
            and len(hallu_objs) != 0
            and image is not None
        ):
            messages_cd = [
                {
                    "role": "user",
                    "content": [
                        content_main,
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            use_ccd = True
        elif method == "vcd" and image is not None:
            messages_cd = [
                {
                    "role": "user",
                    "content": [
                        content_main,
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            use_vcd = True
        elif method == "icd" and image is not None:
            messages_cd = [
                {
                    "role": "user",
                    "content": [
                        content_main,
                        {
                            "type": "text",
                            "text": f"You are a confused objects detector to provide a fuzzy overview or impression of the image. {prompt}",
                        },
                    ],
                }
            ]
        elif method == "pai" and image is not None:
            raise NotImplementedError()
        elif method == "code" and image is not None:
            messages_cd = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{caption} {prompt}"},
                    ],
                }
            ]
        else:
            messages_cd = None

        if image is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        content_main,
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if append:
            text = text + f"{append}"
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        _input_ids = inputs["input_ids"]
        GRABER["input_ids"] = _input_ids

        if image is None:
            GRABER["image_start_pos"] = None
            GRABER["image_end_pos"] = None
        else:
            image_start_pos = torch.where(_input_ids[0] == 151652)[0] + 1
            image_end_pos = torch.where(_input_ids[0] == 151653)[0]
            GRABER["image_start_pos"] = image_start_pos
            GRABER["image_end_pos"] = image_end_pos
            if "image_grid_thw" in inputs:
                GRABER["grid_frames"] = 1
                GRABER["grid_hs"] = inputs["image_grid_thw"][0][1] // 2
                GRABER["grid_ws"] = inputs["image_grid_thw"][0][2] // 2
            elif "video_grid_thw" in inputs:
                GRABER["grid_frames"] = inputs["video_grid_thw"][0][0]
                GRABER["grid_hs"] = inputs["video_grid_thw"][0][1] // 2
                GRABER["grid_ws"] = inputs["video_grid_thw"][0][2] // 2
            else:
                GRABER["grid_frames"] = None
                GRABER["grid_hs"] = None
                GRABER["grid_ws"] = None

        if append is None:
            GRABER["input_ids_offset"] = len(_input_ids[0])
        else:
            if image is None:
                messages_origin = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            else:
                messages_origin = [
                    {
                        "role": "user",
                        "content": [
                            content_main,
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            text_origin = self.processor.apply_chat_template(
                messages_origin, tokenize=False, add_generation_prompt=True
            )
            image_inputs_origin, video_inputs_origin = process_vision_info(
                messages_origin
            )
            inputs_origin = self.processor(
                text=[text_origin],
                images=image_inputs_origin,
                videos=video_inputs_origin,
                padding=True,
                return_tensors="pt",
            )

            inputs_origin = inputs_origin.to("cuda")
            _input_ids_origin = inputs_origin["input_ids"]
            GRABER["input_ids_offset"] = len(_input_ids_origin[0])

        if messages_cd is not None:
            # Preparation for inference
            text_cd: str = self.processor.apply_chat_template(
                messages_cd, tokenize=False, add_generation_prompt=True
            )
            if use_ccd:
                assert hallu_objs
                # tqdm.write(str(hallu_objs))
                hallu_scores = list(hallu_objs.values())
                hallu_objs = list(hallu_objs.keys())
                if repeat_mode == "continuous":
                    hallu_objs = [x for x in hallu_objs for _ in range(repeat)]
                    hallu_scores = [x for x in hallu_scores for _ in range(repeat)]
                elif repeat_mode == "cross":
                    hallu_objs = hallu_objs * repeat
                    hallu_scores = hallu_scores * repeat
                else:
                    raise ValueError(f"repeat_mode must in 'continuous' or 'cross'.")
                for s in hallu_scores:
                    assert s == 0  # else not implemented
                if "<|image_pad|>" in text_cd:
                    text_cd = text_cd.replace(
                        "<|image_pad|>", f"<|image_pad|> {sep.join(hallu_objs)}"
                    )
                elif "<|video_pad|>" in text_cd:
                    text_cd = text_cd.replace(
                        "<|video_pad|>", f"<|video_pad|> {sep.join(hallu_objs)}"
                    )
            image_inputs_cd, video_inputs_cd = process_vision_info(messages_cd)
            inputs_cd = self.processor(
                text=[text_cd],
                images=image_inputs_cd,
                videos=video_inputs_cd,
                padding=True,
                return_tensors="pt",
            )

            inputs_cd = inputs_cd.to("cuda")

            if use_vcd:
                inputs_cd["pixel_values"] = add_diffusion_noise(
                    inputs_cd["pixel_values"], noise_step=noise_step
                )

            for k, v in inputs_cd.items():
                inputs[f"{k}_cd"] = v

        # Inference: Generation of the output

        if messages_cd is not None:
            use_cd = True
        else:
            use_cd = False

        output = self.model.generate(**inputs, **kwargs, use_cd=use_cd, cd_type=cd_type)

        if (
            "return_dict_in_generate" in kwargs.keys()
            and kwargs["return_dict_in_generate"]
        ):
            output_ids = output["sequences"]  # type: ignore
            generated_ids = output_ids
        else:
            generated_ids = output
            output = None

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        # 4. add logs and return
        model_logs = {}
        if hallu_objs:
            model_logs["candidates"] = hallu_objs
        if caption:
            model_logs["caption"] = caption

        return response, output, model_logs if model_logs else None


class MiniGPT4Modified(LM, CTMixin):
    # We test MiniGPT with backbone on HalTrapper
    def __init__(self, backbone="llama") -> None:
        assert backbone == "llama"

        minigpt4_root = get_path_from_table("MiniGPT4 repo root")
        sys.path.append(os.fspath(minigpt4_root))

        from transformers.generation.utils import (
            GenerationMixin,
        )
        from methods_utils.search_methods_4_30_0 import (
            new_greedy_search,
            new_sample,
            new_beam_search,
        )
        from methods_utils.new_modeling_llama_minigpt4 import (
            new_LlamaForCausalLM_forward,
        )
        from minigpt4.models.modeling_llama import LlamaForCausalLM  # type: ignore

        # CD Algorithm
        GenerationMixin.greedy_search = new_greedy_search
        GenerationMixin.sample = new_sample
        GenerationMixin.beam_search = new_beam_search
        # No substantial change, only adds and passes CD args
        LlamaForCausalLM.forward = new_LlamaForCausalLM_forward

        print_note(
            "The loading implimentation of MiniGPT4 model here is really slow, please wait in patience..."
        )

        name = "minigpt4-llama2-7b"

        super().__init__(name)

        from methods_utils.conversation_minigpt4_cd import (
            Chat,
            CONV_VISION_Vicuna0,
            CONV_VISION_LLama2,
            StoppingCriteriaSub,
        )
        from minigpt4.common.registry import registry  # type: ignore
        from minigpt4.common.config import Config  # type: ignore
        from transformers import StoppingCriteriaList
        from types import SimpleNamespace
        import torch

        conv_dict = {
            "pretrain_vicuna0": CONV_VISION_Vicuna0,
            "pretrain_llama2": CONV_VISION_LLama2,
        }

        tqdm.write("Initializing Chat")

        args = SimpleNamespace(
            cfg_path=os.fspath(
                minigpt4_root / "eval_configs/minigpt4_llama2_eval.yaml"
            ),
            gpu_id=0,
            options=[],
        )

        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))

        self.CONV_VISION = conv_dict[model_config.model_type]

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)

        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [
            torch.tensor(ids).to(device="cuda:{}".format(args.gpu_id))
            for ids in stop_words_ids
        ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

        self.chat = Chat(
            model,
            vis_processor,
            device="cuda:{}".format(args.gpu_id),
            stopping_criteria=stopping_criteria,
        )

        self.tokenizer = model.llama_tokenizer

        def update_input_ids_hook(input_ids):
            GRABER["input_ids"] = input_ids
            GRABER["input_ids_offset"] = 59

        self.update_input_ids_hook = update_input_ids_hook

        tqdm.write("Initialization Finished")

    def submit(
        self,
        prompt,
        image=None,
        question_id=None,
        method=None,
        cd_type=None,
        noise_step=None,
        repeat=1,
        repeat_mode="continuous",
        pai_alpha=None,
        append=None,
        sep=" ",
        candidates_number=None,
        ee_threshold=None,
        ig_threshold=None,
        ig_strategy="cos_sim",
        **kwargs,
    ):
        # 1. preprocess
        hallu_objs, caption = self.preprocess_method(
            image,
            method,
            candidates_number,
            ee_threshold,
            ig_threshold,
            ig_strategy,
            question_id,
        )

        if method is None:
            method = "baseline"
        if method == "haltrapper" and hallu_objs is not None and len(hallu_objs) == 0:
            method = "baseline"
        if hallu_objs is None and method == "haltrapper":
            hallu_objs = []

        chat_state = self.CONV_VISION.copy()
        chat_state.messages = []  # This resets the history of chat

        img_list = []

        if image is not None:
            self.chat.upload_img(image, chat_state, img_list)
        self.chat.encode_img(img_list)
        self.chat.ask(prompt, chat_state)

        GRABER["image_start_pos"] = 41
        GRABER["image_end_pos"] = 105  # Hardcode Here!

        response, output = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            method=method,
            cd_type=cd_type,
            noise_step=noise_step,
            hallu_objs=hallu_objs,
            repeat=repeat,
            repeat_mode=repeat_mode,
            pai_alpha=pai_alpha,
            caption=caption,
            append=append,
            update_input_ids_hook=self.update_input_ids_hook,
            sep=sep,
            **kwargs,
        )

        return response, output, None


class QwenVLModified(QwenVL, CTMixin):
    def __init__(self) -> None:
        from transformers.generation.utils import (
            GenerationMixin,
        )
        from methods_utils.search_methods_4_32_0 import (
            new_greedy_search,
            new_sample,
            new_beam_search,
        )

        # CD Algorithm
        GenerationMixin.greedy_search = new_greedy_search
        GenerationMixin.sample = new_sample
        GenerationMixin.beam_search = new_beam_search

        super().__init__()

        from methods_utils.new_modeling_qwen import (
            new_chat,
            new_QWenLMHeadModel_forward,
            new_QWenModel_forward,
        )

        def update_input_ids_hook(input_ids):
            GRABER["input_ids"] = input_ids
            GRABER["input_ids_offset"] = 292

        self.update_input_ids_hook = update_input_ids_hook

        self.model.chat = types.MethodType(new_chat, self.model)
        self.model.forward = types.MethodType(new_QWenLMHeadModel_forward, self.model)
        self.model.transformer.forward = types.MethodType(
            new_QWenModel_forward, self.model.transformer
        )

    def submit(
        self,
        prompt,
        image=None,
        question_id=None,
        method=None,
        cd_type=None,
        noise_step=None,
        repeat=1,
        repeat_mode="continuous",
        pai_alpha=None,
        append=None,
        sep=" ",
        candidates_number=None,
        ee_threshold=None,
        ig_threshold=None,
        ig_strategy="cos_sim",
        **kwargs,
    ):
        # 1. preprocess
        hallu_objs, caption = self.preprocess_method(
            image,
            method,
            candidates_number,
            ee_threshold,
            ig_threshold,
            ig_strategy,
            question_id,
        )

        if image is None:
            query = prompt
        else:
            query = self.tokenizer.from_list_format(
                [
                    # Either a local path or an url
                    {"image": image},
                    {"text": prompt},
                ]
            )

        GRABER["image_start_pos"] = 19
        GRABER["image_end_pos"] = 275

        response, output = self.model.chat(
            self.tokenizer,
            query=query,
            history=None,
            method=method,
            cd_type=cd_type,
            noise_step=noise_step,
            hallu_objs=hallu_objs,
            repeat=repeat,
            repeat_mode=repeat_mode,
            pai_alpha=pai_alpha,
            caption=caption,
            append=append,
            update_input_ids_hook=self.update_input_ids_hook,
            sep=sep,
            **kwargs,
        )
        return response, output, None
