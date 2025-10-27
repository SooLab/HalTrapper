from datetime import datetime
from tqdm import tqdm
import random
import sys
import os

import json
from pathlib import Path
from ._utils._path import safe_open, safe_close
from ._utils._cuda import assert_cuda_set_device
from ._utils._colors import *
from .path_table import get_path_from_table

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Type, Any, IO, ClassVar, Union

if TYPE_CHECKING:
    from transformers.generation.utils import GenerateOutput
    from transformers import PreTrainedTokenizer, PreTrainedModel
    from ._utils._path import PathObj
    from .benchmarks import BenchBase


class LM(ABC):
    model: "PreTrainedModel"
    tokenizer: "PreTrainedTokenizer"

    registry: ClassVar[dict[str, Type["LM"]]] = {}

    def _open_log_file(self, path: "PathObj", mode="X") -> None:
        self._log_file = safe_open(path, mode)
        self._log_file_path = Path(self._log_file.name)

    def _close_log_file(self) -> None:
        try:
            if self._log_file is not None:
                safe_close(self._log_file)
                if self._log_file_path is not None:
                    fp = Path(self._log_file_path)
                    if fp.is_file() and fp.stat().st_size == 0:
                        fp.unlink()
        except:
            pass

        self._log_file = None
        self._log_file_path = None

    def __init__(self, name: str) -> None:
        assert_cuda_set_device()
        self.name = name
        self._log_file_path: Optional[Path] = None
        self._log_file: Optional[IO[Any]] = None
        time_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        self._open_log_file(f"./logs/{name}--{time_str}.jsonl")
        # NOTE: To allow mixins use attributes of LM, super().__init__() is better to set here at the end of this function!
        super().__init__()

    def __del__(self):
        self._close_log_file()

    def __init_subclass__(cls, cmd_name: Optional[str] = None) -> None:
        super().__init_subclass__()

        if cmd_name is None:
            cmd_name = cls.__name__

        cls.registry[cmd_name.lower()] = cls

    @property
    def log_file_path(self) -> Optional[Path]:
        return self._log_file_path

    @log_file_path.setter
    def log_file_path(self, path: Optional["PathObj"]) -> None:
        self.set_log_file_path(path)

    def set_log_file_path(self, path: Optional["PathObj"], mode="x") -> None:
        self._close_log_file()
        if path is not None:
            self._open_log_file(path, mode)

    def __call__(
        self,
        prompt: str,
        image: Optional["PathObj"] = None,
        user_log_dict: Optional[dict[Any, Any]] = None,
        use_log: bool = True,
        question_id: Optional[int] = None,
        **kwargs,
    ) -> tuple[str, Optional["GenerateOutput"], Optional[dict[Any, Any]]]:
        if image is not None:
            image = str(image)

        if user_log_dict is None:
            user_log_dict = {}

        try:
            if "do_sample" not in kwargs.keys():
                kwargs["do_sample"] = True if kwargs["temperature"] > 0 else False
            # if kwargs['do_sample'] == False and kwargs['temperature'] == 0.0:
            #     del kwargs['temperature']
            # if kwargs['do_sample'] == False and kwargs['top_k'] is None:
            #     del kwargs['top_k']
        except:
            pass

        # if 'use_cache' not in kwargs.keys():
        #     kwargs['use_cache'] = True

        response, output, model_log_dict = self.submit(
            prompt, image, question_id, **kwargs
        )

        if model_log_dict == None:
            model_log_dict = {}
        else:
            model_log_dict = {"logs": model_log_dict}

        if "streamer" in kwargs.keys():
            del kwargs["streamer"]

        if use_log:
            item = {
                **user_log_dict,
                "prompt": prompt,
                "image": image,
                "response": response,
                **model_log_dict,
                "model": self.name,
                "inference_time": datetime.now().astimezone().isoformat(),
                "params": {**kwargs},
            }
            if self._log_file is not None:
                self._log_file.seek(0, 2)
                self._log_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                print_warning(
                    "`use_log` is true for this generation, but the log file path is set to None."
                )
        else:
            item = None

        return response, output, item

    @abstractmethod
    def submit(
        self,
        prompt: str,
        image: Optional[str],
        question_id: Optional[int] = None,
        **kwargs,
    ) -> tuple[str, Optional["GenerateOutput"], Optional[dict[Any, Any]]]:
        ...

    def input(self, prompt) -> str:
        return input(prompt)

    def input_multiline(self, prompt, *, strip=True) -> str:
        print(prompt)
        print("""-- MULTILINE -- (Ctrl+D: submit)""")

        prompt = sys.stdin.read()
        print()

        if strip:
            prompt = prompt.strip()

        return prompt

    def interact(self, raise_error: bool = False, **kwargs) -> None:
        image_path = None
        while True:
            try:
                print_line()

                sys.stdin.flush()
                image_input = self.input("Enter image path, 'exit', or 'None': ")

                if image_input == "exit":
                    break
                elif image_input == "None":
                    image_path = None
                elif image_input == "":
                    pass
                else:
                    image_path = image_input

                prompt = self.input_multiline("Enter prompt:")

                print_with_label("Image", image_path)
                print_with_label("Prompt", prompt)

                # try:
                #     from transformers import TextStreamer
                #     streamer = TextStreamer(
                #         self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                #     kwargs['streamer'] = streamer
                # except:
                #     pass

                response, _, _ = self(prompt, image_path, **kwargs)
                print_with_label(self.name, response)

            except Exception as e:
                if raise_error:
                    raise e
                else:
                    print_error(f"Generation failed | {e.__class__.__name__}: {e}.")

    def eval(
        self,
        benchmark: "BenchBase",
        shuffle: bool = False,
        n_samples: Optional[Union[float, int]] = None,
        indices: Optional[list[int]] = None,
        **kwargs,
    ) -> None:
        if indices is None:
            indices = list(range(len(benchmark)))

            if n_samples is not None:
                if n_samples <= 0:
                    raise ValueError(
                        f"`n_samples` must be a positive integer or a float in (0, 1]. Got {n_samples}."
                    )
                if isinstance(n_samples, float) and n_samples <= 1:
                    n_samples = len(benchmark) * n_samples

                n_samples = round(n_samples)
                indices = random.sample(indices, n_samples)
                indices.sort()
                print_note(
                    f"Evaluating a random sample of {n_samples} out of {len(benchmark)} total examples."
                )

            if shuffle:
                random.shuffle(indices)
                print_note("Shuffling the evaluation order.")

        else:
            print_note("Using specified indices when evaling.")

        log_list = []
        print_note(f"Saving evaluation result to {self.log_file_path}")

        for original_idx in tqdm(indices):
            prompt, image_path, user_log_dict = benchmark[original_idx]
            _, _, item = self(
                prompt,
                image_path,
                user_log_dict=user_log_dict,
                question_id=original_idx,
                **kwargs,
            )
            log_list.append(item)

        if self._log_file_path is not None:
            log_file_name = Path(self._log_file_path).stem
        else:
            time_str = datetime.now().strftime("%Y%m%dT%H%M%S")
            log_file_name = Path(f"{self.name}--{time_str}")

        log_file_path = f"./eval_responses/{log_file_name}.json"

        try:
            import torch
            import gc

            del self.model
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

        benchmark.get_score(log_list, log_file_path)


class LLaVA(LM):
    def __init__(self, version: str = "1.5", size: str = "7b") -> None:
        if version == "1.5":
            if size == "7b":
                name = "llava-v1.5-7b"
            elif size == "13b":
                name = "llava-v1.5-13b"
            else:
                raise
        elif version == "1.6":
            if size == "7b":
                name = "llava-v1.6-vicuna-7b"
            elif size == "13b":
                name = "llava-v1.6-vicuna-13b"
            elif size == "34b":
                name = "llava-v1.6-34b"
            else:
                raise
        else:
            raise
        super().__init__(name)

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        self.get_model_name_from_path = get_model_name_from_path

        self.model_path = os.fspath(get_path_from_table(name))

        self.model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_path, None, self.model_name
        )

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

    def image_parser(self, args):
        # Copied from LLaVA: llava/eval/run_llava.py
        out = args.image_file.split(args.sep)
        return out

    def load_image(self, image_file):
        # Copied from LLaVA: llava/eval/run_llava.py
        import requests
        from PIL import Image
        from io import BytesIO

        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def load_images(self, image_files):
        # Copied from LLaVA: llava/eval/run_llava.py
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)
            out.append(image)
        return out

    def new_eval_model_pretrained(
        self, args, disable_conv_mode_warning=False, **kwargs
    ):
        # Copied and modified from LLaVA: llava/eval/run_llava.py
        # Major changes:
        # 1. Support 0 image input;
        # 2. Support `return_dict_in_generate=True` for dev usage.

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

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output = self.model.generate(
                input_ids, images=images_tensor, image_sizes=image_sizes, **kwargs
            )

        if not isinstance(output, torch.Tensor):
            output_ids = output.sequences
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
        self, prompt, image=None, question_id=None, **kwargs
    ) -> tuple[str, Optional["GenerateOutput"], Optional[dict[Any, Any]]]:
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
            },
        )()
        response, output = self.new_eval_model_pretrained(args, **kwargs)
        return response, output, None


class QwenVL(LM):
    def __init__(self) -> None:
        from transformers.generation import GenerationConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = "Qwen-VL-Chat"

        super().__init__(name)

        model_path = os.fspath(get_path_from_table(name))

        # Note: The default behavior now has injection attack prevention off.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # use bf16
        # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, fp16=True).eval()
        # use cpu only
        # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True).eval()
        # use cuda device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda", trust_remote_code=True
        ).eval()

        # Specify hyperparameters for generation
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )

    def submit(self, prompt, image=None, question_id=None, **kwargs):
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
        response, _ = self.model.chat(
            self.tokenizer, query=query, history=None, **kwargs
        )
        return response, None, None  # TODO: Implement output


class Qwen2VL(LM):
    def __init__(self) -> None:
        name = "Qwen2-VL-7B-Instruct"  # FIXME

        super().__init__(name)

        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            os.fspath(get_path_from_table(name)),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )

        # default processer
        self.processor = AutoProcessor.from_pretrained(
            os.fspath(get_path_from_table(name)),
        )
        self.tokenizer = self.processor.tokenizer

    def submit(
        self, prompt, image=None, question_id=None, **kwargs
    ) -> tuple[str, Optional["GenerateOutput"], Optional[dict[Any, Any]]]:
        from qwen_vl_utils import process_vision_info

        if image is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        elif isinstance(image, str):
            if image.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv")):
                video_kwargs = {}
                for key in ["max_pixels", "min_pixels", "fps", "total_pixels"]:
                    if key in kwargs:
                        video_kwargs[key] = kwargs.pop(key)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": image, **video_kwargs},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

        else:
            raise ValueError("`image` must be a string or None.")

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output

        output = self.model.generate(**inputs, **kwargs)

        # TODO: Judge by isinstance(Tensor)
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

        return response, output, None  # type: ignore


class MiniGPT4(LM):
    def __init__(self, backbone: str = "vicuna0") -> None:
        # print_note(
        #     "The loading implimentation of MiniGPT4 model here is really slow, please wait in patience...")

        if "vicuna" in backbone:
            name = "minigpt4-vicuna0-7b"
            llama = False
        elif "llama" in backbone:
            name = "minigpt4-llama2-7b"
            llama = True
        else:
            raise ValueError(
                f"Backbone for MiniGPT-4 model should be either 'vicuna0' or 'llama2'."
            )

        super().__init__(name)

        minigpt4_root = get_path_from_table("MiniGPT4 repo root")
        sys.path.append(os.fspath(minigpt4_root))

        from .conversation_minigpt4 import (
            Chat,
            CONV_VISION_Vicuna0,
            CONV_VISION_LLama2,
            StoppingCriteriaSub,
        )
        from minigpt4.common.registry import registry
        from minigpt4.common.config import Config
        from transformers import StoppingCriteriaList
        from types import SimpleNamespace
        import torch

        conv_dict = {
            "pretrain_vicuna0": CONV_VISION_Vicuna0,
            "pretrain_llama2": CONV_VISION_LLama2,
        }

        print("Initializing Chat")

        if llama:
            cfg_path = os.fspath(
                minigpt4_root / "eval_configs/minigpt4_llama2_eval.yaml"
            )
        else:
            cfg_path = os.fspath(minigpt4_root / "eval_configs/minigpt4_eval.yaml")

        args = SimpleNamespace(cfg_path=cfg_path, gpu_id=0, options=[])

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
        print("Initialization Finished")

    def submit(self, prompt, image, question_id=None, **kwargs):
        chat_state = self.CONV_VISION.copy()
        chat_state.messages = []  # This resets the history of chat

        img_list = []

        if image is not None:
            self.chat.upload_img(image, chat_state, img_list)
            self.chat.encode_img(img_list)
        self.chat.ask(prompt, chat_state)

        response = self.chat.answer(conv=chat_state, img_list=img_list, **kwargs)[0]
        return response, None, None  # TODO: Implement output


class JanusPro(LM):
    def __init__(self) -> None:
        name = "Janus-Pro-7B"

        super().__init__(name)

        import torch
        from transformers import AutoModelForCausalLM
        from janus.models import MultiModalityCausalLM, VLChatProcessor

        self.model: MultiModalityCausalLM = (
            AutoModelForCausalLM.from_pretrained(
                os.fspath(get_path_from_table(name)),
                trust_remote_code=True,
            )
            .to(torch.bfloat16)
            .cuda()
            .eval()
        )

        # default processer
        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            os.fspath(get_path_from_table(name)),
        )
        self.tokenizer = self.processor.tokenizer

    def submit(
        self, prompt, image=None, question_id=None, **kwargs
    ) -> tuple[str, Optional["GenerateOutput"], Optional[dict[Any, Any]]]:
        import torch
        from janus.utils.io import load_pil_images

        if image is None:
            conversation = [
                {
                    "role": "User",
                    "content": prompt,
                },
                {"role": "Assistant", "content": ""},
            ]
        else:
            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image],
                },
                {"role": "Assistant", "content": ""},
            ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        output = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
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

        return response, output, None
