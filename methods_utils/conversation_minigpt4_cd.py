# Derived and modified from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/conversation/conversation.py

import argparse
import time
from threading import Thread
from PIL import Image
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id,
        )

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(input_ids[:, -len(stop) :] == stop).item():
                return True

        return False


CONV_VISION_Vicuna0 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
    "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_LLama2 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
    "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("<s>[INST] ", " [/INST] "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

CONV_VISION_minigptv2 = Conversation(
    system="",
    roles=("<s>[INST] ", " [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)


def safe_get_context_emb(self, prompt, img_list, device):
    prompt_segs = prompt.split("<ImageHere>")
    assert (
        len(prompt_segs) == len(img_list) + 1
    ), "Unmatched numbers of image placeholders and images."
    seg_tokens = [
        self.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0)
        .to(device)
        .input_ids  # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [
        seg_embs[-1]
    ]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def get_input_ids(self, prompt, img_list, device):
    prompt_segs = prompt.split("<ImageHere>")
    assert (
        len(prompt_segs) == len(img_list) + 1
    ), "Unmatched numbers of image placeholders and images."
    seg_tokens = [
        self.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0)
        .to(device)
        .input_ids  # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    # FIXME: -200: Hardcode Here
    image_tensor = torch.tensor([[-200]]).to(device).to(seg_tokens[0].dtype)
    input_ids = []
    for i in range(len(seg_tokens)):
        input_ids.append(seg_tokens[i])
        if i + 1 != len(seg_tokens):
            input_ids.append(image_tensor)
    input_ids = torch.cat(input_ids, dim=1)
    return input_ids


class Chat:
    def __init__(self, model, vis_processor, device="cuda:0", stopping_criteria=None):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)]
            )

    def ask(self, text, conv):
        if (
            len(conv.messages) > 0
            and conv.messages[-1][0] == conv.roles[0]
            and conv.messages[-1][1][-6:] == "</Img>"
        ):  # last message is image.
            conv.messages[-1][1] = " ".join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer_prepare(
        self,
        conv,
        img_list,
        min_length=1,
        repetition_penalty=1,
        length_penalty=1,
        max_length=2000,
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
    ):
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # HalTrapper: Modified Here
        if append is not None:
            prompt += " " + append
            input_ids = get_input_ids(self.model, prompt, img_list, self.device)
            assert update_input_ids_hook is not None
            update_input_ids_hook(input_ids)

        if method == "haltrapper":
            assert hallu_objs is not None
            assert all(x == 0 for x in hallu_objs.values())
            # print(hallu_objs)
            hallu_objs = list(hallu_objs.keys())
            if repeat_mode == "continuous":
                hallu_objs = [x for x in hallu_objs for _ in range(repeat)]
            elif repeat_mode == "cross":
                hallu_objs = hallu_objs * repeat
            elif repeat_mode == "random":
                hallu_objs = hallu_objs * repeat
                random.shuffle(hallu_objs)
            else:
                raise ValueError(
                    f"repeat_mode must in 'continuous', 'cross' or 'random'."
                )

            prompt_cd = prompt.replace(
                "<ImageHere>", f"<ImageHere> {sep.join(hallu_objs)}"
            )
            embs = self.model.get_context_emb(prompt, img_list)
            embs_cd = self.model.get_context_emb(prompt_cd, img_list)

        elif method == "vcd":
            from methods_utils.vcd_add_noise import add_diffusion_noise

            img_list_cd = []
            for img in img_list:
                img_list_cd.append(add_diffusion_noise(img, noise_step))
            embs = self.model.get_context_emb(prompt, img_list)
            embs_cd = self.model.get_context_emb(prompt, img_list_cd)

        elif method == "icd":
            assert "Please help me describe the image in detail." in prompt
            prompt_cd = prompt.replace(
                "Please help me describe the image in detail.",
                "You are a confused objects detector to provide a fuzzy overview or impression of the image. Please help me describe the image in detail.",
            )
            embs = self.model.get_context_emb(prompt, img_list)
            embs_cd = self.model.get_context_emb(prompt_cd, img_list)

        elif method == "pai":
            raise NotImplementedError()

        elif method == "code":
            prompt_cd = prompt.replace("<Img><ImageHere></Img>", caption)
            embs = self.model.get_context_emb(prompt, img_list)
            embs_cd = safe_get_context_emb(self.model, prompt_cd, [], self.device)

        else:
            embs = self.model.get_context_emb(prompt, img_list)
            embs_cd = None
        # HalTrapper: Modification Ends

        current_max_len = embs.shape[1] + kwargs["max_new_tokens"]
        if current_max_len - max_length > 0:
            print(
                "Warning: The number of tokens in current conversation exceeds the max length. "
                "The model will not see the contexts outside the range."
            )
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        # HalTrapper: Modified Here
        if embs_cd is not None:
            current_max_len_cd = embs_cd.shape[1] + kwargs["max_new_tokens"]
            if current_max_len_cd - max_length > 0:
                print(
                    "Warning: The number of tokens in current conversation exceeds the max length. "
                    "The model will not see the contexts outside the range."
                )
            begin_idx_cd = max(0, current_max_len_cd - max_length)
            embs_cd = embs_cd[:, begin_idx_cd:]
            use_cd = True
        else:
            use_cd = False
        # HalTrapper: Modification Ends

        generation_kwargs = dict(
            inputs_embeds=embs,
            inputs_embeds_cd=embs_cd,  # HalTrapper: New argument
            stopping_criteria=self.stopping_criteria,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            use_cd=use_cd,  # HalTrapper: New argument
            cd_type=cd_type,  # HalTrapper: New argument
            attention_mask_cd=None,  # HalTrapper: New argument, FIXME: Hardcode here
            position_ids_cd=None,  # HalTrapper: New argument, FIXME: Hardcode here
            **kwargs,
        )
        return generation_kwargs

    def answer(
        self,
        conv,
        img_list,
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
        **kargs,
    ):
        generation_dict = self.answer_prepare(
            conv,
            img_list,
            method=method,
            cd_type=cd_type,
            noise_step=noise_step,
            hallu_objs=hallu_objs,
            repeat=repeat,
            repeat_mode=repeat_mode,
            pai_alpha=pai_alpha,
            caption=caption,
            append=append,
            update_input_ids_hook=update_input_ids_hook,
            sep=sep,
            **kargs,
        )
        output = self.model_generate(**generation_dict)

        if (
            "return_dict_in_generate" in kargs.keys()
            and kargs["return_dict_in_generate"]
        ):
            output_token = output["sequences"][0]
        else:
            output_token = output[0]
            output = None

        output_text = self.model.llama_tokenizer.decode(
            output_token, skip_special_tokens=True
        )

        output_text = output_text.split("###")[0]  # remove the stop sign '###'
        output_text = output_text.split("Assistant:")[-1].strip()

        conv.messages[-1][1] = output_text
        return output_text, output

    def stream_answer(self, conv, img_list, **kargs):
        generation_kwargs = self.answer_prepare(conv, img_list, **kargs)
        streamer = TextIteratorStreamer(
            self.model.llama_tokenizer, skip_special_tokens=True
        )
        generation_kwargs["streamer"] = streamer
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def model_generate(self, *args, **kwargs):
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():
            output = self.model.llama_model.generate(*args, **kwargs)
        return output

    def encode_img(self, img_list):
        image = img_list[0]
        img_list.pop(0)
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert("RGB")
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)

    def upload_img(self, image, conv, img_list):
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        img_list.append(image)
        msg = "Received."

        return msg
