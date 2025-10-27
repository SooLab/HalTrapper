# Derived and modified from Tongyi Qianwen Materials

from methods_utils.qwen_vl_chat.qwen_generation_utils import *  # type:ignore
import re
from typing import Optional


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
    # HalTrapper: New args
    method: str = "baseline",
    hallu_objs: Optional[List[str]] = None,
    repeat: int = 1,
    repeat_mode: str = "continuous",
    caption: Optional[str] = None,
    append: Optional[str] = None,
    sep: str = " ",
):
    # HalTrapper: Modification Starts
    if method == "haltrapper":
        assert hallu_objs is not None
        assert all(x == 0 for x in hallu_objs.values())
        # print(hallu_objs)
        hallu_objs = list(hallu_objs.keys())
        if repeat_mode == "continuous":
            hallu_objs = [x for x in hallu_objs for _ in range(repeat)]
        elif repeat_mode == "cross":
            hallu_objs = hallu_objs * repeat
        else:
            raise ValueError(f"repeat_mode must in 'continuous' or 'cross'.")
        context_text = f" {sep.join(hallu_objs)}"
    elif method == "icd":
        query = query.replace(
            "</img>\n",
            "</img>\nYou are a confused objects detector to provide a fuzzy overview or impression of the image. ",
        )
        context_text = None
    elif method == "code":
        assert caption is not None
        query = re.sub(r"<img>.*?<\/img>", caption, query)
        context_text = None
    else:  # baseline, do not change
        context_text = None

    # HalTrapper: Modification Ends

    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content, context_text=None):
            response_text, response_tokens = f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(
                content, allowed_special=set(tokenizer.IMAGE_ST)
            )
            # HalTrapper: Modified
            if context_text is not None:
                context_tokens = tokenizer.encode(context_text)
                # HalTrapper: FIXME: 151858 Hardcoded here (</img>)
                eoi_pos = [i for i, x in enumerate(response_tokens) if x == 151858]
                for i in reversed(eoi_pos):
                    response_tokens = (
                        response_tokens[:i] + context_tokens + response_tokens[i:]
                    )
            return response_text, response_tokens

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    "assistant", turn_response
                )
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = (
                    nl_tokens + query_tokens + nl_tokens + response_tokens
                )
                prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        if append is None:
            context_tokens += (
                nl_tokens
                + im_start_tokens
                # HalTrapper: Modified
                + _tokenize_str("user", query, context_text)[1]
                + im_end_tokens
                + nl_tokens
                + im_start_tokens
                + tokenizer.encode("assistant")
                + nl_tokens
            )
            raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"
        else:
            context_tokens += (
                nl_tokens
                + im_start_tokens
                # HalTrapper: Modified
                + _tokenize_str("user", query, context_text)[1]
                + im_end_tokens
                + nl_tokens
                + im_start_tokens
                + tokenizer.encode("assistant")
                + nl_tokens
                + tokenizer.encode(append)
            )
            raw_text += (
                f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n{append}"
            )

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens
