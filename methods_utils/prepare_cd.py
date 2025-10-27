import torch
from transformers.cache_utils import DynamicCache
from copy import deepcopy


def prepare_kwargs_for_cd(input_ids, model_kwargs):
    input_ids_cd = model_kwargs.pop("input_ids_cd", None)
    use_cd = model_kwargs.get("use_cd")

    if use_cd:
        if input_ids_cd is None:
            input_ids_cd = input_ids.clone()
        cd_type = model_kwargs["cd_type"]
        model_kwargs_cd = model_kwargs.copy()

        if "inputs_embeds_cd" in model_kwargs.keys():
            del model_kwargs["inputs_embeds_cd"]
            model_kwargs_cd["inputs_embeds"] = model_kwargs_cd["inputs_embeds_cd"]
            del model_kwargs_cd["inputs_embeds_cd"]
        elif "inputs_embeds" in model_kwargs.keys():
            del model_kwargs_cd["inputs_embeds"]

        if "attention_mask_cd" in model_kwargs.keys():
            del model_kwargs["attention_mask_cd"]
            model_kwargs_cd["attention_mask"] = model_kwargs_cd["attention_mask_cd"]
            del model_kwargs_cd["attention_mask_cd"]
        elif "attention_mask" in model_kwargs.keys():
            del model_kwargs_cd["attention_mask"]

        if "input_scaling_cd" in model_kwargs.keys():
            del model_kwargs["input_scaling_cd"]
            model_kwargs_cd["input_scaling"] = model_kwargs_cd["input_scaling_cd"]
            del model_kwargs_cd["input_scaling_cd"]
        elif "input_scaling" in model_kwargs.keys():
            del model_kwargs_cd["input_scaling"]

        if "position_ids_cd" in model_kwargs.keys():
            del model_kwargs["position_ids_cd"]
            model_kwargs_cd["position_ids"] = model_kwargs_cd["position_ids_cd"]
            del model_kwargs_cd["position_ids_cd"]
        elif "position_ids" in model_kwargs.keys():
            del model_kwargs_cd["position_ids"]

        if "pixel_values_cd" in model_kwargs.keys():
            del model_kwargs["pixel_values_cd"]
            model_kwargs_cd["pixel_values"] = model_kwargs_cd["pixel_values_cd"]
            del model_kwargs_cd["pixel_values_cd"]
        elif "pixel_values" in model_kwargs.keys():
            del model_kwargs_cd["pixel_values"]

        if "pixel_values_videos_cd" in model_kwargs.keys():
            del model_kwargs["pixel_values_videos_cd"]
            model_kwargs_cd["pixel_values_videos"] = model_kwargs_cd[
                "pixel_values_videos_cd"
            ]
            del model_kwargs_cd["pixel_values_videos_cd"]
        elif "pixel_values_videos" in model_kwargs.keys():
            del model_kwargs_cd["pixel_values_videos"]

        if "image_grid_thw_cd" in model_kwargs.keys():
            del model_kwargs["image_grid_thw_cd"]
            model_kwargs_cd["image_grid_thw"] = model_kwargs_cd["image_grid_thw_cd"]
            del model_kwargs_cd["image_grid_thw_cd"]
        elif "image_grid_thw" in model_kwargs.keys():
            del model_kwargs_cd["image_grid_thw"]

        if "video_grid_thw_cd" in model_kwargs.keys():
            del model_kwargs["video_grid_thw_cd"]
            model_kwargs_cd["video_grid_thw"] = model_kwargs_cd["video_grid_thw_cd"]
            del model_kwargs_cd["video_grid_thw_cd"]
        elif "video_grid_thw" in model_kwargs.keys():
            del model_kwargs_cd["video_grid_thw"]

        # HalTrapper: FIXME: Hardcode Here
        if "inputs_embeds" in model_kwargs_cd.keys():
            bs, ls = model_kwargs_cd["inputs_embeds"].shape[:2]
        else:
            bs, ls = input_ids_cd.shape[:2]

        model_kwargs_cd["attention_mask"] = torch.ones(
            bs, ls, dtype=torch.int64, device=model_kwargs["attention_mask"].device
        )
        if "cache_position" in model_kwargs_cd.keys():
            model_kwargs_cd["cache_position"] = torch.arange(
                ls,
                dtype=model_kwargs_cd["cache_position"].dtype,
                device=model_kwargs_cd["cache_position"].device,
            )
        if "past_key_values" in model_kwargs_cd.keys() and isinstance(
            model_kwargs_cd["past_key_values"], DynamicCache
        ):
            model_kwargs_cd["past_key_values"] = deepcopy(
                model_kwargs["past_key_values"]
            )

    else:
        if "inputs_embeds_cd" in model_kwargs.keys():
            del model_kwargs["inputs_embeds_cd"]
        if "attention_mask_cd" in model_kwargs.keys():
            del model_kwargs["attention_mask_cd"]
        if "input_scaling_cd" in model_kwargs.keys():
            del model_kwargs["input_scaling_cd"]
        if "position_ids_cd" in model_kwargs.keys():
            del model_kwargs["position_ids_cd"]
        if "pixel_values_cd" in model_kwargs.keys():
            del model_kwargs["pixel_values_cd"]
        if "image_grid_thw_cd" in model_kwargs.keys():
            del model_kwargs["image_grid_thw_cd"]
        if "pixel_values_videos_cd" in model_kwargs.keys():
            del model_kwargs["pixel_values_videos_cd"]
        if "video_grid_thw_cd" in model_kwargs.keys():
            del model_kwargs["video_grid_thw_cd"]
        model_kwargs_cd = None
        cd_type = None

    return input_ids, input_ids_cd, model_kwargs, model_kwargs_cd, use_cd, cd_type
