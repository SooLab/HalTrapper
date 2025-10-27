# HalTrapper

This is the official PyTorch implementation for our **ICCV 2025** paper:

> **Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context**  
> \> Ge Zheng<sup>1,2*</sup>&emsp; Jiaye Qian<sup>2*</sup>&emsp; Jiajin Tang<sup>2</sup>&emsp; Sibei Yang<sup>1†</sup>  
> \> <sup>1</sup>School of Computer Science and Engineering, Sun Yat-sen University &emsp;<sup>2</sup>ShanghaiTech University

![HalTrapper Pipeline](assets/pipeline.svg)

[![arXiv:2510.20229](https://img.shields.io/badge/arXiv-2510.20229-red)](https://arxiv.org/abs/2510.20229)

## Abstract

Large Vision-Language Models (LVLMs) have made significant progress in recent years but are also prone to hallucination issues. They exhibit more hallucinations in longer, free-form responses, often attributed to accumulated uncertainties. In this paper, we ask: Does increased hallucination result solely from length-induced errors, or is there a deeper underlying mechanism? After a series of preliminary experiments and findings, we suggest that the risk of hallucinations is not caused by length itself but by the increased reliance on context for coherence and completeness in longer responses. Building on these insights, we propose a novel “induce-detect-suppress” framework that actively induces hallucinations through deliberately designed contexts, leverages induced instances for early detection of high-risk cases, and ultimately suppresses potential object-level hallucinations during actual decoding. Our approach achieves consistent, significant improvements across all benchmarks, demonstrating its efficacy. The strong detection and improved hallucination mitigation not only validate our framework but, more importantly, re-validate our hypothesis on context. Rather than solely pursuing performance gains, this study aims to provide new insights and serves as a first step toward a deeper exploration of hallucinations in LVLMs’ longer responses.

## Setup

### Environment Setup

Our code requires Python ≥ 3.9. When evaluating different models, we use specific versions of the `transformers` library for each model family. Due to API changes across different versions of `transformers`, using other versions may result in errors. Our code includes version assertions in certain modules to prevent unexpected behaviors. The versions are listed below:

| Model     | `transformers` Version |
| --------- | ---------------------- |
| LLaVA 1.5 | 4.37.2                 |
| Qwen VL   | 4.32.0                 |
| MiniGPT-4 | 4.30.0                 |
| Qwen2 VL  | 4.45.0                 |
| Janus Pro | 4.48.3                 |

We recommend following the official installation instructions provided on each model's GitHub repository for setting up their dependencies.

Additionally, to evaluate CHAIR and AMBER, install the following:

```sh
pip install spacy nltk "numpy<2"
python -m spacy download en_core_web_lg
```

### Path Setup

You need to specify the paths in [`playground/path_table.py`](playground/path_table.py), replacing the `path/to/xxx` placeholders with your actual paths.

#### COCO & AMBER

To evaluate CHAIR and AMBER, you must download the COCO and AMBER datasets. Links are provided below:

- [COCO Dataset](https://cocodataset.org/)
- [AMBER Repository](https://github.com/junyangwang0410/AMBER/tree/master)

For the COCO dataset, please specify the path to the `val2014/` folder which contains the image files directly. For the AMBER dataset, please use the path to the `data/` folder from the repository above. We assume that images for AMBER are under the `data/image/` folder in AMBER root.

#### MiniGPT-4

To evaluate MiniGPT-4, you need to specify the root path to MiniGPT-4 official repository, then set up the MiniGPT-4 first. We assume that the config file for MiniGPT-4 is located in the MiniGPT-4 repository under `eval_configs/minigpt4_llama2_eval.yaml`. Please ensure the correct configuration file path is specified based on the model architecture you are evaluating.

## Evaluation

### Start of Evaluation

To evaluate on the CHAIR benchmark using greedy decoding, run:

```bash
python decontext.py \
    --model [model] \
    --method [method] \
    --eval chair \
    --fixed True
```

The `--fixed True` flag ensures the evaluation uses a fixed set of 500 questions rather than randomly sampling 500 questions.

To evaluate on the AMBER benchmark using greedy decoding, run:

```bash
python decontext.py \
    --model [model] \
    --method [method] \
    --eval amber \
    --split g \
    --change-prompt True
```

The `--split g` flag specifies evaluation on the generative subset only.

The available options for `[model]` are:
- `llava` for LLaVA v1.5 7B
- `qwenvl` for Qwen VL
- `minigpt4` for MiniGPT 4
- `qwen2vl` for Qwen2 VL
- `januspro` for Janus Pro

The available options for `[method]` are:
- `baseline` for the vanilla model evaluation
- `vcd` for [Visual Contrastive Decoding](https://github.com/DAMO-NLP-SG/VCD)
- `icd` for [Instruction Contrastive Decoding](https://github.com/hillzhang1999/ICD)
- `pai` for [Paying More Attention to Image](https://github.com/LALBJ/PAI)
- `code` for [Countering Description Contrastive Decoding](https://github.com/IVY-LVLM/CODE)
- `haltrapper` for ours

You can add `--sample` to enable nucleus sampling, or `--num_beams 5` to enable beam search.

### Output and Configuration Logging

At the end of evaluation, the results will be printed, and the detailed model inference outputs will be automatically saved in the `result/` directory as a `.jsonl` file. A corresponding configuration record of this inference will also be saved as a YAML file with the suffix `-config.yaml`. If you want to **manually evaluate existing `.jsonl` results only**, run the following commands:

**For CHAIR**

```sh
python -m playground.eval \
    [path/to/model-outputs.jsonl] \
    --eval chair \
    --fixed True
```

**For AMBER**

```sh
python -m playground.eval \
    [path/to/model-outputs.jsonl] \
    --eval amber \
    --split g \
    --change-prompt True
```

### HalTrapper: Cache for Hallucinaion Candidates

Our approach involves two main steps: generating hallucination candidates for each image, and subsequently mitigating hallucinations. Since the candidate generation step is relatively slow and the results can be reused, our implementation automatically stores hallucination candidates for each image processed by a model in the `cache/` folder. You can manually delete these cache files to clear the cached data.

What's more, due to this caching mechanism, inference on the same image will be significantly faster during subsequent runs compared to the initial processing.

## Acknowledgements

Our implementation incorporates or modifies code from the following open-source repositories. We extend our sincere gratitude to the authors of these projects (listed in no particular order):
- [junyangwang0410/AMBER](https://github.com/junyangwang0410/AMBER)
- [IVY-LVLM/CODE](https://github.com/IVY-LVLM/CODE)
- [hillzhang1999/ICD](https://github.com/hillzhang1999/ICD)
- [kinsDev/Janus-Pro](https://github.com/kinsDev/Janus-Pro)
- [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [LALBJ/PAI](https://github.com/LALBJ/PAI)
- [QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [DAMO-NLP-SG/VCD](https://github.com/DAMO-NLP-SG/VCD)

## Citation

If you find our work useful, please cite us as:

```bib
@InProceedings{Zheng_2025_HalTrapper,
    author    = {Zheng, Ge and Qian, Jiaye and Tang, Jiajin and Yang, Sibei},
    title     = {Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {4101-4113}
}
```
