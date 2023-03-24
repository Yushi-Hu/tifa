# TIFA: Accurate and Interpretable Text-to-Image Evaluation with Question Answering

This repository contains the code and models for our paper [TIFA: Accurate and Interpretable Text-to-Image Evaluation with Question Answering](https://arxiv.org/abs/2303.11897). Please refer to the [project page](https://tifa-benchmark.github.io/) for a quick overview.

<img src="static/tifa_webteaser2.png" width="800">

#### Why TIFA?
We have pre-generated the questions with OpenAI APIs, such that users only need to run a VQA model in this repo to benchmark their text-to-image models. Our research shows that TIFA is much more accurate than CLIP, while being fine-grained and interpretable. Meanwhile, this repo also provides tools to allow users to customize their own TIFA benchmark.

#### Leaderboard

<img src="static/tifa_leaderboard.png" width="600">
Want to submit results on the leaderboard? Please email the authors.

**************************** **Updates** ****************************

* TODO: Release Flan-T5 fine-tuned for question generation so that users can generate questions without OpenAI API. Release Huggingface demo.
* 03/24: We released the evaluation package, which includes evaluation code, VQA modules, and question generation modules
* 03/22: We released the TIFA v1.0 captions, questions.


## Quick Links
- [Installation](#Installation)
- [Quick Start](#quick-start)
- [TIFA v1.0 Benchmark](#tifa-v1.0-benchmark)
- [TIFA on customized benchmark](#tifa-on-customized-benchmark)
- [VQA modules](#vqa-modules)
- [Question Generation modules](#question-generation-modules)
  -[Question Generation with GPT-3.5](#quetion-generation-with-gpt-3.5)
  -[Filtering with UnifiedQA](#filter-with-unifiedqa)
- [TIFA on arbitrary image and text](#tifa-on-arbitrary-image-and-text)

## Installation

## Quick Start

## TIFA v1.0 Benchmark

TIFA v1.0 text inputs are in `tifa_v1.0/tifa_v1.0_text_inputs.json` 

You can also <a href="https://raw.githubusercontent.com/Yushi-Hu/tifa/main/tifa_v1.0/tifa_v1.0_text_inputs.json" download>Download here</a>

The GPT-3 pre-generated TIFA v1.0 question and answers are in `tifa_v1.0/tifa_v1.0_question_answers.json`. 

You can also <a href="https://raw.githubusercontent.com/Yushi-Hu/tifa/main/tifa_v1.0/tifa_v1.0_question_answers.json" download>Download here</a>

## TIFA on Customized Benchmark

## VQA Modules

## Question Generation Modules

### Question generation with GPT 3.5

### Filter with UnifiedQA

## TIFA on arbitary image and text

## Details: Customized benchmark format

The text inputs are organized as follows:
```console
[
    {
        "id": "coco_301091",    # the unique text id
        "caption": "On a gray day a surfer carrying a white board walks on a beach.",     # the text
        "coco_val_id": "380711"    # for COCO captions only. The COCO image id corresponding to the caption
    },
    ...
]
```

The question and answers are organized as follows:
```console
[
    {
        "id": "coco_301091",    # the unique text id, correspond to the text inputs file
        "caption": "On a gray day a surfer carrying a white board walks on a beach.",    # the text
        "question": "is this a surfer?",
        "choices": [
            "yes",
            "no"
        ],
        "answer": "yes",
        "answer_type": "animal/human",       # the element type
        "element": "surfer",                 # the element the question is about
        "coco_val_id": "380711"              # for COCO captions only. the COCO image id corresponding to the caption
    },
    ...
]
```

## Citation
If you find our work helpful, please cite us:

```bibtex
@article{Hu2023TIFAAA,
  title={TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering},
  author={Yushi Hu and Benlin Liu and Jungo Kasai and Yizhong Wang and Mari Ostendorf and Ranjay Krishna and Noah A. Smith},
  journal={ArXiv},
  year={2023},
  volume={abs/2303.11897}
}
```

