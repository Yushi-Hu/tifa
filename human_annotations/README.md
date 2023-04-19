# Human Judgments on text-to-image faithfulness


This folder contains the human annotations for 800 tifa_v1.0 images synthesized by minDALL-E, VQ-Diffusion, Stable Diffusion v1.1, v1.5, v2.1. The images can be downloaded via this link: <a href="https://drive.google.com/file/d/16y87Wg1mYwPhfZnLeBHi0HH4RiwD3a5K/view?usp=share_link" download>https://drive.google.com/file/d/16y87Wg1mYwPhfZnLeBHi0HH4RiwD3a5K/view?usp=share_link</a>


`human_annotations.json` contains the human annotation scores. `human_annotations_with_scores.json` contains the human scores + the scores given by caption-based metrics, clipscore, and tifa scores with various VQA models.

`compute_correlation.ipynb` contains the code to compute the correlations between human scores and automatic metrics.
