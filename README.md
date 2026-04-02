# Do Audio-Visual Large Language Models Really See and Hear?

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-paper-b31b1b" alt="arXiv"/>
  </a>
  <a href="https://avllm-interpretability.github.io">
    <img src="https://img.shields.io/badge/Project-Website-blue" alt="Website"/>
  </a>
  <a href="https://huggingface.co/datasets/XXXXXXX">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow" alt="HuggingFace"/>
  </a>
</p>

Code for experiments conducted in the paper, with Qwen 2.5 Omni as the representative model.


## Installation
```bash
pip3 install -r requirements.txt
```

## Experiments

### Logit Lens Experiment
```bash
python src/logitlens_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

### Attention Knockout Experiment
```bash
python src/attention_knockout_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```