---
license: apache-2.0
datasets:
- BabyLM-community/babylm-eus
language:
- en
metrics:
- accuracy
pipeline_tag: text-generation
---
# 🦙 SLlama: Parameter-Efficient Language Model Architecture for Enhanced Linguistic Competence Under Strict Data Constraints
# Overview
Scaling data and model size has driven much of the recent progress in language modeling. However, this approach breaks down in settings with strict data constraints, such as the BabyLM Challenge. Insights from training compute-optimal LLMs show that:

Smaller models trained on more data generally outperform

Larger models trained insufficiently,
highlighting the importance of compact, efficient architectures.

A commonly used compression method is embedding weight tying, but we find that in very small models this technique significantly harms linguistic competence—a limitation that is often overlooked.

# 🦙 Introducing SLlama
To address this issue, we explore architectural strategies that retain the parameter savings of tied embeddings without sacrificing the representational advantages of untied embeddings.

SLlama is a compact variant of Llama-3 that integrates several targeted modifications designed to compress Transformer components efficiently:

Repeated Reduced Hidden Size and Projection (RRHP)

Permutated Weight Attention (PWA)

Shared Projection Multi-Layer Perceptron (SPMLP)

Layer Weight Sharing

These techniques allow SLlama to remain lightweight while preserving expressive capacity.

## Installation
```bash
git clone https://github.com/ovasoft/sllama.git
cd sllama
pip install -r requirements.txt
```
## Babylm Dataset
Download the [Babylm dataset](https://babylm.github.io/) into the path specified in config.yaml

## Usage
You may want to update the configuration file to suit your need.
To train the model from stratch without gpu, run:
```
python train.py
```

To train the model with gpu 0, run:
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

After training (which should take less than 20 minutes on a gpu), the model should be in trainer_config.out_dir specified in the config file

## Evaluation
Consequent to training, evaluate the model using the  [Babylm evaluation pipeline](https://github.com/babylm/evaluation-pipeline-2025)


## Citation

If you use this work, please cite:

```bibtex
@inproceedings{omolaoye-etal-2025-sllama,
    title = "{SL}lama: Parameter-Efficient Language Model Architecture for Enhanced Linguistic Competence Under Strict Data Constraints",
    author = "Omolaoye, Victor Adelakun  and
      Owoyele, Babajide Alamu  and
      de Melo, Gerard",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1198/",
    doi = "10.18653/v1/2025.emnlp-main.1198",
    pages = "23491--23506",
    ISBN = "979-8-89176-332-6"
}