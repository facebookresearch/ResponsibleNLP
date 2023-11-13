# Perturbation Augmentation for Fairer NLP

This folder contains datasets and other artifacts for the Perturbation Augmentation for Fairer NLP project.

[Paper: Rebecca Qian, Candace Ross, Jude Fernandes, Eric Smith, Douwe Kiela, Adina Williams. *Perturbation Augmentation for Fairer NLP.* 2022.](https://aclanthology.org/2022.emnlp-main.646/)

## PANDA
Perturbation Augmentation NLP DAtaset (PANDA) consists of approximately 100K pairs of human-perturbed text snippets (source, perturbed). PANDA can be used for training a learned perturber that can rewrite text along three demographic axes (age, gender, race), in a way that preserves semantic meaning. PANDA can also be used to evaluate the demographic robustness of language models.

## Models
- The perturber seq2seq model that rewrites text along a specified demographic axis and attribute can be found at https://huggingface.co/facebook/perturber .
- The FairBERTa LLM trained on demographically perturbed corpora can be found at https://huggingface.co/facebook/FairBERTa .

Rewind the repo to before November 13, 2023 to view crowdsourcing code (no longer supported).

Please reach out with feedback, issues and suggestions!
