This repository hosts code and datasets relating to Responsible NLP projects from Meta AI.

# Projects

- [`AdvPromptSet`](https://github.com/facebookresearch/ResponsibleNLP/tree/main/AdvPromptSet)
  - **AdvPromptSet**: a comprehensive and challenging adversarial text prompt set with 197,628 prompts of varying toxicity levels and more than 24 sensitive demographic identity groups and combinations.
- [`fairscore`](https://github.com/facebookresearch/ResponsibleNLP/tree/main/fairscore):
  - From [Rebecca Qian, Candace Ross, Jude Fernandes, Eric Smith, Douwe Kiela, Adina Williams. *Perturbation Augmentation for Fairer NLP.* 2022.](https://aclanthology.org/2022.emnlp-main.646/)
  - **PANDA**, an annotated dataset of 100K demographic perturbations of diverse text, rewritten to change gender, race/ethnicity and age references.
  - The perturber, pretrained models, code and other artifacts related to the Perturbation Augmentation for Fairer NLP project will be released shortly.
- [`gender_gap_pipeline`](https://github.com/facebookresearch/ResponsibleNLP/tree/main/gender_gap_pipeline):
  - **The Gender-GAP Pipeline**, from [Benjamin Muller,  Belen Alastruey, Prangthip Hansanti, Elahe Kalbassi, Christophe Ropers, Eric Michael Smith, Adina Williams, Luke Zettlemoyer, Pierre Andrews, Marta R Costa-jussà](https://arxiv.org/pdf/2308.16871.pdf)
- [`holistic_bias`](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias):
  - From [Eric Michael Smith, Melissa Hall, Melanie Kambadur, Eleonora Presani, Adina Williams. *"I'm sorry to hear that": finding bias in language models with a holistic descriptor dataset.* 2022.](https://arxiv.org/pdf/2205.09209.pdf)
  - Code to generate a dataset, **HolisticBias**, consisting of nearly 600 demographic terms in over 450k sentence prompts
  - Code to calculate **Likelihood Bias**, a metric of the amount of bias in a language model, defined on HolisticBias demographic terms
- [`robbie`](https://github.com/facebookresearch/ResponsibleNLP/tree/main/robbie):
  - **ROBBIE**: we test 6 bias/toxicity metrics (including 2 novel ones) across 5 model families and 3 bias/toxicity mitigation techniques, and show that using a broad array of metrics enables much better assessment of safety issues in these models and mitigations.

-----

See [CONTRIBUTING.md](https://github.com/facebookresearch/ResponsibleNLP/blob/main/CONTRIBUTING.md) for how to help out, and see [LICENSE](https://github.com/facebookresearch/ResponsibleNLP/blob/main/LICENSE) for license information.
