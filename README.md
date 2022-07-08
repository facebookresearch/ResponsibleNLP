This repository hosts code and datasets relating to Responsible NLP projects from Meta AI.

# Projects

- [`holistic_bias`](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias):
  - From [Eric Michael Smith, Melissa Hall, Melanie Kambadur, Eleonora Presani, Adina Williams. *"I'm sorry to hear that": finding bias in language models with a holistic descriptor dataset.* 2022.](https://arxiv.org/pdf/2205.09209.pdf)
  - Code to generate a dataset, **HolisticBias**, consisting of nearly 600 demographic terms in over 450k sentence prompts
  - Code to calculate **Likelihood Bias**, a metric of the amount of bias in a language model, defined on HolisticBias demographic terms
- [`fairscore`](https://github.com/facebookresearch/ResponsibleNLP/tree/main/fairscore):
  - From [Rebecca Qian, Candace Ross, Jude Fernandes, Eric Smith, Douwe Kiela, Adina Williams. *Perturbation Augmentation for Fairer NLP.* 2022.](https://dynabench.org/fairer_nlp.pdf)
  - **PANDA**, an annotated dataset of 100K demographic perturbations of diverse text, rewritten to change gender, race/ethnicity and age references.
  - The perturber, pretrained models, code and other artifacts related to the Perturbation Augmentation for Fairer NLP project will be released shortly.

-----

See [CONTRIBUTING.md](https://github.com/facebookresearch/ResponsibleNLP/blob/main/CONTRIBUTING.md) for how to help out, and see [LICENSE](https://github.com/facebookresearch/ResponsibleNLP/blob/main/LICENSE) for license information.
