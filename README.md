# Self-supervised Pre-training of Text Recognizers

This repository contains the code for the paper [Self-supervised Pre-training of Text Recognizers](https://arxiv.org/abs/2405.00420) by [Martin Kišš](https://www.fit.vut.cz/person/ikiss) and [Michal Hradiš](https://www.fit.vut.cz/person/ihradis) for [2024 International Conference on Document Analysis and Recognition](https://icdar2024.net/) (ICDAR 2024).
In this repository you can find implementations of investigated methods, models, and visualizations.

[[arxiv](https://arxiv.org/abs/2405.00420)]

### Paper abstract
In this paper, we investigate self-supervised pre-training methods for document text recognition. Nowadays, large unlabeled datasets can be collected for many research tasks, including text recognition, but it is costly to annotate them. Therefore, methods utilizing unlabeled data are researched. We study self-supervised pre-training methods based on masked label prediction using three different approaches -- Feature Quantization, VQ-VAE, and Post-Quantized AE. We also investigate joint-embedding approaches with VICReg and NT-Xent objectives, for which we propose an image shifting technique to prevent model collapse where it relies solely on positional encoding while completely ignoring the input image. We perform our experiments on historical handwritten (Bentham) and historical printed datasets mainly to investigate the benefits of the self-supervised pre-training techniques with different amounts of annotated target domain data. We use transfer learning as strong baselines. The evaluation shows that the self-supervised pre-training on data from the target domain is very effective, but it struggles to outperform transfer learning from closely related domains. This paper is one of the first researches exploring self-supervised pre-training in document text recognition, and we believe that it will become a cornerstone for future research in this area. We made our implementation of the investigated methods publicly available at https://github.com/DCGM/pero-pretraining.  

