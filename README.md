# SKEL: A Shared Knowledge Extraction and Learning Framework for
Open-Vocabulary Audio-Visual Event Localization

## Abstract
Open-Vocabulary Audio-Visual Event Localization
(OV-AVEL) aims to recognize both seen and unseen sounding
event categories and localize their temporal intervals in videos.
Existing methods usually ignore transferable shared knowledge
among related event categories, making unseen foreground events
more likely to be misclassified as background. Meanwhile, inter
category shared knowledge has been demonstrated to facilitate
generalization to unseen categories in open-vocabulary settings.
However, these shared-knowledge-based methods cannot be di
rectly applied to the OV-AVEL task: (1) existing inter-category
shared knowledge extraction methods are insufficient to charac
terize the complex audio-visual features of events, resulting in un
reliable task-relevant shared knowledge; and (2) external priors
are difficult to support scalable transfer of shared knowledge to
continuously expanding unseen categories. To address these limi
tations, we propose a Shared Knowledge Extraction and Learning
(SKEL) framework for the OV-AVEL task, which consists of
three key components: a Prompt Semantic Enhancement (PSE)
module, a Superclass Semantic Center Learning (SSCL) module,
and a Teacher-Student Learning Strategy (TSLS). The PSE and
SSCL aim to extract reliable task-relevant shared knowledge
by deriving superclass semantic centers through multimodal
prompting and semantic clustering. To further enhance the scala
bility of shared knowledge transfer, TSLS employs distillation to
embed shared knowledge into the student branch and facilitate
its generalization from seen categories to continuously emerging
unseen categories. Extensive experiments demonstrate that SKEL
effectively improves the localization and recognition of unseen
events and outperforms state-of-the-art baselines.

## Data Preparation
### Dataset
The proposed OV-AVEBench dataset is available now. You may directly download the preprocessed audio (.wav) and visual (.png) files from [this link](https://mailhfuteducn-my.sharepoint.com/personal/2018110964_mail_hfut_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F2018110964%5Fmail%5Fhfut%5Fedu%5Fcn%2FDocuments%2FOV%2DAVEBench%2Fovave%5Fdataset%5Fpreprocessed%2Etar%2Egz&parent=%2Fpersonal%2F2018110964%5Fmail%5Fhfut%5Fedu%5Fcn%2FDocuments%2FOV%2DAVEBench&ga=1) to develop your own models for OV-AVEL task. The raw videos are also available at [here](https://mailhfuteducn-my.sharepoint.com/personal/2018110964_mail_hfut_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F2018110964%5Fmail%5Fhfut%5Fedu%5Fcn%2FDocuments%2FOV%2DAVEBench%2FOV%2DAVEBench%5Fraw%5Fvideos%2Etar%2Egz&parent=%2Fpersonal%2F2018110964%5Fmail%5Fhfut%5Fedu%5Fcn%2FDocuments%2FOV%2DAVEBench&ga=1). Please put the downloaded preprocessed data into `ovave_dataset_preprocessed' directory.

### pretrained backbone
Download the ImageBind_Huge from https://github.com/facebookresearch/ImageBind/tree/main

## Train

    bash run.sh
