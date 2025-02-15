# AI Art Neural Constellation
This repo contains the implementation codes for all the experiments performed in the paper titled "AI Art Neural Constellation: Revealing the Collective and Contrastive State of AI-Generated and Human Art" [report](https://arxiv.org/abs/2402.02453).
Motivated by the recent success of generative  machine learning as a way of art creation, this paper aims to demonstrate  comparative and statistical analysis  between 6,000 WikiArt and 3,200 AI-generated artworks. Five important aspects of understanding visual arts are grounded in the analysis; the codes and related files are categorized by the five aspects below. 

### Five Aspects for Visual Analysis and Codes

1. Wofflin's five principles
2. General art principles
3. OOD analysis in CLIP space
4. Time
5. Emotion and Likability

### Dataset
The dataset can be downloaded from [here](https://neuralartconst.s3.us-east-2.amazonaws.com/dataset.zip).

### External Codes Referenced

- StyleGAN1 and StyleGAN2 models are taken from [rosinality/style-based-gan-pytorch](https://github.com/rosinality/style-based-gan-pytorch) and [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) respectively.
- StyleCAN1 and StyleCAN2 models are taken from [Vision-CAIR/WAGA](https://github.com/Vision-CAIR/WAGA).
- StyleCWAN1 and StyleCWAN2 models are taken from [Vision-CAIR/CWAN](https://github.com/Vision-CAIR/CWAN).
- StyleCWAN1 and StyleCWAN2 models are taken from [Vision-CAIR/CWAN](https://github.com/Vision-CAIR/CWAN).
- VQGAN model is taken from [CompVis/VQGAN](https://github.com/CompVis/taming-transformers).
- Diffusion model is taken from [Diffusers](https://huggingface.co/docs/diffusers/en/tutorials/basic_training).
- Proxy Learning is taken from [diana-s-kim/ProxyLearning_torch](https://github.com/diana-s-kim/ProxyLearning_torch).
