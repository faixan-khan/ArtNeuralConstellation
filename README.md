# AI Art Neural Constellation
This repo contains the implementation codes for all the experiments performed in the paper titled "AI Art Neural Constellation: Revealing the Collective and Contrastive State of AI-Generated and Human Art" [report]().
Motivated by the recent success of generative  machine learning as a way of art creation, this paper aims to demonstrate  comparative and statistical analysis  between 6,000 WikiArt and 3,200 AI-generated artworks. Five important aspects to understand visual arts are grounded in the analysis; the codes and related files are categorized by the five aspects as below. 

### Five Aspects for Visual Analysis and Codes

├── (1)Wofflin's five principles
│   ├── average-wofflin-and-correlation.ipynb
├── (2)General art principles
│   ├── proxy learning  [proxy learning repo]()
│   ├── fileinfo_csv
│   │   ├── train.csv, val.csv #human art data filenames and style
│   │   ├── generative.csv #AI art data filenames and model 
│   ├──  style_attribute.py #styles and attributes used in experiments
│   ├──  general_principles_density_analysis_Fig3.py
│   ├──  diff_rg_1800_before_Tab3_1.py
│   ├──  diff_rg_1800_after_Tab3_2.py
├── (3)CLIP_OOD_experiments
│   ├── fileinfo_csv
│   │   ├── clip_generative.csv 
│   │   ├── clip_real.csv
│   │   ├── generative.csv
│   ├── tnse
│   ├──  style_attribute.py #styles and attributes used in experiments
│   ├──  tsneplot_by_pcas_Fig4.py
│   ├──  cllct_id_ood_Fig5.py
│   ├──  ood_95_vs_50_Fig6.py
├── (4)Time
└── (5)Emotion and Likability

### External Codes Referenced

- The implementations for StyleGAN1 and StyleGAN2 models are taken from [rosinality/style-based-gan-pytorch](https://github.com/rosinality/style-based-gan-pytorch) and [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) respectively.
- The implementations for StyleCAN1 and StyleCAN2 models are taken from [Vision-CAIR/WAGA](https://github.com/Vision-CAIR/WAGA).
- The implementations for StyleCWAN1 and StyleCWAN2 models are taken from [Vision-CAIR/CWAN](https://github.com/Vision-CAIR/CWAN).
- The implementations for StyleCWAN1 and StyleCWAN2 models are taken from [Vision-CAIR/CWAN](https://github.com/Vision-CAIR/CWAN).
- The implementations for Proxy Learning  are taken from [diana-s-kim/ProxyLearning_torch](https://github.com/diana-s-kim/ProxyLearning_torch).