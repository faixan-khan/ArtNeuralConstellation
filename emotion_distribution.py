import pandas as pd

generations_csv = {
'Real' : './datasets/processed/combined_wolfflins_emotions/all_data_combined_wofflin_and_emotion',
'StyleGAN2' : './datasets/processed/combined_wolfflins_emotions/combined_StyleGAN2.csv',
'StyleCAN2' : './datasets/processed/combined_wolfflins_emotions/combined_StyleCAN2.csv',
'StyleCWAN1' : './datasets/processed/combined_wolfflins_emotions/combined_StyleCWAN1.csv',
'StyleCWAN2' : './datasets/processed/combined_wolfflins_emotions/combined_StyleCWAN2.csv',
'StableDiffusion' : './datasets/processed/combined_wolfflins_emotions/combined_Diffusion.csv',
'VQGAN' : './datasets/processed/combined_wolfflins_emotions/combined_VQGAN.csv',
}

generated_indexes = {}
generated_csvs = {}

emotions = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness', 'something else']

for k, c in generations_csv.items():
    csv = pd.read_csv(c)
    generated_csvs[k] = csv

for k in generated_csvs.keys():
    generated_csvs[k] = generated_csvs[k][emotions]


dict={}
for k in generated_csvs.keys():
    temp = generated_csvs[k].sum() 
    if k == 'Real':
        temp = temp / (generated_csvs['Real'].shape[0]*5)
    else:    
        temp = temp / (400*5)
    dict[k] = temp * 100

dict['DDPM'] = dict.pop('StableDiffusion')
desired_order = ['Real','StyleGAN2','StyleCAN2','StyleCWAN1','StyleCWAN2','DDPM','VQGAN']
new_dict = {key: dict[key] for key in desired_order}
dict = new_dict

import matplotlib.pyplot as plt
import numpy as np

years = emotions


x = np.arange(len(years))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained',figsize=(15, 5))

for attribute, measurement in dict.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percent of Artworks', fontsize=16)
ax.set_xlabel('Emotions', fontsize=16)
ax.set_title('Emotion Distribution', fontsize=16)
ax.set_xticks(x + width, years, fontsize=16)
ax.legend(loc='upper left', ncols=3, fontsize=16)
ax.set_ylim(0, 30)


plt.savefig('emotions-dist.png')