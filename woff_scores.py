import pickle
import pandas as pd
import numpy as np
import os, glob
from pathlib import Path

dict={}

df_1k = pd.read_csv('./datasets/processed/combined_wofflins_real.csv')
df_5k = pd.read_csv('./datasets/processed/real_wofflin_scores_combined_normalised.csv')
combined = [df_1k, df_5k]
df = pd.concat(combined)

bins = [0,0.2,0.4,0.6,0.8,1.1]

for col in df.columns[1:6]:
    dict[col.lower()] = {}
    dict[col.lower()]['Real'] = []
    total = 0
    for i in range(1,len(bins)):
        lower_bound = bins[i-1]
        upper_bound = bins[i]
        count_within_range = df[(df[col] >= lower_bound) & (df[col] < upper_bound)].shape[0]
        total += count_within_range
        count_within_range = count_within_range / 6000
        dict[col.lower()]['Real'].append(count_within_range)

generations_csv = {
'StyleGAN2' : ('./datasets/features/generated_artworks/StyleGAN2/', './datasets/processed/combined_wofflins_styleGAN2.csv'),
'StyleCAN2' : ('./datasets/features/generated_artworks/StyleCAN2/', './datasets/processed/combined_wofflins_styleCAN2.csv'),
'StyleCWAN1' : ('./datasets/features/generated_artworks/StyleCWAN1/', './datasets/processed/combined_wofflins_styleCWAN1.csv'),
'StyleCWAN2' : ('./datasets/features/generated_artworks/StyleCWAN2/', './datasets/processed/combined_wofflins_styleCWAN2.csv'),
'DDPM' : ('./datasets/features/generated_artworks/StableDiffusion/', './datasets/processed/combined_wofflins_SD.csv'),
'VQGAN' : ('./datasets/features/generated_artworks/VQGAN/', './datasets/processed/combined_wofflins_VQGAN.csv'),
}


generated_indexes = {}
generated_csvs = {}

for k, (g, c) in generations_csv.items():
    items = glob.glob(g + '*')
    items = [i for i in items if 'lowest_shape_entropy' not in i]
    csv = pd.read_csv(c)
    names = list(csv['Input.image'].apply(lambda x: Path(x).name).values)
    indexes = [names.index(Path(i).name) for i in items]
    generated_csvs[k] = csv
    generated_indexes[k] = indexes

for k, (g, c) in generations_csv.items():
    for col in generated_csvs[k].columns[1:6]:
        col_key = col.lower()
        total = 0
        if col == 'absolute-vs-relative':
            col_key = 'absolute-clarity-vs-relative-clarity'
        dict[col_key][k] = []
        
        for i in range(1,len(bins)):
            lower_bound = bins[i-1]
            upper_bound = bins[i]
            count_within_range = generated_csvs[k][(generated_csvs[k][col] >= lower_bound) & (generated_csvs[k][col] < upper_bound)].shape[0]
            count_within_range = count_within_range / 400
            dict[col_key][k].append(count_within_range)
            total += count_within_range

import matplotlib.pyplot as plt
import numpy as np

print(dict)
for col in dict.keys():
    dict[col]['Human'] = dict[col].pop('Real')
    desired_order = ['Human','StyleGAN2','StyleCAN2','StyleCWAN1','StyleCWAN2','DDPM','VQGAN']
    new_dict = {key: dict[col][key] for key in desired_order}
    dict[col] = new_dict

print(dict)


for col in df.columns[1:6]:
    years = [0.2,0.4,0.6,0.8,1.0]


    x = np.arange(len(years))  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0

    col_key = col.lower()
    fig, ax = plt.subplots(layout='constrained',figsize=(15, 5))

    for attribute, measurement in dict[col_key].items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.set_ylabel('Percent of Artworks',fontsize=16)
    ax.set_xlabel('Bins',fontsize=16)
    ax.set_title(col_key,fontsize=16)
    ax.set_xticks(x + width, years,fontsize=12)
    ax.legend(loc='upper left', ncols=3,fontsize=16)
    ax.set_ylim(0, 1)

    plt.savefig(col_key + '_wolf.png')


        