from PIL import Image
import os
import pandas as pd
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

csv_dir = './datasets/processed/likeability'
csv_files = os.listdir(csv_dir)
csvs = {k:pd.read_csv(os.path.join(csv_dir, k)) for k in csv_files}

to_keep = ['highest_NN_dist', 'lowest_NN_dist', 'highest_shape_entropy', 'random']

files = ['StyleGAN1.csv', 'StyleCAN1.csv', 'StyleGAN2.csv', 'StyleCAN2.csv', 'StyleCWAN1.csv', 'StyleCWAN2.csv','StableDiffusion.csv','VQGAN.csv']

for file in files:
    print(file)
    if file == 'StyleGAN2.csv' or file == 'StyleCAN2.csv':
        name = 'Input.image'
        col =  'Input.mean'
        turing = 'Input.is_generated_by_artist'
    else:
        name = 'Input.kn_img_url'
        col =  'Answer.Q1_answer'
        turing = 'Answer.Q2_answer'
    for grp in to_keep:
        mean_rating = csvs[file][[name,col]].groupby(name).mean().filter(like=grp,axis=0).sort_values(by=col,ascending=True).mean()
        print(grp,': ', mean_rating.values)
    print('Turing Score: ',csvs[file][turing].mean())
    print('Mean Score:',csvs[file][col].mean())
    print('Score Std: ',csvs[file][col].std())

painting_high={}
painting_low={}
for i in csvs.keys():
    painting_high[i] = {}
    painting_low[i] = {}
    try:
        name = 'Input.image'
        col =  'Input.mean'
        turing = 'Input.is_generated_by_artist'
        for gr in to_keep:
            temp = csvs[i][[name,col]].groupby(name).mean().filter(like=gr,axis=0).sort_values(by=col)
            temp = temp.reset_index()
            painting_high[i][gr] = temp[name].iloc[-1]
            painting_low[i][gr] = temp[name].iloc[0]
            # print(gr, csvs[i][[name,col]].groupby(name).mean().filter(like=gr,axis=0).sort_values(by=col).iloc[0])
            # print(gr, csvs[i][[name,col]].groupby(name).mean().filter(like=gr,axis=0).sort_values(by=col).iloc[-1])        
    except:
        name = 'Input.kn_img_url'
        col =  'Answer.Q1_answer'
        turing = 'Answer.Q2_answer'
        for gr in to_keep:
            temp = csvs[i][[name,col]].groupby(name).mean().filter(like=gr,axis=0).sort_values(by=col)
            temp = temp.reset_index()
            painting_high[i][gr] = temp[name].iloc[-1]
            painting_low[i][gr] = temp[name].iloc[0]
            # print(gr, csvs[i][[name,col]].groupby(name).mean().filter(like=gr,axis=0).sort_values(by=col).iloc[0])
            # print(gr, csvs[i][[name,col]].groupby(name).mean().filter(like=gr,axis=0).sort_values(by=col).iloc[-1])


order = ['StyleGAN1.csv','StyleCAN1.csv','VQGAN.csv','StableDiffusion.csv']
organise=[]
for gr in to_keep:
    for i in order:
        most_liked = os.path.join(i.split('.')[0],painting_high[i][gr].split('/')[-1])
        organise.append(most_liked)
    for i in order:
        least_liked = os.path.join(i.split('.')[0],painting_low[i][gr].split('/')[-1])
        organise.append(least_liked)

data='./datasets/features/generated_artworks/'
images = [plt.imread(os.path.join(data,image_path)) for image_path in organise]

# Create a grid of images
grid_height = 4
grid_width = 8

fig, axes = plt.subplots(grid_height, grid_width, figsize=(15, 5))

for i, ax in enumerate(axes.flatten()):
    if i < len(images):
        ax.imshow(images[i])
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
plt.savefig('likability.png')