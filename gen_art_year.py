import pickle
import numpy as np
from scipy.io import loadmat
import os
import re
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import argparse
import collections

parser = argparse.ArgumentParser(description='A simple argument parser example.')

# Add an argument to the parser
parser.add_argument('--data_root', type=str, help='parent directory where data is stored', default='./datasets/features/')
parser.add_argument('--full', action='store_false', help='to run nn of full wikiart or just 6000')

# Parse the arguments
args = parser.parse_args()
mat_file = loadmat('./datasets/processed/groundtruth_pruned.mat')

years = []
re_string = '[0-9][0-9][0-9][0-9]\.'
for m in mat_file['groundtruth_pruned'][0][0][0]:
    text = re.findall(re_string, m[0][0])
    if text != []:
        year = int(text[0][:-1])
        if year < 1400 or year > 2010:
            years.append(np.nan)
        else:
            years.append(int(year))
    else:
        years.append(np.nan)

years = np.array(years)
indexes = ~np.isnan(years)

print('Average year for real art: ',years[indexes].mean())

df_1k = pd.read_csv('./datasets/processed/combined_wofflins_real.csv')
df_5k = pd.read_csv('./datasets/processed/real_wofflin_scores_combined_normalised.csv')
combined = [df_1k, df_5k]
df = pd.concat(combined)

df['name'] = df['Input.image'].apply(lambda x: Path(x).name)

def get_year(row):
    try:
        y = int(row['name'].split('.')[0].split('-')[-1])
        if int(y) < 1400:
            y = None  
        if int(y) > 2000:
            y = None 
    except ValueError:
        y = None
        
    return y

df['year'] = df.apply(get_year, axis=1)
df[df['year'].isna()]
df.drop(index=[223,985,1142], inplace=True)

df['year'] = df['year'].astype(int)

if args.full:
    years = years[indexes]
else:
    years = df['year'].values

print(years.shape)

mat_file = loadmat('./datasets/processed/groundtruth_pruned.mat')
files = [Path(f[0][0]).name for f in mat_file['groundtruth_pruned'][0][0][0]]
real_indexes = [files.index(f) for f in df.name.values]

real_features_path = os.path.join(args.data_root, 'real') 
real_features = os.listdir(real_features_path)
generated_features_path = os.path.join(args.data_root,'generated_features')
generated_features_files = sorted(os.listdir(generated_features_path))
real_features = ['resnet50_pretrained_real.pkl']

dict={}

for real in real_features:
    with open(os.path.join(real_features_path, real), 'rb') as f:
        real_feats = pickle.load(f)

    if args.full:
        real_feats = real_feats[indexes]
    else:
        real_feats = real_feats[real_indexes]
    
    neigh = NearestNeighbors(n_neighbors=5,n_jobs=8)
    neigh.fit(real_feats)

    for gen in generated_features_files:
        model = real[:-9]
        name = gen[:gen.rfind('_')]
        if 'StyleGAN1' in gen or 'StyleCAN1'in gen: 
            continue
        if model in name and name in model:
            with open(os.path.join(generated_features_path, gen), 'rb') as f:
                feats = pickle.load(f)
            generated_feats = np.array(list(feats.values())) 
            distances,indices = neigh.kneighbors(generated_feats, 5)
            average_year=[]
            for i in range(distances.shape[0]):
                year_1 = years[indices[i]]
                year_painting_avg = np.mean(year_1)
                average_year.append(year_painting_avg)

            np_average_year = np.array(average_year, dtype=np.int32)
            counter=collections.Counter(np_average_year)
            my_years = [1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
            all_sum=0
            year_wise=[]
            for j in range(len(my_years)-1):
                st = my_years[j]
                et = my_years[j+1]
                ck=0
                for i in sorted(counter.items()):
                    if st <= i[0] < et:
                        ck+=i[1]
                    if i[0] >= et:
                        break
                all_sum+=ck
                year_wise.append(ck)
            key = gen.split('_')[-1].split('.')[0]
            dict[key] = year_wise
            print('Average Year for the mdoel: ',gen.split('_')[-1].split('.')[0], ' is: ',np.mean(np_average_year))

print(dict)

dict['DDPM'] = dict.pop('StableDiffusion')
desired_order = ['StyleGAN2','StyleCAN2','StyleCWAN1','StyleCWAN2','DDPM','VQGAN']
new_dict = {key: dict[key] for key in desired_order}
dict = new_dict

# plotting
import matplotlib.pyplot as plt
import numpy as np

years = ['1400-1450','1450-1500','1500-1550','1550-1600','1600-1650','1650-1700','1700-1750','1750-1800','1800-1850','1850-1900','1900-1950','1950-2000']


x = np.arange(len(years))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(layout='constrained',figsize=(15, 5))
i=1
for attribute, measurement in dict.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i])
    multiplier += 1
    i +=1

ax.tick_params(axis='y', which='major', labelsize=12)
ax.set_ylabel('Number of Artworks', fontsize=16)
ax.set_xlabel('Time Period',fontsize=16)
ax.set_title('Year',fontsize=16)
ax.set_xticks(x + width, years,fontsize=12)
ax.legend(loc='upper left', ncols=3,fontsize=20)
ax.set_ylim(0, 220)

if args.full:
    plt.savefig('Year_WiKiArt.png')
else:
    plt.savefig('Year_WiKiArt.png')


        