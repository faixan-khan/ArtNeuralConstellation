import pickle
from scipy.io import loadmat
import pickle
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import matplotlib as mpl
import matplotlib
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='A simple argument parser example.')

# Add an argument to the parser
parser.add_argument('--data_root', type=str, help='parent directory where data is stored', default='./datasets/features/')
# Parse the arguments
args = parser.parse_args()


df_1k = pd.read_csv('./datasets/processed/combined_wofflins_real.csv')
df_5k = pd.read_csv('./datasets/processed/real_wofflin_scores_combined_normalised.csv')
combined = [df_1k, df_5k]
df = pd.concat(combined)

df['name'] = df['Input.image'].apply(lambda x: Path(x).name)

for j,i in enumerate(df['name']):
    try:
        b=i.split('.')[0]
        year=int(b.split('-')[-1])
        if year < 1400 or year > 2000:
            print('remove :',i, year)
    except:
        print(i,j)

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

df.drop(df.index[[223,1985,2142]], inplace=True)
df['year'] = df.apply(get_year, axis=1)
df['year'] = df['year'].astype(int)

df = df.sort_values(by='year')

mat_file = loadmat('./datasets/processed/groundtruth_pruned.mat')
files = [Path(f[0][0]).name for f in mat_file['groundtruth_pruned'][0][0][0]]
real_indexes = [files.index(f) for f in df.name.values]

all_years = list(range(1500, 2001, 100))
cmap_string = 'jet'
cmap = matplotlib.cm.get_cmap(cmap_string)

cmap_values = (np.array(all_years) - 1400) / 600
colors = cmap(cmap_values)

sample_per_increment = 1000
colors = colors[:, None].repeat(sample_per_increment, axis=1)
colors = colors.reshape(-1, colors.shape[-1])
colors = colors[:df.shape[0]]

real_features_path = os.path.join(args.data_root, 'real') 
real_features = os.listdir(real_features_path)
real_features = ['resnet50_pretrained_real.pkl']
generated_features_path = os.path.join(args.data_root,'generated_features')
generated_features_files = os.listdir(generated_features_path)
# real_features = ['StyleGAN1_real.pkl','StyleCAN1_real.pkl','StyleCAN2_real.pkl','StyleCWAN1_real.pkl','StyleCWAN2_real.pkl','StyleGAN2_real.pkl']
for real in real_features:
    print(real)
    with open(os.path.join(real_features_path, real), 'rb') as f:
        real_feats = pickle.load(f)
    generated_feats = np.zeros((3200, real_feats.shape[1]))
    idx=0
    for gen in generated_features_files:
        model = real[:-9]
        name = gen[:gen.rfind('_')]
        # print(model, name)
        if model in name and name in model:
            # print(gen)
            with open(os.path.join(generated_features_path, gen), 'rb') as f:
                feats = pickle.load(f)
            try:
                values = np.array(list(feats.values())) 
            except:
                values = feats
                print(gen)
            generated_feats[idx*400:(idx+1)*400] = values
            idx+=1

    real_feats = real_feats[real_indexes]
    t = TSNE(n_components=2, random_state=16) 

    all_feats = np.concatenate((real_feats, generated_feats), axis=0)
    print('TSNE GENERATION')
    tsne_transformed_feats = t.fit_transform(all_feats)
    print('TSNE done...')
    sampled_all_feats_embedded = tsne_transformed_feats[:real_feats.shape[0]]
    gen_feats = tsne_transformed_feats[real_feats.shape[0]:]

    fig, ax = plt.subplots(figsize=(15,10))
    ax.margins(0.2,0.2)
    title = model.split('.')[0].split('_')
    try:
        ax.set_title(title[0] + '_' + title[1])
    except:
        ax.set_title(title[0])

    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=1400, vmax=2000), cmap=cmap))
    ax.scatter(sampled_all_feats_embedded[:,0], sampled_all_feats_embedded[:,1],
              s=30,
              edgecolors=colors,
              facecolor='none'
            )
    ax.scatter(gen_feats[:, 0], gen_feats[:, 1],
                s=25,
                marker='+',
                color='black'
            )

    plt.savefig('tsne_plot.png')
