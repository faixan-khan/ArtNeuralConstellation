import pickle
import pandas as pd
import numpy as np
import os

from pathlib import Path
from scipy.io import loadmat
from sklearn.decomposition import PCA

from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torchvision.transforms as transforms
from PIL import Image
import glob 
import torch

from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import pearsonr

def get_principal_components(all_feats, n_components, verbose=False):
    pca = PCA(n_components=n_components, svd_solver='full')
    feats = pca.fit_transform(all_feats)
    if verbose:
        print(pca.explained_variance_ratio_)
        print(np.cumsum(pca.explained_variance_ratio_))
    return feats, len(pca.explained_variance_ratio_)

def get_coefficients(feats, df, n_components, verbose=False):
    pc, components = get_principal_components(feats, n_components, verbose=verbose)
    
    components_to_return = components
    print(f"{components} components required")
    components = min(components, 30)
    coefficients = {}
    principles = ['Linearly-vs-Painterly', 'Planar-vs-Recessional',
       'closed-form-vs-open-form', 'multiplicity-vs-unity',
       'absolute-clarity-vs-relative-clarity']
    
    for p in principles:
        for c in range(components):
            coefficients.setdefault(p,[]).append(pearsonr(df[p].values, pc[:, c])[0])
        
    return pd.DataFrame(coefficients), components_to_return

df_1k = pd.read_csv('./datasets/processed/combined_wofflins_real.csv')
df_5k = pd.read_csv('./datasets/processed/real_wofflin_scores_combined_normalised.csv')
combined = [df_1k, df_5k]
df = pd.concat(combined)

df['name'] = df['Input.image'].apply(lambda x: Path(x).name)
mat_file = loadmat('./datasets/processed/groundtruth_pruned.mat')
files = [Path(f[0][0]).name for f in mat_file['groundtruth_pruned'][0][0][0]]
real_indexes = [files.index(f) for f in df.name.values]

folder = './datasets/features/real/'
principles = ['Linearly-vs-Painterly', 'Planar-vs-Recessional',
       'closed-form-vs-open-form', 'multiplicity-vs-unity',
       'absolute-clarity-vs-relative-clarity']
table_for_n_components = {}
dataframes = {}
for file in sorted(os.listdir(folder)):
    print(file)
    with open(os.path.join(folder, file), 'rb') as f:
        feats = pickle.load(f)
      
    coefficient_df, c = get_coefficients(feats[real_indexes], df, 0.95, verbose=False)
    table_for_n_components[file] = c
    dataframes[file] = coefficient_df.transpose()
    for p in range(5):
        sum=0
        maxi=0
        act_value=0
        index=0
        for i in range(coefficient_df.transpose().shape[1]):
            if(abs(coefficient_df.transpose()[i][p])>maxi):
                maxi = abs(coefficient_df.transpose()[i][p])
                act_value=coefficient_df.transpose()[i][p]
                index=i
            sum+=coefficient_df.transpose()[i][p]
        print(maxi,round(act_value,2),' -- ',index)
    print('##'*50)   

generations_csv = {
'StyleGAN2' : ('./datasets/features/generated_artworks/StyleGAN2/', './datasets/processed/combined_wofflins_styleGAN2.csv'),
'StyleCAN2' : ('./datasets/features/generated_artworks/StyleCAN2/', './datasets/processed/combined_wofflins_styleCAN2.csv'),
'StyleCWAN1' : ('./datasets/features/generated_artworks/StyleCWAN1/', './datasets/processed/combined_wofflins_styleCWAN1.csv'),
'StyleCWAN2' : ('./datasets/features/generated_artworks/StyleCWAN2/', './datasets/processed/combined_wofflins_styleCWAN2.csv'),
'StableDiffusion' : ('./datasets/features/generated_artworks/StableDiffusion/', './datasets/processed/combined_wofflins_SD.csv'),
'VQGAN' : ('./datasets/features/generated_artworks/VQGAN/', './datasets/processed/combined_wofflins_VQGAN.csv'),
}

generated_indexes = {}
generated_csvs = {}

for k, (g, c) in generations_csv.items():
    print(k,g,c)
    items = glob.glob(g + '*')
    items = [i for i in items if 'lowest_shape_entropy' not in i]
    csv = pd.read_csv(c)
    names = list(csv['Input.image'].apply(lambda x: Path(x).name).values)
    indexes = [names.index(Path(i).name) for i in items]
    generated_csvs[k] = csv
    generated_indexes[k] = indexes

folder = './datasets/features/generated_features/'

table_for_n_components = {}
dataframes = {}
for file in sorted(os.listdir(folder)):
#     if file.split('_')[0] != 'vit':
#         print(file)
#         continue
    generation_name = file.split('_')[-1].split('.')[0]
    generation_model = file.split('_')[0]
    if generation_name in ['StyleGAN1','StyleCAN1', 'random']:
        continue
    print(generation_name)
    with open(os.path.join(folder, file), 'rb') as f:
        feats = pickle.load(f)
    feats = np.array(list(feats.values())) 
    generated_csvs[generation_name].columns = ['Input.image', 'multiplicity-vs-unity', 'absolute-clarity-vs-relative-clarity', 'Linearly-vs-Painterly', 'Planar-vs-Recessional',
       'closed-form-vs-open-form']
    coefficient_df, c = get_coefficients(feats, generated_csvs[generation_name], 0.95)
    table_for_n_components[file] = c
    dataframes[file] = coefficient_df.transpose()
    print(file)
    for p in range(5):
        sum=0
        maxi=0
        act_value=0
        index=0
        for i in range(coefficient_df.transpose().shape[1]):
            if(abs(coefficient_df.transpose()[i][p])>maxi):
                maxi=abs(coefficient_df.transpose()[i][p])
                act_value=coefficient_df.transpose()[i][p]
                index=i
            sum+=coefficient_df.transpose()[i][p]
        print(maxi,round(act_value,2),' -- ',index)
        
        print(maxi)
    print('##'*50)