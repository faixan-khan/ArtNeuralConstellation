import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob 

df_1k = pd.read_csv('./datasets/processed/combined_wofflins_real.csv')
df_5k = pd.read_csv('./datasets/processed/real_wofflin_scores_combined_normalised.csv')
combined = [df_1k, df_5k]
df = pd.concat(combined)

print('For Human Generated Art')
for col in df.columns[1:6]:
    print(col,': ',df[col].mean(),'  ',df[col].std())    

generations_csv = {
'StyleGAN2' : ('./dataset/features/generated_artworks/StyleGAN2/', './datasets/processed/combined_wofflins_styleGAN2.csv'),
'StyleCAN2' : ('./dataset/features/generated_artworks/StyleCAN2/', './datasets/processed/combined_wofflins_styleCAN2.csv'),
'StyleCWAN1' : ('./dataset/features/generated_artworks/StyleCWAN1/', './datasets/processed/combined_wofflins_styleCWAN1.csv'),
'StyleCWAN2' : ('./dataset/features/generated_artworks/StyleCWAN2/', './datasets/processed/combined_wofflins_styleCWAN2.csv'),
'StableDiffusion' : ('./dataset/features/generated_artworks/StableDiffusion/', './datasets/processed/combined_wofflins_SD.csv'),
'VQGAN' : ('./dataset/features/generated_artworks/VQGAN/', './datasets/processed/combined_wofflins_VQGAN.csv'),
}

generated_indexes = {}
generated_csvs = {}
print('For Machine Generated Art')

for k, (g, c) in generations_csv.items():
    # print(k,g,c)
    items = glob.glob(g + '*')
    items = [i for i in items if 'lowest_shape_entropy' not in i]
    csv = pd.read_csv(c)
    names = list(csv['Input.image'].apply(lambda x: Path(x).name).values)
    indexes = [names.index(Path(i).name) for i in items]
    generated_csvs[k] = csv
    generated_indexes[k] = indexes

for k, (g, c) in generations_csv.items():
    print(k)
    for col in generated_csvs[k].columns[1:6]:
        print(col,': ',generated_csvs[k][col].mean(),'   ',generated_csvs[k][col].std())    