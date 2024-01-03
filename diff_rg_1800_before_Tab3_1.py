import pandas as pd
import sys
import os
import torch
sys.path.append("..")
import style_attribute
import numpy as np
import matplotlib.pyplot as plt
#compute z statistic human art drawn before 1800 vs. AI art
#to find visual element where generative art, for example of StyleGAN1, are significantly different from real.
visual_elements_15=['non-representational','representational','geometric','abstract','planar','closed','open','rough','perspective','broken','thin','flat','distorted','linear','ambiguous']
models=["StyleGAN1","StyleGAN2","StyleCAN1","StyleCAN2","StyleCWAN1","StyleCWAN2","StableDiffusion","VQGAN"]
attr_idx=[style_attribute.ATTRIBUTES.index(v) for v in visual_elements_15]
z_left,z_right=-3.0, 3.0



def main():
    os.system("mkdir ./diff_rg_plt_1800/")
    df_real=pd.read_csv("./fileinfo_csv/val.csv",header=0)
    df_generative=pd.read_csv("./fileinfo_csv/generative.csv",header=0)
    def get_year(x):
        first=x.split("-")[-1].split(".")[0].split("(")[0]
        try:
             second=x.split("-")[-2].split("-")[0]
        except:
            second="none"

        if first.isnumeric() and len(first)==4:
            first=int(first)
        else:
            first=0

        if second.isnumeric() and len(second)==4:
            second=int(second)
        else:
            second=0
        return max(first, second)
    df_real["year"]=df_real.painting.apply(get_year)
    lgt_real=np.load("./proxy_embedding/embedding_val.npz")["visual_elements"][:,attr_idx]#real
    lgt_generative=np.load("./proxy_embedding/embedding_generative.npz")["visual_elements"][:,attr_idx]#real
    df_real=pd.concat([df_real,pd.DataFrame(lgt_real,columns=visual_elements_15)],axis=1)
    df_real=df_real[(df_real["year"]>0) & (df_real["year"]<1800)]
    df_real.reset_index(inplace=True, drop=True)
    #standardization#
    mean=np.mean(df_real.iloc[:,-15:],axis=0).values
    std=np.std(df_real.iloc[:,-15:],axis=0).values
    df_real.iloc[:,-15:]=(df_real.iloc[:,-15:]-mean)/std
    
    lgt_generative_std=(lgt_generative-mean)/std
    df_generative=pd.concat([df_generative,pd.DataFrame(lgt_generative_std,columns=visual_elements_15)],axis=1)
    
    
    df_generative_=df_generative[df_generative["style"].isin(models)]
    df_generative_.reset_index(inplace=True, drop=True)
    fig,ax = plt.subplots(len(visual_elements_15),figsize=(1.75,8.3))
    fig.tight_layout(pad=1e-5)
    
    for idx,element in enumerate(visual_elements_15):
        #z-score computation
        mean_real=np.mean(df_real[element].values,axis=0)
        mean_generative=np.mean(df_generative_[element].values,axis=0)
        std_generative=np.std(df_generative_[element].values,axis=0)
        z_score=(mean_generative-mean_real)/(std_generative/20.0)#20=sqrt(400)
        ax[idx].scatter(mean_real,0,c="blue",s=7)#zero
        ax[idx].scatter(mean_generative,0,c="red",s=5)#the outlier marking
        ax[idx].set_xlim((-3,3))
        if z_score<z_left or z_score>z_right:
             ax[idx].set_title(element+" with z: "+str(round(z_score,2)),y=0.6,fontsize=7.5,color="red")
        else:
             ax[idx].set_title(element,y=0.6,fontsize=7.5)
        ax[idx].set_yticklabels([])
        ax[idx].tick_params(color="silver")
        plt.setp(ax[idx].spines.values(), color="silver")
        if idx!=0:
            ax[idx].set_xticklabels([])
    print("complete...")
    plt.savefig("./diff_rg_plt_1800/before_1800.png")
                      
if __name__ == "__main__":
    main()
