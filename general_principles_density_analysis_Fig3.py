#draw density figure on 15 elements
#generate densities for Fig.3
import pandas as pd
import sys
import torch
sys.path.append("..")
import style_attribute
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
visual_elements_15=['non-representational','representational','geometric','abstract','planar','closed','open','rough','perspective','broken','thin','flat','distorted','linear','ambiguous']
attr_idx=[style_attribute.ATTRIBUTES.index(v) for v in visual_elements_15]
std=0.5

models=["StyleGAN1","StyleGAN2","StyleCAN1","StyleCAN2","StyleCWAN1","StyleCWAN2","StableDiffusion","VQGAN"]
sig_z=[4.12,-3.7,2.86,4.34,3.96,-3.75,1.55,4.86,-4.62,3.46,-3.09,4.49,3.32,1.85,4.32]

class KDE(nn.Module):#element specific density
    def __init__(self,element):
        super().__init__()
        self.element=element
    def forward(self,x):
         normal_points=1.0/np.sqrt((2.0*np.pi*std**2))*torch.exp(-1*((x-self.element)**2)/(2*std**2))#400x#centroids(data)
         normal_eval=torch.mean(normal_points,dim=1)
         return normal_eval

def main():
    df=pd.read_csv("./fileinfo_csv/val.csv",header=0)
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

    df["year"]=df.painting.apply(get_year)
    lgt=np.load("./proxy_embedding//embedding_val.npz")["visual_elements"][:,attr_idx]#real
    lgt_generative=np.load("./proxy_embedding/embedding_generative.npz")["visual_elements"][:,attr_idx]#real
    df_generative=pd.read_csv("./fileinfo_csv/generative.csv",header=0)



    mean=np.mean(lgt,axis=0)
    std=np.std(lgt,axis=0)
    lgt_std=(lgt-mean)/std
    lgt_generative_std=(lgt_generative-mean)/std

    fig,ax = plt.subplots(3,5,figsize=(15.5,5))
    os.system("mkdir ./all_density_plt/")

    for idx,element in enumerate(visual_elements_15):
        row_=int(idx/5)
        col_=int(idx%5)
        df_=pd.concat([df,pd.DataFrame(lgt_std,columns=visual_elements_15)],axis=1)
    
        kde_real=KDE(df_[element].values)
        a=torch.arange(-7,7,step=0.01).unsqueeze(dim=1)
        result=kde_real(a)
        x=a.numpy()
        y=result.numpy()
        real_art,=ax[row_,col_].plot(x,y,'tab:blue',linestyle='-',label="All: human art",linewidth=1.0)

        #compute center points#
        mean_real_x=np.mean(df_[element].values,axis=0)
        kde_real_y=kde_real(torch.tensor([[mean_real_x]])).numpy()[0]

        
        #year<1800
        df_=pd.concat([df,pd.DataFrame(lgt_std,columns=visual_elements_15)],axis=1)
        df_=df_[(df_["year"]>0) & (df_["year"]<1800)]
        df_.reset_index(inplace=True, drop=True)
        kde_real=KDE(df_[element].values)
        a=torch.arange(-7,7,step=0.01).unsqueeze(dim=1)
        result=kde_real(a)
        x=a.numpy().squeeze()
        y=result.numpy().squeeze()
        before_1800,=ax[row_,col_].plot(x,y,'black',linestyle='-',label="before 1800",linewidth=0.5)
    
        #year>1700 and year<1900
        df_=pd.concat([df,pd.DataFrame(lgt_std,columns=visual_elements_15)],axis=1)
        df_=df_[(df_["year"]>1700) & (df_["year"]<1900)]
        df_.reset_index(inplace=True, drop=True)
        kde_real=KDE(df_[element].values)
        a=torch.arange(-7,7,step=0.01).unsqueeze(dim=1)
        result=kde_real(a)
        x=a.numpy().squeeze()
        y=result.numpy().squeeze()
        year_1700_1900,=ax[row_,col_].plot(x,y,'black',linestyle='-.',label="1700-1900",linewidth=0.5)


        #year>=1700 & year<1900
        df_=pd.concat([df,pd.DataFrame(lgt_std,columns=visual_elements_15)],axis=1)
        df_=df_[(df_["year"]>1800)]
        df_.reset_index(inplace=True, drop=True)
        kde_real=KDE(df_[element].values)
        a=torch.arange(-7,7,step=0.01).unsqueeze(dim=1)
        result=kde_real(a)
        x=a.numpy().squeeze()
        y=result.numpy().squeeze()
        after_1800,=ax[row_,col_].plot(x,y,'black',linestyle=':',label="after 1800",linewidth=0.5)

        #genertive
        df_generative_=pd.concat([df_generative,pd.DataFrame(lgt_generative_std,columns=visual_elements_15)],axis=1)
        df_generative_=df_generative_[df_generative_["style"].isin(models)]
        kde_gen=KDE(df_generative_[element].values)
        a=torch.arange(-7,7,step=0.01).unsqueeze(dim=1)
        result=kde_gen(a)
        x=a.numpy().squeeze()
        y=result.numpy().squeeze()
        generated_art,=ax[row_,col_].plot(x,y,'tab:red',linestyle="-",label="generated_art",linewidth=1.0)
        mean_gen_x=np.mean(df_generative_[element].values,axis=0)
        kde_gen_y=kde_gen(torch.tensor([[mean_gen_x]])).numpy()[0]


        #center plots#
        ax[row_,col_].plot([mean_real_x,mean_real_x],[0,kde_real_y],'dimgray',linestyle="-",linewidth=1.0)
        ax[row_,col_].scatter([mean_real_x,mean_real_x],[0,kde_real_y],c='dimgray',s=2.5)
        ax[row_,col_].plot([mean_gen_x,mean_gen_x],[0,kde_gen_y],'tab:red',linestyle="-",linewidth=1.0)
        ax[row_,col_].scatter([mean_gen_x,mean_gen_x],[0,kde_gen_y],c='tab:red',s=2.5)

        #plot-config
        ax[row_,col_].set_xlim(-2.5,2.5)
        ax[row_,col_].set_yticks([0,0.1,0.2])
        ax[row_,col_].tick_params(axis='x',labelsize=8,color="dimgray",labelcolor="dimgray")
        ax[row_,col_].tick_params(axis='y',labelsize=7.5,color="dimgray",labelcolor="dimgray")
        if row_!=0  or col_!=0:
            ax[row_,col_].set_yticks([])
        ax[row_,col_].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[row_,col_].spines['top'].set_color("dimgray")
        ax[row_,col_].spines['bottom'].set_color("dimgray")
        ax[row_,col_].spines['right'].set_color("dimgray")
        ax[row_,col_].spines['left'].set_color("dimgray")

        
        if sig_z[idx]>3.0 or sig_z[idx]<-3.0:
            ax[row_,col_].set_title("("+str(idx+1)+") "+element+": z-score:"+str(sig_z[idx]),fontsize=10,loc="left",pad=-10,color="tab:red")            
        else:
            ax[row_,col_].set_title("("+str(idx+1)+") "+element+": z-score:"+str(sig_z[idx]),fontsize=10,loc="left",pad=-10,color="black")
            ax[row_,col_].spines['top'].set_color("gray")
            ax[row_,col_].spines['bottom'].set_color("gray")
            ax[row_,col_].spines['right'].set_color("gray")
            ax[row_,col_].spines['left'].set_color("gray")

        
    ax[0,0].set_xlim(-6,6)    
    ax[2,0].legend(handles=[real_art],ncol=1,bbox_to_anchor=(0.5,-0.175),loc='upper right',fontsize=9.5)
    ax[2,1].legend(handles=[generated_art],ncol=1,bbox_to_anchor=(0.5,-0.175),loc='upper right',fontsize=9.5)
    ax[2,2].legend(handles=[before_1800],ncol=1,bbox_to_anchor=(0.5,-0.175),loc='upper right',fontsize=9.5)
    ax[2,3].legend(handles=[year_1700_1900],ncol=1,bbox_to_anchor=(0.5,-0.175),loc='upper right',fontsize=9.5)
    ax[2,4].legend(handles=[after_1800],ncol=1,bbox_to_anchor=(0.5,-0.175),loc='upper right',fontsize=9.5)
    fig.tight_layout(pad=0.1)


    plt.savefig("./all_density_plt/density.png")
    print("complete...")
if __name__ == "__main__":
    main()
