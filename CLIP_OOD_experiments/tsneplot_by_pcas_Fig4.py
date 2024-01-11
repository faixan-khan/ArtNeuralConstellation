import pandas as pd
import os
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,AnnotationBbox)
from matplotlib.lines import Line2D
import matplotlib as mpl
import style_attribute


#tsneplot_by_pcas_Fig4: Fig4, AI art vs human art speparation t-sne plots according to different pca

target=["pca-20","pca-30","pca-50","pca-70","pca-95","pca-all"]
models=["StyleGAN2","StyleCAN1","StableDiffusion","VQGAN"]
x=[20,30,50,70,95,100]
colors=['black','darkgray','cornflowerblue','darkslateblue']


class Visual_2D:
    def __init__(self,df,ax):
        self.df=df
        self.year=df.year.iloc[:-3200]
        self.ax=ax
        self.scatter_()
        self.anno_box=[]
        self.num_samples=df.shape[0]
        
    def scatter_(self):
        color=self.year.tolist()
        self.ax.scatter(self.df.iloc[:-3200,-2].to_numpy(),self.df.iloc[:-3200,-1].to_numpy(),c=color,s=5,picker=4,label="human art")
        self.ax.scatter(self.df.iloc[-3200:,-2].to_numpy(),self.df.iloc[-3200:,-1].to_numpy(),c='tab:orange',s=5,picker=4,alpha=0.35,label="generated art")



def main():
    df=pd.read_csv("./fileinfo_csv/generative_w_info.csv",header=0)
    fig,ax = plt.subplots(2,4,figsize=(16,3.5))
    plt.figure(figsize=(16.5,4))
    ax1=plt.subplot2grid((2, 4), (0, 0), colspan=1)
    for idx,model in enumerate(models):
        print("model:",model)
        num_ood=[]
        for comp in target:
            num_ood.append(df[(df[comp]==1.0) & (df["style"]==model)]["painting"].size)
        if model=="StableDiffusion":
            model="DDPM"
        ax1.plot(x,num_ood,':',c=colors[idx],label=model,linewidth=1.0,marker=".")
    ax1.set_yticks([50,100,200,400])
    ax1.set_xticks([20,30,50,70,95,100])
    ax1.set_ylabel("OOD counts",color="black",fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.legend(loc=2,fontsize=7.5)
    ax1.set_xlabel("PCA percentage (%)",fontsize=9)
    
    #all
    num_ood=[]    
    for comp in target:
        num_ood.append(df[(df[comp]==1.0)]["painting"].size)
        
    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=1)
    ax2.plot(x,num_ood,'-',c="tab:red",label="all (all ood for eight models)",linewidth=1.0,marker=".")

    ax2.annotate("("+str(num_ood[0])+")",xy=(x[0],num_ood[0]),xytext=(x[0],num_ood[0]+50),ha='center', va='top',fontsize=7,color="tab:red")
    ax2.annotate("("+str(num_ood[1])+")",xy=(x[1],num_ood[1]),xytext=(x[1],num_ood[1]+50),ha='center', va='top',fontsize=7,color="tab:red")
    ax2.annotate("("+str(num_ood[2])+")",xy=(x[2],num_ood[2]),xytext=(x[2]+5,num_ood[2]),ha='center', va='top',fontsize=7,color="tab:red")
    ax2.annotate("("+str(num_ood[3])+")",xy=(x[3],num_ood[3]),xytext=(x[3]+5,num_ood[3]),ha='center', va='top',fontsize=7,color="tab:red")
    ax2.annotate("("+str(num_ood[4])+")",xy=(x[4],num_ood[4]),xytext=(x[4]+5,num_ood[4]),ha='center', va='top',fontsize=7,color="tab:red")
    ax2.annotate("("+str(num_ood[5])+")",xy=(x[5],num_ood[5]),xytext=(x[5],num_ood[5]+45),ha='center', va='top',fontsize=7,color="tab:red")
    ax2.tick_params(axis='y', which='major', labelsize=8, colors="tab:red")
    ax2.set_ylabel("OOD counts",color="black",fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=8)


    #legend
    ax2.legend(loc=2,fontsize=7.5)
    ax2.set_xlabel("PCA percentage (%)",fontsize=9)
    ax2.set_xlabel("PCA percentage (%)",fontsize=9)



    
    #50%
    #real#
    df=pd.read_csv("./fileinfo_csv/clip_real.csv",header=0) 
    df["style"]=df["style"].apply(lambda x: style_attribute.STYLE_MERGE[x])
    df["style"]=df["style"].apply(lambda x: style_attribute.STYLES.index(x))    
    df["year"]=df.painting.apply(lambda x: int (x.split("-")[-1].split(".")[0]) if len(x.split("-")[-1].split(".")[0])==4 and x.split("-")[-1].split(".")[0].isnumeric() else 0)
    tsne_lgt=np.load("./tsne/t_sne_transformed_train_embedding_50.npz")["tsne"]
    df=pd.concat([df,pd.DataFrame(tsne_lgt[:-3200,:])],axis=1)
    df=df[df["year"]>0]

    #gen#
    df_gen=pd.read_csv("./fileinfo_csv/clip_generative.csv",header=0)
    df_gen["painting"]=df_gen.apply(lambda x: pd.Series(x["style"]+"/"+x["painting"]),axis=1)
    df_gen=pd.concat([df_gen,pd.DataFrame(tsne_lgt[-3200:,:])],axis=1)
    #df=df.append(df_gen)
    df=pd.concat([df,df_gen],ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    ax3 = plt.subplot2grid((2, 4), (0, 1), rowspan=2)
    Visual_2D(df,ax3)
    ax3.set_xlabel("t-SNE for PCA 50%",color="black",fontsize=9)

    ax3.spines['top'].set_color("dimgray")
    ax3.spines['bottom'].set_color("dimgray")
    ax3.spines['right'].set_color("dimgray")
    ax3.spines['left'].set_color("dimgray")

    ax3.set_ylim(-85,85)
    ax3.set_xlim(-75,75)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.legend(frameon=False)
    
    #70%
    #real#
    df=pd.read_csv("./fileinfo_csv/clip_real.csv",header=0) 
    df["style"]=df["style"].apply(lambda x: style_attribute.STYLE_MERGE[x])
    df["style"]=df["style"].apply(lambda x: style_attribute.STYLES.index(x))    
    df["year"]=df.painting.apply(lambda x: int (x.split("-")[-1].split(".")[0]) if len(x.split("-")[-1].split(".")[0])==4 and x.split("-")[-1].split(".")[0].isnumeric() else 0)
    tsne_lgt=np.load("./tsne/t_sne_transformed_train_embedding_70.npz")["tsne"]
    df=pd.concat([df,pd.DataFrame(tsne_lgt[:-3200,:])],axis=1)
    df=df[df["year"]>0]

    #gen#
    df_gen=pd.read_csv("./fileinfo_csv/clip_generative.csv",header=0)
    df_gen["painting"]=df_gen.apply(lambda x: pd.Series(x["style"]+"/"+x["painting"]),axis=1)
    df_gen=pd.concat([df_gen,pd.DataFrame(tsne_lgt[-3200:,:])],axis=1)
    #df=df.append(df_gen)
    df=pd.concat([df,df_gen],ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    ax4 = plt.subplot2grid((2, 4), (0, 2), rowspan=2)
    Visual_2D(df,ax4)
    ax4.set_xlabel("t-SNE for PCA 70%",color="black",fontsize=9)

    ax4.spines['top'].set_color("dimgray")
    ax4.spines['bottom'].set_color("dimgray")
    ax4.spines['right'].set_color("dimgray")
    ax4.spines['left'].set_color("dimgray")

    ax4.set_ylim(-85,85)
    ax4.set_xlim(-75,75)
    ax4.tick_params(axis='both', which='major', labelsize=8)
    
    #95%
    #real#
    df=pd.read_csv("./fileinfo_csv/clip_real.csv",header=0) 
    df["style"]=df["style"].apply(lambda x: style_attribute.STYLE_MERGE[x])
    df["style"]=df["style"].apply(lambda x: style_attribute.STYLES.index(x))    
    df["year"]=df.painting.apply(lambda x: int (x.split("-")[-1].split(".")[0]) if len(x.split("-")[-1].split(".")[0])==4 and x.split("-")[-1].split(".")[0].isnumeric() else 0)
    tsne_lgt=np.load("./tsne/t_sne_transformed_train_embedding_95.npz")["tsne"]
    df=pd.concat([df,pd.DataFrame(tsne_lgt[:-3200,:])],axis=1)
    df=df[df["year"]>0]

    #gen#
    df_gen=pd.read_csv("./fileinfo_csv/clip_generative.csv",header=0)
    df_gen["painting"]=df_gen.apply(lambda x: pd.Series(x["style"]+"/"+x["painting"]),axis=1)
    df_gen=pd.concat([df_gen,pd.DataFrame(tsne_lgt[-3200:,:])],axis=1)
    #df=df.append(df_gen)
    df=pd.concat([df,df_gen],ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    ax5 = plt.subplot2grid((2, 4), (0, 3), rowspan=2)
    Visual_2D(df,ax5)
    ax5.set_xlabel("t-SNE for PCA 95%",color="black",fontsize=9)

    ax5.spines['top'].set_color("dimgray")
    ax5.spines['bottom'].set_color("dimgray")
    ax5.spines['right'].set_color("dimgray")
    ax5.spines['left'].set_color("dimgray")

    ax5.set_ylim(-85,85)
    ax5.set_xlim(-75,75)
    ax5.tick_params(axis='both', which='major', labelsize=8)

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=1148, vmax=2012)
    cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax5)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout(pad=0.001)
    plt.savefig("ood_tsne.png")


if __name__ == "__main__":
    main()
