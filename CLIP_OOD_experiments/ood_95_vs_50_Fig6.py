import pandas as pd
import os

#ood_95_vs_50_Fig6: how oodness is changed according to 95% vs. 50% of pca


target=["pca-20","pca-30","pca-50","pca-70","pca-95","pca-all"]
img_path="/Users/dianakim/2023_spring/Research/StyleMode_forGenerative_release1/generative_art/"
def main():
        df=pd.read_csv("./fileinfo_csv/generative_w_info.csv",header=0)
        #when ood at 95%pca - see whether id vs. ood at 50%pca
        paintings=df[df["pca-95"]==1.0]["painting"].values
        distances=df[df["pca-95"]==1.0]["pca-95-dist"].values
        sorted_painting=[x for _, x in sorted(zip(distances, paintings),reverse=True)] #larger = oodness
        for idx, img in enumerate(sorted_painting[:10]):
            if df[df["painting"]==img]["pca-50"].values==1:
                print("ood-img",img,"ood in 50")
            else:
                print("ood-img",img,"id in 50")

        #when id at 95%pca - see wheather id vs. ood at 50%pca
        paintings=df[df["pca-95"]==0.0]["painting"].values
        distances=df[df["pca-95"]==0.0]["pca-95-dist"].values
        sorted_painting=[x for _, x in sorted(zip(distances, paintings))] #larger = oodness

        for idx, img in enumerate(sorted_painting[:10]):
            if df[df["painting"]==img]["pca-50"].values==1:
                print("id-img",img,"ood in 50")
            else:
                print("id-img",img,"id in 50")
        
if __name__ == "__main__":
    main()
