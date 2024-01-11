
import pandas as pd
import os


#cllct_id_ood_Fig5: sort id and ood images according to their oodness at 95% pca

target=["pca-20","pca-30","pca-50","pca-70","pca-95","pca-all"]
img_path="/Users/dianakim/2023_spring/Research/StyleMode_forGenerative_release1/generative_art/"
def main():
    #id cllct
    df=pd.read_csv("./fileinfo_csv/generative_w_info.csv",header=0)
    for comp in target:
        id_paintings=df[df[comp]==0.0]["painting"].values
        distances=df[df[comp]==0.0][comp+"-dist"].values
        id_sorted_painting=[x for _, x in sorted(zip(distances, id_paintings))]
        print("comp: ", comp, id_sorted_painting[:50])

        ood_paintings=df[df[comp]==1.0]["painting"].values
        distances=df[df[comp]==1.0][comp+"-dist"].values
        ood_sorted_painting=[x for _, x in sorted(zip(distances, ood_paintings),reverse=True)] #larger = oodness
        print("comp: ", comp, ood_sorted_painting[:50])
        

if __name__ == "__main__":
    main()
