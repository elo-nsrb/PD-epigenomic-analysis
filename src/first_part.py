#Import
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.stats import spearmanr
from matplotlib_venn import venn2

import os
import pickle
from itertools import product
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

#from pydeseq2.dds import DeseqDataSet
#from pydeseq2.ds import DeseqStats
#Definition of the color palette
COLORS_CT = {"NEU":"#75485E",
             "OLD":"#51A3A3","OPCs":"#C0F0F0",
             "MIC":"#CB904D",
             "AST":"#C3E991",
           "bulk":"#CCCCCC", 
        "CTRL":"#EBEBEB",
             "SPOR":"#AB4283",
             "GBA1":"#2D1E2F", 
             "LRRK2":"#F7B32B",
             "LRRK2-PD-":"#fce8bf",
             "LRRK2-PD+":"#F7B32B",
             "GBA1-PD+":"#2D1E2F",
             "GBA1-PD-":"#817882",
           "CAUD":"#12323B", 
             "SMTG":"#FE5F55",
            "MDFG":"#FFD6C0", 
             "HIPP":"#F0C20E", "PTMN":"#B4E1FF","SUNI":"#7A9B76", ##96C22B",
            "Male":"#AB8743","Female":"#45556B", }
palette= {"DOWN": "#F6E8C3", "UP":"#003C30"}
norm=plt.Normalize(-2,2)
tt = "UP"
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",palette[tt]])
cmap = sns.light_palette("#7A9B76", as_cmap=True)# diverging_palette(0, "#7A9B76", as_cmap=True)
cmap = sns.light_palette("#704C5E", as_cmap=True)# diverging_palette(0, "#7A9B76", as_cmap=True)
cmap = sns.light_palette("#45556B", as_cmap=True)# diverging_palette(0, "#7A9B76", as_cmap=True)

color_lib = {
    'GO_Biological_Process_2021':'#39B5E0',
     'GO_Molecular_Function_2021':'#A31ACB',
      'GO_Cellular_Component_2021':'#539165',
}
#Matplotlib parameters

DPI = 300
plt.rcParams['figure.dpi']= DPI

def plot_cond_and_sex(adata, outdir):
    # adata = sc.read_h5ad(atac_input_file_name)
    df = adata.obs.drop_duplicates('PatientID')
    df.rename(columns={'sex':'Sex','condition':'Condition'},inplace=True)
    df = df.groupby(["Sex", "Condition"])['PatientID'].nunique().reset_index()
    df.columns = ["Sex", "Condition", "Number of cases"]
    sns.set(font_scale=1.5, style="white")
    # plot = (so.Plot(data=df,x='Sex',color='Condition')
    #         .add(so.Bar(alpha=1),so.Hist(binwidth=1),so.Dodge(by=['color']))
    #         .scale(color=COLORS_CT)
    #         .theme({"axes.facecolor": "w", "axes.edgecolor": "black"})
    #         .label(title="Parkinson's disease cohort by sex and condition",y="Number of cases",fontsize=10)
    #         .save(loc=f"{outdir}/count_cohort.svg", dpi=DPI)
    #         )
    fig, ax = plt.subplots()
    ax.grid(axis="y")
    sns.barplot(data=df,x='Sex',y="Number of cases", hue='Condition',  palette=COLORS_CT, ax=ax)
    ax.set_title("Parkinson's disease cohort by sex and condition")
    ax.set_ylabel("Number of cases")
    ax.set_xlabel("")
    plt.savefig(f"{outdir}/count_cohort.svg", dpi=DPI)
    plt.show()
    
def plot_cond_and_region(adata, outdir):
    # adata = sc.read_h5ad(atac_input_file_name)
    df = adata.obs#.drop_duplicates('PatientID')
    df.rename(columns={'brain_region':'Brain regions','condition':'Condition'},inplace=True)
    df = df.groupby(['Brain regions', "Condition"])['PatientID'].nunique().reset_index()
    df.columns = ['Brain regions', "Condition", "Number of cases"]
    sns.set(font_scale=1.5, style="white")
    # plot = (so.Plot(data=df,x='Sex',color='Condition')
    #         .add(so.Bar(alpha=1),so.Hist(binwidth=1),so.Dodge(by=['color']))
    #         .scale(color=COLORS_CT)
    #         .theme({"axes.facecolor": "w", "axes.edgecolor": "black"})
    #         .label(title="Parkinson's disease cohort by sex and condition",y="Number of cases",fontsize=10)
    #         .save(loc=f"{outdir}/count_cohort.svg", dpi=DPI)
    #         )
    fig, ax = plt.subplots(figsize=(6,5))
    ax.grid(axis="y")
    sns.barplot(data=df,x='Brain regions',y="Number of cases", hue='Condition',  palette=COLORS_CT, ax=ax)
    ax.set_title("Parkinson's disease cohort by sex and condition")
    ax.set_ylabel("Number of cases")
    ax.set_xlabel("")
    plt.savefig(f"{outdir}/count_cohort_regions.svg", dpi=DPI)
    plt.show()

def plot_cond_and_age(adata, outdir):
    # adata = sc.read_h5ad(atac_input_file_name)
    
    boxdata = (adata
               .obs[['PatientID','Sex','condition','age at death']]
               .drop_duplicates(subset='PatientID')
               [['Sex','condition','age at death']]
               .rename(columns={'age at death':'Age of death','condition':'Condition'})
               )

    boxdata['Age of death'] = boxdata['Age of death'].astype(int)

    box_plot = sns.boxplot(data=boxdata,x='Condition',y='Age of death',orient='v', 
                           palette=COLORS_CT,
                           # order=['CTRL','GBA1','LRRK2','SPOR'], 
                           order=['GBA1','LRRK2','SPOR'], 
                           boxprops=dict(alpha=1))
    box_plot.set(xlabel='Condition', ylabel='Age of death')
    
    plt.title("Age of death by condition",fontsize=10)

    #Compute non-parametric Wilcoxon test between conditions

    pairs=[("GBA1/PD+","SPOR"),("SPOR","CTRL"),("GBA1_PD+","CTRL"),("CTRL","LRRK2"),("LRRK2","GBA1_PD+"),("LRRK2","SPOR")]
    pairs=[("GBA1","SPOR"),("SPOR","CTRL"),("GBA1","CTRL"),("CTRL","LRRK2"),("LRRK2","GBA1"),("LRRK2","SPOR")]
    pairs=[("GBA1","SPOR"),("LRRK2","GBA1"),("LRRK2","SPOR")]

    annotator= Annotator(box_plot,pairs,data=boxdata,x='Condition',y='Age of death', 
                         order=['GBA1','LRRK2','SPOR'])
                         # order=['CTRL','GBA1','LRRK2','SPOR'])
    annotator.configure(test='Mann-Whitney', text_format='star',comparisons_correction='Benjamini-Hochberg')
    annotator.apply_and_annotate()
    
    
    plt.savefig(f"{outdir}/count_age.svg", dpi=DPI)
    plt.show()
def plot_umap_celltype_specific_ct(adata,outdir, nn=10):
    
    adata = adata[adata.obs["celltype"] != "bulk",:]
    sc.pp.neighbors(adata,n_neighbors=nn)
    sc.pp.pca(adata)
    sc.tl.umap(adata)
    umap = adata.obsm["X_umap"]
    adata.obs = adata.obs.rename({"celltype":"Cell-type"},axis=1)
    sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=adata.obs["Cell-type"],palette=COLORS_CT, s=10)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    ax = plt.gca()

    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(f"{outdir}/umap_celltype.svg",dpi=DPI)
    plt.show()

def plot_umap_celltype_specific_ct_from_path(atac_input_file_name,outdir,nn=10):
    
    adata =  sc.read_h5ad(atac_input_file_name)
    adata = adata[adata.obs["celltype"] != "bulk",:]
    sc.pp.neighbors(adata,n_neighbors=nn)
    sc.pp.pca(adata)
    sc.tl.umap(adata)
    umap = adata.obsm["X_umap"]
    adata.obs = adata.obs.rename({"celltype":"Cell-type"},axis=1)
    sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=adata.obs["Cell-type"],palette=COLORS_CT, s=10)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    ax = plt.gca()

    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(f"{outdir}/umap_celltype.svg",dpi=DPI)
    plt.show()
def plot_umap_celltype_specific_ct_covariate_from_path(atac_input_file_name, covariate, outdir, nn=10):
    
    adata =  sc.read_h5ad(atac_input_file_name)
    adata = adata[adata.obs["celltype"] != "bulk",:]
    sc.pp.neighbors(adata,n_neighbors=nn)
    sc.pp.pca(adata)
    sc.tl.umap(adata)
    umap = adata.obsm["X_umap"]
    adata.obs = adata.obs.rename({"celltype":"Cell-type"},axis=1)
    adata.obs = adata.obs.rename({"xxx.expired_age":"Age","sex":"Sex"},axis=1)
    palette_cov = {"Male":"#28587B","Female":"#7F7CAF"}
    if covariate == "Age":
        sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=adata.obs[covariate], s=10)
    else:
        sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=adata.obs[covariate],palette=palette_cov, s=10)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    ax = plt.gca()

    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])
   
    plt.savefig(f"{outdir}/umap_celltype.svg",dpi=DPI)
    plt.show()

def plot_umap_celltype_specific_ct_covariate(adata, covariate, outdir,nn=10):
    
    adata = adata[adata.obs["celltype"] != "bulk",:]
    sc.pp.neighbors(adata,n_neighbors=nn)
    sc.pp.pca(adata)
    sc.tl.umap(adata)
    umap = adata.obsm["X_umap"]
    adata.obs = adata.obs.rename({"celltype":"Cell-type"},axis=1)
    adata.obs = adata.obs.rename({"xxx.expired_age":"Age","sex":"Sex"},axis=1)
    palette_cov = {"Male":"#28587B","Female":"#7F7CAF"}
   
    if covariate=="Sex" :
        sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=adata.obs[covariate],palette=palette_cov, s=10)
    else:
        sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=adata.obs[covariate], s=10)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    ax = plt.gca()

    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])
   
    plt.savefig(f"{outdir}/umap_celltype.svg",dpi=DPI)
    plt.show()
def plot_umap_celltype_specific_br(adata,outdir,nn=10):
    
    adata = adata[adata.obs["celltype"] != "bulk",:]
    sc.pp.neighbors(adata,n_neighbors=nn)
    sc.pp.pca(adata)
    sc.tl.umap(adata)
    umap = adata.obsm["X_umap"]
    adata.obs = adata.obs.rename({"brain_region":"Brain region"},axis=1)
    sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=adata.obs["Brain region"],palette=COLORS_CT, s=10)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    ax = plt.gca()

    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])

    
    plt.savefig(f"{outdir}/umap_brain_region.svg",dpi=DPI)
    plt.show()
    
def plot_umap_celltype_specific_br_from_path(atac_input_file_name,outdir):
    
    adata =  sc.read_h5ad(atac_input_file_name)
    adata = adata[adata.obs["celltype"] != "bulk",:]
    sc.pp.neighbors(adata,n_neighbors=nn)
    sc.pp.pca(adata)
    sc.tl.umap(adata)
    umap = adata.obsm["X_umap"]
    adata.obs = adata.obs.rename({"brain_region":"Brain region"},axis=1)
    sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=adata.obs["Brain region"],palette=COLORS_CT, s=10)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    ax = plt.gca()

    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])

    
    plt.savefig(f"{outdir}/umap_brain_region.svg",dpi=DPI)
    plt.show()
def plot_all_umap(adata,outdir, nn=10):

    adata = adata[adata.obs["celltype"] != "bulk",:]
    sc.pp.neighbors(adata,n_neighbors=nn)
    sc.pp.pca(adata)
    sc.tl.umap(adata)
    umap = adata.obsm["X_umap"]
    adata.obs = adata.obs.rename({"brain_region":"Brain Region",
                                  "celltype":"Cell-Type",
                                  "xxx.expired_age":"Age",
                                  "sex":"Sex",
                                  "condition":"Condition"},axis=1)

    fig,ax = plt.subplots(2,2,figsize=(10,10))
    dict_ax = {(0,0):adata.obs["Cell-Type"],
               (0,1):adata.obs["Brain Region"],
               (1,0):adata.obs["Sex"],
               (1,1):adata.obs["Condition"]}
    for i in range(2):
        for j in range(2):


            sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=dict_ax[(i,j)],palette=COLORS_CT,ax=ax[i][j], s=10)
            
                
            ax[i][j].set_xlabel("UMAP 1")
            ax[i][j].set_ylabel("UMAP 2")
            # Hide X and Y axes label marks
            ax[i][j].xaxis.set_tick_params(labelbottom=False)
            ax[i][j].yaxis.set_tick_params(labelleft=False)

            # Hide X and Y axes tick marks
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

    plt.savefig(f"{outdir}/all_umap.svg",dpi=DPI)
    plt.show()
def plot_all_umap_from_path(atac_input_file_name,outdir,nn=10):
    adata =  sc.read_h5ad(atac_input_file_name)
    adata = adata[adata.obs["celltype"] != "bulk",:]
    sc.pp.neighbors(adata,n_neighbors=nn)
    sc.tl.umap(adata, linewidth=0)
    umap = adata.obsm["X_umap"]
    adata.obs = adata.obs.rename({"brain_region":"Brain Region",
                                  "celltype":"Cell-Type",
                                  "xxx.expired_age":"Age",
                                  "sex":"Sex",
                                  "condition":"Condition"},axis=1)

    fig,ax = plt.subplots(2,2,figsize=(10,10))
    dict_ax = {(0,0):adata.obs["Cell-Type"],
               (0,1):adata.obs["Brain Region"],
               (1,0):adata.obs["Sex"],
               (1,1):adata.obs["Condition"]}
    for i in range(2):
        for j in range(2):

            sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=dict_ax[(i,j)],palette=COLORS_CT,ax=ax[i][j], s=10, linewidth=0)
                
            ax[i][j].set_xlabel("UMAP 1")
            ax[i][j].set_ylabel("UMAP 2")
            # Hide X and Y axes label marks
            ax[i][j].xaxis.set_tick_params(labelbottom=False)
            ax[i][j].yaxis.set_tick_params(labelleft=False)

            # Hide X and Y axes tick marks
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

    plt.savefig(f"{outdir}/all_umap.svg",dpi=DPI)
    plt.show()
def compute_multitesting(input_df):
    
    df = input_df.copy()
    df = (df
          .dropna(subset=["pvalue"]))
    list_ct, list_br, list_comp = list(df["ct"].unique()), list(df["br"].unique()), list(df["comparison"].unique())
    
    for (comp,br,ct) in tqdm(product(list_comp,list_br, list_ct)):

        pvals = df.loc[(df.ct == ct) & (df.br == br) & (df.comparison == comp), "pvalue"] 
        
        df.loc[(df.ct == ct) & (df.br == br) & (df.comparison == comp), "padj"] = multipletests(pvals=pvals, alpha=0.05, method="fdr_bh")[1]
    #Identify significant DE
    
    df["significant"] = False

    df.loc[(np.abs(df["log2FoldChange"])>=0) & (df["padj"]<=0.05),"significant"] = True

    return df



def plot_DE(res_DE,outdir,outname,more_title="", cmap='RdPu', FC=0):
    
    count = res_DE.loc[res_DE.significant==True,["padj","downreg","log2FoldChange","comparison","br","ct"]].groupby(["br","ct","downreg","comparison"]).count()["log2FoldChange"].reset_index()
    total_number = res_DE.loc[:,["padj","downreg","log2FoldChange","comparison","br","ct"]].groupby(["br","ct","comparison"]).count()["log2FoldChange"].reset_index()
    count = count.merge(total_number.rename({"log2FoldChange":"total_num"},axis=1),on=["br","ct","comparison"])
    count["log2FoldChange"] = (count
                            ["log2FoldChange"]
                            .div(count["total_num"])
                            )
    count = count.drop(["total_num"],axis=1)
    
    t = count.set_index(["br","ct","downreg","comparison"])

    all_possibilities = [(br,ct,downreg,comparison)
                      for (br,ct,downreg,comparison) in product(res_DE["br"].unique(),res_DE["ct"].unique(),res_DE["downreg"].unique(),res_DE["comparison"].unique()) ]
    
    for pos in all_possibilities:
        if pos not in t.index:
            t.loc[pos] = 0
    
    count = t.reset_index()
    fig,ax = plt.subplots(len(count.comparison.unique()),2,figsize=(8,10))
    vmin,vmax = 0, 7

    for comp in count.comparison.unique():
        pivot_table_down = count[(count.downreg==True) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_down.columns = pivot_table_down.columns.droplevel(0)
        print(pivot_table_down.columns)
        
        pivot_table_up = count[(count.downreg==False) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_up.columns = pivot_table_up.columns.droplevel(0)
        
        pivot_table_down = pivot_table_down.fillna(0)
        pivot_table_up = pivot_table_up.fillna(0)
        # Determine the common range for the color bar
        vmin = min(pivot_table_down.values.min(), pivot_table_up.values.min(),vmin)
        vmax = max(pivot_table_down.values.max(), pivot_table_up.values.max(),vmax)
    
    vmin,vmax = 0, .07
    for i,comp in enumerate(count.comparison.unique()):
        #if "res_DE_ct_sex" in res_DE_file_name:
        #    count = pd.concat([count,pd.DataFrame(["PTMN",False,"SPOR$GBA1",0]),pd.DataFrame(["CAUD",False,"SPOR$GBA1",0])])

        
        pivot_table_down = count[(count.downreg==True) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_down.columns = pivot_table_down.columns.droplevel(0)
        print(pivot_table_down.columns)
        
        pivot_table_up = count[(count.downreg==False) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_up.columns = pivot_table_up.columns.droplevel(0)
        
        pivot_table_down = pivot_table_down.fillna(0)
        pivot_table_up = pivot_table_up.fillna(0)
        
        
        # cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["#FAF3DD","#F9DC5C","#FFBA08","#D8315B","#770058"])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb_ax = fig.add_axes([1.01,.124,.02,.769])
        cb1 = mpl.colorbar.ColorbarBase(cb_ax,
                                        cmap=cmap,
                                        norm=norm,
                                        orientation='vertical',
                                        
                                        )
        
        cb1.ax.yaxis.set_major_formatter("{x:.1%}")
        
        sns.heatmap(pivot_table_down, annot=True, cmap=cmap,vmin=vmin, vmax=vmax,cbar=False,ax=ax[i][0],fmt=".0%",annot_kws={"c":"black"})
        
        sns.heatmap(pivot_table_up, annot=True,ax=ax[i][1],cmap=cmap,vmin=vmin, vmax=vmax,cbar=False,fmt=".0%",annot_kws={"c":"black"})
        #ax[i][1].set_yticks([])
        ax[i][0].tick_params(axis=u'both', which=u'both',length=0)
        ax[i][1].tick_params(axis=u'both', which=u'both',length=0)
        
        cond0, cond1 = comp.split("$")
        ax[i][1].set_title(f"{more_title}Up-Accessible OCRs - {cond0} vs. {cond1}", size=10)
        print(ax[i][0].get_xticklabels())
        
        ax[i][0].set_title(f"{more_title}Down-Accessible OCRs - {cond0} vs. {cond1}", size=10)
        ax[i][0].set_ylabel("")
        ax[i][1].set_ylabel("")
        ax[i][0].set_xlabel("Brain Region",size=10)
        ax[i][1].set_xlabel("Brain Region",size=10)

        ax[i][0].set_ylabel("Cell-Type",size=10)
        ax[i][1].set_ylabel("Cell-Type",size=10)
        ax[i][0].patch.set_edgecolor('black')  
        ax[i][0].patch.set_linewidth(1)
        ax[i][1].patch.set_edgecolor('black')  
        ax[i][1].patch.set_linewidth(1)  
        # Add a single color bar
        [t.set_visible(True) for t in ax[i][1].get_yticklabels()]
        
    plt.tight_layout()
    plt.savefig(f"{outdir}/res_DE_{outname}.svg",dpi=DPI)
    plt.show()

def plot_DE_sex(res_DE,outdir,outname,more_title="",cmap='RdPu'):
    
    count = res_DE.loc[res_DE.significant==True,["padj","downreg","log2FoldChange","comparison","br","ct"]].groupby(["br","ct","downreg","comparison"]).count()["log2FoldChange"].reset_index()
    total_number = res_DE.loc[:,["padj","downreg","log2FoldChange","comparison","br","ct"]].groupby(["br","ct","comparison"]).count()["log2FoldChange"].reset_index()
    count = count.merge(total_number.rename({"log2FoldChange":"total_num"},axis=1),on=["br","ct","comparison"])
    count["log2FoldChange"] = (count
                            ["log2FoldChange"]
                            .div(count["total_num"])
                            )
    count = count.drop(["total_num"],axis=1)
    
    t = count.set_index(["br","ct","downreg","comparison"])

    all_possibilities = [(br,ct,downreg,comparison)
                      for (br,ct,downreg,comparison) in product(res_DE["br"].unique(),res_DE["ct"].unique(),res_DE["downreg"].unique(),res_DE["comparison"].unique()) ]
    
    for pos in all_possibilities:
        if pos not in t.index:
            t.loc[pos] = 0
    
    count = t.reset_index()
    fig,ax = plt.subplots(len(count.comparison.unique()),2,figsize=(6,4))
    vmin,vmax = float("inf"), -float("inf")

    for comp in count.comparison.unique():
        pivot_table_down = count[(count.downreg==True) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_down.columns = pivot_table_down.columns.droplevel(0)
        print(pivot_table_down.columns)
        
        pivot_table_up = count[(count.downreg==False) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_up.columns = pivot_table_up.columns.droplevel(0)
        
        pivot_table_down = pivot_table_down.fillna(0)
        pivot_table_up = pivot_table_up.fillna(0)
        # Determine the common range for the color bar
        vmin = min(pivot_table_down.values.min(), pivot_table_up.values.min(),vmin)
        vmax = max(pivot_table_down.values.max(), pivot_table_up.values.max(),vmax)
    vmin,vmax = 0,.03
    for i,comp in enumerate(count.comparison.unique()):
        #if "res_DE_ct_sex" in res_DE_file_name:
        #    count = pd.concat([count,pd.DataFrame(["PTMN",False,"SPOR$GBA1",0]),pd.DataFrame(["CAUD",False,"SPOR$GBA1",0])])

        
        pivot_table_down = count[(count.downreg==True) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_down.columns = pivot_table_down.columns.droplevel(0)
        print(pivot_table_down.columns)
        
        pivot_table_up = count[(count.downreg==False) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_up.columns = pivot_table_up.columns.droplevel(0)
        
        pivot_table_down = pivot_table_down.fillna(0)
        pivot_table_up = pivot_table_up.fillna(0)
        
        
        # cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["#FAF3DD","#F9DC5C","#FFBA08","#D8315B","#770058"])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb_ax = fig.add_axes([1.01,.124,.02,.769])
        cb1 = mpl.colorbar.ColorbarBase(cb_ax,
                                        cmap=cmap,
                                        norm=norm,
                                        orientation='vertical',
                                        
                                        )
        
        cb1.ax.yaxis.set_major_formatter("{x:.1%}")
        
        sns.heatmap(pivot_table_down, annot=True, cmap=cmap,vmin=vmin, vmax=vmax,cbar=False,ax=ax[0],fmt=".0%",annot_kws={"c":"black"})
        
        sns.heatmap(pivot_table_up, annot=True,ax=ax[1],cmap=cmap,vmin=vmin, vmax=vmax,cbar=False,fmt=".0%",annot_kws={"c":"black"})
        #ax[i][1].set_yticks([])
        ax[0].tick_params(axis=u'both', which=u'both',length=0)
        ax[1].tick_params(axis=u'both', which=u'both',length=0)
        
        cond0, cond1 = comp.split("$")
        ax[1].set_title(f"{more_title}Up-Accessible OCRs - {cond0} vs. {cond1}", size=6)
        print(ax[0].get_xticklabels())
        
        ax[0].set_title(f"{more_title}Down-Accessible OCRs - {cond0} vs. {cond1}", size=6)
        ax[0].set_ylabel("")
        ax[1].set_ylabel("")
        ax[0].set_xlabel("Brain Region",size=10)
        ax[1].set_xlabel("Brain Region",size=10)

        ax[0].set_ylabel("Cell-Type",size=10)
        ax[1].set_ylabel("Cell-Type",size=10)
        ax[0].patch.set_edgecolor('black')  
        ax[0].patch.set_linewidth(1)
        ax[1].patch.set_edgecolor('black')  
        ax[1].patch.set_linewidth(1)  
        [t.set_visible(True) for t in ax[1].get_yticklabels()]
        # Add a single color bar
        
    plt.tight_layout()
    plt.savefig(f"{outdir}/res_DE_{outname}.svg",dpi=DPI)
    plt.show()

def plot_DE_number(res_DE,outdir,outname,more_title="",cmap='RdPu'):
    
    count = res_DE.loc[res_DE.significant==True,["padj","downreg","log2FoldChange","comparison","br","ct"]].groupby(["br","ct","downreg","comparison"]).count()["log2FoldChange"].reset_index()
    total_number = res_DE.loc[:,["padj","downreg","log2FoldChange","comparison","br","ct"]].groupby(["br","ct","comparison"]).count()["log2FoldChange"].reset_index()
    count = count.merge(total_number.rename({"log2FoldChange":"total_num"},axis=1),on=["br","ct","comparison"])
    count["log2FoldChange"] = (count
                            ["log2FoldChange"]
                            )
    count = count.drop(["total_num"],axis=1)
    
    t = count.set_index(["br","ct","downreg","comparison"])

    all_possibilities = [(br,ct,downreg,comparison)
                      for (br,ct,downreg,comparison) in product(res_DE["br"].unique(),res_DE["ct"].unique(),res_DE["downreg"].unique(),res_DE["comparison"].unique()) ]
    
    for pos in all_possibilities:
        if pos not in t.index:
            t.loc[pos] = 0
    
    count = t.reset_index()
    
    fig,ax = plt.subplots(len(count.comparison.unique()),2,figsize=(8,10))
    vmin,vmax = float("inf"), -float("inf")

    for comp in count.comparison.unique():
        pivot_table_down = count[(count.downreg==True) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_down.columns = pivot_table_down.columns.droplevel(0)
        print(pivot_table_down.columns)
        
        pivot_table_up = count[(count.downreg==False) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_up.columns = pivot_table_up.columns.droplevel(0)
        
        pivot_table_down = pivot_table_down.fillna(0)
        pivot_table_up = pivot_table_up.fillna(0)
        # Determine the common range for the color bar
        vmin = min(pivot_table_down.values.min(), pivot_table_up.values.min(),vmin)
        vmax = max(pivot_table_down.values.max(), pivot_table_up.values.max(),vmax)
    vmin,vmax = 0, 1200
    for i,comp in enumerate(count.comparison.unique()):
        #if "res_DE_ct_sex" in res_DE_file_name:
        #    count = pd.concat([count,pd.DataFrame(["PTMN",False,"SPOR$GBA1",0]),pd.DataFrame(["CAUD",False,"SPOR$GBA1",0])])

        
        pivot_table_down = count[(count.downreg==True) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_down.columns = pivot_table_down.columns.droplevel(0)
        print(pivot_table_down.columns)
        
        pivot_table_up = count[(count.downreg==False) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_up.columns = pivot_table_up.columns.droplevel(0)
        
        pivot_table_down = pivot_table_down.fillna(0)
        pivot_table_up = pivot_table_up.fillna(0)
        
        # cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["#FAF3DD","#F9DC5C","#FFBA08","#D8315B","#770058"])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb_ax = fig.add_axes([1.01,.124,.02,.769])
        cb1 = mpl.colorbar.ColorbarBase(cb_ax,
                                        cmap=cmap,
                                        norm=norm,
                                        orientation='vertical',
                                        
                                        )
        
        #cb1.ax.yaxis.set_major_formatter("{x:.1%}")
        
        sns.heatmap(pivot_table_down, annot=True, cmap=cmap,vmin=vmin, vmax=vmax,cbar=False,ax=ax[i][0],fmt='g',annot_kws={"c":"black"})
        
        sns.heatmap(pivot_table_up, annot=True,ax=ax[i][1],cmap=cmap,vmin=vmin, vmax=vmax,cbar=False,fmt='g',annot_kws={"c":"black"})
        #ax[i][1].set_yticks([])
        ax[i][0].tick_params(axis=u'both', which=u'both',length=0)
        ax[i][1].tick_params(axis=u'both', which=u'both',length=0)
        cond0, cond1 = comp.split("$")
        ax[i][1].set_title(f"{more_title}Up-Accessible OCRs - {cond0} vs. {cond1}",size=10)
        ax[i][0].set_xlabel("Brain Region",size=10)
        ax[i][1].set_xlabel("Brain Region",size=10)
        ax[i][0].set_title(f"{more_title}Down-Accessible OCRs - {cond0} vs. {cond1}", size=10)
        ax[i][0].set_ylabel("Cell-Type",size=10)
        ax[i][1].set_ylabel("Cell-Type",size=10)
        ax[i][0].patch.set_edgecolor('black')  
        ax[i][0].patch.set_linewidth(1)
        ax[i][1].patch.set_edgecolor('black')  
        ax[i][1].patch.set_linewidth(1)  

        [t.set_visible(True) for t in ax[i][1].get_yticklabels()]
    # Add a single color bar
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/res_DE_{outname}.svg",dpi=DPI)
    plt.show()

def plot_DE_number_sex(res_DE,outdir,outname,more_title="",cmap='RdPu'):
    
    count = res_DE.loc[res_DE.significant==True,["padj","downreg","log2FoldChange","comparison","br","ct"]].groupby(["br","ct","downreg","comparison"]).count()["log2FoldChange"].reset_index()
    total_number = res_DE.loc[:,["padj","downreg","log2FoldChange","comparison","br","ct"]].groupby(["br","ct","comparison"]).count()["log2FoldChange"].reset_index()
    count = count.merge(total_number.rename({"log2FoldChange":"total_num"},axis=1),on=["br","ct","comparison"])
    count["log2FoldChange"] = (count
                            ["log2FoldChange"]
                            )
    count = count.drop(["total_num"],axis=1)
    
    t = count.set_index(["br","ct","downreg","comparison"])

    all_possibilities = [(br,ct,downreg,comparison)
                      for (br,ct,downreg,comparison) in product(res_DE["br"].unique(),res_DE["ct"].unique(),res_DE["downreg"].unique(),res_DE["comparison"].unique()) ]
    
    for pos in all_possibilities:
        if pos not in t.index:
            t.loc[pos] = 0
    
    count = t.reset_index()
    
    fig,ax = plt.subplots(len(count.comparison.unique()),2,figsize=(6,4))
    vmin,vmax = float("inf"), -float("inf")

    for comp in count.comparison.unique():
        pivot_table_down = count[(count.downreg==True) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_down.columns = pivot_table_down.columns.droplevel(0)
        print(pivot_table_down.columns)
        
        pivot_table_up = count[(count.downreg==False) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_up.columns = pivot_table_up.columns.droplevel(0)
        
        pivot_table_down = pivot_table_down.fillna(0)
        pivot_table_up = pivot_table_up.fillna(0)
        # Determine the common range for the color bar
        vmin = min(pivot_table_down.values.min(), pivot_table_up.values.min(),vmin)
        vmax = max(pivot_table_down.values.max(), pivot_table_up.values.max(),vmax)
    vmin, vmax = 0, 500
    for i,comp in enumerate(count.comparison.unique()):
        #if "res_DE_ct_sex" in res_DE_file_name:
        #    count = pd.concat([count,pd.DataFrame(["PTMN",False,"SPOR$GBA1",0]),pd.DataFrame(["CAUD",False,"SPOR$GBA1",0])])

        
        pivot_table_down = count[(count.downreg==True) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_down.columns = pivot_table_down.columns.droplevel(0)
        print(pivot_table_down.columns)
        
        pivot_table_up = count[(count.downreg==False) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({"log2FoldChange":""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_up.columns = pivot_table_up.columns.droplevel(0)
        
        pivot_table_down = pivot_table_down.fillna(0)
        pivot_table_up = pivot_table_up.fillna(0)
        
        # cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["#FAF3DD","#F9DC5C","#FFBA08","#D8315B","#770058"])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb_ax = fig.add_axes([1.01,.124,.02,.769])
        cb1 = mpl.colorbar.ColorbarBase(cb_ax,
                                        cmap=cmap,
                                        norm=norm,
                                        orientation='vertical',
                                        
                                        )
        
        #cb1.ax.yaxis.set_major_formatter("{x:.1%}")
        
        sns.heatmap(pivot_table_down, annot=True, cmap=cmap,vmin=vmin, vmax=vmax,cbar=False,ax=ax[0],fmt='g',annot_kws={"c":"black"})
        
        sns.heatmap(pivot_table_up, annot=True,ax=ax[1],cmap=cmap,vmin=vmin, vmax=vmax,cbar=False,fmt='g',annot_kws={"c":"black"})
        ax[1].set_yticks(["AST","EXC","INH","MIC","OLD","OPCs"])
        ax[0].tick_params(axis=u'both', which=u'both',length=0)
        ax[1].tick_params(axis=u'both', which=u'both',length=0)
        cond0, cond1 = comp.split("$")
        ax[1].set_title(f"{more_title}Up-Accessible OCRs - {cond0} vs. {cond1}",size=6)
        ax[0].set_xlabel("Brain Region",size=10)
        ax[1].set_xlabel("Brain Region",size=10)
        ax[0].set_title(f"{more_title}Down-Accessible OCRs - {cond0} vs. {cond1}", size=6)
        ax[0].set_ylabel("Cell-Type",size=10)
        ax[1].set_ylabel("Cell-Type",size=10)
        ax[0].patch.set_edgecolor('black')  
        ax[0].patch.set_linewidth(1)
        ax[1].patch.set_edgecolor('black')  
        ax[1].patch.set_linewidth(1)  

        
    # Add a single color bar
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/res_DE_{outname}.svg",dpi=DPI)
    plt.show()


import gseapy as gp
# TO ADD GENE ONTOLOGY / HOMER
def get_GO(de_res):

    #from gseapy import barplot, dotplot
    lib_name = ['GO_Biological_Process_2021', 'GO_Molecular_Function_2021', 'GO_Cellular_Component_2021']
    list_lib = []
    comparisons = ['SPOR$CTRL','SPOR$GBA1']
    for lib in lib_name:
        list_df = []
        
        for comp in comparisons:
            for ct in de_res["ct"].unique():
                for br in de_res["br"].unique():
                    background = list(de_res.loc[(de_res["ct"]==ct) & (de_res["br"]==br) & (de_res["comparison"]==comp),"nearestGeneChip"].unique())
                    for downreg in [True,False]: 
                        
                        de_down = list(de_res.loc[(de_res["ct"]==ct) & (de_res["br"]==br) & (de_res["comparison"]==comp) & (de_res["downreg"]==downreg) & (de_res["significant"]==True),"nearestGeneChip"].unique())
                    
                
                        pre_res =gp.enrichr(de_down, gene_sets=lib, organism='Human',
                                        background=background)
                    
                        res_df = pre_res.res2d
                        res_df['celltype'] = ct
                        res_df['brain_region'] = br
                        res_df['comparison'] = comp
                        res_df['downreg'] = downreg
                        
                        list_df.append(res_df)

        list_lib.append(pd.concat(list_df))

    df_go = pd.concat(list_lib) 
    return df_go

def dotplot(df, cutoff_pval=0.05, cutoff_overlap=0.1,term_num=42, figsize=(10,20), scale=50, comparison="", title=""):
    """Visualize enrichr or gsea results.
    
    :param df: GSEApy DataFrame results. 
    :param cutoff: p-adjust cut-off. 
    :param term_num: number of enriched terms to show.
    :param scale: dotplot point size scale.
    :return:  a dotplot for enrichr terms. 
    """
    #enrichr results
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # pvalue cut off
    df = df[(df['Adjusted P-value'] <= cutoff_pval) & (df['hits_ratio']>=cutoff_overlap)]
    print(len(df))
    if len(df) < 1:
        print("Warning: No enrich terms when cuttoff = %s"%cutoff_pval )
        return None
    #sorting the dataframe for better visualization
    df = df.sort_values(by=['Adjusted P-value'], ascending=True)
    df = df.head(term_num)
    df = df.sort_values(by=['Gene_set','Combined Score'],ascending=[True,False])
    #ylabels = df['Term'].str.split('$').str[0].to_list()
    #print('ylab',len(ylabels))
    #x axis values
    x = [i for i in range(0,len(df.celltype))]
    x_labels = df['celltype'].values

    # y axis index and values
    #y=  [i for i in range(0,len(df))]
    print(len(df))
    labels = df.Term.values
    #str.split('(GO')[0]
    area = np.pi * (df['Count'] *scale)**2
    print('x:',len(x))
    #print('y:',len(y))
    print('area:',len(area))
    #create scatter plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sc = sns.scatterplot(
                   x="celltype", 
                   y=df["Term"],
                   size=df["Count"],
                   sizes = (50,900),
                   hue='-log10(q-val)',
                   data=df,
                   legend=True,
                   style="brain_region",
                   axes=ax,
                   markers=['x']) #'s','X','o'
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    #ax.yaxis.set_major_locator(plt.FixedLocator(y))
    #ax.yaxis.set_major_formatter(plt.FixedFormatter(labels))
    ax.set_ylim([-1, len(df)])
    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)  
       
    #colorbar

    norm = plt.Normalize(df['-log10(q-val)'].min(), df['-log10(q-val)'].max())
    sm = plt.cm.ScalarMappable( norm=norm, cmap="gnuplot2") #cmap="RdBu"
    sm.set_array([])
    sc.figure.colorbar(sm, shrink=0.5, pad=0.03, location='bottom').set_label("-log$_{10}$(FDR)")
    
    #color celltype
    for label in ax.get_xticklabels():
      label.set(color=COLORS_CT[label.get_text()])
    #color lib name
    for label in ax.get_yticklabels():
      label.set(color=color_lib[label.get_text().split('$')[1]],fontsize=14)
    ax.set(ylabel=None, xlabel=None,yticklabels=df['Term'].str.split('$').str[0].unique())
    print("axlen",len(ax.get_xticklabels())) #
    print("lablen",len(df['Term'].str.split('$').str[0].to_list()))
    '''ax.legend({
        'Biological process': color_lib['GO_Biological_Process_2021'],
        'Molecular function': color_lib['GO_Molecular_Function_2021'],
        'Cellular component': color_lib['GO_Cellular_Component_2021'],
    })'''
    #turn off all spines and ticks
    #ax2.axis('off')
    '''t = ax.text(-2.4, 20, "Combined score",
            ha="center", va="center", rotation=90, size=10,
            bbox=dict(boxstyle="rarrow,pad=0.3",
                      fc="lightblue", ec="steelblue", lw=2))
    '''
    #plt.tight_layout()
    #canvas.print_figure('test', bbox_inches='tight')   
    plt.suptitle(f"{title}",fontsize=20, y=0.90) 
    #plt.savefig(f"../Plots/enrichr_OCR_GBA1_vs_CTRL_promoters.jpg")
    plt.show()


color_lib = {
    'GO_Biological_Process_2021':'#39B5E0',
     'GO_Molecular_Function_2021':'#A31ACB',
      'GO_Cellular_Component_2021':'#539165',
}


def dotplot(df,outdir,outname, cutoff_pval=0.05, cutoff_overlap=0.,term_num=42, figsize=(10,20), scale=50, comparison="", title=""):
    """Visualize enrichr or gsea results.
    
    :param df: GSEApy DataFrame results. 
    :param cutoff: p-adjust cut-off. 
    :param term_num: number of enriched terms to show.
    :param scale: dotplot point size scale.
    :return:  a dotplot for enrichr terms. 
    """
    #enrichr results
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    df['Term'] = df['Term'].apply(lambda x: x.split('(')[0])
    df['Term'] = df['Term'] + '$' +df['Gene_set']
    df['-log10(q-val)'] = -np.log10(df["Adjusted P-value"])
    # pvalue cut off
    df = df[(df['Adjusted P-value'] <= cutoff_pval) & (df["Odds Ratio"]>=cutoff_overlap)]
    print(len(df))
    if len(df) < 1:
        print("Warning: No enrich terms when cuttoff = %s"%cutoff_pval )
        return None
    #sorting the dataframe for better visualization
    df = df.sort_values(by=['Adjusted P-value'], ascending=True)
    df = df.head(term_num)
    df = df.sort_values(by=['Gene_set','Combined Score'],ascending=[True,False])
    #ylabels = df['Term'].str.split('$').str[0].to_list()
    #print('ylab',len(ylabels))
    #x axis values
    x = [i for i in range(0,len(df.celltype))]
    x_labels = df['celltype'].values

    # y axis index and values
    #y=  [i for i in range(0,len(df))]
    print(len(df))
    labels = df.Term.values
    #str.split('(GO')[0]
    #area = np.pi * (df['Count'] *scale)**2
    print('x:',len(x))
    #print('y:',len(y))
    #print('area:',len(area))
    #create scatter plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sc = sns.scatterplot(
                   x="celltype", 
                   y=df["Term"],
                   #size=df["Count"],
                   sizes = (50,900),
                   hue='-log10(q-val)',
                   data=df,
                   legend=True,
                   style="brain_region",
                   axes=ax,
                   markers=['x']) #'s','X','o'
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    #ax.yaxis.set_major_locator(plt.FixedLocator(y))
    #ax.yaxis.set_major_formatter(plt.FixedFormatter(labels))
    ax.set_ylim([-1, len(df)])
    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)  
       
    #colorbar

    norm = plt.Normalize(df['-log10(q-val)'].min(), df['-log10(q-val)'].max())
    sm = plt.cm.ScalarMappable( norm=norm, cmap="gnuplot2") #cmap="RdBu"
    sm.set_array([])
    sc.figure.colorbar(sm, shrink=0.5, pad=0.03, location='bottom').set_label("-log$_{10}$(FDR)")
    
    #color celltype
    for label in ax.get_xticklabels():
      label.set(color=COLORS_CT[label.get_text()])
    #color lib name
    for label in ax.get_yticklabels():
      label.set(color=color_lib[label.get_text().split('$')[1]],fontsize=14)
    ax.set(ylabel=None, xlabel=None,yticklabels=df['Term'].str.split('$').str[0].unique())
    print("axlen",len(ax.get_xticklabels())) #
    print("lablen",len(df['Term'].str.split('$').str[0].to_list()))
    '''ax.legend({
        'Biological process': color_lib['GO_Biological_Process_2021'],
        'Molecular function': color_lib['GO_Molecular_Function_2021'],
        'Cellular component': color_lib['GO_Cellular_Component_2021'],
    })'''
    #turn off all spines and ticks
    #ax2.axis('off')
    '''t = ax.text(-2.4, 20, "Combined score",
            ha="center", va="center", rotation=90, size=10,
            bbox=dict(boxstyle="rarrow,pad=0.3",
                      fc="lightblue", ec="steelblue", lw=2))
    '''
    #plt.tight_layout()
    #canvas.print_figure('test', bbox_inches='tight')   
    plt.suptitle(f"{title}",fontsize=20, y=0.90) 
    plt.savefig(f"{outdir}/gene_ontology_{outname}.svg",dpi=DPI)
    plt.show()



pval = 0.05
logFC = 0
backgrounds = {}
# adata = sc.read_h5ad("/Users/raphaelzaghroun/Downloads/data_PD/adata/ct/adata_ct.h5ad")

def differential_expression_age(adata):
    
    #Data cleaning 
    adata.obs.dropna(axis=1, how="all", inplace=True) # drop columns with all NAN's
    adata2 = adata[(adata.obs.celltype != 'bulk') & (adata.obs.brain_region != 'HIPP')&(adata.obs.condition != 'LRRK')].copy() #& 

    #make the names of the nearest genes unique
    adata2.var['old_index'] = adata2.var.index.astype(int)


    #Here choose the comparisaon you want to do
    adata3 = adata2[:,:] #we compare now SPOR vs. CTRL #adata2.var.shortAnnotChip=="Promoter" ((adata2.obs.condition!="CTRL"))
    adata3.var_names = adata3.var.nearestGeneChip.astype('string').values
    adata3.var_names_make_unique()
    adata3.var.index.names = ['names']
    #WGCNA Analysis: 

    comparisons = [
        "GBA1$CTRL",
        "SPOR$CTRL",
        "SPOR$GBA1"
    ] #comparison 1 against 2 without 3
    comparisons = ["SPOR$CTRL","SPOR$GBA1","GBA1$CTRL"]
    comp_all = [
        "GBA1$All",
        "SPOR$All",
        "CTRL$All"
    ]
    
    comp_dfs = []
    for ct in adata3.obs.celltype.unique():
        
        comp_adata = adata3[(adata3.obs.celltype==ct),:]
        comp_adata.var.reset_index(inplace=True)
        zero_peaks_cols = np.where(((comp_adata.X == np.float32(0)).all(axis=0)==True).tolist())[0].tolist()
        
        comp_adata = comp_adata[:,~comp_adata.var.index.isin(zero_peaks_cols)]
        
        for br in adata3.obs.brain_region.unique():
            print(f"{br}_{ct}")
            ad_ct_br = comp_adata[(comp_adata.obs.brain_region==br),:] #(comp_adata.var.shortAnnotChip=='Promoter')
            ad_ct_br.var_names = ad_ct_br.var.nearestGeneChip.astype('string').values
            ad_ct_br.var_names_make_unique()
            ad_ct_br.var.index.names = ['names']
            backgrounds[f"{br}_{ct}"] = ad_ct_br.var.nearestGeneChip.unique().to_list()
            #Wilcoxon's Test between conditions (here GBA1 vs. cutoff_overlap)
            print(f"{br}_{ct} DE analysis")
            
            for comp in comparisons: #comparisons
                print(comp)
                cond2, cond1 = comp.split('$')
                if cond1!="All":
                    curr = ad_ct_br[ad_ct_br.obs.condition.isin([cond1, cond2])]
                else:
                    curr = ad_ct_br
                curr.var.reset_index(inplace=True)
                zero_peaks_cols_comp = np.where((curr.X == np.float32(0)).all(axis=0))[0]
                comparison_ad = curr[:, ~curr.var.index.isin(zero_peaks_cols_comp)]
                comparison_ad.var_names = comparison_ad.var.nearestGeneChip.astype(str).values
                comparison_ad.var_names_make_unique()
                comparison_ad.var.index.names = ['names']
                comparison_ad.X = 10000000*comparison_ad.X
                comparison_ad.X = comparison_ad.X.astype(int)
                counts_df = pd.DataFrame(data=comparison_ad.X,columns=comparison_ad.var.index)
                clin_df = comparison_ad.obs[['condition','sex']]
                clin_df.reset_index(inplace=True)
                
                dds = DeseqDataSet(
                    counts=counts_df,
                    clinical=clin_df,
                    design_factors=["sex","condition"],
                    ref_level=['condition',cond2],
                    refit_cooks=False,
                    n_cpus=8,
                )
                dds.deseq2()
                stat_res_cond1_vs_cond2 = DeseqStats(dds, contrast=["condition", cond1, cond2], n_cpus=8)
                stat_res_cond1_vs_cond2.summary()
                group_df = stat_res_cond1_vs_cond2.results_df
                
                #Add variables 
                group_df = group_df.merge(adata3.var, left_on='names', right_on='names')
                group_df['-log10(pvals)'] = -np.log10(group_df.pvalue)
                group_df['-log10(qvals)'] = -np.log10(group_df.padj)
                group_df['ranks'] = group_df['-log10(qvals)']*group_df['log2FoldChange']
                group_df['significant'] = (group_df['-log10(qvals)']>=1.3010) & (np.abs(group_df['log2FoldChange'])>=logFC) # qvals < 0.05 & abs(logfoldchanges) > 0.5
                group_df['downreg'] = (group_df['-log10(qvals)']>=1.3010) & (group_df['log2FoldChange']<=-logFC)
                
                conditions = [
                group_df['significant']& group_df['downreg'] == True,
                group_df['significant']& ~group_df['downreg'] == True,
                group_df['significant']==False
                ]

                choices = [
                    'down','up', 'not significant'
                ]
                group_df['type_reg'] = np.select(conditions, choices, default='not significant')
                group_df['comparison'] = comp
                group_df["br"] = br
                group_df["ct"] = ct
                comp_dfs.append(group_df.copy())
    peaks_df = pd.concat(comp_dfs)
    return peaks_df

def differential_expression_no_correction(adata):
    
    #Data cleaning 
    adata.obs.dropna(axis=1, how="all", inplace=True) # drop columns with all NAN's
    adata2 = adata[(adata.obs.celltype != 'bulk') & (adata.obs.brain_region != 'HIPP')&(adata.obs.condition != 'LRRK2')].copy() #& 

    #make the names of the nearest genes unique
    adata2.var['old_index'] = adata2.var.index.astype(int)


    #Here choose the comparisaon you want to do
    adata3 = adata2[:,:] #we compare now SPOR vs. CTRL #adata2.var.shortAnnotChip=="Promoter" ((adata2.obs.condition!="CTRL"))
    adata3.var_names = adata3.var.nearestGeneChip.astype('string').values
    adata3.var_names_make_unique()
    adata3.var.index.names = ['names']
    #WGCNA Analysis: 

    comparisons = [
        "GBA1$CTRL",
        "SPOR$CTRL",
        "SPOR$GBA1"
    ] #comparison 1 against 2 without 3
    comparisons = ["SPOR$CTRL","SPOR$GBA1","GBA1$CTRL"]
    comp_all = [
        "GBA1$All",
        "SPOR$All",
        "CTRL$All"
    ]
    
    comp_dfs = []
    for ct in adata3.obs.celltype.unique():
        
        comp_adata = adata3[(adata3.obs.celltype==ct),:]
        comp_adata.var.reset_index(inplace=True)
        zero_peaks_cols = np.where(((comp_adata.X == np.float32(0)).all(axis=0)==True).tolist())[0].tolist()
        
        comp_adata = comp_adata[:,~comp_adata.var.index.isin(zero_peaks_cols)]
        
        for br in adata3.obs.brain_region.unique():
            print(f"{br}_{ct}")
            ad_ct_br = comp_adata[(comp_adata.obs.brain_region==br),:] #(comp_adata.var.shortAnnotChip=='Promoter')
            ad_ct_br.var_names = ad_ct_br.var.nearestGeneChip.astype('string').values
            ad_ct_br.var_names_make_unique()
            ad_ct_br.var.index.names = ['names']
            backgrounds[f"{br}_{ct}"] = ad_ct_br.var.nearestGeneChip.unique().to_list()
            #Wilcoxon's Test between conditions (here GBA1 vs. cutoff_overlap)
            print(f"{br}_{ct} DE analysis")
            
            for comp in comparisons: #comparisons
                print(comp)
                cond2, cond1 = comp.split('$')
                if cond1!="All":
                    curr = ad_ct_br[ad_ct_br.obs.condition.isin([cond1, cond2])]
                else:
                    curr = ad_ct_br
                curr.var.reset_index(inplace=True)
                zero_peaks_cols_comp = np.where((curr.X == np.float32(0)).all(axis=0))[0]
                comparison_ad = curr[:, ~curr.var.index.isin(zero_peaks_cols_comp)]
                comparison_ad.var_names = comparison_ad.var.nearestGeneChip.astype(str).values
                comparison_ad.var_names_make_unique()
                comparison_ad.var.index.names = ['names']
                comparison_ad.X = 10000000*comparison_ad.X
                comparison_ad.X = comparison_ad.X.astype(int)
                counts_df = pd.DataFrame(data=comparison_ad.X,columns=comparison_ad.var.index)
                clin_df = comparison_ad.obs[['condition','sex']]
                clin_df.reset_index(inplace=True)
                
                dds = DeseqDataSet(
                    counts=counts_df,
                    clinical=clin_df,
                    design_factors=["condition"],
                    ref_level=['condition',cond2],
                    refit_cooks=False,
                    n_cpus=8,
                )
                dds.deseq2()
                stat_res_cond1_vs_cond2 = DeseqStats(dds, contrast=["condition", cond1, cond2], n_cpus=8)
                stat_res_cond1_vs_cond2.summary()
                group_df = stat_res_cond1_vs_cond2.results_df
                
                #Add variables 
                group_df = group_df.merge(adata3.var, left_on='names', right_on='names')
                group_df['-log10(pvals)'] = -np.log10(group_df.pvalue)
                group_df['-log10(qvals)'] = -np.log10(group_df.padj)
                group_df['ranks'] = group_df['-log10(qvals)']*group_df['log2FoldChange']
                group_df['significant'] = (group_df['-log10(qvals)']>=1.3010) & (np.abs(group_df['log2FoldChange'])>=logFC) # qvals < 0.05 & abs(logfoldchanges) > 0.5
                group_df['downreg'] = (group_df['-log10(qvals)']>=1.3010) & (group_df['log2FoldChange']<=-logFC)
                
                conditions = [
                group_df['significant']& group_df['downreg'] == True,
                group_df['significant']& ~group_df['downreg'] == True,
                group_df['significant']==False
                ]

                choices = [
                    'down','up', 'not significant'
                ]
                group_df['type_reg'] = np.select(conditions, choices, default='not significant')
                group_df['comparison'] = comp
                group_df["br"] = br
                group_df["ct"] = ct
                comp_dfs.append(group_df.copy())
    peaks_df = pd.concat(comp_dfs)
    return peaks_df


def differential_expression_btwn_sex(adata):
    
    #Data cleaning 
    adata.obs.dropna(axis=1, how="all", inplace=True) # drop columns with all NAN's
    adata2 = adata[(adata.obs.celltype != 'bulk') & (adata.obs.brain_region != 'HIPP') & (adata.obs.condition != 'LRRK')].copy() #& 

    #make the names of the nearest genes unique
    adata2.var['old_index'] = adata2.var.index.astype(int)


    #Here choose the comparisaon you want to do
    adata3 = adata2[:,:] #we compare now SPOR vs. CTRL #adata2.var.shortAnnotChip=="Promoter" ((adata2.obs.condition!="CTRL"))
    adata3.var_names = adata3.var.nearestGeneChip.astype('string').values
    adata3.var_names_make_unique()
    adata3.var.index.names = ['names']
    #WGCNA Analysis: 

    comparisons = [
        "Male$Female"
    ] 
    
    condition_ref = "SPOR"
    comp_dfs = []
    for ct in adata3.obs.celltype.unique():
        
        comp_adata = adata3[(adata3.obs.celltype==ct) & (adata3.obs.condition == condition_ref),:]
        comp_adata.var.reset_index(inplace=True)
        zero_peaks_cols = np.where(((comp_adata.X == np.float32(0)).all(axis=0)==True).tolist())[0].tolist()
        
        comp_adata = comp_adata[:,~comp_adata.var.index.isin(zero_peaks_cols)]
        
        for br in adata3.obs.brain_region.unique():
            print(f"{br}_{ct}")
            ad_ct_br = comp_adata[(comp_adata.obs.brain_region==br),:] #(comp_adata.var.shortAnnotChip=='Promoter')
            ad_ct_br.var_names = ad_ct_br.var.nearestGeneChip.astype('string').values
            ad_ct_br.var_names_make_unique()
            ad_ct_br.var.index.names = ['names']
            backgrounds[f"{br}_{ct}"] = ad_ct_br.var.nearestGeneChip.unique().to_list()
            #Wilcoxon's Test between conditions (here GBA1 vs. cutoff_overlap)
            print(f"{br}_{ct} DE analysis")
            
            for comp in comparisons: #comparisons
                print(comp)
                cond2, cond1 = comp.split('$')
                curr = ad_ct_br
                curr.var.reset_index(inplace=True)
                zero_peaks_cols_comp = np.where((curr.X == np.float32(0)).all(axis=0))[0]
                comparison_ad = curr[:, ~curr.var.index.isin(zero_peaks_cols_comp)]
                comparison_ad.var_names = comparison_ad.var.nearestGeneChip.astype(str).values
                comparison_ad.var_names_make_unique()
                comparison_ad.var.index.names = ['names']
                comparison_ad.X = 1000*comparison_ad.X
                comparison_ad.X = comparison_ad.X.astype(int)
                counts_df = pd.DataFrame(data=comparison_ad.X,columns=comparison_ad.var.index)
                clin_df = comparison_ad.obs[['condition','sex']]
                clin_df.reset_index(inplace=True)
                
                dds = DeseqDataSet(
                    counts=counts_df,
                    clinical=clin_df,
                    design_factors=["sex"],
                    ref_level=['sex',cond2],
                    refit_cooks=True,
                    n_cpus=8,
                )
                dds.deseq2()
                stat_res_cond1_vs_cond2 = DeseqStats(dds, contrast=["sex", cond1, cond2], n_cpus=8,cooks_filter=False)
                stat_res_cond1_vs_cond2.summary()
                group_df = stat_res_cond1_vs_cond2.results_df
                
                #Add variables 
                group_df = group_df.merge(adata3.var, left_on='names', right_on='names')
                group_df['-log10(pvals)'] = -np.log10(group_df.pvalue)
                group_df['-log10(qvals)'] = -np.log10(group_df.padj)
                group_df['ranks'] = group_df['-log10(qvals)']*group_df['log2FoldChange']
                group_df['significant'] = (group_df['-log10(qvals)']>=1.3010) & (np.abs(group_df['log2FoldChange'])>=logFC) # qvals < 0.05 & abs(logfoldchanges) > 0.5
                group_df['downreg'] = (group_df['-log10(qvals)']>=1.3010) & (group_df['log2FoldChange']<=-logFC)
                
                conditions = [
                group_df['significant']& group_df['downreg'] == True,
                group_df['significant']& ~group_df['downreg'] == True,
                group_df['significant']==False
                ]

                choices = [
                    'down','up', 'not significant'
                ]
                group_df['type_reg'] = np.select(conditions, choices, default='not significant')
                group_df['comparison'] = comp
                group_df["br"] = br
                group_df["ct"] = ct
                comp_dfs.append(group_df.copy())
    peaks_df = pd.concat(comp_dfs)
    return peaks_df
