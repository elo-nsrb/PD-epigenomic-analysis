import anndata as ad
import scanpy as sc
from pydeseq2.default_inference import DefaultInference
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from statsmodels.stats.multitest import multipletests 
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
DPI=300


def prepareDataBulk(atac_input_file_name, deconv_path,peak_annot_path):
    df = pd.read_csv(atac_input_file_name)
    adata = ad.AnnData(df.drop(["Unnamed: 0","Sample_num", "Region","Group"], axis=1))
    adata.obs["filename"] = df["Unnamed: 0"].values
    adata.obs["Sample_num"] = df["Sample_num"].values
    adata.obs["Region"] = df["Region"].values
    adata.obs["Group"] = df["Group"].values
    adata.obs["PatientID"] = adata.obs["Sample_num"].str.split("_",expand=True)[3]
    adata.obs["PatientID"] = adata.obs["PatientID"].str.replace("xx","")
    adata.obs["PatientID"] = adata.obs["PatientID"].str.replace("x","_")
    adata
    
    adata.var["peakid"] = adata.var_names.tolist()
    adata.var
    
    peakannot = pd.read_csv(peak_annot_path,sep="\t")
    peakannot["peakid"] = peakannot["seqnames"].astype(str) + "." + (peakannot["start"] -1).astype(str) + "." + (peakannot["end"]).astype(str)
    peakannot
    adata.var =adata.var.merge(peakannot, on="peakid")
    
    meta = sc.read_h5ad(deconv_path).obs.drop(["celltype"],axis=1).drop_duplicates()
    meta = meta[['Neuritic plaque density binary',
     'Braak score NFT binary',
                 "xxx.Braak score",
                 'xxx.Unified LB Stage',
     'AD path',
     'PD clinical',
           'Neocortical LB',
     'Brainstem/Limbic LB',
     'ApoE_4',
     'ApoE_2',
     'Cognitive Imp',
     'PD path','Sex','ApoE','age at death',"group", "PatientID",'MutationType']].drop_duplicates()
 
    mis = [it for it in meta.PatientID.tolist() if it not in adata.obs.PatientID.tolist()]
    meta = meta[~meta.PatientID.isin(mis)]
    adata.obs = adata.obs.merge(meta, on="PatientID", how="left")
    
    adata.obs.rename(columns={"Region":"brain_region"}, inplace=True)
    adata.var.reset_index(inplace=True, drop=True)
    adata.obs["ADNC"] = adata.obs["AD path"].values
    adata.obs["CI"] = adata.obs["Cognitive Imp"].values
    adata.obs["Braak Score"] = adata.obs["xxx.Braak score"].map({"I":1, "II":1,"III":2,
                                                                 "IV":2, "V":3, "VI":3}).astype(float).values
    adata.obs["LB Stages"] = adata.obs['xxx.Unified LB Stage'].map({"0. No Lewy bodies":0.,"lV. Neocortical":3.,
                                                     "lll. Brainstem/Limbic":2.,"lla. Brainstem Predominant":1.,
                                                               "llb. Limbic Predominant":2.}).values
    adata.obs["Total path"] = adata.obs["Braak Score"].values.astype(float) + adata.obs["LB Stages"]

    return adata



def prepareData(adata):
    adata.obs["condition"] = adata.obs["condition"].astype(str)

    adata.obs["LB Stage"] = adata.obs['xxx.Unified LB Stage'].map({"0. No Lewy bodies":"No LB","lV. Neocortical":"LB Neocortical" ,
                                                         "lll. Brainstem/Limbic":"LB Brainstem Limbic","lla. Brainstem Predominant":"LB Brainstem Limbic",
                                                                   "llb. Limbic Predominant":"LB Brainstem Limbic"}).values
    
    adata.obs["LB Stages"] = adata.obs['xxx.Unified LB Stage'].map({"0. No Lewy bodies":0,"lV. Neocortical":4 ,
                                                         "lll. Brainstem/Limbic":3,"lla. Brainstem Predominant":1,
                                                                   "llb. Limbic Predominant":2}).values
    adata.obs["groupClinical"] = adata.obs["condition"].astype(str).values.copy()
    adata.obs.loc[(adata.obs["Cognitive Imp"]=="yes")&(adata.obs["condition"]=="GBA1_PD+"), "groupClinical"] = "GBA1-PD+-CI+"
    adata.obs.loc[(adata.obs["Cognitive Imp"]=="no")&(adata.obs["condition"]=="GBA1_PD+"), "groupClinical"] = "GBA1-PD+-CI-"
    adata.obs.loc[(adata.obs["Cognitive Imp"]=="yes")&(adata.obs["condition"]=="SPOR"), "groupClinical"] = "SPOR-CI+"
    adata.obs.loc[(adata.obs["Cognitive Imp"]=="no")&(adata.obs["condition"]=="SPOR"), "groupClinical"] = "SPOR-CI-"
    
    adata.obs["groupPath"] = adata.obs["condition"].astype(str).values.copy()
    adata.obs.loc[(adata.obs["AD path"]== 'moderate/high')&(adata.obs["condition"]=="GBA1_PD+"), "groupPath"] = "GBA1-PD+-AD+"
    adata.obs.loc[(adata.obs["AD path"]=='low/no')&(adata.obs["condition"]=="GBA1_PD+"), "groupPath"] = "GBA1-PD+-AD-"
    adata.obs.loc[(adata.obs["AD path"]== 'moderate/high')&(adata.obs["condition"]=="SPOR"), "groupPath"] = "SPOR-AD+"
    adata.obs.loc[(adata.obs["AD path"]=='low/no')&(adata.obs["condition"]=="SPOR"), "groupPath"] = "SPOR-AD-"
    
    # adata.obs["groupLBPath"] = adata.obs["condition"].astype(str).values.copy()
    # adata.obs.loc[(adata.obs["LB Stage"]=="LB Neocortical")&(adata.obs["condition"]=="GBA1-PD+"), "groupLBPath"] = "GBA1-PD+-Neocortical"
    # adata.obs.loc[(adata.obs["LB Stage"]=="LB Brainstem Limbic")&(adata.obs["condition"]=="GBA1-PD+"), "groupLBPath"] = "GBA1-PD+-Brainstem-Limbic"
    # adata.obs.loc[(adata.obs["LB Stage"]== "LB Neocortical")&(adata.obs["condition"]=="SPOR"), "groupLBPath"] = "SPOR-Neocortical"
    # adata.obs.loc[(adata.obs["LB Stage"]=="LB Brainstem Limbic")&(adata.obs["condition"]=="SPOR"), "groupLBPath"] = "SPOR-Brainstem-Limbic"
    
    adata.obs["condition"] = adata.obs["condition"].str.replace("_","-")

    ## remove gba1-pd- and lrrk2-pd-
    adata = adata[adata.obs.condition!="LRRK2-PD-"]
    adata = adata[adata.obs.condition!="GBA1-PD-"]
    adata.obs["condition2"] = adata.obs["condition"].replace({"LRRK2-PD-":"LRRK2", "LRRK2-PD+":"LRRK2"}).values
    adata.obs["ApoE4"] = adata.obs["ApoE_4"].replace({0:"E4-" ,1:"E4+"})
    adata.obs["CI"] = adata.obs["Cognitive Imp"].values
    adata.obs["Dementia nos"] = adata.obs["xxx.dementia_nos"].values

    adata.obs["AD dementia/MCI"] = adata.obs["xxx.AD"].values
    adata.obs["ADNC"] = adata.obs["AD path"].values
    adata.obs.loc[adata.obs["xxx.MCI"]=="yes", "AD dementia/MCI"] = "yes"
    return adata

def differential_expression(adata,logFC=0, method='t-test_overestim_var',columns="condition", corr_=None, refit=True):
    
    #Data cleaning 
    adata.obs.dropna(axis=1, how="all", inplace=True) # drop columns with all NAN's
    adata = adata[(adata.obs.celltype != 'bulk') & (adata.obs.condition != 'LRRK')].copy() #& 

    adata.var_names = adata.var.peakid.values
    adata.var["names"] = adata.var.peakid.values
   
    if columns == "condition":
        comparisons = [
              # "SPOR$CTRL",
            # "SPOR$LRRK2-PD+",
            # "GBA1-PD+$LRRK2-PD+",
             # "LRRK2-PD+$All",
           "SPOR$GBA1-PD+",
              "GBA1-PD+$All",
             # "GBA1-PD+$CTRL",
            "GBA1$All",
            "SPOR$All",
           
        ] #comparison 1 against 2 without 3
    elif columns == 'condition1':
        comparisons = [
              "SPOR$GBA1-PD+"]
        columns = "condition"
    elif columns == 'group':
        comparisons = [
              "SPOR$GBA1", "SPOR$All", "GBA1$All"]
        columns = "group"
    elif columns == 'group_separate':
        comparisons = [
              "SPOR$GBA1", "SPOR$LRRK2", "GBA1$LRRK2"]
        columns = "group"
    elif columns == 'condition2':
        comparisons = [
              "LRRK2$All"]
        columns = "condition2"
    elif columns == "groupPath":
        comparisons = [
       
            "GBA1-PD+-AD+$GBA1-PD+-AD-",
            "SPOR-AD+$GBA1-PD+-AD+",
            "SPOR-AD-$GBA1-PD+-AD-",
            "SPOR-AD+$SPOR-AD-",
            
        ] #co
    elif columns == "groupClinical":
        comparisons = [
        "SPOR-CI-+$GBA1-PD+-CI-",
                "SPOR-CI+$GBA1-PD+-CI+",
            "GBA1-PD+-CI+$GBA1-PD+-CI-",
    
           
            "SPOR-CI+$SPOR-CI-",
        ] #co
    elif columns == "LB Stage":
        comparisons=["LB Brainstem Limbic$LB Neocortical",
                    
                    #"LB Brainstem Limbic$All",
                    #"LB Neocortical$All",
                    "LB Brainstem Limbic$No LB",
                    "LB Neocortical$No LB"]
        adata = adata[~(adata.obs[columns].isna())]
    elif columns == "groupLBPath":
        comparisons=["GBA1-PD+-Neocortical$SPOR-Neocortical",
                    "SPOR-Brainstem-Limbic$SPOR-Neocortical",
                     "GBA1-PD+-Brainstem-Limbic$SPOR-Brainstem-Limbic",
                     "GBA1-PD+-Brainstem-Limbic$GBA1-PD+-Neocortical",
               ]
        adata = adata[~(adata.obs[columns].isna())]
    elif columns == "AD path":
        comparisons = ['low/no$moderate/high']
        adata = adata[~(adata.obs[columns].isna())]
    elif columns == "Cognitive Imp":
        comparisons = ['no$yes']
        adata = adata[~(adata.obs[columns].isna())]
    elif columns == "ApoE4":
        comparisons = ['E4+$E4-']
        adata = adata[~(adata.obs[columns].isna())]   
    print("ok")
    comp_dfs = []
    for ct in adata.obs.celltype.unique():
        
        comp_adata = adata[(adata.obs.celltype==ct),:]
      
        for br in adata.obs.brain_region.unique():
            print(f"{br}_{ct}")
            ad_ct_br = comp_adata[(comp_adata.obs.brain_region==br),:] #(comp_adata.var.shortAnnotChip=='Promoter')
            ad_ct_br = ad_ct_br[:,  ad_ct_br.X.sum(0)!=0]
            print(f"{br}_{ct} DE analysis")
            
            for comp in comparisons: #comparisons
                print(comp)
                cond2, cond1 = comp.split('$')
                
  
                if cond1!="All":
                    curr = ad_ct_br[ad_ct_br.obs[columns].isin([cond1, cond2])]
                    if cond2=="GBA1":
                        curr.obs.condition = curr.obs[columns].replace({"GBA1_PD+":"GBA1", 'GBA1_PD-':"GBA1"})
                    if cond2=="LRRK2":
                        curr.obs.condition = curr.obs[columns].replace({"LRRK2_PD+":"LRRK2", 'LRRK2_PD-':"LRRK2"})
                      
                else:
                    curr = ad_ct_br.copy()
                    if cond2=="GBA1":
                        curr.obs.condition = curr.obs[columns].replace({"GBA1_PD+":"GBA1", 'GBA1_PD-':"GBA1", })
                    if cond2=="LRRK2":
                        curr.obs.condition = curr.obs[columns].replace({"LRRK2_PD+":"LRRK2", 'LRRK2_PD-':"LRRK2"})
                    curr.obs[columns] = curr.obs[columns].apply(lambda x: x if x==cond2 else "All")
                print(curr.obs.groupby([columns]).size())

                if curr.obs[columns].nunique() >1:
                # if True:
                    curr.var.reset_index(inplace=True)
                    
                    comparison_ad = curr[:, curr.X.sum(0)!=0]
                    # sc.pp.highly_variable_genes(comparison_ad, n_top_genes=2000,  flavor ="seurat_v3")#inplace=True,
                    # comparison_ad = comparison_ad[:, comparison_ad.var.highly_variable]
                    print(comparison_ad.shape)
                    comparison_ad.X = (comparison_ad.X*1000).astype(int)
                    comparison_ad.var_names =  comparison_ad.var.peakid.values
                    comparison_ad.var["names"] = comparison_ad.var.index.tolist()
                    # df_ = comparison_ad.to_df()
                    # print(df_)
                    # df_["PatientID"] = comparison_ad.obs["PatientID"].values
                    # counts_df = df_.groupby(["PatientID"])[comparison_ad.var.index.tolist()].sum()
                    # print(counts_df.shape) 
                    # clin_df = df_.groupby(["PatientID"])[comparison_ad.var.index.tolist()].sum().reset_index()
                    # clin_df = clin_df[["PatientID"]].merge(comparison_ad.obs[['condition','Sex',"PatientID"]].drop_duplicates(), on="PatientID", how="inner")
                    #clin_df.index = clin_df["PatientID"].values
                    #print(clin_df.shape)
                    counts_df = pd.DataFrame(data=comparison_ad.X,columns=comparison_ad.var_names)
                    clin_df = comparison_ad.obs[[columns,"PatientID"]+corr_]
                    clin_df.reset_index(inplace=True)
                    clin_df = clin_df[[columns,"PatientID"]+corr_]
                   
                    if False:
                        dat = ad.AnnData(counts_df)
                        dat.obs = clin_df
                        dat.var["gene"] = comparison_ad.var_names.tolist()
        
                        group_df = de.test.wald(
                                    data=dat,
                                    formula_loc="~ 1 + " + columns + "+ Sex",
                                    factor_loc_totest=columns,
                            # coef_to_test=[cond1],
                                )
                        # sc.tl.rank_genes_groups(dat, groupby="condition", method=method, 
                        #                     rankby_abs=True,
                        #                    gene_symbols="gene",
                        #                     #gene_symbols="peakid",
                        #                     key_added = method)
                        # group_df = sc.get.rank_genes_groups_df(dat, group=cond1, 
                        #                             gene_symbols="gene",
                        #                             key=method, #log2fc_min=0.25, 
                        #                             )#['names'].squeeze().str.strip().tolist()
        
                        # group_df["pvalue"] = group_df["pvals"].values
                        # group_df["log2FoldChange"] = group_df["logfoldchanges"].values
                        # group_df["padj"] = group_df["pvals_adj"].values
        
                        group_df["pvalue"] = group_df["pval"].values
                        group_df["log2FoldChange"] = group_df["log2fc"].values
                        group_df["padj"] =  multipletests(
                                        group_df["pval"], alpha=0.05, method="fdr_bh")[1]
                    if True:
                        if corr_ is None:
                            design_factors = [columns]
                        else:
                            design_factors = [columns] + corr_
                        inference = DefaultInference(n_cpus=8)
                        dds = DeseqDataSet(
                            counts=counts_df,
                            metadata=clin_df,
                            design_factors=design_factors,
                            ref_level=[columns,cond2], ## ref kevek ==cond2
                            refit_cooks=refit,
                            inference=inference
                           # n_cpus=8,
                        )
                        dds.deseq2()
                        stat_res_cond1_vs_cond2 = DeseqStats(dds, contrast=[columns, cond1, cond2])
                        stat_res_cond1_vs_cond2.summary()
                        group_df = stat_res_cond1_vs_cond2.results_df
                
                    #Add variables 
                    # print(group_df.head(5))
                    group_df = group_df.join( comparison_ad.var)#, left_on='names', right_on='names')
                    # print(group_df.head(5))
                    group_df['-log10(pvals)'] = -np.log10(group_df.pvalue)
                    group_df['-log10(qvals)'] = -np.log10(group_df.padj)
                    group_df['ranks'] = group_df['-log10(qvals)']*group_df['log2FoldChange']
                    group_df['significant'] = (group_df['-log10(qvals)']>=-np.log10(0.05)) & (np.abs(group_df['log2FoldChange'])>=logFC) # qvals < 0.05 & abs(logfoldchanges) > 0.5
                    group_df['downreg'] = (group_df['-log10(qvals)']>=-np.log10(0.05)) & (group_df['log2FoldChange']<=-logFC)
                    
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


def plot_DE_number_one(res_DE,outdir,outname, more_title="",cmap='RdPu', lf_key="lf"):
    
    count = res_DE.loc[res_DE.significant==True,["padj","downreg",lf_key,"comparison","br","ct"]].groupby(["br","ct","downreg","comparison"]).count()[lf_key].reset_index()
    total_number = res_DE.loc[:,["padj","downreg",lf_key,"comparison","br","ct"]].groupby(["br","ct","comparison"]).count()[lf_key].reset_index()
    count = count.merge(total_number.rename({lf_key:"total_num"},axis=1),on=["br","ct","comparison"])
    count[lf_key] = (count
                            [lf_key]
                            )
    count = count.drop(["total_num"],axis=1)
    
    t = count.set_index(["br","ct","downreg","comparison"])

    all_possibilities = [(br,ct,downreg,comparison)
                      for (br,ct,downreg,comparison) in product(res_DE["br"].unique(),res_DE["ct"].unique(),res_DE["downreg"].unique(),res_DE["comparison"].unique()) ]
    
    for pos in all_possibilities:
        if pos not in t.index:
            t.loc[pos] = 0
    
    count = t.reset_index()
    sns.set(font_scale=1.2, style="white")
    fig,ax = plt.subplots(1,2,figsize=(8,4))
    vmin,vmax = float("inf"), -float("inf")

    for comp in count.comparison.unique():
        pivot_table_down = count[(count.downreg==True) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({lf_key:""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_down.columns = pivot_table_down.columns.droplevel(0)
        print(pivot_table_down.columns)
        
        pivot_table_up = count[(count.downreg==False) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({lf_key:""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_up.columns = pivot_table_up.columns.droplevel(0)
        
        pivot_table_down = pivot_table_down.fillna(0)
        pivot_table_up = pivot_table_up.fillna(0)
        # Determine the common range for the color bar
        vmin = min(pivot_table_down.values.min(), pivot_table_up.values.min(),vmin)
        vmax = max(pivot_table_down.values.max(), pivot_table_up.values.max(),vmax)
    # vmin,vmax = 0, 
    # vmax=1000
    for i,comp in enumerate(count.comparison.unique()):
        #if "res_DE_ct_sex" in res_DE_file_name:
        #    count = pd.concat([count,pd.DataFrame(["PTMN",False,"SPOR$GBA1",0]),pd.DataFrame(["CAUD",False,"SPOR$GBA1",0])])

        
        pivot_table_down = count[(count.downreg==True) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({lf_key:""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
        pivot_table_down.columns = pivot_table_down.columns.droplevel(0)
        print(pivot_table_down.columns)
        
        pivot_table_up = count[(count.downreg==False) & (count.ct != "bulk") & (count.comparison==comp)].drop(["downreg"],axis=1).rename({lf_key:""},axis=1).drop(["comparison"],axis=1).pivot(index="ct",columns="br").astype(float)
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
        #ax[i][1].set_yticks([])
        ax[0].tick_params(axis=u'both', which=u'both',length=0)
        ax[1].tick_params(axis=u'both', which=u'both',length=0)
        cond0, cond1 = comp.split("$")
        ax[1].set_title(f"{more_title}Up-Accessible OCRs - {cond1} vs. {cond0}",size=10)
        ax[0].set_xlabel("Brain Region",size=10)
        ax[1].set_xlabel("Brain Region",size=10)
        ax[0].set_title(f"{more_title}Down-Accessible OCRs - {cond1} vs. {cond0}", size=10)
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
    # ax=ax.flatten()
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
    # vmin,vmax = 0, 400
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
def filter_sig(peaks_df,logFC = 0.5, pvalue=0.05):
    peaks_df['significant'] = (peaks_df['-log10(qvals)']>=-np.log10(pvalue)) & (np.abs(peaks_df['log2FoldChange'])>=logFC) # qvals < 0.05 & abs(logfoldchanges) > 0.5
    peaks_df['downreg'] = (peaks_df['-log10(qvals)']>=-np.log10(pvalue)) & (peaks_df['log2FoldChange']<=-logFC)
                    
    conditions = [
                   peaks_df['significant']& peaks_df['downreg'] == True,
                   peaks_df['significant']& ~peaks_df['downreg'] == True,
                   peaks_df['significant']==False
                    ]
    
    choices = [ 'down','up', 'not significant']
    peaks_df['type_reg'] = np.select(conditions, choices, default='not significant')
    return peaks_df
# Calculate peak variance and cumulative variance
def calculate_cumulative_variance(adata):
    # Calculate variance for each peak
    peak_variance = adata.X.var(axis=0)
    
    # Sort variances in descending order
    sorted_variance = np.sort(peak_variance)[::-1]
    
    # Calculate cumulative sum of variances
    cumsum_variance = np.cumsum(sorted_variance)
    
    # Calculate percentage of total variance explained
    total_variance = np.sum(peak_variance)
    percent_variance = (cumsum_variance / total_variance) * 100
    
    # Create a DataFrame for plotting
    variance_df = pd.DataFrame({
        'n_peaks': range(1, len(sorted_variance) + 1),
        'cumulative_variance': percent_variance,
        'individual_variance': sorted_variance
    })
    
    return variance_df

# Create plots to visualize the variance
def plot_variance_metrics(variance_df):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot cumulative variance
    sns.lineplot(data=variance_df, 
                x='n_peaks', 
                y='cumulative_variance',
                ax=ax1, color="k")
    ax1.set_title('Cumulative Variance Explained')
    ax1.set_xlabel('Number of Peaks')
    ax1.set_ylabel('Cumulative Variance (%)')
    
    # Add horizontal lines at common thresholds
    thresholds = [50, 80, 90, 95, 99]
    color = "#b88c9e" #b88c9e'
    for threshold in thresholds:
        ax1.axhline(y=threshold, color=color, linestyle='--', alpha=0.7)
        n_peaks = variance_df[variance_df['cumulative_variance'] >= threshold].iloc[0]['n_peaks']
        ax1.text(len(variance_df), threshold, f'{threshold}% ({n_peaks:,.0f} peaks)', 
                verticalalignment='bottom')
    
    # Plot individual variances (elbow plot)
    sns.lineplot(data=variance_df.head(10000),  # Limit to top 10000 for visibility
                x='n_peaks',
                y='individual_variance',
                color="k",
                ax=ax2)
    ax2.set_title('Individual Peak Variances (Top 10000)')
    ax2.set_xlabel('Peak Rank')
    ax2.set_ylabel('Variance')
    
    plt.tight_layout()
    return fig