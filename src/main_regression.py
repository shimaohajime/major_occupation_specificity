'''
Code to run the wage/employment regressions at major and individual level and replicate the tables.
It assumes that the MOS (formerly denoted as MCI, the major-complexity index) is computed in compute_MOS.py,
and the other indices are computed in compute_other_indices.py.
'''



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

from matplotlib import pyplot as plt
import seaborn as sns

import copy

## Measure execution time
import time

start_time = time.time()

# Read data

if True:
    df_wage = pd.read_csv(f'data/processed_data/acs/df_reg_{"degfield"}_soc{2}.csv',index_col=0)
    df_wage_ = pd.read_csv(f'data/processed_data/acs/df_reg_{"degfieldd"}_soc{2}.csv',index_col=0)
    df_wage['major_d'] = df_wage_['major']
    df_wage_d_ = pd.read_csv(f'data/processed_data/acs/df_reg_{"degfield"}_soc{4}.csv',index_col=0)
    df_wage['occupation_4'] = df_wage_d_['occupation']

    df_emp = pd.read_csv(f'data/processed_data/acs/df_emp_{"degfield"}_soc{2}.csv',index_col=0)
    df_emp_ = pd.read_csv(f'data/processed_data/acs/df_emp_{"degfieldd"}_soc{2}.csv',index_col=0)
    df_emp['major_d'] = df_emp_['major']
    df_emp_d_ = pd.read_csv(f'data/processed_data/acs/df_emp_{"degfield"}_soc{4}.csv',index_col=0)
    # df_emp['occupation_4'] = df_emp_d_['occupation']

    # Average income (incwage_cpiu_2010) by majors
    wage_by_major = df_wage.groupby(['major']).agg({'incwage_cpiu_2010':'mean'})
    wage_by_major_d = df_wage.groupby(['major_d']).agg({'incwage_cpiu_2010':'mean'})

    # Read the indices
    df_index_by_major = pd.read_csv(f'results/acs/other_index_{"degfield"}_soc{2}.csv',index_col=0)
    df_index_by_major_d = pd.read_csv(f'results/acs/other_index_{"degfieldd"}_soc{2}.csv',index_col=0)
    df_index_by_major_4 = pd.read_csv(f'results/acs/other_index_{"degfield"}_soc{4}.csv',index_col=0)
    df_index_by_major_d4 = pd.read_csv(f'results/acs/other_index_{"degfieldd"}_soc{4}.csv',index_col=0)

    df_index_by_major.columns = [f'{col}_degfield_soc2' for col in df_index_by_major.columns]
    df_index_by_major_d.columns = [f'{col}_degfieldd_soc2' for col in df_index_by_major_d.columns]
    df_index_by_major_4.columns = [f'{col}_degfield_soc4' for col in df_index_by_major_4.columns]
    df_index_by_major_d4.columns = [f'{col}_degfieldd_soc4' for col in df_index_by_major_d4.columns]

    df_index_by_major = pd.concat([df_index_by_major, df_index_by_major_4],axis=1)
    df_index_by_major_d = pd.concat([df_index_by_major_d, df_index_by_major_d4],axis=1)

    df_gini = pd.read_csv(f'results/acs/gini_{"degfield"}.csv',index_col=0)
    df_index_by_major = df_index_by_major.merge(df_gini,how='left',left_index=True,right_index=True)



    # Read MCI
    for soc_level in [2,4]:
        for binarize_method in ['rca']:#,'fixedr'
            if binarize_method == 'rca':
                binarize_threshold = 1.
            elif binarize_method == 'fixedr':
                binarize_threshold = 0.05 
            df_mci = pd.read_csv(f'results/acs/MCI_Maj_{binarize_method}{binarize_threshold:.2f}_{"degfield"}_soc{soc_level}.csv',index_col=0)
            df_mci = df_mci.loc[:,df_mci.columns.str.startswith('mci')]
            df_mci.columns = [f'{col}_{binarize_method}{binarize_threshold:.2f}_degfield_soc{soc_level}' for col in df_mci.columns]
            df_index_by_major = pd.concat([df_index_by_major, df_mci],axis=1)

            df_mci_d = pd.read_csv(f'results/acs/MCI_Maj_{binarize_method}{binarize_threshold:.2f}_{"degfieldd"}_soc{soc_level}.csv',index_col=0)
            df_mci_d = df_mci_d.loc[:,df_mci_d.columns.str.startswith('mci')]
            df_mci_d.columns = [f'{col}_{binarize_method}{binarize_threshold:.2f}_degfieldd_soc{soc_level}' for col in df_mci_d.columns]
            df_index_by_major_d = pd.concat([df_index_by_major_d, df_mci_d],axis=1)



    # Normalize Indices
    scaler = StandardScaler()
    df_index_by_major = pd.DataFrame(scaler.fit_transform(df_index_by_major),columns=df_index_by_major.columns,index=df_index_by_major.index)
    df_index_by_major_d = pd.DataFrame(scaler.fit_transform(df_index_by_major_d),columns=df_index_by_major_d.columns,index=df_index_by_major_d.index)


    # Limit the indices to the main ones
    df_index_by_major_main = df_index_by_major.loc[:,['mci_b_rca1.00_degfield_soc4','mci_b_from_hhi_rca1.00_degfield_soc4','mci_b_from_top3_rca1.00_degfield_soc4',\
                                                    'mci2_b_rca1.00_degfield_soc4','mci2_b_from_hhi_rca1.00_degfield_soc4','mci2_b_from_top3_rca1.00_degfield_soc4',\
                                                        'Blom_hhi_degfield_soc4','Altonji_top3_share_degfield_soc4','gini_degfield']]
    df_index_by_major_main.rename(columns={'mci_b_rca1.00_degfield_soc4':'MCI','mci_b_from_hhi_rca1.00_degfield_soc4':'MCI_HHI','mci_b_from_top3_rca1.00_degfield_soc4':'MCI_TOP3',\
                                        'mci2_b_rca1.00_degfield_soc4':'MCI2','mci2_b_from_hhi_rca1.00_degfield_soc4':'MCI2_HHI','mci2_b_from_top3_rca1.00_degfield_soc4':'MCI2_TOP3',\
                                            'Blom_hhi_degfield_soc4':'HHI','Altonji_top3_share_degfield_soc4':'TOP3','gini_degfield':'GINI'},inplace=True)
    df_index_by_major_d_main = df_index_by_major_d.loc[:,['mci_b_rca1.00_degfieldd_soc4','mci_b_from_hhi_rca1.00_degfieldd_soc4','mci_b_from_top3_rca1.00_degfieldd_soc4',\
                                                        'mci2_b_rca1.00_degfieldd_soc4','mci2_b_from_hhi_rca1.00_degfieldd_soc4','mci2_b_from_top3_rca1.00_degfieldd_soc4',\
                                                            'Blom_hhi_degfieldd_soc4','Altonji_top3_share_degfieldd_soc4']]
    df_index_by_major_d_main.rename(columns={'mci_b_rca1.00_degfieldd_soc4':'MCI_d','mci_b_from_hhi_rca1.00_degfieldd_soc4':'MCI_HHI_d','mci_b_from_top3_rca1.00_degfieldd_soc4':'MCI_TOP3_d',\
                                            'mci2_b_rca1.00_degfieldd_soc4':'MCI2_d','mci2_b_from_hhi_rca1.00_degfieldd_soc4':'MCI2_HHI_d','mci2_b_from_top3_rca1.00_degfieldd_soc4':'MCI2_TOP3_d',\
                                            'Blom_hhi_degfieldd_soc4':'HHI_d','Altonji_top3_share_degfieldd_soc4':'TOP3_d'},inplace=True)
    df_index_by_major_d_main.index.names = ['major_d']

    df_index_by_major_ext = df_index_by_major_main.copy()
    df_index_by_major_ext['MCI_w'] = df_index_by_major['mci_w_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI_c'] = df_index_by_major['mci_c_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI2_w'] = df_index_by_major['mci2_w_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI2_c'] = df_index_by_major['mci2_c_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI_w_HHI'] = df_index_by_major['mci_w_from_hhi_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI_c_HHI'] = df_index_by_major['mci_c_from_hhi_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI2_w_HHI'] = df_index_by_major['mci2_w_from_hhi_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI2_c_HHI'] = df_index_by_major['mci2_c_from_hhi_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI_w_TOP3'] = df_index_by_major['mci_w_from_top3_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI_c_TOP3'] = df_index_by_major['mci_c_from_top3_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI2_w_TOP3'] = df_index_by_major['mci2_w_from_top3_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI2_c_TOP3'] = df_index_by_major['mci2_c_from_top3_rca1.00_degfield_soc4']

    df_index_by_major_ext['MCI_GINI'] = df_index_by_major['mci_b_gini_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI_w_GINI'] = df_index_by_major['mci_w_gini_rca1.00_degfield_soc4']
    df_index_by_major_ext['MCI_c_GINI'] = df_index_by_major['mci_c_gini_rca1.00_degfield_soc4']

    df_index_by_major_ext['MCI_HHI_SOC2'] = df_index_by_major['mci_b_from_hhi_rca1.00_degfield_soc2']


    df_index_by_major_d_ext = df_index_by_major_d_main.copy()
    df_index_by_major_d_ext['MCI_d_w'] = df_index_by_major_d['mci_w_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI_d_c'] = df_index_by_major_d['mci_c_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI2_d_w'] = df_index_by_major_d['mci2_w_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI2_d_c'] = df_index_by_major_d['mci2_c_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI_d_w_HHI'] = df_index_by_major_d['mci_w_from_hhi_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI_d_c_HHI'] = df_index_by_major_d['mci_c_from_hhi_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI2_d_w_HHI'] = df_index_by_major_d['mci2_w_from_hhi_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI2_d_c_HHI'] = df_index_by_major_d['mci2_c_from_hhi_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI_d_w_TOP3'] = df_index_by_major_d['mci_w_from_top3_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI_d_c_TOP3'] = df_index_by_major_d['mci_c_from_top3_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI2_d_w_TOP3'] = df_index_by_major_d['mci2_w_from_top3_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI2_d_c_TOP3'] = df_index_by_major_d['mci2_c_from_top3_rca1.00_degfieldd_soc4']

    df_index_by_major_d_ext['MCI_GINI_d'] = df_index_by_major_d['mci_b_gini_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI_w_GINI_d'] = df_index_by_major_d['mci_w_gini_rca1.00_degfieldd_soc4']
    df_index_by_major_d_ext['MCI_c_GINI_d'] = df_index_by_major_d['mci_c_gini_rca1.00_degfieldd_soc4']

    df_index_by_major_d_ext['MCI_HHI_d_SOC2'] = df_index_by_major_d['mci_b_from_hhi_rca1.00_degfieldd_soc2']


    # Add major characteristics
    cip0 = pd.read_stata('data/raw_data/acs/cip0_final.dta')
    df_index_by_major_d = df_index_by_major_d.merge(cip0,how='left',left_index=True,right_on='ACS_Majorcode')
    ## Set the index to ACS_Majorcode and rename it as major_d
    df_index_by_major_d = df_index_by_major_d.set_index('ACS_Majorcode')
    df_index_by_major_d.index.name = 'major_d'
    
    df_index_by_major_d_ext['Category'] = df_index_by_major_d['GroupName']
    df_index_by_major_d_ext['STEM'] = df_index_by_major_d['STEM']


    ## Add the major
    df_degfield = pd.read_csv('data/processed_data/acs/degfield_mapping.csv', index_col=0)
    df_degfieldd = pd.read_csv('data/processed_data/acs/degfieldd_mapping.csv', index_col=0)
    df_index_by_major_ext = df_index_by_major_ext.merge(df_degfield, left_index=True, right_index=True, how='left')
    df_index_by_major_d_ext = df_index_by_major_d_ext.merge(df_degfieldd, left_index=True, right_index=True, how='left')

    df_index_by_major_ext.to_csv('results/acs/df_index_by_major_ext.csv')
    df_index_by_major_d_ext.to_csv('results/acs/df_index_by_major_d_ext.csv')


    ## Merge the df_index with df at the major level
    # df_wage = df_wage.merge(df_index_by_major_main,on='major',how='left')
    # df_wage = df_wage.merge(df_index_by_major_d_main,on='major_d',how='left')
    df_wage = df_wage.merge(df_index_by_major_ext,on='major',how='left')
    df_wage = df_wage.merge(df_index_by_major_d_ext,on='major_d',how='left')
    df_emp = df_emp.merge(df_index_by_major_ext,on='major',how='left')
    df_emp = df_emp.merge(df_index_by_major_d_ext,on='major_d',how='left')


    ## MCI in earlier iterations.
    df_mci_iter_hhi = pd.read_csv(f'results/acs/mci_b_from_hhi_over_iteration_rca1.00_degfieldd_soc4.csv', index_col=0)
    ## Add MCI_HHI_ to the column names
    degfieldd_s = df_mci_iter_hhi['degfieldd_s']
    df_mci_iter_hhi.drop(columns=['degfieldd_s'],inplace=True)
    df_mci_iter_hhi.columns = ['MCI_HHI_'+col for col in df_mci_iter_hhi.columns]
    hhi = df_mci_iter_hhi['MCI_HHI_iter_0']/100
    ## Normalize
    scaler = StandardScaler()
    df_mci_iter_hhi = pd.DataFrame(scaler.fit_transform(df_mci_iter_hhi),columns=df_mci_iter_hhi.columns,index=df_mci_iter_hhi.index)
    df_mci_iter_hhi['degfieldd_s'] = degfieldd_s
    ## Compare the rankings
    df_for_table_ = df_mci_iter_hhi.copy()
    df_for_table_['HHI'] = hhi
    df_for_table_['Ranking'] = df_for_table_['HHI'].rank(ascending=False,method='first')
    df_for_table_.sort_values('HHI',ascending=False)[['Ranking','HHI','degfieldd_s']].to_csv('results/acs/mci_b_from_hhi_iter0_ranking.csv')
    df_for_table_['Ranking'] = df_for_table_['MCI_HHI_iter_25'].rank(ascending=False,method='first')
    df_for_table_.sort_values('MCI_HHI_iter_25',ascending=False)[['Ranking','MCI_HHI_iter_25','degfieldd_s']].to_csv('results/acs/mci_b_from_hhi_iter25_ranking.csv')


    ## Ranking on degfield not degfieldd
    temp = pd.read_csv(f'results/acs/mci_b_from_hhi_over_iteration_rca1.00_degfield_soc4.csv', index_col=0)[['iter_25','degfield_s']]
    temp.rename(columns={'iter_25':'MCI_HHI_iter_25'},inplace=True)
    temp['Ranking'] = temp['MCI_HHI_iter_25'].rank(ascending=False,method='first')
    temp.sort_values('MCI_HHI_iter_25',ascending=False)[['Ranking','MCI_HHI_iter_25','degfield_s']].to_csv('results/acs/mci_b_from_hhi_iter25_ranking_degfield.csv')


    ## Ranking based on GINI
    '''
    df_mci_iter_gini = pd.read_csv(f'results/acs/mci_b_from_gini_over_iteration_rca1.00_degfield_soc4.csv', index_col=0)
    df_mci_iter_hhi_coarse = pd.read_csv(f'results/acs/mci_b_from_hhi_over_iteration_rca1.00_degfield_soc4.csv', index_col=0)
    # pd.concat([df_mci_iter_hhi_coarse['iter_10'],df_mci_iter_gini['iter_10']],axis=1).corr()

    degfieldd_s = df_mci_iter_gini['degfield_s']
    df_mci_iter_gini.drop(columns=['degfield_s'],inplace=True)
    df_mci_iter_gini.columns = ['MCI_GINI_'+col for col in df_mci_iter_gini.columns]
    gini = df_mci_iter_gini['MCI_GINI_iter_0']/100
    ## Normalize
    scaler = StandardScaler()
    df_mci_iter_gini = pd.DataFrame(scaler.fit_transform(df_mci_iter_gini),columns=df_mci_iter_gini.columns,index=df_mci_iter_gini.index)
    df_mci_iter_gini['degfield_s'] = degfieldd_s
    df_mci_iter_gini['GINI'] = gini
    df_mci_iter_gini.sort_values('GINI',ascending=False).to_csv('results/acs/mci_b_from_gini_change_over_iter.csv')

    df_mci_iter_gini.sort_values('MCI_GINI_iter_10',ascending=False)
    df_mci_iter_gini.set_index('degfield_s')[f'MCI_GINI_iter_10'].rank(ascending=False,method='first')
    '''
    ## Create variables that are ranking at each iteration

    df_mci_iter_hhi_degfield = pd.read_csv(f'results/acs/mci_b_from_hhi_over_iteration_rca1.00_degfield_soc4.csv', index_col=0)
    iter_ranking = pd.DataFrame(index=df_mci_iter_hhi_degfield.degfield_s)
    for iter in range(26):
        iter_ranking[f'Ranking_iter_{iter}'] = df_mci_iter_hhi_degfield.set_index('degfield_s')[f'iter_{iter}'].rank(ascending=False,method='first')
    iter_ranking = iter_ranking.T



    ## Merge with the regression df
    df_wage = df_wage.merge(df_mci_iter_hhi, left_on='major_d', right_index=True, how='left')
    df_emp = df_emp.merge(df_mci_iter_hhi, left_on='major_d', right_index=True, how='left')

    df_wage = df_wage.dropna()
    df_emp = df_emp.dropna()

    ## In df_wage and df_emp, turn the SubgroupName into a dummy variable
    df_wage = pd.get_dummies(df_wage, columns=['Category'], drop_first=False)
    df_wage = df_wage.drop(columns=['Category_Other'])
    df_emp = pd.get_dummies(df_emp, columns=['Category'], drop_first=False)
    df_emp = df_emp.drop(columns=['Category_Other'])

    ##
    wage_summary_stat = df_wage[['sex','white','hispanic','experience','incwage_cpiu_2010']].describe().T
    wage_summary_stat.drop(columns=['25%','50%','75%'],inplace=True)
    wage_summary_stat.rename(columns={'mean':'Mean','std':'Std', 'min':'Min','max':'Max','count':'N'},inplace=True)
    wage_summary_stat = wage_summary_stat[['N','Mean','Std','Min','Max' ]]
    wage_summary_stat.to_csv('results/acs/summary_stat_df_wage.csv')

    emp_summary_stat = df_emp[['sex','white','hispanic','experience','fulltime']].describe().T
    emp_summary_stat.drop(columns=['25%','50%','75%'],inplace=True)
    emp_summary_stat.rename(columns={'mean':'Mean','std':'Std', 'min':'Min','max':'Max','count':'N'},inplace=True)
    emp_summary_stat = emp_summary_stat[['N','Mean','Std','Min','Max' ]]
    emp_summary_stat.to_csv('results/acs/summary_stat_df_emp.csv')

    ##
    if False:
        df_wage.to_csv('results/acs/main_merged_wage_df.csv')
        df_emp.to_csv('results/acs/main_merged_emp_df.csv')






def reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI','MCI2'],other_indices=['HHI','TOP3'], add_exp_int=False, cluster=True, cluster_vec=None,weighted_reg=False, weight_vec=None, logit=False,\
                         main_together=True, other_together=True, other_each=True, other_together_only=False):
    results_ind = []
    col_names = []
    model = sm.Logit if logit else sm.OLS
    cov_type = 'cluster' if cluster else 'nonrobust'
    cov_kwds = {'groups': cluster_vec} if cluster else None
    fit_args = {'cov_type':cov_type, 'cov_kwds':cov_kwds, 'maxiter':200} if logit else {'cov_type':cov_type, 'cov_kwds':cov_kwds}

    # print(f'main_indices: {main_indices}, other_indices: {other_indices}, len: {len(main_indices)}, {len(other_indices)}')
    
    #1. Regression with each index only
    for idx in main_indices:
        X_ = pd.concat([X,df_reg[[idx]]], axis=1)
        if add_exp_int:
            X_[f'{idx}_exp'] = X_['experience']*X_[idx]
            X_[f'{idx}_exp_sq'] = X_['experience_sq']*X_[idx]
        result = model(y, sm.add_constant( X_.copy() )).fit(**fit_args)
        results_ind.append(copy.deepcopy(result) )
        col_names.append(y.name)

    if other_each:
        for idx in other_indices:
            X_ = pd.concat([X,df_reg[[idx]]], axis=1)
            if add_exp_int:
                X_[f'{idx}_exp'] = X_['experience']*X_[idx]
                X_[f'{idx}_exp_sq'] = X_['experience_sq']*X_[idx]
            result = model(y, sm.add_constant( X_.copy() )).fit(**fit_args)
            results_ind.append(copy.deepcopy(result) )
            col_names.append(y.name)


    #2. MCI controlling for each of other indices
    if other_each:
        for idx2 in other_indices:
            for idx in main_indices:
                if idx2 != idx:
                    X_ = pd.concat([X,df_reg[[idx,idx2]]], axis=1)
                    if add_exp_int:
                        X_[f'{idx}_exp'] = X_['experience']*X_[idx]
                        X_[f'{idx}_exp_sq'] = X_['experience_sq']*X_[idx]
                        X_[f'{idx2}_exp'] = X_['experience']*X_[idx2]
                        X_[f'{idx2}_exp_sq'] = X_['experience_sq']*X_[idx2]
                    result = model(y, sm.add_constant( X_.copy() )).fit(**fit_args)
                    results_ind.append(copy.deepcopy(result) )
                    col_names.append(y.name)

    #3. MCI controlling for all the other indices
    if other_together_only and len(other_indices)>1:
        X_ = pd.concat([X,df_reg[other_indices]], axis=1)
        if add_exp_int:
            for idx2 in other_indices:
                X_[f'{idx2}_exp'] = X_['experience']*X_[idx2]
                X_[f'{idx2}_exp_sq'] = X_['experience_sq']*X_[idx2]
        result = model(y, sm.add_constant( X_.copy() )).fit(**fit_args)
        results_ind.append(copy.deepcopy(result) )
        col_names.append(y.name)


    if other_together and len(other_indices)>1:
        for idx in main_indices:
            X_ = pd.concat([X,df_reg[[idx]+other_indices]], axis=1)
            if add_exp_int:
                X_[f'{idx}_exp'] = X_['experience']*X_[idx]
                X_[f'{idx}_exp_sq'] = X_['experience_sq']*X_[idx]
                for idx2 in other_indices:
                    X_[f'{idx2}_exp'] = X_['experience']*X_[idx2]
                    X_[f'{idx2}_exp_sq'] = X_['experience_sq']*X_[idx2]
            result = model(y, sm.add_constant( X_.copy() )).fit(**fit_args)
            results_ind.append(copy.deepcopy(result) )
            col_names.append(y.name)

    #4. MCI and MCI2
    if main_together and len(main_indices)>1:
        X_ = pd.concat([X,df_reg[main_indices]], axis=1)
        if add_exp_int:
            for idx in main_indices:
                X_[f'{idx}_exp'] = X_['experience']*X_[idx]
                X_[f'{idx}_exp_sq'] = X_['experience_sq']*X_[idx]
        result = model(y, sm.add_constant( X_.copy() ) ).fit(**fit_args)
        results_ind.append(copy.deepcopy(result) )
        col_names.append(y.name)

    #5. MCI, MCI2 controlling for each of other indices
    if other_each:
        if main_together and len(main_indices)>1:
            for idx2 in other_indices:
                X_ = pd.concat([X,df_reg[main_indices+[idx2]]], axis=1)
                if add_exp_int:
                    for idx in main_indices:
                        X_[f'{idx}_exp'] = X_['experience']*X_[idx]
                        X_[f'{idx}_exp_sq'] = X_['experience_sq']*X_[idx]
                    X_[f'{idx2}_exp'] = X_['experience']*X_[idx2]
                    X_[f'{idx2}_exp_sq'] = X_['experience_sq']*X_[idx2]
                result = model(y, sm.add_constant( X_.copy() ) ).fit(**fit_args)
                results_ind.append(copy.deepcopy(result) )
                col_names.append(y.name)

    #6. MCI, MCI2 controlling for all the other indices
    if other_together and len(other_indices)>1:
        if main_together and len(main_indices)>1:
            X_ = pd.concat([X,df_reg[main_indices+other_indices]], axis=1)
            if add_exp_int:
                for idx in main_indices:
                    X_[f'{idx}_exp'] = X_['experience']*X_[idx]
                    X_[f'{idx}_exp_sq'] = X_['experience_sq']*X_[idx]
                for idx2 in other_indices:
                    X_[f'{idx2}_exp'] = X_['experience']*X_[idx2]
                    X_[f'{idx2}_exp_sq'] = X_['experience_sq']*X_[idx2]
            result = model(y, sm.add_constant( X_.copy() ) ).fit(**fit_args)
            results_ind.append(copy.deepcopy(result) )
            col_names.append(y.name)

    return results_ind, col_names

def sort_table_rows(summary, main_var, control_var):

    table_df = pd.DataFrame(summary.tables[0])
    new_index = []
    for i in range(table_df.shape[0]):
        if table_df.index[i] == '':
            new_index.append( table_df.index[i-1] + '_std' )
        else:
            new_index.append( table_df.index[i] )
    table_df.index = new_index

    sort_var = main_var +control_var
    sorted_index = []
    for var in sort_var:
        sorted_index.append(var)
        sorted_index.append(var+'_std')
    sorted_index = sorted_index+ ['R-squared Adj.','No. observations']
    table_df = table_df.reindex(sorted_index)

    for idx in table_df.index:
        if '_std' in idx:
            table_df.rename(index={idx:''}, inplace=True)

    return table_df




'''
1. Major-level analyses
'''
# Collapse df_reg at the major level
# Find continuous and categorical variables
numeric_df = df_wage.select_dtypes(include=['number'])
numeric_df['major_d'] = df_wage['major_d']
df_wage_major = numeric_df.groupby('major_d').mean()
# df_wage_major = df_wage.groupby('major_d').mean()


df_wage_major['log_wage'] = df_wage_major['incwage_cpiu_2010'].apply(np.log1p)
X_major = df_wage_major[['white','hispanic','sex']]
y_major = df_wage_major['log_wage']


## Correlation
corr_indices = df_wage_major[['HHI_d','TOP3_d','GINI','MCI_HHI_d','MCI_TOP3_d','MCI_GINI_d','MCI_d']].corr()
corr_indices.to_csv('results/acs/corr_indices.csv')

corr_mci_coarse = df_wage_major[['MCI_HHI_d','MCI_HHI_d_SOC2','MCI_HHI','MCI_HHI_SOC2']].corr()
corr_mci_coarse.to_csv('results/acs/corr_mci_coarse.csv')

## Correlation with NSSE variables - CIP4 and degfieldd.
df_cip4_mapping = pd.read_stata('data/raw_data/acs/CIP4_Final.dta')
dta_nsse = pd.read_stata('data/raw_data/acs/NSSE_CIP4.dta',iterator=True)
# nsse_var_label = dta_nsse.variable_labels()
# nsse_var_label = pd.DataFrame.from_dict(nsse_var_label, orient='index', columns=['label'])
df_nsse = pd.read_stata('data/raw_data/acs/NSSE_CIP4.dta')
df_nsse = df_nsse.merge(df_cip4_mapping, left_on='CIP4', right_on='CIP4',how='left')
df_nsse = df_nsse.merge(df_wage_major, left_on='ACS_Majorcode',right_index=True,how='left')


## Aggregating continuous variables only
# df_cip4_level = df_nsse.groupby('CIP4').mean().dropna()
numeric_df = df_nsse.select_dtypes(include=['number'])
numeric_df['CIP4'] = df_nsse['CIP4']
df_cip4_level = numeric_df.groupby('CIP4').mean()


nsse_corr = df_cip4_level.corrwith(df_cip4_level['MCI_HHI_d']).reset_index()
# nsse_corr = nsse_corr.merge(nsse_var_label, left_on='index', right_index=True,how='left').set_index('index').rename(columns={0:'correlation'}).dropna()
nsse_corr.to_csv('results/acs/nsse_corr_cip4.csv')
nsse_corr.to_csv('results/paper/nsse_corr_cip4.csv')

nsse_hhi_corr = df_cip4_level.corrwith(df_cip4_level['HHI_d']).reset_index()# nsse_corr = nsse_corr.merge(nsse_var_label, left_on='index', right_index=True,how='left').set_index('index').rename(columns={0:'correlation'}).dropna()
nsse_hhi_corr.rename(columns={0:'HHI'},inplace=True)
nsse_hhi_corr.to_csv('results/acs/nsse_hhi_corr_cip4.csv')
nsse_hhi_corr.to_csv('results/paper/nsse_hhi_corr_cip4.csv')


## Correlation with NSSE variables - CIP2 and degfield.
df_cip2_mapping = pd.read_stata('data/raw_data/acs/CIP2_Final.dta')
dta_nsse = pd.read_stata('data/raw_data/acs/NSSE_CIP2.dta',iterator=True)
# nsse_var_label = dta_nsse.variable_labels()
# nsse_var_label = pd.DataFrame.from_dict(nsse_var_label, orient='index', columns=['label'])
df_nsse = pd.read_stata('data/raw_data/acs/NSSE_CIP2.dta')
df_nsse = df_nsse.merge(df_cip2_mapping, left_on='CIP2', right_on='CIP2',how='left')
df_nsse = df_nsse.merge(df_wage_major, left_on='ACS_Majorcode',right_index=True,how='left')


# df_cip2_level = df_nsse.groupby('CIP2').mean().dropna()
numeric_df = df_nsse.select_dtypes(include=['number'])
numeric_df['CIP2'] = df_nsse['CIP2']
df_cip2_level = numeric_df.groupby('CIP2').mean()


nsse_corr = df_cip2_level.corrwith(df_cip2_level['MCI_HHI']).reset_index()
# nsse_corr = nsse_corr.merge(nsse_var_label, left_on='index', right_index=True,how='left').set_index('index').rename(columns={0:'correlation'}).dropna()
nsse_corr.to_csv('results/acs/nsse_corr_cip2.csv')
nsse_corr.to_csv('results/paper/nsse_corr_cip2.csv')

nsse_hhi_corr = df_cip2_level.corrwith(df_cip2_level['HHI']).reset_index()# nsse_corr = nsse_corr.merge(nsse_var_label, left_on='index', right_index=True,how='left').set_index('index').rename(columns={0:'correlation'}).dropna()
nsse_hhi_corr.rename(columns={0:'HHI'},inplace=True)
nsse_hhi_corr.to_csv('results/acs/nsse_hhi_corr_cip2.csv')
nsse_hhi_corr.to_csv('results/paper/nsse_hhi_corr_cip2.csv')




target_var ='wage'
## 1.1 Regression with HHI-based MCI with early iterations.
print('Major-level regression with HHI-based MCI with early iterations.')
results_iter, col_names_iter = reg_with_mci_indices(X_major, y_major, df_wage_major, main_indices = [f'MCI_HHI_iter_{i}' for i in [0,1,2,5,10,25]],other_indices=[], add_exp_int=False, cluster=False, cluster_vec=None,weighted_reg=False, weight_vec=None, main_together=False)
summary_wage_iter = summary_col(results_iter, stars=True,
                            # model_names=col_names_iter,
                            info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                            'No. observations' : lambda x: f"{int(x.nobs):d}"}
                                            )
table_df = sort_table_rows(summary_wage_iter, [f'MCI_HHI_iter_{i}' for i in [0,1,2,5,10,25]], ['HHI','const','sex','white','hispanic'])
table_df.to_csv(f'results/acs/regression_{target_var}_major_hhi_iter.csv')


## Original HHI controlled
X_major_with_hhi = X_major.copy()
X_major_with_hhi['HHI'] = df_wage_major['MCI_HHI_iter_0']
## 1.1 Regression with HHI-based MCI with early iterations.
results_iter, col_names_iter = reg_with_mci_indices(X_major_with_hhi, y_major, df_wage_major, main_indices = [f'MCI_HHI_iter_{i}' for i in [1,2,5,10,25]],other_indices=[], add_exp_int=False, cluster=False, cluster_vec=None,weighted_reg=False, weight_vec=None, main_together=False)
summary_wage_iter = summary_col(results_iter, stars=True,
                            model_names=col_names_iter,
                            info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                            'No. observations' : lambda x: f"{int(x.nobs):d}"})
table_df = sort_table_rows(summary_wage_iter, [f'MCI_HHI_iter_{i}' for i in [1,2,5,10,25]], ['HHI','const','sex','white','hispanic'])
table_df.to_csv(f'results/acs/regression_{target_var}_major_hhi_iter_with_hhi.csv')


result_folder = 'results/acs/'


'''
2. Individual-level regressions
'''

## Define the baseline features.
X_wage = pd.concat([pd.get_dummies(df_wage['year'], prefix='year',drop_first=True),df_wage[['white','hispanic','sex']],df_wage[['experience','experience_sq']]], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
y_wage = df_wage['incwage_cpiu_2010'].apply(np.log1p).rename('log_wage')
major_wage = df_wage['major']
## Generate an integer index for each major-year pair for clustering
major_year_wage = df_wage['major_d'].astype(str) + df_wage['year'].astype(str)
major_year_wage = major_year_wage.astype('category').cat.codes

X_emp = pd.concat([pd.get_dummies(df_emp['year'], prefix='year',drop_first=True),df_emp[['white','hispanic','sex']],df_emp[['experience','experience_sq']]], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
y_emp = df_emp['fulltime']#2- df_emp['empstat'] 
major_emp = df_emp['major']
## Generate an integer index for each major-year pair for clustering
major_year_emp = df_emp['major_d'].astype(str) + df_emp['year'].astype(str)
major_year_emp = major_year_emp.astype('category').cat.codes


def wage_emp_choice(target_var):
    if target_var=='wage':
        return df_wage, X_wage, y_wage, major_year_wage
    elif target_var=='emp':
        return df_emp, X_emp, y_emp, major_year_emp

## Regression with vanilla MCI
for target_var in ['wage','emp']:
    print(f'---Regression for {target_var}---')
    df_reg, X, y, major_year = wage_emp_choice(target_var)

    if False:

        print('Individual-level regression with vanilla MCI.')
        results_ind_d, col_names_d = reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI_d'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None)
        #,'MCI2_d'
        summary_wage_ind_d = summary_col(results_ind_d, stars=True,
                                    model_names=col_names_d,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        table_df = sort_table_rows(summary_wage_ind_d, ['MCI_d'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d.csv')
        del results_ind_d

        ## Regression with HHI-based MCI
        print('Individual-level regression with HHI-based MCI.')
        results_ind_d_hhi, col_names_d_hhi = reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI_HHI_d'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None)
        summary_wage_ind_d_hhi = summary_col(results_ind_d_hhi, stars=True,
                                    model_names=col_names_d_hhi,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        #,'MCI2_HHI_d'
        table_df = sort_table_rows(summary_wage_ind_d_hhi, ['MCI_HHI_d'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi.csv')
        del results_ind_d_hhi


    if False:
        ## Regression with Category dummy
        print('Individual-level regression with category dummy.')
        results_ind_d_stem, col_names_d_stem = reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI_HHI_d'],other_indices=['STEM'], add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None)
        results_ind_d_cat, col_names_d_cat = reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI_HHI_d'],other_indices=df_reg.columns[df_reg.columns.str.startswith('Category')].tolist(), add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None, other_each=False,other_together=True,other_together_only=True)
        summary_wage_ind_d_cat = summary_col(results_ind_d_stem+results_ind_d_cat[1:], stars=True,
                                    # model_names=col_names_d_cat,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        table_df = sort_table_rows(summary_wage_ind_d_cat, ['MCI_HHI_d'], ['STEM']+ df_reg.columns[df_reg.columns.str.startswith('Category')].tolist() + ['const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_cat.csv')
        del results_ind_d_stem, results_ind_d_cat

    if False:
        ## Regression with degfield-level MCI and SOC2-based MCI.
        print('Individual-level regression with MCI based on different major/occupation category.')
        results_ind_coarse, col_names_d_coarse = reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI_HHI_d','MCI_HHI_d_SOC2','MCI_HHI','MCI_HHI_SOC2'],other_indices=[], add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None, main_together=False)
        summary_wage_ind_d_coarse = summary_col(results_ind_coarse, stars=True,
                                    model_names=col_names_d_coarse,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        table_df = sort_table_rows(summary_wage_ind_d_coarse, ['MCI_HHI_d','MCI_HHI_d_SOC2','MCI_HHI','MCI_HHI_SOC2'],  ['const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_coarse.csv')
        del results_ind_coarse, col_names_d_coarse






    if False:

        ## Regression with GINI-based MCI
        print('Individual-level regression with GINI-based MCI.')
        results_ind_d_gini, col_names_d_gini = reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI_GINI_d'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None)
        summary_wage_ind_d_gini = summary_col(results_ind_d_gini, stars=True,
                                    model_names=col_names_d_gini,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        #,'MCI2_GINI_d'
        table_df = sort_table_rows(summary_wage_ind_d_gini, ['MCI_GINI_d'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_gini.csv')
        del results_ind_d_gini

        ## Regression with TOP3-based MCI
        print('Individual-level regression with TOP3-based MCI.')
        results_ind_d_top3, col_names_d_top3 = reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI_TOP3_d'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None)
        summary_wage_ind_d_top3 = summary_col(results_ind_d_top3, stars=True,
                                    model_names=col_names_d_top3,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        #,'MCI2_TOP3_d'
        table_df = sort_table_rows(summary_wage_ind_d_top3, ['MCI_TOP3_d'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_top3.csv')
        del results_ind_d_top3


        ## Regression with weighted MCI
        print('Individual-level regression with weighted MCI.')
        results_ind_d_weighted, col_names_d_weighted = reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI_d_w','MCI2_d_w'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None)
        summary_wage_ind_d_weighted = summary_col(results_ind_d_weighted, stars=True,
                                    model_names=col_names_d_weighted,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        table_df = sort_table_rows(summary_wage_ind_d_weighted, ['MCI_d_w','MCI2_d_w'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_weighted.csv')
        del results_ind_d_weighted

        ## Regression with controlled MCI
        print('Individual-level regression with controlled MCI.')
        results_ind_d_controlled, col_names_d_controlled = reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI_d_c','MCI2_d_c'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None)
        summary_wage_ind_d_controlled = summary_col(results_ind_d_controlled, stars=True,
                                    model_names=col_names_d_controlled,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        table_df = sort_table_rows(summary_wage_ind_d_controlled, ['MCI_d_c','MCI2_d_c'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_controlled.csv')
        del results_ind_d_controlled


        ## Regression with experience in 3-13 (HHI-based MCI)
        print('Individual-level regression with HHI-based MCI and experience in 3-13.')
        experience_start = 3
        experience_end = 13
        df_reg_in_age_bin = df_reg[(df_reg['experience']>=experience_start) & (df_reg['experience']<=experience_end)]
        X_in_age_bin = X.loc[df_reg_in_age_bin.index]
        y_in_age_bin = y.loc[df_reg_in_age_bin.index]
        major_year_in_age_bin = major_year.loc[df_reg_in_age_bin.index]

        results_ind_d_age_bin, col_names_d_age_bin = reg_with_mci_indices(X_in_age_bin, y_in_age_bin, df_reg_in_age_bin, main_indices = ['MCI_HHI_d'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year_in_age_bin,weighted_reg=False, weight_vec=None)
        #,'MCI2_HHI_d'
        summary_wage_ind_d_age_bin = summary_col(results_ind_d_age_bin, stars=True,
                                    model_names=col_names_d,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        table_df = sort_table_rows(summary_wage_ind_d_age_bin, ['MCI_HHI_d'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_age_bin.csv')
        del results_ind_d_age_bin

    if False:
        ## Regression with earlier iterations (HHI-based MCI)
        print('Individual-level regression with HHI-based MCI and earlier iterations.')
        results_ind_d_hhi, col_names_d_hhi = reg_with_mci_indices(X, y, df_reg, main_indices = [f'MCI_HHI_iter_{i}' for i in [1,2,5,10,25]],other_indices=[], add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None, main_together = False)
        summary_wage_ind_d_hhi = summary_col(results_ind_d_hhi, stars=True,
                                    model_names=col_names_d_hhi,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        #,'MCI2_HHI_d'
        table_df = sort_table_rows(summary_wage_ind_d_hhi, [f'MCI_HHI_iter_{i}' for i in [1,2,5,10,25]], ['const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_iter.csv')
        del results_ind_d_hhi


    if True:
        ## Regression with sliding window of the age bin
        print('Individual-level regression with HHI-based MCI and sliding window of the age bin.')
        idx = 'MCI_HHI_d'
        idx2 = 'HHI_d'
        coef_seq = []
        coef_hhi_seq = []
        std_seq = []
        std_hhi_seq = []
        
        conf_up = []
        conf_down = []
        conf_up_hhi = []
        conf_down_hhi = []

        for experience_start in range(3,28):
            experience_end = experience_start + 10
            print(f'Age in {experience_start+22}-{experience_end+22}')
            df_reg_in_age_bin = df_reg[(df_reg['experience']>=experience_start) & (df_reg['experience']<=experience_end)]
            X_in_age_bin = X.loc[df_reg_in_age_bin.index]
            y_in_age_bin = y.loc[df_reg_in_age_bin.index]
            major_year_in_age_bin = major_year.loc[df_reg_in_age_bin.index]

            X_ = pd.concat([X_in_age_bin,df_reg_in_age_bin[[idx]]], axis=1).astype(float)
            result = sm.OLS(y_in_age_bin, sm.add_constant( X_.copy() )).fit()
            coef_seq.append(result.params[idx])
            std_seq.append(result.bse[idx])
            conf_up.append(result.conf_int().loc[idx][1])
            conf_down.append(result.conf_int().loc[idx][0])

            X2_ = pd.concat([X_in_age_bin,df_reg_in_age_bin[[idx2]]], axis=1).astype(float)
            result = sm.OLS(y_in_age_bin, sm.add_constant( X2_.copy() )).fit()
            coef_hhi_seq.append(result.params[idx2])
            std_hhi_seq.append(result.bse[idx2])
            conf_up_hhi.append(result.conf_int().loc[idx2][1])
            conf_down_hhi.append(result.conf_int().loc[idx2][0])

        coef_seq = np.array(coef_seq)
        std_seq = np.array(std_seq)
        coef_hhi_seq = np.array(coef_hhi_seq)
        std_hhi_seq = np.array(std_hhi_seq)
        x = np.arange(3,28)+5.+22

        #Plot coef_seq
        fig, ax = plt.subplots()
        ax.plot(x, coef_seq, label='Coefficient of MOS', color='black')
        ax.fill_between(x, conf_down, conf_up, color='gray', alpha=0.5)
        # ax.fill_between(x, coef_seq - std_seq, coef_seq + std_seq, color='gray', alpha=0.5)
        # Decided not to include HHI
        # ax.plot(x, coef_hhi_seq, label='Coefficient of HHI', color='red')
        # ax.fill_between(x, coef_hhi_seq - std_hhi_seq, coef_hhi_seq + std_hhi_seq, color='pink', alpha=0.5)
        ax.set_xlabel('Age')
        # ax.set_ylabel(f'Coefficient on {target_var}')
        ax.legend()
        # ax.set_title('Estimated Coefficients with Standard Error')
        fig.savefig(f'results/acs/{target_var}_reg_age_window.png')
        fig.savefig(f'results/paper/{target_var}_reg_age_window.png')
        plt.close()


