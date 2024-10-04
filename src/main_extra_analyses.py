'''
Wage and employment regressions robustness check.
It assumes that the MOS (formerly denoted as MCI, the major-complexity index) is computed in compute_MOS.py,
and the other indices are computed in compute_other_indices.py.

'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.inspection import partial_dependence
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

from matplotlib import pyplot as plt
import seaborn as sns

import copy
from scipy.ndimage import gaussian_filter1d

## Measure execution time
import time

# df_wage = pd.read_csv('results/acs/main_merged_wage_df.csv', index_col=0)
# df_emp = pd.read_csv('results/acs/main_merged_emp_df.csv', index_col=0)

# df_wage_major_year = df_wage.groupby(['degfieldd','year']).mean()
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


def compute_mci_eig(A, At=None, mci_init=None, oci_init=None):
    if mci_init is None:
        assert np.all(np.isin(A,[0,1])), "A must be binary matrix if mci_init is not provided."
        mci_init = A.sum(axis=1)
    if oci_init is None:
        assert np.all(np.isin(A,[0,1])), "A must be binary matrix if oci_init is not provided."
        oci_init = A.sum(axis=0)

    A_tilde = np.linalg.inv(np.diag(mci_init.values)) @ A.values @ np.linalg.inv(np.diag(oci_init.values)) @ A.values.T 
    eigvals, eigvecs = np.linalg.eig(A_tilde)
    return pd.Series(eigvecs[:,1], index=A.index)


def compute_mci_hajime(A, At=None, mci_init=None, iter=250, output_other_side=False, normalize=False):
    M, N = A.shape
    if mci_init is None:
        assert np.all(np.isin(A,[0,1])), "A must be binary matrix if mci_init is not provided."
        mci_init = A.sum(axis=1)
    mci = mci_init.copy().values.reshape([-1,1])
    if At is None:
        At = A.T
    Aw = A/np.sum(A,axis=1).values.reshape(M,1)
    Atw = At/np.sum(At,axis=1).values.reshape(N,1)

    result_mci = pd.DataFrame(index=A.index)
    result_mci[f'iter_{0}'] = mci
    result_oci = pd.DataFrame(index=A.columns)
    for i in range(iter):
        oci = np.dot(Atw,mci)
        mci = np.dot(Aw,oci)
        # oci = (oci - oci.mean())/oci.std()
        # mci = (mci - mci.mean())/mci.std()
        oci = (oci - oci.min())/(oci.max()-oci.min())
        mci = (mci - mci.min())/(mci.max()-mci.min())
        result_mci[f'iter_{i+1}'] = mci
        result_oci[f'iter_{i+1}'] = oci
    if output_other_side:
        return result_mci, result_oci
    else:
        return result_mci



binarize_method = 'rca'
binarize_threshold = 1.
soc_level = 4
major_var = 'degfieldd'
iterations = 25

df_mci = pd.read_csv(f'data/processed_data/acs/df_mci_{major_var}_soc{soc_level}.csv',index_col=0)
df_adj = df_mci.groupby(['major','occupation']).size().unstack(fill_value=0)
print(df_mci.shape)

df_wage = pd.read_csv(f'data/processed_data/acs/df_reg_{"degfield"}_soc{2}.csv',index_col=0)
df_wage_ = pd.read_csv(f'data/processed_data/acs/df_reg_{"degfieldd"}_soc{2}.csv',index_col=0)
df_wage['major_d'] = df_wage_['major']
df_wage_d_ = pd.read_csv(f'data/processed_data/acs/df_reg_{"degfield"}_soc{4}.csv',index_col=0)
df_wage['occupation_4'] = df_wage_d_['occupation']

df_emp = pd.read_csv(f'data/processed_data/acs/df_emp_{"degfield"}_soc{2}.csv',index_col=0)
df_emp_ = pd.read_csv(f'data/processed_data/acs/df_emp_{"degfieldd"}_soc{2}.csv',index_col=0)
df_emp['major_d'] = df_emp_['major']
df_emp_d_ = pd.read_csv(f'data/processed_data/acs/df_emp_{"degfield"}_soc{4}.csv',index_col=0)


df_index_by_major_d4 = pd.read_csv(f'results/acs/other_index_{"degfieldd"}_soc{4}.csv',index_col=0)
df_gini = pd.read_csv(f'results/acs/gini_{"degfield"}.csv',index_col=0)

df_index_by_major_d4 = df_index_by_major_d4.reset_index().rename(columns={'major':'major_d','Blom_hhi':'HHI_d','Altonji_top3_share':'TOP3_d','OD':'OD_d'})
df_gini = df_gini.reset_index().rename(columns={'gini_degfield':'GINI'})
scaler = StandardScaler()
df_index_by_major_d4['HHI_d'] = scaler.fit_transform(df_index_by_major_d4['HHI_d'].values.reshape(-1,1))
df_index_by_major_d4['TOP3_d'] = scaler.fit_transform(df_index_by_major_d4['TOP3_d'].values.reshape(-1,1))

## Normalize OD as well?
df_index_by_major_d4['OD_d'] = scaler.fit_transform(df_index_by_major_d4['OD_d'].values.reshape(-1,1))

df_gini['GINI'] = scaler.fit_transform(df_gini['GINI'].values.reshape(-1,1))


# Add major characteristics
cip0 = pd.read_stata('data/raw_data/acs/cip0_final.dta')
df_index_by_major_d4 = df_index_by_major_d4.merge(cip0,how='left',left_on='major_d',right_on='ACS_Majorcode')
df_index_by_major_d4['Category'] = df_index_by_major_d4['GroupName']


mci = pd.read_csv(f'results/acs/MCI_Maj_{binarize_method}{binarize_threshold:.2f}_{"degfieldd"}_soc{soc_level}.csv',index_col=0)
df_index_by_major_d4['MCI_HHI_d'] = scaler.fit_transform(mci[['mci_b_from_hhi']])

df_wage = df_wage.merge(df_index_by_major_d4[['major_d','HHI_d','TOP3_d','MCI_HHI_d','Category','STEM','OD_d']],on='major_d')
df_wage = df_wage.merge(df_gini[['major','GINI']],on='major')
df_emp = df_emp.merge(df_index_by_major_d4[['major_d','HHI_d','TOP3_d','MCI_HHI_d']],on='major_d')
df_emp = df_emp.merge(df_gini[['major','GINI']],on='major')

def wage_emp_choice(target_var):
    if target_var=='wage':
        return df_wage
    elif target_var=='emp':
        return df_emp
def create_Xy(df_reg, target_var, extra_control = []):
    X_ = pd.concat([pd.get_dummies(df_reg['year'], prefix='year',drop_first=True),df_reg[['white','hispanic','sex']],df_reg[['experience','experience_sq']+extra_control] ], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
    y_ = df_reg['incwage_cpiu_2010'].apply(np.log1p).rename('log_wage') if target_var=='wage' else df_reg['fulltime']
    major_ = df_reg['major']
    ## Generate an integer index for each major-year pair for clustering
    major_year_ = df_reg['major_d'].astype(str) + df_reg['year'].astype(str)
    major_year_ = major_year_.astype('category').cat.codes
    return X_, y_, major_year_


## Emp regression with logistic model
if False:
    for target_var in ['emp']:
        df_reg = df_wage if target_var=='wage' else df_emp
        X_, y_, major_year_ = create_Xy(df_reg, target_var)
        X_['MCI'] = df_reg['MCI_HHI_d']
        X_['HHI'] = df_reg['HHI_d']
        X_['TOP3'] = df_reg['TOP3_d']
        X_['GINI'] = df_reg['GINI']

        # model = LogisticRegression(penalty='none',solver='lbfgs',max_iter=10000)
        # model.fit(X_,y_)
        # partial_dependence(model, X_, [0])
        model = sm.Logit(y_,X_)
        result = model.fit()
        print(result.summary())

        X2_ = X_.copy()
        X3_ = X_.copy()
        X2_['MCI'] = X_['MCI'] - X_['MCI'].std()
        X3_['MCI'] = X_['MCI'] + X_['MCI'].std()
        print( (result.predict(X_) - result.predict(X2_) ).mean() )
        print( (result.predict(X3_) - result.predict(X_) ).mean() )

        model2 = sm.Probit(y_,X_)
        result2 = model2.fit()
        print(result2.summary())
        print( (result2.predict(X_) - result2.predict(X2_) ).mean() )
        print( (result2.predict(X3_) - result2.predict(X_) ).mean() )

## Regression with OD
if False:
    for target_var in ['wage']:#,'emp'
        print(f'Individual-level {target_var} regression with OD.')
        df_reg = df_wage if target_var=='wage' else df_emp
        X_, y_, major_year_ = create_Xy(df_reg, target_var)
        # X_ = pd.concat([pd.get_dummies(df_reg['year'], prefix='year',drop_first=True),df_reg[['white','hispanic','sex']],df_reg[['experience','experience_sq']]], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
        # y_ = df_reg['incwage_cpiu_2010'].apply(np.log1p).rename('log_wage') if target_var=='wage' else df_reg['fulltime']
        # major_ = df_reg['major']
        # ## Generate an integer index for each major-year pair for clustering
        # major_year_ = df_wage['major_d'].astype(str) + df_wage['year'].astype(str)
        # major_year_ = major_year_.astype('category').cat.codes

        results_ind_d_od, col_names_d_od = reg_with_mci_indices(X_, y_, df_reg, main_indices = ['MCI_HHI_d'],other_indices=['HHI_d','TOP3_d','GINI','OD_d'], add_exp_int=False, cluster=True, cluster_vec=major_year_,weighted_reg=False, weight_vec=None)
        summary_wage_ind_d_hhi30 = summary_col(results_ind_d_od, stars=True,
                                    model_names=col_names_d_od,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        table_df = sort_table_rows(summary_wage_ind_d_hhi30, ['MCI_HHI_d'], ['HHI_d','TOP3_d','GINI','OD_d','const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_with_od.csv')
        del results_ind_d_od



## Regression with Major FE
if False:
    for target_var in ['wage','emp']:#
        print(f'Individual-level {target_var} regression with major dummies.')
        df_reg = wage_emp_choice(target_var) 
        X_ = pd.concat([pd.get_dummies(df_reg['year'], prefix='year',drop_first=True),df_reg[['white','hispanic','sex']],df_reg[['experience','experience_sq']], pd.get_dummies(df_reg['major'], prefix='major',drop_first=True)], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
        y_ = df_reg['incwage_cpiu_2010'].apply(np.log1p).rename('log_wage') if target_var=='wage' else df_reg['fulltime']
        major_wage = df_reg['major']
        ## Generate an integer index for each major-year pair for clustering
        major_year_ = df_reg['major_d'].astype(str) + df_reg['year'].astype(str)
        major_year_ = major_year_.astype('category').cat.codes
        result_ = sm.OLS(y_, sm.add_constant( X_.copy() ) ).fit(cov_type='cluster',cov_kwds={'groups': major_year_})
        print(f'Adj. R-squared: {result_.rsquared_adj:.4f}')
        del result_

        print(f'Individual-level {target_var} regression with major_d dummies.')
        X_ = pd.concat([pd.get_dummies(df_reg['year'], prefix='year',drop_first=True),df_reg[['white','hispanic','sex']],df_reg[['experience','experience_sq']], pd.get_dummies(df_reg['major_d'], prefix='occupation',drop_first=True)], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
        y_ = df_reg['incwage_cpiu_2010'].apply(np.log1p).rename('log_wage') if target_var=='wage' else df_reg['fulltime']
        model = LinearRegression(fit_intercept=True).fit(X_, y_)
        yhat = model.predict(X_)
        SS_Residual = sum((y_-yhat)**2)       
        SS_Total = sum((y_-np.mean(y_))**2)     
        r_squared = 1 - (float(SS_Residual))/SS_Total
        adjusted_r_squared = 1 - (1-r_squared)*(len(y_)-1)/(len(y_)-X_.shape[1]-1)
        print(f'Adj. R-squared: {adjusted_r_squared:.4f}')
        # major_wage = df_reg['major']
        # ## Generate an integer index for each major-year pair for clustering
        # major_year_ = df_reg['major_d'].astype(str) + df_reg['year'].astype(str)
        # major_year_ = major_year_.astype('category').cat.codes
        # result_ = sm.OLS(y_, sm.add_constant( X_.copy() ) ).fit(cov_type='cluster',cov_kwds={'groups': major_year_})
        # print(f'Adj. R-squared: {result_.rsquared_adj:.4f}')
        # del result_


## Regression with occupation dummies
for target_var in ['wage']:#,'emp'
    print(f'Individual-level {target_var} regression with occupation dummies.')
    df_reg = wage_emp_choice(target_var) 
    X_ = pd.concat([pd.get_dummies(df_reg['year'], prefix='year',drop_first=True),df_reg[['white','hispanic','sex']],df_reg[['experience','experience_sq']], pd.get_dummies(df_reg['occupation'], prefix='occupation',drop_first=True)], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
    y_ = df_reg['incwage_cpiu_2010'].apply(np.log1p).rename('log_wage') if target_var=='wage' else df_reg['fulltime']
    major_wage = df_reg['major']
    ## Generate an integer index for each major-year pair for clustering
    major_year_ = df_reg['major_d'].astype(str) + df_reg['year'].astype(str)
    major_year_ = major_year_.astype('category').cat.codes
    results_, _ = reg_with_mci_indices(X_, y_, df_reg, main_indices = ['MCI_HHI_d'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year_, weighted_reg=False, weight_vec=None)
    summary_ = summary_col(results_, stars=True,
                                # model_names=col_names_d_hhi,
                                info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                'No. observations' : lambda x: f"{int(x.nobs):d}"})
    table_df = sort_table_rows(summary_, ['MCI_HHI_d'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
    table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_occ_dummies.csv')
    del results_

    ## Regression with occupation dummies AND major category dummies
    if True:
        print(f'Individual-level {target_var} regression with occupation dummies AND major categories.')
        df_reg = pd.get_dummies(df_reg, columns=['Category'], drop_first=False)
        df_reg = df_reg.drop(columns=['Category_Other'])
        X_ = pd.concat([pd.get_dummies(df_reg['year'], prefix='year',drop_first=True),df_reg[['white','hispanic','sex']],df_reg[['experience','experience_sq']], pd.get_dummies(df_reg['occupation'], prefix='occupation',drop_first=True)], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
        # X_ = pd.concat([X_, pd.get_dummies(df_reg['Category'], prefix='Category', drop_first=False)],axis=1)
        # X_ = X_.drop(columns=['Category_Other'])
        results_ind_d_stem, col_names_d_stem = reg_with_mci_indices(X_, y_, df_reg, main_indices = ['MCI_HHI_d'],other_indices=['STEM'], add_exp_int=False, cluster=True, cluster_vec=major_year_,weighted_reg=False, weight_vec=None)
        results_ind_d_cat, col_names_d_cat = reg_with_mci_indices(X_, y_, df_reg, main_indices = ['MCI_HHI_d'],other_indices=df_reg.columns[df_reg.columns.str.startswith('Category')].tolist(), add_exp_int=False, cluster=True, cluster_vec=major_year_,weighted_reg=False, weight_vec=None, other_each=False,other_together=True,other_together_only=True)
        summary_wage_ind_d_cat = summary_col(results_ind_d_stem+results_ind_d_cat[1:], stars=True,
                                    # model_names=col_names_d_cat,
                                    info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                    'No. observations' : lambda x: f"{int(x.nobs):d}"})
        table_df = sort_table_rows(summary_wage_ind_d_cat, ['MCI_HHI_d'], ['STEM']+ df_reg.columns[df_reg.columns.str.startswith('Category')].tolist() + ['const','sex','white','hispanic','experience','experience_sq'])
        table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_cat_and_occ.csv')
        del results_ind_d_stem, results_ind_d_cat






## Regression with interaction terms
for target_var in ['wage','emp']:#
    print(f'Individual-level {target_var} regression with interaction terms.')
    df_reg = wage_emp_choice(target_var) 
    df_reg['MCI*sex'] = df_reg['MCI_HHI_d'] * df_reg['sex']
    df_reg['MCI*white'] = df_reg['MCI_HHI_d'] * df_reg['white']
    df_reg['MCI*hispanic'] = df_reg['MCI_HHI_d'] * df_reg['hispanic']
    X_, y_, major_year_ = create_Xy(df_reg, target_var)
    controls = ['MCI*sex','MCI*white','MCI*hispanic']
    results1_, _ = reg_with_mci_indices(X_, y_, df_reg, main_indices = ['MCI_HHI_d'],other_indices=controls, add_exp_int=False, cluster=True, cluster_vec=major_year_,weighted_reg=False, weight_vec=None,other_each=True)
    summary_ = summary_col(results1_, stars=True,
                                # model_names=col_names_d_hhi,
                                info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                'No. observations' : lambda x: f"{int(x.nobs):d}"})
    table_df = sort_table_rows(summary_, ['MCI_HHI_d'], controls+['const','sex','white','hispanic','experience','experience_sq'])
    table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_interaction.csv')
    del results1_


## Regression with NSSE features.
## Use CIP2-degfield mapping
df_cip2_mapping = pd.read_stata('data/raw_data/acs/CIP2_Final.dta')

dta_nsse = pd.read_stata('data/raw_data/acs/NSSE_CIP2.dta',iterator=True)
df_nsse = pd.read_stata('data/raw_data/acs/NSSE_CIP2.dta')
df_nsse = df_nsse.merge(df_cip2_mapping, left_on='CIP2', right_on='CIP2',how='left')

cip2_in_mapping = df_cip2_mapping['CIP2'].unique()
cip2_in_nsse = df_nsse['CIP2'].unique()
print(f'CIP2 in df_cip2_mapping but not in NSSE: {set(cip2_in_mapping) - set(cip2_in_nsse)}')
print(f'CIP2 in NSSE but not in df_cip2_mapping: {set(cip2_in_nsse) - set(cip2_in_mapping)}')
print(f'Number of dropped CIP2 in NSSE: {len(set(cip2_in_mapping) - set(cip2_in_nsse))}')


for target_var in ['wage','emp']:
    df_reg = wage_emp_choice(target_var) 
    major_d_in_df_reg = df_reg['major_d'].unique()
    ACS_Majorcode_in_df_nsse = df_nsse['ACS_Majorcode'].unique()
    print(f'Major in df_reg but not in ACS_Majorcode: {set(major_d_in_df_reg) - set(ACS_Majorcode_in_df_nsse)}')
    print(f'Major in ACS_Majorcode but not in df_reg: {set(ACS_Majorcode_in_df_nsse) - set(major_d_in_df_reg)}')

    df_reg_with_nsse = df_reg.merge(df_nsse,left_on='major_d',right_on='ACS_Majorcode',how='left')
    df_reg_with_nsse.dropna(inplace=True)
    print('Sample with NSSE:',df_reg_with_nsse.shape[0])
    print('Matched CIP2 majors:',len(df_reg_with_nsse['CIP2'].unique() ))
    
    #Wage Regression with NSSE features
    X_, y_, major_year_ = create_Xy(df_reg_with_nsse, target_var)
    # X_ = pd.concat([pd.get_dummies(df_reg_with_nsse['year'], prefix='year',drop_first=True),df_reg_with_nsse[['white','hispanic','sex']],df_reg_with_nsse[['experience','experience_sq']] ], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
    # y_ = df_reg_with_nsse['incwage_cpiu_2010'].apply(np.log1p).rename('log_wage') if target_var=='wage' else df_reg_with_nsse['fulltime']
    # major_wage = df_reg_with_nsse['major']
    # ## Generate an integer index for each major-year pair for clustering
    # major_year_ = df_reg_with_nsse['major_d'].astype(str) + df_reg_with_nsse['year'].astype(str)
    # major_year_ = major_year_.astype('category').cat.codes

    print(f'Individual-level {target_var} regression with NSSE featuers.')
    controls = ['SATV', 'SATM','InternationalStudent','Edu_F_BAabove','Edu_M_BAabove']
    results1_, _ = reg_with_mci_indices(X_, y_, df_reg_with_nsse, main_indices = ['MCI_HHI_d'],other_indices=controls, add_exp_int=False, cluster=True, cluster_vec=major_year_,weighted_reg=False, weight_vec=None,other_each=False)
    summary_ = summary_col(results1_, stars=True,
                                # model_names=col_names_d_hhi,
                                info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                'No. observations' : lambda x: f"{int(x.nobs):d}"})
    table_df = sort_table_rows(summary_, ['MCI_HHI_d'], controls+['const','sex','white','hispanic','experience','experience_sq'])
    table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_nsse_control.csv')
    del results1_




## NLSY regression
df_nlsy = pd.read_stata('data/raw_data/nlsy/Combined_NLSY.dta')
df_nlsy['fulltime'] = (df_nlsy['Weeks']>=40) & (df_nlsy['Hours']>=1200)
df_nlsy['CIP2'] = df_nlsy['CIP2'].astype(int)
df_nlsy['experience'] = df_nlsy['Age_BDate'] - 22
df_nlsy['experience_sq'] = df_nlsy['experience']**2

cip2_to_degfield = pd.read_stata('data/raw_data/acs/degfield_CIP2.dta')
df_nlsy = df_nlsy.merge(cip2_to_degfield, on='CIP2',how='left')#

mci_for_cip2 = pd.read_csv(f'results/acs/MCI_Maj_{binarize_method}{binarize_threshold:.2f}_{"degfield"}_soc{soc_level}.csv',index_col=0)
mci_for_cip2 = mci_for_cip2[['mci_b_from_hhi']].reset_index().rename(columns={'major':'degfield','mci_b_from_hhi':'MCI_HHI'})
mci_for_cip2['MCI_HHI'] = scaler.fit_transform(mci_for_cip2[['MCI_HHI']])
df_nlsy = df_nlsy.merge(mci_for_cip2, on='degfield',how='left')#

#Log the family income
df_nlsy['Income_HH_15_19'] = np.log1p(df_nlsy['Income_HH_15_19'])

df_wage_nlsy = df_nlsy.copy()[(df_nlsy['Income_2010']>501) & (df_nlsy['fulltime']==1)]
df_emp_nlsy = df_nlsy
sat_total = ['ACT_SAT']
afqt_total = ['AFQT']
trans_total = ['Trans_HS_GPA_Overall']
sat_vars = ['SAT_M', 'SAT_V']
afqt_vars= ['ASVAB_GS',
       'ASVAB_AR', 'ASVAB_WK', 'ASVAB_PC', 'ASVAB_AI', 'ASVAB_SI', 'ASVAB_MK',
       'ASVAB_MC', 'ASVAB_EI', 'ASVAB_AO', 'ASVAB_CS', 'ASVAB_NO']
trans_vars = ['Trans_HS_GPA_Overall', 'Trans_HS_Eng_Overall',
       'Trans_HS_Foreign_Overall', 'Trans_HS_Math_Overall',
       'Trans_HS_SocialScience_Overall', 'Trans_HS_LifeScience_Overall']
parent_vars = ['Income_HH_15_19', 'Dad_Edu', 'Mom_Edu']


for target_var in ['wage','emp']:
    df_reg = df_wage_nlsy.copy() if target_var=='wage' else df_emp_nlsy.copy()
    X = df_reg[['Female','Black', 'Hispanic','experience','experience_sq','MCI_HHI']]
    X = pd.concat([X,pd.get_dummies(df_reg['Year'], prefix='year',drop_first=True)],axis=1)
    y = np.log1p(df_reg['Income_2010']) if target_var=='wage' else df_reg['fulltime']
    major_year_ = df_reg['degfield'].astype(str) + df_reg['Year'].astype(str)
    major_year_ = major_year_.astype('category').cat.codes

    # The total score 
    results_ = []
    ## Sample restriction applied to create consistent samles
    X_ = pd.concat([X.copy(),df_reg[sat_total+afqt_total+trans_total+parent_vars]],axis=1)
    idx = ~X_.isna().any(axis=1)

    print(f'Sample size: {idx.sum()}, number of individuals: {df_reg[idx]["ID"].unique().shape[0]}')

    ## Regression
    result1_ = sm.OLS(y[idx], sm.add_constant( X.copy()[idx] )).fit(cov_type='cluster',cov_kwds={'groups':major_year_[idx]})
    results_.append(copy.copy(result1_) )
    del X_, result1_
    for c in [sat_total,afqt_total,trans_total,parent_vars]:
        X_ = pd.concat([X.copy(),df_reg[c]],axis=1)
        # idx = ~X_.isna().any(axis=1)
        result1_ = sm.OLS(y[idx], sm.add_constant( X_[idx] )).fit(cov_type='cluster',cov_kwds={'groups':major_year_[idx]})
        results_.append(copy.copy(result1_) )
        del X_, result1_
    X_ = pd.concat([X.copy(),df_reg[sat_total+afqt_total+trans_total+parent_vars]],axis=1)
    result1_ = sm.OLS(y[idx], sm.add_constant( X_[idx] )).fit(cov_type='cluster',cov_kwds={'groups':major_year_[idx]})
    results_.append(result1_)
    del result1_

    summary_ = summary_col(results_, stars=True,
                                # model_names=col_names_d_hhi,
                                info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                'No. observations' : lambda x: f"{int(x.nobs):d}"})
    table_df = sort_table_rows(summary_, ['MCI_HHI'], sat_total+afqt_total+trans_total+parent_vars)
    table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_nlsy.csv')



    # Score in each subject
    results_ = []
    X_ = pd.concat([X.copy()],axis=1)
    ## In here, we allow the sample size to vary.
    idx = ~X_.isna().any(axis=1)
    result1_ = sm.OLS(y[idx], sm.add_constant( X_[idx] )).fit(cov_type='cluster',cov_kwds={'groups':major_year_[idx]})
    results_.append(copy.copy(result1_) )
    del X_, result1_
    for c in [sat_vars,afqt_vars,trans_vars,parent_vars]:
        X_ = pd.concat([X.copy(),df_reg[c]],axis=1)
        ## In here, we allow the sample size to vary.
        idx = ~X_.isna().any(axis=1)
        result1_ = sm.OLS(y[idx], sm.add_constant( X_[idx] )).fit(cov_type='cluster',cov_kwds={'groups':major_year_[idx]})
        results_.append(copy.copy(result1_) )
        del X_, result1_
    X_ = pd.concat([X.copy(),df_reg[sat_vars+afqt_vars+trans_vars+parent_vars]],axis=1)
    idx = ~X_.isna().any(axis=1)
    result1_ = sm.OLS(y[idx], sm.add_constant( X_[idx] )).fit(cov_type='cluster',cov_kwds={'groups':major_year_[idx]})
    results_.append(result1_)
    del result1_

    summary_ = summary_col(results_, stars=True,
                                # model_names=col_names_d_hhi,
                                info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                'No. observations' : lambda x: f"{int(x.nobs):d}"})
    table_df = sort_table_rows(summary_, ['MCI_HHI'], sat_vars+afqt_vars+trans_vars+parent_vars)
    table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_nlsy_detail.csv')








## MCI only with age 25-30
df_mci30 = df_mci[df_mci['experience']<=30-22]
print(df_mci30.shape)
df_adj30 = df_mci30.groupby(['major','occupation']).size().unstack(fill_value=0)
occupation_too_small = df_adj30.sum(axis=0)[ df_adj30.sum(axis=0)<=100/2].index
df_adj30.drop(columns=occupation_too_small,inplace=True)
print(df_adj30.shape)
df_mci30 = df_mci30[~df_mci30['occupation'].isin(occupation_too_small)]
df_adj30 = df_mci30.groupby(['major','occupation']).size().unstack(fill_value=0)
print(df_adj30.shape)

major_share = (df_adj30.T/df_adj30.sum(axis=1) ).T #Share of an occupation within a major
occupation_share = df_adj30.sum(axis=0)/df_adj30.sum().sum()
rca = major_share/occupation_share
major_occupation_matrix = (rca>binarize_threshold).astype(int)
major_weight = df_adj30.divide(df_adj30.sum(axis=1), axis=0)
frac = major_weight.transpose()
Blom_hhi = (frac**2).sum()
mci_init_hhi = Blom_hhi*100
mci_b_from_hhi30 = compute_mci_hajime(major_occupation_matrix, mci_init=mci_init_hhi,iter=iterations)

## Wage regression with MCI under 30
scaler = StandardScaler()
mci_b_from_hhi30 = pd.DataFrame(scaler.fit_transform(mci_b_from_hhi30),index=mci_b_from_hhi30.index,columns=mci_b_from_hhi30.columns).reset_index().rename(columns={'major':'major_d'})
df_wage30 = df_wage.merge(mci_b_from_hhi30[['major_d','iter_25']], left_on='major_d', right_on='major_d', how='left').rename(columns={'iter_25':'MCI_HHI30'})
df_emp30 = df_emp.merge(mci_b_from_hhi30[['major_d','iter_25']], left_on='major_d', right_on='major_d', how='left').rename(columns={'iter_25':'MCI_HHI30'})

for target_var in ['wage','emp']:
    print(f'Individual-level {target_var} regression with HHI-based MCI under 30.')
    df_reg = df_wage30 if target_var=='wage' else df_emp30
    X_, y_, major_year_ = create_Xy(df_reg, target_var)
    # X_ = pd.concat([pd.get_dummies(df_reg['year'], prefix='year',drop_first=True),df_reg[['white','hispanic','sex']],df_reg[['experience','experience_sq']]], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
    # y_ = df_reg['incwage_cpiu_2010'].apply(np.log1p).rename('log_wage') if target_var=='wage' else df_reg['fulltime']
    # major_ = df_reg['major']
    # ## Generate an integer index for each major-year pair for clustering
    # major_year_ = df_wage['major_d'].astype(str) + df_wage['year'].astype(str)
    # major_year_ = major_year_.astype('category').cat.codes

    results_ind_d_hhi30, col_names_d_hhi = reg_with_mci_indices(X_, y_, df_reg, main_indices = ['MCI_HHI30'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year_,weighted_reg=False, weight_vec=None)
    summary_wage_ind_d_hhi30 = summary_col(results_ind_d_hhi30, stars=True,
                                model_names=col_names_d_hhi,
                                info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                'No. observations' : lambda x: f"{int(x.nobs):d}"})
    table_df = sort_table_rows(summary_wage_ind_d_hhi30, ['MCI_HHI30'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
    table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_under30.csv')
    del results_ind_d_hhi30




## MCI only with female
df_mci_female = df_mci[df_mci['sex']==1]
df_adj_female = df_mci_female.groupby(['major','occupation']).size().unstack(fill_value=0)
occupation_too_small = df_adj_female.sum(axis=0)[ df_adj_female.sum(axis=0)<=100/2].index
df_adj_female.drop(columns=occupation_too_small,inplace=True)
print(df_adj_female.shape)
df_mci_female = df_mci[~df_mci['occupation'].isin(occupation_too_small)]
df_adj_female = df_mci_female.groupby(['major','occupation']).size().unstack(fill_value=0)
print(df_adj_female.shape)

major_share = (df_adj_female.T/df_adj_female.sum(axis=1) ).T #Share of an occupation within a major
occupation_share = df_adj_female.sum(axis=0)/df_adj_female.sum().sum()
rca = major_share/occupation_share
major_occupation_matrix = (rca>binarize_threshold).astype(int)
major_weight = df_adj_female.divide(df_adj_female.sum(axis=1), axis=0)
frac = major_weight.transpose()
Blom_hhi = (frac**2).sum()
mci_init_hhi = Blom_hhi*100
mci_b_from_hhi_female = compute_mci_hajime(major_occupation_matrix, mci_init=mci_init_hhi,iter=iterations)

## Wage regression with MCI only female
scaler = StandardScaler()
mci_b_from_hhi_female = pd.DataFrame(scaler.fit_transform(mci_b_from_hhi_female),index=mci_b_from_hhi_female.index,columns=mci_b_from_hhi_female.columns).reset_index().rename(columns={'major':'major_d'})
df_wage_female = df_wage.merge(mci_b_from_hhi_female[['major_d','iter_25']], left_on='major_d', right_on='major_d', how='left').rename(columns={'iter_25':'MCI_HHI_female'})
df_wage_female = df_wage_female[df_wage_female['sex']==1]
df_emp_female = df_emp.merge(mci_b_from_hhi_female[['major_d','iter_25']], left_on='major_d', right_on='major_d', how='left').rename(columns={'iter_25':'MCI_HHI_female'})
df_emp_female = df_emp_female[df_emp_female['sex']==1]


for target_var in ['wage','emp']:
    print('Individual-level regression with HHI-based MCI with female only.')
    df_reg = df_wage_female if target_var=='wage' else df_emp_female
    X_, y_, major_year_ = create_Xy(df_reg, target_var)
    results_ind_d_hhi_female, col_names_d_hhi = reg_with_mci_indices(X_, y_, df_reg, main_indices = ['MCI_HHI_female'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year_,weighted_reg=False, weight_vec=None)
    summary_wage_ind_d_hhi_female = summary_col(results_ind_d_hhi_female, stars=True,
                                model_names=col_names_d_hhi,
                                info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                                'No. observations' : lambda x: f"{int(x.nobs):d}"})
    table_df = sort_table_rows(summary_wage_ind_d_hhi_female, ['MCI_HHI_female'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
    table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_female.csv')
    del results_ind_d_hhi_female


# X_wage_female = pd.concat([pd.get_dummies(df_wage_female['year'], prefix='year',drop_first=True),df_wage_female[['white','hispanic']],df_wage_female[['experience','experience_sq']]], axis=1) #pd.get_dummies(df_wage['major'], prefix='major',drop_first=True),
# y_wage_female = df_wage_female['incwage_cpiu_2010'].apply(np.log1p).rename('log_wage')
# major_wage_female = df_wage_female['major']
# ## Generate an integer index for each major-year pair for clustering
# major_year_wage_female = df_wage_female['major_d'].astype(str) + df_wage_female['year'].astype(str)
# major_year_wage_female = major_year_wage_female.astype('category').cat.codes
# target_var = 'wage'

# df_reg, X, y, major_year = df_wage_female, X_wage_female, y_wage_female, major_year_wage_female
# results_ind_d_hhi_female, col_names_d_hhi = reg_with_mci_indices(X, y, df_reg, main_indices = ['MCI_HHI_female'],other_indices=['HHI_d','TOP3_d','GINI'], add_exp_int=False, cluster=True, cluster_vec=major_year,weighted_reg=False, weight_vec=None)
# summary_wage_ind_d_hhi_female = summary_col(results_ind_d_hhi_female, stars=True,
#                             model_names=col_names_d_hhi,
#                             info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
#                                             'No. observations' : lambda x: f"{int(x.nobs):d}"})
# table_df = sort_table_rows(summary_wage_ind_d_hhi_female, ['MCI_HHI_female'], ['HHI_d','TOP3_d','GINI','const','sex','white','hispanic','experience','experience_sq'])
# table_df.to_csv(f'results/acs/summary_{target_var}_ind_d_hhi_female.csv')
# del results_ind_d_hhi_female



## MCI by year
occupation_list_year = []
for year in range(2009,2020):
    print(year)
    df_mci_year = df_mci[df_mci['year']==year]
    df_adj_year = df_mci_year.groupby(['major','occupation']).size().unstack(fill_value=0)
    occupation_too_small = df_adj_year.sum(axis=0)[ df_adj_year.sum(axis=0)<=100].index
    ## Drop if occupation is too small
    df_adj_year.drop(columns=occupation_too_small,inplace=True)
    print(df_adj_year.shape)
    occupation_list_year.append(df_adj_year.columns.tolist())

occupation_list_in_all_years = list(set.intersection(*map(set,occupation_list_year)))

df_mci = df_mci[df_mci['occupation'].isin(occupation_list_in_all_years)]
print('New df_mci:',df_mci.shape)


matrix_by_year = {}
major_weight_by_year = {}
for year in range(2009,2020):
    print(year)
    df_mci_year = df_mci[df_mci['year']==year]
    df_adj_year = df_mci_year.groupby(['major','occupation']).size().unstack(fill_value=0)
    major_share = (df_adj_year.T/df_adj_year.sum(axis=1) ).T #Share of an occupation within a major
    occupation_share = df_adj_year.sum(axis=0)/df_adj_year.sum().sum()
    rca = major_share/occupation_share
    major_occupation_matrix = (rca>binarize_threshold).astype(int)
    major_weight = df_adj_year.divide(df_adj_year.sum(axis=1), axis=0)

    matrix_by_year[year] = major_occupation_matrix
    major_weight_by_year[year] = major_weight

mci_by_year = pd.DataFrame(index=major_weight_by_year[2009].index)
for year in range(2009,2020):
    frac = major_weight_by_year[year].transpose()
    Blom_hhi = (frac**2).sum()
    mci_init_hhi = Blom_hhi*100
    mci_b_from_hhi = compute_mci_hajime(matrix_by_year[year], mci_init=mci_init_hhi,iter=iterations)

    mci_by_year[f'MCI_HHI_{year}'] = mci_b_from_hhi[f'iter_{iterations}']
    mci_by_year[f'HHI_{year}'] = Blom_hhi


print(mci_by_year.corr().iloc[::2,::2])
mci_by_year.to_csv(f'data/processed_data/acs/MCI_by_year_{major_var}_soc{soc_level}.csv')

scaler = StandardScaler()
mci_by_year_scaled = pd.DataFrame(scaler.fit_transform(mci_by_year),index=mci_by_year.index,columns=mci_by_year.columns)



## Future wage prediction
df_wage = pd.read_csv(f'data/processed_data/acs/df_reg_{"degfieldd"}_soc{4}.csv',index_col=0)
df_grouped = df_wage.groupby(['year','major'])['sex','white','hispanic','incwage_cpiu_2010'].mean().reset_index()
df_grouped['log_wage'] = np.log1p(df_grouped['incwage_cpiu_2010'])
df_grouped['MCI'] = None
# df_pivot = df_grouped.pivot(index='year', columns='major')
for year in range(2009,2020):
    for m in mci_by_year_scaled.index.unique():
        df_grouped.loc[(df_grouped['year']==year)&(df_grouped['major']==m),'MCI'] = mci_by_year_scaled.loc[m,f'MCI_HHI_{year}']
        df_grouped.loc[(df_grouped['year']==year)&(df_grouped['major']==m),'HHI'] = mci_by_year_scaled.loc[m,f'HHI_{year}']


results_ = []
result_hhi_ = []
for y_ahead in [0,1,2,5,10]:
    df_grouped['y_future'] = np.log1p(df_grouped.groupby('major')['incwage_cpiu_2010'].shift(-y_ahead))
    df_ = df_grouped.dropna()

    if y_ahead==0:
        X1 = df_[['sex','white','hispanic','MCI']].astype(float)
    else:
        X1 = df_[['sex','white','hispanic','log_wage','MCI']].astype(float)
    y = df_['y_future']
    result = sm.OLS(y, sm.add_constant(X1)).fit()
    results_.append(copy.copy(result))

    if y_ahead==0:
        X2 = df_[['sex','white','hispanic','HHI']]
    else:
        X2 = df_[['sex','white','hispanic','log_wage','HHI']]
    result_hhi = sm.OLS(y, sm.add_constant(X2)).fit()
    result_hhi_.append(copy.copy(result_hhi))



summary_ = summary_col(results_, stars=True,
                            model_names=['0 year','1 year','2 year','5 year','10 year'],
                            info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                            'No. observations' : lambda x: f"{int(x.nobs):d}"})
table_df = sort_table_rows(summary_, ['MCI'], ['log_wage'])
table_df.to_csv(f'results/acs/summary_wage_pred.csv')

summary_hhi_ = summary_col(result_hhi_, stars=True,
                            model_names=['0 year','1 year','2 year','5 year','10 year'],
                            info_dict={#'R-squared' : lambda x: f"{x.rsquared:.2f}",
                                            'No. observations' : lambda x: f"{int(x.nobs):d}"})
table_df_hhi = sort_table_rows(summary_hhi_, ['HHI'], ['log_wage'])
table_df_hhi.to_csv(f'results/acs/summary_wage_pred_hhi.csv')




## MCI by year with N year window
## Use degfield as major?
major_var = 'degfieldd'
df_mci = pd.read_csv(f'data/processed_data/acs/df_mci_{major_var}_soc{soc_level}.csv',index_col=0)
occupation_list_year = []
y_window = 5
for year_start in range(2009,2021-y_window):
    print(year_start,'-',year_start+y_window)
    df_mci_year = df_mci[(df_mci['year']>=year_start) & (df_mci['year']<year_start+y_window)]
    df_adj_year = df_mci_year.groupby(['major','occupation']).size().unstack(fill_value=0)
    occupation_too_small = df_adj_year.sum(axis=0)[ df_adj_year.sum(axis=0)<=100].index
    ## Drop if occupation is too small
    df_adj_year.drop(columns=occupation_too_small,inplace=True)
    print(df_adj_year.shape)
    occupation_list_year.append(df_adj_year.columns.tolist())

occupation_list_in_all_years = list(set.intersection(*map(set,occupation_list_year)))

df_mci = df_mci[df_mci['occupation'].isin(occupation_list_in_all_years)]
print('New df_mci:',df_mci.shape)
print('Number of occupations:',len(occupation_list_in_all_years))

matrix_by_year = {}
major_weight_by_year = {}
for year_start in range(2009,2021-y_window):
    print(year_start,'-',year_start+y_window)
    df_mci_year = df_mci[(df_mci['year']>=year_start) & (df_mci['year']<year_start+y_window)]
    df_adj_year = df_mci_year.groupby(['major','occupation']).size().unstack(fill_value=0)
    major_share = (df_adj_year.T/df_adj_year.sum(axis=1) ).T #Share of an occupation within a major
    occupation_share = df_adj_year.sum(axis=0)/df_adj_year.sum().sum()
    rca = major_share/occupation_share
    major_occupation_matrix = (rca>binarize_threshold).astype(int)
    major_weight = df_adj_year.divide(df_adj_year.sum(axis=1), axis=0)

    matrix_by_year[year_start] = major_occupation_matrix
    major_weight_by_year[year_start] = major_weight

mci_by_year = pd.DataFrame(index=major_weight_by_year[2009].index)
for year_start in range(2009,2021-y_window):
    frac = major_weight_by_year[year_start].transpose()
    Blom_hhi = (frac**2).sum()
    mci_init_hhi = Blom_hhi*100
    mci_b_from_hhi = compute_mci_hajime(matrix_by_year[year_start], mci_init=mci_init_hhi,iter=iterations)

    mci_by_year[f'MCI_HHI_{year_start}'] = mci_b_from_hhi[f'iter_{iterations}']
    # mci_by_year[f'HHI_{year_start}'] = Blom_hhi

mci_by_year = pd.DataFrame(scaler.fit_transform(mci_by_year), index=mci_by_year.index, columns=mci_by_year.columns) 
df_name = pd.read_csv(f'data/processed_data/acs/{major_var}_mapping.csv', index_col=0)

##
mci_ranking_by_year = pd.DataFrame(index=mci_by_year.index)
for year_start in range(2009,2021-y_window):
    mci_ranking_by_year[f'MCI_HHI_{year_start}'] = mci_by_year[f'MCI_HHI_{year_start}'].rank(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
k = 0
for m in mci_by_year.index:
    original_line = mci_ranking_by_year.loc[m].values
    if np.abs(original_line.max() - original_line.min()) >=7:
        ax.plot(np.arange(2009,2021-y_window),original_line,linewidth=2.,alpha=1.,label=df_name.loc[m].values[0]   )#, color=sns.color_palette("tab20")[k]
        k += 1
    else:
        ax.plot(np.arange(2009,2021-y_window),original_line,linewidth=0.5,alpha=0.5, label='_nolegend_',color='grey' )

ax.invert_yaxis()
ax.set_xlabel('Year')
ax.set_ylabel('Ranking')
# ax.set_title('Ranking of MCI based on GINI')
ax.set_xticklabels([f'{2009+i}' for i in range(11)])
## Add the legend outside the plot, but the legend needs to be inside the plot to be able to use the label
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
## Reduce the font size of the legend
plt.setp(ax.get_legend().get_texts(), fontsize='7')
## Fit the legend to figure size
fig.tight_layout()
fig.savefig(f'results/acs/MCI_change_over_year.png',dpi=300)

## Ranking of majors in the first and last period
mci_ranking_by_year.iloc[:,[0,-1]].merge(df_name, left_index=True, right_index=True).sort_values(by='MCI_HHI_2009',ascending=True).to_csv(f'results/acs/mci_ranking_2009-13_vs_15-19.csv')

print(mci_ranking_by_year.iloc[:,[0,-1]].corr() )



