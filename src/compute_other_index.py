'''
Compute the indices other than MOS such as Gini and Top3  for comparison.
It assumes the data is prepared b ACS_data_prep.py.
'''
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression



## Compute gini
#NOTE: With degfield, we don't have to drop any occuaption.
#NOTE: With degfieldd, we have to drop Engineering/architecture, Editors/writers/performers, Research/scientists/technical, Other/military (Dropped) 
#in order to have a fully connected graph.
df_SOC2_11OCC = pd.read_excel('data/raw_data/acs/SOC2_11OCC.xlsx',sheet_name='Sheet1')
# df = pd.read_csv(f'data/processed_data/acs/df_mci_degfield_soc2.csv',index_col=0)
# df = pd.read_csv(f'data/processed_data/acs/df_mci_degfieldd_soc2.csv',index_col=0)
df_wage = pd.read_csv(f'data/processed_data/acs/df_reg_degfield_soc2.csv', index_col=0)

## Apply 25-35 age restriction
df_wage = df_wage[(df_wage['experience']>=25-22) & (df_wage['experience']<=35-22)]



df = df_wage.merge(df_SOC2_11OCC,left_on='occupation',right_on='SOC2')
df['occupation'] = df['Code_11OCC']
## Drop military occupations from GINI computation
df = df[df['occupation']!=12]

occupation_list = df['occupation'].unique()
major_list = df['major'].unique()
occupation_list = df['occupation'].unique()
print(f'Number of (major,occupation):{len(major_list)},{len(occupation_list)}')
print(occupation_list)
n_major = len(major_list)
n_occupation = len(occupation_list)

# Need to restrict the major and occupations so that they form a fully connected graph.
# Otherwise the wage regression will not be identified.
df_adj = df.groupby(['major','occupation']).size().unstack(fill_value=np.nan)
# List of majors that contain 0
occupation_list_refined = df_adj.dropna(axis=1).columns
n_occupation = len(occupation_list_refined)
print('Number of occupations after refinement:',n_occupation)
print(occupation_list_refined)

# Restrict df so that 'occupation' is in occupation_list_refined
df = df[df['occupation'].isin(occupation_list_refined)]
df_adj = df_adj[occupation_list_refined]

## Experiment
# df = df[df['year']<=2015]
# df_adj = df.groupby(['major','occupation']).size().unstack(fill_value=np.nan)[occupation_list_refined]


## Confusing notation in the original paper. After clarifying with the authors, the weight is across majors within occupation.
# df_w = df_adj/df_adj.sum(axis=0) #df_adj.div(df_adj.sum(axis=1),axis=0)
df_w = (df_adj.T/df_adj.sum(axis=1)).T #df_adj.div(df_adj.sum(axis=1),axis=0)


## df_w.sum(axis=0)==1

gamma = {}
for occ in occupation_list_refined:
    print(f'Occupation {occ}:')
    df_occ = df[df['occupation']==occ]
    assert len(df_occ['major'].unique())==n_major, 'Number of majors does not match'
    log_earn = np.log(df_occ['incwage_cpiu_2010']+1.)
    X_occ = pd.concat([pd.get_dummies(df_occ['major'], prefix='major',drop_first=True),pd.get_dummies(df_occ['year'], prefix='year',drop_first=True),df_occ[['white','hispanic','sex']],df_occ[['experience','experience_sq']]], axis=1)
    assert pd.get_dummies(df_occ['major'], prefix='major').shape[1]==n_major, 'Number of majors does not match'
    model = LinearRegression(fit_intercept=True).fit(X_occ, log_earn)
    # gamma_o = model.coef_[:n_major] - np.mean(model.coef_[:n_major])
    coef = pd.Series(model.coef_, index=X_occ.columns)

    # Find majors that are dropped and not in the model but in major_list
    missing_majors = [major for major in major_list if f'major_{major}' not in coef.index]
    print(f'Missing majors: {missing_majors}')
    assert len(missing_majors)<=1, 'More than one major is missing.'
    coef[f'major_{missing_majors[0]}'] = 0. 

    gamma_o = coef.loc[coef.index.str.startswith('major')].values - np.mean(coef.loc[coef.index.str.startswith('major')].values)
    # add the coefficient of the first (omitted) major
    gamma[occ] = pd.Series(gamma_o, index =coef.loc[coef.index.str.startswith('major')].index)

# Turn gamma into a dataframe.
df_gamma = pd.DataFrame(gamma)
df_gamma.index = df_gamma.index.str.replace('major_','').astype(int)

# Compute GINI
'''
# Seems to have a bug. Need to check.
abs_diff = np.abs(df_gamma.values[:, :, np.newaxis] - df_gamma.values[:, np.newaxis, :])
gini = ( 1/(2*n_occupation**2) )*np.sum(np.sum(abs_diff,axis=2)*df_w ,axis=1 )
gini.rename('gini_degfield',inplace=True)
gini.to_csv(f'results/acs/gini_degfield.csv')
'''
         
# Compute GINI by iterations.
gini_iter = {}
for m in np.sort(major_list):
    val = 0.
    for j in occupation_list_refined:
        for k in occupation_list_refined:
            ## Wrong notation in the original paper. Weight was supposed to be w[m,j] instead of w[m,k].
            val += (1/(2*n_occupation**2) )*np.abs(df_gamma.loc[m,j]-df_gamma.loc[m,k])#*df_w.loc[m,j]#*df_w.loc[m,k]
    gini_iter[m] = val
gini_iter = pd.Series(gini_iter)
gini_iter.rename('gini_degfield',inplace=True)
gini_iter.index.name = 'major'
gini_iter.to_csv(f'results/acs/gini_degfield.csv')

# Compute GINI with weights
gini_iter2 = {}
for m in np.sort(major_list):
    val = 0.
    for j in occupation_list_refined:
        for k in occupation_list_refined:
            ## Wrong notation in the original paper. Weight was supposed to be w[m,j] instead of w[m,k].
            val += (1/(2*n_occupation**2) )*np.abs(df_gamma.loc[m,j]-df_gamma.loc[m,k])*df_w.loc[m,j]#*df_w.loc[m,k]
    gini_iter2[m] = val
gini_iter2 = pd.Series(gini_iter2)
gini_iter2.rename('gini_degfield',inplace=True)
gini_iter2.index.name = 'major'
gini_iter2.to_csv(f'results/acs/gini_with_weights_degfield.csv')



## Compute top3 and hhi
for soc_level in [2,4]:
    for major_var in ['degfield', 'degfieldd']:


        # Transition matrix
        major_weight = pd.read_csv(f'data/processed_data/acs/major_weight_{major_var}_soc{soc_level}.csv',index_col=0)
        occupation_weight = pd.read_csv(f'data/processed_data/acs/occupation_weight_{major_var}_soc{soc_level}.csv',index_col=0)
        # df_mci = pd.read_csv(f'data/processed_data/acs/df_mci_{major_var}_soc{soc_level}.csv',index_col=0)

        frac = major_weight.transpose()
        # Share of top 3 occupations
        Altonji_top3_share = frac.apply(lambda x: x.nlargest(3).sum())
        # HHI
        Blom_hhi = (frac**2).sum()

        Altonji_top3_share_occ = occupation_weight.apply(lambda x: x.nlargest(3).sum())
        Blom_hhi_occ = (occupation_weight**2).sum()

        '''
        data_dir = 'data/raw_data/acs/'
        df = pd.read_stata(data_dir+f'usa_00016.dta',convert_categoricals=False)
        # Predict log earnings in a particular occupation from major, year, and other characteristics.
        # Characteristics include gender, race, experience and experience squared.
        '''

        ## Extra: compute occupational distinctiveness
        s_is_occ_weight = True
        if s_is_occ_weight:
            S = occupation_weight
        else:
            adj_matrix = pd.read_csv(f'data/processed_data/acs/adj_{major_var}_soc{soc_level}.csv',index_col=0)
            S = adj_matrix/adj_matrix.sum().sum()
        OD = pd.Series(index=S.index)
        for m in S.index:
            val = 0.
            for o in S.columns:
                val +=np.abs( S.loc[m,o] - S.sum(axis=0)[o] )
                # if s_is_occ_weight:
                #     val +=np.abs( 2*S.loc[m,o] - 1. )
                # else:
                #     val +=np.abs( S.loc[m,o] - S.sum(axis=0)[o] )
            OD[m] = val






        # Combine with other indices
        df_index = pd.DataFrame({'Altonji_top3_share':Altonji_top3_share,'Blom_hhi':Blom_hhi,'OD':OD})
        # df_index = pd.DataFrame({'Altonji_top3_share':Altonji_top3_share,'Blom_hhi':Blom_hhi,'gini':gini})
        df_index.to_csv(f'results/acs/other_index_{major_var}_soc{soc_level}.csv')

        df_index_occ = pd.DataFrame({'Altonji_top3_share':Altonji_top3_share_occ,'Blom_hhi':Blom_hhi_occ})
        df_index_occ.to_csv(f'results/acs/other_index_occ_{major_var}_soc{soc_level}.csv')