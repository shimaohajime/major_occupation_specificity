'''
Code to process the ACS data for the MOS project.
'''


import pandas as pd
import numpy as np
import datetime
import pyreadstat
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier

node_size_cutoff = {'soc2':100,'soc4':100,'degfield':100,'degfieldd':100}

# Read data
data_dir = 'data/raw_data/acs/'
df = pd.read_stata(data_dir+f'ACS_combined.dta',convert_categoricals=False)

## Save a mapping between degfield and degfield_s
df_degfield = df.groupby('degfield').first()['degfield_s']
df_degfieldd = df.groupby('degfieldd').first()['degfieldd_s']
df_degfield.to_csv('data/processed_data/acs/degfield_mapping.csv')
df_degfieldd.to_csv('data/processed_data/acs/degfieldd_mapping.csv')



# Preprocessing by Xiao
## Drop if year>=2020
df = df[df['year']<2020]
## drop if degfield2 != 0
df = df[df['degfield2']==0]
# Sex to be 0,1
df['sex'] = df['sex'] - 1.
## Drop if school==2
df = df[df['school']!=2]

## Drop if occsoc is missing or blank
df = df[df['occsoc'].notna()]
df = df[df['occsoc']!='']
## Redefine the occupation code
### replace occsoc="4740XX"  if occsoc=="47XXXX" // 
df['occsoc'] = df['occsoc'].str.replace('47XXXX','4740XX')
df['occsoc2'] = df['occsoc'].str[:2]
df['occsoc3'] = df['occsoc'].str[:3]
df = df[df['occsoc3']!='   ']
### If occsoc3==559, then keep 4 digits to construct occsoc4, otherwise keep 3 digits and add 0 at the end.
df['occsoc4_'] = df['occsoc'].str[:4]
df['occsoc4'] = df['occsoc'].str[:4]
df['occsoc4'] = df['occsoc3'] + '0'
df.loc[df['occsoc3']=='559','occsoc4'] = df.loc[df['occsoc3']=='559','occsoc'].str[:4]
### SOC 6 is just occsoc 
df['occsoc6'] = df['occsoc']


## Redefine the major code
df.loc[df['degfieldd']==4003, 'degfield'] = 36
df.loc[df['degfieldd']==4003, 'degfieldd'] = 3611
df.loc[df['degfieldd']==4008, 'degfield'] = 50
df.loc[df['degfieldd']==4008, 'degfieldd'] = 5098

## Drop too small majors when constructing the graph
df_mci = df[(df['age']>=25) & (df['age']<=35)].copy()
df_mci = df_mci[df_mci['wkswork2']>=4]
df_mci = df_mci[df_mci['uhrswork']>=30] 
degfieldd_list = df['degfieldd'].unique()
n_degfieldd = len(degfieldd_list)
degfieldd_count = df_mci.groupby(['degfieldd']).size()
major_too_small = degfieldd_count[degfieldd_count<=node_size_cutoff[f'degfieldd']].index
print('Dropping individuals who belong to the following majors:')
print(major_too_small)
df = df[~df['degfieldd'].isin(major_too_small)]
print(f'Original majors: {n_degfieldd}, after dropping: {len(df["degfieldd"].unique())}')


## Fork the dataframe for the employment regression before applying full-time constraint.
df_with_unemp = df.copy()


## Apply full-time constraint
## Drop if wkswork2<4
df = df[df['wkswork2']>=4]
## Drop if uhrswork<30
df = df[df['uhrswork']>=30] 
## Construct the explanatory variables
df['experience'] = (df['age'] - 22).astype(float)
df['experience_sq'] = df['experience']**2
#df['white'] = 1 if race==0 and 0 otherwise
df['white'] = (df['race']==1).astype(int)
df['hispanic'] = (df['hispan']!=0).astype(int)
df.dropna(inplace=True, subset=['year','sex','white','hispanic','experience','experience_sq','incwage_cpiu_2010','age','empstat','wkswork2','uhrswork'])

## Similar construction to df_with_unemp
df_with_unemp['experience'] = (df_with_unemp['age'] - 22).astype(float)
df_with_unemp['experience_sq'] = df_with_unemp['experience']**2
df_with_unemp['white'] = (df_with_unemp['race']==1).astype(int)
df_with_unemp['hispanic'] = (df_with_unemp['hispan']!=0).astype(int)
### Define the full-time variable
df_with_unemp['fulltime'] = ( (df_with_unemp['uhrswork']>=30) & (df_with_unemp['wkswork2']>=4) ).astype(int)
df_with_unemp.dropna(inplace=True, subset=['year','sex','white','hispanic','experience','experience_sq','fulltime','age','wkswork2','uhrswork'])



for soc_level in [2,4]:
    for major_var in ['degfield','degfieldd']:

        ## 'degfield' is the major at coarse level, and degfieldd is more detailed. 
        ## "occsoc' is the occupation code. Work with the first 2 or 4 digits now.
        ## Some coding scheme change in occsoc within this period. Need to make them consistent at the end.
        df['major'] = df[major_var]
        df['occupation'] = df[f'occsoc{soc_level}']
        df_with_unemp['major'] = df_with_unemp[major_var]


        # df for MCI
        ## Drop if age<25 or age>35 for MCI
        df_mci = df[(df['age']>=25) & (df['age']<=35)].copy()
        ## Keep if empstat==1
        # df_mci = df_mci[df_mci['empstat']==1]
        # df_mci.drop(columns=['empstat'],inplace=True)
        df_mci = df_mci[['major','occupation','year','white','hispanic','sex','experience','experience_sq','incwage_cpiu_2010']]

        # Construct bipartite network of major-occupation

        # Unique major and occupation
        major_list = df['major'].unique()
        occupation_list = df['occupation'].unique()
        n_major = len(major_list)
        n_occupation = len(occupation_list)
        print(f'Number of (major,occupation):{len(major_list)},{len(occupation_list)}')

        # Count matrix
        df_adj = df_mci.groupby(['major','occupation']).size().unstack(fill_value=0)
        occupation_too_small = df_adj.sum(axis=0)[ df_adj.sum(axis=0)<=node_size_cutoff[f'soc{soc_level}'] ].index
        # major_too_small = df_adj.sum(axis=1)[df_adj.sum(axis=1)<=node_size_cutoff[f'{major_var}'] ].index
        ## Drop if occupation is too small
        df_adj.drop(columns=occupation_too_small,inplace=True)
        # df_adj.drop(index=major_too_small,inplace=True)
        df_adj.to_csv(f'data/processed_data/acs/adj_{major_var}_soc{soc_level}.csv')
        print(f'Adjusted to:{len(major_list)},{len(occupation_list)}')

        df_mci = df_mci[df_mci['occupation'].isin(df_adj.columns)]
        # df_mci = df_mci[df_mci['major'].isin(df_adj.index)]
        print(f'Number of individuals in df_mci:{len(df_mci)}')
        df_mci.to_csv(f'data/processed_data/acs/df_mci_{major_var}_soc{soc_level}.csv')


        # df for wage regression
        ## Drop if age<23 and age>60 for wage regression
        print(f'Number of individuals in df_reg:{len(df)}   ')
        df_reg = df[(df['age']>=25) & (df['age']<=60)].copy()
        print(f'After age restriction in df_reg:{len(df_reg)}')
        # df_reg = df_reg[df_reg['empstat']==1]
        df_reg = df_reg[['major','occupation','year','white','hispanic','sex','experience','experience_sq','incwage_cpiu_2010']]
        ## Drop individuals from df_reg if occupation is too small
        # df_reg = df_reg[df_reg['occupation'].isin(df_adj.columns)]
        df_reg = df_reg[df_reg['major'].isin(df_adj.index)]
        print(f'After major/occupation restriction in df_reg:{len(df_reg)}')
        ## Drop if incwage_cpiu_2010<500
        df_reg = df_reg[df_reg['incwage_cpiu_2010']>=500]
        print(f'After wage threshold in df_reg:{len(df_reg)}')
        df_reg.to_csv(f'data/processed_data/acs/df_reg_{major_var}_soc{soc_level}.csv')

        # df for employment regression
        df_emp = df_with_unemp[(df_with_unemp['age']>=25) & (df_with_unemp['age']<=60)].copy()
        # df_emp = df_emp[df_emp['empstat'].isin([1,2])]
        df_emp = df_emp[['major','year','white','hispanic','sex','experience','experience_sq','fulltime']]
        # df_emp = df_emp[df_emp['occupation'].isin(df_adj.columns)]
        df_emp = df_emp[df_emp['major'].isin(df_adj.index)]
        print(f'Number of individuals in df_emp:{len(df_emp)}')
        print(f'Employment rate in df_emp:{df_emp["fulltime"].mean()}')
        df_emp.to_csv(f'data/processed_data/acs/df_emp_{major_var}_soc{soc_level}.csv')


        # Weighted matrix
        major_weight = df_adj.divide(df_adj.sum(axis=1), axis=0)
        occupation_weight = df_adj.divide(df_adj.sum(axis=0), axis=1)

        major_weight.to_csv(f'data/processed_data/acs/major_weight_{major_var}_soc{soc_level}.csv')
        occupation_weight.to_csv(f'data/processed_data/acs/occupation_weight_{major_var}_soc{soc_level}.csv')

        # Construct the binary matrix.
        ## Methods: Threshold with certain number, or revealed comparative advantage.

        for binarize_method in ['rca']:#,'fixedr','fixedn'
            major_share = (df_adj.T/df_adj.sum(axis=1) ).T #Share of an occupation within a major
            if binarize_method == 'rca':
                binarize_threshold = 1.
                occupation_share = df_adj.sum(axis=0)/df_adj.sum().sum()
                rca = major_share/occupation_share
                major_occupation_matrix = (rca>binarize_threshold).astype(int)

                # a = df_adj
                # b = df_adj.sum(axis=1)
                # c = df_adj.sum(axis=0)
                # d = df_adj.sum().sum()
                # r = ((a.T/b).T)/(c/d)

                # m = 35
                # o = '1130'
                # a1 = df_adj.loc[35,'1130']
                # b1 = df_adj.loc[35,:].sum()
                # c1 = df_adj.loc[:, '1130'].sum()
                # d1 = df_adj.sum().sum()
                # r1 = ((a1/b1))/(c1/d1)
            elif binarize_method == 'fixedr':
                binarize_threshold = .05
                major_occupation_matrix = (major_share>binarize_threshold).astype(int)
            elif binarize_method == 'fixedn':
                binarize_threshold = 10
                major_occupation_matrix = (df_adj>binarize_threshold).astype(int)
            major_occupation_matrix.to_csv(f'data/processed_data/acs/major_occupation_matrix_binary_{binarize_method}{binarize_threshold:.2f}_{major_var}_soc{soc_level}.csv')


            # Controlled MCI
            if True:
                # major->occupation regression
                print(datetime.datetime.now())
                X_major = pd.concat([pd.get_dummies(df_mci['major'], prefix='major',drop_first=True),df_mci[['white','hispanic']],pd.get_dummies(df_mci['sex'], prefix='sex',drop_first=True)], axis=1) #pd.get_dummies(df_mci['year'], prefix='year',drop_first=True),,df_mci[['experience','experience_sq']]
                y_occ = df_mci['occupation']
                # clf = LogisticRegression(random_state=42, max_iter=10000, solver='sag').fit(X_major, y_occ)
                clf = SGDClassifier(loss='log_loss',penalty=None,fit_intercept=True,max_iter=10000,early_stopping=True).fit(X_major, y_occ)
                ## Coefficient matrix
                coef_matrix_ = pd.DataFrame(clf.coef_,columns=X_major.columns,index=clf.classes_)
                ## Find majors that are dropped and not in the model but in major_list
                missing_majors = [major for major in major_list if f'major_{major}' not in coef_matrix_.columns]
                coef_matrix_[f'major_{missing_majors[0]}'] = 0.
                ## major_coef_matrix is a subset of coef_matrix_ where the columns are coef_matrix_.columns.str.startswith('major')
                major_cols = coef_matrix_.columns[coef_matrix_.columns.str.startswith('major')]
                major_coef_matrix = coef_matrix_[major_cols]
                ## Normalize
                major_coef_matrix = coef_matrix_.loc[:,coef_matrix_.columns.str.startswith('major')].transpose()
                major_coef_matrix = (major_coef_matrix - major_coef_matrix.min().min())/(major_coef_matrix.max().max()-major_coef_matrix.min().min()  )
                major_controlled_weight = major_coef_matrix.divide(major_coef_matrix.sum(axis=1), axis=0)
                print(datetime.datetime.now())

                # occupation->major regression
                print(datetime.datetime.now())
                X_occ = pd.concat([pd.get_dummies(df_mci['occupation'], prefix='occ',drop_first=True),df_mci[['white','hispanic']],pd.get_dummies(df_mci['sex'], prefix='sex',drop_first=True) ], axis=1) #pd.get_dummies(df_mci['year'], prefix='year',drop_first=True),,df_mci[['experience','experience_sq']]
                y_major = df_mci['major']
                # clf = LogisticRegression(random_state=42, max_iter=10000, solver='sag').fit(X_occ, y_major)
                clf = SGDClassifier(loss='log_loss',penalty=None,fit_intercept=True,max_iter=10000,early_stopping=True).fit(X_occ, y_major)
                ## Coefficient matrix
                coef_matrix_ = pd.DataFrame(clf.coef_,columns=X_occ.columns,index=clf.classes_)
                ## Find occupations that are dropped and not in the model but in occupation_list
                missing_occ = [occ for occ in occupation_list if f'occ_{occ}' not in coef_matrix_.columns]
                coef_matrix_[f'occ_{missing_occ[0]}'] = 0.
                ## occupation_coef_matrix is a subset of coef_matrix_ where the columns are coef_matrix_.columns.str.startswith('occ')
                occ_cols = coef_matrix_.columns[coef_matrix_.columns.str.startswith('occ')]
                occupation_coef_matrix = coef_matrix_[occ_cols]
                ## Normalize
                occupation_coef_matrix = (occupation_coef_matrix - occupation_coef_matrix.min().min())/(occupation_coef_matrix.max().max()-occupation_coef_matrix.min().min()  )
                occupation_controlled_weight = occupation_coef_matrix.divide(occupation_coef_matrix.sum(axis=0), axis=1)
                print(datetime.datetime.now())


                # Save the controlled weight matrix
                ## Remove 'major_' and 'occ_' from the index and columns 
                major_controlled_weight.index = major_controlled_weight.index.str.replace('major_','')
                occupation_controlled_weight.columns = occupation_controlled_weight.columns.str.replace('occ_','')
                ## Sort the index and columns
                major_controlled_weight = major_controlled_weight.reindex(sorted(major_controlled_weight.index), axis=0)
                major_controlled_weight = major_controlled_weight.reindex(sorted(major_controlled_weight.columns), axis=1)
                occupation_controlled_weight = occupation_controlled_weight.reindex(sorted(occupation_controlled_weight.index), axis=0)
                occupation_controlled_weight = occupation_controlled_weight.reindex(sorted(occupation_controlled_weight.columns), axis=1)
                ## Name the index as 'major' and columns as 'occupation'
                major_controlled_weight.index.name = 'major'
                major_controlled_weight.columns.name = 'occupation'
                occupation_controlled_weight.index.name = 'major'
                occupation_controlled_weight.columns.name = 'occupation'

                major_controlled_weight.to_csv(f'data/processed_data/acs/major_coltrolled_weight_{major_var}_soc{soc_level}.csv')
                occupation_controlled_weight.to_csv(f'data/processed_data/acs/occupation_coltrolled_weight_{major_var}_soc{soc_level}.csv')

            if binarize_method == 'rca':
                df_name = df_degfield if major_var == 'degfield' else df_degfieldd
                df_adj_with_name = df_adj.merge(df_name, left_index=True, right_index=True)
                df_adj_with_name = df_adj_with_name.set_index(f'{major_var}_s')
                major_occupation_matrix_with_name = major_occupation_matrix.merge(df_name, left_index=True, right_index=True)
                major_occupation_matrix_with_name = major_occupation_matrix_with_name.set_index(f'{major_var}_s')

                df_adj_with_name.to_csv(f'data/processed_data/acs/adjacency_matrix_{major_var}_soc{soc_level}_with_name.csv')
                major_occupation_matrix_with_name.to_csv(f'data/processed_data/acs/major_occupation_matrix_{major_var}_soc{soc_level}_with_name.csv')

