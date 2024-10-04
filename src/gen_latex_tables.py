'''
Generate Latex tables from the regression results.
'''

import pandas as pd
import numpy as np

def write_latex_table_summary(table_df,result_folder,table_name,caption='', drop_d=False,write_index=True,decimal=4):
    n_cols = len(table_df.columns)
    # Add one for the index column
    column_format = 'l' + 'c' * n_cols

    latex_df = table_df.copy()
    latex_df = latex_df.fillna('')
    latex_df = latex_df.replace("<NA>", "")
    latex_df = latex_df.replace("nan", "")

    if drop_d:
        latex_df.index = latex_df.index.str.replace('_d', '', regex=True)
        latex_df.columns = latex_df.columns.str.replace('_d', '', regex=True)


    latex_table = latex_df.to_latex(column_format=column_format,
                                    caption=caption,
                                    bold_rows=False, 
                                    # multicolumn_format='c', 
                                    multicolumn=False, 
                                    escape=True,
                                    index=write_index,
                                    index_names = False,
                                    float_format=f"{{:0.{decimal}f}}".format
                                    )

    latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
    latex_table = latex_table.replace('\\midrule', '\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline\\hline')
    latex_table = latex_table.replace('begin{table}', 'begin{table}[ht]')

    latex_table = latex_table.replace('female', 'Female')
    latex_table = latex_table.replace('hispanic', 'Hispanic')
    latex_table = latex_table.replace('white', 'White')
    latex_table = latex_table.replace('experience', 'Experience')
    latex_table = latex_table.replace('experience_sq', 'Experience\_sq')
    


    # Save to a .tex file
    with open(result_folder+table_name, 'w') as tf:
        # tf.write('\\begin{table}[ht]\n')
        # tf.write('\\centering\n')
        tf.write(latex_table)
        # tf.write('\\end{table}\n')



def write_latex_table(table_df,result_folder,table_name,caption='', no_control=False,keep_col=False,drop_d=True,decimal=4):
    n_cols = len(table_df.columns)
    # Add one for the index column
    column_format = 'l' + 'c' * n_cols

    latex_df = table_df.copy()

    latex_df = latex_df.reset_index()
    latex_df['index'] = latex_df['index'].fillna('')
    latex_df = latex_df.set_index('index')
    latex_df.index.name = ''

    latex_df = latex_df.fillna('')
    latex_df = latex_df.replace("<NA>", "")
    latex_df = latex_df.replace("nan", "")

    latex_df.index = latex_df.index.str.replace("sex", "female")

    if drop_d:
        latex_df.index = latex_df.index.str.replace('_d', '', regex=True)
        latex_df.columns = latex_df.columns.str.replace('_d', '', regex=True)

    # latex_df.index = latex_df.index.str.replace('_', '\\_', regex=True)

    if not keep_col:
        # Rename columns to be "(1)", "(2)", etc.
        latex_df.columns = ['(' + str(i+1) + ')' for i in range(len(latex_df.columns))]
        latex_df = latex_df.fillna('')

    if no_control:
        start_index = latex_df.index.tolist().index('const')
        end_index = latex_df.index.tolist().index('R-squared Adj.')
        latex_df = pd.concat([latex_df.iloc[:start_index], latex_df.iloc[end_index:]])

    latex_table = latex_df.to_latex(column_format=column_format,
                                    caption=caption,
                                    bold_rows=False, 
                                    # multicolumn_format='c', 
                                    multicolumn=False, 
                                    escape=True,
                                    index_names = False,
                                    float_format=f"{{:0.{decimal}f}}".format
    )

    latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
    latex_table = latex_table.replace('\\midrule', '\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline\\hline')
    latex_table = latex_table.replace('begin{table}', 'begin{table}[htbp]')

    latex_table = latex_table.replace('female', 'Female')
    latex_table = latex_table.replace('hispanic', 'Hispanic')
    latex_table = latex_table.replace('white', 'White')
    latex_table = latex_table.replace('experience', 'Experience')
    latex_table = latex_table.replace('experience_sq', 'Experience\_sq')



    # Save to a .tex file
    with open(result_folder+table_name, 'w') as tf:
        # tf.write('\\begin{table}[ht]\n')
        # tf.write('\\centering\n')
        tf.write(latex_table)
        # tf.write('\\end{table}\n')

result_folder_paper = 'results/paper/'
result_folder_slides = 'results/slides/'

# Correlation between indices
corr_indices = pd.read_csv('results/acs/corr_indices.csv',index_col=0)
## Corrleation between HHI, TOP3, GINI, MCI
corr1 = corr_indices.loc[['HHI_d','TOP3_d','GINI','MCI_HHI_d'],['HHI_d','TOP3_d','GINI','MCI_HHI_d']]
## Turn corr1 into lower triangle by setting upper triangle to be nan
corr1 = corr1.where(np.tril(np.ones(corr1.shape)).astype(np.bool))
write_latex_table_summary(corr1,result_folder_paper,table_name=f'table_corr_index.tex',caption='Correlation of MCI with other indices.',drop_d=True)
## Correlation between MCI and other indices
corr2 = corr_indices.loc[['MCI_HHI_d','MCI_TOP3_d','MCI_GINI_d','MCI_d'],['MCI_HHI_d','MCI_TOP3_d','MCI_GINI_d','MCI_d']]
corr2.rename(columns={'MCI_d':'MCI_spread'},index={'MCI_d':'MCI_spread'},inplace=True)
corr2 = corr2.where(np.tril(np.ones(corr2.shape)).astype(np.bool))
write_latex_table_summary(corr2,result_folder_paper,table_name=f'table_corr_mci_across_init.tex',caption='Correlation of MCI with different initialization.',drop_d=True)


# Correlation between MCI in different methods
corr_mci_coarse = pd.read_csv('results/acs/corr_mci_coarse.csv',index_col=0)
corr_mci_coarse = corr_mci_coarse.where(np.tril(np.ones(corr_mci_coarse.shape)).astype(np.bool))
write_latex_table_summary(corr_mci_coarse,result_folder_paper,table_name=f'table_corr_mci_coarse.tex',caption='Correlation of MCI from different aggregation level of major/occupation.',drop_d=False)



# Correlation against NSSE features
nsse_corr = pd.read_csv('results/acs/nsse_corr.csv',index_col=0)
write_latex_table_summary(nsse_corr,result_folder_paper,table_name=f'table_corr_nsse.tex')



# Ranking before and after iterations
ranking_iter0 = pd.read_csv('results/acs/mci_b_from_hhi_iter0_ranking.csv',index_col=0).rename(columns={'Ranking':'Before Iteration'})
ranking_iter25 = pd.read_csv('results/acs/mci_b_from_hhi_iter25_ranking.csv',index_col=0).rename(columns={'Ranking':'After Iteration'})
ranking = ranking_iter0.merge(ranking_iter25[['After Iteration',  'MCI_HHI_iter_25']],how='inner',left_index=True,right_index=True)
ranking = ranking[['degfieldd_s','Before Iteration','After Iteration',  'HHI','MCI_HHI_iter_25']] #

ranking[['degfieldd_s','Before Iteration', 'HHI']].to_csv('results/paper/ranking_hhi.csv')
ranking[['degfieldd_s','After Iteration', 'MCI_HHI_iter_25']].rename(columns={'MCI_HHI_iter_25':'MCI'}).to_csv('results/paper/ranking_mci.csv')


## Ranking is an integer, so we need to convert it to string
ranking.reset_index(inplace=True)
ranking['Before Iteration'] = ranking['Before Iteration'].astype(int)
ranking['After Iteration'] = ranking['After Iteration'].astype(int)
ranking.rename(columns={'degfieldd_s':'Major Name','major':'Major Code'},inplace=True)
ranking =ranking[['Major Code','Major Name','HHI','Before Iteration','After Iteration']]
# write_latex_table(ranking,result_folder_paper,table_name=f'table_ranking_change.tex',caption='Change of ranking before and after iterations.',keep_col=True,drop_d=True)
write_latex_table_summary(ranking,result_folder_paper,table_name=f'table_ranking_change.tex',caption='Change of ranking before and after iterations.', drop_d=False,write_index=False)



# Summary stat csv of df_mci. same for df_wage and df_emp are generated in main_individual_regression.py
for soc_level in [2,4]:
    for major_var in ['degfield','degfieldd']:
        df_mci = pd.read_csv(f'data/processed_data/acs/df_mci_{major_var}_soc{soc_level}.csv', index_col=0)
        df_mci_summary_stat = df_mci[['sex','white','hispanic','experience']].describe().T
        df_mci_summary_stat.drop(columns=['25%','50%','75%'],inplace=True)
        df_mci_summary_stat.rename(columns={'mean':'Mean','std':'Std', 'min':'Min','max':'Max','count':'N'},inplace=True)
        df_mci_summary_stat = df_mci_summary_stat[['N','Mean','Std','Min','Max' ]]
        df_mci_summary_stat.to_csv(f'results/acs/summary_stat_df_mci_{major_var}_soc{soc_level}.csv')

# Summary stat tables.
mci_summary_stat = pd.read_csv(f'results/acs/summary_stat_df_mci_degfieldd_soc4.csv',index_col=0)
mci_summary_stat=(mci_summary_stat*1000).astype(int)/1000
mci_summary_stat = pd.concat([pd.DataFrame(index=['Panel A: MCI Construction']),mci_summary_stat])
mci_summary_stat['N'] = mci_summary_stat['N'].astype('Int64').apply(lambda x: '{:,}'.format(x) if pd.notna(x) else x).astype(str)

wage_summary_stat = pd.read_csv('results/acs/summary_stat_df_wage.csv',index_col=0)
wage_summary_stat=(wage_summary_stat*1000).astype(int)/1000
wage_summary_stat = pd.concat([pd.DataFrame(index=['Panel B: Wage Regression']),wage_summary_stat])
wage_summary_stat.loc['incwage_cpiu_2010'] = wage_summary_stat.loc['incwage_cpiu_2010'].astype(int).apply(lambda x: '{:,}'.format(x) if pd.notna(x) else x).astype(str)
wage_summary_stat.loc[['sex','white','hispanic','experience'],'N'] = wage_summary_stat.loc[['sex','white','hispanic','experience'],'N'].astype('Int64').apply(lambda x: '{:,}'.format(x) if pd.notna(x) else x).astype(str)

emp_summary_stat = pd.read_csv('results/acs/summary_stat_df_emp.csv',index_col=0)
emp_summary_stat=(emp_summary_stat*1000).astype(int)/1000
emp_summary_stat = pd.concat([pd.DataFrame(index=['Panel C: Employment Regression']),emp_summary_stat])
emp_summary_stat['N'] = emp_summary_stat['N'].astype('Int64').apply(lambda x: '{:,}'.format(x) if pd.notna(x) else x).astype(str)

summary_stat = pd.concat([mci_summary_stat,wage_summary_stat,emp_summary_stat])

summary_stat = summary_stat[['Mean','Std','Min','Max','N']]

write_latex_table_summary(summary_stat,result_folder_paper,table_name=f'summary_stat_df.tex',caption='Summary statistics of the data')





# target_var = 'wage'
# target_var_name = 'Wage'
base_index ='hhi'
base_index_name = 'HHI'
for target_var, target_var_name in zip(['wage', 'emp'], ['Wage','Employment']):
    #1. baseline regression with other index
    table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_{base_index}.csv',index_col=0)
    write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}.tex',caption=f'{target_var_name} Regression with {base_index_name}-based MCI')
    # write_latex_table(table_df,result_folder_slides,table_name=f'presentation_{target_var}_ind_d_{base_index}.tex',caption=f'{target_var_name} Regression with {base_index_name}-based MCI',no_control=True)

    #2 regression over iterations
    table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_{base_index}_iter.csv',index_col=0)
    ## For all the index "MCI_HHI_iter_{i}", double the number of iterations to be MCI_HHI_iter_{2*i}
    table_df.rename(index={f'MCI_HHI_iter_{i}':f'MCI_HHI_iter_{2*i}' for i in[1,2,5,10,25]},inplace=True)
    write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_iter.tex',caption=f'{target_var_name} Regression with {base_index_name}-based MCI  over iterations',no_control=True)
    # write_latex_table(table_df,result_folder_slides,table_name=f'presentation_{target_var}_ind_d_{base_index}_iter.tex',caption=f'{target_var_name} Regression with {base_index_name}-based MCI over iterations',no_control=True)

    #3 regression with major categories
    table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_{base_index}_cat.csv',index_col=0)
    # Remove "Category_" from the index if any.
    table_df.index = table_df.index.str.replace('Category_','',regex=True)
    write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_cat.tex',caption=f'{target_var_name} Regression with MCI and major categories')
    # write_latex_table(table_df,result_folder_slides,table_name=f'presentation_{target_var}_ind_d_{base_index}_cat.tex',caption=f'{target_var_name} Regression with MCI and major categories',no_control=True)

    #4 regression with MCI based on coarse categories
    table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_{base_index}_coarse.csv',index_col=0)
    table_df = table_df.rename(index={'MCI_HHI':'Coarse Major-SOC4','MCI_HHI_d':'Detailed Major-SOC4','MCI_HHI_SOC2':'Coarse Major-SOC2','MCI_HHI_d_SOC2':'Detailed Major-SOC2'})
    write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_coarse.tex',caption=f'{target_var_name} Regression with MCI on coarse categories',no_control=True,drop_d=False)

    #5 regression with NSSE control
    table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_{base_index}_nsse_control.csv',index_col=0)
    write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_nsse.tex',caption=f'{target_var_name} Regression with major characteristics from NSSE.')
    # write_latex_table(table_df,result_folder_slides,table_name=f'presentation_{target_var}_ind_d_{base_index}_nsse.tex',caption=f'{target_var_name} Regression with Coarse HHI-based MCI',no_control=True)

    #6 regression with occupation category control
    if target_var=='wage':
        table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_{base_index}_occ_dummies.csv',index_col=0)
        write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_occ.tex',caption=f'{target_var_name} Regression with occupation category dummies.',no_control=True)
        # write_latex_table(table_df,result_folder_slides,table_name=f'presentation_{target_var}_ind_d_{base_index}_occ.tex',caption=f'{target_var_name} Regression with occupation category controls',no_control=True)

    #7 regression with MCI based on under-30.
    table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_{base_index}_under30.csv',index_col=0)
    write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_under30.tex',caption=f'{target_var_name} Regression with MCI based on under-30 population',no_control=True)

    #8 regression with female-only
    table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_{base_index}_female.csv',index_col=0)
    write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_female.tex',caption=f'{target_var_name} Regression with MCI based on female only.',no_control=True)

    #9 regression with interaction terms
    table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_{base_index}_interaction.csv',index_col=0)
    table_df = table_df.loc[:,~table_df.loc['MCI_HHI_d'].isnull()]
    write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_interaction.tex',caption=f'{target_var_name} Regression with interaction terms.',no_control=False)

    #10 regression with NLSY data
    table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_hhi_nlsy.csv',index_col=0)
    write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_nlsy.tex',caption=f'{target_var_name} Regression with NLSY data.',no_control=False)
    table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_hhi_nlsy_detail.csv',index_col=0)
    write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_nlsy_detail.tex',caption=f'{target_var_name} Regression with NLSY data.',no_control=False)

    #11 regression with occupation dummies
    if target_var=='wage':
        table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_hhi_occ_dummies.csv',index_col=0)
        write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_occ_dummies.tex',caption=f'{target_var_name} Regression with occupation dummies.',no_control=False)

        table_df = pd.read_csv(f'results/acs/summary_{target_var}_ind_d_hhi_cat_and_occ.csv',index_col=0)
        write_latex_table(table_df,result_folder_paper,table_name=f'table_{target_var}_ind_d_{base_index}_cat_and_occ.tex',caption=f'{target_var_name} Regression with occupation dummies and major categories.',no_control=False)



# future wage prediction
table_df = pd.read_csv(f'results/acs/summary_wage_pred.csv',index_col=0)
write_latex_table(table_df,result_folder_paper,table_name=f'table_wage_pred.tex',caption=f'Future wage prediction from MCI.',no_control=False,keep_col=True)

table_df = pd.read_csv(f'results/acs/summary_wage_pred_hhi.csv',index_col=0)
write_latex_table(table_df,result_folder_paper,table_name=f'table_wage_pred_hhi.tex',caption=f'Future wage prediction from HHI.',no_control=False,keep_col=True)


# Ranking in first and second half of the perids
table_df = pd.read_csv('results/acs/mci_ranking_2009-13_vs_15-19.csv',index_col=0)
table_df.reset_index(inplace=True)
table_df['MCI_HHI_2009'] = table_df['MCI_HHI_2009'].astype(int)
table_df['MCI_HHI_2015'] = table_df['MCI_HHI_2015'].astype(int)
table_df.rename(columns={'MCI_HHI_2009':'2009-13','MCI_HHI_2015':'2015-2019','degfieldd_s':'Major Name','index':'Major Code'},inplace=True)
table_df = table_df[['Major Code','Major Name','2009-13','2015-2019']]
write_latex_table_summary(table_df,result_folder_paper,table_name=f'table_mci_ranking.tex',caption=f'MCI ranking in 2009-13 vs 2015-19.',write_index=False)

#binary_matrix = pd.read_csv(f'data/processed_data/acs/major_occupation_matrix_binary_{"rca"}{1.:.2f}_{"degfieldd"}_soc{4}.csv',index_col=0)
