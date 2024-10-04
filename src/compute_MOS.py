'''
Code to compute the MOS.
It assumes the ACS data is processed by ACS_data_prep.py.
'''
import pandas as pd
import numpy as np
from scipy import stats
import src.ECI_package_experimenting as MCI
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


def compute_mci(adj_matrix, major_weight=None, occupation_weight=None, asym=False, iterations=250):
    '''
    Compute the  MCI.
    If asym=False, use a single adjacency matrix.
    If asym=True, use two different matrices for m->o and o->m.
    '''
    #Read data
    major_list = adj_matrix.index
    occ_list = adj_matrix.keys()

    #Check for the single-degree nodes
    print('Occupations linked to only one major:',  adj_matrix.keys()[adj_matrix.sum()==1].tolist())
    print('Majors linked to only one occupation:',adj_matrix.transpose().keys()[adj_matrix.transpose().sum()==1].tolist())

    #Compute MCI
    #NOTE:
    ## MCI.eci_compute MCI either binary or weighted, but does not normalize the weights.
    ## The experiment that standardize the values every iteration needs to be coded separately.
    ## The controlled MCI requires two weight matrices, which seems to be coded as eci_compute_withWeights.
    if asym:
        [MCI_Maj_b, MCI_Occ_b] = MCI.eci_compute_withWeights_std(adj_matrix.values,iterations, major_weight.values, occupation_weight.values) 
    else:
        [MCI_Maj_b, MCI_Occ_b] = MCI.eci_compute(adj_matrix.values,iterations) 


    # Generate rankings.
    col_iter = ["iter_"+str(x) for x in range(iterations+1)]
    MCI_Maj_b_df = pd.DataFrame(MCI_Maj_b,index = major_list, columns = col_iter)
    MCI_Occ_b_df = pd.DataFrame(MCI_Occ_b,index = occ_list, columns = col_iter)

    ## Ranking change over iterations. Only even(odd) iteration for major(occupation)
    MCI_Maj_rank_b = pd.DataFrame(MCI_Maj_b_df,index = major_list).apply(MCI.rank_list).iloc[:,0:iterations:2]
    MCI_Occ_rank_b = pd.DataFrame(MCI_Occ_b_df,index = occ_list).apply(MCI.rank_list).iloc[:,1:iterations:2]

    ## Plot ranking change (MOR_plot)
    # figsize = (5,10)
    # title = ''
    # xlabel='Iterations'
    # fig, ax = plt.subplots(1,figsize=figsize)
    # ax.plot(MCI_Maj_rank_b)
    # ax.set_xlabel(xlabel)
    # ax.set_title(title)

    return MCI_Maj_b_df, MCI_Occ_b_df, MCI_Maj_rank_b, MCI_Occ_rank_b

def plot_ranking_change(MCI_Maj_rank_b):
    # Plot ranking change (MOR_plot)
    figsize = (5,10)
    title = ''
    xlabel='Iterations'
    fig, ax = plt.subplots(1,figsize=figsize)
    ax.plot(MCI_Maj_rank_b)
    ax.set_xlabel(xlabel)
    ax.set_title(title)


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


def compute_mci_hajime(A, At=None, mci_init=None, iter=250, output_other_side=False):
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
        oci = (oci - oci.min())/(oci.max()-oci.min()) + 1e-5
        mci = (mci - mci.min())/(mci.max()-mci.min()) + 1e-5
        result_mci[f'iter_{i+1}'] = mci
        result_oci[f'iter_{i+1}'] = oci
    if output_other_side:
        return result_mci, result_oci
    else:
        return result_mci


if __name__ == '__main__':
    iterations = 25#10#250

    # major_name = pd.read_csv('data/processed_data/acs/degfield_mapping.csv', index_col=0)
    # major_d_name = pd.read_csv('data/processed_data/acs/degfieldd_mapping.csv', index_col=0)

    df_degfield = pd.read_csv('data/processed_data/acs/degfield_mapping.csv', index_col=0)
    df_degfieldd = pd.read_csv('data/processed_data/acs/degfieldd_mapping.csv', index_col=0)

    for soc_level in [2,4]:
        for major_var in ['degfield', 'degfieldd']:
            for binarize_method in ['rca']: #'fixedr'
                if binarize_method == 'rca':
                    binarize_threshold = 1.
                elif binarize_method == 'fixedr':
                    binarize_threshold = 0.05 
                major_name = df_degfield if major_var == 'degfield' else df_degfieldd
                major_name.index.name = 'major'

                print(f'---Processing {major_var} soc{soc_level} {binarize_method}{binarize_threshold:.2f}---')
                
                # Read data
                adj_matrix = pd.read_csv(f'data/processed_data/acs/adj_{major_var}_soc{soc_level}.csv',index_col=0)
                binary_matrix = pd.read_csv(f'data/processed_data/acs/major_occupation_matrix_binary_{binarize_method}{binarize_threshold:.2f}_{major_var}_soc{soc_level}.csv',index_col=0)
                major_weight = pd.read_csv(f'data/processed_data/acs/major_weight_{major_var}_soc{soc_level}.csv',index_col=0)
                occupation_weight = pd.read_csv(f'data/processed_data/acs/occupation_weight_{major_var}_soc{soc_level}.csv',index_col=0)
                major_controlled_weight = pd.read_csv(f'data/processed_data/acs/major_coltrolled_weight_{major_var}_soc{soc_level}.csv',index_col=0)
                occupation_controlled_weight = pd.read_csv(f'data/processed_data/acs/occupation_coltrolled_weight_{major_var}_soc{soc_level}.csv',index_col=0)


                print(f'Original matrix size: {binary_matrix.shape}')
                major_list = binary_matrix.index
                occupation_list = binary_matrix.columns
                # Restrict the major/occupatin to be in controlled weight too.
                major_list = major_list[major_list.isin(major_controlled_weight.index)]
                occupation_list = occupation_list[occupation_list.isin(occupation_controlled_weight.columns)]
                print(f'After restricting to controlled weight: {len(major_list)} majors, {len(occupation_list)} occupations')
                # Remove the zero major/occupation 
                while (binary_matrix.sum(axis=1)==0).sum()>0 or (binary_matrix.sum(axis=0)==0).sum()>0:
                    major_list =  major_list[major_list.isin( binary_matrix.index[binary_matrix.sum(axis=1)>0])]
                    occupation_list = occupation_list[occupation_list.isin(binary_matrix.columns[binary_matrix.sum(axis=0)>0]) ]
                    # Update the matrices
                    binary_matrix = binary_matrix.loc[major_list][occupation_list]
                    print(f'After removing zero major/occupation: {len(major_list)} majors, {len(occupation_list)} occupations')

                binary_matrix = binary_matrix.loc[major_list][occupation_list]
                adj_matrix = adj_matrix.loc[major_list][occupation_list]
                major_weight = major_weight.loc[major_list][occupation_list]
                occupation_weight = occupation_weight.loc[major_list][occupation_list]
                major_controlled_weight = major_controlled_weight.loc[major_list][occupation_list]
                occupation_controlled_weight = occupation_controlled_weight.loc[major_list][occupation_list]


                # MCI_Maj_b_df, MCI_Occ_b_df, MCI_Maj_rank_b, MCI_Occ_rank_b = compute_mci(binary_matrix, iterations=iterations)
                # # MCI_Maj_b_df.to_csv('results/acs/MCI_Maj_b_{binarize_method}_{major_var}_soc{soc_level}.csv')

                # MCI_Maj_w_df, MCI_Occ_w_df, MCI_Maj_rank_w, MCI_Occ_rank_w  = compute_mci(binary_matrix, major_weight=major_weight, occupation_weight=occupation_weight, asym=True, iterations=iterations)
                # # MCI_Maj_w_df.to_csv('results/acs/MCI_Maj_w_{binarize_method}_{major_var}_soc{soc_level}.csv')

                # MCI_Maj_c_df, MCI_Occ_c_df, MCI_Maj_rank_c, MCI_Occ_rank_c  = compute_mci(binary_matrix, major_weight=major_controlled_weight, occupation_weight=occupation_controlled_weight, asym=True, iterations=iterations)
                # # MCI_Maj_c_df.to_csv('results/acs/MCI_Maj_c_{binarize_method}_{major_var}_soc{soc_level}.csv')

                mci_b_H = compute_mci_hajime(binary_matrix, iter=iterations)
                mci_init = binary_matrix.sum(axis=1)
                mci_w_H = compute_mci_hajime(A=major_weight, At=occupation_weight.T, mci_init=mci_init, iter=iterations) 
                mci_c_H = compute_mci_hajime(A=major_controlled_weight, At=occupation_controlled_weight.T, mci_init=mci_init, iter=iterations)
                spread_index = binary_matrix.sum(axis=1)


                mci_b_eig = compute_mci_eig(binary_matrix)


                # Start from oci 
                _, mci2_b_H = compute_mci_hajime(binary_matrix.T, iter=iterations, output_other_side=True)
                oci_init = binary_matrix.sum(axis=0)
                _, mci2_w_H = compute_mci_hajime(A=occupation_weight.T, At=major_weight, mci_init=oci_init, iter=iterations, output_other_side=True)
                _, mci2_c_H = compute_mci_hajime(A=occupation_controlled_weight.T, At=major_controlled_weight, mci_init=oci_init, iter=iterations, output_other_side=True)


                # assert np.corrcoef(MCI_Maj_b_df.iloc[:,-1], mci_b_H.iloc[:,-1])[0,1] > 0.99
                # assert np.corrcoef(MCI_Maj_w_df.iloc[:,-1], mci_w_H.iloc[:,-1])[0,1] > 0.99
                # assert np.corrcoef(MCI_Maj_c_df.iloc[:,-1], mci_c_H.iloc[:,-1])[0,1] > 0.99

                # mci_b_H.to_csv(f'results/acs/MCI_Maj_b_{binarize_method}_{major_var}_soc{soc_level}.csv')
                # mci_w_H.to_csv(f'results/acs/MCI_Maj_w_{binarize_method}_{major_var}_soc{soc_level}.csv')
                # mci_c_H.to_csv(f'results/acs/MCI_Maj_c_{binarize_method}_{major_var}_soc{soc_level}.csv')
                # mci2_b_H.to_csv(f'results/acs/MCI_Maj2_b_{binarize_method}_{major_var}_soc{soc_level}.csv')
                # mci2_w_H.to_csv(f'results/acs/MCI_Maj2_w_{binarize_method}_{major_var}_soc{soc_level}.csv')
                # mci2_c_H.to_csv(f'results/acs/MCI_Maj2_c_{binarize_method}_{major_var}_soc{soc_level}.csv')


                frac = major_weight.transpose()
                # Share of top 3 occupations
                Altonji_top3_share = frac.apply(lambda x: x.nlargest(3).sum())
                Altonji_top3_share_occ = occupation_weight.apply(lambda x: x.nlargest(3).sum())
                # HHI
                Blom_hhi = (frac**2).sum()
                Blom_hhi_occ = (occupation_weight**2).sum()

                # other_index =  pd.read_csv(f'results/acs/other_index_{major_var}_soc{soc_level}.csv', index_col=0)
                # mci_init = other_index['Blom_hhi'].loc[binary_matrix.index]
                mci_init_hhi = Blom_hhi*100
                mci_b_from_hhi = compute_mci_hajime(binary_matrix, mci_init=mci_init_hhi,iter=iterations)
                mci_w_from_hhi = compute_mci_hajime(A=major_weight, At=occupation_weight.T, mci_init=mci_init_hhi, iter=iterations)
                mci_c_from_hhi = compute_mci_hajime(A=major_controlled_weight, At=occupation_controlled_weight.T, mci_init=mci_init_hhi, iter=iterations)

                mci_init_top3 = Altonji_top3_share*100
                mci_b_from_top3 = compute_mci_hajime(binary_matrix, mci_init=mci_init_top3,iter=iterations)
                mci_w_from_top3 = compute_mci_hajime(A=major_weight, At=occupation_weight.T, mci_init=mci_init_top3, iter=iterations)
                mci_c_from_top3 = compute_mci_hajime(A=major_controlled_weight, At=occupation_controlled_weight.T, mci_init=mci_init_top3, iter=iterations)

                df_gini = pd.read_csv(f'results/acs/gini_{"degfield"}.csv',index_col=0)
                
                if major_var == 'degfield':
                    mci_init_gini = df_gini.loc[binary_matrix.index]*100
                    mci_b_from_gini = compute_mci_hajime(binary_matrix, mci_init=mci_init_gini,iter=iterations)
                    mci_w_from_gini = compute_mci_hajime(A=major_weight, At=occupation_weight.T, mci_init=mci_init_gini, iter=iterations)
                    mci_c_from_gini = compute_mci_hajime(A=major_controlled_weight, At=occupation_controlled_weight.T, mci_init=mci_init_gini, iter=iterations)
                elif major_var == 'degfieldd':
                    coarse_major = binary_matrix.index.astype(str).str[:2].astype(int)
                    # Turn coarse_major into a series
                    coarse_major = pd.Series(coarse_major, index=binary_matrix.index)
                    coarse_major[binary_matrix.index==5098] = 40
                    # The union of df_gini.index and coarse_major
                    mci_init_gini = df_gini.loc[coarse_major]*100
                    mci_b_from_gini = compute_mci_hajime(binary_matrix, mci_init=mci_init_gini,iter=iterations)
                    mci_w_from_gini = compute_mci_hajime(A=major_weight, At=occupation_weight.T, mci_init=mci_init_gini, iter=iterations)
                    mci_c_from_gini = compute_mci_hajime(A=major_controlled_weight, At=occupation_controlled_weight.T, mci_init=mci_init_gini, iter=iterations)



                ## Plot the change of ranking
                iter_ranking = pd.DataFrame(index=mci_b_from_hhi.index)
                iter_ranking.index.name = 'major'
                for iter in range(mci_b_from_hhi.shape[1]):
                    iter_ranking[f'Ranking_iter_{iter}'] = mci_b_from_hhi[f'iter_{iter}'].rank(ascending=False,method='first')
                iter_ranking = iter_ranking.T

                fig, ax = plt.subplots(figsize=(12, 6))
                ## Define the color of plots for each degfield_s. Need 35 distinct colors.
                k = 0
                change_cut = (iter_ranking.max(axis=0)-iter_ranking.min(axis=0)).sort_values().iloc[-15]
                for m in iter_ranking.columns:
                    original_line = iter_ranking[m]
                    # smoothed_line = gaussian_filter1d(original_line, sigma=2)
                    ## Make the line very transparent if the change is not too big
                    if np.abs(original_line.max() - original_line.min()) >=change_cut:
                        ax.plot(iter_ranking.index,original_line,linewidth=2.,alpha=1.,label=major_name.loc[m].values[0], color=sns.color_palette("tab20", 35)[k]   )
                        k += 1
                    else:
                        ax.plot(iter_ranking.index,original_line,linewidth=0.5,alpha=0.5, label='_nolegend_',color='grey' )
                ax.invert_yaxis()
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Ranking')
                ax.set_xticklabels([f'{2*i}' for i in range(iter_ranking.shape[0])])
                ## Add the legend outside the plot, but the legend needs to be inside the plot to be able to use the label
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ## Reduce the font size of the legend
                # plt.setp(ax.get_legend().get_texts(), fontsize='7')
                ## Fit the legend to figure size
                fig.tight_layout()
                # plt.show()
                # ax.legend(gini_iter_ranking.columns)
                fig.savefig(f'results/paper/MCI_change_over_iteration_{major_var}_soc{soc_level}.png',dpi=300)
                # plt.close(fig)



                '''
                iterations_tes = 100
                mci_init_tes = (Altonji_top3_share>=0).astype(int)
                mci_b_from_tes = compute_mci_hajime(binary_matrix, mci_init=mci_init_tes,iter=iterations_tes)
                mci_tes = (mci_b_from_tes.iloc[:,-1] - mci_b_from_tes.iloc[:,-1].mean() )/ mci_b_from_tes.iloc[:,-1].std()

                mci_init_tes2 = mci_init_tes.copy()
                mci_init_tes2.iloc[0] = 0.
                mci_b_from_tes2 = compute_mci_hajime(binary_matrix, mci_init=mci_init_tes2,iter=iterations_tes)
                mci_tes2 = (mci_b_from_tes2.iloc[:,-1] - mci_b_from_tes2.iloc[:,-1].mean() )/ mci_b_from_tes2.iloc[:,-1].std()

                mci_init_tes3 = mci_init_tes.copy()
                mci_init_tes3.iloc[0:30] = 0.
                mci_b_from_tes3 = compute_mci_hajime(binary_matrix, mci_init=mci_init_tes3,iter=iterations_tes)
                mci_tes3 = (mci_b_from_tes3.iloc[:,-1] - mci_b_from_tes3.iloc[:,-1].mean() )/ mci_b_from_tes3.iloc[:,-1].std()

                mci_init_tes4 = mci_init_tes.copy()
                mci_init_tes4.iloc[1:] = 0.
                mci_b_from_tes4 = compute_mci_hajime(binary_matrix, mci_init=mci_init_tes4,iter=iterations_tes)
                mci_tes4 = (mci_b_from_tes4.iloc[:,-1] - mci_b_from_tes4.iloc[:,-1].mean() )/ mci_b_from_tes4.iloc[:,-1].std()

                mci_init_tes5 = binary_matrix.sum(axis=1)
                mci_b_from_tes5 = compute_mci_hajime(binary_matrix, mci_init=mci_init_tes5,iter=iterations_tes)
                mci_tes5 = (mci_b_from_tes5.iloc[:,-1] - mci_b_from_tes5.iloc[:,-1].mean() )/ mci_b_from_tes5.iloc[:,-1].std()

                mci_init_tes6 = (Altonji_top3_share>=0).astype(int)*.001
                mci_b_from_tes6 = compute_mci_hajime(binary_matrix, mci_init=mci_init_tes,iter=iterations_tes)
                mci_tes6 = (mci_b_from_tes6.iloc[:,-1] - mci_b_from_tes6.iloc[:,-1].mean() )/ mci_b_from_tes6.iloc[:,-1].std()

                
                mci_init_tes7 = pd.Series(np.random.randint(1,16,len(Altonji_top3_share)), index = Altonji_top3_share.index)
                mci_b_from_tes7 = compute_mci_hajime(binary_matrix, mci_init=mci_init_tes7,iter=iterations_tes)
                mci_tes7 = (mci_b_from_tes7.iloc[:,-1] - mci_b_from_tes7.iloc[:,-1].mean() )/ mci_b_from_tes7.iloc[:,-1].std()

                mci_init_tes8 = pd.Series(np.random.randint(1,16,len(Altonji_top3_share)), index = Altonji_top3_share.index)
                mci_b_from_tes8 = compute_mci_hajime(binary_matrix, mci_init=mci_init_tes8,iter=iterations_tes)
                mci_tes8 = (mci_b_from_tes8.iloc[:,-1] - mci_b_from_tes8.iloc[:,-1].mean() )/ mci_b_from_tes8.iloc[:,-1].std()

                mci_init_tes9 = pd.Series(np.random.uniform(0,1,len(Altonji_top3_share)), index = Altonji_top3_share.index)
                mci_b_from_tes9 = compute_mci_hajime(binary_matrix, mci_init=mci_init_tes9,iter=iterations_tes)
                mci_tes9 = (mci_b_from_tes9.iloc[:,-1] - mci_b_from_tes9.iloc[:,-1].mean() )/ mci_b_from_tes9.iloc[:,-1].std()

                mci_init_tes10 = pd.Series(np.random.uniform(0,1,len(Altonji_top3_share)), index = Altonji_top3_share.index)
                mci_b_from_tes10 = compute_mci_hajime(binary_matrix, mci_init=mci_init_tes10,iter=iterations_tes)
                mci_tes10 = (mci_b_from_tes10.iloc[:,-1] - mci_b_from_tes10.iloc[:,-1].mean() )/ mci_b_from_tes10.iloc[:,-1].std()
                

                tes_init = pd.concat([mci_init_tes7, mci_init_tes8, mci_init_tes9, mci_init_tes10], axis=1)
                tes=pd.concat([mci_tes7, mci_tes8, mci_tes9, mci_tes10], axis=1)


                np.corrcoef([mci_b_from_tes.iloc[:,-1], mci_b_from_tes2.iloc[:,-1]])
                np.corrcoef([mci_b_from_tes.iloc[:,-1], mci_b_from_tes6.iloc[:,-1]])
                (mci_tes==0).mean()
                '''


                # other_index_occ =  pd.read_csv(f'results/acs/other_index_occ_{major_var}_soc{soc_level}.csv', index_col=0)
                # oci_init = other_index_occ['Blom_hhi'].loc[binary_matrix.columns.astype(int)]
                oci_init_hhi = Blom_hhi_occ
                _, mci2_b_from_hhi = compute_mci_hajime(binary_matrix.T, mci_init=oci_init_hhi,iter=iterations, output_other_side=True)
                _, mci2_w_from_hhi = compute_mci_hajime(A=occupation_weight.T, At=major_weight, mci_init=oci_init_hhi, iter=iterations, output_other_side=True)
                _, mci2_c_from_hhi = compute_mci_hajime(A=occupation_controlled_weight.T, At=major_controlled_weight, mci_init=oci_init_hhi, iter=iterations, output_other_side=True)

                oci_init_top3 = Altonji_top3_share_occ
                _, mci2_b_from_top3 = compute_mci_hajime(binary_matrix.T, mci_init=oci_init_top3,iter=iterations, output_other_side=True)
                _, mci2_w_from_top3 = compute_mci_hajime(A=occupation_weight.T, At=major_weight, mci_init=oci_init_top3, iter=iterations, output_other_side=True)
                _, mci2_c_from_top3 = compute_mci_hajime(A=occupation_controlled_weight.T, At=major_controlled_weight, mci_init=oci_init_top3, iter=iterations, output_other_side=True)

                df_mci = pd.DataFrame({'Altonji_top3_share':Altonji_top3_share,'Blom_hhi':Blom_hhi,'mci_b':mci_b_H.iloc[:,-1],\
                                        'mci_w':mci_w_H.iloc[:,-1], 'mci_c':mci_c_H.iloc[:,-1], 'mci2_b':mci2_b_H.iloc[:,-1], 'mci2_w':mci2_w_H.iloc[:,-1], 'mci2_c':mci2_c_H.iloc[:,-1], 'mci_b_from_hhi':mci_b_from_hhi.iloc[:,-1],\
                                        'mci_w_from_hhi':mci_w_from_hhi.iloc[:,-1], 'mci_c_from_hhi':mci_c_from_hhi.iloc[:,-1], 'mci2_b_from_hhi':mci2_b_from_hhi.iloc[:,-1], 'mci2_w_from_hhi':mci2_w_from_hhi.iloc[:,-1], 'mci2_c_from_hhi':mci2_c_from_hhi.iloc[:,-1],\
                                             'mci_b_from_top3':mci_b_from_top3.iloc[:,-1], 'mci_w_from_top3':mci_w_from_top3.iloc[:,-1], 'mci_c_from_top3':mci_c_from_top3.iloc[:,-1], 'mci2_b_from_top3':mci2_b_from_top3.iloc[:,-1], 'mci2_w_from_top3':mci2_w_from_top3.iloc[:,-1], 'mci2_c_from_top3':mci2_c_from_top3.iloc[:,-1],\
                                                'mci_spread_index':spread_index})

                df_mci['mci_b_gini'] = mci_b_from_gini.iloc[:,-1]
                df_mci['mci_w_gini'] = mci_w_from_gini.iloc[:,-1]
                df_mci['mci_c_gini'] = mci_c_from_gini.iloc[:,-1]


                # Merge df_mci with major_name by major and degfield
                df_mci = df_mci.merge(major_name, left_index=True, right_index=True, how='left')
                df_mci.to_csv(f'results/acs/MCI_Maj_{binarize_method}{binarize_threshold:.2f}_{major_var}_soc{soc_level}.csv')



                for mci, mci_name in zip([mci_b_H, mci_w_H, mci_c_H, mci2_b_H, mci2_w_H, mci2_c_H, mci_b_from_hhi, mci_w_from_hhi, mci_c_from_hhi, mci2_b_from_hhi, mci2_w_from_hhi, mci2_c_from_hhi, mci_b_from_top3, mci_w_from_top3, mci_c_from_top3, mci2_b_from_top3, mci2_w_from_top3, mci2_c_from_top3], ['mci_b', 'mci_w', 'mci_c', 'mci2_b', 'mci2_w', 'mci2_c', 'mci_b_from_hhi', 'mci_w_from_hhi', 'mci_c_from_hhi', 'mci2_b_from_hhi', 'mci2_w_from_hhi', 'mci2_c_from_hhi', 'mci_b_from_top3', 'mci_w_from_top3', 'mci_c_from_top3', 'mci2_b_from_top3', 'mci2_w_from_top3', 'mci2_c_from_top3']):
                    ## Add major name to mci
                    mci = mci.merge(major_name, left_index=True, right_index=True, how='left')
                    mci.to_csv(f'results/acs/{mci_name}_over_iteration_{binarize_method}{binarize_threshold:.2f}_{major_var}_soc{soc_level}.csv')
                    
                for mci, mci_name in zip([mci_b_from_gini, mci_w_from_gini, mci_c_from_gini], ['mci_b_from_gini', 'mci_w_from_gini', 'mci_c_from_gini']):
                    mci = mci.merge(major_name, left_index=True, right_index=True, how='left')
                    mci.to_csv(f'results/acs/{mci_name}_over_iteration_{binarize_method}{binarize_threshold:.2f}_{major_var}_soc{soc_level}.csv')







    # data_source = 'nscg'
    # year = 2003
    # data_dir = 'data/processed_data/' + data_source + '/'
    # result_dir = 'data/processed_data/' + data_source + '/'
    # binary_thresh = 1
    # iterations = 10

    # compute_mci(data_source, year, data_dir, result_dir, binary_thresh, iterations)
