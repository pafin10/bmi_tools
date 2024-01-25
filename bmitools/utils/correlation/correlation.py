import numpy as np
import scipy
import scipy.stats
from scipy import stats
import os
import parmap
from tqdm import tqdm

#
def compute_correlations_parallel(data_dir,
                                  rasters,
                                  rasters_DFF,
                                  n_cores,
                                  #method='all',
                                  binning_window=30,
                                  subsample=5,
                                  scale_by_DFF=False,
                                  corr_parallel_flag=True,
                                  zscore=False,
                                  n_tests_zscore=1000,
                                  recompute_correlation=False,
                                  min_number_bursts=0):

    """
    This function computes pairwise Pearson correlations between different rasters in a parallelized manner.

    Parameters:
    data_dir (str): The directory where the data is stored.
    rasters (np.array): The rasters to be processed.
    rasters_DFF (np.array): The rasters to be processed after applying DeltaF/F.
    n_cores (int): The number of cores to be used for parallel processing.
    binning_window (int, optional): The size of the binning window. Default is 30.
    subsample (int, optional): The subsampling rate. Default is 5.
    scale_by_DFF (bool, optional): A flag indicating whether to scale by DeltaF/F. Default is False.
    corr_parallel_flag (bool, optional): A flag indicating whether to run the correlation computation in parallel. Default is True.
    zscore (bool, optional): A flag indicating whether to compute the z-score. Default is False.
    n_tests_zscore (int, optional): The number of tests to be performed for z-score computation. Default is 1000.
    recompute_correlation (bool, optional): A flag indicating whether to recompute the correlation. Default is False.
    min_number_bursts (int, optional): The minimum number of bursts required. Default is 0.

    Returns:
    None. The function saves the computed correlations to a .npz file in 'data_dir'.
    
    Note: If a file with the same name already exists in 'data_dir' and 'recompute_correlation' is False,
          the function will return without doing anything.
    """
    # make a small class to hold all the input variables
    class C:
        pass

    c1 = C()
    c1.n_cores = n_cores
    c1.n_tests = n_tests_zscore
    #c1.correlation_method = method
    c1.binning_window = binning_window
    c1.subsample = subsample
    c1.scale_by_DFF = scale_by_DFF
    c1.corr_parallel_flag = corr_parallel_flag
    c1.zscore = zscore
    c1.rasters = rasters
    c1.rasters_DFF = rasters_DFF
    c1.recompute_correlation = recompute_correlation
    c1.min_number_bursts = min_number_bursts
    
    #
    print ("... computing pairwise pearson correlation ...")
    print (" RASTERS IN: ", rasters.shape)
    print (" BINNING WINDOW: ", binning_window)

    #
    c1.data_dir = data_dir

    # convert object into a dictionary
    c1 = c1.__dict__

    #############################################################
    #############################################################
    #############################################################
    # run parallel
    ids = np.arange(rasters.shape[0])
    if corr_parallel_flag:
        parmap.map(correlations_parallel2,
                    ids,
                    c1,
                    pm_processes=n_cores,
                    pm_pbar = True
                    )
    else:
        for k in tqdm(ids, desc='computing intercell correlation'):
            correlations_parallel2(k,
                                   c1)

# this computes the correlation for a single cell against all others and then saves it to disk
def correlations_parallel2(id, 
                           c1):
    """
    This function computes the correlation between different rasters in a parallelized manner.

    Parameters:
    id (int): The ID of the raster to be processed.
    c1 (dict): A dictionary containing various parameters and data needed for the computation.

    The dictionary 'c1' has the following keys:
    - data_dir (str): The directory where the data is stored.
    - rasters (np.array): The rasters to be processed.
    - rasters_DFF (np.array): The rasters to be processed after applying DeltaF/F.
    - binning_window (int): The size of the binning window.
    - subsample (int): The subsampling rate.
    - scale_by_DFF (bool): A flag indicating whether to scale by DeltaF/F.
    - zscore (bool): A flag indicating whether to compute the z-score.
    - n_tests (int): The number of tests to be performed.
    - recompute_correlation (bool): A flag indicating whether to recompute the correlation.
    - min_number_bursts (int): The minimum number of bursts required.

    Returns:
    None. The function saves the computed correlations to a .npz file in 'data_dir'.
    
    Note: If a file with the same name already exists in 'data_dir' and 'recompute_correlation' is False, 
          the function will return without doing anything.
    """
    # extract values from dicionary c1
    data_dir = c1["data_dir"]
    rasters = c1["rasters"]
    rasters_DFF = c1["rasters_DFF"]
    binning_window = c1["binning_window"]
    subsample = c1["subsample"]
    scale_by_DFF = c1["scale_by_DFF"]
    zscore = c1["zscore"]
    n_tests = c1["n_tests"]
    recompute_correlation = c1["recompute_correlation"]
    min_number_bursts = c1["min_number_bursts"]

    # 
    fname_out = os.path.join(data_dir,
                                str(id)+ '.npz'
                                )

    # not used for now, but may wish to skip computation if file already exists
    if os.path.exists(fname_out) and recompute_correlation==False:
        return

    #        
    temp1 = rasters[id][::subsample]

    # scale by rasters_DFF
    if scale_by_DFF:
        temp1 = temp1*rasters_DFF[id][::subsample]

    # bin data in chunks of size binning_window
    if binning_window!=1:
        tt = []
        for q in range(0, temp1.shape[0], binning_window):
            temp = np.sum(temp1[q:q + binning_window])
            tt.append(temp)
        temp1 = np.array(tt)

    #
    corrs = []
    for p in range(rasters.shape[0]):
        temp2 = rasters[p][::subsample]
        
        # scale by rasters_DFF
        if scale_by_DFF:
            temp2 = temp2*rasters_DFF[p][::subsample]
        
        # 
        if binning_window!=1:
            tt = []
            for q in range(0, temp2.shape[0], binning_window):
                temp = np.sum(temp2[q:q + binning_window])
                tt.append(temp)
            temp2 = np.array(tt)
        
        #
        corr, corr_z = get_corr2(temp1, temp2, zscore, n_tests, min_number_bursts)

        # 
        corrs.append([id, p, corr[0], corr[1], corr_z[0]])

    #
    corrs = np.vstack(corrs)
    #
    np.savez(fname_out, 
             binning_window = binning_window,
             subsample = subsample,
             scale_by_DFF = scale_by_DFF,
             zscore_flag = zscore,
             id = id,
             compared_cells = corrs[:,1],
             pearson_corr = corrs[:,2],
             pvalue_pearson_corr = corrs[:,3],
             z_score_pearson_corr = corrs[:,4],
             n_tests = n_tests,
            )

    #return corrs



def get_corr2(temp1, temp2, zscore, n_tests=1000, min_number_bursts=0):
    """
    This function calculates the Pearson correlation coefficient between two arrays of data, temp1 and temp2. 
    If zscore is True, the function also calculates the z-score of the correlation coefficient based on n_tests random shuffles of temp2.
    
    :param temp1: 1D numpy array of data
    :param temp2: 1D numpy array of data
    :param zscore: boolean, if True calculate z-score of correlation coefficient
    :param n_tests: int, number of random shuffles to perform when calculating z-score (default=1000)
    :param min_number_bursts: int, minimum number of bursts
    :return: tuple containing the Pearson correlation coefficient between temp1 and temp2, and the z-score of the correlation coefficient (if zscore=True) or np.nan (if zscore=False)
    """
    # check if all values are the same 
    if len(np.unique(temp1))==1 or len(np.unique(temp2))==1:
        corr = [np.nan,1]
        return corr, [np.nan]
    
    # check if number bursts will be below self.min_num_bursts
    if np.sum(temp1!=0)<min_number_bursts or np.sum(temp2!=0)<min_number_bursts:
        corr = [np.nan,1]
        return corr, [np.nan]

    # if using dynamic correlation we need to compute the correlation for 1000 shuffles
    corr_original = scipy.stats.pearsonr(temp1, temp2)

    # make array and keep track
    corr_array = []
    corr_array.append(corr_original[0])
                                
    #
    if zscore:
        corr_s = []
        for k in range(n_tests):
            # choose a random index ranging from 0 to the length of the array minus 1
            idx = np.random.randint(-temp2.shape[0], temp2.shape[0],1)
            #idx = np.random.randint(temp2.shape[0])
            temp2_shuffled = np.roll(temp2, idx)
            corr_s = scipy.stats.pearsonr(temp1, temp2_shuffled)
            corr_array.append(corr_s[0])

        # compute z-score
        corr_z = stats.zscore(corr_array)

    else:
        corr_z = [np.nan]

    return corr_original, corr_z