import numpy as np
from scipy.signal import convolve2d
from sklearn.decomposition import PCA

def remove_compartment_stripes(Lib, chrom, h_candi, v_candi, window=10000000, smooth_factor=1):

    res = Lib.binsize
    min_matrix = 10
    kernel = np.ones((smooth_factor, smooth_factor))
    step = window//2

    # consistent expected value
    cM = Lib.matrix(balance=True, sparse=True).fetch(chrom)
    tmp = np.isfinite(Lib.bins().fetch(chrom)['weight'].values)
    maxdis = min(cM.shape[0]-1, window//res)
    pre_cal = {}
    for i in range(maxdis+1):
        if i > 0:
            valid = tmp[:-i] * tmp[i:]
        else:
            valid = tmp
        current = cM.diagonal(i)[valid]
        if current.size > 0:
            v = current.mean()
            pre_cal[i] = v

    start = 0
    while Lib.chromsizes[chrom] - start > res*min_matrix:
        end = min(Lib.chromsizes[chrom], start+window)
        mask = np.isnan(Lib.bins().fetch((chrom,start,end))['weight'].values) # gaps
        tmp = np.logical_not(mask) # valid rows or columns
        if tmp.sum() < min_matrix:
            start += step
            continue
        cM = Lib.matrix(balance=True, sparse=False).fetch((chrom, start, end))
        cM[np.isnan(cM)] = 0
        # smooth matrix
        smooth = convolve2d(cM, kernel, mode='same')
        # expected matrix
        expected = np.zeros_like(cM)
        idx = range(expected.shape[0])
        for i in idx:
            if i in pre_cal:
                if i > 0:
                    expected[idx[:-i], idx[i:]] = pre_cal[i]
                    expected[idx[i:], idx[:-i]] = pre_cal[i]
                else:
                    expected[idx, idx] = pre_cal[i]
        expected = convolve2d(expected, kernel, mode='same') # smooth the expected matrix with same smoothing factor
        # PCA
        obs_exp = np.zeros_like(expected)
        obs_exp[expected!=0] = smooth[expected!=0] / expected[expected!=0]
        # remove gaps
        index = list(np.where(mask)[0])
        convert_index = np.where(np.logical_not(mask))[0]
        temp = np.delete(obs_exp, index, 0)
        new_obs_exp = np.delete(temp, index, 1)
        # pearson correlation matrix
        pearson = np.corrcoef(new_obs_exp)
        pca = PCA(n_components=3, whiten=True, svd_solver='arpack')
        fp = pca.fit_transform(pearson)[:,0]
        # map back to original coordinates
        n = obs_exp.shape[0]
        arr = np.zeros(n)
        arr[convert_index] = fp # PC1
        with open('chr21.bed','w') as out:
            for i in range(arr.size):
                if arr[i] > 0:
                    tmp = ['chr21', str(start+i*res), str(start+i*res+res)]
                    out.write('\t'.join(tmp)+'\n')

        # correct horizontal stripes
        for h in h_candi:
            rh = h - start//res
            if (rh < 0) or (rh + 1 > arr.size):
                continue
            tmp = h_candi[h]
            for d in tmp:
                count = 0
                for i in range(d[0], d[1]):
                    ri = i - start//res
                    if (ri < 0) or (ri + 1 > arr.size):
                        continue
                    if (arr[rh] > 0.5) and (arr[ri] > 0.5) and ((arr[rh:ri+1]<-0.5).sum() > 0):
                        count += 1
                    else:
                        if (arr[rh] < -0.5) and (arr[ri] < -0.5) and ((arr[rh:ri+1]>0.5).sum() > 0):
                            count += 1
                if (count / (d[1]-d[0]) > 0.2):
                    tmp.remove(d)
        
        # correct vertical stripes
        for v in v_candi:
            rv = v - start//res
            if (rv < 0) or (rv + 1 > arr.size):
                continue
            tmp = v_candi[v]
            for d in tmp:
                count = 0
                for i in range(d[0], d[1]):
                    ri = i - start//res
                    if (ri < 0) or (ri + 1 > arr.size):
                        continue
                    if (arr[rv] > 0.5) and (arr[ri] > 0.5) and ((arr[ri:rv+1]<-0.5).sum() > 0):
                        count += 1
                    else:
                        if (arr[rv] < -0.5) and (arr[ri] < -0.5) and ((arr[ri:rv+1]>0.5).sum() > 0):
                            count += 1
                if (count / (d[1]-d[0]) > 0.2):
                    tmp.remove(d)
        
        start += step
    
    h_stripes = {}
    v_stripes = {}
    for h in h_candi:
        if len(h_candi[h]):
            h_stripes[h] = h_candi[h]
    for v in v_candi:
        if len(v_candi[v]):
            v_stripes[v] = v_candi[v]
    
    return h_stripes, v_stripes