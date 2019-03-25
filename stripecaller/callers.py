import logging
import numpy as np
from scipy import sparse
from scipy.stats import poisson
from scipy.signal import convolve2d
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

def _horizontal_stripe(M, cM, maxapart, res, l_n, b_n, siglevel, fold, chromLen):

    maxapart = maxapart // res
    # only works for upper triangular matrix
    # valid pixels
    xi, yi = cM.nonzero()
    mask = ((yi - xi) <= maxapart) & ((yi - xi) > (l_n + b_n)) & \
           (xi + 1 > l_n) & (yi + l_n < chromLen) # basic filtering
    xi, yi = xi[mask], yi[mask]

    top_p = np.zeros(xi.size)
    bottom_p = np.ones(xi.size) # different initialization of top / bottom p-values
    top_fold = np.ones(xi.size) * 20
    bottom_fold = np.ones(xi.size)

    # observed -- sum of local (horizontal) raw signal
    pool_i = np.tile(xi, (l_n*2+1, 1))
    pool_j = np.r_[[yi+i for i in range(-l_n, l_n+1)]]
    o_array = np.asarray(M[pool_i, pool_j].sum(axis=0)).ravel()
    # bias estimation
    tmp = np.median(cM[pool_i, pool_j].toarray(), axis=0)
    biases = np.zeros(tmp.size)
    mask = tmp != 0
    biases[mask] = o_array[mask] / tmp[mask]

    # bottom expected
    pool_j = np.tile(yi, (b_n, 1))
    pool_i = np.r_[[xi+i for i in range(l_n+1,l_n+b_n+1)]]
    e_array = np.median(cM[pool_i, pool_j].toarray(), axis=0) * biases
    mask = e_array > 0
    bottom_p[mask] = poisson(e_array[mask]).sf(o_array[mask])
    bottom_fold[mask] = o_array[mask] / e_array[mask]
    # top expected
    fm = (xi > (l_n + b_n - 1)) # additional filtering
    txi, tyi = xi[fm], yi[fm]
    pool_j = np.tile(tyi, (b_n, 1))
    pool_i = np.r_[[txi-i for i in range(l_n+1,l_n+b_n+1)]]
    e_array = np.median(cM[pool_i, pool_j].toarray(), axis=0) * biases
    mask = e_array > 0
    tmp = poisson(e_array[mask]).sf(o_array[fm][mask])
    tmp_ = np.zeros(txi.size)
    tmp_[mask] = tmp
    top_p[fm] = tmp_
    tmp_ = np.ones(txi.size) * 20
    tmp_[mask] = o_array[mask] / e_array[mask]
    top_fold[fm] = tmp_

    candi_mask = (top_p < siglevel) & (bottom_p < siglevel) & (top_fold > fold) & (bottom_fold > fold)
    xi = xi[candi_mask]
    yi = yi[candi_mask]

    return xi, yi

def _vertical_stripe(M, cM, maxapart, res, l_n, b_n, siglevel, fold, chromLen):

    maxapart = maxapart // res
    # only works for upper triangular matrix
    # valid pixels
    xi, yi = cM.nonzero()
    mask = ((yi - xi) <= maxapart) & ((yi - xi) > (l_n + b_n)) & \
           (xi + 1 > l_n) & (yi + l_n < chromLen) # basic filtering
    xi, yi = xi[mask], yi[mask]

    right_p = np.zeros(xi.size)
    left_p = np.ones(xi.size)
    right_fold = np.ones(xi.size) * 20
    left_fold = np.ones(xi.size)

    # observed -- sum of local (vertical) raw signal
    pool_j = np.tile(yi, (l_n*2+1, 1))
    pool_i = np.r_[[xi+i for i in range(-l_n, l_n+1)]]
    o_array = np.asarray(M[pool_i, pool_j].sum(axis=0)).ravel()
    # bias estimation
    tmp = np.median(cM[pool_i, pool_j].toarray(), axis=0)
    biases = np.zeros(tmp.size)
    mask = tmp != 0
    biases[mask] = o_array[mask] / tmp[mask]

    # left expected
    pool_i = np.tile(xi, (b_n, 1))
    pool_j = np.r_[[yi-i for i in range(l_n+1,l_n+b_n+1)]]
    e_array = np.median(cM[pool_i, pool_j].toarray(), axis=0) * biases
    mask = e_array > 0
    left_p[mask] = poisson(e_array[mask]).sf(o_array[mask])
    left_fold[mask] = o_array[mask] / e_array[mask]
    # right expected
    fm = (yi + l_n + b_n < chromLen) # additional filtering
    txi, tyi = xi[fm], yi[fm]
    pool_i = np.tile(txi, (b_n, 1))
    pool_j = np.r_[[tyi+i for i in range(l_n+1,l_n+b_n+1)]]
    e_array = np.median(cM[pool_i, pool_j].toarray(), axis=0) * biases
    mask = e_array > 0
    tmp = poisson(e_array[mask]).sf(o_array[fm][mask])
    tmp_ = np.zeros(txi.size)
    tmp_[mask] = tmp
    right_p[fm] = tmp_
    tmp_ = np.ones(txi.size) * 20
    tmp_[mask] = o_array[mask] / e_array[mask]
    right_fold[fm] = tmp_

    candi_mask = (left_p < siglevel) & (right_p < siglevel) & (left_fold > fold) & (right_fold > fold)
    xi = xi[candi_mask]
    yi = yi[candi_mask]

    return xi, yi

def consecutive_runs(arr):

    arr.sort()
    pieces = np.split(arr, np.where(np.diff(arr)!=1)[0]+1)
    maxlen = max(map(len, pieces))

    return pieces, maxlen

def extend_stretches(pieces, min_seed_len=6, max_gap=5, min_stripe_len=9):

    filtered = [p for p in pieces if len(p)>=min_seed_len] # can't be empty
    stripes = []
    seed = filtered[0]
    for p in filtered[1:]:
        if p[0] - seed[-1] < (max_gap + 2):
            seed = np.r_[seed, p]
        else:
            if seed[-1] - seed[0] + 1 >= min_stripe_len:
                stripes.append([seed[0], seed[-1]+1])
            seed = p
    
    if seed[-1] - seed[0] + 1 >= min_stripe_len:
        stripes.append([seed[0], seed[-1]+1])
    
    return stripes

def call_stripes(M, cM, maxapart, res, l_n, b_n, siglevel, fold, chromLen,
                 min_seed_len=6, max_gap=5, min_stripe_len=9):
    
    # call horizontal stripes
    xi, yi = _horizontal_stripe(M, cM, maxapart, res, l_n, b_n, siglevel, fold, chromLen)
    h_stripes = {}
    for anchor_x in np.unique(xi):
        ys = yi[xi==anchor_x]
        pieces, maxlen = consecutive_runs(ys)
        if maxlen < min_seed_len:
            continue
        tmp = extend_stretches(pieces,
                               min_seed_len=min_seed_len,
                               max_gap=max_gap,
                               min_stripe_len=min_stripe_len)
        h_stripes[anchor_x] = tmp
    
    # call vertical stripes
    xi, yi = _vertical_stripe(M, cM, maxapart, res, l_n, b_n, siglevel, fold, chromLen)
    v_stripes = {}
    for anchor_y in np.unique(yi):
        xs = xi[yi==anchor_y]
        pieces, maxlen = consecutive_runs(xs)
        if maxlen < min_seed_len:
            continue
        tmp = extend_stretches(pieces,
                               min_seed_len=min_seed_len,
                               max_gap=max_gap,
                               min_stripe_len=min_stripe_len)
        v_stripes[anchor_y] = tmp

    return h_stripes, v_stripes
        
def remove_compartment_stripes(Lib, chrom, h_stripes, v_stripes, window=10000000, step=5000000, smooth_factor=1):

    res = Lib.binsize
    min_matrix = 10
    kernel = np.ones((smooth_factor, smooth_factor))

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

        code = arr > 0
        # correct horizontal stripes
        for h in h_stripes:
            rh = h - start//res
            if (rh < 0) or (rh + 1 > arr.size):
                continue
            tmp = h_stripes[h]
            count = 0
            for i in range(tmp[0], tmp[1]):
                ri = i - start//res
                if (ri < 0) or (ri + 1 > arr.size):
                    continue
                if (code[rh]==code[ri]) and (code[rh:ri+1].sum() > 0) and (code[rh:ri+1].sum() < code[rh:ri+1].size):
                    count += 1
            if count / (tmp[1]-tmp[0]) > 0.3:
                del h_stripes[h]
        
        # correct vertical stripes
        for v in v_stripes:
            rv = v - start//res
            if (rv < 0) or (rv + 1 > arr.size):
                continue
            tmp = v_stripes[v]
            count = 0
            for i in range(tmp[0], tmp[1]):
                ri = i - start//res
                if (ri < 0) or (ri + 1 > arr.size):
                    continue
                if (code[rv]==code[ri]) and (code[ri:rv+1].sum() > 0) and (code[ri:rv+1].sum() < code[ri:rv+1].size):
                    count += 1
            if count / (tmp[1]-tmp[0]) > 0.3:
                del v_stripes[v]
    
    return h_stripes, v_stripes








