import logging
import numpy as np
from scipy import sparse
from scipy.stats import poisson

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