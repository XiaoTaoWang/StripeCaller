import logging, bisect, copy
from collections import defaultdict, Counter
import numpy as np
from scipy import sparse
from scipy.stats import poisson
from scipy.signal import find_peaks, peak_widths

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
    e_array = np.median(cM[pool_i, pool_j].toarray(), axis=0) * biases[fm]
    mask = e_array > 0
    tmp = poisson(e_array[mask]).sf(o_array[fm][mask])
    tmp_ = np.zeros(txi.size)
    tmp_[mask] = tmp
    top_p[fm] = tmp_
    tmp_ = np.ones(txi.size) * 20
    tmp_[mask] = o_array[fm][mask] / e_array[mask]
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
    e_array = np.median(cM[pool_i, pool_j].toarray(), axis=0) * biases[fm]
    mask = e_array > 0
    tmp = poisson(e_array[mask]).sf(o_array[fm][mask])
    tmp_ = np.zeros(txi.size)
    tmp_[mask] = tmp
    right_p[fm] = tmp_
    tmp_ = np.ones(txi.size) * 20
    tmp_[mask] = o_array[fm][mask] / e_array[mask]
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

def call_stripes(key, M, cM, maxapart, res, l_n, b_n, siglevel, fold, chromLen,
                 min_seed_len=6, max_gap=5, min_stripe_len=9):
    
    # call horizontal stripes
    logger.info('Chrom: {0}, calling horizontal stripes ...'.format(key))
    # first run
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
        if len(tmp):
            h_stripes[anchor_x] = tmp

    if len(h_stripes):
        xi, yi = local_cluster(h_stripes, min_count=min_stripe_len, res=res)
        # second run
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
            if len(tmp):
                h_stripes[anchor_x] = tmp
    
    # call vertical stripes
    logger.info('Chrom: {0}, calling vertical stripes ...'.format(key))
    # first run
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
        if len(tmp):
            v_stripes[anchor_y] = tmp
    
    if len(v_stripes):
        yi, xi = local_cluster(v_stripes, min_count=min_stripe_len, res=res)
        # second run
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
            if len(tmp):
                v_stripes[anchor_y] = tmp

    return h_stripes, v_stripes

def local_cluster(candi_dict, min_count=9, min_dis=50000, wlen=400000, res=10000):

    min_dis = max(min_dis//res, 1)
    wlen = min(wlen//res, 10)

    count = defaultdict(int)
    for k in candi_dict:
        count[k] = sum([v[1]-v[0] for v in candi_dict[k]])
    refidx = range(min(count)-1, max(count)+2) # extend 1 bin
    signal = np.r_[[count[i] for i in refidx]]
    summits = find_peaks(signal, height=min_count, distance=min_dis)[0]
    sorted_summits = [(signal[i],i) for i in summits]
    sorted_summits.sort(reverse=True)

    peaks = set()
    records = {}
    for _, i in sorted_summits:
        tmp = peak_widths(signal, [i], rel_height=1, wlen=wlen)[2:4]
        li, ri = int(np.round(tmp[0][0])), int(np.round(tmp[1][0]))
        lb = refidx[li]
        rb = refidx[ri]
        if not len(peaks):
            peaks.add((refidx[i], lb, rb))
            for b in range(lb, rb+1):
                records[b] = (refidx[i], lb, rb)
        else:
            for b in range(lb, rb+1):
                if b in records:
                    # merge anchors
                    m_lb = min(lb, records[b][1])
                    m_rb = max(rb, records[b][2])
                    summit = records[b][0] # always the highest summit
                    peaks.remove(records[b])
                    break
            else: # loop terminates normally
                m_lb, m_rb, summit = lb, rb, refidx[i]
            peaks.add((summit, m_lb, m_rb))
            for b in range(m_lb, m_rb+1):
                records[b] = (summit, m_lb, m_rb)
    
    anchor_ = []
    coords_ = []
    for p in sorted(peaks):
        tmp = []
        for i in range(p[1], p[2]+1):
            if i in candi_dict:
                for c in candi_dict[i]:
                    tmp.extend(range(c[0], c[1]))
        tmp = sorted(set(tmp))
        anchor_.extend([p[0]]*len(tmp))
        coords_.extend(tmp)
    
    anchor_ = np.r_[anchor_]
    coords_ = np.r_[coords_]

    return anchor_, coords_

def remove_compartment_stripes(stripes, TADs, chrom):

    ref = TADs[chrom]
    tmp = copy.deepcopy(stripes)
    for k in stripes:
        for d in stripes[k]:
            count_pixel = 0
            count_tads = []
            for i in range(d[0], d[1]):
                inter = [i, k] if i < k else [k, i]
                cache = check_in(inter, ref)
                if len(cache):
                    count_pixel += 1
                for t in cache:
                    count_tads.append(t)
            if (not count_pixel) and (not len(count_tads)):
                tmp[k].remove(d)
                continue
            count_tads = Counter(count_tads).most_common(1)[0]
            if (count_pixel / (d[1] - d[0]) < 0.9) and (count_tads[1] / (count_tads[0][1] - count_tads[0][0]) < 0.5):
                tmp[k].remove(d)
    
    corrected = {}
    for k in tmp:
        if len(tmp[k]):
            corrected[k] = tmp[k]
    
    return corrected

def remove_loop_pixels(stripes, loops, chrom, min_stripe_len=9):

    corrected = {}
    for k in stripes:
        tmp = []
        for d in stripes[k]:
            for i in range(d[0], d[1]):
                if i < k:
                    if not (i, k) in loops[chrom]:
                        tmp.append(i)
                else:
                    if not (k, i) in loops[chrom]:
                        tmp.append(i)
        tmp = np.r_[tmp]
        if len(tmp) < min_stripe_len:
            continue
        pieces, _ = consecutive_runs(tmp)
        domains = []
        for p in pieces:
            if len(p) >= min_stripe_len:
                domains.append([p[0], p[-1]+1])
        if len(domains):
            corrected[k] = domains
    
    return corrected

def check_in(coord, List, mismatch=1):

    cache = set()
    idx = max(0, bisect.bisect(List, coord)-1)
    for q in List[idx:]:
        if ((q[0] <= coord[0]) and (q[1] >= coord[1])) or \
           ((abs(q[0]-coord[0]) <= mismatch) and (q[0] <= coord[1] <= q[1])) or \
           ((abs(q[1]-coord[1]) <= mismatch) and (q[0] <= coord[0] <= q[1])):
            cache.add(tuple(q))
    
    return cache

def load_TADs(fil, res):

    tads = {}
    with open(fil, 'r') as source:
        for line in source:
            parse = line.rstrip().split()
            chrom, s, e = parse[0], int(parse[1]), int(parse[2])
            if len(parse)==4:
                if parse[-1]!='0':
                    continue
            if not chrom in tads:
                tads[chrom] = []
            tads[chrom].append([s//res, e//res])
    for c in tads:
        tads[c].sort()
    
    return tads

def load_loops(fil, res, mismatch=1):

    loops = {}
    with open(fil, 'r') as source:
        for line in source:
            parse = line.rstrip().split()
            chrom, s1, e1, s2, e2 = parse[0], int(parse[1]), int(parse[2]), int(parse[4]), int(parse[5])
            p1 = (s1 + e1) // (2*res)
            p2 = (s2 + e2) // (2*res)
            if not chrom in loops:
                loops[chrom] = set()
            for i in range(max(p1-mismatch,0), p1+mismatch+1):
                for j in range(max(p2-mismatch,0), p2+mismatch+1):
                    loops[chrom].add((i, j))
    
    return loops