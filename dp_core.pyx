# cython: language_level=3

"""
Copyright 2019 Brian Thompson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

cimport numpy as np
cimport cython


def make_x_y_offsets(alignment_types):
    # alignment types for which we will precompute costs

    # deletion/insertion is added later
    for x, y in alignment_types:
        assert (x > 0)
        assert (y > 0)

    x_offsets = np.array([x for x, y in alignment_types], dtype=np.int32)  # MUST **NOT** INCLUDE (0,1), (1,0)
    y_offsets = np.array([y for x, y in alignment_types], dtype=np.int32)  # MUST **NOT** INCLUDE (0,1), (1,0)
    return x_offsets, y_offsets


def make_dense_costs(np.ndarray[float, ndim=3] vecs0,  # itput
                     np.ndarray[float, ndim=3] vecs1,  # input
                     np.ndarray[float, ndim=2] norm0,  # input
                     np.ndarray[float, ndim=2] norm1,  # input
                     int offset0 = 0,  # index into vecs0/norms0
                     int offset1 = 0,  # index into vecs1/norms1
                     ):
    """
    Make a full N*M feature matrix. By default, makes 1-1 alignments, 
       can build others by specifying offset0, offset1 to index into
       vecs0, norms0 and vecs1, norms1 respectivly. 
    """
    assert vecs0.shape[0] > offset0
    assert vecs1.shape[0] > offset1
    assert norm0.shape[0] > offset0
    assert norm1.shape[0] > offset1

    cdef int size0 = np.shape(vecs0)[1]
    assert norm0.shape[1] == size0

    cdef int size1 = np.shape(vecs1)[1]
    assert norm1.shape[1] == size1

    cdef int vecsize = np.shape(vecs0)[2]
    assert vecs1.shape[2] == vecsize

    cdef int xi, yi
    cdef float sumx

    cdef np.ndarray[float, ndim=2] costs = np.empty((size0, size1), dtype=np.float32)

    for xi in range(size0):
        for yi in range(size1):
            sumx = 0.0
            for jj in range(vecsize):
                sumx += vecs0[offset0, xi, jj] * vecs1[offset1, yi, jj]

            costs[xi, yi] = 2.0 * (1.0 - sumx) / (1e-6 + norm0[offset0, xi] + norm1[offset1, yi])
            # normalize by alignment type  
            costs[xi, yi] = costs[xi, yi] * (offset0 + 1) * (offset1 + 1)

    return costs


def dense_dp(np.ndarray[float, ndim=2] alignment_cost, float pen):
    """
    Compute cost matrix (csum) and backpointers (bp) 
    from full 2-D 1-1 alignment costs matrix (alignment_cost) 
    """

    size0 = alignment_cost.shape[0]
    size1 = alignment_cost.shape[1]
    # csum and traceback matrix are both on nodes
    #   so they are +1 in each dimension compared to the jump costs matrix
    # For anything being used in accumulation, use float64
    cdef np.ndarray[double, ndim=2] csum = np.empty((size0 + 1, size1 + 1), dtype=np.float64)
    cdef np.ndarray[int, ndim=2] bp = np.empty((size0 + 1, size1 + 1), dtype=np.int32)

    # bp and csum are nodes, 
    #   while alignment_cost is the cost of going between the nodes
    # Size of nodes should be one larger than alignment costs
    b0, b1 = np.shape(bp)
    c0, c1 = np.shape(csum)
    j0, j1 = np.shape(alignment_cost)
    assert (b0 == c0 == j0 + 1)
    assert (b1 == c1 == j1 + 1)

    cdef int cmax = np.shape(csum)[1]
    cdef int rmax = np.shape(csum)[0]
    cdef int c, r
    cdef double cost0, cost1, cost2

    # initialize the all c-direction deletion path
    for c in range(cmax):
        csum[0, c] = c * pen
        bp[0, c] = 1

    # initialize the all r-direction deletion path
    for r in range(rmax):
        csum[r, 0] = r * pen
        bp[r, 0] = 2

    # Initial cost is 0.0
    csum[0, 0] = 0.0  # noop
    bp[0, 0] = 4  # should not matter

    # Calculate the rest recursively
    for c in range(1, cmax):
        for r in range(1, rmax):

            # alignment_cost indexes are off by 1 wrt
            #   csum/bp, since csum/bp are nodes
            cost0 = csum[r - 1, c - 1] + alignment_cost[r - 1, c - 1]
            cost1 = csum[r, c - 1] + pen
            cost2 = csum[r - 1, c] + pen

            csum[r, c] = cost0
            bp[r, c] = 0

            if cost1 < csum[r, c]:
                csum[r, c] = cost1
                bp[r, c] = 1
            if cost2 < csum[r, c]:
                csum[r, c] = cost2
                bp[r, c] = 2

    return csum, bp


def score_path(np.ndarray[int, ndim=1] xx,
               np.ndarray[int, ndim=1] yy,
               np.ndarray[float, ndim=1] norm1,
               np.ndarray[float, ndim=1] norm2,
               np.ndarray[float, ndim=2] vecs1,
               np.ndarray[float, ndim=2] vecs2,
               np.ndarray[float, ndim=1] out):
    cdef int xi, yi, ii, jj
    cdef float outx
    cdef int lenxy = xx.shape[0]
    cdef int vecsize = vecs1.shape[1]

    for ii in range(lenxy):
        xi = xx[ii]
        yi = yy[ii]
        outx = 0.0
        for jj in range(vecsize):
            outx += vecs1[xi, jj] * vecs2[yi, jj]
        out[ii] = 2.0 * (1.0 - outx) / (norm1[xi] + norm2[yi])


# Bounds checking and wraparound slow things down by about 2x
# Division by 0 checking has minimal speed impact
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)  # use c-style division (no division-by-zero check)
def make_sparse_costs(np.ndarray[float, ndim=3] vecs0,  # intput: num aligns X num sents X dim
                      np.ndarray[float, ndim=3] vecs1,  # input
                      np.ndarray[float, ndim=2] norms0,  # intput: num aligns X num sents
                      np.ndarray[float, ndim=2] norms1,  # input
                      x_y_path,
                      alignment_types,
                      int width_over2):
    """
    Make features for DP, *for lines running across approximate path*, *for each alignment type*
    x_offsets, y_offsets should not include (0,1), (1,0)

    Basically, we take the feature matrix, rotate it 45 degress, 
       and compute a "wavy" matrix for the features.
    It's like the diagonal but it moves around to hopefully always include the true path.
    """

    cdef np.ndarray[int, ndim=2] x_y_path_ = np.array(x_y_path).astype(np.int32)

    assert (vecs0.shape[0] == norms0.shape[0])
    assert (vecs1.shape[0] == norms1.shape[0])

    assert (vecs0.shape[1] == norms0.shape[1])
    assert (vecs1.shape[1] == norms1.shape[1])

    # check how many overlaps vectors were passed in
    num_overlaps_in_vecs0 = vecs0.shape[0]
    num_overlaps_in_vecs1 = vecs1.shape[0]

    # check how many overlaps were requested
    # edge case: alignment_types could be empty
    #    In that case, we should just return insertions/deletions
    #    and max_x_overlap == max_y_overlap == 0
    max_x_overlap = max([0] + [x for x, y in alignment_types])  # add [0] in case alignment_types is empty
    max_y_overlap = max([0] + [y for x, y in alignment_types])  # add [0] in case alignment_types is empty

    # note: alignment types are specified 1-based, but vectors are stored 0-based
    if max_x_overlap > num_overlaps_in_vecs0:
        raise Exception('%d x overlaps requrested (via alignment_types), but vecs0 only has %d' % (
            max_x_overlap, num_overlaps_in_vecs0))
    if max_y_overlap > num_overlaps_in_vecs1:
        raise Exception('%d y overlaps requrested (via alignment_types), but vecs1 only has %d' % (
            max_y_overlap, num_overlaps_in_vecs1))

    # number of sentences in each document
    cdef int xsize = vecs0.shape[1]
    cdef int ysize = vecs1.shape[1]

    # vector diminsions should match
    assert (vecs0.shape[2] == vecs1.shape[2])

    cdef np.ndarray[int, ndim=1] x_offsets, y_offsets
    x_offsets, y_offsets = make_x_y_offsets(alignment_types)

    # reserve outputs
    a_len = x_y_path_.shape[0]
    b_len = 2 * width_over2
    cdef np.ndarray[float, ndim=3] a_b_feats = np.empty((len(alignment_types), a_len, b_len), dtype=np.float32)
    cdef np.ndarray[int, ndim=1] b_offset = np.empty(a_len).astype(np.int32)

    cdef int x, y, aa, bb, xx, yy, a_idx, b_idx, bb2, x_offset, y_offset, ii_align, x_offset_idx, y_offset_idx
    cdef int vecsize = vecs0.shape[2]
    cdef int num_alignments = x_offsets.shape[0]

    cdef float sumx, feat
    cdef float inf = np.inf

    for ii in range(x_y_path_.shape[0]):
        x = x_y_path_[ii, 0]
        y = x_y_path_[ii, 1]

        # convert xy to ab cords
        aa = x + y
        bb = y

        a_idx = aa
        b_offset[aa] = bb - width_over2
        for b_idx, bb2 in enumerate(range(bb - width_over2, bb + width_over2)):
            # convert ab to xy cords
            xx = aa - bb2
            yy = bb2

            for ii_align in range(num_alignments):
                x_offset = x_offsets[ii_align]
                x_offset_idx = x_offset - 1  # overlaps start at 1, vectors stored 0-based
                y_offset = y_offsets[ii_align]
                y_offset_idx = y_offset - 1

                if 0 <= xx < xsize and 0 <= yy < ysize:
                    sumx = 0.0
                    for jj in range(vecsize):
                        sumx += vecs0[x_offset_idx, xx, jj] * vecs1[y_offset_idx, yy, jj]
                    feat = 2.0 * x_offset * y_offset * (1.0 - sumx) / (
                            1e-6 + norms0[x_offset_idx, xx] + norms1[y_offset_idx, yy])

                else:
                    feat = inf

                a_b_feats[ii_align, a_idx, b_idx] = feat

    return a_b_feats, b_offset


def sparse_dp(np.ndarray[float, ndim=3] a_b_costs,
              np.ndarray[int, ndim=1] b_offset_in,
              alignment_types,
              double del_penalty,
              int x_in_size,
              int y_in_size):
    """
    Do DP along a path, using features saved off along path.
    x_offsets, y_offsets should not include (0,1), (1,0)

    xsize, ysize refer to the costs a_b_csum, but in x/y space

    As in the simpler full-DP case, 
       we compute cumulative costs and backpointers on notes,
       and there are COSTS associated with moving between them.

    This means the size of the notes  +1,+1 larger (in x,y) than the COSTS.
    
    So the size of a_b_csum, a_b_xp, a_b_yp are all one larger in x and y compared to the costs

    In order to save memory (and time, vs a sparse matrix with hashes to look up values), let:
             a = x + y
             b = x - y

    b_offsets tells us how far from the left edge the features are computed for.
         basically it's like we are computing along the diagonal, 
         but we shift the diagonal around based on our belief
         about where the alignments are. 

    b_offsets is used for both costs AND csum, backpointers, so it needs to be 
        +2 longer (it is in the a-direction) than the costs (in the a direction)

    """
    cdef np.ndarray[int, ndim=1] x_offsets, y_offsets
    x_offsets, y_offsets = make_x_y_offsets(alignment_types)

    # make x/y offsets, including (0,1), (1,), i.e. including deletion and insertion
    x_offsets = np.concatenate([x_offsets, np.array([0, 1], dtype=np.int32)])
    y_offsets = np.concatenate([y_offsets, np.array([1, 0], dtype=np.int32)])

    cdef int a_in_size = a_b_costs.shape[1]
    cdef int b_in_size = a_b_costs.shape[2]

    cdef int a_out_size = a_in_size + 2
    cdef int b_out_size = b_in_size

    cdef int x_out_size = x_in_size + 1
    cdef int y_out_size = y_in_size + 1

    # costs are the costs of going between nodes.
    # in x,y for the nodes, we basically add a buffer 
    #   at x=0 and y=0, and shift the cost by (x=+1,y=+1)
    # In a,b space, this means adding two points (for the buffer)
    #      at the beginning, and shifting by (a=+0,b=+1) since
    #      a=x+y and b=y
    # for the first two points, we can simply replicate the
    #    original b_offset, since it should be -width_over2
    # i.e. b_offset_in[0] == -width_over2
    extra_two_points = np.array([b_offset_in[0], b_offset_in[0]], dtype=np.int32)
    cdef np.ndarray[int, ndim=1] b_offset_out = np.concatenate([extra_two_points, b_offset_in + 1])

    # outputs
    # For anything being used in accumulation, use float64
    cdef np.ndarray[double, ndim=2] a_b_csum = np.zeros((a_in_size + 2, b_in_size),
                                                        dtype=np.float64) + np.inf  # error cumulative sum
    cdef np.ndarray[int, ndim=2] a_b_xp = np.zeros((a_in_size + 2, b_in_size), dtype=np.int32) - 2  # backpointer for x
    cdef np.ndarray[int, ndim=2] a_b_yp = np.zeros((a_in_size + 2, b_in_size), dtype=np.int32) - 2  # backpointer for y

    cdef int num_alignments = x_offsets.shape[0]
    cdef double inf = np.inf
    cdef int xx_out, yy_out, ii_align, x_offset, y_offset
    cdef int aa_in_cost, bb_in_cost, aa_out, bb_out, aa_out_prev, bb_out_prev, xx_in_cost, yy_in_cost, xx_out_prev, yy_out_prev

    cdef double alignment_cost, total_cost, prev_cost

    # increasing in a is the same as going along diagonals in x/y, so DP order works
    #  (and any ordering is fine in b - nothing depends on values adjacent on diagonal in x/y)
    for aa_out in range(a_in_size + 2):
        for bb_out in range(b_in_size):
            #xx_out, yy_out = ab2xy_w_offset(aa_out, bb_out, b_offset_out)
            yy_out = bb_out + b_offset_out[aa_out]
            xx_out = aa_out - yy_out

            # edge case: all deletions in y-direction
            if xx_out == 0 and 0 <= yy_out < y_out_size:
                a_b_csum[aa_out, bb_out] = del_penalty * yy_out
                a_b_xp[aa_out, bb_out] = 0
                a_b_yp[aa_out, bb_out] = 1

            # edge case: all deletions in x-direction
            elif yy_out == 0 and 0 <= xx_out < x_out_size:
                a_b_csum[aa_out, bb_out] = del_penalty * xx_out
                a_b_xp[aa_out, bb_out] = 1
                a_b_yp[aa_out, bb_out] = 0

            else:
                # initialize output to inf
                a_b_csum[aa_out, bb_out] = inf
                a_b_xp[aa_out, bb_out] = -42
                a_b_yp[aa_out, bb_out] = -42

                for ii_align in range(num_alignments):
                    x_offset = x_offsets[ii_align]
                    y_offset = y_offsets[ii_align]

                    # coords of location of alignment cost, in input x/y space
                    xx_in_cost = xx_out - 1  # features were front padded,
                    yy_in_cost = yy_out - 1  #   so offset is always 1

                    # the coords of location of previous cumsum cost, in input x/y space
                    xx_out_prev = xx_out - x_offset
                    yy_out_prev = yy_out - y_offset

                    if 0 <= xx_in_cost < x_in_size and 0 <= yy_in_cost < y_in_size and 0 <= xx_out_prev < x_out_size and 0 <= yy_out_prev < y_out_size:
                        # convert x,y to a,b
                        aa_in_cost = xx_in_cost + yy_in_cost
                        bb_in_cost = yy_in_cost - b_offset_in[aa_in_cost]

                        aa_out_prev = xx_out_prev + yy_out_prev
                        bb_out_prev = yy_out_prev - b_offset_out[aa_out_prev]

                        if 0 <= aa_in_cost < a_in_size and 0 <= bb_in_cost < b_in_size and 0 <= aa_out_prev < a_out_size and 0 <= bb_out_prev < b_out_size:
                            if x_offset == 0 or y_offset == 0:
                                alignment_cost = del_penalty
                            else:
                                alignment_cost = a_b_costs[ii_align, aa_in_cost, bb_in_cost]

                            prev_cost = a_b_csum[aa_out_prev, bb_out_prev]

                            total_cost = prev_cost + alignment_cost

                            if total_cost < a_b_csum[aa_out, bb_out]:
                                a_b_csum[aa_out, bb_out] = total_cost
                                a_b_xp[aa_out, bb_out] = x_offset
                                a_b_yp[aa_out, bb_out] = y_offset

    return a_b_csum, a_b_xp, a_b_yp, b_offset_out
