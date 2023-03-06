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

import logging
import sys
from ast import literal_eval
from collections import OrderedDict
from math import ceil
from time import time

import numpy as np

import pyximport
pyximport.install(setup_args={'include_dirs':np.get_include()}, inplace=True, reload_support=True)

from dp_core import make_dense_costs, score_path, sparse_dp, make_sparse_costs, dense_dp

logger = logging.getLogger('vecalign')  # set up in vecalign.py


def preprocess_line(line):
    line = line.strip()
    if len(line) == 0:
        line = 'BLANK_LINE'
    return line


def yield_overlaps(lines, num_overlaps):
    lines = [preprocess_line(line) for line in lines]
    for overlap in range(1, num_overlaps + 1):
        for out_line in layer(lines, overlap):
            # check must be here so all outputs are unique
            out_line2 = out_line[:10000]  # limit line so dont encode arbitrarily long sentences
            yield out_line2


def read_in_embeddings(text_file, embed_file):
    """
    Given a text file with candidate sentences and a corresponing embedding file,
       make a maping from candidate sentence to embedding index, 
       and a numpy array of the embeddings
    """
    sent2line = dict()
    with open(text_file, 'rt', encoding="utf-8") as fin:
        for ii, line in enumerate(fin):
            if line.strip() in sent2line:
                raise Exception('got multiple embeddings for the same line')
            sent2line[line.strip()] = ii

    line_embeddings = np.fromfile(embed_file, dtype=np.float32, count=-1)
    if line_embeddings.size == 0:
        raise Exception('Got empty embedding file')

    laser_embedding_size = line_embeddings.size // len(sent2line)  # currently hardcoded to 1024
    if laser_embedding_size != 1024:
        logger.warning('expected an embedding size of 1024, got %s', laser_embedding_size)
    logger.info('laser_embedding_size determined to be %d', laser_embedding_size)
    line_embeddings.resize(line_embeddings.shape[0] // laser_embedding_size, laser_embedding_size)
    return sent2line, line_embeddings


def make_doc_embedding(sent2line, line_embeddings, lines, num_overlaps):
    """
    lines: sentences in input document to embed
    sent2line, line_embeddings: precomputed embeddings for lines (and overlaps of lines)
    """

    lines = [preprocess_line(line) for line in lines]

    vecsize = line_embeddings.shape[1]

    vecs0 = np.empty((num_overlaps, len(lines), vecsize), dtype=np.float32)

    for ii, overlap in enumerate(range(1, num_overlaps + 1)):
        for jj, out_line in enumerate(layer(lines, overlap)):
            try:
                line_id = sent2line[out_line]
            except KeyError:
                logger.warning('Failed to find overlap=%d line "%s". Will use random vector.', overlap, out_line)
                line_id = None

            if line_id is not None:
                vec = line_embeddings[line_id]
            else:
                vec = np.random.random(vecsize) - 0.5
                vec = vec / np.linalg.norm(vec)

            vecs0[ii, jj, :] = vec

    return vecs0


def make_norm1(vecs0):
    """
    make vectors norm==1 so that cosine distance can be computed via dot product
    """
    for ii in range(vecs0.shape[0]):
        for jj in range(vecs0.shape[1]):
            norm = np.sqrt(np.square(vecs0[ii, jj, :]).sum())
            vecs0[ii, jj, :] = vecs0[ii, jj, :] / (norm + 1e-5)


def layer(lines, num_overlaps, comb=' '):
    """
    make front-padded overlapping sentences
    """
    if num_overlaps < 1:
        raise Exception('num_overlaps must be >= 1')
    out = ['PAD', ] * min(num_overlaps - 1, len(lines))
    for ii in range(len(lines) - num_overlaps + 1):
        out.append(comb.join(lines[ii:ii + num_overlaps]))
    return out


def read_alignments(fin):
    alignments = []
    with open(fin, 'rt', encoding="utf-8") as infile:
        for line in infile:
            fields = [x.strip() for x in line.split(':') if len(x.strip())]
            if len(fields) < 2:
                raise Exception('Got line "%s", which does not have at least two ":" separated fields' % line.strip())
            try:
                src = literal_eval(fields[0])
                tgt = literal_eval(fields[1])
            except:
                raise Exception('Failed to parse line "%s"' % line.strip())
            alignments.append((src, tgt))

    # I know bluealign files have a few entries entries missing,
    #   but I don't fix them in order to be consistent previous reported scores
    return alignments


def print_alignments(alignments, scores=None, src_lines=None, tgt_lines=None, ofile=sys.stdout):
    if scores is None:
        scores = [None for _ in alignments]
    for (x, y), s in zip(alignments, scores):
        if s is None:
            print('%s:%s' % (x, y), file=ofile)
        else:
            print('%s:%s:%.6f' % (x, y, s), file=ofile)
        if src_lines is not None and tgt_lines is not None:
            print(' '*40, 'SRC: ', ' '.join([src_lines[i].replace('\n', ' ').strip() for i in x]), file=ofile)
            print(' '*40, 'TGT: ', ' '.join([tgt_lines[i].replace('\n', ' ').strip() for i in y]), file=ofile)


class DeletionKnob(object):
    """
    A good deletion penalty is dependent on normalization, and probably language, domain, etc, etc
    I want a way to control deletion penalty that generalizes well...
    Sampling costs and use percentile seems to work fairly well.
    """
    def __init__(self, samp, res_min, res_max):

        self.res_min = res_min
        self.res_max = res_max

        if self.res_min >= self.res_max:
            logger.warning('res_max <= res_min, increasing it')
            self.res_max = self.res_min + 1e-4

        num_bins = 1000
        num_pts = 30

        self.hist, self.bin_edges = np.histogram(samp, bins=num_bins,
                                                 range=[self.res_min, self.res_max],
                                                 density=True)

        dx = self.bin_edges[1] - self.bin_edges[0]
        self.cdf = np.cumsum(self.hist) * dx

        interp_points = [(0, self.res_min), ]
        for knob_val in np.linspace(0, 1, num_pts - 1)[1:-1]:
            cdf_idx = np.searchsorted(self.cdf, knob_val)
            cdf_val = self.res_min + cdf_idx / float(num_bins) * (self.res_max - self.res_min)
            interp_points.append((knob_val, cdf_val))
        interp_points.append((1, self.res_max))
        self.x, self.y = zip(*interp_points)

    def percentile_frac_to_del_penalty(self, knob_val):
        del_pen = np.interp([knob_val], self.x, self.y)[0]
        return del_pen


def make_alignment_types(max_alignment_size):
    # return list of all (n,m) where n+m <= max_alignment_size
    # does not include deletions, i.e. (1, 0) or (0, 1)
    alignment_types = []
    for x in range(1, max_alignment_size):
        for y in range(1, max_alignment_size):
            if x + y <= max_alignment_size:
                alignment_types.append((x, y))
    return alignment_types


def make_one_to_many_alignment_types(max_alignment_size):
    # return list of all (1, m) where m <= max_alignment_size
    # does not include deletions, i.e. (1, 0) or (0, 1)
    alignment_types = []
    for m in range(1, max_alignment_size + 1):
        alignment_types.append((1, m))
    return alignment_types


def ab2xy_w_offset(aa, bb_idx, bb_offset):
    bb_from_side = bb_idx + bb_offset[aa]
    xx = aa - bb_from_side
    yy = bb_from_side
    return (xx, yy)


def xy2ab_w_offset(xx, yy, bb_offset):
    aa = xx + yy
    bb_from_side = yy
    bb = bb_from_side - bb_offset[aa]
    return aa, bb


def process_scores(scores, alignments):
    # floating point sometimes gives negative numbers, which is a little unnerving ...
    scores = np.clip(scores, a_min=0, a_max=None)

    for ii, (x_algn, y_algn) in enumerate(alignments):
        # deletion penalty is pretty arbitrary, just report 0
        if len(x_algn) == 0 or len(y_algn) == 0:
            scores[ii] = 0.0
        # report sores un-normalized by alignment sizes
        #    (still normalized with random vectors, though)
        else:
            scores[ii] = scores[ii] / len(x_algn) / len(y_algn)

    return scores


def sparse_traceback(a_b_csum, a_b_xp, a_b_yp, b_offset, xsize, ysize):
    alignments = []
    xx = xsize
    yy = ysize

    cum_costs = []

    while True:
        aa, bb = xy2ab_w_offset(xx, yy, b_offset)

        cum_costs.append(a_b_csum[aa, bb])

        xp = a_b_xp[aa, bb]
        yp = a_b_yp[aa, bb]

        if xx == yy == 0:
            break

        if xx < 0 or yy < 0:
            raise Exception('traceback bug')

        x_side = list(range(xx - xp, xx))
        y_side = list(range(yy - yp, yy))
        alignments.append((x_side, y_side))

        xx = xx - xp
        yy = yy - yp

    alignments.reverse()
    cum_costs.reverse()
    costs = np.array(cum_costs[1:]) - np.array(cum_costs[:-1])
    # "costs" are scaled by x_alignment_size * y_alignment_size
    #     and the cost of a deletion is del_penalty
    # "scores": 0 for deletion/insertion, 
    #    and cosine distance, *not* scaled 
    #    by len(x_alignment)*len(y_alignment)
    scores = process_scores(scores=costs, alignments=alignments)

    return alignments, scores


def dense_traceback(x_y_tb):
    xsize, ysize = x_y_tb.shape

    xx = xsize - 1
    yy = ysize - 1

    alignments = []
    while True:
        if xx == yy == 0:
            break
        bp = x_y_tb[xx, yy]
        if bp == 0:
            xp, yp = 1, 1
            alignments.append(([xx - 1], [yy - 1]))
        elif bp == 1:
            xp, yp = 0, 1
            alignments.append(([], [yy - 1]))
        elif bp == 2:
            xp, yp = 1, 0
            alignments.append(([xx - 1], []))
        else:
            raise Exception('got unknown value')

        xx = xx - xp
        yy = yy - yp

    alignments.reverse()

    return alignments


def append_slant(path, xwidth, ywidth):
    """
    Append quantized approximation to a straight line
       from current x,y to a point at (x+xwidth, y+ywidth)
    """
    NN = xwidth + ywidth
    xstart, ystart = path[-1]
    for ii in range(1, NN + 1):
        x = xstart + round(xwidth * ii / NN)
        y = ystart + round(ywidth * ii / NN)
        # In the case of ties we want them to round differently,
        #   so explicitly make sure we take a step of 1, not 0 or 2
        lastx, lasty = path[-1]
        delta = x + y - lastx - lasty
        if delta == 1:
            path.append((x, y))
        elif delta == 2:
            path.append((x - 1, y))
        elif delta == 0:
            path.append((x + 1, y))


def alignment_to_search_path(algn):
    """
    Given an alignment, make searchpath.
    Searchpath must step exactly one position in x XOR y at each time step.
    
    In the case of a block of deletions, the order found by DP is not meaningful.
    To make things consistent and to improve the probability of recovering 
       from search errors, we search an approximately straight line
       through a block of deletions. We do the same through a many-many 
       alignment, even though we currently don't refine a many-many alignment...
    """
    path = [(0, 0), ]
    xdel, ydel = 0, 0
    ydel = 0
    for x, y in algn:
        if len(x) and len(y):
            append_slant(path, xdel, ydel)
            xdel, ydel = 0, 0
            append_slant(path, len(x), len(y))
        elif len(x):
            xdel += len(x)
        elif len(y):
            ydel += len(y)

    append_slant(path, xdel, ydel)

    return path


def extend_alignments(course_alignments, size0, size1):
    """
    extend alignments to include new endpoints size0, size1
    if alignments are larger than size0/size1, raise exception
    """
    # could be a string of deletions or insertions at end, so cannot just grab last one
    xmax = 0  # maximum x value in course_alignments
    ymax = 0  # maximum y value in course_alignments
    for x, y in course_alignments:
        for xval in x:
            xmax = max(xmax, xval)
        for yval in y:
            ymax = max(ymax, yval)

    if xmax > size0 or ymax > size1:
        raise Exception('asked to extend alignments but already bigger than requested')

    # do not duplicate xmax/ymax, do include size0/size1 
    extra_x = list(range(xmax + 1, size0 + 1))
    extra_y = list(range(ymax + 1, size1 + 1))

    logger.debug('extending alignments in x by %d and y by %d', len(extra_x), len(extra_y))

    if len(extra_x) == 0:
        for yval in extra_y:
            course_alignments.append(([], [yval]))
    elif len(extra_y) == 0:
        for xval in extra_x:
            course_alignments.append(([xval], []))
    else:
        course_alignments.append((extra_x, extra_y))


def upsample_alignment(algn):
    def upsample_one_alignment(xx):
        return list(range(min(xx) * 2, (max(xx) + 1) * 2))

    new_algn = []
    for xx, yy in algn:
        if len(xx) == 0:
            for yyy in upsample_one_alignment(yy):
                new_algn.append(([], [yyy]))
        elif len(yy) == 0:
            for xxx in upsample_one_alignment(xx):
                new_algn.append(([xxx], []))
        else:
            new_algn.append((upsample_one_alignment(xx), upsample_one_alignment(yy)))
    return new_algn


def make_del_knob(e_laser,
                  f_laser,
                  e_laser_norms,
                  f_laser_norms,
                  sample_size):
    e_size = e_laser.shape[0]
    f_size = f_laser.shape[0]

    if e_size > 0 and f_size > 0 and sample_size > 0:

        if e_size * f_size < sample_size:
            # dont sample, just compute full matrix
            sample_size = e_size * f_size
            x_idxs = np.zeros(sample_size, dtype=np.int32)
            y_idxs = np.zeros(sample_size, dtype=np.int32)
            c = 0
            for ii in range(e_size):
                for jj in range(f_size):
                    x_idxs[c] = ii
                    y_idxs[c] = jj
                    c += 1
        else:
            # get random samples
            x_idxs = np.random.choice(range(e_size), size=sample_size, replace=True).astype(np.int32)
            y_idxs = np.random.choice(range(f_size), size=sample_size, replace=True).astype(np.int32)

        # output
        random_scores = np.empty(sample_size, dtype=np.float32)

        score_path(x_idxs, y_idxs,
                   e_laser_norms, f_laser_norms,
                   e_laser, f_laser,
                   random_scores, )

        min_score = 0
        max_score = max(random_scores)  # could bump this up... but its probably fine

    else:
        # Not much we can do here...
        random_scores = np.array([0.0, 0.5, 1.0])  # ???
        min_score = 0
        max_score = 1  # ????

    del_knob = DeletionKnob(random_scores, min_score, max_score)

    return del_knob


def compute_norms(vecs0, vecs1, num_samples, overlaps_to_use=None):
    # overlaps_to_use = 10  # 10 matches before

    overlaps1, size1, dim = vecs1.shape
    overlaps0, size0, dim0 = vecs0.shape
    assert (dim == dim0)

    if overlaps_to_use is not None:
        if overlaps_to_use > overlaps1:
            raise Exception('Cannot use more overlaps than provided. You may want to re-run make_verlaps.py with a larger -n value')
    else:
        overlaps_to_use = overlaps1

    samps_per_overlap = ceil(num_samples / overlaps_to_use)

    if size1 and samps_per_overlap:
        # sample other size (from all overlaps) to compre to this side
        vecs1_rand_sample = np.empty((samps_per_overlap * overlaps_to_use, dim), dtype=np.float32)
        for overlap_ii in range(overlaps_to_use):
            idxs = np.random.choice(range(size1), size=samps_per_overlap, replace=True)
            random_vecs = vecs1[overlap_ii, idxs, :]
            vecs1_rand_sample[overlap_ii * samps_per_overlap:(overlap_ii + 1) * samps_per_overlap, :] = random_vecs

        norms0 = np.empty((overlaps0, size0), dtype=np.float32)
        for overlap_ii in range(overlaps0):
            e_laser = vecs0[overlap_ii, :, :]
            sim = np.matmul(e_laser, vecs1_rand_sample.T)
            norms0[overlap_ii, :] = 1.0 - sim.mean(axis=1)

    else:  # no samples, no normalization
        norms0 = np.ones((overlaps0, size0)).astype(np.float32)

    return norms0


def downsample_vectors(vecs1):
    a, b, c = vecs1.shape
    half = np.empty((a, b // 2, c), dtype=np.float32)
    for ii in range(a):
        # average consecutive vectors
        for jj in range(0, b - b % 2, 2):
            v1 = vecs1[ii, jj, :]
            v2 = vecs1[ii, jj + 1, :]
            half[ii, jj // 2, :] = v1 + v2
        # compute mean for all vectors
        mean = np.mean(half[ii, :, :], axis=0)
        for jj in range(0, b - b % 2, 2):
            # remove mean
            half[ii, jj // 2, :] = half[ii, jj // 2, :] - mean
    # make vectors norm==1 so dot product is cosine distance
    make_norm1(half)
    return half


def vecalign(vecs0,
             vecs1,
             final_alignment_types,
             del_percentile_frac,
             width_over2,
             max_size_full_dp,
             costs_sample_size,
             num_samps_for_norm,
             norms0=None,
             norms1=None):
    if width_over2 < 3:
        logger.warning('width_over2 was set to %d, which does not make sense. increasing to 3.', width_over2)
        width_over2 = 3

    # make sure input embeddings are norm==1
    make_norm1(vecs0)
    make_norm1(vecs1)

    # save off runtime stats for summary
    runtimes = OrderedDict()

    # Determine stack depth
    s0, s1 = vecs0.shape[1], vecs1.shape[1]
    max_depth = 0
    while s0 * s1 > max_size_full_dp ** 2:
        max_depth += 1
        s0 = s0 // 2
        s1 = s1 // 2

    # init recursion stack
    # depth is 0-based (full size is 0, 1 is half, 2 is quarter, etc)
    stack = {0: {'v0': vecs0, 'v1': vecs1}}

    # downsample sentence vectors
    t0 = time()
    for depth in range(1, max_depth + 1):
        stack[depth] = {'v0': downsample_vectors(stack[depth - 1]['v0']),
                        'v1': downsample_vectors(stack[depth - 1]['v1'])}
    runtimes['Downsample embeddings'] = time() - t0

    # compute norms for all depths, add sizes, add alignment types
    t0 = time()
    for depth in stack:
        stack[depth]['size0'] = stack[depth]['v0'].shape[1]
        stack[depth]['size1'] = stack[depth]['v1'].shape[1]
        stack[depth]['alignment_types'] = final_alignment_types if depth == 0 else [(1, 1)]

        if depth == 0 and norms0 is not None:
            if norms0.shape != vecs0.shape[:2]:
                print('norms0.shape:', norms0.shape)
                print('vecs0.shape[:2]:', vecs0.shape[:2])
                raise Exception('norms0 wrong shape')
            stack[depth]['n0'] = norms0
        else:
            stack[depth]['n0'] = compute_norms(stack[depth]['v0'], stack[depth]['v1'], num_samps_for_norm)

        if depth == 0 and norms1 is not None:
            if norms1.shape != vecs1.shape[:2]:
                print('norms1.shape:', norms1.shape)
                print('vecs1.shape[:2]:', vecs1.shape[:2])
                raise Exception('norms1 wrong shape')
            stack[depth]['n1'] = norms1
        else:
            stack[depth]['n1'] = compute_norms(stack[depth]['v1'], stack[depth]['v0'], num_samps_for_norm)

    runtimes['Normalize embeddings'] = time() - t0

    # Compute deletion penalty for all depths
    t0 = time()
    for depth in stack:
        stack[depth]['del_knob'] = make_del_knob(e_laser=stack[depth]['v0'][0, :, :],
                                                 f_laser=stack[depth]['v1'][0, :, :],
                                                 e_laser_norms=stack[depth]['n0'][0, :],
                                                 f_laser_norms=stack[depth]['n1'][0, :],
                                                 sample_size=costs_sample_size)
        stack[depth]['del_penalty'] = stack[depth]['del_knob'].percentile_frac_to_del_penalty(del_percentile_frac)
        logger.debug('del_penalty at depth %d: %f', depth, stack[depth]['del_penalty'])
    runtimes['Compute deletion penalties'] = time() - t0
    tt = time() - t0
    logger.debug('%d x %d full DP make features: %.6fs (%.3e per dot product)',
                 stack[max_depth]['size0'], stack[max_depth]['size1'], tt,
                 tt / (stack[max_depth]['size0'] + 1e-6) / (stack[max_depth]['size1'] + 1e-6))
    # full DP at maximum recursion depth
    t0 = time()
    stack[max_depth]['costs_1to1'] = make_dense_costs(stack[max_depth]['v0'],
                                                      stack[max_depth]['v1'],
                                                      stack[max_depth]['n0'],
                                                      stack[max_depth]['n1'])

    runtimes['Full DP make features'] = time() - t0
    t0 = time()
    _, stack[max_depth]['x_y_tb'] = dense_dp(stack[max_depth]['costs_1to1'], stack[max_depth]['del_penalty'])
    stack[max_depth]['alignments'] = dense_traceback(stack[max_depth]['x_y_tb'])
    runtimes['Full DP'] = time() - t0

    # upsample the path up to the top resolution
    compute_costs_times = []
    dp_times = []
    upsample_depths = [0, ] if max_depth == 0 else list(reversed(range(0, max_depth)))
    for depth in upsample_depths:
        if max_depth > 0:  # upsample previoius alignment to current resolution
            course_alignments = upsample_alignment(stack[depth + 1]['alignments'])
            # features may have been truncated when downsampleing, so alignment may need extended
            extend_alignments(course_alignments, stack[depth]['size0'], stack[depth]['size1'])  # in-place
        else:  # We did a full size 1-1 search, so search same size with more alignment types
            course_alignments = stack[0]['alignments']

        # convert couse alignments to a searchpath
        stack[depth]['searchpath'] = alignment_to_search_path(course_alignments)

        # compute ccosts for sparse DP
        t0 = time()
        stack[depth]['a_b_costs'], stack[depth]['b_offset'] = make_sparse_costs(stack[depth]['v0'], stack[depth]['v1'],
                                                                                stack[depth]['n0'], stack[depth]['n1'],
                                                                                stack[depth]['searchpath'],
                                                                                stack[depth]['alignment_types'],
                                                                                width_over2)

        tt = time() - t0
        num_dot_products = len(stack[depth]['b_offset']) * len(stack[depth]['alignment_types']) * width_over2 * 2
        logger.debug('%d x %d sparse DP (%d alignment types, %d window) make features: %.6fs (%.3e per dot product)',
                     stack[max_depth]['size0'], stack[max_depth]['size1'],
                     len(stack[depth]['alignment_types']), width_over2 * 2,
                     tt, tt / (num_dot_products + 1e6))

        compute_costs_times.append(time() - t0)
        t0 = time()
        # perform sparse DP
        stack[depth]['a_b_csum'], stack[depth]['a_b_xp'], stack[depth]['a_b_yp'], \
        stack[depth]['new_b_offset'] = sparse_dp(stack[depth]['a_b_costs'], stack[depth]['b_offset'],
                                                 stack[depth]['alignment_types'], stack[depth]['del_penalty'],
                                                 stack[depth]['size0'], stack[depth]['size1'])

        # performace traceback to get alignments and alignment scores
        # for debugging, avoid overwriting stack[depth]['alignments']
        akey = 'final_alignments' if depth == 0 else 'alignments'
        stack[depth][akey], stack[depth]['alignment_scores'] = sparse_traceback(stack[depth]['a_b_csum'],
                                                                                stack[depth]['a_b_xp'],
                                                                                stack[depth]['a_b_yp'],
                                                                                stack[depth]['new_b_offset'],
                                                                                stack[depth]['size0'],
                                                                                stack[depth]['size1'])
        dp_times.append(time() - t0)

    runtimes['Upsample DP compute costs'] = sum(compute_costs_times[:-1])
    runtimes['Upsample DP'] = sum(dp_times[:-1])

    runtimes['Final DP compute costs'] = compute_costs_times[-1]
    runtimes['Final DP'] = dp_times[-1]

    # log time stats
    max_key_str_len = max([len(key) for key in runtimes])
    for key in runtimes:
        if runtimes[key] > 5e-5:
            logger.info(key + ' took ' + '.' * (max_key_str_len + 5 - len(key)) + ('%.4fs' % runtimes[key]).rjust(7))

    return stack
