#!/usr/bin/env python3

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


This is a standalone example of creating a document vector from sentence vectors
following https://aclanthology.org/2020.emnlp-main.483 

"""


import numpy as np


NUM_TIME_SLOTS = 16
PERT_G = 20


def pert_pdf_normalized(xx, peak, g=PERT_G, low=-0.001, high=1.001):
    # A PERT distribution is a Beta distribution reparameterized by (low, peak, high, g).
    a, b, c = low, peak, high
    mu = (a + g * b + c) / (g + 2)
    if mu == b:  # special case where g has no effect
        a1 = a2 = 3.0
    else:
        a1 = ((mu - a) * (2 * b - a - c)) / ((b - mu) * (c - a))
        a2 = a1 * (c - mu) / (mu - a)
    t = (xx - a) / (c - a)  # map support to [0, 1]
    yy = t ** (a1 - 1) * (1 - t) ** (a2 - 1)
    return yy / np.sum(yy)


# Cache a bank of PERT distributions
_num_banks = 100
_xx = np.linspace(start=0, stop=1, num=NUM_TIME_SLOTS)
PERT_BANKS = []
for _pp in np.linspace(0, 1, num=_num_banks):
    PERT_BANKS.append(pert_pdf_normalized(_xx, peak=_pp))


np.set_printoptions(threshold=50, precision=5)
    

def build_doc_embedding(sent_vecs, sent_counts):
    # ensure sentence counts are >= 1
    sent_counts = np.clip(sent_counts, a_min=1, a_max=None)

    # scale each sent vec by 1/count
    sent_weights = 1.0/np.array(sent_counts)

    scaled_sent_vecs = np.multiply(sent_vecs.T, sent_weights).T

    # equally space sentences
    sent_centers = np.linspace(0, 1, len(scaled_sent_vecs))

    # find weighting for each sentence, for each time slot
    sentence_loc_weights = np.zeros((len(sent_centers), NUM_TIME_SLOTS))

    for sent_ii, p in enumerate(sent_centers):
        bank_idx = int(p * (len(PERT_BANKS) - 1))  # find the nearest cached pert distribution
        sentence_loc_weights[sent_ii, :] = PERT_BANKS[bank_idx]
        
    # make each chunk vector
    doc_chunk_vec = np.matmul(scaled_sent_vecs.T, sentence_loc_weights).T

    # concatenate chunk vectors into a single vector for the full document
    doc_vec = doc_chunk_vec.flatten()

    # normalize document vector
    doc_vec = doc_vec / (np.linalg.norm(doc_vec) + 1e-5)

    return doc_vec


def demo():

    rng = np.random.default_rng(42)

    # USER TODO: Replace sent_vecs with laser/LaBSE/etc embeddings of each sentence in your document,
    #    after projecting the sentence embeddings into a lower-dimensional space using something like PCA (see paper for details).
    sent_emb_size = 32  # Document embedding size will be sent_emb_size * NUM_TIME_SLOTS
    n_sents = 7
    sent_vecs = rng.random((n_sents, sent_emb_size))-0.5

    # USER TODO: Replace sent_counts with the number of times each sentence has been seen in your corpus.
    sent_counts = rng.integers(low=1, high=50, size=n_sents)

    doc_emb = build_doc_embedding(sent_vecs, sent_counts)

    print('Document Embedding:', doc_emb)
    print('Document Embedding Size:', doc_emb.shape)


if __name__ == '__main__':
    demo()
