
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

from mcerp import PERT  # pip install mcerp  # see https://github.com/tisimst/mcerp/blob/master/mcerp/__init__.py


NUM_TIME_SLOTS = 16
PERT_G = 20


# PERT is very slow (50ms per distribution) so we cache a bank of PERT distributions
_num_banks = 100
_xx = np.linspace(start=0, stop=1, num=NUM_TIME_SLOTS)
PERT_BANKS = []
for _pp in np.linspace(0, 1, num=_num_banks):
    if _pp == 0.5:  # some special case that makes g do nothing
        _pp += 0.001
    pert = PERT(low=-0.001, peak=_pp, high=1.001, g=PERT_G, tag=None)
    _yy = pert.rv.pdf(_xx)
    _yy = _yy / sum(_yy)  # normalize
    PERT_BANKS.append(_yy)


np.set_printoptions(threshold=50, precision=5)
    

def build_doc_embedding(sent_vecs, sent_counts):
    # ensure sentence counds are >= 1
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

    # Replace sent_vecs with laser/LaBSE/etc embeddings of each sentence in your document,
    #    after projecting the sentence embeddings into a lower-dimensional space using something like PCA (see paper for details).
    sent_emb_size = 32  # Document embedding size will be sent_emb_size * NUM_TIME_SLOTS
    n_sents = 7
    sent_vecs = np.random.rand(n_sents, sent_emb_size)-0.5
    
    # Replace sent_counts with the number of times each sentence has been seen in your corpus.
    sent_counts = np.random.randint(low=1, high=50, size=n_sents)

    doc_emb = build_doc_embedding(sent_vecs, sent_counts)
    
    print('Document Embedding:', doc_emb)
    print('Document Embedding Size:', doc_emb.shape)


if __name__ == '__main__':
    demo()
