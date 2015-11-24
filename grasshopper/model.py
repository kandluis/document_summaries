'''
Main entry point for our text summarization Grasshopper.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvar.edu)
'''
from shared.util import tf_idf, cosineSim, clean, stationary
import numpy as np

def docToMatrix(D, vec_fun=tf_idf, sim_fun=cosineSim):
    '''
    Given a document d which consists of a set of sentences, converts it into a
    |D|_s x |D|_s matrix with weights given by the specifiend similarity
    function. The similary function should take as input vector representations
    as output by the vec_fun.
    '''
    # Convert sentences to vector representations!
    sentenceVectors = [vec_fun(s) for s in D]

    # Compute similaliry
    n = len(D)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = sim_fun(sentenceVectors[i], sentenceVectors[j])
    return M


def grasshopper(W, r, lambda, k):
    '''
    Implements the Grasshopper algorithm described in the following paper:
    Improving Diversity in Ranking Using Absorbing Random Walks:
    http://www.david-andrzejewski.com/publications/hlt-naacl_2007.pdf

    For a Matlab Implementation, see
        http://pages.cs.wisc.edu/~jerryzhu/pub/grasshopper.m

    INPUTS:
        W: Matrix representation of an directed, weighed graph. Self-edges are
            allowed. Undirected graphs should be symmetric.
        r: A probability distribution over the sentences, with higher
            probability implying higher ranking.
        lamb: How much to weight the prior in comparison to the input graph
            weights.
        k: return the top k items as ranked by Grasshopper

    OUTPUT
        res: a list of (index, prob) of length k, where index corresponds to
        the index in the input graph, and prob is the probability assigned to
        it when selected.

    The algorithm is modified to take advantage of the fact that we only care
    about the expected number of states.
    '''
    # Let's do some basis error checking!
    n, m = W.shape
    assert(n == m)  # Sizes should be equal
    assert(np.min(W) >= 0)  # No negative edges
    assert(np.sum(r) == 1)  # r is a distribution
    assert(0 <= lamb and lamb <= 1)  # lambda is valid
    assert(0 < k and k <= n)  # Summary can't be longer than document!

    # Normalize the rows of W to create the transition matrix P'
    P = W / np.sum(W, axis=1)
    hatP = lamb * P - (1 - lamb) * r

    res = []
    for i in xrange(k):
        # Compute most likely state.
        q = stationary(hatP)
        maxIdx, maxProb = np.argmax(q), np.max(q)
        res.append((maxIdx, maxProb))

        # Transfrom state into absorbtion state
        hatP[maxIdx, :] = np.zeros(len(q))
        hatP[maxIdx, maxIdx] = 1.0

    # Return the results!
    return res