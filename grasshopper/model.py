'''
Main entry point for our text summarization Grasshopper.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvar.edu)
'''
from shared import utils
import numpy as np


def docToMatrix(D, vec_fun=utils.frequency, sim_fun=utils.cosineSim):
    '''
    Given a document d which consists of a set of sentences, converts it into a
    |D|_s x |D|_s matrix with weights given by the specified similarity
    function. The similarity function should take as input vector
    representation as output by the vec_fun.
    '''
    # Convert sentences to vector representations!
    sentenceVectors = [vec_fun(s) for s in D]

    # Compute similarity
    n = len(D)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = sim_fun(sentenceVectors[i], sentenceVectors[j])
    return M


def grasshopper(W, r, lamb, k, epsilon=0.0000001):
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
        res: a list of (index, steps) of length k, where index corresponds to
        the index in the input graph, and prob is the probability assigned to
        it when selected. The first element contains prob rather than steps.

    The algorithm is modified to take advantage of the fact that we only care
    about the expected number of states.
    '''
    # Let's do some basis error checking!
    n, m = W.shape
    assert n == m  # Sizes should be equal
    assert np.min(W) >= 0  # No negative edges
    assert abs(np.sum(r) - 1) < epsilon  # r is a distribution
    assert 0 <= lamb and lamb <= 1  # lambda is valid
    assert 0 < k and k <= n  # Summary can't be longer than document!

    # Normalize the rows of W to create the transition matrix P'
    P = W / np.sum(W, axis=1)
    hatP = lamb * P + (1 - lamb) * r

    assert hatP.shape == (n, m)  # Shape should not change!

    # To store results.
    absorbed = []
    nonAbsorbed = range(n)
    probs = []

    # Calculate the most probable state!
    q = utils.stationary(hatP)
    absorbed.append(np.argmax(q))
    probs.append(np.max(q))
    nonAbsorbed.remove(np.argmax(q))

    # Pick the ramaining k-1 items by picking out the most-visited node one by
    # one. once picked out, the item turns into an absorbing node.
    while (len(absorbed) < k):
        # Compute expected number fo times each node will be visited before
        # random walk is absorbed by absorbing nodes. Averaged over all start
        # nodes.
         # Compute the inverse of the fundamental matrix!
        N = np.linalg.inv(
            np.identity(len(nonAbsorbed)) - hatP[nonAbsorbed, nonAbsorbed])

        # Compute the expected visit counts
        nuvisit = np.sum(N, axis=0)
        nvisit = np.zeros(n)
        nvisit[nonAbsorbed] = nuvisit

        # Find the new absorbing state
        absorbState = np.argmax(nvisit)
        absorbVisit = max(nvisit)
        # Store the results
        absorbed.append(absorbState)
        probs.append(absorbVisit)
        nonAbsorbed.remove(absorbState)

    # Return the results!
    return zip(absorbed, probs)
