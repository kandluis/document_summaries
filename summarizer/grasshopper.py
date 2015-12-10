'''
Main entry point for our text summarization Grasshopper Algorithm.

This module contains functions pertaining specifically to Grasshopper.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvard.edu)
'''
from . import utils
import numpy as np


def stationary(Mat, epsilon=0.01):
    '''
    Given numpy matrix Mat, returns the vector s such that sX = s, where s is
    normalized to be a probability distribution.
    So we have sX = sI -> s(X-I) = 0, so we need to find ker(X-I).
    We use the linealg package in numpy to take care of this for us.
    '''
    values, vectors = np.linalg.eig(Mat.T)
    index = np.nonzero((abs(np.real(values) - 1.0) < epsilon) &
                       (abs(np.imag(values)) < epsilon))[0][0]
    # print values
    q = vectors[:, index]
    assert(abs((q**2).sum() - 1) < epsilon)

    return q / np.sum(q)  # convert into probability distribution


def invertMatrixTheorem(A, Ainv, indx):
    '''
    Computes the inverse of a matrix with one row and column removed using
    the matrix inversion lemma. It needs a matrix A, the inverse of A,
    and the row and column index which needs to be removed.
    '''
    n, m = A.shape
    assert(n == m)  # square matrix

    # Remove row and compute inverse
    u = np.zeros(n).reshape((n, 1))
    u[indx, 0] = -1

    v = A[indx, :].reshape((1, n))
    v[0, indx] = v[0, indx] - 1

    T1 = v.dot(Ainv).reshape((1, n))
    T2 = Ainv.dot(u).reshape((n, 1))
    T = Ainv - T2.dot(T1) / (1 + T1.dot(u))

    # Remove column and compute inverse.
    w = A[:, indx].reshape((n, 1))
    w[indx, 0] = 0

    R1 = T.dot(w)
    # R1.shape = (n, 1)
    R2 = u.T.dot(T)
    # R2.shape = (1, n)

    R = T - R1.dot(R2) / (1 + R2.dot(w))

    # Remove redundant rows
    R = np.delete(R, (indx), axis=0)
    R = np.delete(R, (indx), axis=1)

    return R


def docToMatrix(D, vec_fun=utils.frequency, sim_fun=utils.cosineSim):
    '''
    Given a document d which consists of a set of sentences, converts it into a
    |D|_s x |D|_s matrix with weights given by the specified similarity
    function. The similarity function should take as input vector
    representation as output by the vec_fun.
    '''
    # Convert sentences to vector representations!
    sentenceVectors = vec_fun(D)

    # Compute similarity
    n = len(D)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = sim_fun(sentenceVectors[i], sentenceVectors[j])
    return M


def grasshopper(W, r, lamb, k, epsilon=0.0001):
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
    q = stationary(hatP)
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
        nuvisit = np.sum(N, axis=1)
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
    return absorbed


def modifiedGrasshopper(W, r, lamb, k, epsilon=0.0001):
    '''
    Implements a modified version of the Grasshopper algorithm which
    follows the same procedure but rather than creating an absorbing state
    at each step, changes the outgoing probability of the selected sentence
    so it is distributed uniformly among all other sentences.
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

    # Select sentences based on the most probable stationary state
    selected = []
    probs = []
    while (len(selected) < k):
        q = stationary(hatP)
        # We don't want to select any previously selected sentences
        q[selected] = 0
        state = np.argmax(q)
        selected.append(state)
        probs.append(q[state])

        # Modify the matrix so the weight is distributed uniformly to all
        # other sentences.
        out = np.ones(len(P)) / (len(P) - 1)
        out[state] = 0.
        hatP[state, :] = out

    return selected


def run_abstract(D, k, algo):
    '''
    Runs the optimized grasshopper algorithm.

    This is the main API for the Grasshopper algorithms.

    '''
    D = [s for d in D for s in d]

    # Clean the document
    cleanDoc, mapping = utils.cleanDocument(D)
    WClean = docToMatrix(
        cleanDoc, sim_fun=utils.threshHoldCosineSim)

    # According to the paper, alpha = 0.25 and lambda = 0.5 where the
    # best parameters
    lamb = 0.5
    alpha = 0.25
    r = utils.decayDistribution(alpha, len(WClean))
    summary = algo(WClean, r, lamb, k)
 
    # Extract the summary
    return summary, D, mapping


def run_modified_grasshopper(D, k):
    return run_abstract(D, k, modifiedGrasshopper)


def run_grassHopper(D, k):
    return run_abstract(D, k, grasshopper)
