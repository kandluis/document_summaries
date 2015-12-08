'''
Main entry point for our text summarization using our baseline algorithm.

The baseline algorithm consists of assigning a weight to each sentence.

We define the weight of the

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvard.edu)
'''
from . import utils
import numpy as np


def geom(p, k):
    return (1.0 - p)**k * p


def concatDocs(D):
    sents = []
    for doc in D:
        sents += doc
    return sents


def baseline(D, k, bytes=665):
    '''
    Baseline simply takes the first bytes in the doc.
    '''
    D = "".join(["".join(s) for s in concatDocs(D)])
    return [D[:bytes]]


def geomPriorBaseline(D, k, bytes=665, p=0.02):
    D = concatDocs(D)
    sentences, mapping = utils.cleanDocument(D)
    probs = np.array([geom(p, i) for i in xrange(len(sentences))])
    probs = probs / sum(probs)
    # Keep choosing until bytes is met
    k = 1
    while True:
        summary = np.random.choice(xrange(len(sentences)), size=k,
                                   replace=False, p=probs)
        res = [D[mapping[i]] for i in sorted(summary)]
        k += 1
        if len("".join(res)) >= bytes:
            break

    return res


def modifiedGeomPriorBaseline(D, k, bytes=665, p=0.02):
    D = concatDocs(D)
    sentences, mapping = utils.cleanDocument(D)
    probs = np.array([geom(p, i) for i in xrange(1, len(sentences))])
    probs = probs / sum(probs)
    # Keep choosing until bytes is met
    k = 1
    while True:
        summary = np.random.choice(xrange(1, len(sentences)), size=k,
                                   replace=False, p=probs)
        summary = np.append(0, summary)
        res = [D[mapping[i]] for i in sorted(summary)]
        k += 1
        if len("".join(res)) >= bytes:
            break

    return res


def wordFreqBaseline(D, k, bytes=665):
    D = concatDocs(D)
    sentences, mapping = utils.cleanDocument(D)
    freqs = utils.MyCounter()
    for sentence in sentences:
        for word in sentence:
            freqs[word] += 1.0

    summary = []
    summary_words = set()
    res = []
    while len("".join(res)) < bytes:
        sent_scores = [sum([freqs[word] for word in sentence
                            if word not in summary_words]) / len(sentence) for sentence in sentences]
        selected = sent_scores.index(max(sent_scores))
        summary.append(selected)
        summary_words = summary_words.union(sentences[selected])
        res.append(D[mapping[selected]])

    # print mapping
    return [D[mapping[i]] for i in sorted(summary)]
