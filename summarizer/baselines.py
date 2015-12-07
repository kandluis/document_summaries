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


def geomPriorBaseline(D, k, p=0.02):
    D = concatDocs(D)
    sentences, mapping = utils.cleanDocument(D)
    probs = np.array([geom(geom_p, i) for i in xrange(len(sentences))])
    probs = probs / sum(probs)
    summary = np.random.choice(xrange(len(sentences)), size=summary_size,
                               replace=False, p=probs)
    return [D[mapping[i]] for i in summary_sents]


def modifiedGeomPriorBaseline(D, k, p=0.02):
    D = concatDocs(D)
    sentences, mapping = utils.cleanDocument(D)
    probs = np.array([geom(geom_p, i) for i in xrange(1, len(sentences))])
    probs = probs / sum(probs)
    summary = np.random.choice(xrange(1, len(sentences)), size=summary_size,
                               replace=False, p=probs)
    summary = np.append(0, summary)
    return [D[mapping[i]] for i in summary_sents]


def wordFreqBaseline(D, k):
    D = concatDocs(D)
    sentences, mapping = utils.cleanDocument(D)
    freqs = {}
    for sentence in sentences:
        for word in sentence:
            if word not in freqs:
                freqs[word] = 1.0
            else:
                freqs[word] += 1.0

    summary = []
    summary_words = set()
    while len(summary) < summary_size:
        sent_scores = [sum([freqs[word] for word in sentence
                            if word not in summary_words]) / len(sentence) for sentence in sentences]
        selected = sent_scores.index(max(sent_scores))
        summary.append(selected)
        summary_words = summary_words.union(sentences[selected])

    return [D[mapping[i]] for i in summary_sents]
