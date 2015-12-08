'''
Main entry point for our text summarization TextRank.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvard.edu)
'''


import nltk
import numpy as np
import string
from . import utils
from itertools import combinations
from nltk.corpus import stopwords

# similarity function


def sent_sim(s1, s2):
    common = float(sum([word in s2 for word in s1]))
    if common == 0:
        return 0.0
    else:
        return common / (np.log(len(s1)) + np.log(len(s2)))


def score_sentence(G, v, d, scores):
    return (1.0 - d) + d * sum([float(scores[i]) * G[v][i] /
                                sum(G.T[i]) for i in xrange(len(G[v]))])


def concatDocs(D):
    sents = []
    for doc in D:
        sents += doc
    return sents


def textRank(D, k, bytes=665):
    D = concatDocs(D)
    stop = stopwords.words('english') + [i for i in string.punctuation]
    sentences, mapping = utils.cleanDocument(D)
    G = np.diag([1.0] * len(sentences))
    tagged_sentences = [nltk.pos_tag(sentences[i])
                        for i in xrange(len(sentences))]
    scores = [1.0] * len(sentences)
    filtered_sentences = [[word[0].lower() for word in sentence
                           if word[0] not in stop and word[1] in
                           ['NN', 'JJ', 'VB', 'NP', 'NNS', 'RB', 'VBN', 'VBG']
                           and len(word[0]) > 2] for sentence in tagged_sentences]
    # populate graph
    for i, j in combinations(range(len(filtered_sentences)), 2):
        G[i][j] = sent_sim(filtered_sentences[i], filtered_sentences[j])

    converged = False
    while not converged:
        converged = True
        for node in xrange(len(sentences)):
            old_score = scores[node]
            scores[node] = score_sentence(G, node, 0.85, scores)
            if abs(scores[node] - old_score) > 0.0001:
                converged = False

    summary_sents = sorted([index for (score, index) in sorted([(scores[i], i)
                                                                for i in xrange(len(scores))], reverse=True)[:k]])

    return [D[mapping[i]] for i in summary_sents]


def modifiedTextRank(D, k):

    return
