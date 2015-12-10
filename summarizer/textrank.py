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
        #check for edge case where both sentences are 1 word
        if (len(s1) + len(s2)) != 2:
            return common / (np.log(len(s1)) + np.log(len(s2)))
        else:
            return 1


def score_sentence(G, v, d, scores):
    return (1.0 - d) + d * sum([float(scores[i]) * G[v][i] /
                                sum(G.T[i]) for i in xrange(len(G[v]))])


def concatDocs(D):
    sents = []
    for doc in D:
        sents += doc
    return sents


def textRank(D, k):
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
    iters = 0
    while not converged and iters < 30:
        iters += 1
        converged = True
        for node in xrange(len(sentences)):
            old_score = scores[node]
            scores[node] = score_sentence(G, node, 0.85, scores)
            if abs(scores[node] - old_score) > 0.01:
                converged = False
    print iters
    summary_sents = [index for (score, index) in sorted([(scores[i], i)
                                                                for i in xrange(len(scores))], reverse=True)[:k]]
    print summary_sents
    return summary_sents, D, mapping


def modifiedTextRank(D, k):
    D = concatDocs(D)
    stop = stopwords.words('english') + [i for i in string.punctuation]
    sentences, mapping = utils.cleanDocument(D)
    G = np.diag([1.0] * len(sentences))
    tagged_sentences = [nltk.pos_tag(sentences[i])
                        for i in xrange(len(sentences))]
    scores = [1.0] * len(sentences)
    summary = []
    summary_words = set()
    while len(summary) < min(k, len(sentences)):
        scores = [1.0] * len(sentences)

        filtered_sentences = [[word[0].lower() for word in sentence
                               if word[0] not in stop and word[1] in
                               ['NN', 'JJ', 'VB', 'NP', 'NNS', 'RB', 'VBN', 'VBG'] 
                               and word[0] not in summary_words
                               and len(word[0]) > 2] for sentence in tagged_sentences]
        # populate graph
        for i, j in combinations(range(len(filtered_sentences)), 2):
            G[i][j] = sent_sim(filtered_sentences[i], filtered_sentences[j])

        converged = False
        iters = 0
        while not converged and iters < 30:
            iters += 1
            converged = True
            for node in xrange(len(sentences)):
                old_score = scores[node]
                scores[node] = score_sentence(G, node, 0.85, scores)
                if abs(scores[node] - old_score) > 0.0001:
                    converged = False
        new_sent = sorted([(scores[i], i) for i in xrange(len(scores))], reverse=True)[0][1]
        summary.append(new_sent)
        summary_words = summary_words.union(sentences[new_sent])


    return summary, D, mapping
