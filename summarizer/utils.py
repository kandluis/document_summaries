'''
Contains useful utility functions use throughout other areas in our code.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvard.edu)

Harvard University.
'''
from collections import Counter
from nltk.corpus import stopwords
from bs4 import BeautifulSoup as BSHTML

import numpy as np
import re
import os
import nltk.stem

regex = re.compile('[^a-zA-Z]')
myStopwords = set(stopwords.words('english'))


class MyCounter(Counter):
    """
    Adapted from Berkley Packman Framework
    A myCounter keeps track of counts for a set of keys.

    The myCounter class is an extension of the the standard Counter class.

    The myCounter also includes additional functionality useful in implementing
    the vectorization of sentences. In particular, counters can be normalized,
    multiplied, etc.
    """

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = self.items()

        def compare(x, y):
            return sign(y[1] - x[1])

        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0:
            return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return MyCounter(dict.copy(self))

    def __mul__(self, y):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

'''
Distribution Functions
'''


def decayDistribution(alpha, l):
    dist = np.array([i**(-alpha) for i in range(1, l + 1)])
    return dist / dist.sum()


def expDecayDistribution(alpha, l):
    dist = np.array([alpha**i for i in range(l)])
    return dist / dist.sum()


'''
Summary cleaning functions
'''


def cleanMultiDocModelSummaries(dir):
    '''
    Given a directory containing model summaries for DUC 2003,
    parses them into an input expected by our system.
    '''

    for name in os.listdir(dir):
        tmp = name.split('.')
        newName = 'Summary.{}.{}.txt'.format(tmp[0][1:], tmp[-1])
        newFile = os.path.join(dir, newName)
        oldFile = os.path.join(dir, name)
        os.rename(oldFile, newFile)


def cleanSingleDocModelSummaries(dir):
    '''
    Given a directory containing model summaries for DUC 2003,
    parses them into the input expected for single document summaries
    by our system.
    '''
    for name in os.listdir(dir):
        tmp = name.split('.')
        ID = tmp[0][1:]
        Person = tmp[4]
        DOCID = tmp[-2][3:] + tmp[-1]
        newName = 'Summary.{}{}.{}.txt'.format(ID, DOCID, Person)
        newFile = os.path.join(dir, newName)
        oldFile = os.path.join(dir, name)
        os.rename(oldFile, newFile)


def cleanOriginalDocs(dir):
    '''
    Given a directory containing the set of original documents from the
    DUC 2003 conference, renames and parses them into the format expected
    by our system.
    '''
    # Rename directories
    _, dirs, _ = os.walk(dir).next()
    for subdir in dirs:
        ID = subdir[1:-1]
        newDir = os.path.join(dir, ID)
        oldDir = os.path.join(dir, subdir)
        os.rename(oldDir, newDir)

        for name in os.listdir(newDir):
            # Rename the documents themselves
            tmp = name.split('.')
            fielID = tmp[0][3:] + tmp[1] + '.txt'
            newFile = os.path.join(newDir, ID)
            oldFile = os.path.join(newDir, name)
            os.rename(oldFile, newFile)

            # Extract the text!
            with open(newFile, 'r') as txt:
                HTML = BSHTML(txt.read(), 'xml')
                text = HTML.TEXT.text
                sentences = [s.replace('\n', '') for s in text.split('. ')]
                with open(os.path.join(newDir, 'Parsed.' + name), 'w') as f:
                    for s in sentences:
                        f.write("{}\n".format(s))


'''
Document Cleaning
'''


def clean(w, stemmer):
    newW = regex.sub('', w)
    return stemmer.stem(newW)


def cleanDocument(D, keepStopwords=False):
    '''
    Given a document W consisting of a list of sentences,
    returns a cleaned version of the document.

    Clean versions contain all lower case and alphabetical words,
    with stop words removed. Non-alphabetic words are ignored.

    Additionally, it returns a dictionary mapping sentence indexes
    in the clean document to sentence indexes in the original
    document.
    '''
    res = []
    resI = 0
    toRes = {}
    for i, s in enumerate(D):
        newS = [clean(w.lower(), nltk.stem.porter.PorterStemmer())
                for w in s if keepStopwords or w.lower() not in myStopwords]

        newS = filter(lambda x: x != '', newS)
        if newS != []:
            res.append(newS)
            toRes[resI] = i
            resI += 1

    return res, toRes


'''
Similarity measures
'''


def cosineSim(v1, v2):
    '''
    Note that input vectors are sparse!
    '''
    nv1, nv2 = v1.copy(), v2.copy()
    nv1.normalize()
    nv2.normalize()
    return nv1 * nv2


def threshHoldCosineSim(v1, v2, threshold=0.01):
    r = cosineSim(v1, v2)
    return 0. if r < threshold else 1.


'''
Vector representations of sentences.
'''


def absoluteWordFrequencies(D):
    # Count frequencies of terms
    c = MyCounter()
    for s in D:
        for w in s:
            c[w] += 1

    return c


def frequency(D, normalized=False):
    '''
    Given a documents, converts it to a matrix of sentence vectors.
    '''
    # Count frequencies of terms
    freqs = absoluteWordFrequencies(D)
    if normalized:
        freqs.normalize()

    # Construct vectors
    res = []
    for s in D:
        v = MyCounter()
        for w in s:
            v[w] = freqs[w]
        res.append(v)

    return res


def logFrequency(D):
    '''
    Scales logarithmically
    '''
    res = frequency(D)
    for i in range(len(res)):
        for w in res[i]:
            res[i][w] = 1 + np.log(res[i][w])

    return res


def booleanFrequencies(D):
    '''
    Vectors are binary, 1/0 if term occurs.
    '''
    res = []
    for s in D:
        v = MyCounter()
        for w in s:
            v[w] = 1
        res.append(v)

    return res


def IDF(t, corpus):
    N = float(len(corpus))
    containedIn = 1.
    for D in corpus:
        freqs = booleanFrequencies(D)
        if t in freqs:
            containedIn += 1.

    return np.log(N / containedIn)
