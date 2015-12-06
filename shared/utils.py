'''
Contains useful utility functions use throughout other areas in our code.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvard.edu)

Harvard University.
'''
from collections import Counter
import numpy as np
import re
from nltk.corpus import stopwords
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


def clean(w, stemmer):
    newW = regex.sub('', w)
    return stemmer.stem(newW)


def decayDistribution(alpha, l):
    dist = np.array([i**(-alpha) for i in range(1, l + 1)])
    return dist / dist.sum()


def expDecayDistribution(alpha, l):
    dist = np.array([alpha**i for i in range(l)])
    return dist / dist.sum()


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
            toRes[i] = resI
            resI += 1

    return res, toRes


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
