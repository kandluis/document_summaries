'''
Contains useful utility functions use throughout other areas in our code.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvar.edu)

Harvard University.
'''
from collections import Counter
import numpy as np
import re
from nltk.corpus import stopwords

# Global variables to improve performance on common words.
englishStops = stopwords.words('english')
regex = re.compile('[^a-zA-Z]')


class MyCounter(Counter):
    """
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


def stationary(Mat, epsilon=0.0001):
    '''
    Given numpy matrix Mat, returns the vector s such that sX = s, where s is
    normalized to be a probability distribution.
    So we have sX = sI -> s(X-I) = 0, so we need to find ker(X-I).
    We use the linealg package in numpy to take care of this for us.
    '''
    values, vectors = np.linalg.eig(Mat.T)

    # Due to floating point imprecision, need to use epsilon values!
    index = np.nonzero(abs(values - 1.0) < epsilon)
    q = vectors[:, index]

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
    u = np.zeros(n)
    u[indx] = -1

    v = A[indx, :]

    T1 = v.dot(Ainv)
    T1.shape = (1, n)
    T2 = Ainv.dot(u.T)
    T2.shape = (n, 1)
    T = Ainv - T2.dot(T1) / (1 + T1.dot(u.T))

    # Remove column and compute inverse.
    w = A[:, indx]
    w.shape = n
    w[indx] = 0

    R1 = T.dot(w)
    R1.shape = (n, 1)
    R2 = u.T.dot(T)
    R2.shape = (1, n)

    R = T - R1.dot(R2) / (1 + R2.dot(w))

    # Remove redundant rows
    R = np.delete(R, (indx), axis=0)
    R = np.delete(R, (indx), axis=1)

    return R


def clean(w):
    '''
    Given a word, removes all non-alphabetic starting/ending characters.
    Returns None when the word is empty (consists of purely non-alphabetic characters).
    '''
    r = regex.sub('', w)
    return r if r != '' else None


def thresholdCosineSim(v1, v2, threshold=0.01):
    score = cosineSim(v1, v2)
    return 0 if score < threshold else score


def cosineSim(v1, v2):
    '''
    Note that input vectors are sparse!
    '''
    nv1, nv2 = v1.copy(), v2.copy()
    nv1.normalize()
    nv2.normalize()
    return nv1 * nv2


def tf_idf(sentence):
    '''
    Given a sentence, converts the sentence to a TF-IDF representations. The
    representation is sparse, with the key being the term.
    '''
    v1 = MyCounter()
    for word in sentence:
        # Remove starting/ending punctuations/spaces
        # Make sure whatever is left is an english word
        if word not in englishStops:
            cleanWord = clean(word)
            if cleanWord is not None:
                v1[cleanWord.lower()] += 1

    return v1
