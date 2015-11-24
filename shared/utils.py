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


def stationary(Mat):
    '''
    Given numpy matrix Mat, returns the vector s such that sX = s, where s is
    normalized to be a probabiliy distribution.
    So we have sX = sI -> s(X-I) = 0, so we need to find ker(X-I)
    '''
    res = np.ones(len(Mat))
    return res / np.sum(res)


def clean(w):
    '''
    Given a word, removes all non-alphabetic starting/ending characters. It
    then proceeds to check if remaining characters are alphabetic. If so,
    return those characters. If not, returns None
    '''
    # TODO(nautilik)
    return w


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
    representation is parse, with the key being the term.
    '''
    v1 = MyCounter()
    # TODO(nautilik): Need to avoid stop words, non-english words, etc.
    for word in sentence:
        # Remove starting/ending punctuations/spaces
        # Make sure whatever is left is an english word
        cleanWord = clean(word)
        if cleanWord is not None:
            v1[cleanWord.lower()] += 1

    return v1