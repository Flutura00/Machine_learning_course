import itertools
import numpy as np
import itertools
import operator
import functools
from Machine_learning_course.plot_functions import *
from Machine_learning_course.perceptron_functions import *

def make_polynomial_feature_fun(order):
    # raw_features is d by n
    # return is k by n where k = sum_{i = 0}^order  multichoose(d, i)
    def f(raw_features):
        d, n = raw_features.shape
        result = []   # list of column vectors
        for j in range(n):
            features = []
            for o in range(order+1):
                indexTuples = \
                          itertools.combinations_with_replacement(range(d), o)
                for it in indexTuples:
                    features.append(mul(raw_features[i, j] for i in it))
            result.append(cv(features))
        return np.hstack(result)
    return f
func = make_polynomial_feature_fun(5)
print(func)


def y(x, th, th0):
   return np.dot(np.transpose(th), x) + th0
# Takes a list of numbers and returns a row vector: 1 x n

def make_polynomial_feature_fun(order):
    # raw_features is d by n
    # return is k by n where k = sum_{i = 0}^order  multichoose(d, i)
    def f(raw_features):
        d, n = raw_features.shape
        result = []   # list of column vectors
        for j in range(n):
            features = []
            for o in range(order+1):
                indexTuples = \
                          itertools.combinations_with_replacement(range(d), o)
                for it in indexTuples:
                    features.append(mul(raw_features[i, j] for i in it))
            result.append(cv(features))
        return np.hstack(result)
    return f


def one_hot_internal(x, k):
    # Make an empty column vector
    v = np.zeros((k, 1))
    # Set an entry to 1
    v[x-1, 0] = 1
    return v

def test_one_hot(sub):
    if one_hot_internal(3, 5).tolist() == sub(3, 5).tolist() and one_hot_internal(4, 7).tolist() == sub(4, 7).tolist():
        print("Passed! \n")
    else: print("Test Failed")


def test_linear_classifier_with_features(dataFun, learner, feature_fun,
                             draw = True, refresh = True, pause = True):
    raw_data, labels = dataFun()
    data = feature_fun(raw_data) if feature_fun else raw_data
    if draw:
        ax = plot_data(raw_data, labels)

        def hook(params):
            (th, th0) = params
            plot_nonlin_sep(
                lambda x1,x2: int(positive(feature_fun(cv([x1, x2])), th, th0)),
                ax = ax)
            plot_data(raw_data, labels, ax)
            plt.pause(0.005)
            print('th', th.T, 'th0', th0)
            if pause: input('press enter here to continue:')
    else:
        hook = None
    th, th0 = learner(data, labels, hook = hook)
    if hook: hook((th, th0))
    print("Final score", int(score(data, labels, th, th0)))
    print("Params", np.transpose(th), th0)

def make_polynomial_feature_fun(order):
    # raw_features is d by n
    # return is k by n where k = sum_{i = 0}^order  multichoose(d, i)
    def f(raw_features):
        d, n = raw_features.shape
        result = []   # list of column vectors
        for j in range(n):
            features = []
            for o in range(order+1):
                indexTuples = \
                          itertools.combinations_with_replacement(range(d), o)
                for it in indexTuples:
                    features.append(mul(raw_features[i, j] for i in it))
            result.append(cv(features))
        return np.hstack(result)
    return f

def test_with_features(dataFun, order = 2, draw=True, pause=True):
    test_linear_classifier_with_features(
        dataFun,                        # data
        perceptron,                     # learner
        make_polynomial_feature_fun(order), # feature maker
        draw=draw,
        pause=pause)