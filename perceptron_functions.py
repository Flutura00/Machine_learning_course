import numpy as np

import functools
import operator
def cv(value_list):
    return np.transpose(rv(value_list))
def rv(value_list):
    return np.array([value_list])

def mul(seq):
    return functools.reduce(operator.mul, seq, 1)


def positive(x, th, th0):
   return np.sign(th.T@x + th0)

def perceptron(data, labels, params = {}, hook = None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
    return theta, theta_0

def positive_origin(x, th):
    return np.sign(th.T @ x)

# Here goes the perceptron without offset, through the origin,do I always add a dimension on the data?yes:
def perceptron_origin(data, labels, params={}, hook=None):
    # first add a dimension of ones to the data?
    data = np.vstack([data, np.ones(len(data[0]))])
    # if T not in params, default to 100
    T = params.get('T', 100)
    (d, n) = data.shape # 2 dimensions and 5 datapoints n
    theta = np.zeros((d, 1))
    for t in range(T):
        for i in range(n):
            x = data[:, i:i + 1] # so the ith data point
            y = labels[:, i:i + 1] # same label
            if y * positive_origin(x, theta) <= 0.0: # the sign as defined before, with current x and current y.
                theta = theta + y * x
    return theta

# Averaged perceptron # so what are we averaging?
import numpy as np
# x is dimension d by 1
# th is dimension d by 1
# th0 is dimension 1 by 1
# return 1 by 1 matrix of +1, 0, -1
def positive(x, th, th0):
    return np.sign(th.T @ x + th0)

def averaged_perceptron(data, labels, params={}, hook=None):
    T = params.get('T', 100)
    (d, n) = data.shape
    theta = np.zeros((d, 1));
    theta_0 = np.zeros((1, 1))
    theta_sum = theta.copy()
    theta_0_sum = theta_0.copy()
    for t in range(T): # yep!
        for i in range(n):
            x = data[:, i:i + 1]
            y = labels[:, i:i + 1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
            theta_sum = theta_sum + theta # we average the actual slope!
            theta_0_sum = theta_0_sum + theta_0 # and the actual theta!
    theta_avg = theta_sum / (T * n) # then divide with number of steps taken in total!
    theta_0_avg = theta_0_sum / (T * n)

    if hook:
        hook((theta_avg, theta_0_avg))
    return theta_avg, theta_0_avg

def length(col_v): # Takes a d by 1 matrix;  returns a scalar of their lengths. so if point x comes from origin 0,0 then we get this length. and it all makes sense when we do the distance
    return np.sum(col_v * col_v) ** 0.5

def length_many(col_v):
    return np.sqrt(np.sum(np.square(col_v), axis=1))

def normalize(col_v):
    return col_v / length(col_v)

def margin_one(x,y,th):
    length = np.sum(th * th) ** 0.5
    print(length) # what should I expect here?
    sign_dist = y*(th.T @ x)/length
    sign_dist = np.absolute(sign_dist)
    return np.amin(sign_dist) # the minimum of


def length_many(col_v):
    return np.sqrt(np.sum(np.square(col_v), axis=1))
def signed_dist(x, th, th0):
    return ((th.T @ x) + th0) / length(th) # x is d by 1, which means 1 point! theta is also d by 1, one equation
def normalize_many(col_v):
    return np.divide(col_v, length_many(col_v))

# Takes a 2D matrix;  returns last column as 2D matrix
def index_final_col(A):
    return A[:, -1:]
#%%
# Matrix multiplication
def transform(data):
    return (np.dot(data, np.array([[1], [1]])))

def signed_dist(x, th, th0):
    return ((th.T @ x) + th0) / length(th)


def score_mat(data, labels, ths, th0s):
    pos = np.sign(np.dot(np.transpose(ths), data) + np.transpose(th0s)) # this is just a matrix thing that gives you results according to dimensionality of the data. if you give one theta - theta0 pair it gives you one score. if you give two, 2 scores...
    return np.sum(pos == labels, axis=1, keepdims=True) # the matrix of theta and theta0 scores. so for each pair it gives a score.

def best_separator(data, labels, ths, th0s):
    best_index = np.argmax(score_mat(data, labels, ths, th0s)) # this argmax makes it possible without a for loop, to go through each separator one by one, each theta, and collect the maxmimum score from the score function. The argmax goes through the matrix and extracts one row value. argmax only gives ORDER OF THE ELEMENT. then you put it down there at the return! and return your thing.
    return ths[:, best_index], th0s[:, best_index:best_index + 1]


# Evaluating a classifier, now that we selected our best classifier up there, we just look at its slope! what is actually done is, the perceptron is run with the training data, data_train, and the labels train. while the scoring function is run with the resulting theta and theta zero we get from the perceptron, and using the data_test instead of data train!
def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)
def eval_classifier(learner,data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    return score(data_test, labels_test, th, th0) / data_test.shape[1]



#%%
# Evaluating a learning algorithm with a fixed dataset
# now instead of a fata generator, we have particular data. we shuffle them(already done) and then separate them as training and test data!
import numpy as np
def xval_learning_alg(learner, data, labels, k):
    s_data = np.array_split(data, k, axis=1) # we split data into k sets of data, so we can iterate k times!
    s_labels = np.array_split(labels, k, axis=1)
    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i + 1:], axis=1) # here, we take all sets of data that we already split, but let one out, leave one out cross validation, not one data point, but a subset of datapoints.
        labels_train = np.concatenate(s_labels[:i] + s_labels[i + 1:], axis=1)
        data_test = np.array(s_data[i]) # then we use the one subset we let out in training, we use it for testing.
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                     data_test, labels_test) # the score sum is averaged in the end.
    return score_sum / k
