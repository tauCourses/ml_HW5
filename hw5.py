import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

dir_path = repr(os.path.dirname(os.path.realpath(sys.argv[0]))).strip("'")

from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0, 8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_size = 2000
train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
train_labels = (labels[train_idx[:train_data_size]] == pos) * 2 - 1

# validation_data_unscaled = data[train_idx[6000:], :].astype(float)
# validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_size = 2000
test_data_unscaled = data[60000 + test_idx[:test_data_size], :].astype(float)
test_labels = (labels[60000 + test_idx[:test_data_size]] == pos) * 2 - 1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
# validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)


def test_error(c, s):
    return 1 - sum([1 for x in s if c(x[0]) == x[1]]) / float(len(s))


def sort_by_pixel(s, j, d):
    if not (j in cached_permutation_sorted_by):
        S_with_indexes = zip(s, [i for i in range(len(s))])
        S_with_indexes.sort(key=lambda x: x[0][0][j])  # sort by the j pixel
        cached_permutation_sorted_by[j] = [index for _, index in S_with_indexes]
    return [s[i] for i in cached_permutation_sorted_by[j]], [d[i] for i in cached_permutation_sorted_by[j]]


cached_permutation_sorted_by = {}


def erm_for_decision_stump(origin_s, origin_d):
    number_of_pixels = len(origin_s[0][0])
    best_j = 1
    best_theta = 0
    best_F = float("inf")
    best_direction = -1
    for direction in range(2):
        direction = direction * 2 - 1
        starting__f = sum([origin_d[i] for i in range(len(origin_s)) if origin_s[i][1] == direction])
        for j in range(number_of_pixels):
            S, D = sort_by_pixel(origin_s, j, origin_d)
            current_f = starting__f
            if best_F > current_f:
                best_direction = direction
                best_F = current_f
                best_j = j
                best_theta = S[0][0][j] - 1
            for i in range(len(S)):
                current_f -= direction * S[i][1] * D[i]
                if best_F > current_f and (len(S) == i + 1 or S[i][0][j] != S[i + 1][0][j]):
                    best_direction = direction
                    best_F = current_f
                    best_j = j
                    if i + 1 == len(S):
                        best_theta = (S[i][0][j] * 2 + 1) / 2
                    else:
                        best_theta = (S[i][0][j] + S[i + 1][0][j]) / 2

    return lambda x: -best_direction if (x[best_j] > best_theta) else best_direction, best_j, best_theta, best_direction


def empirical_error(D, S, h):
    return sum([D[i] for i in range(len(D)) if S[i][1] != h(S[i][0])])


def ada_boost(S, T, tests):
    D = [float(1) / float(len(S)) for _ in S]
    test_errors = []
    empirical_errors = []
    h_array = []
    w_array = []
    for t in range(T):
        h, j, theta, direction = erm_for_decision_stump(S, D)
        e = empirical_error(D, S, h)
        w = 0.5 * numpy.log(float(1 - e) / float(e))
        D_t = [(D[i] * numpy.exp(-w * S[i][1] * h(S[i][0]))) for i in range(len(D))]
        D = D_t / sum(D_t)
        h_array.append(h)
        w_array.append(w)
        empirical_errors_ = test_error(lambda x: 1 if sum([w * h(x) for w, h in zip(w_array, h_array)]) > 0 else -1, S)
        empirical_errors.append(empirical_errors_)
        test_error_ = test_error(lambda x: 1 if sum([w * h(x) for w, h in zip(w_array, h_array)]) > 0 else -1, tests)
        test_errors.append(test_error_)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(empirical_errors, label="empirical error")
    ax.plot(test_errors, label="test error")
    plt.xlabel("iterations", fontsize=18)
    plt.ylabel("error", fontsize=16)
    plt.legend()
    fig.savefig(os.path.join(dir_path, "errors"))
    fig.clf()


ada_boost(zip(train_data, train_labels), 100, zip(test_data, test_labels))
