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


def test_error(C, S):
    return sum([1 for x in S if C(x[0]) == x[1]]) / float(len(S))


def sort_by_pixel(S, j, D):
    array = zip(S,D)
    array.sort(key=lambda x: x[0][0][j])  # sort by the j pixel
    return array[0], array[1]

def ERM_for_decition_stump(S, D):
    number_of_pixels = len(S[0][0])
    best_j = 1
    best_theta = 0
    best_F = float("inf")
    starting__f = sum([D[i] for i in range(len(S)) if S[i][1] == 1])
    for j in range(number_of_pixels):
        S, D = sort_by_pixel(S, j, D)
        current_f = starting__f
        if best_F > current_f:
            best_F = current_f
            best_j = j
            best_theta = S[0][0][j] - 1
        for i in range(len(S)):
            current_f -= S[i][1] * D[i]
            if best_F > current_f:
                best_F = current_f
                best_j = j
                if i + 1 == len(S):
                    best_theta = (S[i][0][j] + 1) / 2
                else:
                    best_theta = (S[i][0][j] + S[i + 1][0][j]) / 2

    print best_j
    return lambda x: 1 if (x[best_j] <= best_theta) else -1


def empirical_eroor(D, S, h):
    return sum([D[i] for i in range(len(D)) if S[i][1] != h(S[i][0])])


def ada_boost(S, T, tests):
    D = [float(1) / float(len(S)) for _ in S]
    test_errors = []
    empirical_error = []

    h_array = []
    w_array = []
    for t in range(T):
        print D
        h = ERM_for_decition_stump(S, D)
        e = empirical_eroor(D, S, h)
        print e
        w = 0.5 * numpy.log(float(1 - e) / float(e))
        print w
        D_t = [(D[i] * numpy.exp(-w * S[i][1] * h(S[i][0]))) for i in range(len(D))]
        D = D_t / sum(D_t)
        h_array.append(h)
        w_array.append(w)
        empirical_error.append(e)
        test_errors.append(
            test_error(lambda x: 1 if sum([w * h(x) for w, h in zip(w_array, h_array)]) <= 0 else -1, tests))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(empirical_error, label="empirical error")
    ax.plot(test_errors, label="test error")
    plt.xlabel("iterations", fontsize=18)
    plt.ylabel("error", fontsize=16)
    plt.legend()
    fig.savefig(os.path.join(dir_path, "errors"))
    fig.clf()


ada_boost(zip(train_data, train_labels), 10, zip(test_data, test_labels))
