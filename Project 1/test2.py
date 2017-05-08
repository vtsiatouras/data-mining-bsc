import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.metrics.pairwise
import scipy


def train(X_train, y_train):
    # do nothing
    return


def predict(X_train, y_train, X_test, k):
    # create list for distances and targets
    distances = []
    targets = []
    for i in range(X_train.shape[0]):
        # first we compute the euclidean distance
        # distance = sklearn.metrics.pairwise.euclidean_distances(X_test, X_train[i]) #50%
        # distance = sklearn.metrics.pairwise.pairwise_distances(X_test, X_train[i]) #53%
        distance = sklearn.metrics.pairwise.cosine_distances(X_test, X_train[i]) #96%
        distance = distance[0][0]

        # add it to list of distances
        distances.append([distance, i])
    # print(distances)
    # sort the list
    distances = sorted(distances)
    # print(distances)
    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    most_common_target = Counter(targets).most_common(1)[0][0]
    print("targets: ", targets[:k])
    print("most common: ", most_common_target)

    return most_common_target
    # return targets[0]


def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
    # train on the input data
    train(X_train, y_train)
    # loop over all observations
    # print(X_test.shape)
    # print(X_train.shape)
    # print(y_train)
    for i in range(X_test.shape[0]):
        predictions.append(predict(X_train, y_train, X_test[i], k))
    predictions = np.asarray(predictions)
    return predictions
