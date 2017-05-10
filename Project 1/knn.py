import numpy as np
from collections import Counter
import sklearn.metrics.pairwise


def predict(X_train, y_train, X_test, k):
    distances = []
    topKcategories = []
    # For every document from the train set
    for i in range(X_train.shape[0]):
        # Calculate the cosine distance from the current test document
        distance = sklearn.metrics.pairwise.cosine_distances(X_test, X_train[i])
        distance = distance[0][0]
        # Add the distance to a list along with the documents index
        distances.append([distance, i])
    # Sort the list
    distances = sorted(distances)
    # For the top K elements of the list
    for i in range(k):
        index = distances[i][1]
        topKcategories.append(y_train[index])
    # Find the most common category
    most_common_category = Counter(topKcategories).most_common(1)[0][0]
    return most_common_category


def kNearestNeighbor(X_train, y_train, X_test, k):
    predictions = []
    # For every document from the test set
    for i in range(X_test.shape[0]):
        # Predict its category
        # and store its category on the predictions list
        predictions.append(predict(X_train, y_train, X_test[i], k))
    predictions = np.asarray(predictions)
    return predictions
