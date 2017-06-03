import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from operator import itemgetter
import math
import csv

attributeType = ["qualitative", "numerical", "qualitative", "qualitative", "numerical", "qualitative", "qualitative",
                 "numerical", "qualitative", "qualitative", "numerical", "qualitative", "numerical", "qualitative",
                 "qualitative", "numerical", "qualitative", "numerical", "qualitative", "qualitative"]


def performLabelEncoding(dataframe):
    le = preprocessing.LabelEncoder()
    i = 0
    # For every column
    for column in dataframe:
        # Excluding the last two
        if i == 20:
            break
        # If attribute is qualitative
        if attributeType[i] == "qualitative":
            # Label encode it
            dataframe[column] = le.fit_transform(dataframe[column])
        i += 1
    return dataframe


def createPlots(dataframe):
    good = dataframe[dataframe["Label"] == 1]
    bad = dataframe[dataframe["Label"] == 2]
    i = 0
    # For every column
    for column in dataframe:
        # Excluding the last two
        if i == 20:
            break
        # If attribute is qualitative
        if attributeType[i] == "qualitative":
            plt.title(column + " Good")
            good[column].value_counts().plot(kind='bar')
            name = "output/Attribute" + str(i + 1) + "_" + "good.png"
            savefig(name)
            plt.figure()
            plt.title(column + " Bad")
            bad[column].value_counts().plot(kind='bar')
            name = "output/Attribute" + str(i + 1) + "_" + "bad.png"
            savefig(name)
            if i < 19:
                plt.figure()
        # If attribute is numerical
        elif attributeType[i] == "numerical":
            plt.title(column + " Good")
            good.boxplot(column)
            name = "output/Attribute" + str(i + 1) + "_" + "good.png"
            savefig(name)
            plt.figure()
            plt.title(column + " Bad")
            bad.boxplot(column)
            name = "output/Attribute" + str(i + 1) + "_" + "bad.png"
            savefig(name)
            if i < 19:
                plt.figure()
        i += 1


def classifiers(dataframe):
    kf = KFold(n_splits=10)
    attributeColumns = dataframe.iloc[:, 0:20]
    svm_accuracy = 0
    # Run SVM
    print("Running SVM...(this might take some time)")
    for train_index, test_index in kf.split(dataframe):
        X_train_counts = np.array(attributeColumns)[train_index]
        X_test_counts = np.array(attributeColumns)[test_index]
        clf_cv = svm.SVC(gamma=1.0, C=1.0, kernel="rbf").fit(X_train_counts,
                                                                np.array(dataframe["Label"])[train_index])
        yPred = clf_cv.predict(X_test_counts)
        svm_accuracy += accuracy_score(np.array(dataframe["Label"])[test_index], yPred)
    svm_accuracy /= 10
    print("SVM Accuracy: ", svm_accuracy)
    rf_accuracy = 0
    # Run Random Forests
    print("Running Random Forest...")
    for train_index, test_index in kf.split(dataframe):
        X_train_counts = np.array(attributeColumns)[train_index]
        X_test_counts = np.array(attributeColumns)[test_index]
        clf_cv = RandomForestClassifier().fit(X_train_counts, np.array(dataframe["Label"])[train_index])
        yPred = clf_cv.predict(X_test_counts)
        rf_accuracy += accuracy_score(np.array(dataframe["Label"])[test_index], yPred)
    rf_accuracy /= 10
    print("Random Forest Accuracy: ", rf_accuracy)
    nb_accuracy = 0
    # Run Naive Bayes
    print("Running Naive Bayes...")
    for train_index, test_index in kf.split(dataframe):
        X_train_counts = np.array(attributeColumns)[train_index]
        X_test_counts = np.array(attributeColumns)[test_index]
        clf_cv = MultinomialNB().fit(X_train_counts, np.array(dataframe["Label"])[train_index])
        yPred = clf_cv.predict(X_test_counts)
        nb_accuracy += accuracy_score(np.array(dataframe["Label"])[test_index], yPred)
    nb_accuracy /= 10
    print("Naive Bayes Accuracy: ", nb_accuracy)
    # Output to a .csv file
    out_file = open("output/EvaluationMetric_10fold.csv", 'w')
    wr = csv.writer(out_file, delimiter="\t")
    firstLine = ["Statistic Measure", "Naive Bayes", "Random Forest", "SVM"]
    wr.writerow(firstLine)
    secondLine = ["Accuracy", nb_accuracy, rf_accuracy, svm_accuracy]
    wr.writerow(secondLine)


def predictions(dataframe, test_dataframe):
    test_dataframe = performLabelEncoding(test_dataframe)
    # Convert to numpy array only the attributes (exclude label & id)
    X_train = np.array(dataframe.iloc[:, 0:20])
    X_test = np.array(test_dataframe.iloc[:, 0:20])
    clf_cv = RandomForestClassifier().fit(X_train, np.array(dataframe["Label"]))
    predicted = clf_cv.predict(X_test)
    # Output to a .csv file
    out_file = open("output/testSet_Predictions.csv", 'w')
    wr = csv.writer(out_file, delimiter="\t")
    firstLine = ["Client_ID", "Predicted_Label"]
    # Write the first line
    wr.writerow(firstLine)
    # For every prediction
    for i in range(len(test_dataframe)):
        # If its good
        if predicted[i] == 1:
            line = [int(test_dataframe["Id"][i]), "Good"]
        # If its bad
        else:
            line = [int(test_dataframe["Id"][i]), "Bad"]
        # Write the line
        wr.writerow(line)


def entropy(dataframe, attribute):
    attributeFrequency = {}
    entropy = 0.0
    # For every row of the dataframe, count the frequencies per value
    for i in range(len(dataframe)):
        value = dataframe[attribute][i]
        if value in attributeFrequency:
            attributeFrequency[value] += 1.0
        else:
            attributeFrequency[value] = 1.0
    # For each value apply the entropy formula
    for frequency in attributeFrequency.values():
        entropy += (-frequency / len(dataframe)) * math.log(frequency / len(dataframe), 2)
    return entropy


def informationGain(dataframe, attribute):
    attributeFrequency = {}
    subsetEntropy = 0.0
    # For every row of the dataframe, count the frequencies per value
    for i in range(len(dataframe)):
        value = dataframe[attribute][i]
        if value in attributeFrequency:
            attributeFrequency[value] += 1.0
        else:
            attributeFrequency[value] = 1.0
    # For each value apply the information gain formula
    for keyValue in attributeFrequency.keys():
        weight = attributeFrequency[keyValue] / sum(attributeFrequency.values())
        dataframeSubset = pd.DataFrame()
        # Create a subset of the dataframe
        for i in range(len(dataframe)):
            value = dataframe[attribute][i]
            if value == keyValue:
                dataframeSubset.append(dataframe.iloc[i, 0:20])
        subsetEntropy += weight * entropy(dataframeSubset, attribute)
    return entropy(dataframe, attribute) - subsetEntropy


def featureSelection(dataframe, encodedDataframe):
    print("Calculating information gain for every attribute...", end=' ')
    attributeInfoGain = []
    # For every column
    i = 0
    for column in dataframe:
        # Excluding the last two
        if i == 20:
            break
        i += 1
        ig = informationGain(dataframe, column)
        attributeInfoGain.append((column, ig))
    accuracyArray = []
    attributeInfoGain.sort(key=itemgetter(1))
    print("Done!")
    for t in attributeInfoGain:
        print(t[0], "%.2f" % t[1])
    attributeColumns = encodedDataframe.iloc[:, 0:20]
    for attribute, infoGain in attributeInfoGain:
        kf = KFold(n_splits=10)
        rf_accuracy = 0
        # Run Random Forests
        print("Running Random Forest with", attributeColumns.shape[1], "features...", end=' ')
        for train_index, test_index in kf.split(attributeColumns):
            X_train_counts = np.array(attributeColumns)[train_index]
            X_test_counts = np.array(attributeColumns)[test_index]
            clf_cv = RandomForestClassifier().fit(X_train_counts, np.array(encodedDataframe["Label"])[train_index])
            yPred = clf_cv.predict(X_test_counts)
            rf_accuracy += accuracy_score(np.array(encodedDataframe["Label"])[test_index], yPred)
        rf_accuracy /= 10
        print("Accuracy: ", rf_accuracy)
        accuracyArray.append(rf_accuracy)

        attributeColumns = attributeColumns.drop(attribute, axis=1)
        print(attribute, "with information gain %.2f" % infoGain, "removed\n")
        sh = attributeColumns.shape
        if sh[1] == 0:
            break
    x_axis = [i for i in range(1, 21)]
    x_axis_reversed = [i for i in reversed(range(1, 21))]
    t = []
    for i in range(0, 19):
        t.append((x_axis, accuracyArray))
    plt.figure()
    plt.plot(x_axis, accuracyArray)
    plt.xticks(x_axis, x_axis_reversed)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    savefig("output/attribute_removal_accuracy_penalty.png")


def createBins(dataframe):
    i = 0
    # For every column
    for column in dataframe:
        # Excluding the last two
        if i == 20:
            break
        # If attribute is numerical
        if attributeType[i] == "numerical":
            # Create bins
            dataframe[column] = pd.cut(dataframe[column], bins=5, labels=False)
        i += 1
    return dataframe


if __name__ == "__main__":
    os.makedirs(os.path.dirname("output/"), exist_ok=True)
    start_time = time.time()
    dataframe = pd.read_csv('./datasets/train.tsv', sep='\t')
    test_dataframe = pd.read_csv('./datasets/test.tsv', sep='\t')
    A = np.array(dataframe)
    length = A.shape[0]
    print("Size of input: ", length)
    createPlots(dataframe)
    # Create a copy of the dataframe to prevent overwriting the original dataframe
    encoded_dataframe = dataframe.copy()
    encoded_dataframe = performLabelEncoding(encoded_dataframe)
    categorical_dataframe = dataframe.copy()
    categorical_dataframe = createBins(categorical_dataframe)

    classifiers(encoded_dataframe)
    predictions(encoded_dataframe, test_dataframe)
    featureSelection(dataframe, encoded_dataframe)
    print("Total execution time %s seconds" % (time.time() - start_time))
