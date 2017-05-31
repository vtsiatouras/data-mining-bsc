import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv
import matplotlib.mlab as mlab

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
    for column in dataframe:
        if i == 20:
            break
        if attributeType[i] == "qualitative":
            a = ['<0', '<200', '200<...<2000', '2000<']
            ax = good[column].value_counts().plot(kind='bar')
            ax.set_xticklabels(a, rotation=0)
            # Creating legend and title for the figure. Legend created with figlegend(), title with suptitle()
            suptitle = plt.suptitle('Example Plot', x=0.45, y=0.87, fontsize=18)
            # plt.xlabel('Age')
            # plt.ylabel('# of Applicants')
            name = "output/Attribute" + str(i + 1) + "_" + "good.png"
            savefig(name)
            plt.figure()
            bad[column].value_counts().plot(kind='bar')
            name = "output/Attribute" + str(i + 1) + "_" + "bad.png"
            savefig(name)
            # plt.xlabel('Age')
            # plt.ylabel('# of Applicants')
            if i < 19:
                plt.figure()
        elif attributeType[i] == "numerical":
            good.boxplot(column)
            # plt.xlabel('Age')
            # plt.ylabel('# of Applicants')
            name = "output/Attribute" + str(i + 1) + "_" + "good.png"
            savefig(name)
            plt.figure()
            bad.boxplot(column)
            # plt.xlabel('Age')
            # plt.ylabel('# of Applicants')
            name = "output/Attribute" + str(i + 1) + "_" + "bad.png"
            savefig(name)
            if i < 19:
                plt.figure()
        i += 1
        # plt.show()


def classifiers(dataframe):
    # dataframe = performLabelEncoding(dataframe)
    kf = KFold(n_splits=10)
    attributes = dataframe.iloc[:, 0:20]
    svm_accuracy = 0
    # Run SVM
    print("Running SVM...(this might take some time)")
    for train_index, test_index in kf.split(dataframe):
        X_train_counts = np.array(dataframe.iloc[:, 0:20])[train_index]
        X_test_counts = np.array(dataframe.iloc[:, 0:20])[test_index]
        clf_cv = svm.SVC(gamma=1.0, C=1.0, kernel="rbf").fit(X_train_counts,
                                                                np.array(dataframe["Label"])[train_index])
        yPred = clf_cv.predict(X_test_counts)
        svm_accuracy += accuracy_score(np.array(dataframe["Label"])[test_index], yPred)
        print(accuracy_score(np.array(dataframe["Label"])[test_index], yPred))
    svm_accuracy /= 10
    print("SVM Accuracy: ", svm_accuracy)
    rf_accuracy = 0
    # Run Random Forests
    print("Running Random Forest...")
    for train_index, test_index in kf.split(dataframe):
        X_train_counts = np.array(dataframe.iloc[:, 0:20])[train_index]
        X_test_counts = np.array(dataframe.iloc[:, 0:20])[test_index]
        clf_cv = RandomForestClassifier().fit(X_train_counts, np.array(dataframe["Label"])[train_index])
        yPred = clf_cv.predict(X_test_counts)
        rf_accuracy += accuracy_score(np.array(dataframe["Label"])[test_index], yPred)
    rf_accuracy /= 10
    print("Random Forest Accuracy: ", rf_accuracy)
    nb_accuracy = 0
    # Run Naive Bayes
    print("Running Naive Bayes...")
    for train_index, test_index in kf.split(dataframe):
        X_train_counts = np.array(dataframe.iloc[:, 0:20])[train_index]
        X_test_counts = np.array(dataframe.iloc[:, 0:20])[test_index]
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
    out_file = open("output/testSet_categories.csv", 'w')
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


if __name__ == "__main__":
    os.makedirs(os.path.dirname("output/"), exist_ok=True)
    start_time = time.time()
    dataframe = pd.read_csv('./datasets/train.tsv', sep='\t')
    test_dataframe = pd.read_csv('./datasets/test.tsv', sep='\t')
    A = np.array(dataframe)
    length = A.shape[0]
    print("Size of input: ", length)
    # createPlots(dataframe)
    # Create a copy of the dataframe to prevent overwriting the original dataframe
    encoded_dataframe = dataframe.copy()
    encoded_dataframe = performLabelEncoding(encoded_dataframe)
    # classifiers(encoded_dataframe)
    predictions(encoded_dataframe, test_dataframe)
    print("Total execution time %s seconds" % (time.time() - start_time))
