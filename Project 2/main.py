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
import matplotlib.mlab as mlab

attributeType = ["qualitative", "numerical", "qualitative", "qualitative", "numerical", "qualitative", "qualitative",
                 "numerical", "qualitative", "qualitative", "numerical", "qualitative", "numerical", "qualitative",
                 "qualitative", "numerical", "qualitative", "numerical", "qualitative", "qualitative"]


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
    kf = KFold(n_splits=10)
    accuracy = 0
    # Run SVM
    print("Running SVM...(this might take some time)")
    for train_index, test_index in kf.split(dataframe):
        X_train_counts = np.array(dataframe)[train_index]
        X_test_counts = np.array(dataframe)[test_index]
        clf_cv = svm.SVC(gamma=1.0, C=1.0, kernel="linear").fit(X_train_counts,
                                                                np.array(dataframe["Label"])[train_index])
        yPred = clf_cv.predict(X_test_counts)
        accuracy += accuracy_score(np.array(dataframe["Label"])[test_index], yPred)
    print("SVM Accuracy: ", accuracy / 10)
    accuracy = 0
    # Run Random Forests
    print("Running Random Forests...")
    for train_index, test_index in kf.split(dataframe):
        X_train_counts = np.array(dataframe)[train_index]
        X_test_counts = np.array(dataframe)[test_index]
        clf_cv = RandomForestClassifier().fit(X_train_counts, np.array(dataframe["Label"])[train_index])
        yPred = clf_cv.predict(X_test_counts)
        accuracy += accuracy_score(np.array(dataframe["Label"])[test_index], yPred)
    print("Random Forests Accuracy: ", accuracy / 10)
    accuracy = 0
    # Run Naive Bayes
    print("Running Naive Bayes...")
    for train_index, test_index in kf.split(dataframe):
        X_train_counts = np.array(dataframe)[train_index]
        X_test_counts = np.array(dataframe)[test_index]
        clf_cv = MultinomialNB().fit(X_train_counts, np.array(dataframe["Label"])[train_index])
        yPred = clf_cv.predict(X_test_counts)
        accuracy += accuracy_score(np.array(dataframe["Label"])[test_index], yPred)
    print("Naive Bayes Accuracy: ", accuracy / 10)


if __name__ == "__main__":
    os.makedirs(os.path.dirname("output/"), exist_ok=True)
    start_time = time.time()
    dataframe = pd.read_csv('./datasets/train.tsv', sep='\t')
    test_dataframe = pd.read_csv('./datasets/test.tsv', sep='\t')
    A = np.array(dataframe)
    print(A)
    length = A.shape[0]
    print("Size of input: ", length)
    # print(dataframe)
    # print(dataframe["Attribute1"])
    # createPlots(dataframe)
    classifiers(dataframe)
    print("Total execution time %s seconds" % (time.time() - start_time))
