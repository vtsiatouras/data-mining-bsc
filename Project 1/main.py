import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance
from wordcloud import WordCloud, STOPWORDS
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, precision_recall_fscore_support#todo
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from pylab import figure, axes, pie, title, show
from matplotlib.pyplot import savefig
from sklearn.ensemble import RandomForestClassifier


import csv
import os

categories = ["Politics", "Film", "Football", "Business", "Technology"]


def createStopwords():
    # Create stopword set
    myStopwords = STOPWORDS
    myStopwords.update(ENGLISH_STOP_WORDS)
    # Add extra stopwords
    myStopwords.update(["said", "say", "year", "will", "make", "time", "new", "says"])
    return myStopwords


def wordcloud(dataframe, length, myStopwords):
    # Create an empty array which will contain all the words per category
    word_string = ["", "", "", "", ""]
    # For every row
    for row in range(0, length):
        ind = categories.index(str(dataframe.ix[row][4]))
        # Copy the content of the articles to word_string
        word_string[ind] += dataframe.ix[row][3]
        # Copy three times the title of the articles to word_string for extra weight
        for i in range(0, 3):
            word_string[ind] += dataframe.ix[row][2]
    print(myStopwords)
    for i in range(0, 5):
        wordcloud = WordCloud(stopwords=myStopwords,
                              background_color='white',
                              width=1200,
                              height=1000
                              ).generate(word_string[i])
        plt.figure()
        plt.imshow(wordcloud)
        plt.title(categories[i])
        plt.axis('off')
        plt.draw()
    plt.show()


def clustering(dataframe, repeats, myStopwords):
    num_clusters = 5
    # define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(stop_words=myStopwords)
    # Only process the content, not the title
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataframe["Content"])
    # Convert it to an array
    tfidf_matrix_array = tfidf_matrix.toarray()
    # Run K-means with cosine distance as the metric
    kclusterer = KMeansClusterer(num_clusters, distance=cosine_distance, repeats=repeats)
    # Output to assigned_clusters
    assigned_clusters = kclusterer.cluster(tfidf_matrix_array, assign_clusters=True)
    # categories = ["Politics", "Film", "Football", "Business", "Technology"]  # todo
    # cluster_size counts how many elements each cluster contains
    cluster_size = [0, 0, 0, 0, 0]
    # Create a 5x5 array and fill it with zeros
    matrix = [[0 for x in range(5)] for y in range(5)]
    # For every category
    for category in categories:
        # For every article
        for row in range(0, len(assigned_clusters)):
            # Compare the cluster number with the category number
            if assigned_clusters[row] == categories.index(category):
                ind = categories.index(dataframe.ix[row][4])
                matrix[categories.index(category)][ind] += 1
    # Count how many elements each cluster contains
    for row in range(0, len(assigned_clusters)):
        cluster_size[assigned_clusters[row]] += 1
    for x in range(5):
        for y in range(5):
            # Calculate frequency
            matrix[x][y] /= cluster_size[x]
            # Only keep the 2 first decimal digits
            matrix[x][y] = format(matrix[x][y], '.2f')
    # Output to a .csv file
    out_file = open("output/clustering_KMeans.csv", 'w')
    wr = csv.writer(out_file, delimiter="\t")
    newCategories = categories
    newCategories.insert(0, "\t")
    wr.writerow(newCategories)
    for x in range(5):
        newMatrix = matrix[x]
        clusterName = "Cluster " + str(x + 1)
        newMatrix.insert(0, clusterName)
        wr.writerow(matrix[x])


def classification(classifier, dataframe, myStopwords):
    count_vect = CountVectorizer(stop_words=myStopwords)
    count_vect.fit(dataframe["Content"])
    kf = KFold(n_splits=10)
    fold = 0
    accuracy = 0
    precision = 0
    f_measure = 0
    recall = 0
    clf = 0
    for train_index, test_index in kf.split(dataframe["Content"]):
        X_train_counts = count_vect.transform(np.array(dataframe["Content"])[train_index])
        X_test_counts = count_vect.transform(np.array(dataframe["Content"])[test_index])
        if classifier == "svm":
            clf = svm.SVC(C=2.0, cache_size=200, gamma=0.0001, kernel='rbf', probability=True)
        elif classifier == "nb":
            clf = MultinomialNB()
        elif classifier == "rf":
            clf = RandomForestClassifier()
        else:
            print("Wrong classifier name. Accepted classifiers are: \"svm\", \"nb\", \"rf\" ")
        clf_cv = clf.fit(X_train_counts, np.array(dataframe["Category"])[train_index])
        yPred = clf_cv.predict(X_test_counts)
        f = f1_score(np.array(dataframe["Category"])[test_index], yPred, average=None)
        accuracy += accuracy_score(np.array(dataframe["Category"])[test_index], yPred)
        f_measure += sum(f) / float(len(f))
        recall += recall_score(np.array(dataframe["Category"])[test_index], yPred, average='macro')
        precision += precision_score((dataframe["Category"])[test_index], yPred, average='macro')
        fold += 1
        print("Fold " + str(fold))
        report = classification_report(np.array(dataframe["Category"])[test_index], yPred, target_names=categories)
        print(report)

    accuracy /= 10
    precision /= 10
    f_measure /= 10
    recall /= 10
    print("Precision: ", precision)
    print("Accuracy: ", accuracy)
    print("F Measure: ", f_measure)
    print("Recall: ", recall)
    # define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X = tfidf_vectorizer.fit_transform(dataframe["Content"])
    y = tfidf_vectorizer.fit_transform(dataframe["Category"])
    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2, 3, 4])
    n_classes = y.shape[1]
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(clf)
    if classifier == "svm":
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    else:
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print("AUC(micro): ", roc_auc["micro"])
    #  ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    savefig('output/roc_10fold.png')
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print("AUC(macro): ", roc_auc["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    savefig('output/roc_10fold_detailed.png')
    plt.show()






if __name__ == "__main__":
    os.makedirs(os.path.dirname("output/"), exist_ok=True)
    dataframe = pd.read_csv('./Documentation/train_set.csv', sep='\t')
    test_dataframe = pd.read_csv('./Documentation/test_set.csv', sep='\t')
    A = np.array(dataframe)
    length = A.shape[0]
    print(length)
    myStopwords = createStopwords()
    # wordcloud(dataframe, length, myStopwords)
    # clustering(dataframe, 2, myStopwords)
    classification("svm", dataframe, myStopwords)
