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
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

import csv
import os


def wordcloud(dataframe, length):
    categories = ["Politics", "Film", "Football", "Business", "Technology"]
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
    # Create stopword set
    myStopwords = STOPWORDS
    myStopwords.update(ENGLISH_STOP_WORDS)
    # Add extra stopwords
    myStopwords.update(["said", "say", "year", "will", "make", "time", "new", "says"])
    # For every category, create a wordcloud
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


def clustering(dataframe, repeats):
    num_clusters = 5
    # define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # Only process the content, not the title
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataframe["Content"])
    # Convert it to an array
    tfidf_matrix_array = tfidf_matrix.toarray()
    # Run K-means with cosine distance as the metric
    kclusterer = KMeansClusterer(num_clusters, distance=cosine_distance, repeats=repeats)
    # Output to assigned_clusters
    assigned_clusters = kclusterer.cluster(tfidf_matrix_array, assign_clusters=True)
    categories = ["Politics", "Film", "Football", "Business", "Technology"]  # todo
    # cluster_size counts how many elements each cluster contains
    cluster_size = [0, 0, 0, 0, 0]
    # Create a 5x5 array and fill it with zeros
    matrix = [[0 for x in range(5)] for y in range(5)]
    # For every catergory
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
    wr.writerow(categories)
    for x in range(5):
        wr.writerow(matrix[x])


def svmClassifier(dataframe, test_dataframe):
    categories = ["Politics", "Film", "Football", "Business", "Technology"]  # todo
    # Create stopword set
    myStopwords = STOPWORDS
    myStopwords.update(ENGLISH_STOP_WORDS)
    # Add extra stopwords
    myStopwords.update(["said", "say", "year", "will", "make", "time", "new", "says"]) #todo

    count_vect = CountVectorizer(stop_words=myStopwords)
    count_vect.fit(dataframe["Content"])

    kf = KFold(n_splits=10)
    # kf.get_n_splits(twenty_train.data)

    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    fold = 0
    for train_index, test_index in kf.split(dataframe["Content"]):
        X_train_counts = count_vect.transform(np.array(dataframe["Content"])[train_index])
        X_test_counts = count_vect.transform(np.array(dataframe["Content"])[test_index])

        clf = svm.SVC(C=2.0, cache_size=200, gamma=0.0001, kernel='rbf', probability=True)
        #clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3,
        #              gamma=0.0001, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
        #             tol=0.001, verbose=False)
        clf_cv = clf.fit(X_train_counts, np.array(dataframe["Category"])[train_index])

        # y_score = clf_cv.predict_proba(np.array(dataframe["Content"])[test_index])
        # fpr, tpr, _ = roc_curve(np.array(dataframe["Content"])[test_index], y_score[:, 1])
        # plt.plot(fpr, tpr, 'b', alpha=0.15)
        # tpr = interp(base_fpr, fpr, tpr)
        # tpr[0] = 0.0
        # tprs.append(tpr)
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        yPred = clf_cv.predict(X_test_counts)
        print("Accuracy: ", accuracy_score(np.array(dataframe["Category"])[test_index], yPred))
        print("F-measure: ", f1_score(np.array(dataframe["Category"])[test_index], yPred, average=None))

        bin_categories = label_binarize(dataframe["Category"], classes=[0, 1, 2, 3, 4])
        n_classes = bin_categories.shape[1]
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(np.array(dataframe["Category"]), yScore[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        #
        # # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(np.array(dataframe["Category"]).ravel(), yScore.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        #print("AUC: ", roc_auc_score(np.array(dataframe["Category"])[test_index], yPred))

        #false_positive_rate, true_positive_rate, thresholds = roc_curve(np.array(X_train_counts, yPred))
        #print("AUC: ", auc(false_positive_rate, true_positive_rate))
        fold += 1
        print("Fold " + str(fold))
        a = classification_report(yPred, np.array(dataframe["Category"])[test_index], target_names=categories)
        print(a)



if __name__ == "__main__":
    os.makedirs(os.path.dirname("output/"), exist_ok=True)
    dataframe = pd.read_csv('./Documentation/train_set_tiny.csv', sep='\t')
    test_dataframe = pd.read_csv('./Documentation/test_set.csv', sep='\t')
    A = np.array(dataframe)
    length = A.shape[0]
    print(length)
    # wordcloud(dataframe, length)
    # clustering(dataframe, 2)
    from sklearn.metrics import average_precision_score
    y_true = np.array(["Politics", "Business", "Politics", "Technology"])
    y_scores = np.array(["Politics", "Politics", "Football", "Technology"])
    print(accuracy_score(y_true, y_scores))

    svmClassifier(dataframe, test_dataframe)
    # for i in range(A.shape[0]):
    #     text = ""
    #     for j in range(A.shape[1]):
    #         text += str(A[i, j]) + ","
    #     print(text)

    # the histogram of the data
    # plt.hist(dataframe["RowNum"], facecolor='green')
    # plt.xlabel('Age')
    # plt.ylabel('# of Applicants')
    # plt.show()

    # cnt = Counter()
    # categories = ["Politics", "Film", "Football", "Business", "Technology"]
    # i = 0
    # for category in dataframe["Category"]:
    #     i += 1
    #     cnt[category] += 1
    # print(i)
    # print(cnt)
    # for category in categories:
    #     print(category + "  " + str(cnt[category]) + " Documents")
    #
    # my_additional_stop_words = ['Antonia', 'Nikos', 'Nikolas']
    # stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    # count_vect = CountVectorizer(stop_words=stop_words)
    # count_vect.fit(dataframe["Content"])
    # X_train_counts = count_vect.transform(dataframe["Content"])
    # print(X_train_counts.shape)
    #
    # clf = MultinomialNB().fit(X_train_counts, dataframe["Category"])
    # docs_new = ['referee is goal', 'OpenGL on the GPU is  fast']
    # X_new_counts = count_vect.transform(docs_new)
    # print(X_new_counts)
    #
    # predicted = clf.predict(X_new_counts)
    #
    # for doc, category in zip(docs_new, predicted):
    #     print('%r => %s' % (doc, category))
    #
    # document = """
    # is to offer a free vote to MPs on David Cameron’s proposals for UK to bomb Isis in Syria but will make clear that Labour party policy is to oppose airstrikes. The leader will also press Cameron to delay the vote until Labour’s concerns about the justification for the bombing are addressed, as part of a deal he has thrashed out with the deputy leader, Tom Watson, and other senior members of the shadow cabinet over the weekend. His decision averts the threat of a mass shadow cabinet walkout while making it clear that his own firmly held opposition to airstrikes is official Labour party policy, backed by the membership. It will also create a dilemma for Downing Street about whether to press ahead with the vote this week, because undecided Labour MPs are likely to be tempted to back Corbyn’s call for a longer timetable. Cameron has been expected to try for a vote on Wednesday but he has said he will not do so unless he is sure there is a clear majority in favour of strikes. It is understood has been no discussion with No 10 about Labour’s proposals to put off the vote. """
    #
    # X_new_counts = count_vect.transform([document])
    #
    # predicted = clf.predict(X_new_counts)
    #
    # print("Predicted category => " + str(predicted[0]))
    #
    # text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
    #                      ('tfidf', TfidfTransformer()),
    #                      ('clf', MultinomialNB()),
    #                      ])
    #
    # text_clf = text_clf.fit(dataframe["Content"], dataframe["Category"])
    #
    # docs_test = dataframe["Content"]
    # predicted = text_clf.predict(docs_test)
    # print(np.mean(predicted == dataframe["Category"]))
    #
    # print(classification_report(predicted, dataframe["Category"], target_names=categories))
    #
    # kf = KFold(n_splits=5)
    # # kf.get_n_splits(dataframe["Content"])
    # fold = 0
    # for train_index, test_index in kf.split(dataframe["Content"]):
    #     X_train_counts = count_vect.transform(np.array(dataframe["Content"])[train_index])
    #     X_test_counts = count_vect.transform(np.array(dataframe["Content"])[test_index])
    #
    #     clf_cv = MultinomialNB().fit(X_train_counts, np.array(dataframe["Category"])[train_index])
    #     yPred = clf_cv.predict(X_test_counts)
    #     fold += 1
    #     print("Fold " + str(fold))
    #     print(classification_report(yPred, np.array(dataframe["Category"])[test_index], target_names=categories))
    #
    #
    #
    # vectorizer = TfidfVectorizer()
    # vectorizer.fit_transform(dataframe["Content"])
    # X_train_tfidf = vectorizer.transform(dataframe["Content"])
    #
    # svd = TruncatedSVD(n_components=500)
    # X_lsi = svd.fit_transform(X_train_tfidf)
    #
    #
    # clfSVD=SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)\
    #         .fit(X_lsi,dataframe["Category"])
    # # clfSVD = GaussianNB().fit(X_lsi, twenty_train.target)
    #
    # X_test_lsi = svd.transform(vectorizer.transform(dataframe["Content"]))
    # predictedSVD = clfSVD.predict(X_test_lsi)
    #
    # print(np.mean(predictedSVD == dataframe["Category"]))
