
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
    categories = ["Politics", "Film", "Football", "Business", "Technology"] #todo
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


def svmClassifier(dataframe):
    categories = ["Politics", "Film", "Football", "Business", "Technology"]  # todo
    # Create stopword set
    myStopwords = STOPWORDS
    myStopwords.update(ENGLISH_STOP_WORDS)
    # Add extra stopwords
    myStopwords.update(["said", "say", "year", "will", "make", "time", "new", "says"])

    count_vect = CountVectorizer(stop_words=myStopwords)
    count_vect.fit(dataframe["Content"])
    X_train_counts = count_vect.transform(dataframe["Content"])
    print(X_train_counts.shape)

    clf = svm.SVC(C=2.0, cache_size=200, gamma=0.0001, kernel='rbf')
    clf.fit(X_train_counts, dataframe["Category"])

    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)
    document = """ It is not appropriate for students to wear a full veil in the classroom or for people to go through airport security with their faces covered, has said. But the deputy prime minister said he did not want to see a state ban on the wearing of religious items of clothing in particular circumstances. His comments came as a in public places. The Home Office minister Jeremy Browne called for a national debate on whether the state should step in to prevent young women having the veil imposed upon them. His intervention was sparked by a row amid public protests. Browne said he was ""instinctively uneasy"" about restricting religious freedoms, but he added there may be a case to act to protect girls who were too young to decide for themselves whether they wished to wear the veil or not. ""I am instinctively uneasy about restricting the freedom of individuals to observe the religion of their choice,"" . ""But there is genuine debate about whether girls should feel a compulsion to wear a veil when society deems children to be unable to express personal choices about other areas like buying alcohol, smoking or getting married. ""We should be very cautious about imposing religious conformity on a society which has always valued freedom of expression."" Responding to his comments, Clegg said: ""I think there is a debate going on already in households and communities up and down the country. ""My own view, very strongly held, is that we shouldn't end up like other countries issuing edicts or laws from parliament telling people what they should or should not wear. ""This is a free country and people going about their own business should be free to wear what they wish. I think it is very un-British to start telling people what pieces of clothing they should wear. ""I think there are exceptions to that as far as the full veil is concerned – security at airports, for instance. It is perfectly reasonable for us to say the full veil is clearly not appropriate there. ""And I think in the classroom, there is an issue, of course, about teachers being able to address their students in a way where they can address them face-to-face. I think it is quite difficult in the classroom to be able to do that."" A number of Conservative MPs have voiced dismay at the way the Birmingham Metropolitan College case was handled. The college had originally banned niqabs and burqas from its campuses eight years ago on the grounds that students should be easily identifiable at all times. But when a 17-year-old prospective student complained to her local newspaper that she was being discriminated against, a campaign sprang up against the ban, attracting 8,000 signatures to an online petition in just 48 hours. Following the college's decision to withdraw the rule, Downing Street said David Cameron would support a ban in his children's schools, although the decision should rest with the headteacher. However, the prime minister has been coming under growing pressure from his own MPs for a rethink on Department for Education guidelines in order to protect schools and colleges from being ""bullied"". The Tory backbencher Dr Sarah Wollaston said the veils were ""deeply offensive"" and were ""making women invisible"", and called for the niqab to be banned in schools and colleges. : ""It would be a perverse distortion of freedom if we knowingly allowed the restriction of communication in the very schools and colleges which should be equipping girls with skills for the modern world. We must not abandon our cultural belief that women should fully and equally participate in society."" Mohammed Shafiq, chief executive of the Ramadhan Foundation, said he was disgusted by Browne's calls to consider banning Muslim girls and young women from wearing the veil in public places. ""This is another example of the double standards that are applied to Muslims in our country by some politicians,"" he said. ""Whatever one's religion they should be free to practise it according to their own choices and any attempt by the government to ban Muslim women will be strongly resisted by the Muslim community. ""We take great pride in the United Kingdom's values of individual freedom and freedom of religion and any attempt by illiberal male politicians to dictate to Muslim women what they should wear will be challenged."" He added: ""We would expect these sorts of comments from the far right and authoritarian politicians and not from someone who allegedly believes in liberal values and freedom."""
    document1 = """corner penalty referee ball"""
    document2 = """ and Cornell researchers have denied that the controversial “emotion contagion” experiment was funded by the US Department of Defence (DoD). The social network told the Guardian that the study was entirely self-funded and that Facebook is categorically not a willing participant in the DoD’s , which funds research into the modelling of dynamics, risks and tipping points for large-scale civil unrest across the world, under the supervision of various US military agencies. “While Prof Hancock, like many researchers, has conducted work funded by the federal government during his career, at no time did Professor Hancock or his postdoctoral associate Jamie Guillory request or receive outside funding to support their work on this PNAS paper,” John Carberry, director of media relations at Cornell University where the academic work took place, told the Guardian. “Initial wording in an article and press releases generated by Cornell University that indicated outside funding sources was an unfortunate error missed during the editorial review process.” Several that because one of the key academic researchers on the study, , previously had ties with the Minerva Initiative that the Facebook “emotion contagion” work could have been in service of the US military. The Minerva initiative , the year of the global banking crisis, and was established to allow the department to partner with universities ""to improve DoD's basic understanding of the social, cultural, behavioural, and political forces that shape regions of the world of strategic importance to the US"". Cornell University is one of the institutions actively engaged with the Minerva Initiative and currently has a study funded through till 2017 managed by the US Air Force Office of Scientific which aims to develop an empirical model ""of the dynamics of social movement mobilisation and contagions"". The project aims to determine ""the critical mass (tipping point)"" of social contagions by studying their ""digital traces"" in the cases of ""the 2011 Egyptian revolution, the 2011 Russian Duma elections, the 2012 Nigerian fuel subsidy crisis and the 2013 Gazi park protests in Turkey.” Cornell denies that the Facebook “emotion contagion” study that was linked with this project. “Prof Hancock did submit a research grant proposal to the DoD’s Minerva program in 2008 to study language use in support of US efforts to engage social scientists on national security issues, but that proposal was not funded,” explained Carberry. “A similar research project was funded in 2009 by the National Science Foundation. Neither project involved studying emotional contagion or Facebook in any way.” “At no time prior to his work on this paper did Prof Hancock seek federal funding for this work, or any work studying emotional contagion on Facebook,” insisted Carberry. Other US universities including Washington and Maryland are involved in studies directly funded and commissioned by Minerva and the DoD, while the US military also has its own in-house research institutions conducting further studies and projects. Facebook did not want to provide a comment for publication. The row over the ethical implications of the Facebook “emotion contagion” study still rumbles on, despite for the “poorly communicated” study by the social network’s chief operating officer and number two, Sheryl Sandberg. Her apology contrasted with comments made on the same day by Facebook’s head of global policy Monika Bickert, and explained that it was necessary for innovation. She added that legislation which could limit involvednovation was “concerning”"""
    document3 = """At least 12,000 people have signed an online form asking Google to remove links about them from their search results in only 24 hours since it offered the service. activists have cautiously welcomed Google's decision to provide the form for Europeans to make the request. Chief executive Larry Page said was ""trying now to be more European, and think about [data collection] maybe more from a European context"". The firm says it has received a surge of data removal requests since a , Europe's highest court, which said search engines were subject to data protection rules and so should remove ""outdated, wrong or irrelevant"" information from their indexes unless there was a public interest in keeping it. About 40% of the requests have come from Germany and 13% from the UK, Google told the FT. The biggest proportion of removal requests, about 31%, related to frauds and scams. Google said it had set up a where people can request the removal of particular links, though it does not commit to removing them within any time limit. Decisions will be made by people, not algorithms, and information will start to be removed from mid-June. Although links to articles will be removed, the ECJ ruling says the original articles can remain. Jim Killock, executive director of the Open Rights Group, which campaigns for privacy and free speech, said: ""There are clear privacy issues from time to time about material published on the web. To actually have a mechanism to deal with this seems to be the right way to go."" The ECJ ruling did not specify how Google and other search engines should weigh up user requests, saying only that they should balance the needs of data privacy and public interest – for example relating to politicians seeking to have information about themselves removed. Page : ""I wish we'd been more involved in a real debate in Europe. That's one of the things we've taken from this, that we're starting the process of really going and talking to people."" Google will also set up a 10-strong committee of senior executives and outside experts who will try to develop a long-term approach to requests. Among those already named are Wikipedia founder Jimmy Wales, UN special rapporteur Frank La Rue, University of Leuven director Peggy Valcke, the former Spanish data protection chief Jose Luis Pinar, and Luciano Floridi, information ethics philospher at the Oxford Institute. The web form only applies to searches in . In the US, the first amendment means that freedom of speech and publication trump data protection rights, which are minimal. Dina Shiloh, of the law firm Mishcon de Reya, who has advised companies and individuals over online reputation management, called the move ""baby steps"" and said Facebook and Google's video subsidiary YouTube already had forms to request data removal. ""Essentially, this is a clash that was ripe to happen. You have Europe's privacy rights, which are very different to the understanding in the US. Privacy is not dead in the EU."" But she said the ECJ ruling would inevitably lead to court cases if Google turns down requests. ""We're entering new territory with the internet."" Google already removes about a million links per month from its index, mainly at the request of music and film copyright holders, said Professor Viktor Mayer-Schönberger, who follows internet governance and registration at the Oxford Internet Institute. ""In that way, they are already editing the web, and have always – there are links to terrorist stuff, neo-Nazi stuff, to child abuse images. Government agencies contact Google and have them take stuff down. ""The real question is, is this going to be more repressive than the other things they are doing? Is it going to negatively impact the trajectory of the internet? I don't think it will."" This article was changed on 2 June. Dina Shiloh was misquoted as saying: ""This is a clash that was right to happen"". This should have read: ""This is a clash that was ripe to happen."""
    X_new_counts = count_vect.transform([document3])
    predicted = clf.predict(X_new_counts)
    print("Predicted category => " + str(predicted[0]))






if __name__ == "__main__":
    os.makedirs(os.path.dirname("output/"), exist_ok=True)
    dataframe = pd.read_csv('./Documentation/train_set_tiny.csv', sep='\t')

    A = np.array(dataframe)
    length = A.shape[0]
    print(length)
    # wordcloud(dataframe, length)
    # clustering(dataframe, 2)
    svmClassifier(dataframe)
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
