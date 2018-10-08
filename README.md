# Data Mining

## About 

This repository contains the assignments for the course "Data Mining" of the dept. Informatics & Telecommunications of University of Athens. The 2 directories contain 2 Python applications that implement some machine learning & data mining algorithms such as Naive Bayes classification, Support Vector Machines, k-Means, k-NN etc.

## Project 1

In this project we implemented an app that classifies text with multiple ways. The algorithms are trained by a large dataset (12267 data points). Each element of this csv file has 'id':unique article number, 'title':article number, 'content', article's content, 'category': the category that the article belongs. The categories are 'Politics', 'Film', 'Football', 'Business' and 'Technology'.

Initially, all the stopwords are filtered out of all the texts of the train set. Next the programm generates wordclouds of each category.

![WordClouds](https://github.com/VangelisTsiatouras/data-mining-di/blob/master/readme_assets/wordclouds.png)

### Clustering using K-Means

Using `nltk.cluster` we implemented a function that split the train set into 5 clusters without using the varaiable 'category'. This unsupervised clustering split the dataset pretty well as its shown below:

|           | Politics | Film | Football | Business | Technology |
|-----------|----------|------|----------|----------|------------|
| Cluster 1 | 0.00     | 0.00 | 0.99     | 0.00     | 0.00       |
| Cluster 2 | 0.02     | 0.01 | 0.09     | 0.08     | 0.80       |
| Cluster 3 | 0.09     | 0.00 | 0.00     | 0.90     | 0.01       |
| Cluster 4 | 0.97     | 0.00 | 0.00     | 0.02     | 0.00       |
| Cluster 5 | 0.01     | 0.96 | 0.00     | 0.00     | 0.02       |

Also the metric that used for the clustering was cosine similarity.

### Classification

The classification algorithms that are used for this section are Support Vector Machines (SVM), Random Forests, Naive Bayes and K-Nearest Neighbor (simple brute force implementation). Also every algorithm we used is validated and evaluated with 10-fold Cross Validation using the metrics:

* Precision / Recall / F-Measure
* Accuracy
* AUC
* ROC plot

Some results

```
Naive Bayes
	Precision:  0.953217245448
	Accuracy:  0.957524419964
	F Measure:  0.953964727484
	Recall:  0.954935194128
	AUC(macro):  0.988626377942

Random Forests
	Precision:  0.907870031643
	Accuracy:  0.906571087455
	F Measure:  0.895458598759
	Recall:  0.889491758916
	AUC(macro):  0.984640271653

SVM
	Precision:  0.95827233923
	Accuracy:  0.960459203006
	F Measure:  0.957437015182
	Recall:  0.957005044416
	AUC(macro):  0.993629547693

K-Nearest Neighbors
	Precision:  0.943017209083
	Accuracy:  0.944972361809
	F Measure:  0.94160036097
	Recall:  0.941381308206
```

ROC curves for SVM

![ROC_SVM](https://github.com/VangelisTsiatouras/data-mining-di/blob/master/readme_assets/roc_10fold_detailed.png)

### Frameworks & Libraries used

* [Pandas](https://pandas.pydata.org/)
* [NumPy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [SciPy](https://www.scipy.org/)
* [NLTK](https://www.nltk.org/)


## Project 2

In this project we worked mostly in data exploration, data evaluation and feature selection. More specific, we had to create a classification model that evaluates creditworthiness of borrowers. The train dataset in each row contains the attributes of each borrower and a label if the borrower is 'good' or 'bad'. More info about the attributes of the dataset can be found [here]().

### Classification & Feature Selection

For the classification we implemented 3 different classifiers which are Naive Bayes, Random Forest and SVM (we used sklearn library). The results of each classifier are:

|    | Naive Bayes | Random Forests | SVM |
|----|-------------|----------------|-----|
| Accuracy | 0.6375 | 0.74625 | 0.70125 |

The best classifier is Random Forests and we chose this for the estimation of the test set.

After that we calculated the [Information Gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition) of each attribute

| Attribute | Information Gain |
|-----------|------------------|
| Attribute20 | 0.21 |
| Attribute10 | 0.52 |
| Attribute18 | 0.6 |
| Attribute14 | 0.83 |
| Attribute19 | 0.97 |
| Attribute16 | 1.12 |
| Attribute15 | 1.14 |
| Attribute17 | 1.41 |
| Attribute9 | 1.52 |
| Attribute3 | 1.71 |
| Attribute6 | 1.72 |
| Attribute8 | 1.81 |
| Attribute1 | 1.82 |
| Attribute11 | 1.85 |
| Attribute12 | 1.95 |
| Attribute7 | 2.15 |
| Attribute4 | 2.69 |
| Attribute2 | 3.73 |
| Attribute13 | 5.26 |
| Attribute5 | 9.52 |

To find the most optimal amount of Attributes, we remove attributes one by one in the order that are stored in the array above ('Attribute20' -> 'Attribute10' -> 'Attribute18' -> ...). Next is calculated the average accuracy for 10-fold cross validation using Random Forests. The plot below shows the result of this process.

![Feature Selection]()

The graph shows that with the attirbutes 20, 10, 18, 14, 19, 16, 15, 17, 9 and 3 removed, the accuracy of Random Forests can achieve ~76%. That makes the classification 2% more accurate.

### Frameworks & Libraries used

* [Pandas](https://pandas.pydata.org/)
* [NumPy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)