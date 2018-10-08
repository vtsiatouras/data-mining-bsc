# Data Mining

## About 

This repository contains the assignments for the course "Data Mining" of the dept. Informatics & Telecommunications of University of Athens. The 2 directories contain 2 Python applications that implement some machine learning & data mining algorithms such as Naive Bayes classification, Support Vector Machines, k-Means, k-NN etc.

## Project 1

In this project we implemented an app that classifies text with multiple ways. The algorithms are trained by a large dataset (12267 data points). Each element of this csv file has 'id':unique article number, 'title':article number, 'content', article's content, 'category': the category that the article belongs. The categories are 'Politics', 'Film', 'Football', 'Business' and 'Technology'.

Initially, all the stopwords are filtered out of all the texts of the train set. Next the programm generates wordclouds of each category.

![WordClouds](https://github.com/VangelisTsiatouras/data-mining-di/blob/master/readme_assets/wordclouds.png)

### Clustering using K-Means

Using `nltk.cluster` we implemented a function that split the train set into 5 clusters without using the varaiable 'category'. This unsupervised clustering split the dataset pretty well as its shown below:

[TABLE]

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
