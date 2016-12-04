# Identify Fraud from Enron email

## Enron Submission Free-Response Questions

**1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]**

The goal of this project is to use the Enron Email Corpus Dataset and related financial data to build a Machine Learning Algorithm able to identify persons of interest (POI) in the Enron Scandal.

The Enron Email Corpus is one of the biggest dataset of email conversations openly available. It features conversations from executives of the Enron Corporation which were made available by the Federal Energy Regulatory Commission after the company's collapse.

Machine Learning is a good tool for such a project for at least two reasons:

- The Enron email corpus is a huge dataset: trying to process data and recognize patterns using regular techniques would be long and inefficient.
- We don't really know what we are looking for, should we suspect people with huge bonuses ? A lot of shared receipts with POI ? It's hard to know and this is the kind of problem at which ML is good.

The dataset we'll use is composed of 146 executives, 21 of which are persons of interest. This is quite unbalanced and we'll need to take this into account when we evaluate our model.

The dataset has a total of 3066 data points.
While preparing the dataset for the project I was able to identify a few outliers:
- "TOTAL" which may refer to the company, not an individual, had bonus and salary orders of magnitude higher than the average values for these variables.
- "LOCKHART EUGENE E" had only "NaN" values.
- "THE TRAVEL AGENCY IN THE PARK" also nearly only had "NaN" values.

I also used Pandas to compute values of total payments and stock value from the dataset and compared with the values on the FindLaw pdf file. When finding non-matching values, I updated the dataset with the values from the file.

**2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]**

In my POI identifier I ended up using a mix of financial and email features.
I imported most of the features available and created two new ones: from_poi_ratio and to_poi_ratio. They are respectively the fraction of messages received from POI among all messages received and the fraction of messages send to POI among all messages send. Including the fractions rather than the absolutes values should create more useful features because the data points are normalized.

I preprocessed all the features using StandardScaler and then used SelectKbest in combinaison with PCA within a GridSearchCV to get the best features for the final pipeline.

Getting later on the features scores from SelectKBest, it appears that to_poi_ratio, (score: 8.43), is the fourth most important feature after 'from_poi_to_this_person', 'Bonus' and 'deferred_income'. The GridSearch feature selection process selected the two main components from the PCA. 

When running the exact same model but removing the two custom features I go from ~0.34 precision, ~0.32 recall to ~0.20 precision, ~0.19 recall.

**3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]**

I ended up using a K-Nearest Neighbors Classifier after trying Adaboost, Gradient Boosting Classifier, SVC, Decision Tree, RandomForest and others.

It was the only one which gave me consistently statisfying results on both precision and recall. Other algorithms like RandomForest were much better on precision but I wasn't able to get them to perform well on recall. With other algorithms like SVC I was able to have a near perfect recall but low precision.

For each algorithm I tried different combinaisons of parameters within grid_search and alternated between f1 and recall as scoring parameters. 

We choose parameters to adapt the Algorithm to the problem at hand. Data comes in all sorts and forms and so no single Algorithm can perform well for all our needs.
When choosing the parameters, the first risk is to not tune the algorithm well enough and to have low performance.
The second risk is to overfit the training data. When this happens the model is unlikely to generalize well onto unseen data and so has a lower performance.

The answer to those problems is validation. The goal of validation is to give an estimate of the performance of our model on a independent dataset and to check and prevent overfitting. The way this works is that we split the data between training and testing sets, and we train the model with the training set and test with the testing set, that is unseen data.

In our case, we use StratifiedShuffleSplit to split the data into randomized folds (subsets of our datasets) that preserve the percentage of samples for each class.
We fit our grid search and use the resulting best parameters within the pipeline.
We use the pipeline as the final model. The scoring validation was made using a classification_report with our test data and predictions from test features. as well as using the final `tester.py` file.


**4. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]**

I chose precision and recall as reference metrics.
Average precision: 0.348, which means 34.8% of people who are predicted as POI are really POI.
Average recall: 0.326, means that among the POI, 32.6% are identified by our classifier.
