# Identify Fraud from Enron email

## Enron Submission Free-Response Questions

**1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]**

The goal of this project is to use the Enron Email Corpus Dataset and related financial data to build a Machine Learning Algorithm able to identify persons of interest (POI) in the Enron Scandal.

The Enron Email Corpus is one of the biggest dataset of email conversations openly available. It features conversations from executives of the Enron Corporation which were made available by the Federal Energy Regulatory Commission after the company's collapse.

Machine Learning is a good tool for such a project for at least two reasons:

- The Enron email corpus is a huge dataset: trying to process data and recognize patterns using regular techniques would be long and inefficient.
- We don't really know what we are looking for, should we suspect people with huge bonuses ? A lot of shared receipts with POI ? It's hard to know and this is the kind of problem at which ML is good.

While preparing the dataset for the project I was able to identify a few outliers using data visualization. I also decided to use a script to remove other extreme values from the data (top and below 2% percentiles).

___

**2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]**

In my POI identify I ended up using a mix of financial and email features:
- salary (Financial)
- bonus (Financial)
- deferral_payments (Financial)
- restricted_stock_deferred (Financial)
- long_term_incentive (Financial)
- from_this_person_to_poi (Email)
- shared_receipt_with_poi (Email)

I ran a simple decision tree and checked the score after adding and removing different features.

I tried feature scaling like standardization using scikit learn StandardScaler but did not gain precision, I however did gain a little precision using PCA with 4 components, which I decided to use for the final classifier.

I created two features from_poi_ratio and to_poi_ratio which are respectively the fraction of messages received from POI among all messages received and the fraction of messages send to POI among all messages send.
___

**3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]**

I ended up using a RandomForestClassifier. I picked this one after trying many other classifiers including Adaboost, SVM, KNN with different parameters. The average performance of RandomForest seemed nearly always better on precision, not on recall.

To choose which parameters to use in the classifier I used GridSearchCV with recall set as score parameter to improve the recall score.

I got and used as a result:

- 20 trees
- entropy as the information gain function
- no bootstrap
- 2 min_samples_split
- 4 max_features

___

**4. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]**

I chose precision and recall as reference metrics.
Average precision: 0.42, which means 42% of people who are predicted as POI are really POI.
Average recall: 0.22, means that among the POI, 22% are identified by our classifier.
