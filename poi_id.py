# coding: utf-8
#!/usr/bin/python
from __future__ import division
import os
import sys
sys.path.append("../tools/")
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
import pandas as pd

# features selected
features_list = ['poi',
                 'bonus',
                 'salary',
                 'deferral_payments',
                 'deferred_income',
                 'director_fees',
                 'exercised_stock_options',
                 'expenses',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'loan_advances',
                 'long_term_incentive',
                 'other',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'shared_receipt_with_poi',
                 'to_messages',
                 'total_payments',
                 'total_stock_value']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[314]:

# Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# We load the dataset into a dataframe
# and compute total for payments and stock values
# We'll then compare the results with the provided financial pdf file.

df = pd.DataFrame.from_dict(data_dict).T
df.replace('NaN', np.nan, inplace=True)

payment_total_fields = ['salary',
                        'bonus',
                        'long_term_incentive',
                        'deferred_income',
                        'deferral_payments',
                        'loan_advances',
                        'other',
                        'expenses',
                        'director_fees']

total_stock_value_fields = ['exercised_stock_options',
                            'restricted_stock',
                            'restricted_stock_deferred']

with pd.option_context('display.max_rows', 999, 'display.max_columns', 3):
    # Set to true to print the results
    if False:
        print(df[payment_total_fields].sum(axis=1))
    if False:
        print(df[total_stock_value_fields].sum(axis=1))

# After printing the results and comparing with the file
# BELFER ROBERT and BHATNAGAR SANJAY appeared to have incorrect total values
# so I updated the values within the dataset using the ones on the pdf file.
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 0
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['exercised_stock_options'] = 0
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = 0

data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['other'] = 0
data_dict['BHATNAGAR SANJAY']['director_fees'] = 0
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290


# Create new features
for v in data_dict.values():
    from_poi_to_this_person = v["from_poi_to_this_person"]
    to_messages = v["to_messages"]
    from_this_person_to_poi = v["from_this_person_to_poi"]
    from_messages = v["from_messages"]

    v["from_poi_ratio"] = (float(from_poi_to_this_person) / float(to_messages) if
                           to_messages not in [0, "NaN"] and from_poi_to_this_person
                           not in [0, "NaN"] else 0.0)
    v["to_poi_ratio"] = (float(from_this_person_to_poi) / float(from_messages) if
                         from_messages not in [0, "NaN"] and from_this_person_to_poi
                         not in [0, "NaN"] else 0.0)

features_list.append("from_poi_ratio")
features_list.append("to_poi_ratio")

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# We use StratifiedShuffleSplit because the dataset is small and unbalanced
cv = StratifiedShuffleSplit(labels, n_iter=100, test_size=0.75, random_state=42)
for train_index, test_index in cv:
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for ii in train_index:
        X_train.append(features[ii])
        y_train.append(labels[ii])
    for jj in test_index:
        X_test.append(features[jj])
        y_test.append(labels[jj])


# In[319]:


feature_selection = FeatureUnion([
        ('kbest', SelectKBest(f_classif)),
        ('pca', PCA())
    ])

pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', feature_selection),
        # ('clf', RandomForestClassifier())
        # ('clf', AdaBoostClassifier())
        # ('clf', GradientBoostingClassifier())
        # ('clf', SVC())
        # ('clf', DecisionTreeClassifier())
        ('clf', KNeighborsClassifier(weights='distance', algorithm='ball_tree'))
    ])

grid_search = GridSearchCV(pipeline, {
        'feature_selection__kbest__k': [2, 3, 4, 5, 7, 10],
        'feature_selection__pca__n_components': [2, 5, 10, ],

        # ADABOOST
        # 'clf__algorithm' : ['SAMME', 'SAMME.R'],
        # 'clf__n_estimators': [25, 50, 100],
        # 'clf__learning_rate': [.5, 1., 1.5],

        # GRADIENT BOOSTING
        # 'clf__loss' : ['deviance', 'exponential'],
        # 'clf__learning_rate': [0.1, 0.3, 0.5],
        # 'clf__n_estimators': [25, 50, 100],
        # 'clf__max_depth': [3, 5, 10],
        # 'clf__min_samples_split': [1, 3, 5, 10],

        # SVM
        # 'clf__kernel': [ 'sigmoid', 'poly','rbf'],
        # 'clf__C': [1, 5, 10, 20, 200, 1000],
        # 'clf__class_weight' :[None, 'balanced'],

        # RANDOM FOREST
        # 'clf__n_estimators': [25, 30, 50, 80, 100],
        # 'clf__min_samples_split': [1, 3, 5, 10],
        # 'clf__criterion': ['gini', 'entropy'],
        # 'clf__max_depth': [3, 6, 8, 11, 15, 20]

        # KNC
        'clf__n_neighbors': [2, 4, 6, 10],
        'clf__weights': ['distance', 'uniform'],
        'clf__algorithm': ['kd_tree', 'ball_tree', 'auto', 'brute'],

    }, scoring='recall')


grid_search.fit(X_train, y_train)

clf = pipeline.set_params(**grid_search.best_params_)
pipeline.fit(X_train, y_train)

print(grid_search.best_params_)
print(dir(grid_search))

report = classification_report(y_test, clf.predict(X_test))
print report

# dump classifier and dta
dump_classifier_and_data(clf, my_dataset, features_list)

# Getting the feature Scores
k = grid_search.get_params(True)['estimator__feature_selection__transformer_list'][0][1]
features_scores = zip(features_list[1:], k.scores_)
for f, s in sorted(features_scores, key=lambda x: x[1], reverse=True):
    print('%s: %s' % (f, s))
