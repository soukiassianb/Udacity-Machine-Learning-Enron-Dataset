#!/usr/bin/python
import os
import sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../tools/")
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np


# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
 # financial features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
 #
 # email features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list = ['poi',
                 'salary',
                 'bonus',
                 'deferral_payments',
                 'restricted_stock_deferred',
                 'long_term_incentive',
                 'restricted_stock',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Task 2: Remove outliers
def get_array(data_dict, value):
    return np.array([int(v[value]) for v in data_dict.values() if v[value] != 'NaN'])


def clean_outliers(data_dict, feature, percentile):
    """Remove values above and below given percentile"""
    high_threshold = np.percentile(get_array(data_dict, feature), 100 - percentile)
    low_threshold = np.percentile(get_array(data_dict, feature), percentile)
    for k, v in data_dict.items():
        if v[feature] > high_threshold or v[feature] < low_threshold:
            data_dict[k][feature] = 'NaN'
    return data_dict

for feature in features_list[1:]:
    data_dict = clean_outliers(data_dict, feature, 2)

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2, f_classif
# from sklearn import preprocessing


# print(np.array(features_train).shape)
# # features_train = SelectKBest(f_classif, k=2).fit_transform(features_train, labels_train)
# features_train = preprocessing.scale(features_train)
# features_test = preprocessing.scale(features_test)
# print(features_train)


# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=40, criterion="gini")  # 0.918918918919

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

clf.fit(features_train, labels_train)

print(clf.score(features_test, labels_test))

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
