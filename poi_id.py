#!/usr/bin/python
from __future__ import division
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

# features selected
features_list = ['poi',
                 'salary',
                 'bonus',
                 'deferral_payments',
                 'restricted_stock_deferred',
                 'long_term_incentive',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


def get_array(data_dict, value):
    return np.array([int(v[value]) for v in data_dict.values() if v[value] != 'NaN'])


def clean_extreme_values(data_dict, feature, percentile):
    """Remove values above and below given percentile"""
    high_threshold = np.percentile(get_array(data_dict, feature), 100 - percentile)
    low_threshold = np.percentile(get_array(data_dict, feature), percentile)
    for k, v in data_dict.items():
        if v[feature] > high_threshold or v[feature] < low_threshold:
            data_dict[k][feature] = 'NaN'
    return data_dict

for feature in features_list[1:]:
    data_dict = clean_extreme_values(data_dict, feature, 2)


# Create new features
for k, v in data_dict.items():
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


def doPCA(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca

pca = doPCA(features, 4)
features = pca.transform(features)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


def findRandomForestParameters():
    """
    Use grid_search to determine parameters for our RandomForestClassifier
    """
    clf = RandomForestClassifier(random_state=42)
    params = {
        "n_estimators": range(20, 60),
        "criterion": ["gini", "entropy"],
        "max_features": range(3, 5),
        "min_samples_split": range(2, 4),
        "bootstrap": [True, False]
    }
    grid_search = GridSearchCV(clf, params, n_jobs=-1, cv=2)
    grid_search.fit(features_train, labels_train)
    print grid_search.best_params_

# Load our classifier with parameters
# determiend by findRandomForestParameters
clf = RandomForestClassifier(max_features=3,
                             min_samples_split=3,
                             bootstrap=True,
                             criterion='entropy',
                             n_estimators=28)
# clf = RandomForestClassifier(n_estimators=40)

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()



# fit our classifier
clf.fit(features_train, labels_train)

# dump classifier and data
dump_classifier_and_data(clf, my_dataset, features_list)
