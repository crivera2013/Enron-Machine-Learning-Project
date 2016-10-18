#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np 
import pandas as pd 



from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
 # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#fix the records of these individuals
data_dict['BHATNAGAR SANJAY'] = {'bonus': 'NaN',
                                 'deferral_payments': 'NaN',
                                 'deferred_income': 'NaN',
                                 'director_fees': 'NaN',
                                 'email_address': 'sanjay.bhatnagar@enron.com',
                                 'exercised_stock_options': 15456290,
                                 'expenses': 137864,
                                 'from_messages': 29,
                                 'from_poi_to_this_person': 0,
                                 'from_this_person_to_poi': 1,
                                 'loan_advances': 'NaN',
                                 'long_term_incentive': 'NaN',
                                 'other': 'NaN',
                                 'poi': False,
                                 'restricted_stock': 2604490,
                                 'restricted_stock_deferred': -2604490,
                                 'salary': 'NaN',
                                 'shared_receipt_with_poi': 463,
                                 'to_messages': 523,
                                 'total_payments': 137864,
                                 'total_stock_value': 15456290} 
data_dict['BELFER ROBERT'] = {'bonus': 'NaN',
                              'deferral_payments': 'NaN',
                              'deferred_income': -102500,
                              'director_fees': 102500,
                              'email_address': 'NaN',
                              'exercised_stock_options': 'NaN',
                              'expenses': 3285,
                              'from_messages': 'NaN',
                              'from_poi_to_this_person': 'NaN',
                              'from_this_person_to_poi': 'NaN',
                              'loan_advances': 'NaN',
                              'long_term_incentive': 'NaN',
                              'other': 'NaN',
                              'poi': False,
                              'restricted_stock': -44093,
                              'restricted_stock_deferred': 44093,
                              'salary': 'NaN',
                              'shared_receipt_with_poi': 'NaN',
                              'to_messages': 'NaN',
                              'total_payments': 3285,
                              'total_stock_value': 'NaN'}


### Task 2: Remove outliers
df = pd.DataFrame.from_dict(data_dict, orient='index')
# get rid of the "Total" employee which is not an employee but in fact a sum of all employees.
# As such it would be a huge outlier on the dataset that needs to be removed.

### add missing individuals

df = df.drop('TOTAL')
#### Employees with 'NaN' financial data values have salaries so low that it means that those 'NaN' values
# are actually just zeroes and should be replaced as such so that they don't disrupt the algorithm
df = df.replace('NaN',0)

### Task 3: Create new feature(s)
#  total_compensation should identify all highly paid enron employees who might otherwise hide their payments
# by receiving them in a myriad of ways

df['true_total_payments'] = df['salary']+df['bonus']+df['long_term_incentive']+df['deferred_income']+df['deferral_payments']+df['loan_advances']+df['other']+df['expenses']+df['director_fees']

df['true_total_stock_value'] = df['exercised_stock_options']+df['restricted_stock']+df['restricted_stock_deferred']

df['total_compensation'] = df['true_total_payments'] + df['true_total_stock_value']

# determine redundant features by finding if certain features are highly correlating
# redundant features are then removed from the features list

#### this section was found from code from the udacity formums
#### located here:  https://discussions.udacity.com/t/final-project-sharing-my-code-and-results/169287
"""
feat_mat = df.corr()
features = list(feat_mat.columns.values)
for feature in features:
	try:
		
		print feature
		print feat_mat[column].nlargest(4)
	except:
		pass
"""
features_list = ['poi','salary','bonus','long_term_incentive','deferred_income','deferral_payments',
	'loan_advances','other','expenses','director_fees','total_payments','exercised_stock_options',
	'restricted_stock','restricted_stock_deferred','total_stock_value','true_total_payments',
	'true_total_stock_value','total_compensation','from_poi_to_this_person','from_this_person_to_poi',
	'from_messages','shared_receipt_with_poi','to_messages']
#features_list = ['poi','to_messages','deferral_payments','bonus','restricted_stock','restricted_stock_deferred','expenses','from_messages','from_this_person_to_poi','director_fees','deferred_income',
#	'long_term_incentive','from_poi_to_this_person','total_compensation','true_total_stock_value','true_total_payments']

### Store to my_dataset for easy export below.

# FINAL FEATURES LIST
#features_list = ['poi','bonus','salary','true_total_stock_value','other','expenses','true_total_payments','from_messages','shared_receipt_with_poi']
features_list = ['poi','salary','bonus','total_stock_value','expenses','total_compensation']

my_dataset = df.T.to_dict()

#print df.shape
#fire =  df.loc[df['poi'] ==1]
#print fire.shape

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Before trying classifiers, enact validation strategy by using cross_validation to increase the size of the 
# dataset so that classifier results arent actually outliers of what they really are.
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)



 # Provided to give you a starting point. Try a variety of classifiers.  

print "............................................"
print "decision tree classifier"
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

from sklearn.ensemble import GradientBoostingClassifier

#clf.feature_importances_


#test_classifier(clf, my_dataset, features_list)

#print "acc = .82, precision = 0.31, recall = 0.29"

print "............................................"
print "GaussianNB"

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
#test_classifier(clf, my_dataset, features_list)

#print "acc = 0.87, precision = .5, Recall = 0.234"




print "............................................"
from sklearn.ensemble import AdaBoostClassifier


print "AdaBoostClassifer"
print " without tuning the metrics"
from sklearn.ensemble import AdaBoostClassifier


clf = AdaBoostClassifier(n_estimators = 101, learning_rate = 2, algorithm = 'SAMME.R')
clf = clf.fit(features_train, labels_train)

#print clf.feature_importances_
#test_classifier(clf, my_dataset, features_list)

print "............................................"
#with total dataset, acc = .81, precision = .309, recall = .313

### Task 5: Tune your classifier to achieve better than .3 precision and recall 

### using our testing script. Check the tester.py sript in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


## helpful code was found here: https://github.com/FCH808/FCH808.github.io/tree/master/Intro%20to%20Machine%20Learning/ud120-projects/final_project
#  and here: http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit


clf = AdaBoostClassifier()
range1 = range(1,200,20)
range2 = [1.,1.5,2.,2.5]
parameters = {'n_estimators':range1,'learning_rate':range2}
#scoring_metric = 'precision'
scoring_metric = 'recall'
#clf = GridSearchCV(clf, param_grid=parameters, cv=10,
                           #scoring=scoring_metric)

#clf.fit(features_train,labels_train)

#print(clf.best_params_)

print "AdaBoost Classifier fully tuned and validated"

clf = AdaBoostClassifier(n_estimators = 101, learning_rate = 2, algorithm = 'SAMME.R')
clf.fit(features_train, labels_train)
test_classifier(clf, my_dataset, features_list)
#print "acc = 0.848, precision = 0.407, recall = 0.3035"


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
