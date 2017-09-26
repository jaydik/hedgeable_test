import pandas as pd
import numpy as np
import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

print('Running 01-svm.py')

# Set the working directory to the script dir
os.chdir(os.path.dirname(sys.argv[0]))

# Read in the data
X_train = pd.read_csv('../data/processed/X_train.csv', index_col=0, header=None)
y_train = pd.read_csv('../data/processed/y_train.csv', index_col=0, header=None)

# Just for print diagnostics
size_training_set = X_train.shape[0]
num_ads = y_train.sum()[1]
pct_ads = round((y_train.sum() / size_training_set * 100)[1], 3)

print('There are {} total training examples.'.format(size_training_set))
print('There are {} ad examples ({}%).'.format(num_ads, pct_ads))

# We want to tune the model with a range of parameters and choose the best
param_grid = {'C': [0.01, 0.5, 1, 10],
              'kernel': ['linear', 'rbf', 'sigmoid']}

# Use a support vector machine
mdl = SVC()

# Set up the GridSearch
clf = GridSearchCV(mdl, param_grid, verbose=10)

print('Training Model')
clf.fit(X_train, np.ravel(y_train))

print('Saving model to code/models/svm.pkl')
joblib.dump(clf, 'models/svm.pkl')

print('\n\n')
