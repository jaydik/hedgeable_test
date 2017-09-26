import pandas as pd
import numpy as np
import os
import sys
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

print('Running 02-evaluate_model.py')

# Set the working directory to the script dir
os.chdir(os.path.dirname(sys.argv[0]))

print('Reading data')
X_test = pd.read_csv('../data/processed/X_test.csv', index_col=0, header=None)
y_test = pd.read_csv('../data/processed/y_test.csv', index_col=0, header=None)

print('Loading model')
clf = joblib.load('models/svm.pkl')

print('Predicting')
preds = clf.predict(X_test)

print('Calculating metrics')
acc = round(accuracy_score(preds, y_test) * 100, 3)
pre = round(precision_score(preds, y_test) * 100, 3)
rec = round(recall_score(preds, y_test) * 100, 3)

print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(pre))
print('Recall: {}%'.format(rec))

print('\n\n')
