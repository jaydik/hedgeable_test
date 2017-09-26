import pandas as pd
import numpy as np
import os
import sys
from sklearn.externals import joblib

print('Running 03-arbitrary_dataset.py')

# Set the working directory to the script dir
os.chdir(os.path.dirname(sys.argv[0]))

print('Reading data')
X_test = pd.read_csv('../data/raw/test_data.csv', index_col=0, header=None)

print('Importing model')
clf = joblib.load('models/svm.pkl')

print('Predicting')
preds = pd.DataFrame(clf.predict(X_test))

print('Printing predictions to data/output/predictions.csv')
preds.to_csv('../data/output/predictions.csv', index=True, header=False)

print('\n\n')
