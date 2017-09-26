import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print('Running 00-split_data.py')

# Set the working directory to the script dir
os.chdir(os.path.dirname(sys.argv[0]))

# Import the data
data = pd.read_csv('../data/raw/data',
                   header=None,
                   low_memory=False,
                   converters={0: str.strip, 1: str.strip, 2: str.strip})

# Nulls were represented by "?" in the dataset, let's fix that
data = data.replace('?', np.nan)
print("There are {} total rows in the dataset.".format(len(data.index)))

# For now we'll eliminate the null rows, maybe we'll try to use them later...
data_nonull = data.dropna(axis=0, how='any')
print("There are {} non-null rows in the dataset.".format(len(data_nonull.index)))

labels = data_nonull[1558].replace({'nonad.': 0, 'ad.': 1})
X = data_nonull.drop(1558, axis=1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    labels,
                                                    test_size=0.1,
                                                    random_state=8675309)

# Print csvs for later
print("Printing files to proj/data/processed...")
X_train.to_csv('../data/processed/X_train.csv', index=True, header=False)
X_test.to_csv('../data/processed/X_test.csv', index=True, header=False)
y_train.to_csv('../data/processed/y_train.csv', index=True, header=False)
y_test.to_csv('../data/processed/y_test.csv', index=True, header=False)

print('\n\n')
