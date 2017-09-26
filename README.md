# Dickerson Hedgeable Technical Test

If you'd like to run on an arbitrary dataset, make sure the file is named 'test_data.csv'.
Also, make sure the file is in data/raw/test_data.csv. The data needs to be the same shape as
the original dataset, except for the ad/nonad column, which needs to not be in the file.
However, there needs to be an index column to keep track of the instances.

Once you've copied the test_data.csv to the proj/data/raw folder, you can run the RUN_ALL.sh file
to train the model (using the included data) and predict for that dataset. The predictions will be saved to proj/data/output folder.
