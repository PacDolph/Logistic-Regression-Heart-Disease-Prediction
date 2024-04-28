# This is a sample Python script.
import numpy as np
from preprocessing import DataPreprocessing as dp
from model_construction import Model
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

saved_filename = 'Logistic_Regression_Serialised.sav'
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'E:/Documents/Python Documents/HeartDisease/data/processed.cleveland.data'
    names = ('age','sex','cp','trestbps','chol','fbs',
             'restecg', 'thalach', 'exang', 'oldpeak', 'slop', 'ca'
             ,'thal','num')
    dataset = pd.read_csv(path, names=names)
    dataset = dp(dataset)

    # data preprocessing steps
    # overview
    print("Data overview:")
    dataset.overview()

    # drop unrelated columns
    dropped_cols = ['sex', 'trestbps', 'fbs',]
    dataset.drop_features(dropped_cols)

    print("Type check:\n")
    dataset.type_check()
    # find rows that have different data types
    print("Find flawed data points\n")
    sus_col = ['ca', 'thal']
    num_of_rows = dataset.data.shape[0]
    num_of_cols = dataset.data.shape[1]
    sus_row = []
    for i in sus_col:
        for j in range(num_of_rows):
            try:
                float(dataset.data[i][j])
            except ValueError:
                print(j, i, dataset.data[i][j])
                sus_row.append(j)
    print(sus_row)
    dataset.modify_data(dataset.data.drop(sus_row, axis=0))
    dataset.data.reset_index()

    print("Duplicate check:\n")
    dataset.handle_duplicate()
    print("Missing value check:\n")
    dataset.handle_missing_value()

    train, test = dataset.training_test_split(0.8)
    training_set = dp(train)
    test_set = dp(test)
    train_X, train_Y = training_set.set_x_and_y('num')
    test_X, test_Y = test_set.set_x_and_y('num')

    # build model!
    logit = Model(train_X, train_Y)
    # labels need to be normalised to fit logistic regression model
    logit.normalise(4)
    # grid settings
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_value = [100, 10, 1.0, 0.1, 0.01]
    cross_val = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # logit.lr_model_selection(solvers, penalty, c_value, cross_val)

    # logit.lr_train()
    # logit.lr_test(test_X, test_Y)
    total_X, total_Y = dataset.set_x_and_y('num')
    # saved_filename = 'Logistic_Regression_Serialised.sav'
    logit.lr_model_prepare('l2', 10, 'newton-cg', total_X, total_Y, saved_filename)

    # validation
