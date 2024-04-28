# data preprocessing, which includes engineer preprocessing, other preprocessing such as duplicate
# and missing values handling,
import pandas
import pandas as pd
from sklearn.model_selection import train_test_split

# this module might not generalise that well

# make sure the range is correct, data cleaning, missing value handling, duplicate handling;


class DataPreprocessing:
    # check range of a column. We need the minimum value, maximum value and index of the column.
    def __init__(self, dataframe):
        self.data: pandas.DataFrame = dataframe

    def modify_data(self, new_data):
        self.data = new_data

    def drop_features(self, list_of_features):
        old_data = self.data
        new_data = old_data.drop(labels=list_of_features, axis=1)
        self.modify_data(new_data)

    def overview(self):
        df = self.data
        print("Overview:\n",df.head())
    def range_check(self, name_of_column, min, max):
        df = self.data
        row_num = 0
        for value in df[name_of_column]:
            row_num = row_num+1
            if value > max or value < min:
                print("row "+row_num+" out of range with value "+value)

    # check type, including errors or missing values in forms other than null;
    # also need to check that numerical data follow the same format
    def type_check(self):
        df = self.data
        print(df.dtypes)

    # display null values;
    # for missing values, we need to decide how to handle them.
    # either delete column/row, or impute with some rules.
    def handle_missing_value(self):
        df = self.data
        # print(df.isnull())

    # to inspect for outliers, use box plot
    def handle_outliers(self):
        df = self.data

    def handle_duplicate(self):
        df = self.data
        return df.duplicated()

    def set_x_and_y(self, target):
        dataset = self.data
        dataset_Y = dataset[target]
        dataset_X = dataset.drop(labels=target,axis=1)
        return dataset_X, dataset_Y

    # first split the dataset, then use set_x_and_y to
    # extract the Y separately
    def training_test_split(self, train_size):
        df = self.data
        train = df.sample(frac=train_size, random_state=86)
        test = df.drop(train.index)
        return train, test
