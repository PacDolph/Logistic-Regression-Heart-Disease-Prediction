import csv
import io
class DataCheck:
    def __init__(self, data_string):
        self.data_string = data_string
        self.data_df = None

    def attribute_check(self):
        csv_file = self.data_string
        with open(csv_file, 'rb') as file:
            row_num = 1
            for row in file:
                if row.count(',') != 12:
                    return "Row "+row_num+" doesn't have correct number of columns."
                row_num = row_num + 1
        return "passed"
