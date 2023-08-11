# This code will convert the original labels provided by 3 reviewers and summarize to 1 label per type per scan

import pandas as pd
import numpy as np

FILE_NAME = 'reads.csv'
raw_data = pd.read_csv(FILE_NAME)
#print(raw_data.head())#check the format of the original data


#create a copy of only certain columns
label_data = raw_data[['name', 'Category']].copy()


def ReaderSummary(raw_data, label_data, column_name):
    # this function will summarize the reviews from 3 reviewrs using a group vote method
    # >=2 say yes -> yes
    # otherwise, no

    # input: a string specify the column name
    # the raw_data frame
    # the to-be label data frame

    # output: add the value to the corresponding data frame

    R1_column_name = 'R1:' + column_name
    R2_column_name = 'R2:' + column_name
    R3_column_name = 'R3:' + column_name

    R1 = np.array(raw_data[R1_column_name])
    R2 = np.array(raw_data[R2_column_name])
    R3 = np.array(raw_data[R3_column_name])

    output = (R1 + R2 + R3) >= 2
    label_data[column_name] = pd.Series(output, index=label_data.index)

    return label_data

#for all fo the columns run the summary function
column_name_list = ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH',
                    'BleedLocation-Left', 'BleedLocation-Right', 'ChronicBleed',
                   'Fracture', 'CalvarialFracture', 'OtherFracture', 'MassEffect',
                   'MidlineShift']

for column in column_name_list:
    label_data = ReaderSummary(raw_data, label_data, column)

label_data.head()

SAVE_FILE_NAME = 'summary_label.csv'
label_data.to_csv(SAVE_FILE_NAME)

label_data.sort_values(by=['name'])