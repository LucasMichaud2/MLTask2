import pandas as pd
import Data_treatment
import Task3_treatment
import models
import merge
import os

# input file path in every python file


train_features_file_path = '/Users/lucasmichaud/Desktop/MLTask2/train_features.csv'  # -> input train_featiures file path
test_features_file_path = '/Users/lucasmichaud/Desktop/MLTask2/test_features.csv'  # -> input test_features file path
train_labels_file_path = '/Users/lucasmichaud/Desktop/MLTask2/train_labels.csv'  # -> input train_label file path
working_directory = os.path.dirname(train_features_file_path)
task3_train = 'task3_train.csv'
task3_test = 'task3_test.csv'
task3_results = 'task3_results.csv'
data_train_task1_bis = 'data_train_task1_bis.csv'
data_test_task1_bis = 'data_test_task1_bis.csv'
DATA_Labels = 'DATA_Labels.csv'

df_train_set = pd.read_csv(train_features_file_path)
df_test_set = pd.read_csv(test_features_file_path)
df_train_labels = pd.read_csv(train_labels_file_path)
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']


def treatment_task3(train_set, test_set):
    df_normalized_train_set, mean_train, std_train = Task3_treatment.vital_treatment_train(train_set)
    df_normalized_test_set = Task3_treatment.vital_treatment_test(test_set, mean_train, std_train)
    df_normalized_train_set.to_csv(working_directory + task3_train)
    df_normalized_test_set.to_csv(working_directory + task3_test)
    return mean_train, std_train


mean_train1, std_train1 = treatment_task3(df_train_set, df_test_set)
normalized_train_set = pd.read_csv(working_directory + task3_train)
normalized_test_set = pd.read_csv(working_directory + task3_test)


def main_task3(train_set, test_set, train_labels, data_set, mean_test, std_test):
    train_vector = Task3_treatment.prepare_regression(train_set)
    test_vector = Task3_treatment.prepare_regression(test_set)
    y_train = Task3_treatment.y_treatment(data_set, train_labels)
    task3 = Task3_treatment.vital_nn(train_vector, y_train, test_vector)
    mean_test.rename(index={'RRate': 'LABEL_RRate', 'ABPm': 'LABEL_ABPm', 'SpO2': 'LABEL_SpO2',
                            'Heartrate': 'LABEL_Heartrate'}, inplace=True)
    std_test.rename(index={'RRate': 'LABEL_RRate', 'ABPm': 'LABEL_ABPm', 'SpO2': 'LABEL_SpO2',
                           'Heartrate': 'LABEL_Heartrate'}, inplace=True)
    task3 = (task3 * std_test) + mean_test

    task3.to_csv(working_directory + task3_results)
    return print('task3 done')


def main_task1():
    Data_treatment.main2(df_train_set, df_test_set, df_train_labels)
    dataset = pd.read_csv(working_directory + data_train_task1_bis)
    dataset.rename(columns={'Unnamed: 0': 'feat'}, inplace=True)
    dataset = dataset.drop(['feat'], axis=1)
    data_labels = pd.read_csv(working_directory + DATA_Labels)
    data_labels.rename(columns={'Unnamed: 0': 'feat'}, inplace=True)
    data_labels = data_labels.drop(['feat'], axis=1)
    test_set = pd.read_csv(working_directory + data_test_task1_bis)
    test_set.rename(columns={'Unnamed: 0': 'feat'}, inplace=True)
    test_set = test_set.drop(['feat'], axis=1)
    x_train = models.x_train_gen(dataset)
    x_test = models.x_train_gen(test_set)
    y_train = models.y_train_gen(data_labels)
    models.nn_network_test(x_train, y_train, x_test)
    return 0


def main_task2():
    Data_treatment.main2(df_train_set, df_test_set, df_train_labels)
    dataset = pd.read_csv(working_directory + data_train_task1_bis)
    dataset.rename(columns={'Unnamed: 0': 'feat'}, inplace=True)
    dataset = dataset.drop(['feat'], axis=1)
    test_set = pd.read_csv(working_directory + data_test_task1_bis)
    test_set.rename(columns={'Unnamed: 0': 'feat'}, inplace=True)
    test_set = test_set.drop(['feat'], axis=1)
    all_labels = pd.read_csv(train_labels_file_path)
    x_train = models.x_train_gen(dataset)
    x_test = models.x_train_gen(test_set)
    Y_sepsis = models.sepsis_label_data(all_labels)
    models.logistic_reg_sepsis(x_train, Y_sepsis, x_test)
    return 0


def merge_results():
    x_test = pd.read_csv(test_features_file_path)
    task1 = pd.read_csv(working_directory + models.task1_results)
    task2 = pd.read_csv(working_directory + models.task2_results)
    task3 = pd.read_csv(working_directory + task3_results)
    pid_ordered = merge.sorting(x_test)
    merge.join(pid_ordered, task1, task2, task3)
    return print('Done')


main_task1()
main_task2()
main_task3(normalized_train_set, normalized_test_set, df_train_labels, df_train_set, mean_train1, std_train1)
merge_results()
