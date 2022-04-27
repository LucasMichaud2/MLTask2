import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

train_features_file_path = '/Users/lucasmichaud/Desktop/MLTask2/train_features.csv'
working_directory = os.path.dirname(train_features_file_path)
np.set_printoptions(threshold=sys.maxsize)
data_train_task1_bis = 'data_train_task1_bis.csv'
data_test_task1_bis = 'data_test_task1_bis.csv'
DATA_Labels = 'DATA_Labels.csv'

train_set = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask2/train_features.csv')
test_set = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask2/test_features.csv')
train_label = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask2/train_labels.csv')


def data_normalization_train(df_train_data):
    """
    Normalize the data based on the mean and the STD of the whole train set
    """

    df_train_data = df_train_data.drop(["pid"], axis=1)
    mean_train_data = df_train_data.mean(axis=0)
    std_train_data = df_train_data.std(axis=0)
    normalized_train_data = (df_train_data - mean_train_data) / std_train_data

    return normalized_train_data, mean_train_data, std_train_data


def preprocess_train(data_set):
    """
    Input the raw dataset, treat all the nans : patient y patient assign the mean value and if all nans, assign the
    global mean.
    Then create a new dataframe with only the mean min max std of each patients for each features.
    Normalize the dataset.
    :param data_set: Dataframe containing the patients metrics
    :return: preprocess train data (dataframe
    """
    data = []
    data1 = data_set.drop(['Time'], axis=1)
    mean_dataset = data1.mean()  # variante
    for patient, features in tqdm(data_set.groupby('pid')):
        features1 = features.drop(['Time'], axis=1)
        patient_mean = features1.mean(axis=0)

        for col in features1:

            if np.isnan(patient_mean[col]):
                patient_mean[col] = mean_dataset[col]  # normalyl

            if features1[col].isna().sum() == 12:

                features1.loc[:, features1.isna().all()] = mean_dataset[col]  # 0 normally
            else:
                features1[col].fillna(patient_mean[col], inplace=True)

        patient_min = features1.min(axis=0)
        patient_max = features1.max(axis=0)
        patient_std = features1.interpolate(method='linear', axis=0).diff().mean()
        age = features1['Age'].mean()  # get the value to inout in STD
        pid = features1['pid'].mean()
        patient_features = pd.concat([patient_mean, patient_min, patient_max, patient_std], axis=1)
        patient_features = patient_features.transpose()
        patient_features.rename(index={0: 'mean', 1: 'min', 2: 'max', 3: 'STD'}, inplace=True)
        patient_features.at['STD', 'pid'] = pid
        patient_features.at['STD', 'Age'] = age
        data.append(patient_features)

    data_filled = pd.concat(data, axis=0)

    pid_column = data_filled['pid']
    data_for_model = data_normalization_train(data_filled)[0]
    data_for_model.insert(0, "pid", pid_column)

    return data_for_model, data_filled, mean_dataset


def preprocess_test(test_set, data_raw, mean_data):
    """
    Same function as previous except we do it with the test set, normalizing with the same normalization as the train
    set.
    :param test_set: Dataframe
    :param data_raw: train set Dataframe
    :param mean_data:
    :return: preprocessed data frame
    """
    data = []
    for patient, features in tqdm(test_set.groupby('pid')):
        features1 = features.drop(['Time'], axis=1)
        patient_mean = features1.mean(axis=0)

        for col in features1:

            if np.isnan(patient_mean[col]):
                patient_mean[col] = mean_data[col]

            if features1[col].isna().sum() == 12:

                features1.loc[:, features1.isna().all()] = mean_data[col]
            else:
                features1[col].fillna(patient_mean[col], inplace=True)

        patient_min = features1.min(axis=0)
        patient_max = features1.max(axis=0)
        patient_std = features1.interpolate(method='linear', axis=0).diff().mean()
        age = features1['Age'].mean()  # get the value to inout in STD
        pid = features1['pid'].mean()
        patient_features = pd.concat([patient_mean, patient_min, patient_max, patient_std], axis=1)
        patient_features = patient_features.transpose()
        patient_features.rename(index={0: 'mean', 1: 'min', 2: 'max', 3: 'STD'}, inplace=True)
        patient_features.at['STD', 'pid'] = pid
        patient_features.at['STD', 'Age'] = age
        data.append(patient_features)

    data_filled = pd.concat(data, axis=0)

    pid_column = data_filled['pid']
    data_filled = data_filled.drop(['pid'], axis=1)

    data_for_model = (data_filled - data_normalization_train(data_raw)[1]) / data_normalization_train(data_raw)[2]

    data_for_model.insert(0, "pid", pid_column)

    return data_for_model


def label_process(labels):
    """
    labels are the y set to train the model on. Sorted by pid as the train set is also sorted by pid
    :param labels: dataframe containing binary data
    :return: ordered label set
    """
    labels_train = labels.drop(["LABEL_Sepsis", "LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"], axis=1)
    ordered_labels = labels_train.sort_values(by='pid', ascending=True, axis=0)

    return ordered_labels


def main2(data_train, data_test, train_label):
    data_train_task1, data_filled, mean_d = preprocess_train(data_train)
    data_test_task1 = preprocess_test(data_test, data_filled, mean_d)
    y_train_task1 = label_process(train_label)

    data_train_task1.to_csv(working_directory + data_train_task1_bis)
    data_test_task1.to_csv(working_directory + data_test_task1_bis)
    y_train_task1.to_csv(working_directory + DATA_Labels)
    return 0


