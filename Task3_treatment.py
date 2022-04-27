import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import HistGradientBoostingRegressor
import os

train_features_file_path = '/Users/lucasmichaud/Desktop/MLTask2/train_features.csv'
working_directory = os.path.dirname(train_features_file_path)
data_set = pd.read_csv(train_features_file_path)
labels_set = pd.read_csv(working_directory + '/train_labels.csv')
test_set = pd.read_csv(working_directory + '/test_features.csv')
task_3_data_for_model = 'task_3_data_for_model.csv'
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']


# only concerned labels
def data_treatment(data_set):
    features = data_set.drop(['Time'], axis=1)
    no_pid = data_set.drop(['Time', 'pid'], axis=1)
    mean_train_data = no_pid.mean(axis=0)  # Attention ce code est faux
    std_train_data = no_pid.std(axis=0)
    data_for_model = []
    for patient, data in tqdm(features.groupby('pid')):
        patient_mean = data.mean(axis=0)

        for col in data:

            if np.isnan(patient_mean[col]):
                patient_mean[col] = 0

            if data[col].isna().sum() == 12:

                data.loc[:, data.isna().all()] = 0
            else:
                data[col].fillna(patient_mean[col], inplace=True)

        pid_col = data['pid']
        data = data.drop(['pid'], axis=1)
        normalized_data = (data - mean_train_data) / std_train_data
        normalized_data.insert(0, "pid", pid_col)
        data_for_model.append(normalized_data)
    final_data = pd.concat(data_for_model, axis=0)
    final_data.to_csv(working_directory + task_3_data_for_model)
    return 0


# normalized_data = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask2/Task3_data_model.csv')


def prepare_regression(data):
    data_vector = np.array([]).reshape(0, 64)
    for patient, features in tqdm(data.groupby('pid')):
        no_pid = features.drop(['pid'], axis=1)

        no_pid = no_pid.drop(['Unnamed: 0'], axis=1)

        flatten_data = no_pid.to_numpy().flatten()

        flatten_data = np.nan_to_num(flatten_data)

        data_vector = np.vstack([data_vector, flatten_data])

    return data_vector


def mean_std(data):
    mean_features = data[['RRate', 'ABPm', 'SpO2', 'Heartrate']].mean()
    mean_features.rename(index={'RRate': 'LABEL_RRate', 'ABPm': 'LABEL_ABPm', 'SpO2': 'LABEL_SpO2',
                                'Heartrate': 'LABEL_Heartrate'}, inplace=True)
    std_features = data[['RRate', 'ABPm', 'SpO2', 'Heartrate']].std()
    std_features.rename(index={'RRate': 'LABEL_RRate', 'ABPm': 'LABEL_ABPm', 'SpO2': 'LABEL_SpO2',
                               'Heartrate': 'LABEL_Heartrate'}, inplace=True)

    return mean_features, std_features


def y_treatment(features_data, data):
    labels = data[['pid', 'LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']]
    ordered_labels = labels.sort_values(by='pid', ascending=True, axis=0)
    y_train_data = ordered_labels.drop(['pid'], axis=1)
    normalized_y_data = (y_train_data - mean_std(features_data)[0]) / mean_std(features_data)[1]
    y_data = normalized_y_data.to_numpy()
    return y_data


def regression_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = HistGradientBoostingRegressor(loss="squared_error").fit(X_train, Y_train)
    vitals_pred = model.predict(X_train)

    return 0


def vital_nn(X, Y, X_test):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    model = MLPRegressor(hidden_layer_sizes=20, max_iter=500, random_state=2).fit(X, Y)
    vital_pred = model.predict(X_test)
    df_submission = pd.DataFrame(vital_pred)
    df_submission.columns = VITALS

    # df_true = pd.DataFrame(Y_test)
    # df_true.columns = VITALS
    return df_submission  # get_score_t3(df_true, df_submission)


def get_score_t3(df_true, df_submission):
    task3 = np.mean(
        [0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    return task3


def vital_treatment_train(data_set):
    features = data_set[['pid', 'RRate', 'ABPm', 'SpO2', 'Heartrate']]
    no_pid = features.drop(['pid'], axis=1)
    mean_train_data = no_pid.mean(axis=0)  # Attention ce code est faux
    std_train_data = no_pid.std(axis=0)
    data_for_model = []
    for patient, data in tqdm(features.groupby('pid')):
        patient_mean = data.mean(axis=0)
        patient_mean1 = pd.DataFrame(patient_mean).transpose()
        patient_std = data.interpolate(method='linear', axis=0).diff().mean()
        patient_std1 = pd.DataFrame(patient_std).transpose()
        patient_std1['pid'] = patient_mean1['pid']
        patient_max = data.max(axis=0)
        patient_min = data.min(axis=0)
        patient_max = pd.DataFrame(patient_max).transpose()
        patient_min = pd.DataFrame(patient_min).transpose()

        for col in data:

            if np.isnan(patient_mean[col]):
                patient_mean[col] = 0

            if data[col].isna().sum() == 12:

                data.loc[:, data.isna().all()] = 0
            else:
                data[col].fillna(patient_mean[col], inplace=True)

        data = pd.concat([data, patient_mean1], axis=0)
        data = pd.concat([data, patient_std1], axis=0)
        data = pd.concat([data, patient_max, patient_min], axis=0)

        pid_col = data['pid']

        data = data.drop(['pid'], axis=1)
        normalized_data = (data - mean_train_data) / std_train_data
        normalized_data.insert(0, "pid", pid_col)
        data_for_model.append(normalized_data)
    final_data = pd.concat(data_for_model, axis=0)
    return final_data, mean_train_data, std_train_data


def vital_treatment_test(test_set, mean_train, std_train):
    features = test_set[['pid', 'RRate', 'ABPm', 'SpO2', 'Heartrate']]
    data_for_model = []
    for patient, data in tqdm(features.groupby('pid')):
        patient_mean = data.mean(axis=0)
        patient_mean1 = pd.DataFrame(patient_mean).transpose()
        patient_std = data.interpolate(method='linear', axis=0).diff().mean()
        patient_std1 = pd.DataFrame(patient_std).transpose()
        patient_std1['pid'] = patient_mean1['pid']
        patient_max = data.max(axis=0)
        patient_min = data.min(axis=0)
        patient_max = pd.DataFrame(patient_max).transpose()
        patient_min = pd.DataFrame(patient_min).transpose()
        for col in data:

            if np.isnan(patient_mean[col]):
                patient_mean[col] = 0

            if data[col].isna().sum() == 12:

                data.loc[:, data.isna().all()] = 0
            else:
                data[col].fillna(patient_mean[col], inplace=True)
        data = pd.concat([data, patient_mean1], axis=0)
        data = pd.concat([data, patient_std1], axis=0)
        data = pd.concat([data, patient_max, patient_min], axis=0)
        pid_col = data['pid']
        data = data.drop(['pid'], axis=1)
        normalized_data = (data - mean_train) / std_train
        normalized_data.insert(0, "pid", pid_col)
        data_for_model.append(normalized_data)
    final_data = pd.concat(data_for_model, axis=0)
    return final_data



