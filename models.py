import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn import model_selection, linear_model, feature_selection
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.naive_bayes import GaussianNB
import os

np.set_printoptions(threshold=sys.maxsize)

# creation of the Xtrain set
train_features_file_path = '/Users/lucasmichaud/Desktop/MLTask2/train_features.csv'
working_directory = os.path.dirname(train_features_file_path)
dataset = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask2/data_train_task1_bis.csv')
dataset.rename(columns={'Unnamed: 0': 'feat'}, inplace=True)
dataset = dataset.drop(['feat'], axis=1)
data_labels = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask2/DATA_Labels.csv')
data_labels.rename(columns={'Unnamed: 0': 'feat'}, inplace=True)
data_labels = data_labels.drop(['feat'], axis=1)
test_set = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask2/data_test_task1_bis.csv')
test_set.rename(columns={'Unnamed: 0': 'feat'}, inplace=True)
test_set = test_set.drop(['feat'], axis=1)
all_labels = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask2/train_labels.csv')
task1_results = 'task1_results.csv'
task2_results = 'task2_results2.csv'

TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']


def x_train_gen(data):
    """
    convert the dataframe containing the preprocess features into a numpy array containing all the data for a patient
    on one row in order to into in the neural network
    :param data:
    :return:
    """
    data_vector = np.array([]).reshape(0, 137)  # reshape the size of
    for patient, features in tqdm(data.groupby('pid')):
        features1 = features.drop(['pid'], axis=1)
        age = features1.iat[0, 0]
        features2 = features1.drop(['Age'], axis=1)
        features3 = features2.to_numpy().flatten()
        features3 = np.nan_to_num(features3)
        features4 = np.append(age, features3)
        data_vector = np.vstack([data_vector, features4])
    return data_vector


def y_train_gen(data):
    """
    convert data into a numpy array for NN
    :param data:
    :return:
    """
    features1 = data.drop(['pid'], axis=1)
    features2 = features1.values
    return features2


def nn_network_train(X, Y):
    """
    Neural netwrok for task 1 for training: split the train data into train/test in order to find the parameters
    and try all the variations possible
    :param X: input for NN  numpy array
    :param Y: output of NN to train on
    :return: score of the model
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.2, random_state=2)
    model = MLPClassifier(random_state=1, max_iter=600, hidden_layer_sizes=20).fit(X, Y)
    predictions = model.predict_proba(X_test)
    df_submission = pd.DataFrame(predictions)
    df_true = pd.DataFrame(Y_test)
    df_submission.columns = TESTS
    df_true.columns = TESTS

    return get_score1(df_true, df_submission)


def nn_network_test(X, Y, X_test):
    """

    :param X: input for NN  numpy array
    :param Y: output of NN to train on
    :return: Predictions for the test set (submission
    """

    model = MLPClassifier(random_state=1, max_iter=600, hidden_layer_sizes=20).fit(X, Y)
    predictions = model.predict_proba(X_test)
    df_submission = pd.DataFrame(predictions)
    df_submission.columns = TESTS
    df_submission.to_csv(working_directory + task1_results)

    return print('task1 done')


def get_score1(df_true, df_submission):
    task1 = np.mean([metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS])
    return task1


def get_score2(df_true, df_submission):
    task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
    return task2


def sepsis_label_data(train_label):
    """
    convert into numpy array for task2 regression
    :param train_label:
    :return: numpy array train data
    """
    ordered_labels = train_label.sort_values(by='pid', ascending=True, axis=0)
    sepsis_label = ordered_labels['LABEL_Sepsis'].to_numpy()
    sepsis_label = sepsis_label.reshape(len(sepsis_label), 1)
    return sepsis_label


def sepsis_nn(X, Y):
    """
    NN model for task 2. not ideal
    :param X:
    :param Y:
    :return:
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.2, random_state=2)
    sepsis_model = MLPClassifier(random_state=3, max_iter=600, hidden_layer_sizes=20, warm_start=False).fit(X_train,
                                                                                                            Y_train)
    sepsis_pred = sepsis_model.predict_proba(X_test)
    sepsis_pred = sepsis_pred[:, 1]
    sepsis_pred = sepsis_pred.reshape(len(sepsis_pred), 1)
    df_submission = pd.DataFrame(sepsis_pred)
    df_submission.columns = ['LABEL_Sepsis']
    df_true = pd.DataFrame(Y_test)
    df_true.columns = ['LABEL_Sepsis']
    return get_score2(df_true, df_submission)


def logistic_reg_sepsis_train(X, Y):
    """
    model with train data splitted to evaluate parameters
    :param X:
    :param Y:
    :return:
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.2, random_state=2)
    sepsis_model = linear_model.LogisticRegression(penalty='l1', class_weight='balanced', random_state=2, solver='saga',
                                                   max_iter=10000).fit(X, Y)
    opt_feature = feature_selection.SelectFromModel(sepsis_model, prefit=True)
    X_train_opt = opt_feature.transform(X)
    X_test_opt = opt_feature.transform(X_test)
    sepsis_model_opt = LogisticRegression(penalty='l2', class_weight='balanced', random_state=2, solver='sag',
                                          max_iter=5000).fit(X_train_opt, Y)
    sepsis_pred = sepsis_model_opt.predict_proba(X_test_opt)
    sepsis_pred = sepsis_pred[:, 1]
    sepsis_pred = sepsis_pred.reshape(len(sepsis_pred), 1)
    df_submission = pd.DataFrame(sepsis_pred)
    df_submission.columns = ['LABEL_Sepsis']
    df_true = pd.DataFrame(Y_test)
    df_true.columns = ['LABEL_Sepsis']

    return get_score2(df_true, df_submission)


def logistic_reg_sepsis(X, Y, X_test):
    """
    final model for Sepsis diagnostic
    :param X:
    :param Y:
    :param X_test:
    :return: Probabilties for testing Sepsis
    """

    sepsis_model = linear_model.LogisticRegression(penalty='l1', class_weight='balanced', random_state=2, solver='saga',
                                                   max_iter=10000).fit(X, Y)
    opt_feature = feature_selection.SelectFromModel(sepsis_model, prefit=True)
    X_train_opt = opt_feature.transform(X)
    X_test_opt = opt_feature.transform(X_test)
    sepsis_model_opt = LogisticRegression(penalty='l2', class_weight='balanced', random_state=2, solver='sag',
                                          max_iter=5000).fit(X_train_opt, Y)
    sepsis_pred = sepsis_model_opt.predict_proba(X_test_opt)
    sepsis_pred = sepsis_pred[:, 1]
    sepsis_pred = sepsis_pred.reshape(len(sepsis_pred), 1)
    df_submission = pd.DataFrame(sepsis_pred)
    df_submission.columns = ['LABEL_Sepsis']

    df_submission.to_csv(working_directory + task2_results)
    return print('task2 done')


def sepsis_gaussian(X, Y):
    """
    Third model, not ideal
    :param X:
    :param Y:
    :return:
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.2, random_state=2)
    sepsis_model = GaussianNB().fit(X_train, Y_train)
    sepsis_pred = sepsis_model.predict_proba(X_test)
    sepsis_pred = sepsis_pred[:, 1]
    sepsis_pred = sepsis_pred.reshape(len(sepsis_pred), 1)
    df_submission = pd.DataFrame(sepsis_pred)
    df_submission.columns = ['LABEL_Sepsis']
    df_true = pd.DataFrame(Y_test)
    df_true.columns = ['LABEL_Sepsis']
    return get_score1(df_true, df_submission)


