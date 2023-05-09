import copy
from statistics import mean, stdev
from utils.parser import parse_csv_file
from perceptron import SimpleLinealPerceptron, SimpleNonLinealPerceptron, ActivationFunc, UpdateMode
from config import load_config
from normalize import feature_scaling

import numpy as np
import csv
import random
import pandas as pd

from sklearn.model_selection import train_test_split

acceptable_X_distance = 1.5
acceptable_y_distance = 10

def are_same_class(val1X, val1y, val2X, val2y):
        x_dist = np.linalg.norm(val1X - val2X)
        y_dist = abs(val1y - val2y)
        return True if x_dist <= acceptable_X_distance and y_dist <= acceptable_y_distance else False


def stratified_split(X, y, test_size=0.2):
        data_len = len(X)
        train_indexes = []
        test_indexes = []
        is_on_test = False

        # Split train and test set values if values are the same
        for i in np.arange(data_len):
                for j in train_indexes:
                        if i != j and are_same_class(X[i], y[i], X[j], y[j]):
                                # print('Value on line ' + str(i+2) + ' is of same class with line ' + str(j+2))
                                test_indexes.append(i)
                                is_on_test = True
                                break
                if is_on_test is False:
                        train_indexes.append(i)
                is_on_test = False
        
        # Adjust test set size based on required
        desired_test_size_diff = len(test_indexes) - int(data_len * test_size)
        if desired_test_size_diff > 0:                                  # test set has more values than needed
                new_train_indexes = random.sample(test_indexes, int(desired_test_size_diff))
                for new_train_index in new_train_indexes:
                        train_indexes.append(new_train_index)
                        test_indexes.remove(new_train_index)
        elif desired_test_size_diff < 0:                                # test set has less values than needed
                new_test_indexes = random.sample(train_indexes, int(-desired_test_size_diff))
                for new_test_index in new_test_indexes:
                        test_indexes.append(new_test_index)
                        train_indexes.remove(new_test_index)
        
        
        return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]


testSetProportions = [i/100 for i in range(5, 95, 5)]
resultsMap = {}
averageListTanhTest = []
semAverageListTanhTest = []
averageListTanhTraining = []
semAverageListTanhTraining = []
averageListLogisticTest = []
semAverageListLogisticTest = []
averageListLogisticTraining = []
semAverageListLogisticTraining = []
averageListLinealTest = []
semAverageListLinealTest = []
averageListLinealTraining = []
semAverageListLinealTraining = []

input = parse_csv_file('input_files/TP3-ej2-conjunto.csv')
perceptron_config = load_config()

lineal_perceptron_config = perceptron_config["lineal"]

lineal_num_inputs = int(lineal_perceptron_config["number_of_inputs"])
lineal_epochs = int(lineal_perceptron_config["epochs"])
lineal_learning_rate = float(lineal_perceptron_config["learning_rate"])
lineal_accepted_error = float(lineal_perceptron_config["accepted_error"])

no_lineal_perceptron_config = perceptron_config["no-lineal"]

no_lineal_num_inputs = int(no_lineal_perceptron_config["number_of_inputs"])
no_lineal_epochs = int(no_lineal_perceptron_config["epochs"])
no_lineal_learning_rate = float(no_lineal_perceptron_config["learning_rate"])
no_lineal_accepted_error = float(no_lineal_perceptron_config["accepted_error"])
beta = float(no_lineal_perceptron_config["beta"])

for proportion in testSetProportions:
        resultsMap[proportion] = {
                ActivationFunc.LOGISTIC: {'train': [], 'test':[]},
                ActivationFunc.TANH: {'train': [], 'test':[]},
                'lineal': {'train': [], 'test':[]}
        }

for i in range(10):
        for testPercentage in testSetProportions:
                X = input[:, 0:3]
                y = input[:, 3]

                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testPercentage)

                X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=testPercentage)

                # print(X_train)
                # print(X_test)

                inputCopy = copy.deepcopy(input)

                lineal_perceptron = SimpleLinealPerceptron(lineal_num_inputs, lineal_learning_rate, lineal_epochs, lineal_accepted_error, UpdateMode.BATCH)
                train_lineal_mse, errors = lineal_perceptron.train(X_train, y_train)
                test_lineal_mse = lineal_perceptron.mse_error(X_test, y_test)
                # test_lineal_mse, errors = lineal_perceptron.train(X_test, y_test)
        
                non_lineal_logistic_perceptron = SimpleNonLinealPerceptron(no_lineal_num_inputs, no_lineal_learning_rate, no_lineal_epochs, no_lineal_accepted_error, UpdateMode.BATCH, beta, ActivationFunc.LOGISTIC)
                # ynorm_logistic = feature_scaling(y_train, 0, 1)
                # ynorm_test_logistic = feature_scaling(y_test, 0, 1)
                train_logistic_mse, errors = non_lineal_logistic_perceptron.train(X_train, y_train)
                test_logistic_mse = non_lineal_logistic_perceptron.mse_error(X_test, y_test)
                # test_logistic_mse, errors = non_lineal_logistic_perceptron.train(X_test, y_test)
                
                non_lineal_tanh_perceptron = SimpleNonLinealPerceptron(no_lineal_num_inputs, no_lineal_learning_rate, no_lineal_epochs, no_lineal_accepted_error, UpdateMode.BATCH, beta, ActivationFunc.TANH)
                # ynorm_tanh = feature_scaling(y_train, -1, 1)
                # ynorm_test_tanh = feature_scaling(y_test, -1, 1)
                train_tanh_mse, errors = non_lineal_tanh_perceptron.train(X_train, y_train)
                test_tanh_mse = non_lineal_tanh_perceptron.mse_error(X_test, y_test)

                resultsMap[testPercentage][ActivationFunc.LOGISTIC]['train'].append(train_logistic_mse)
                resultsMap[testPercentage][ActivationFunc.TANH]['train'].append(train_tanh_mse)
                resultsMap[testPercentage]['lineal']['train'].append(train_lineal_mse)
                resultsMap[testPercentage][ActivationFunc.LOGISTIC]['test'].append(test_logistic_mse)
                resultsMap[testPercentage][ActivationFunc.TANH]['test'].append(test_tanh_mse)
                resultsMap[testPercentage]['lineal']['test'].append(test_lineal_mse)
        
for proportion in testSetProportions:
        averageListLogisticTest.append(mean(resultsMap[proportion][ActivationFunc.LOGISTIC]['test']))
        semAverageListLogisticTest.append(stdev(resultsMap[proportion][ActivationFunc.LOGISTIC]['test']))
        averageListLogisticTraining.append(mean(resultsMap[proportion][ActivationFunc.LOGISTIC]['train']))
        semAverageListLogisticTraining.append(stdev(resultsMap[proportion][ActivationFunc.LOGISTIC]['train']))
        averageListTanhTest.append(mean(resultsMap[proportion][ActivationFunc.TANH]['test']))
        semAverageListTanhTest.append(stdev(resultsMap[proportion][ActivationFunc.TANH]['test']))
        averageListTanhTraining.append(mean(resultsMap[proportion][ActivationFunc.TANH]['train']))
        semAverageListTanhTraining.append(stdev(resultsMap[proportion][ActivationFunc.TANH]['train']))
        averageListLinealTest.append(mean(resultsMap[proportion]['lineal']['test']))
        semAverageListLinealTest.append(stdev(resultsMap[proportion]['lineal']['test']))
        averageListLinealTraining.append(mean(resultsMap[proportion]['lineal']['train']))
        semAverageListLinealTraining.append(stdev(resultsMap[proportion]['lineal']['train']))

with open('ej2_sets_proportions.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['method', 'test_proportion', 'test_error', 'test_error_std', 'train_error', 'train_error_std'])
        for proportion in testSetProportions:
                writer.writerow(['Lineal', proportion, mean(resultsMap[proportion]['lineal']['test']), stdev(resultsMap[proportion]['lineal']['test']), mean(resultsMap[proportion]['lineal']['train']), stdev(resultsMap[proportion]['lineal']['train'])])
                writer.writerow(['Tanh', proportion, mean(resultsMap[proportion][ActivationFunc.TANH]['test']), stdev(resultsMap[proportion][ActivationFunc.TANH]['test']), mean(resultsMap[proportion][ActivationFunc.TANH]['train']), stdev(resultsMap[proportion][ActivationFunc.TANH]['train'])])
                writer.writerow(['Logistic', proportion, mean(resultsMap[proportion][ActivationFunc.LOGISTIC]['test']), stdev(resultsMap[proportion][ActivationFunc.LOGISTIC]['test']), mean(resultsMap[proportion][ActivationFunc.LOGISTIC]['train']), stdev(resultsMap[proportion][ActivationFunc.LOGISTIC]['train'])])
