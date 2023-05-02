import copy
from statistics import mean
from utils.parser import parse_csv_file
from perceptron import SimpleLinealPerceptron, SimpleNonLinealPerceptron, ActivationFunc
from config import load_config, ex2_test_size
from normalize import feature_scaling

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

testSetProportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
resultsMap = {}
averageListTanhTest = []
averageListTanhTraining = []
averageListLogisticTest = []
averageListLogisticTraining = []
averageListLinealTest = []
averageListLinealTraining = []

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
        resultsMap[proportion] = {ActivationFunc.LOGISTIC: {'train': [], 'test':[]},
                                  ActivationFunc.TANH: {'train': [], 'test':[]},
                                  'lineal': {'train': [], 'test':[]}
                                }

for i in range(5):
        for testPercentage in testSetProportions:
                input_train, input_test = train_test_split(input, test_size=testPercentage)
                
                X_train = input_train[:,:-1]
                y_train = input_train[:,-1]
 
                X_test = input_test[:,:-1]
                y_test = input_test[:,-1]

                inputCopy = copy.deepcopy(input)

                lineal_perceptron = SimpleLinealPerceptron(lineal_num_inputs, lineal_learning_rate, lineal_epochs, lineal_accepted_error)
                train_lineal_mse = lineal_perceptron.train(X_train, y_train)
                test_lineal_mse = lineal_perceptron.train(X_test, y_test)
        
                non_lineal_logistic_perceptron = SimpleNonLinealPerceptron(no_lineal_num_inputs, no_lineal_learning_rate, no_lineal_epochs, no_lineal_accepted_error, beta, ActivationFunc.LOGISTIC)
                ynorm_logistic = feature_scaling(y_train, 0, 1)
                ynorm_test_logistic = feature_scaling(y_test, 0, 1)
                train_logistic_mse = non_lineal_logistic_perceptron.train(X_train, ynorm_logistic)
                test_logistic_mse = non_lineal_logistic_perceptron.train(X_test, ynorm_test_logistic)
                
                non_lineal_tanh_perceptron = SimpleNonLinealPerceptron(no_lineal_num_inputs, no_lineal_learning_rate, no_lineal_epochs, no_lineal_accepted_error, beta, ActivationFunc.TANH)
                ynorm_tanh = feature_scaling(y_train, -1, 1)
                ynorm_test_tanh = feature_scaling(y_test, -1, 1)
                train_tanh_mse = non_lineal_tanh_perceptron.train(X_train, ynorm_tanh)
                test_tanh_mse = non_lineal_tanh_perceptron.train(X_test, ynorm_test_tanh)

                resultsMap[testPercentage][ActivationFunc.LOGISTIC]['train'].append(train_logistic_mse)
                resultsMap[testPercentage][ActivationFunc.TANH]['train'].append(train_tanh_mse)
                resultsMap[testPercentage]['lineal']['train'].append(train_lineal_mse)
                resultsMap[testPercentage][ActivationFunc.LOGISTIC]['test'].append(test_logistic_mse)
                resultsMap[testPercentage][ActivationFunc.TANH]['test'].append(test_tanh_mse)
                resultsMap[testPercentage]['lineal']['test'].append(test_lineal_mse)
        
for proportion in testSetProportions:
        averageListLogisticTest.append(mean(resultsMap[proportion][ActivationFunc.LOGISTIC]['test']))
        averageListLogisticTraining.append(mean(resultsMap[proportion][ActivationFunc.LOGISTIC]['train']))
        averageListTanhTest.append(mean(resultsMap[proportion][ActivationFunc.TANH]['test']))
        averageListTanhTraining.append(mean(resultsMap[proportion][ActivationFunc.TANH]['train']))
        averageListLinealTest.append(mean(resultsMap[proportion]['lineal']['test']))
        averageListLinealTraining.append(mean(resultsMap[proportion]['lineal']['train']))

width = 0.3
x = np.arange(len(testSetProportions))
plt.xlabel("Test set %", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.title("Error on both Sets (Tanh)")
plt.bar(x, averageListTanhTest,width, label='Test')
plt.bar(x+width, averageListTanhTraining, width,label ='Training')
plt.xticks(x + width / 2, ('10', '20', '30','40','50','60','70','80'))
plt.legend()
plt.show()

plt.xlabel("Test set %", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.title("Error on both Sets (Logistica)")
plt.bar(x, averageListLogisticTest, width, label='Test')
plt.bar(x+width, averageListLogisticTraining, width, label='Training')
plt.xticks(x + width / 2, ('10', '20', '30','40','50','60','70','80'))
plt.legend()
plt.show()

plt.xlabel("Test set %", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.title("Error on both Sets (Lineal)")
plt.bar(x, averageListLinealTest, width, label='Test')
plt.bar(x+width, averageListLinealTraining,width, label='Training')
plt.xticks(x + width / 2, ('10', '20', '30','40','50','60','70','80'))
plt.legend()
plt.show()


        