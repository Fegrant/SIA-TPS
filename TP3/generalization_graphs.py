import imp
from multilayer_perceptron import MultilayerPerceptron

runs = 10

errors = []
errors_std = []
for i in np.arange(runs):
    mlp = MultilayerPerceptron([35] + [25] + [25] + [10], momentum)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    set_size = len(X)
    train_set_size = len(X_train)
    test_set_size = set_size - train_set_size

    # Xt = add_noise(X, noise)
    mlp.train(X, y, epochs, learning_rate)

    mlp_errors = []

    for i in np.arange(set_size):
        predicted_values = mlp.predict(X[i])[0]
        # Averiguo cual es el valor que quiero
        highest_error = 0
        for j in np.arange(10):
            if predicted_values[j] > highest_error and y[i][j] != 1:
                highest_error = predicted_values[j]
            if y[i][j] == 1:
                num_expected = j
        error = 1 - (predicted_values[num_expected] - highest_error) if predicted_values[num_expected] > highest_error else 1
        mlp_errors.append(error)
    errors.append(np.mean(np.array(mlp_errors)))




with open('generalization.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['perceptron_amount', 'error', 'error_std'])
    rate_count = 0
    rate_idx = 0
    for i in np.arange(len(errors)):
        writer.writerow([inner_perceptrons[i], errors[i], errors_std[i]])