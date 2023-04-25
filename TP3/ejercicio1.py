import numpy as np

# Définition de la fonction d'activation Escalon/Heaviside
# english: Heaviside step function

def heaviside(x):
    return np.where(x >= 0, 1, 0)

# Définition de l'algorithme d'apprentissage du perceptron
# (english) Perceptron learning algorithm


def perceptron_learning(X, y, learning_rate=0.1, max_epochs=100):
    # Initialisation des poids et du biais
    # english: Weights and bias initialization
    w = np.zeros(X.shape[1])
    b = 0
    converged = False
    epoch = 0

    while not converged and epoch < max_epochs:
        # Mise à jour du compteur d'itérations
        # english: Update iteration counter
        epoch += 1

        # Pour chaque exemple d'entraînement
        # english: For each training example
        for i in range(X.shape[0]):
            # Prédiction de la sortie du perceptron
            # english: Perceptron output prediction
            y_pred = heaviside(np.dot(w, X[i]) + b)

            # Calcul de l'erreur
            e = y[i] - y_pred

            print(e)

            # Mise à jour des poids et du biais
            # english: Weights and bias update
            w += learning_rate * e * X[i]
            b += learning_rate * e

        # Vérification de la convergence
        # english: Check convergence
        y_pred = heaviside(np.dot(X, w) + b)
        if np.array_equal(y, y_pred):
            converged = True

    return w, b


# Exemple d'utilisation avec une tâche logique AND
# english: Example of use with an AND logic task
X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
y = np.array([-1, -1, -1, 1])

w, b = perceptron_learning(X, y)

# Affichage des résultats
# english: Display results
print("Weights: ", w)
print("Bias: ", b)
print("Predictions: ", heaviside(np.dot(X, w) + b))
