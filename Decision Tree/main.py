import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def generate_random_dataset():
    
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))
    return X, y

def train_decision_tree(X, y, max_depth):
    regressor = DecisionTreeRegressor(max_depth=max_depth)
    regressor.fit(X, y)
    return regressor

def predict_and_plot(regressor, X_test, label, color):
    y_pred = regressor.predict(X_test)
    plt.plot(X_test, y_pred, color=color, label=f"max_depth={label}", linewidth=2)

def plot_results(X, y, X_test):
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()

def main():
    X, y = generate_random_dataset()

    depth_low = 2
    depth_high = 5

    regressor_low = train_decision_tree(X, y, depth_low)
    regressor_high = train_decision_tree(X, y, depth_high)

    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

    predict_and_plot(regressor_low, X_test, label=depth_low, color="cornflowerblue")
    predict_and_plot(regressor_high, X_test, label=depth_high, color="yellowgreen")

    plot_results(X, y, X_test)

if __name__ == "__main__":
    main()
