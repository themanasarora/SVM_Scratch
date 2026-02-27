import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from SVMkernel import SVM

# -----------------------------
# Generate Nonlinear Data
# -----------------------------
X, y = make_circles(n_samples=200, noise=0.1, factor=0.3)
y = np.where(y == 0, -1, 1)

# -----------------------------
# Train Kernel SVM
# -----------------------------
clf = SVM(kernel="rbf", gamma=5, n_iter=2000)
clf.fit(X, y)

predictions = clf.predict(X)
accuracy = np.mean(predictions == y)

print("Accuracy:", accuracy)

# -----------------------------
# Visualization
# -----------------------------
def plot_decision_boundary(X, y, model):

    # Create grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # Flatten grid to feed into model
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict on grid
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # Plot contour
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")

    plt.title("Kernel SVM Decision Boundary")
    plt.show()

plot_decision_boundary(X, y, clf)