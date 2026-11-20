import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0])
marks = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])


def gradient_descent(x, y, lr, epochs):
    m = 0
    c = 0
    n = len(x)
    losses = []
    for epoch in range(1, epochs + 1):
        y_hat = m * x + c
        dm = (-2 / n) * np.sum(x * (y - y_hat))
        dc = (-2 / n) * np.sum(y - y_hat)
        m = m - lr * dm
        c = c - lr * dc
        mse = np.mean((y - (m * x + c)) ** 2)
        losses.append(mse)
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE:", mse)
    return m, c, losses


model = LinearRegression().fit(hours.reshape(-1, 1), marks)
print("scikit-learn -> m =", round(model.coef_[0], 4), ", c =", round(model.intercept_, 4))

for lr in [0.001, 0.01, 0.05]:
    print("\nLearning rate:", lr)
    m, c, losses = gradient_descent(hours, marks, lr, 1000)
    print("Final m =", m, ", c =", c)
    plt.plot(losses, label="lr = " + str(lr))

plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Loss Curve")
plt.yscale("log")
plt.legend()
plt.show()
