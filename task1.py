import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

hours = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0])
marks = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])

X = hours.reshape(-1, 1)
model = LinearRegression()
model.fit(X, marks)

print("Slope:", round(model.coef_[0], 3))
print("Intercept:", round(model.intercept_, 3))

print("Marks for 5.5 hours:", round(model.predict([[5.5]])[0], 2))
print("Marks for 8 hours:", round(model.predict([[8]])[0], 2))

y_pred = model.predict(X)

print("MAE:", round(mean_absolute_error(marks, y_pred), 3))
print("MSE:", round(mean_squared_error(marks, y_pred), 3))
print("R2:", round(r2_score(marks, y_pred), 3))

plt.scatter(hours, marks, label="Actual data")
plt.plot(hours, y_pred, color="red", label="Regression line")
plt.xlabel("Hours studied")
plt.ylabel("Marks")
plt.title("Hours Studied vs Marks")
plt.legend()
plt.show()
