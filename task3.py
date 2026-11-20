import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    "experience": [1, 2, 3, None, 5, 6, 7, 8, 9, 10, 3, 5],
    "education": ["BS", "BS", "MS", "BS", "MS", "PhD", "MS", "PhD", "BS", "PhD", "BS", "MS"],
    "salary_k": [45, 50, 70, 58, 85, 120, 95, 140, 90, 160, 55, 88]
}

df = pd.DataFrame(data)
print(df.isnull().sum())

df["experience"] = df["experience"].fillna(df["experience"].median())

df = pd.get_dummies(df, columns=["education"], drop_first=True)
print(df.head())

X = df.drop("salary_k", axis=1)
y = df["salary_k"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2:", round(r2_score(y_test, y_pred), 3))

new_person = pd.DataFrame([[4, True, False]], columns=X.columns)
prediction = model.predict(scaler.transform(new_person))
print("Predicted salary for MS with 4 years:", round(prediction[0], 1), "k")
