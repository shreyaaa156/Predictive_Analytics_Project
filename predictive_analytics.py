import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset/sales_data.csv")

print(data.head())

plt.plot(data['Year'], data['Sales'])

plt.title("Historical Sales Data")

plt.xlabel("Year")

plt.ylabel("Sales")

plt.show()

X = data[['Year']]

y = data['Sales']

model = LinearRegression()

model.fit(X, y)

future_years = pd.DataFrame({
    'Year': [2025, 2026, 2027, 2028]
})

predictions = model.predict(future_years)

print(predictions)

plt.scatter(data['Year'], data['Sales'])

plt.plot(
    future_years,
    predictions
)

plt.title("Sales Forecasting")

plt.xlabel("Year")

plt.ylabel("Predicted Sales")

plt.show()

score = model.score(X, y)

print("Accuracy:", score)

plt.savefig("outputs/sales_forecast.png")