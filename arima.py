import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Sales Data
sales = pd.Series([100, 120, 130, 150, 170, 160, 180, 200])

# Step 2: Plot original data
plt.figure(figsize=(6,4))
plt.plot(sales, marker='o')
plt.title("Original Sales Data")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.grid()
plt.show()

# Step 3: Build ARIMA Model
model = ARIMA(sales, order=(1,1,1))
model_fit = model.fit()

# Step 4: Forecast next 3 values
forecast = model_fit.forecast(steps=3)

print("\nForecasted Sales:")
print(forecast)

# Step 5: Plot forecast
plt.figure(figsize=(6,4))
plt.plot(sales, label="Actual", marker='o')

# Future index
future_index = range(len(sales), len(sales) + 3)

plt.plot(future_index, forecast, label="Forecast", marker='o')

plt.title("Sales Forecast")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()