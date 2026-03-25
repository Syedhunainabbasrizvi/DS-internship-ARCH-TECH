import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 2. Load Dataset (Online CSV)
url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
df = pd.read_csv(url)

# 3. Data Preprocessing

# Rename columns (easy access)
df.rename(columns={
    'AAPL.Open': 'Open',
    'AAPL.High': 'High',
    'AAPL.Low': 'Low',
    'AAPL.Close': 'Close',
    'AAPL.Volume': 'Volume'
}, inplace=True)

# Select required columns
data = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Handle missing values (pandas 3.x compatible)
data = data.ffill()

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 4. Train-Test Split
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 5. Features & Target
X_train = train_data[:, :-1]   # Open, High, Low, Close
y_train = train_data[:, 3]     # Close

X_test = test_data[:, :-1]
y_test = test_data[:, 3]

# 6. Train Model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("Model Evaluation:")
print("RMSE:", rmse)
print("MAE:", mae)

# 9. Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.title("Stock Price Prediction (Linear Regression)")
plt.xlabel("Time")
plt.ylabel("Normalized Price")
plt.legend()
plt.show()