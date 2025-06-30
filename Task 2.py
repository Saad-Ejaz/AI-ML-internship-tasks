# 📦 Import required libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 📥 Step 1: Download historical stock data (Tesla in this case)
data = yf.download('TSLA', start='2023-06-01', end='2024-06-01')
data = data.reset_index()  # Reset index to make 'Date' a column
print("First 5 rows of data:")
print(data.head())

# 🧹 Step 2: Select relevant features and create target
# We'll use today's 'Open', 'High', 'Low', 'Volume' to predict tomorrow's 'Close'
df = data[['Open', 'High', 'Low', 'Volume', 'Close']]
df['Target'] = df['Close'].shift(-1)  # Next day's close as the target
df = df.dropna()  # Remove the last row which has NaN in 'Target'

# 🎯 Step 3: Define features (X) and target (y)
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Target']

# ✂️ Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ⚙️ Step 4: Train a Linear Regression model (optional)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 🌲 Step 5: Train a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 📊 Step 6: Plot actual vs predicted prices
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Prices', linewidth=2)
plt.plot(y_pred_rf, label='Predicted Prices (RF)', linestyle='--')
plt.title('Actual vs Predicted Stock Closing Prices (Tesla)')
plt.xlabel('Days')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# 🧮 Step 7: Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Mean Squared Error (Random Forest): {mse_rf:.2f}")