import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf

# Fix the random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and Prepare the Data
file_path = r'C:\\Users\\Admin\\Downloads\\Bitcoin_history.csv'
bitcoin_data = pd.read_csv(file_path)

# Convert 'Date' to datetime and 'Price' to float
bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'], format='%d-%b-%y')
bitcoin_data['Price'] = bitcoin_data['Price'].str.replace(',', '').astype(float)
bitcoin_data.set_index('Date', inplace=True)

# Scale the data (LSTMs work better with scaled data)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(bitcoin_data['Price'].values.reshape(-1, 1))

# Prepare the data for the LSTM model
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # Look back 60 time steps (60 days)
X, y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model with Dropout layers to prevent overfitting
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))  # Dropout layer
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=50, callbacks=[early_stopping])

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get the actual values
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
mae = mean_absolute_error(y_test_inverse, test_predict)
rmse = np.sqrt(mean_squared_error(y_test_inverse, test_predict))
r2 = r2_score(y_test_inverse, test_predict)

print(f"LSTM Model - MAE: {mae}")
print(f"LSTM Model - RMSE: {rmse}")
print(f"LSTM Model - R-squared: {r2}")

# Prepare the data for plotting
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step):len(scaled_data), :] = test_predict

# Plotting the results with corrected axis scales
plt.figure(figsize=(14, 7))
plt.plot(bitcoin_data.index, bitcoin_data['Price'], label='Actual Price', color='blue')
plt.plot(bitcoin_data.index, train_predict_plot, label='Train Predict', color='orange')
plt.plot(bitcoin_data.index, test_predict_plot, label='Test Predict', color='green')
plt.legend(loc='upper left')
plt.title('Bitcoin Price Prediction using LSTM with Dropout and Early Stopping')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# ARIMA Model for Comparison
arima_model = ARIMA(bitcoin_data['Price'][:train_size], order=(5, 1, 0))
arima_fit = arima_model.fit()

# Forecast using the ARIMA model
arima_predictions = arima_fit.forecast(steps=len(bitcoin_data) - train_size)

# Evaluate the ARIMA model
arima_mae = mean_absolute_error(bitcoin_data['Price'][train_size:], arima_predictions)
arima_rmse = np.sqrt(mean_squared_error(bitcoin_data['Price'][train_size:], arima_predictions))
arima_r2 = r2_score(bitcoin_data['Price'][train_size:], arima_predictions)

print(f"ARIMA Model - MAE: {arima_mae}")
print(f"ARIMA Model - RMSE: {arima_rmse}")


# Plotting ARIMA predictions
plt.figure(figsize=(14, 7))
plt.plot(bitcoin_data.index, bitcoin_data['Price'], label='Actual Price', color='blue')
plt.plot(bitcoin_data.index[train_size:], arima_predictions, label='ARIMA Predicted Price', color='red')
plt.legend(loc='upper left')
plt.title('Bitcoin Price Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

