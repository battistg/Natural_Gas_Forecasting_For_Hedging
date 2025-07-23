# Loading Libraries & Data
import warnings
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from statsmodels.tsa.stattools import acf

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from statsmodels.graphics.tsaplots import plot_acf

from scipy import stats
import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
import matplotlib.dates as mdates

# Fix seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load the TTF dataset
data = pd.read_csv("data")
print("Dataset Info:")
print(data.info())

# Data Preparation
# Convert Date column with correct format (YYYY-MM-DD)
data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
print("\nFirst 3 rows:")
print(data.head(3))

print("\nLast 3 rows:")
print(data.tail(3))

# Renaming columns
data = data.rename({'Date': 'date', 'Price': 'gas_price'}, axis=1)

# Setting Date as index
data = data.set_index('date')

# Sort data by date
data = data.sort_index()

# Drop year 2014 

data = data[data.index >= '2015-01-01']  

print(f"\nFiltered data range: {data.index[0]} to {data.index[-1]}")
print(f"Filtered total records: {len(data)}")
print(f"Records dropped: {4017 - len(data)} (year 2014)")



print(data.tail(3))

print(f"\nData range: {data.index[0]} to {data.index[-1]}")
print(f"Total records: {len(data)}")

# Exploratory Data Analysis


# Missing Values
print("Missing values check:")
print(data.isnull().sum())


# Line Chart
plt.figure(figsize=(15, 6))
plt.plot(data.index, data['gas_price'], color='black')
plt.title('Natural Gas TTF Spot Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Histogram
plt.figure(figsize=(10, 5))
plt.hist(data['gas_price'], bins=50, color='blue', edgecolor='black')
plt.title('Price Distribution of TTF Spot Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Check if Time Series is Stationary
def test_stationarity(timeseries):
    # Determining rolling statistics
    rolmean = timeseries.rolling(25).mean()
    rolstd = timeseries.rolling(25).std()
    
    # Plot rolling statistics
    plt.figure(figsize=(20,10))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    
    print(dfoutput)

test_stationarity(data['gas_price'])

# Estimating and Eliminating Trends

# Exponential Weighted Moving Average
ts_sqrt = np.sqrt(data)
expwighted_avg = ts_sqrt.ewm(halflife=25).mean()
ts_sqrt_ewma_diff = ts_sqrt - expwighted_avg
test_stationarity(ts_sqrt_ewma_diff.dropna())

# Data Modeling

# Train Test Split
data = data.sort_index()
train = data[data.index <= '2023-12-31']
test = data[data.index >= '2024-01-01']


print("\nTrain data range:")
print(train.head(2))
print(train.tail(2))

print("\nTest data range:")
print(test.head(2))
print(test.tail(2))

# Plot train/test split
ax = train.plot(figsize=(20, 10), color='b')
test.plot(ax=ax, color='black')
plt.legend(['train set', 'test set'])
plt.show()



# Calculate log returns
data['log_returns'] = np.log(data['gas_price'] / data['gas_price'].shift(1))

# Drop NaN values created by the shift operation from the entire DataFrame
data = data.dropna()


# Autocorrelation Function on log returns
acf_vals_returns = acf(data['log_returns'], nlags=15)

# Lags from 1 to 15
for lag in range(1, 16):
    print(f"Autocorrelation at lag {lag}: {acf_vals_returns[lag]:.4f}")


# lags
lags_to_test = [1,2,3,4,5,6,7,8,9, 10,11,12,13,14, 15]

# test on log returns
ljung_results_returns = acorr_ljungbox(data['log_returns'], lags=lags_to_test, return_df=True)

for lag in lags_to_test:
    stat = ljung_results_returns.loc[lag, 'lb_stat']
    pval = ljung_results_returns.loc[lag, 'lb_pvalue']
    print(f"Lag {lag}:  Ljung-Box Stat = {stat:.2f},  p-value = {pval:.4f}")
    

slot_arima = 30

# === Rolling Forecast on TRAIN ===
print("\n--- ARIMA Rolling Forecast on Train Set ---")
rolling_train_preds = []
history_train = list(train['gas_price'][:slot_arima])
train_dates = train.index[slot_arima:]

for date in train_dates:
    model = ARIMA(history_train, order=(1, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)[0]
    rolling_train_preds.append(forecast)
    history_train.append(train.loc[date, 'gas_price'])

# Evaluation (Train)
y_train_true = train['gas_price'][slot_arima:]
r2_arima_train = r2_score(y_train_true, rolling_train_preds)
mse_arima_train = mean_squared_error(y_train_true, rolling_train_preds)
rmse_arima_train = sqrt(mse_arima_train)

print(f"Train R² (rolling): {r2_arima_train:.4f}")
print(f"Train MSE: {mse_arima_train:.4f}")
print(f"Train RMSE: {rmse_arima_train:.4f}")

# === Rolling Forecast on TEST ===
print("\n--- ARIMA Rolling Forecast on Test Set ---")
rolling_test_preds = []
history_test = list(train['gas_price'])

for date in test.index:
    model = ARIMA(history_test, order=(1, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)[0]
    rolling_test_preds.append(forecast)
    history_test.append(test.loc[date, 'gas_price'])

# Evaluation (Test)
r2_arima_test = r2_score(test['gas_price'], rolling_test_preds)
mse_arima_test = mean_squared_error(test['gas_price'], rolling_test_preds)
rmse_arima_test = sqrt(mse_arima_test)

print(f"Test R² (rolling): {r2_arima_test:.4f}")
print(f"Test MSE: {mse_arima_test:.4f}")
print(f"Test RMSE: {rmse_arima_test:.4f}")

# Plot - Test
plt.figure(figsize=(15, 5))
plt.plot(test.index, test['gas_price'], label='Actual', color='black')
plt.plot(test.index, rolling_test_preds, label='ARIMA Forecast', color='orange')
plt.title("ARIMA Day-Ahead Rolling Forecast on Test Set")
plt.xlabel("Date")
plt.ylabel("Gas Price")
plt.legend()
plt.grid(True)
plt.show()



# === Calcolo residui ARIMA ===
residuals_arima = test['gas_price'].values - np.array(rolling_test_preds)

# === Ljung-Box Test ===
lags_to_test = [5, 10, 15]
ljung_results = acorr_ljungbox(residuals_arima, lags=lags_to_test, return_df=True)

for lag in lags_to_test:
    stat = ljung_results.loc[lag, 'lb_stat']
    pval = ljung_results.loc[lag, 'lb_pvalue']
    print(f"Lag {lag}: Ljung-Box Stat = {stat:.2f}, p-value = {pval:.4f}")

# === Plot dei Residui ===
plt.figure(figsize=(15, 4))
plt.plot(test.index, residuals_arima, color='gray')
plt.title('Residuals - ARIMA')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.grid(True)
plt.show()

# === Plot ACF dei Residui ===
plt.figure(figsize=(10, 4))
plot_acf(residuals_arima, lags=20)
plt.title('Autocorrelation Function - ARIMA Residuals')
plt.show()


# === QQ-PLOT dei residui ARIMA ===
plt.figure(figsize=(6, 6))
sm.qqplot(residuals_arima, line='s')
plt.title("QQ-Plot of ARIMA Residuals (Test Set)")
plt.grid(True)
plt.show()

# === Kolmogorov–Smirnov Test ===

# Standardize residuals before KS test
residuals_arima_std = (residuals_arima - np.mean(residuals_arima)) / np.std(residuals_arima)

ks_stat, ks_pvalue = stats.kstest(residuals_arima_std, 'norm')
print(f"KS statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")

    
# LSTM Model


# Data Normalization for LSTM
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit scaler on training data and transform both train and test
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

print(f"Original data range: {train.min().values[0]:.2f} to {train.max().values[0]:.2f}")
print(f"Scaled data range: {train_scaled.min():.2f} to {train_scaled.max():.2f}")

# Prepare data for LSTM
slot = 30
val_split_idx = int(len(train_scaled) * 0.85)
train_univ_scaled = train_scaled[:val_split_idx]
val_univ_scaled = train_scaled[val_split_idx - slot:]

x_train_univ, y_train_univ = [], []
for i in range(slot, len(train_univ_scaled)):
    x_train_univ.append(train_univ_scaled[i-slot:i, 0])
    y_train_univ.append(train_univ_scaled[i, 0])

x_train_univ = np.array(x_train_univ)
y_train_univ = np.array(y_train_univ)
x_train_univ = x_train_univ.reshape((x_train_univ.shape[0], slot, 1))

x_train = x_train_univ
y_train = y_train_univ

# Validation
x_val_univ, y_val_univ = [], []
for i in range(slot, len(val_univ_scaled)):
    x_val_univ.append(val_univ_scaled[i-slot:i, 0])
    y_val_univ.append(val_univ_scaled[i, 0])

x_val_univ = np.array(x_val_univ)
y_val_univ = np.array(y_val_univ)
x_val_univ = x_val_univ.reshape((x_val_univ.shape[0], slot, 1))

# index for a=validation
val_dates = train.index[val_split_idx:]
val_index = val_dates[slot:]

print(f"Training data shape: {x_train_univ.shape}, {y_train_univ.shape}")
print(f"Scaled y_train range: {y_train_univ.min():.3f} to {y_train_univ.max():.3f}")

# Build LSTM model
lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.layers.LSTM(units=96, input_shape=(slot, 1), return_sequences=True))
lstm_model.add(tf.keras.layers.Dropout(0.3))  # Add dropout to prevent overfitting
lstm_model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
lstm_model.add(tf.keras.layers.Dropout(0.3))
lstm_model.add(tf.keras.layers.LSTM(units=32, return_sequences=False))
lstm_model.add(tf.keras.layers.Dropout(0.3))
lstm_model.add(tf.keras.layers.Dense(units=1))

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.summary()

# Train LSTM model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

history = lstm_model.fit(
    x_train_univ, y_train_univ,
    validation_data=(x_val_univ, y_val_univ),
    epochs=100, batch_size=32, verbose=1, shuffle=False,
    callbacks=[early_stopping, reduce_lr]
)

# Model Evaluation - Train Data
yp_train_lstm = lstm_model.predict(x_train_univ)

# INVERSE TRANSFORM to original scale
yp_train_lstm_rescaled = scaler.inverse_transform(yp_train_lstm)
y_train_rescaled = scaler.inverse_transform(y_train_univ.reshape(-1, 1))

# Rebuilding dataframe
train_compare = pd.DataFrame({
    'gas_price': y_train_rescaled.flatten(),
    'gp_pred': yp_train_lstm_rescaled.flatten()
}, index=train.index[slot:val_split_idx])


print("Train comparison:")
print(train_compare.head(3))
print(train_compare.tail(3))

# Plot training results
plt.figure(figsize=(15, 5))
plt.plot(train_compare['gas_price'], color='red', label="Actual Natural Gas Price")
plt.plot(train_compare['gp_pred'], color='blue', label='Predicted Price')
plt.title("Natural Gas Price Prediction on Train Data")
plt.xlabel('Time')
plt.ylabel('Natural gas price')
plt.legend(loc='best')
plt.show()

# Test Data Evaluation
dataset_total_scaled = np.vstack((train_scaled, test_scaled))
inputs = dataset_total_scaled[len(dataset_total_scaled) - len(test_scaled) - slot:]

x_test = []
for i in range(slot, len(test_scaled) + slot):
    x_test.append(inputs[i-slot:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pred_price = lstm_model.predict(x_test)

# INVERSE TRANSFORM to original scale
pred_price_rescaled = scaler.inverse_transform(pred_price)

b = pd.DataFrame(pred_price_rescaled, columns=['gp_pred'])
b.index = test.index
test_compare = pd.concat([test, b], axis=1)

print("Test comparison:")
print(test_compare.head(3))
print(test_compare.tail(3))

# Plot test results
plt.figure(figsize=(15,5))
plt.plot(test_compare['gas_price'], color='red', label="Actual Natural Gas Price")
plt.plot(test_compare['gp_pred'], color='blue', label='Predicted Price')
plt.title("Natural Gas Price Prediction On Test Data")
plt.xlabel('Time')
plt.ylabel('Natural gas price')
plt.legend(loc='best')
plt.show()

# Calculate metrics
mse_train = mean_squared_error(train_compare['gas_price'], train_compare['gp_pred'])
mse_test = mean_squared_error(test_compare['gas_price'], test_compare['gp_pred'])
rmse_train = sqrt(mse_train)
rmse_test = sqrt(mse_test)
r2_train = r2_score(train_compare['gas_price'], train_compare['gp_pred'])
r2_test = r2_score(test_compare['gas_price'], test_compare['gp_pred'])

print("LSTM Results:")
print("Train Data:\nMSE: {:.6f}, RMSE: {:.4f}, R²: {:.4f}".format(mse_train, rmse_train, r2_train))
print("Test Data:\nMSE: {:.6f}, RMSE: {:.4f}, R²: {:.4f}".format(mse_test, rmse_test, r2_test))

# Plot training vs validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Univariate - Loss vs Val Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# === Calcolo dei residui ===
residuals_test = test_compare['gas_price'].values - test_compare['gp_pred'].values

# === ACF dei residui ===
plt.figure(figsize=(10, 5))
plot_acf(residuals_test, lags=15)
plt.title("Autocorrelation of LSTM Residuals (Test Set)")
plt.show()

# === Ljung-Box Test ===
print("\n=== Ljung-Box Test on LSTM Residuals (Test Set) ===")
lags_to_test = [5, 10, 15]
ljung_results = acorr_ljungbox(residuals_test, lags=lags_to_test, return_df=True)

for lag in lags_to_test:
    stat = ljung_results.loc[lag, 'lb_stat']
    pval = ljung_results.loc[lag, 'lb_pvalue']
    print(f"Lag {lag}: Ljung-Box Stat = {stat:.2f}, p-value = {pval:.4f}")
    
    
import scipy.stats as stats

# === Standardizzazione dei residui ===
residuals_std = (residuals_test - np.mean(residuals_test)) / np.std(residuals_test)

# === QQ-Plot ===
plt.figure(figsize=(8, 6))
stats.probplot(residuals_std, dist="norm", plot=plt)
plt.title("QQ-Plot of LSTM Residuals (Test Set)")
plt.grid(True)
plt.show()

# === Kolmogorov–Smirnov Test ===
ks_stat, ks_pval = stats.kstest(residuals_std, 'norm')
print("\n=== Kolmogorov–Smirnov Test on LSTM Residuals ===")
print(f"KS Statistic = {ks_stat:.4f}")
print(f"p-value      = {ks_pval:.4f}")


# CNN-LSTM-2 Ensemble Model

# Build CNN-LSTM-2 Ensemble model
cnn_model = tf.keras.Sequential()

# CNN layers for pattern detection and feature extraction
cnn_model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='tanh', input_shape=(slot, 1)))
cnn_model.add(tf.keras.layers.Dropout(0.3))
cnn_model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='tanh'))
cnn_model.add(tf.keras.layers.Dropout(0.3))

# LSTM for time sequence
cnn_model.add(tf.keras.layers.LSTM(units=70, return_sequences=False))
cnn_model.add(tf.keras.layers.Dropout(0.3))
cnn_model.add(tf.keras.layers.Dense(units=16))
cnn_model.add(tf.keras.layers.Dropout(0.3))
cnn_model.add(tf.keras.layers.Dense(units=1))


cnn_model.compile(loss='mean_squared_error', optimizer='adam')
cnn_model.summary()

# Train CNN-LSTM-2 Ensemble model
early_stopping_cnn = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr_cnn = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

history_cnn = cnn_model.fit(
    x_train_univ, y_train_univ,
    validation_data=(x_val_univ, y_val_univ),
    epochs=100, batch_size=32, verbose=1, shuffle=False,
    callbacks=[early_stopping_cnn, reduce_lr_cnn]
)

# Model Evaluation - Train Data 
print("\n=== CNN-LSTM-2 Ensemble Model EVALUATION ===")
yp_train_cnn = cnn_model.predict(x_train_univ)

# INVERSE TRANSFORM to original scale
yp_train_cnn_rescaled = scaler.inverse_transform(yp_train_cnn)
y_train_rescaled = scaler.inverse_transform(y_train_univ.reshape(-1, 1))

train_index_cnn = train.index[slot:slot + len(y_train_rescaled)]

train_compare_cnn = pd.DataFrame({
    'gas_price': y_train_rescaled.flatten(),
    'gp_pred_cnn': yp_train_cnn_rescaled.flatten()
}, index=train_index_cnn)
print("Train comparison (CNN-LSTM-2):")
print(train_compare_cnn.head(3))
print(train_compare_cnn.tail(3))

# Plot training results (CNN-LSTM-2 Ensemble)
plt.figure(figsize=(15, 5))
plt.plot(train_compare_cnn['gas_price'], color='red', label="Actual Natural Gas Price")
plt.plot(train_compare_cnn['gp_pred_cnn'], color='orange', label='Predicted Price (CNN)')
plt.title("Natural Gas Price Prediction on Train Data (CNN-Only)")
plt.xlabel('Time')
plt.ylabel('Natural gas price')
plt.legend(loc='best')
plt.show()

# Test Data Evaluation (CNN-LSTM-2 Ensemble)
dataset_total_scaled = np.vstack((train_scaled, test_scaled))
inputs = dataset_total_scaled[len(dataset_total_scaled) - len(test_scaled) - slot:]
x_test = []
for i in range(slot, len(test_scaled) + slot):
    x_test.append(inputs[i-slot:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
pred_price_cnn = cnn_model.predict(x_test)

# INVERSE TRANSFORM to original scale
pred_price_cnn_rescaled = scaler.inverse_transform(pred_price_cnn)

b_cnn = pd.DataFrame(pred_price_cnn_rescaled, columns=['gp_pred_cnn'])
b_cnn.index = test.index
test_compare_cnn = pd.concat([test, b_cnn], axis=1)

print("Test comparison (CNN-LSTM-2 Ensemble):")
print(test_compare_cnn.head(3))
print(test_compare_cnn.tail(3))

# Plot test results (CNN-LSTM-2 Ensemble)
plt.figure(figsize=(15,5))
plt.plot(test_compare_cnn['gas_price'], color='red', label="Actual Natural Gas Price")
plt.plot(test_compare_cnn['gp_pred_cnn'], color='orange', label='Predicted Price (CNN-LSTM-2)')
plt.title("Natural Gas Price Prediction On Test Data (CNN-LSTM-2)")
plt.xlabel('Time')
plt.ylabel('Natural gas price')
plt.legend(loc='best')
plt.show()

# Calculate metrics (CNN-LSTM-2 Ensemble)
mse_train_cnn = mean_squared_error(train_compare_cnn['gas_price'], train_compare_cnn['gp_pred_cnn'])
mse_test_cnn = mean_squared_error(test_compare_cnn['gas_price'], test_compare_cnn['gp_pred_cnn'])
rmse_train_cnn = sqrt(mse_train_cnn)
rmse_test_cnn = sqrt(mse_test_cnn)
r2_train_cnn = r2_score(train_compare_cnn['gas_price'], train_compare_cnn['gp_pred_cnn'])
r2_test_cnn = r2_score(test_compare_cnn['gas_price'], test_compare_cnn['gp_pred_cnn'])

print("CNN-LSTM-2 Results:")
print("Train Data:\nMSE: {:.6f}, RMSE: {:.4f}, R²: {:.4f}".format(mse_train_cnn, rmse_train_cnn, r2_train_cnn))
print("Test Data:\nMSE: {:.6f}, RMSE: {:.4f}, R²: {:.4f}".format(mse_test_cnn, rmse_test_cnn, r2_test_cnn))

# Plot training vs validation loss (CNN-LSTM-2 Ensemble)
plt.figure(figsize=(10, 6))
plt.plot(history_cnn.history['loss'], label='Train Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.title('CNN-LSTM-2 - Loss vs Val Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()



# === Calcolo residui ===
residuals_cnn = test_compare_cnn['gas_price'] - test_compare_cnn['gp_pred_cnn']

# === Ljung-Box Test ===
print("\n=== Ljung-Box Test on CNN-LSTM-2 Residuals (Test Set) ===")
lags_to_test = [5, 10, 15]
ljung_results = acorr_ljungbox(residuals_cnn, lags=lags_to_test, return_df=True)

for lag in lags_to_test:
    stat = ljung_results.loc[lag, 'lb_stat']
    pval = ljung_results.loc[lag, 'lb_pvalue']
    print(f"Lag {lag}: Ljung-Box Stat = {stat:.2f}, p-value = {pval:.4f}")

# === Plot dei Residui ===
plt.figure(figsize=(15, 4))
plt.plot(test_compare_cnn.index, residuals_cnn, color='gray')
plt.title('Residuals - CNN-LSTM-2 Ensemble')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.grid(True)
plt.show()

# === Plot ACF dei Residui ===
plt.figure(figsize=(10, 4))
plot_acf(residuals_cnn, lags=20)
plt.title('Autocorrelation Function - Residuals CNN-LSTM-2')
plt.show()


# === Standardizzazione dei residui CNN-LSTM-2 ===
residuals_cnn_std = (residuals_cnn - np.mean(residuals_cnn)) / np.std(residuals_cnn)

# === QQ-Plot dei residui ===
plt.figure(figsize=(8, 6))
stats.probplot(residuals_cnn_std, dist="norm", plot=plt)
plt.title("QQ-Plot of CNN-LSTM-2 Residuals (Test Set)")
plt.grid(True)
plt.show()

# === Kolmogorov–Smirnov Test ===
ks_stat_cnn, ks_pval_cnn = stats.kstest(residuals_cnn_std, 'norm')
print("\n=== Kolmogorov–Smirnov Test on CNN-LSTM-2 Residuals ===")
print(f"KS Statistic = {ks_stat_cnn:.4f}")
print(f"p-value      = {ks_pval_cnn:.4f}")


# ===== EXOGENOUS FEATURES TESTING =====
print("\n=== EXOGENOUS FEATURES TESTING ===")

# ===== ADDED: Function to reset random seeds =====
def reset_random_seeds():
    """Reset all random seeds for reproducibility"""
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Additional TensorFlow reproducibility settings
    tf.config.experimental.enable_op_determinism()

# Reset seeds at the beginning
reset_random_seeds()

# Load exogenous data
exog_data = pd.read_csv("data.csv")
exog_data['Date'] = pd.to_datetime(exog_data['Date'], format="%d/%m/%Y")
exog_data = exog_data.set_index('Date').sort_index()

# Select key exogenous variables
exog_features = ['VIX_USD', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'Brent_Fut_Price_USD', 'Coal_Fut_Price_USD', 'HenryHub_Fut_FrontMonth_USD', 'NBPgas_Fut_FrontMonth_GBP'
                 ]

# Merge with original data
merged_data = pd.merge(data, exog_data[exog_features], left_index=True, right_index=True, how='inner')

# ===== CORREZIONE: Split PRIMA di scaling =====
# Prepare multivariate data - SPLIT FIRST
train_merged = merged_data[merged_data.index <= '2023-12-31']
test_merged = merged_data[merged_data.index >= '2024-01-01']

print(f"Train merged shape: {train_merged.shape}")
print(f"Test merged shape: {test_merged.shape}")

# ===== CORREZIONE: Single consistent scaler =====
# Scale all features with ONE scaler fitted on train only
from sklearn.preprocessing import MinMaxScaler
scaler_multi = MinMaxScaler(feature_range=(0, 1))
train_scaled_multi = scaler_multi.fit_transform(train_merged.values)
test_scaled_multi = scaler_multi.transform(test_merged.values)

print(f"Scaler fitted on train data range: [{train_merged.min().min():.3f}, {train_merged.max().max():.3f}]")
print(f"Train scaled range: [{train_scaled_multi.min():.3f}, {train_scaled_multi.max():.3f}]")
print(f"Test scaled range: [{test_scaled_multi.min():.3f}, {test_scaled_multi.max():.3f}]")

# Prepare sequences for multivariate models
slot = 7
n_features = train_scaled_multi.shape[1]

# ===== MULTIVARIATE TRAIN / VALIDATION SPLIT =====
val_split_date = '2023-10-31'
val_split_idx = train_merged.index <= val_split_date

# Use the SAME scaler for consistency
train_scaled_partial_multi = train_scaled_multi[val_split_idx]
val_scaled_multi = train_scaled_multi[~val_split_idx]

print(f"Training samples: {len(train_scaled_partial_multi)}")
print(f"Validation samples: {len(val_scaled_multi)}")

# Multivariate sequences function
def create_multivariate_sequences(data, slot):
    x, y = [], []
    for i in range(slot, len(data)):
        x.append(data[i-slot:i, :])
        y.append(data[i, 0])  # Gas price è la prima colonna
    return np.array(x), np.array(y)

x_train_multi, y_train_multi = create_multivariate_sequences(train_scaled_partial_multi, slot)
x_val_multi, y_val_multi = create_multivariate_sequences(val_scaled_multi, slot)

print(f"x_train_multi: {x_train_multi.shape}, x_val_multi: {x_val_multi.shape}")

# === CORRELATION MATRIX ===
corr_matrix = merged_data.corr()

# numerical values
print("\nCorrelation matrix (top 10 gas_price correlations):")
print(corr_matrix['gas_price'].sort_values(ascending=False).head(10))

# heatmap 
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix - Gas Price & Exogenous Features")
plt.show()

# ===== IMPROVED: Model building with seed reset =====

# Reset seeds before building each model
reset_random_seeds()

# Multivariate LSTM
lstm_multi = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=96, input_shape=(slot, n_features), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1)
])
lstm_multi.compile(loss='mean_squared_error', optimizer='adam')

# Reset seeds before building next model
reset_random_seeds()

# Multivariate CNN-LSTM Ensemble
cnn_multi = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='tanh', input_shape=(slot, n_features), padding='same'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='tanh', padding='same'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=70, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=16),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1)
])
cnn_multi.compile(loss='mean_squared_error', optimizer='adam')


# Train multivariate models

# ===== IMPROVED: Deterministic callbacks =====
# Use deterministic callbacks for reproducibility
early_stopping_multi = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1  # Added verbose for better tracking
)
reduce_lr_multi = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=10, 
    min_lr=1e-6,
    verbose=1  # Added verbose for better tracking
)

# Reset seeds before training each model
reset_random_seeds()
lstm_multi.fit(
    x_train_multi, y_train_multi,
    validation_data=(x_val_multi, y_val_multi),
    epochs=100, batch_size=32, verbose=0,
    callbacks=[early_stopping_multi, reduce_lr_multi]
)

reset_random_seeds()
cnn_multi.fit(
    x_train_multi, y_train_multi,
    validation_data=(x_val_multi, y_val_multi),
    epochs=100, batch_size=32, verbose=0,
    callbacks=[early_stopping_multi, reduce_lr_multi]
)




# Concatenate train and test for proper windowing
dataset_total_multi = np.vstack((train_scaled_multi, test_scaled_multi))
print(f"Total dataset shape: {dataset_total_multi.shape}")

# Create test sequences with proper indexing
inputs_multi = dataset_total_multi[len(train_scaled_multi) - slot:]
print(f"Input buffer shape: {inputs_multi.shape}")

x_test_multi = []
for i in range(slot, len(inputs_multi)):
    x_test_multi.append(inputs_multi[i-slot:i, :])

x_test_multi = np.array(x_test_multi)
print(f"Test sequences shape: {x_test_multi.shape}")


# Evaluate multivariate models
def evaluate_multivariate_model(model, x_train, y_train, x_test, model_name):
    from math import sqrt
    
    # Training predictions
    y_pred_train = model.predict(x_train, verbose=0).flatten()
    r2_train = r2_score(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = sqrt(mse_train)

    # Test predictions
    y_pred_test = model.predict(x_test, verbose=0).flatten()
    
    # ===== CORREZIONE: Align test targets with test sequences =====
    y_test_scaled = test_scaled_multi[:len(x_test), 0]  # Take only gas_price column and align length
    
   
    print(f"  x_test shape: {x_test.shape}")
    print(f"  y_pred_test shape: {y_pred_test.shape}")
    print(f"  y_test_scaled shape: {y_test_scaled.shape}")
    print(f"  Lengths match: {len(y_pred_test) == len(y_test_scaled)}")
    
    # Calculate metrics
    r2_test = r2_score(y_test_scaled, y_pred_test)
    mse_test = mean_squared_error(y_test_scaled, y_pred_test)
    rmse_test = sqrt(mse_test)

    print(f"\n{model_name} (Multivariate):")
    print(f"Train - R²: {r2_train:.4f}, MSE: {mse_train:.6f}, RMSE: {rmse_train:.4f}")
    print(f"Test  - R²: {r2_test:.4f}, MSE: {mse_test:.6f}, RMSE: {rmse_test:.4f}")

    return {
        'train_r2': r2_train, 'train_mse': mse_train, 'train_rmse': rmse_train,
        'test_r2': r2_test, 'test_mse': mse_test, 'test_rmse': rmse_test,
        'test_predictions': y_pred_test,
        'test_true': y_test_scaled
    }

# Evaluate all multivariate models
results_lstm_multi = evaluate_multivariate_model(lstm_multi, x_train_multi, y_train_multi, x_test_multi, "LSTM")
results_cnn_multi = evaluate_multivariate_model(cnn_multi, x_train_multi, y_train_multi, x_test_multi, "CNN-LSTM")


# === RESCALE MULTIVARIATE PREDICTIONS TO ORIGINAL SCALE ===

def inverse_multivariate_predictions(y_pred_scaled, y_true_scaled, scaler, feature_index=0):
    n_samples = len(y_pred_scaled)
    n_features = scaler.n_features_in_
    
    # For true values
    dummy_true = np.zeros((n_samples, n_features))
    dummy_true[:, feature_index] = y_true_scaled
    y_true_rescaled = scaler.inverse_transform(dummy_true)[:, feature_index]
    
    # For predictions
    dummy_pred = np.zeros((n_samples, n_features))
    dummy_pred[:, feature_index] = y_pred_scaled
    y_pred_rescaled = scaler.inverse_transform(dummy_pred)[:, feature_index]
    
    return y_true_rescaled, y_pred_rescaled

# ===== LSTM MULTIVARIATE =====
# Train metrics rescaled
y_pred_lstm_train_scaled = lstm_multi.predict(x_train_multi, verbose=0).flatten()
y_train_lstm_scaled = y_train_multi
y_true_lstm_train, y_pred_lstm_train = inverse_multivariate_predictions(
    y_pred_lstm_train_scaled, y_train_lstm_scaled, scaler_multi)
mse_train_lstm_rescaled = mean_squared_error(y_true_lstm_train, y_pred_lstm_train)
rmse_train_lstm_rescaled = sqrt(mse_train_lstm_rescaled)
r2_train_lstm_rescaled = r2_score(y_true_lstm_train, y_pred_lstm_train)

# Test metrics rescaled
y_pred_lstm_scaled = results_lstm_multi['test_predictions']
y_test_scaled_multi = results_lstm_multi['test_true']
y_true_lstm, y_pred_lstm = inverse_multivariate_predictions(
    y_pred_lstm_scaled, y_test_scaled_multi, scaler_multi)
mse_test_lstm_rescaled = mean_squared_error(y_true_lstm, y_pred_lstm)
rmse_test_lstm_rescaled = sqrt(mse_test_lstm_rescaled)
r2_test_lstm_rescaled = r2_score(y_true_lstm, y_pred_lstm)

# ===== CNN-LSTM ENSEMBLE =====
# Train metrics rescaled
y_pred_cnn_train_scaled = cnn_multi.predict(x_train_multi, verbose=0).flatten()
y_train_cnn_scaled = y_train_multi
y_true_cnn_train, y_pred_cnn_train = inverse_multivariate_predictions(
    y_pred_cnn_train_scaled, y_train_cnn_scaled, scaler_multi)
mse_train_cnn_rescaled = mean_squared_error(y_true_cnn_train, y_pred_cnn_train)
rmse_train_cnn_rescaled = sqrt(mse_train_cnn_rescaled)
r2_train_cnn_rescaled = r2_score(y_true_cnn_train, y_pred_cnn_train)

# Test metrics rescaled
y_pred_cnn_scaled = results_cnn_multi['test_predictions']
y_true_cnn, y_pred_cnn = inverse_multivariate_predictions(
    y_pred_cnn_scaled, y_test_scaled_multi, scaler_multi)
mse_test_cnn_rescaled = mean_squared_error(y_true_cnn, y_pred_cnn)
rmse_test_cnn_rescaled = sqrt(mse_test_cnn_rescaled)
r2_test_cnn_rescaled = r2_score(y_true_cnn, y_pred_cnn)


# Print metrics
print("LSTM Multi (rescaled):")
print(f"  Train - R²: {r2_train_lstm_rescaled:.4f}, MSE: {mse_train_lstm_rescaled:.4f}, RMSE: {rmse_train_lstm_rescaled:.4f}")
print(f"  Test  - R²: {r2_test_lstm_rescaled:.4f}, MSE: {mse_test_lstm_rescaled:.4f}, RMSE: {rmse_test_lstm_rescaled:.4f}")

print("CNN-LSTM Ensemble Multi (rescaled):")
print(f"  Train - R²: {r2_train_cnn_rescaled:.4f}, MSE: {mse_train_cnn_rescaled:.4f}, RMSE: {rmse_train_cnn_rescaled:.4f}")
print(f"  Test  - R²: {r2_test_cnn_rescaled:.4f}, MSE: {mse_test_cnn_rescaled:.4f}, RMSE: {rmse_test_cnn_rescaled:.4f}")


# Residual analysis
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

# ===== LSTM MULTIVARIATE =====
residuals_lstm_multi = y_true_lstm - y_pred_lstm

lags_to_test = [5, 10, 15]
ljung_results = acorr_ljungbox(residuals_lstm_multi, lags=lags_to_test, return_df=True)

for lag in lags_to_test:
    stat = ljung_results.loc[lag, 'lb_stat']
    pval = ljung_results.loc[lag, 'lb_pvalue']
    print(f"Lag {lag}: Ljung-Box Stat = {stat:.2f}, p-value = {pval:.4f}")

plt.figure(figsize=(8, 5))
plot_acf(residuals_lstm_multi, lags=20)
plt.title("Autocorrelation Function - LSTM Multivariate Residuals")
plt.show()

# ===== CNN-LSTM MULTIVARIATE =====
residuals_cnn_multi = y_true_cnn - y_pred_cnn

ljung_results = acorr_ljungbox(residuals_cnn_multi, lags=lags_to_test, return_df=True)

for lag in lags_to_test:
    stat = ljung_results.loc[lag, 'lb_stat']
    pval = ljung_results.loc[lag, 'lb_pvalue']
    print(f"Lag {lag}: Ljung-Box Stat = {stat:.2f}, p-value = {pval:.4f}")

plt.figure(figsize=(8, 5))
plot_acf(residuals_cnn_multi, lags=20)
plt.title("Autocorrelation Function - CNN-LSTM Multivariate Residuals")
plt.show()

import scipy.stats as stats

# === Funzione di utilità per QQ-Plot e KS test ===
def qq_ks_test(residuals, model_name):
    residuals_std = (residuals - np.mean(residuals)) / np.std(residuals)
    
    # QQ-Plot
    plt.figure(figsize=(6, 5))
    stats.probplot(residuals_std, dist="norm", plot=plt)
    plt.title(f"QQ-Plot - Residuals ({model_name})")
    plt.grid(True)
    plt.show()
    
    # KS Test
    ks_stat, ks_pval = stats.kstest(residuals_std, 'norm')
    print(f"\n=== KS Test - Residuals ({model_name}) ===")
    print(f"KS Statistic = {ks_stat:.4f}")
    print(f"p-value      = {ks_pval:.4f}")
    
    
# === LSTM MULTIVARIATE ===
qq_ks_test(residuals_lstm_multi, "LSTM Multivariate")

# === CNN-LSTM MULTIVARIATE ===
qq_ks_test(residuals_cnn_multi, "CNN-LSTM Multivariate")



# ===== CNN-LSTM REGIME CLASSIFIER (SOLO CLASSIFICAZIONE) =====

# Reset seeds for reproducibility
reset_random_seeds()

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# 1. features preparation
full_data_classifier = data.copy()
full_data_classifier['log_return'] = np.log(full_data_classifier['gas_price']).diff()
full_data_classifier['vol_30'] = full_data_classifier['log_return'].rolling(window=30).std()
momentum = full_data_classifier['log_return'].rolling(window=15).sum()
vol = full_data_classifier['log_return'].rolling(window=15).std()
full_data_classifier['trend_strength'] = momentum / (vol + 1e-8)

# Add features
full_data_classifier['price_change'] = full_data_classifier['gas_price'].pct_change()
full_data_classifier['vol_5'] = full_data_classifier['log_return'].rolling(window=5).std()
full_data_classifier['vol_60'] = full_data_classifier['log_return'].rolling(window=60).std()
full_data_classifier['momentum_5'] = full_data_classifier['log_return'].rolling(window=5).sum()
full_data_classifier['momentum_30'] = full_data_classifier['log_return'].rolling(window=30).sum()

# 2. thresholds calculation
full_data_clean_classifier = full_data_classifier.dropna()
global_vol_q33, global_vol_q66 = full_data_clean_classifier['vol_30'].quantile([0.33, 0.66])
global_dir_q25, global_dir_q75 = full_data_clean_classifier['trend_strength'].quantile([0.25, 0.75])

print(f"Global volatility thresholds: Low < {global_vol_q33:.4f} < Medium < {global_vol_q66:.4f} < High")
print(f"Global direction thresholds: Bear < {global_dir_q25:.4f} < Neutral < {global_dir_q75:.4f} < Bull")

# classification
full_data_clean_classifier['vol_regime'] = 'Medium'
full_data_clean_classifier['dir_regime'] = 'Neutral'

full_data_clean_classifier.loc[full_data_clean_classifier['vol_30'] < global_vol_q33, 'vol_regime'] = 'Low'
full_data_clean_classifier.loc[full_data_clean_classifier['vol_30'] > global_vol_q66, 'vol_regime'] = 'High'
full_data_clean_classifier.loc[full_data_clean_classifier['trend_strength'] < global_dir_q25, 'dir_regime'] = 'Bear'
full_data_clean_classifier.loc[full_data_clean_classifier['trend_strength'] > global_dir_q75, 'dir_regime'] = 'Bull'

# 3. Split train/test
train_classifier = full_data_clean_classifier[full_data_clean_classifier.index <= '2023-12-31']
test_classifier = full_data_clean_classifier[full_data_clean_classifier.index >= '2024-01-01']

print(f"Train regime distribution (Vol): {train_classifier['vol_regime'].value_counts()}")
print(f"Train regime distribution (Dir): {train_classifier['dir_regime'].value_counts()}")
print(f"Test regime distribution (Vol): {test_classifier['vol_regime'].value_counts()}")
print(f"Test regime distribution (Dir): {test_classifier['dir_regime'].value_counts()}")

# 4. sequences
feature_cols_classifier = ['gas_price', 'vol_30', 'trend_strength', 'price_change', 'vol_5', 'vol_60', 'momentum_5', 'momentum_30']
scaler_classifier = MinMaxScaler(feature_range=(0, 1))

train_features_classifier = scaler_classifier.fit_transform(train_classifier[feature_cols_classifier].values)
test_features_classifier = scaler_classifier.transform(test_classifier[feature_cols_classifier].values)

slot_classifier = 5  

def create_classification_sequences(features, vol_labels, dir_labels, slot):
   x, y_vol, y_dir = [], [], []
   for i in range(slot, len(features)):
       x.append(features[i-slot:i, :])
       y_vol.append(vol_labels.iloc[i])
       y_dir.append(dir_labels.iloc[i])
   return np.array(x), y_vol, y_dir

# Training sequences
x_train_classifier, y_vol_train_raw, y_dir_train_raw = create_classification_sequences(
   train_features_classifier, train_classifier['vol_regime'], 
   train_classifier['dir_regime'], slot_classifier)

# Test sequences
x_test_classifier, y_vol_test_raw, y_dir_test_raw = create_classification_sequences(
   test_features_classifier, test_classifier['vol_regime'], 
   test_classifier['dir_regime'], slot_classifier)

# 5. Encode labels 
vol_encoder_classifier = LabelEncoder()
dir_encoder_classifier = LabelEncoder()

y_vol_train_encoded = vol_encoder_classifier.fit_transform(y_vol_train_raw)
y_dir_train_encoded = dir_encoder_classifier.fit_transform(y_dir_train_raw)
y_vol_test_encoded = vol_encoder_classifier.transform(y_vol_test_raw)
y_dir_test_encoded = dir_encoder_classifier.transform(y_dir_test_raw)

from tensorflow.keras.utils import to_categorical
y_vol_train_cat = to_categorical(y_vol_train_encoded)
y_dir_train_cat = to_categorical(y_dir_train_encoded)

print(f"Volatility classes: {vol_encoder_classifier.classes_}")
print(f"Direction classes: {dir_encoder_classifier.classes_}")
print(f"Training sequences shape: {x_train_classifier.shape}")

# 6. Calculate class weights
vol_class_weights = compute_class_weight('balanced', classes=np.unique(y_vol_train_encoded), y=y_vol_train_encoded)
dir_class_weights = compute_class_weight('balanced', classes=np.unique(y_dir_train_encoded), y=y_dir_train_encoded)

vol_class_weight_dict = {i: vol_class_weights[i] for i in range(len(vol_class_weights))}
dir_class_weight_dict = {i: dir_class_weights[i] for i in range(len(dir_class_weights))}

print(f"Vol class weights: {vol_class_weight_dict}")
print(f"Dir class weights: {dir_class_weight_dict}")

# Evlauation function
def evaluate_classifier(model, x_train, y_train_encoded, x_test, y_test_encoded, encoder, class_name):
   # Training predictions
   y_train_pred = model.predict(x_train, verbose=0)
   y_train_pred_classes = np.argmax(y_train_pred, axis=1)
   
   # Test predictions
   y_test_pred = model.predict(x_test, verbose=0)
   y_test_pred_classes = np.argmax(y_test_pred, axis=1)
   
   # Training metrics
   train_accuracy = accuracy_score(y_train_encoded, y_train_pred_classes)
   train_f1 = f1_score(y_train_encoded, y_train_pred_classes, average='weighted')
   train_recall = recall_score(y_train_encoded, y_train_pred_classes, average='weighted')
   train_balanced_accuracy = balanced_accuracy_score(y_train_encoded, y_train_pred_classes)
   
   # Test metrics
   test_accuracy = accuracy_score(y_test_encoded, y_test_pred_classes)
   test_f1 = f1_score(y_test_encoded, y_test_pred_classes, average='weighted')
   test_recall = recall_score(y_test_encoded, y_test_pred_classes, average='weighted')
   test_balanced_accuracy = balanced_accuracy_score(y_test_encoded, y_test_pred_classes)
   
   print(f"\n{class_name.upper()} RESULTS:")
   print(f"TRAIN - Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Recall: {train_recall:.4f}, Balanced Accuracy: {train_balanced_accuracy:.4f}")
   print(f"TEST  - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, Recall: {test_recall:.4f}, Balanced Accuracy: {test_balanced_accuracy:.4f}")
   
   return (y_train_pred_classes, y_test_pred_classes, test_balanced_accuracy)

# 7. Build classifiers 
def build_lstm_classifier(input_shape, num_classes):

   reset_random_seeds()
   model = tf.keras.Sequential([
       tf.keras.layers.LSTM(units=96, input_shape=input_shape, return_sequences=True),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.LSTM(units=64, return_sequences=True),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.LSTM(units=32, return_sequences=False),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(units=num_classes, activation='softmax')
   ])
   model.compile(
       optimizer='adam',
       loss='categorical_crossentropy', 
       metrics=['accuracy']
   )
   return model

def build_cnn_lstm_classifier(input_shape, num_classes):
   
   reset_random_seeds()
   model = tf.keras.Sequential([
       tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),  
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),  
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.LSTM(units=70, return_sequences=False),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(units=16, activation='relu'), 
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(units=num_classes, activation='softmax')
   ])
   model.compile(
       optimizer='adam',
       loss='categorical_crossentropy', 
       metrics=['accuracy']
   )
   return model

# 8. Time-based validation split function
def create_time_based_validation_split(x_data, y_data, val_split_ratio=0.15):
    
    split_idx = int(len(x_data) * (1 - val_split_ratio))
    
    x_train_split = x_data[:split_idx]
    y_train_split = y_data[:split_idx]
    x_val_split = x_data[split_idx:]
    y_val_split = y_data[split_idx:]
    
    return x_train_split, y_train_split, x_val_split, y_val_split

# Build models for univariate classification
reset_random_seeds()
vol_lstm_classifier = build_lstm_classifier((slot_classifier, len(feature_cols_classifier)), len(vol_encoder_classifier.classes_))
reset_random_seeds()
dir_lstm_classifier = build_lstm_classifier((slot_classifier, len(feature_cols_classifier)), len(dir_encoder_classifier.classes_))
reset_random_seeds()
vol_cnn_classifier = build_cnn_lstm_classifier((slot_classifier, len(feature_cols_classifier)), len(vol_encoder_classifier.classes_))
reset_random_seeds()
dir_cnn_classifier = build_cnn_lstm_classifier((slot_classifier, len(feature_cols_classifier)), len(dir_encoder_classifier.classes_))

# 9. Train classifiers with time-based validation
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1)


# Time-based validation splits
x_train_vol, y_train_vol, x_val_vol, y_val_vol = create_time_based_validation_split(x_train_classifier, y_vol_train_cat)
x_train_dir, y_train_dir, x_val_dir, y_val_dir = create_time_based_validation_split(x_train_classifier, y_dir_train_cat)


reset_random_seeds()
vol_lstm_history = vol_lstm_classifier.fit(
   x_train_vol, y_train_vol, 
   validation_data=(x_val_vol, y_val_vol),
   epochs=100, batch_size=32, verbose=0, 
   class_weight=vol_class_weight_dict,
   callbacks=[early_stopping, reduce_lr]
)


reset_random_seeds()
dir_lstm_history = dir_lstm_classifier.fit(
   x_train_dir, y_train_dir,
   validation_data=(x_val_dir, y_val_dir), 
   epochs=100, batch_size=32, verbose=0, 
   class_weight=dir_class_weight_dict,
   callbacks=[early_stopping, reduce_lr]
)


reset_random_seeds()
vol_cnn_history = vol_cnn_classifier.fit(
   x_train_vol, y_train_vol,
   validation_data=(x_val_vol, y_val_vol), 
   epochs=100, batch_size=64, verbose=0, 
   class_weight=vol_class_weight_dict,
   callbacks=[early_stopping, reduce_lr]
)


reset_random_seeds()
dir_cnn_history = dir_cnn_classifier.fit(
   x_train_dir, y_train_dir,
   validation_data=(x_val_dir, y_val_dir), 
   epochs=100, batch_size=64, verbose=0, 
   class_weight=dir_class_weight_dict,
   callbacks=[early_stopping, reduce_lr]
)

# Evaluate univariate classifiers

vol_lstm_train_pred, vol_lstm_test_pred, vol_lstm_test_acc = evaluate_classifier(
    vol_lstm_classifier, x_train_classifier, y_vol_train_encoded, x_test_classifier, y_vol_test_encoded, vol_encoder_classifier, 
)

dir_lstm_train_pred, dir_lstm_test_pred, dir_lstm_test_acc = evaluate_classifier(
    dir_lstm_classifier, x_train_classifier, y_dir_train_encoded, x_test_classifier, y_dir_test_encoded, dir_encoder_classifier,
)

vol_cnn_train_pred, vol_cnn_test_pred, vol_cnn_test_acc = evaluate_classifier(
    vol_cnn_classifier, x_train_classifier, y_vol_train_encoded, x_test_classifier, y_vol_test_encoded, vol_encoder_classifier, 
)

dir_cnn_train_pred, dir_cnn_test_pred, dir_cnn_test_acc = evaluate_classifier(
    dir_cnn_classifier, x_train_classifier, y_dir_train_encoded, x_test_classifier, y_dir_test_encoded, dir_encoder_classifier, 
)

# ===== MULTIVARIATE CLASSIFICATION WITH EXOGENOUS FEATURES =====


# Reset seeds for reproducibility
reset_random_seeds()

# Load exogenous data
exog_data = pd.read_csv("data.csv")
exog_data['Date'] = pd.to_datetime(exog_data['Date'], format="%d/%m/%Y")
exog_data = exog_data.set_index('Date').sort_index()

# Select key exogenous variables
exog_features = ['VIX_USD', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'Brent_Fut_Price_USD', 'Coal_Fut_Price_USD', 'HenryHub_Fut_FrontMonth_USD', 'NBPgas_Fut_FrontMonth_GBP']

# Merge with classification data
merged_classifier_data = pd.merge(full_data_clean_classifier, exog_data[exog_features], left_index=True, right_index=True, how='inner')


# Split multivariate data
train_classifier_multi = merged_classifier_data[merged_classifier_data.index <= '2023-12-31']
test_classifier_multi = merged_classifier_data[merged_classifier_data.index >= '2024-01-01']


# Prepare multivariate features
feature_cols_classifier_multi = feature_cols_classifier + exog_features
scaler_classifier_multi = MinMaxScaler(feature_range=(0, 1))

train_features_classifier_multi = scaler_classifier_multi.fit_transform(train_classifier_multi[feature_cols_classifier_multi].values)
test_features_classifier_multi = scaler_classifier_multi.transform(test_classifier_multi[feature_cols_classifier_multi].values)

# Create multivariate sequences
x_train_classifier_multi, y_vol_train_raw_multi, y_dir_train_raw_multi = create_classification_sequences(
   train_features_classifier_multi, train_classifier_multi['vol_regime'], 
   train_classifier_multi['dir_regime'], slot_classifier)

x_test_classifier_multi, y_vol_test_raw_multi, y_dir_test_raw_multi = create_classification_sequences(
   test_features_classifier_multi, test_classifier_multi['vol_regime'], 
   test_classifier_multi['dir_regime'], slot_classifier)

# Encode labels for multivariate
vol_encoder_classifier_multi = LabelEncoder()
dir_encoder_classifier_multi = LabelEncoder()

y_vol_train_encoded_multi = vol_encoder_classifier_multi.fit_transform(y_vol_train_raw_multi)
y_dir_train_encoded_multi = dir_encoder_classifier_multi.fit_transform(y_dir_train_raw_multi)
y_vol_test_encoded_multi = vol_encoder_classifier_multi.transform(y_vol_test_raw_multi)
y_dir_test_encoded_multi = dir_encoder_classifier_multi.transform(y_dir_test_raw_multi)

y_vol_train_cat_multi = to_categorical(y_vol_train_encoded_multi)
y_dir_train_cat_multi = to_categorical(y_dir_train_encoded_multi)


# Calculate class weights for multivariate
vol_class_weights_multi = compute_class_weight('balanced', classes=np.unique(y_vol_train_encoded_multi), y=y_vol_train_encoded_multi)
dir_class_weights_multi = compute_class_weight('balanced', classes=np.unique(y_dir_train_encoded_multi), y=y_dir_train_encoded_multi)

vol_class_weight_dict_multi = {i: vol_class_weights_multi[i] for i in range(len(vol_class_weights_multi))}
dir_class_weight_dict_multi = {i: dir_class_weights_multi[i] for i in range(len(dir_class_weights_multi))}

# Build multivariate classifiers
reset_random_seeds()
vol_lstm_classifier_multi = build_lstm_classifier((slot_classifier, len(feature_cols_classifier_multi)), len(vol_encoder_classifier_multi.classes_))
reset_random_seeds()
dir_lstm_classifier_multi = build_lstm_classifier((slot_classifier, len(feature_cols_classifier_multi)), len(dir_encoder_classifier_multi.classes_))
reset_random_seeds()
vol_cnn_classifier_multi = build_cnn_lstm_classifier((slot_classifier, len(feature_cols_classifier_multi)), len(vol_encoder_classifier_multi.classes_))
reset_random_seeds()
dir_cnn_classifier_multi = build_cnn_lstm_classifier((slot_classifier, len(feature_cols_classifier_multi)), len(dir_encoder_classifier_multi.classes_))

# Train multivariate classifiers with time-based validation
# Time-based validation splits for multivariate
x_train_vol_multi, y_train_vol_multi, x_val_vol_multi, y_val_vol_multi = create_time_based_validation_split(x_train_classifier_multi, y_vol_train_cat_multi)
x_train_dir_multi, y_train_dir_multi, x_val_dir_multi, y_val_dir_multi = create_time_based_validation_split(x_train_classifier_multi, y_dir_train_cat_multi)


reset_random_seeds()
vol_lstm_history_multi = vol_lstm_classifier_multi.fit(
   x_train_vol_multi, y_train_vol_multi,
   validation_data=(x_val_vol_multi, y_val_vol_multi), 
   epochs=100, batch_size=32, verbose=0, 
   class_weight=vol_class_weight_dict_multi,
   callbacks=[early_stopping, reduce_lr]
)


reset_random_seeds()
dir_lstm_history_multi = dir_lstm_classifier_multi.fit(
   x_train_dir_multi, y_train_dir_multi,
   validation_data=(x_val_dir_multi, y_val_dir_multi), 
   epochs=100, batch_size=32, verbose=0, 
   class_weight=dir_class_weight_dict_multi,
   callbacks=[early_stopping, reduce_lr]
)


reset_random_seeds()
vol_cnn_history_multi = vol_cnn_classifier_multi.fit(
   x_train_vol_multi, y_train_vol_multi,
   validation_data=(x_val_vol_multi, y_val_vol_multi), 
   epochs=100, batch_size=64, verbose=0, 
   class_weight=vol_class_weight_dict_multi,
   callbacks=[early_stopping, reduce_lr]
)


reset_random_seeds()
dir_cnn_history_multi = dir_cnn_classifier_multi.fit(
   x_train_dir_multi, y_train_dir_multi,
   validation_data=(x_val_dir_multi, y_val_dir_multi), 
   epochs=100, batch_size=64, verbose=0, 
   class_weight=dir_class_weight_dict_multi,
   callbacks=[early_stopping, reduce_lr]
)

# Evaluate multivariate classifiers

vol_lstm_train_pred_multi, vol_lstm_test_pred_multi, vol_lstm_test_acc_multi = evaluate_classifier(
    vol_lstm_classifier_multi, x_train_classifier_multi, y_vol_train_encoded_multi, x_test_classifier_multi, y_vol_test_encoded_multi, vol_encoder_classifier_multi, "Multivariate LSTM Volatility"
)

dir_lstm_train_pred_multi, dir_lstm_test_pred_multi, dir_lstm_test_acc_multi = evaluate_classifier(
    dir_lstm_classifier_multi, x_train_classifier_multi, y_dir_train_encoded_multi, x_test_classifier_multi, y_dir_test_encoded_multi, dir_encoder_classifier_multi, "Multivariate LSTM Direction"
)

vol_cnn_train_pred_multi, vol_cnn_test_pred_multi, vol_cnn_test_acc_multi = evaluate_classifier(
    vol_cnn_classifier_multi, x_train_classifier_multi, y_vol_train_encoded_multi, x_test_classifier_multi, y_vol_test_encoded_multi, vol_encoder_classifier_multi, "Multivariate CNN-LSTM Volatility"
)

dir_cnn_train_pred_multi, dir_cnn_test_pred_multi, dir_cnn_test_acc_multi = evaluate_classifier(
    dir_cnn_classifier_multi, x_train_classifier_multi, y_dir_train_encoded_multi, x_test_classifier_multi, y_dir_test_encoded_multi, dir_encoder_classifier_multi, "Multivariate CNN-LSTM Direction"
)

# Confusion matrices

# Create confusion matrices for all models
models_data = [
    ("LSTM Vol (Uni)", vol_lstm_train_pred, y_vol_train_encoded, vol_lstm_test_pred, y_vol_test_encoded, vol_encoder_classifier),
    ("LSTM Dir (Uni)", dir_lstm_train_pred, y_dir_train_encoded, dir_lstm_test_pred, y_dir_test_encoded, dir_encoder_classifier),
    ("CNN-LSTM Vol (Uni)", vol_cnn_train_pred, y_vol_train_encoded, vol_cnn_test_pred, y_vol_test_encoded, vol_encoder_classifier),
    ("CNN-LSTM Dir (Uni)", dir_cnn_train_pred, y_dir_train_encoded, dir_cnn_test_pred, y_dir_test_encoded, dir_encoder_classifier),
    ("LSTM Vol (Multi)", vol_lstm_train_pred_multi, y_vol_train_encoded_multi, vol_lstm_test_pred_multi, y_vol_test_encoded_multi, vol_encoder_classifier_multi),
    ("LSTM Dir (Multi)", dir_lstm_train_pred_multi, y_dir_train_encoded_multi, dir_lstm_test_pred_multi, y_dir_test_encoded_multi, dir_encoder_classifier_multi),
    ("CNN-LSTM Vol (Multi)", vol_cnn_train_pred_multi, y_vol_train_encoded_multi, vol_cnn_test_pred_multi, y_vol_test_encoded_multi, vol_encoder_classifier_multi),
    ("CNN-LSTM Dir (Multi)", dir_cnn_train_pred_multi, y_dir_train_encoded_multi, dir_cnn_test_pred_multi, y_dir_test_encoded_multi, dir_encoder_classifier_multi)
]

# Plot confusion matrices 
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for idx, (model_name, train_pred, train_true, test_pred, test_true, encoder) in enumerate(models_data):
    # Train confusion matrix
    cm_train = confusion_matrix(train_true, train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', xticklabels=encoder.classes_, 
               yticklabels=encoder.classes_, ax=axes[idx*2], cmap='Blues', cbar=False)
    axes[idx*2].set_title(f'{model_name} - Train CM', fontsize=10)
    axes[idx*2].set_ylabel('True Label')
    axes[idx*2].set_xlabel('Predicted Label')
    
    # Test confusion matrix  
    cm_test = confusion_matrix(test_true, test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', xticklabels=encoder.classes_, 
               yticklabels=encoder.classes_, ax=axes[idx*2+1], cmap='Oranges', cbar=False)
    axes[idx*2+1].set_title(f'{model_name} - Test CM', fontsize=10)
    axes[idx*2+1].set_ylabel('True Label')
    axes[idx*2+1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Summary
print("\n=== CLASSIFICATION COMPARISON SUMMARY ===")
print("UNIVARIATE MODELS:")
print(f"LSTM Volatility:     Test Balanced Acc = {vol_lstm_test_acc:.3f}")
print(f"LSTM Direction:      Test Balanced Acc = {dir_lstm_test_acc:.3f}")
print(f"CNN-LSTM Volatility: Test Balanced Acc = {vol_cnn_test_acc:.3f}")
print(f"CNN-LSTM Direction:  Test Balanced Acc = {dir_cnn_test_acc:.3f}")

print("\nMULTIVARIATE MODELS:")
print(f"LSTM Volatility:     Test Balanced Acc = {vol_lstm_test_acc_multi:.3f}")
print(f"LSTM Direction:      Test Balanced Acc = {dir_lstm_test_acc_multi:.3f}")
print(f"CNN-LSTM Volatility: Test Balanced Acc = {vol_cnn_test_acc_multi:.3f}")
print(f"CNN-LSTM Direction:  Test Balanced Acc = {dir_cnn_test_acc_multi:.3f}")

print("\nIMPROVEMENT WITH EXOGENOUS FEATURES:")
print(f"LSTM Volatility:     {vol_lstm_test_acc_multi - vol_lstm_test_acc:+.3f}")
print(f"LSTM Direction:      {dir_lstm_test_acc_multi - dir_lstm_test_acc:+.3f}")
print(f"CNN-LSTM Volatility: {vol_cnn_test_acc_multi - vol_cnn_test_acc:+.3f}")
print(f"CNN-LSTM Direction:  {dir_cnn_test_acc_multi - dir_cnn_test_acc:+.3f}")



# random forest

full_data_classifier_corrected = full_data_classifier.copy()


features_with_rolling = ['vol_30', 'trend_strength', 'vol_5', 'vol_60', 'momentum_5', 'momentum_30']

print("Original feature calculation includes current day - FIXING...")
for feature in features_with_rolling:
    full_data_classifier_corrected[feature] = full_data_classifier_corrected[feature].shift(1)
    

full_data_classifier_corrected['price_change'] = full_data_classifier_corrected['price_change'].shift(1)



# Use the thresholds
print(f"Using global volatility thresholds: Low < {global_vol_q33:.4f} < Medium < {global_vol_q66:.4f} < High")
print(f"Using global direction thresholds: Bear < {global_dir_q25:.4f} < Neutral < {global_dir_q75:.4f} < Bull")

# Apply classification with corrected features
full_data_classifier_corrected['vol_regime'] = 'Medium'
full_data_classifier_corrected['dir_regime'] = 'Neutral'

# Use corrected vol_30 and trend_strength for classification
full_data_classifier_corrected.loc[full_data_classifier_corrected['vol_30'] < global_vol_q33, 'vol_regime'] = 'Low'
full_data_classifier_corrected.loc[full_data_classifier_corrected['vol_30'] > global_vol_q66, 'vol_regime'] = 'High'
full_data_classifier_corrected.loc[full_data_classifier_corrected['trend_strength'] < global_dir_q25, 'dir_regime'] = 'Bear'
full_data_classifier_corrected.loc[full_data_classifier_corrected['trend_strength'] > global_dir_q75, 'dir_regime'] = 'Bull'


# Clean data 
full_data_clean_classifier_corrected = full_data_classifier_corrected.dropna()

print(f"Data after cleaning: {full_data_clean_classifier_corrected.shape}")
print(f"Regime columns created: {'vol_regime' in full_data_clean_classifier_corrected.columns}, {'dir_regime' in full_data_clean_classifier_corrected.columns}")

# Check regime distributions

# Update splits with corrected data  
train_classifier_corrected = full_data_clean_classifier_corrected[full_data_clean_classifier_corrected.index <= '2021-12-31']
test_classifier_corrected = full_data_clean_classifier_corrected[full_data_clean_classifier_corrected.index >= '2022-01-01']


# Prepare corrected features with new scaler
train_features_classifier_corrected = train_classifier_corrected[feature_cols_classifier].values
test_features_classifier_corrected = test_classifier_corrected[feature_cols_classifier].values

# Create new scaler for corrected data
scaler_classifier_corrected = MinMaxScaler(feature_range=(0, 1))
train_features_classifier_corrected = scaler_classifier_corrected.fit_transform(train_features_classifier_corrected)
test_features_classifier_corrected = scaler_classifier_corrected.transform(test_features_classifier_corrected)


def create_rf_sequences(features, vol_labels, dir_labels, slot):
    
    x, y_vol, y_dir = [], [], []
    for i in range(slot, len(features)):
        # Flatten the sequence: (slot, n_features) -> (slot * n_features,)
        sequence_flattened = features[i-slot:i, :].flatten()
        x.append(sequence_flattened)
        y_vol.append(vol_labels.iloc[i])
        y_dir.append(dir_labels.iloc[i])
    return np.array(x), y_vol, y_dir

# RF evaluation
def evaluate_rf(model, X_train, y_train, X_test, y_test, encoder, class_name):
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
    
    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
    
    print(f"\n{class_name.upper()} RESULTS:")
    print(f"TRAIN - Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Recall: {train_recall:.4f}, Balanced Accuracy: {train_balanced_accuracy:.4f}")
    print(f"TEST  - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, Recall: {test_recall:.4f}, Balanced Accuracy: {test_balanced_accuracy:.4f}")
    
    return (y_train_pred, y_test_pred, test_balanced_accuracy)


# Create sequences with CORRECTED data
X_train_rf, y_vol_train_rf, y_dir_train_rf = create_rf_sequences(
    train_features_classifier_corrected, train_classifier_corrected['vol_regime'], 
    train_classifier_corrected['dir_regime'], slot_classifier)

X_test_rf, y_vol_test_rf, y_dir_test_rf = create_rf_sequences(
    test_features_classifier_corrected, test_classifier_corrected['vol_regime'], 
    test_classifier_corrected['dir_regime'], slot_classifier)

# Encode labels (same encoders as deep learning)
y_vol_train_rf_enc = vol_encoder_classifier.transform(y_vol_train_rf)
y_dir_train_rf_enc = dir_encoder_classifier.transform(y_dir_train_rf)
y_vol_test_rf_enc = vol_encoder_classifier.transform(y_vol_test_rf)
y_dir_test_rf_enc = dir_encoder_classifier.transform(y_dir_test_rf)

print(f"RF corrected input shape - Train: {X_train_rf.shape}, Test: {X_test_rf.shape}")

# Build and train RF models
rf_vol = RandomForestClassifier(
    n_estimators=50,        
    max_depth=8,           
    min_samples_split=20, 
    min_samples_leaf=10,   
    max_features='sqrt',  
    class_weight='balanced', 
    random_state=42
)

rf_dir = RandomForestClassifier(
    n_estimators=50,        
    max_depth=8,           
    min_samples_split=20, 
    min_samples_leaf=10,   
    max_features='sqrt',   
    class_weight='balanced', 
    random_state=42
)


rf_vol.fit(X_train_rf, y_vol_train_rf_enc)
rf_dir.fit(X_train_rf, y_dir_train_rf_enc)

# Evaluate with simplified metrics
rf_vol_train_pred, rf_vol_test_pred, rf_vol_test_acc = evaluate_rf(rf_vol, X_train_rf, y_vol_train_rf_enc, X_test_rf, y_vol_test_rf_enc, vol_encoder_classifier, "RF Volatility (Univariate)")
rf_dir_train_pred, rf_dir_test_pred, rf_dir_test_acc = evaluate_rf(rf_dir, X_train_rf, y_dir_train_rf_enc, X_test_rf, y_dir_test_rf_enc, dir_encoder_classifier, "RF Direction (Univariate)")

# Multiv Randm Forest


# Apply same corrections to multivariate data
merged_classifier_data_corrected = merged_classifier_data.copy()

# Shift the same rolling features
for feature in features_with_rolling:
    if feature in merged_classifier_data_corrected.columns:
        merged_classifier_data_corrected[feature] = merged_classifier_data_corrected[feature].shift(1)
        
# Also shift price_change
if 'price_change' in merged_classifier_data_corrected.columns:
    merged_classifier_data_corrected['price_change'] = merged_classifier_data_corrected['price_change'].shift(1)

# Clean and update multivariate splits
merged_classifier_data_clean_corrected = merged_classifier_data_corrected.dropna()
train_classifier_multi_corrected = merged_classifier_data_clean_corrected[merged_classifier_data_clean_corrected.index <= '2023-12-31']
test_classifier_multi_corrected = merged_classifier_data_clean_corrected[merged_classifier_data_clean_corrected.index >= '2024-01-01']

# Prepare corrected multivariate features
train_features_classifier_multi_corrected = train_classifier_multi_corrected[feature_cols_classifier_multi].values
test_features_classifier_multi_corrected = test_classifier_multi_corrected[feature_cols_classifier_multi].values

# Create new scaler for corrected multivariate data
scaler_classifier_multi_corrected = MinMaxScaler(feature_range=(0, 1))
train_features_classifier_multi_corrected = scaler_classifier_multi_corrected.fit_transform(train_features_classifier_multi_corrected)
test_features_classifier_multi_corrected = scaler_classifier_multi_corrected.transform(test_features_classifier_multi_corrected)

# Create multivariate sequences with corrected data
X_train_rf_multi, y_vol_train_rf_multi, y_dir_train_rf_multi = create_rf_sequences(
    train_features_classifier_multi_corrected, train_classifier_multi_corrected['vol_regime'], 
    train_classifier_multi_corrected['dir_regime'], slot_classifier)

X_test_rf_multi, y_vol_test_rf_multi, y_dir_test_rf_multi = create_rf_sequences(
    test_features_classifier_multi_corrected, test_classifier_multi_corrected['vol_regime'], 
    test_classifier_multi_corrected['dir_regime'], slot_classifier)

# Encode labels (multivariate encoders)
y_vol_train_rf_multi_enc = vol_encoder_classifier_multi.transform(y_vol_train_rf_multi)
y_dir_train_rf_multi_enc = dir_encoder_classifier_multi.transform(y_dir_train_rf_multi)
y_vol_test_rf_multi_enc = vol_encoder_classifier_multi.transform(y_vol_test_rf_multi)
y_dir_test_rf_multi_enc = dir_encoder_classifier_multi.transform(y_dir_test_rf_multi)

print(f"RF multivariate corrected input shape - Train: {X_train_rf_multi.shape}, Test: {X_test_rf_multi.shape}")

# Build and train multivariate RF models 
rf_vol_multi = RandomForestClassifier(
    n_estimators=50,        
    max_depth=10,          
    min_samples_split=25,  
    min_samples_leaf=12,   
    max_features='sqrt', 
    class_weight='balanced', 
    random_state=42
)

rf_dir_multi = RandomForestClassifier(
    n_estimators=50,        
    max_depth=10,          
    min_samples_split=25,  
    min_samples_leaf=12,   
    max_features='sqrt',  
    class_weight='balanced', 
    random_state=42
)


rf_vol_multi.fit(X_train_rf_multi, y_vol_train_rf_multi_enc)
rf_dir_multi.fit(X_train_rf_multi, y_dir_train_rf_multi_enc)

# Evaluate 
rf_vol_train_pred_multi, rf_vol_test_pred_multi, rf_vol_test_acc_multi = evaluate_rf(rf_vol_multi, X_train_rf_multi, y_vol_train_rf_multi_enc, X_test_rf_multi, y_vol_test_rf_multi_enc, vol_encoder_classifier_multi, "RF Volatility (Multivariate)")
rf_dir_train_pred_multi, rf_dir_test_pred_multi, rf_dir_test_acc_multi = evaluate_rf(rf_dir_multi, X_train_rf_multi, y_dir_train_rf_multi_enc, X_test_rf_multi, y_dir_test_rf_multi_enc, dir_encoder_classifier_multi, "RF Direction (Multivariate)")

# Confusion matrix


fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Models data for confusion matrices
rf_models = [
    ("RF Vol (Uni)", rf_vol_train_pred, y_vol_train_rf_enc, rf_vol_test_pred, y_vol_test_rf_enc, vol_encoder_classifier),
    ("RF Dir (Uni)", rf_dir_train_pred, y_dir_train_rf_enc, rf_dir_test_pred, y_dir_test_rf_enc, dir_encoder_classifier),
    ("RF Vol (Multi)", rf_vol_train_pred_multi, y_vol_train_rf_multi_enc, rf_vol_test_pred_multi, y_vol_test_rf_multi_enc, vol_encoder_classifier_multi),
    ("RF Dir (Multi)", rf_dir_train_pred_multi, y_dir_train_rf_multi_enc, rf_dir_test_pred_multi, y_dir_test_rf_multi_enc, dir_encoder_classifier_multi)
]

for idx, (name, y_train_pred, y_train_true, y_test_pred, y_test_true, encoder) in enumerate(rf_models):
    # Train CM
    cm_train = confusion_matrix(y_train_true, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', xticklabels=encoder.classes_, 
               yticklabels=encoder.classes_, ax=axes[0, idx], cmap='Blues', cbar=False)
    axes[0, idx].set_title(f'{name} - Train', fontsize=10)
    
    # Test CM
    cm_test = confusion_matrix(y_test_true, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', xticklabels=encoder.classes_, 
               yticklabels=encoder.classes_, ax=axes[1, idx], cmap='Oranges', cbar=False)
    axes[1, idx].set_title(f'{name} - Test', fontsize=10)

plt.tight_layout()
plt.show()

# Smmary
print("\n=== RANDOM FOREST BASELINE SUMMARY ===")
print("UNIVARIATE:")
print(f"Volatility:  Test Balanced Acc = {rf_vol_test_acc:.3f}")
print(f"Direction:   Test Balanced Acc = {rf_dir_test_acc:.3f}")

print("\nMULTIVARIATE:")
print(f"Volatility:  Test Balanced Acc = {rf_vol_test_acc_multi:.3f}")
print(f"Direction:   Test Balanced Acc = {rf_dir_test_acc_multi:.3f}")

print("\nIMPROVEMENT WITH EXOGENOUS FEATURES (Random Forest):")
print(f"Volatility:  {rf_vol_test_acc_multi - rf_vol_test_acc:+.3f}")
print(f"Direction:   {rf_dir_test_acc_multi - rf_dir_test_acc:+.3f}")



# ===== FEATURE IMPORTANCE FOR RANDOM FOREST MODELS =====
print("\n--- Feature Importance: Random Forest Classifiers ---")

def plot_feature_importance(model, feature_names, title, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Build feature names for sequence input
def expand_feature_names(base_features, slot):
    return [f"{feat}_t-{i}" for i in reversed(range(slot)) for feat in base_features]

# Univariate
univariate_feature_names = expand_feature_names(feature_cols_classifier, slot_classifier)
plot_feature_importance(rf_vol, univariate_feature_names, "Feature Importance - RF Volatility (Univariate)")
plot_feature_importance(rf_dir, univariate_feature_names, "Feature Importance - RF Direction (Univariate)")

# Multivariate
multivariate_feature_names = expand_feature_names(feature_cols_classifier_multi, slot_classifier)
plot_feature_importance(rf_vol_multi, multivariate_feature_names, "Feature Importance - RF Volatility (Multivariate)")
plot_feature_importance(rf_dir_multi, multivariate_feature_names, "Feature Importance - RF Direction (Multivariate)")



# Regime classification visualisation

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Define colors for regimes
vol_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
dir_colors = {'Bear': 'green', 'Neutral': 'gray', 'Bull': 'red'}

# Plot the volatility time series
ax1.plot(full_data_clean_classifier.index, full_data_clean_classifier['vol_30'], 
         color='blue', linewidth=1, alpha=0.8, label='30-day Volatility')

# Create background colors for volatility regimes
for regime in ['Low', 'Medium', 'High']:
    regime_mask = full_data_clean_classifier['vol_regime'] == regime
    regime_dates = full_data_clean_classifier.index[regime_mask]
    
    if len(regime_dates) > 0:
        # Create continuous segments for each regime
        segments = []
        start_idx = None
        
        for i, date in enumerate(full_data_clean_classifier.index):
            if regime_mask.iloc[i]:
                if start_idx is None:
                    start_idx = date
            else:
                if start_idx is not None:
                    segments.append((start_idx, date))
                    start_idx = None
        
        # Don't forget the last segment if it ends at the data end
        if start_idx is not None:
            segments.append((start_idx, full_data_clean_classifier.index[-1]))
        
        # Plot background colors for each segment
        for start_date, end_date in segments:
            ax1.axvspan(start_date, end_date, alpha=0.3, color=vol_colors[regime], 
                       label=f'{regime} Volatility' if segments.index((start_date, end_date)) == 0 else "")

# Add horizontal threshold lines
ax1.axhline(y=global_vol_q33, color='green', linestyle='--', alpha=0.7, linewidth=1, label=f'Low Threshold ({global_vol_q33:.4f})')
ax1.axhline(y=global_vol_q66, color='red', linestyle='--', alpha=0.7, linewidth=1, label=f'High Threshold ({global_vol_q66:.4f})')

ax1.set_ylabel('Volatility', fontsize=12)
ax1.set_title('Volatility Regime Classification', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=10)

# Direction plot
# Plot the trend strength time series
ax2.plot(full_data_clean_classifier.index, full_data_clean_classifier['trend_strength'], 
         color='purple', linewidth=1, alpha=0.8, label='Trend Strength')

# Create background colors for direction regimes
for regime in ['Bear', 'Neutral', 'Bull']:
    regime_mask = full_data_clean_classifier['dir_regime'] == regime
    regime_dates = full_data_clean_classifier.index[regime_mask]
    
    if len(regime_dates) > 0:
        # Create continuous segments for each regime
        segments = []
        start_idx = None
        
        for i, date in enumerate(full_data_clean_classifier.index):
            if regime_mask.iloc[i]:
                if start_idx is None:
                    start_idx = date
            else:
                if start_idx is not None:
                    segments.append((start_idx, date))
                    start_idx = None
        
        # Don't forget the last segment if it ends at the data end
        if start_idx is not None:
            segments.append((start_idx, full_data_clean_classifier.index[-1]))
        
        # Plot background colors for each segment
        for start_date, end_date in segments:
            ax2.axvspan(start_date, end_date, alpha=0.3, color=dir_colors[regime], 
                       label=f'{regime} Market' if segments.index((start_date, end_date)) == 0 else "")

# Add horizontal threshold lines
ax2.axhline(y=global_dir_q25, color='red', linestyle='--', alpha=0.7, linewidth=1, label=f'Bear Threshold ({global_dir_q25:.4f})')
ax2.axhline(y=global_dir_q75, color='green', linestyle='--', alpha=0.7, linewidth=1, label=f'Bull Threshold ({global_dir_q75:.4f})')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8, label='Neutral Line')

ax2.set_ylabel('Trend Strength', fontsize=12)
ax2.set_title('Direction Regime Classification', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=10)

# Format x-axis
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_minor_locator(mdates.MonthLocator([1, 7]))

# Rotate and align the tick labels so they look better
fig.autofmt_xdate()

# Adjust layout
plt.tight_layout()
plt.show()





