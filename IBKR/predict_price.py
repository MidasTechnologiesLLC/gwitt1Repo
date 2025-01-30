import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load the training and testing data
training_data = pd.read_csv("3_years_training_data.csv")
testing_data = pd.read_csv("3_month_testing_data.csv")

# Drop unnecessary columns
training_data = training_data.drop(columns=["Unnamed: 0", "Date"])
testing_data = testing_data.drop(columns=["Unnamed: 0", "Date"])

# Create lagged features for the model
def create_lagged_features(data, n_lags=3):
    df = data.copy()
    for lag in range(1, n_lags + 1):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)  # Remove rows with NaN values due to shifting
    return df

# Apply lagged features to the training and testing datasets
training_data = create_lagged_features(training_data)
testing_data = create_lagged_features(testing_data)

# Separate features and target
X_train = training_data.drop(columns=["Close"]).values
y_train = training_data["Close"].values
X_test = testing_data.drop(columns=["Close"]).values
y_test = testing_data["Close"].values

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(X_train.shape[1],)),
    Dense(32, activation='sigmoid'),
    Dense(16, activation='sigmoid'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model on the test set
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Neural Network MSE: {mse:.2f}")
print(f"Neural Network MAE: {mae:.2f}")

# Prepare the latest data to predict tomorrow's price
latest_data = testing_data.tail(1).drop(columns=["Close"])
latest_data_scaled = scaler.transform(latest_data)

# Predict tomorrow's close price
tomorrow_pred = model.predict(latest_data_scaled)
print(f"Predicted Close Price for Tomorrow: {tomorrow_pred[0][0]:.2f}")
