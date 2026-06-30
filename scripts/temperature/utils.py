"""
utils.py

Project:
    Prediction of Daily Average Temperature of Major Indian cities using ERA5 Reanalysis Data(2000-2010)

Workflow:
    1. Load city temperature data
    2. Convert Kelvin to Celsius
    3. Scale data using Min-Max normalization
    4. Create sliding-window sequences
    5. Build and train an LSTM model
    6. Evaluate forecasting performance
    7. Visualize predictions 
    8. Comparison table

Author: Rupin Yadav
Institute: IISER Mohali
"""


# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from sklearn.metrics import r2_score

from keras.layers import (
    LSTM,
    Dense,
    Input
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# ============================================================
# REPRODUCIBILITY
# ============================================================
# Fix random seeds so that results remain consistent across runs.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)



# ============================================================
# DATA LOADING
# ============================================================
def load_city(city_name):

    df = pd.read_csv(
        f"../data/temperature{city_name}.txt",
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["date", "temperature"]
    )
 # Convert temperature values to numeric format,(Invalid values are replaced with NaN)
 
    df["temperature"] = pd.to_numeric(
        df["temperature"],
        errors="coerce"
    )


# Convert Kelvin → Celsius
    df["temperature"] = df["temperature"] - 273.15

    return df

# ============================================================
# DATA SCALING
# ============================================================
def scale_data(df):

    scaler = MinMaxScaler(
        feature_range=(0,1)
    )

    scaled = scaler.fit_transform(
        df["temperature"]
        .values
        .reshape(-1,1)
    )

    return scaled, scaler
# ============================================================
# SLIDING WINDOW SEQUENCE CREATION
# ============================================================
def create_sequences(data, seq_length):
    X = []
    y = []

    for i in range(seq_length, len(data)):

        X.append(data[i-seq_length:i])

        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    return X, y


# ============================================================
# MODEL ARCHITECTURE
# ============================================================
"""
    Build and compile the LSTM forecasting model.

    Architecture:
        Input Layer
            ↓
        LSTM (100 units)
            ↓
        Dense Output Layer

    Parameters
    ----------
    sequence_length : int

    Returns
    -------
    keras.Model
        Compiled LSTM model.
"""
def build_model(sequence_length):

    model = Sequential()

    model.add(
        Input(shape=(sequence_length,1))
    )

    model.add(
        LSTM(100)
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


# ============================================================
# TRAINING AND EVALUATION
# ============================================================
def train_city_model(
    city_name,
    sequence_length=30,
    epochs=20,
    batch_size=32
):
    """
    Train an LSTM model for a single city.

    Pipeline:
        Load Data
            ↓
        Scale Data
            ↓
        Create Sequences
            ↓
        Train-Test Split (80:20)
            ↓
        Train LSTM
            ↓
        Evaluate Metrics

    Parameters
    ----------
    city_name : str

    sequence_length : int, default=30
        Number of past days used for prediction.

    epochs : int, default=20
        Number of complete training passes.

    batch_size : int, default=32
        Number of samples processed before updating weights.

    Returns
    -------
    dict
        Dictionary containing model, metrics,
        predictions, actual values, and dates.
    """

    df = load_city(city_name)
    scaled, scaler = scale_data(df)
    X, y = create_sequences(
        scaled,
        sequence_length
    )

    train_size = int(len(X)*0.8)

    dates = pd.to_datetime(df["date"])
    test_dates = dates.iloc[
        sequence_length + train_size:
    ].reset_index(drop=True)
    
    dates = pd.to_datetime(df["date"])

    train_start = dates.iloc[0]
    train_end = dates.iloc[sequence_length + train_size - 1]

    test_start = dates.iloc[sequence_length + train_size]
    test_end = dates.iloc[-1]

    print("Training Period:")
    print(train_start.date(), "to", train_end.date())

    print("\nTesting Period:")
    print(test_start.date(), "to", test_end.date())

    X_train = X[:train_size]
    X_test = X[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    # Build model
    model = build_model(sequence_length)

    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test,y_test),
        verbose=0
    )

    # generating predictions
    predictions = model.predict(X_test)
     # Convert predictions back to Celsius
    predictions = scaler.inverse_transform(
        predictions.reshape(-1,1)
    )

    y_actual = scaler.inverse_transform(
        y_test.reshape(-1,1)
    )

    # ------------------------
    # Performance Metrics
    # ------------------------
    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_actual,
        predictions
    )

    correlation = np.corrcoef(
         y_actual.flatten(),
         predictions.flatten()
   
    )[0,1]

    r2 = r2_score(
        y_actual,
        predictions
    )

    return {

        "city": city_name,
        "model": model,
        "history": history,
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
        "r2": r2,
        "predictions": predictions,
        "actual": y_actual,
        "dates": test_dates
    }


# ============================================================
# VISUALIZATION
# ============================================================
def plot_predictions(results):

    dates = results["dates"]
   

    plt.figure(figsize=(12,5))

    plt.plot(
        dates,
        results["actual"].flatten(),
        label=f'Actual Temperature'
    )

    plt.plot(
        dates,
        results["predictions"].flatten(),
        label=f'Predicted Temperature'
    )

    plt.title(
        f'{results["city"].title()} Forecast'
    )

    plt.xlabel("Year")

    plt.ylabel("Temperature (°C)")

    plt.gca().xaxis.set_major_locator(
        mdates.MonthLocator(interval=3)
    )

    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter(' %b %Y')
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"../plots/{results['city']}_forecast.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# ============================================================
# COMPARISON TABLE 
# ============================================================
def create_comparison_table(results):

    comparison = pd.DataFrame({

        "City":
            [r["city"] for r in results],

        " RMSE":
            [r["rmse"] for r in results],

        " MAE":
            [r["mae"] for r in results],

        " R²":
            [r["r2"] for r in results],

        " Correlation":
             [r["correlation"] for r in results],
        
    })

    return comparison 