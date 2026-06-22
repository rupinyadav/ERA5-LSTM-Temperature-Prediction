"""
hourlyutils.py
Project: Prediction of Hourly Temperature of Major Indian cities using ERA5 Reanalysis Data.(LSTM Model)

This module provides utilities for:

1. Loading and preprocessing ERA5 hourly temperature data.
2. Scaling temperature values using MinMaxScaler.
3. Creating sliding-window sequences for LSTM training.
4. Building and training LSTM models.
5. Evaluating model performance using:
   - RMSE
   - MAE
   - Correlation Coefficient
   - R² Score
6. Visualizing predictions.
7. Comparing performance across multiple cities.
8. Evaluating models on unseen holdout datasets.

Author: Rupin Yadav

"""
# ============================================================================
# IMPORTS
# ============================================================================
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)
from keras.models import Sequential
from sklearn.metrics import r2_score

from keras.layers import (
    LSTM,
    Dense,
    Input
)
# ============================================================================
# REPRODUCIBILITY
# ============================================================================
# Fixed random seed to ensure reproducible model training
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def load_hourly_city(city_name):
    """
    Load hourly temperature data for a city.

    The function:
    - Reads ERA5 hourly temperature data from a text file.
    - Converts temperatures from Kelvin to Celsius.
    - Combines date and time columns into a single datetime column.

    Parameters
    ----------
    city_name : str
        Name of the city.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - date
        - time
        - temperature (°C)
        - datetime
    """

    df = pd.read_csv(
        f"../data/{city_name}_hourly_2000_2010.txt",
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["date", "time", "temperature"]
    )

    df["temperature"] = pd.to_numeric(
        df["temperature"],
        errors="coerce"
    )


# Convert Kelvin → Celsius
    df["temperature"] = df["temperature"] - 273.15

# combine timestamp column
    df["datetime"]= pd.to_datetime(
        df["date"]+ " " + df["time"]
    )
    
    return df


#Scaling function
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

#Sequence function(encapsulates sliding window preprocessing)
def create_sequences(data, seq_length):
    """
    Create sliding-window sequences for LSTM training.

    Example:
    If seq_length = 48,
    the model uses the previous 48 hourly observations
    to predict the next hour.

    Parameters
    ----------
    data : ndarray
        Scaled temperature values.

    seq_length : int
        Number of previous time steps used as input.

    """
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


# ============================================================================
# MODEL CONSTRUCTION
# ============================================================================ 
def build_model(sequence_length):
    """
    Build and compile an LSTM model.

    Architecture:
    Input Layer
        ↓
    LSTM(100 units)
        ↓
    Dense(1)

    Parameters
    ----------
    sequence_length : int
        Number of input time steps.

    Returns
    -------
    keras.Model
        Compiled LSTM model.
    """
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

#Train an LSTM model for hourly temperature forecasting.
def train_hourly_model(
    city_name,
    sequence_length=48,
    epochs=20,
    batch_size=32
):

    # load data
    df = load_hourly_city(city_name)

    # scale
    scaled, scaler = scale_data(df)

    # sequences
    X, y = create_sequences(
        scaled,
        sequence_length
    )

    # split
    train_size = int(len(X)*0.8)

    # dates corresponding to y values
    dates = df["datetime"]
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

    # predictions
    predictions = model.predict(X_test)

    predictions = scaler.inverse_transform(
        predictions.reshape(-1,1)
    )

    y_actual = scaler.inverse_transform(
        y_test.reshape(-1,1)
    )

    # metrics
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
        "dates": test_dates,
        "scaler":scaler
    }

# ============================================================
# VISUALIZATION
# ============================================================
def plot_week(results):
    """
Plot actual and predicted temperatures for the first week
(168 hours) of the testing period.

Useful for visual inspection of short-term forecast accuracy.
"""
    plt.figure(figsize=(15,5))

    plt.plot(
        results["dates"][:168],
        results["actual"].flatten()[:168],
        label="Actual"
    )

    plt.plot(
        results["dates"][:168],
        results["predictions"].flatten()[:168],
        label="Predicted"
    )

    plt.title(
        f"{results['city']} Hourly Forecast (1 Week)"
    )

    plt.xlabel("Date-Time")

    plt.ylabel("Temperature (°C)")

    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter('%d-%b-%Y\n%H:%M')
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


def plot_week_by_date(results, start_date):
    """
Plot a 7-day forecast window starting from a user-selected date.

This allows inspection of model performance during
specific weather periods.
"""
    # convert user input to pandas datetime
    start_date = pd.to_datetime(start_date)

    # find first matching index
    idx = np.where(
        results["dates"] >= start_date
    )[0]

    if len(idx) == 0:

        print(
            "Selected date is outside the available range."
        )

        return

    idx = idx[0]

    # 168 hours = 7 days
    end_idx = min(
        idx + 168,
        len(results["dates"])
    )

    plt.figure(figsize=(15,5))

    plt.plot(
        results["dates"][idx:end_idx],
        results["actual"].flatten()[idx:end_idx],
        label="Actual Temperature"
    )

    plt.plot(
        results["dates"][idx:end_idx],
        results["predictions"].flatten()[idx:end_idx],
        label="Predicted Temperature"
    )

    plt.title(
        f"{results['city'].title()} Hourly Forecast\n"
        f"Starting {start_date.strftime('%d-%b-%Y')}"
    )

    plt.xlabel("Date-Time")

    plt.ylabel("Temperature (°C)")

    plt.gca().xaxis.set_major_locator(
        mdates.HourLocator(interval=12)
    )

    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter(
            '%d-%b-%Y\n%H:%M'
        )
    )

    plt.legend()

    plt.tight_layout()

    plt.show()   



def plot_hourly_predictions(results):
    """
Plot the entire testing period forecast.

The figure is also saved automatically in the plots folder
for reporting and comparison purposes.
"""
    dates = results["dates"]
   

    plt.figure(figsize=(15,5))

    plt.plot(
        results["dates"],
        results["actual"].flatten(),
        label=f'Actual Temperature'
    )

    plt.plot(
        results["dates"],
        results["predictions"].flatten(),
        label=f'Predicted Temperature'
    )

    plt.title(
        f'{results["city"].title()} Hourly Forecast'
    )

    plt.xlabel("Time")

    plt.ylabel("Temperature (°C)")

    plt.gca().xaxis.set_major_locator(
        mdates.MonthLocator(interval=3)
    )

    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter('%b %Y')
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
    """
Create a summary table comparing test-set performance
across multiple cities.
"""
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


def evaluate_holdout_period(
    city_name,
    model,
    scaler,
    holdout_file,
    sequence_length=48,
    start_date="2026-06-01",
    end_date="2026-06-06 23:00:00"
):
    """
Evaluate a trained model on completely unseen data.

The holdout dataset is not used during training
or testing and provides a realistic estimate
of model generalization ability.

"""
    df_holdout = pd.read_csv(
        holdout_file,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["date","time","temperature"]
    )

    df_holdout["datetime"] = pd.to_datetime(
        df_holdout["date"] + " " + df_holdout["time"]
    )

    df_holdout["temperature"] = (
        pd.to_numeric(df_holdout["temperature"])
        - 273.15
    )

    scaled = scaler.transform(
        df_holdout["temperature"]
        .values
        .reshape(-1,1)
    )

    X_holdout, y_holdout = create_sequences(
        scaled,
        sequence_length
    )

    predictions = model.predict(X_holdout)

    predictions = scaler.inverse_transform(
        predictions
    )

    actual = scaler.inverse_transform(
        y_holdout.reshape(-1,1)
    )

    dates = (
        df_holdout["datetime"]
        .iloc[sequence_length:]
        .reset_index(drop=True)
    )

    mask = (
        dates >= start_date
    ) & (
        dates <= end_date
    )

    dates = dates[mask]

    predictions = predictions[mask]

    actual = actual[mask]

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    mae = mean_absolute_error(
        actual,
        predictions
    )

    correlation = np.corrcoef(
        actual.flatten(),
        predictions.flatten()
    )[0,1]

    r2 = r2_score(
        actual,
        predictions
    )

    return {
        "city":city_name,

        "dates": dates,

        "actual": actual,

        "predictions": predictions,

        "rmse": rmse,

        "mae": mae,

        "correlation": correlation,

        "r2": r2
    }

def create_test_comparison_table(results):
    """
Create a comparison table summarizing testing-period
forecast performance for multiple cities.
"""
    comparison = pd.DataFrame({

        "City":
            [r["city"] for r in results],

        "RMSE":
            [r["rmse"] for r in results],

        "MAE":
            [r["mae"] for r in results],

        "R²":
            [r["r2"] for r in results],

        "Correlation":
            [r["correlation"] for r in results]
    })

    comparison.index = comparison.index + 1

    return comparison

def create_holdout_comparison_table(results):
    """
Create a comparison table summarizing holdout-period
forecast performance for multiple cities.
"""
    comparison = pd.DataFrame({

        "City":
            [r["city"] for r in results],

        "RMSE":
            [r["rmse"] for r in results],

        "MAE":
            [r["mae"] for r in results],

        "R²":
            [r["r2"] for r in results],

        "Correlation":
            [r["correlation"] for r in results]
    })

    comparison.index = comparison.index + 1

    return comparison


def create_full_comparison_table(
    test_results,
    holdout_results
):
    """
Combine test-set and holdout-set metrics into
a single comparison table.

Useful for assessing whether performance
degrades on unseen future data.
"""
    comparison = pd.DataFrame({

        "City":
            [r["city"] for r in test_results],
        
         "Test MAE":
            [r["mae"] for r in test_results],

        "Holdout MAE":
            [r["mae"] for r in holdout_results],

        "Test RMSE":
            [r["rmse"] for r in test_results],

        "Holdout RMSE":
            [r["rmse"] for r in holdout_results],

        "Test R²":
            [r["r2"] for r in test_results],

        "Holdout R²":
            [r["r2"] for r in holdout_results],

        "Test Corr":
            [r["correlation"] for r in test_results],

        "Holdout Corr":
            [r["correlation"] for r in holdout_results]
    })
    comparison.index = comparison.index + 1

    return comparison

def plot_holdout_period(results):
    """
Visualize actual versus predicted temperatures
for the selected holdout period.

This plot helps evaluate model behaviour
on unseen future observations.
"""
    plt.figure(figsize=(15,5))

    plt.plot(
        results["dates"],
        results["actual"].flatten(),
        label="Actual"
    )

    plt.plot(
        results["dates"],
        results["predictions"].flatten(),
        label="Predicted"
    )

    plt.title(
        f'{results["city"].title()} Hourly Holdout Forecast'
    )

    plt.xlabel("Date-Time")

    plt.ylabel("Temperature (°C)")

    plt.legend()

    plt.tight_layout()

    plt.show()