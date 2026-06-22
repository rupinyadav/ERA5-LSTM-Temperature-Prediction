"""
holdout_utils.py

Utility functions for training and evaluating LSTM temperature
forecasting models using a temporal holdout validation strategy.

Project:
    Prediction of Daily Average Temperature of Major Indian cities using ERA5 Reanalysis Data.

Validation Strategy:
    Development Period : 1980_2020
    Holdout Period     : 2021_2025

Workflow:
    1. Load development and holdout datasets
    2. Scale development data
    3. Apply same scaler to holdout data
    4. Create sliding-window sequences
    5. Train LSTM on development data
    6. Evaluate on test subset
    7. Forecast completely unseen holdout years
    8. Generate visualizations and comparison tables

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

from keras.layers import (
    LSTM,
    Dense,
    Input
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# REPRODUCIBILITY
# ============================================================
# Fix random seeds to ensure that training results
# remain consistent across multiple runs.

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# ============================================================
# DATA LOADING
# ============================================================
def load_period_file(filename):

    df = pd.read_csv(
        f"../data/{filename}",
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["date", "temperature"]
    )

    df["temperature"] = pd.to_numeric(
        df["temperature"],
        errors="coerce"
    )

    df["temperature"] = (
        df["temperature"] - 273.15
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df

# ============================================================
# DATA SCALING
# ============================================================
def scale_development_data(df):

    scaler = MinMaxScaler(
        feature_range=(0,1)
    )

    scaled = scaler.fit_transform(
        df["temperature"]
        .values
        .reshape(-1,1)
    )

    return scaled, scaler


def transform_holdout_data(
    df,
    scaler
):
    """
    Apply the previously fitted scaler to holdout data.

    Important:
    ----------
    The scaler is NOT re-fitted on holdout data.

    This simulates a real forecasting scenario where
    future observations are unavailable during training.

    Parameters
    ----------
    df : pandas.DataFrame

    scaler : MinMaxScaler

    Returns
    -------
    numpy.ndarray
        Scaled holdout temperatures.
    """

    scaled = scaler.transform(
        df["temperature"]
        .values
        .reshape(-1,1)
    )

    return scaled

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


def train_holdout_model(
    city_name,
    sequence_length=30,
    epochs=20,
    batch_size=32
):

    df_dev = load_period_file(
        f"{city_name}_1980_2020.txt"
    )

    df_holdout = load_period_file(
         f"{city_name}_2021_2025.txt"
   )

    scaled_dev, scaler = scale_development_data(
         df_dev
    )

    scaled_holdout = transform_holdout_data(
        df_holdout,
        scaler
    )

    X_dev, y_dev = create_sequences(
        scaled_dev,
        sequence_length
    )

    X_holdout, y_holdout = create_sequences(
        scaled_holdout,
        sequence_length
    )
    train_size = int(len(X_dev)*0.8)
    # dates corresponding to y values
    dates = df_dev["date"]
    test_dates = dates.iloc[
        sequence_length + train_size:
    ].reset_index(drop=True)
    
    holdout_dates = (
        df_holdout["date"]
        .iloc[sequence_length:]
        .reset_index(drop=True)
    
    )


    train_start = dates.iloc[0]

    train_end = dates.iloc[
        sequence_length + train_size - 1
    
    ]

    test_start = test_dates.iloc[0]

    test_end = test_dates.iloc[-1]

    holdout_start = holdout_dates.iloc[0]
    holdout_end = holdout_dates.iloc[-1]

    print("Training:")
    print(train_start.date(), "to", train_end.date())

    print("\nTesting:")
    print(test_start.date(), "to", test_end.date())

    print("\nHoldout:")
    print(holdout_start.date(), "to", holdout_end.date())
    
    X_train = X_dev[:train_size]
    X_test = X_dev[train_size:]

    y_train = y_dev[:train_size]
    y_test = y_dev[train_size:]

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

    # generating Predictions
    predictions = model.predict(X_test)

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

    # holdout predictions

    holdout_predictions = model.predict(
        X_holdout
    )

    holdout_predictions = (
        scaler.inverse_transform(
            holdout_predictions.reshape(-1,1)
        )
    )

    holdout_actual = (
        scaler.inverse_transform(
             y_holdout.reshape(-1,1)
        )
    )
#holdout metrices
    holdout_rmse = np.sqrt(
        mean_squared_error(
            holdout_actual,
            holdout_predictions
         )
    )

    holdout_mae = mean_absolute_error(
        holdout_actual,
        holdout_predictions
    
    )

    holdout_correlation = np.corrcoef(
         holdout_actual.flatten(),
         holdout_predictions.flatten()
   
    )[0,1]

    holdout_r2 = r2_score(
        holdout_actual,
        holdout_predictions
    )

    return {

        "city": city_name,
        "model": model,
        "history": history,
        "train_start": train_start,
        "train_end": train_end,

        "test_start": test_start,
        "test_end": test_end,
 
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        #test result
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
        "r2": r2,
        "predictions": predictions,
        "actual": y_actual,
        "dates": test_dates,

        #holdout results
        "holdout_rmse": holdout_rmse,
        "holdout_mae": holdout_mae,
        "holdout_correlation": holdout_correlation,
        "holdout_r2": holdout_r2,
        "holdout_predictions": holdout_predictions,
        "holdout_actual": holdout_actual,
        "holdout_dates": holdout_dates,

        "test_start": test_dates.iloc[0],
        "test_end": test_dates.iloc[-1],

        "holdout_start": holdout_dates.iloc[0],
        "holdout_end": holdout_dates.iloc[-1]
    }

# ============================================================
# VISUALIZATION
# ============================================================
def plot_predictions(results):

    dates = results["dates"]
   

    plt.figure(figsize=(15,5))

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
        mdates.YearLocator()
    )

    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter(' %Y')

    )
    

    plt.legend(loc ="upper right")

    plt.tight_layout()

    plt.savefig(
        f"../plots/{results['city']}_forecast.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_holdout_forecast(results):

    dates = results["holdout_dates"]

    plt.figure(figsize=(15,5))

    plt.plot(
        dates,
        results["holdout_predictions"].flatten(),
        label="Forecasted Temperature"
    )

    plt.title(
        f'{results["city"].title()} Temperature Forecast of unseen data (2021-2025)'
    )

    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")

    plt.gca().xaxis.set_major_locator(
        mdates.YearLocator()
    )

    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter('%Y')
    )

    plt.xticks(rotation=45)

    plt.legend(loc="upper right")

    plt.tight_layout()

    plt.savefig(
        f"../plots/{results['city']}_forecast_only.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_holdout_predictions(results):
        dates = results["holdout_dates"]
       
        
        plt.figure(figsize=(15,5))

        plt.plot(
            dates,
            results["holdout_actual"].flatten(),
            label="Actual Temperature"
        
        )

        plt.plot(
            dates,
            results["holdout_predictions"].flatten(),
            label="Predicted Temperature"
        
        )

        plt.title(
            f'{results["city"].title()} Holdout Forecast'
           
        )
        plt.xlabel("Year")
        plt.ylabel("Temperature (°C)")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            f"../plots/{results['city']}_holdout_forecast.png",
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
   
def create_Comparison_table(results):

    Comparison = pd.DataFrame({

        "City":
            [r["city"] for r in results],

        "Test RMSE":
            [r["rmse"] for r in results],

        "Holdout RMSE":
            [r["holdout_rmse"] for r in results],

        "Test MAE":
            [r["mae"] for r in results],

        "Holdout MAE":
            [r["holdout_mae"] for r in results],

        "Test R²":
            [r["r2"] for r in results],

        "Holdout R²":
            [r["holdout_r2"] for r in results],
        "Test Correlation":
             [r["correlation"] for r in results],
        "Holdout Correlation":
              [r["correlation"] for r in results]

    })

    return Comparison
   
