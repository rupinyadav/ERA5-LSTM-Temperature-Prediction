# ERA5 Temperature Forecasting using LSTM Networks

## Overview

This project investigates temperature forecasting using Long Short-Term Memory (LSTM) neural networks and ERA5 reanalysis climate data. The work focuses on learning temporal patterns in historical temperature records and evaluating the ability of deep learning models to predict future temperatures for major Indian cities.

The repository contains two complementary studies:

1. **Daily Temperature Forecasting (1980–2020)**
2. **Hourly Temperature Forecasting (2000–2010)**

Both studies use ERA5 temperature data but explore forecasting performance at different temporal resolutions.

---

## Objectives

* Extract city-level temperature data from ERA5 climate datasets
* Develop preprocessing pipelines for climate time series
* Create sliding-window sequences for supervised learning
* Train and evaluate LSTM-based forecasting models
* Compare model performance across multiple Indian cities
* Investigate the impact of temporal resolution (daily vs hourly) on forecasting accuracy

---

## Dataset

### Source

ERA5 Reanalysis Dataset

### Variable Used

* 2-meter Air Temperature (T2M)

### Study Periods

#### Daily Forecasting

* 1980–2020
* Daily average temperature

#### Hourly Forecasting

* 2000–2010
* Hourly temperature observations

### Cities Included

* Delhi
* Mumbai
* Bengaluru
* Hyderabad
* Chennai
* Kolkata
* Jaipur
* Ahmedabad
* Pune
* Surat

---

## Project Components

### Experiment 1: Daily Temperature Forecasting

This study focuses on long-term daily temperature prediction using ERA5 daily average temperature data.

#### Workflow

* Data extraction from ERA5 NetCDF files
* Data preprocessing
* Min-Max normalization
* Sliding window sequence generation
* LSTM model training
* Forecast evaluation

#### Goal

To analyze the capability of LSTM networks in learning long-term temperature trends and seasonal variations across Indian cities.

---

### Experiment 2: Hourly Temperature Forecasting

This study investigates high-resolution temperature forecasting using hourly ERA5 observations.

#### Workflow

* Data extraction from ERA5 NetCDF files
* Data preprocessing
* Min-Max normalization
* Sliding window sequence generation
* LSTM model training
* short-term Forecast evaluation

#### Goal

To understand model performance on finer temporal scales and compare it with daily forecasting results.

---

## Methodology

### Data Preprocessing

* Temperature scaling using MinMaxScaler
* Sliding-window sequence creation
* Train-Test split (80:20)

### Model Architecture

The forecasting model is implemented using TensorFlow/Keras.

Architecture:

Input Layer → LSTM Layer → Dense Layer → Temperature Prediction

### Training Configuration

* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* Framework: TensorFlow/Keras

---

## Evaluation Metrics

Model performance is evaluated using:

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Correlation Analysis
* R² Score 


## Repository Structure

```text
ERA5_LSTM_Project/

├── data/
│   └── City-wise temperature datasets

├── notebooks/
│   ├── Daily forecasting notebooks
│   └── Hourly forecasting notebooks

├── scripts/
│   ├── Daily Model building scripts
│   └── Hourly Model building scripts

├── plots/
│   └── Visualizations

├── outputs/
│   └── Predictions and evaluation outputs

├── README.md
└── .gitignore
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Climate Data Operators (CDO)
* Jupyter Notebook
* VS Code
* Git & GitHub

---

## Future Work

* Multi-step temperature forecasting
* Rainfall prediction using ERA5 data

---

## Author

**Rupin Yadav**

BS-MS (Mathematics)

Indian Institute of Science Education and Research (IISER) Mohali

Research Interests:

* Climate Data Analysis
* Machine Learning
* Time Series Forecasting
* Applied Mathematics
