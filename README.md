[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/A-vEqCXL)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=17069359)

# AIPI520 Final Project
## Hungyin Chen & Haochen Li

# 0. Project Overview
Stock market trends are inherently non-linear and influenced by various factors. Following are our aim and measurement:
- Aim: our aim is to use the historical data to predict next day S&P 500 price change rate.
- Measurement: the measurement for all models are Mean Squared Error (MSE), since the aim of the model is predicting numerical value.

The general steps taken for this project are:
- Create sequences of stock data for time-series prediction.
- Train ARIMA models (statistic model) as the non deep learning model for predicting close price change percentage.
- Train LSTM-based deep learning models for predicting close price change percentage.
- Experiment with hyperparameter tuning to improve model performances.

The project includes features like data preprocessing, model training, evaluation, and visualization of predictions. The link to demo video is: https://youtu.be/e_10wB1-jgU


# 1. Running Instruction
- Create venv `python -m venv .venv`
- Activate venv `source .venv/bin/activate`
- Install packages `pip install -r requirements.txt`
- run `main.py` to get result, it will run both deep learning and non deep learning models, so may take a few hours.

# 2. Data sourcing and processing pipeline <br>

- We sourced the dataset from yfinance library. The detailed pipeline code is in *./utils/load_data.py*. Follwing is the explanation and pipeline.

- We picked daily S&P 500 stock data including Date, Volume, Open price, Close price, High price, Low Price of each day. We picked data from 2020 to 2023 to avoid Volume NaN issue. After getting the data, we created percentage changes for each price column as new features. Then we cleaned the table by keeping only the percentage change.

- The result is then stored in *./data/sp500_data.csv* for future use by both models. Our aim is to use the historical data to predict next day S&P 500 price change rate.

- For deep learning, additional pipeline was built to preprocess stock data into sequences for time-series analysis.

# 3. Project structure <br>
- /data: stores data
    - sp500_data.csv: file that stores the retrived and processed from data yfinance
- /utils: scripts
    - load_data.py: script used to data sourcing, feature engineering, data cleaning and saving.
    - dlmodel.py: script for deep learning models, with hyperparameter tuning; results are recorded in *4. models* section. 
    - nondlmodel.py: stript for non deep learning models, with hyperparameter tuning; results are recorded in *4. models* section.
- .gitignore: file to mark the file that does not need to be tracked
- main.py: run this function to train and evaluate all models.
- README.md: our writeup submission.
- requirement.txt: the libraries required to run this project. 
    
# 4. Models
## 4.1 Non Deep Learning Models
- Model: ARIMA(p,d,q) https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
- Hyperparameters:
    - p: The number of lag observations included in the model (autoregressive part).
    - d: The number of times that the raw observations are differenced to make the time series stationary (integrated part).
    - q: The size of the moving average window (moving average part).
- Hyperparameter tuning results
    - ARIMA(1,1,1): Base model; MSE = 0.00018002194040200658
    - ARIMA(2,1,2): General model; MSE = 0.0001798034530505031
    - ARIMA(0,1,1): Moving average over autoregressive; MSE = 0.0001846434052537533
    - ARIMA(1,0,1): Stationary; MSE = 0.00018125316737461887
    - ARIMA(3,1,0): Autoregressive-focus and no moving average; MSE = 0.00023445822030472924
- **Best performance: ARIMA(2,1,2) with MSE = 0.0001798034530505031**

## 4.2 Deep Learning Models
### 4.2.1 Features
- Preprocesses stock data into sequences for time-series analysis.
- LSTM network with configurable hyperparameters:
    - LSTM units
    - Dropout rate
    - Learning rate
    - Batch size
- TimeSeriesSplit for time-series cross-validation.
- Automatic hyperparameter tuning using grid search.
- Visualization of model performance with matplotlib.
### 4.2.2 Hyperparameter Tuning
The script uses grid search for hyperparameter optimization:
- Parameters:
    - `lstm_units`: Number of units in the LSTM layers.
    - `dropout_rate`: Dropout rate to prevent overfitting.
    - `learning_rate`: Learning rate for the Adam optimizer.
    - `batch_size`: Batch size for training.
The best parameters and their corresponding Mean Squared Error (MSE) are displayed after tuning.

### 4.2.3 Deep Learning Model Results
The final model's performance is evaluated on a test set, and predictions are visualized:
- Evaluation Metric:
    - Mean Squared Error (MSE) for model accuracy.
- Best Parameters:
    - `lstm_units`: 128
    - `dropout_rate`: 0.1
    - `learning_rate`: 0.001
    - `batch_size`: 16
- **Best MSE= 0.000170569299**
- Visualization: A comparison of true vs. predicted values is plotted to assess performance.
![SP500 Data Chart](https://i.imgur.com/8WQhPyf.png)

# 5. Comparision & Discussion
## 5.1 Expected Performance:
- MSE is low for both deep learning and non deep learning models: 
    - The low Mean Squared Error (MSE) indicates that the model is performing well on average, capturing the overall trend of the data and minimizing the prediction error across most points
- Both models are time-consuming: 
    - Evaluation time for each model is long due to time-series check, the models have to be fitted again in new step to generate prediction result for evaluation.
- Performance (MSE based): 
    - performance of the deep learning model (MSE=0.000170569299) is slightly better than non-deep learning model (MSE = 0.0001798034530505031).

## 5.2 Unexpected Performance:
- Despite the low MSE, the model struggles to predict sharp spikes or sudden fluctuations in the true values. This suggests that the model might be prioritizing minimizing errors in stable regions, leading to a trade-off where it underfits high-variance portions of the data.

## 5.3 Comparision
LSTM (deep learning model) can handle more features as input, while ARIMA (non deep learning model) can only handle signgle feature. This indicates for time series data, deep learning model has better potential in future improvement, where we can include more features in the model input.

## 5.4 Suggestions for Improvement:
- To address these unexpected behaviors while maintaining the low MSE, we might can perform more extensive hyperparameter tuning, especially for dropout rates, learning rates, and the number of LSTM units.
- To better track the market performance, we may also consider other technical features such as MACD and KDJ to aid the performance.


Below are requirements from the original assignment description.
-------
1. Project Description <br>
Build a model to generate predictions (regression or classification) on structured data, using a chosen dataset. 

2. Dataset equirements 
    - it cannot be from Kaggle
    - it cannot be a dataset you have worked with before.  
    
3. Conduct an end-to-end modeling process including data cleaning, feature engineering, modeling, and evaluation.  You may work individually or in a team of up to 4 people.
In your project you must train and compare two different types of models: 
    - A non deep learning model (any algorithm)
    - A deep learning model (any type, but must be trained from scratch - no APIs or transfer learning using existing pre-trained models)

Both models must be optimized using feature engineering and hyperparameter tuning. You should be sure to document your optimization process in your writeup.
