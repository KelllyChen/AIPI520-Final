# AR model / learner -> Auther@ Harry
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found. Using zeros as starting parameters.")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters.")
def load_data_from_csv(file_path):
    # Load data from CSV file into a DataFrame
    data = pd.read_csv(file_path, index_col=None)
    return data['Close.Change.Pct']

def evaluate_arima_model(X, arima_order):
    print("Evaluating ARIMA model with order: ", arima_order)

    # Prepare training dataset
    train_size = int(len(X) * 0.8) # 80% of data for training
    train, test = X.iloc[0:train_size], X.iloc[train_size:]
    history = train.copy()
    predictions = []

    # Walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(method_kwargs={"maxiter": 500, "disp": 0})
        yhat = model_fit.forecast(steps=1).iloc[0]

        # prepare for the next step
        predictions.append(yhat)
        history = pd.concat([history, pd.Series([test.iloc[t]], index=[test.index[t]])])  

        if (t + 1) % 100 == 0:
            print(f"Progress: {t+1}/{len(test)}")
    
    predictions, test = zip(*[(pred, test.iloc[i]) for i, pred in enumerate(predictions) if not np.isnan(pred)])
    # Calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

def nondlmain():
    file_path = 'data/sp500_data.csv'
    data = load_data_from_csv(file_path)
    order_list = [(1,1,1),(2,1,2),(0,1,1),(1,0,1),(3,1,0)] 

    for order in order_list:
        result = f"{order} MSE: {evaluate_arima_model(data, order)}"
        print(result)
        with open('nondlresult.txt', 'a') as result_file:
            result_file.write(result + '\n')

if __name__ == "__main__":
    main()