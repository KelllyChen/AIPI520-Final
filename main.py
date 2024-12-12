from utils.load_data import load_SP500
from utils.nondlmodel import nondlmain
from utils.dlmodel import main as dlmain


if __name__ == "__main__":
    # load S&P 500 data
    load_SP500()
    # run ARIMA models
    nondlmain()
    # run LSTM models
    dlmain()