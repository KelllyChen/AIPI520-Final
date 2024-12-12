import yfinance as yf
import os
from sklearn.preprocessing import StandardScaler

# load S&P 500 data to data folder
def load_SP500():
    # get data reference
    ticker_symbol = '^GSPC'
    sp500 = yf.Ticker(ticker_symbol)
    data = sp500.history(start='2000-01-01', end='2024-01-01', interval='1d')

    # data preprocessing
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    
    # feature engineering -> percentage change & scaling
    data['Open.Change.Pct'] = data['Open'].pct_change()
    data['Close.Change.Pct'] = data['Close'].pct_change()
    data['High.Change.Pct'] = data['High'].pct_change()
    data['Low.Change.Pct'] = data['Low'].pct_change()

    scaler = StandardScaler()
    data['Volume'] = scaler.fit_transform(data['Volume'].values.reshape(-1,1))

    # feature selection
    data.drop(columns=['Open', 'High', 'Low', 'Close','Dividends','Stock Splits'], inplace=True)
    data = data.iloc[1:]
    data.dropna(axis=0,inplace=True)

    # save data to csv
    try: 
        if os.path.exists('data/sp500_data.csv'):
            os.remove('data/sp500_data.csv')
        data.to_csv('data/sp500_data.csv',index=False)
        print("Data saved to csv")
        return True
    except Exception as e:
        print(e)
        print("Failed to save data to csv")
        return False

if __name__ == "__main__":
    load_SP500()