import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def MAPE(y_test, y_pred):
    y_test = np.expm1(y_test)
    y_pred = np.expm1(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test) * 100) 

def RMSE(y_test, y_pred):
    y_test = np.expm1(y_test)
    y_pred = np.expm1(y_pred)
    return np.sqrt(((y_test - y_pred) ** 2).mean())

def load_data(file_name):
    nowdir = os.path.dirname(os.path.realpath(__file__))
    return pd.read_excel(nowdir + '/' + file_name)

def prepare_data(df, variable, target):
    X = df[variable]
    y = df[target]
    X = np.array(X).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    y = np.log1p(y)
    return X, y

def evaluate_model(X, y, iterations):
    RMSEerror_train = []
    RMSEerror_test = []
    MAPEerror_train = []
    MAPEerror_test = []

    for i in range(1, iterations + 1):
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=i)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=True, random_state=i)
        
        y_pred = np.mean(y_train)
        
        MAPE_train = MAPE(y_val, y_pred)
        MAPE_test = MAPE(y_test, y_pred)
        RMSE_train = RMSE(y_val, y_pred)
        RMSE_test = RMSE(y_test, y_pred)
        
        RMSEerror_train.append(RMSE_train)
        RMSEerror_test.append(RMSE_test)
        MAPEerror_train.append(MAPE_train)
        MAPEerror_test.append(MAPE_test)

    return {
        'MAPE_train_mean': np.mean(MAPEerror_train),
        'MAPE_train_std': np.std(MAPEerror_train),
        'MAPE_test_mean': np.mean(MAPEerror_test),
        'MAPE_test_std': np.std(MAPEerror_test),
        'RMSE_train_mean': np.mean(RMSEerror_train),
        'RMSE_train_std': np.std(RMSEerror_train),
        'RMSE_test_mean': np.mean(RMSEerror_test),
        'RMSE_test_std': np.std(RMSEerror_test)
    }

if __name__ == "__main__":
    df = load_data('2nd_lifetime_estimation.xlsx')
    variable = 'log(Var (ΔQ0.2C-1C(V)))'
    target = 'cycle'
    X, y = prepare_data(df, variable, target)
    results = evaluate_model(X, y, 1000)
    for metric, value in results.items():
        print(f'{metric}: {value}')