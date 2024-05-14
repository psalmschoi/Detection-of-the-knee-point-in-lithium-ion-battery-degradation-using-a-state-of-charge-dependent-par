import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def MAPE(y_test, y_pred):
    y_test = np.expm1(y_test)
    y_pred = np.expm1(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test) * 100)

def RMSE(y_test, y_pred):
    y_test = np.expm1(y_test)
    y_pred = np.expm1(y_pred)
    return np.sqrt(((y_test - y_pred) ** 2).mean())

def prepare_data(df, variable, target):
    X = np.array(df[variable]).reshape(-1,1)
    y = np.log1p(df[target].values.reshape(-1,1))
    return X, y

def train_and_evaluate(X, y, alphas, iterations):
    data_all = pd.DataFrame()
    RMSE_train = []
    RMSE_test = []
    RMSE_params = []
    for i in range(1, iterations + 1):
        print(str(i) +'/ ' + str(iterations))
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=i)
        kf = KFold(n_splits=5, shuffle=True, random_state=i)
        param_grid = {'alpha': alphas}
        ridge_gsv = GridSearchCV(Ridge(), param_grid, scoring=make_scorer(RMSE, greater_is_better=False), cv=kf, refit=True)
        ridge_gsv.fit(X_trainval, y_trainval)
        y_pred = ridge_gsv.predict(X_test)
        y_list = np.hstack([np.expm1(y_test), np.expm1(y_pred)])
        df_tmp = pd.DataFrame(y_list)
        data_all = data_all.append(df_tmp)
        score = ridge_gsv.score(X_test, y_test)
        RMSE_train.append(ridge_gsv.best_score_)
        RMSE_test.append(score)
        RMSE_params.append(ridge_gsv.best_params_)
    return RMSE_train, RMSE_test, RMSE_params, data_all

def process_and_round_data(data_all):
    data_all.columns = ['True', 'Predicted']
    data_all_round = data_all.round(4)
    print(data_all_round)
    data_truepred = pd.DataFrame(data_all_round['True'].unique(), columns=['True'])
    data_truepred['Predicted'] = data_truepred['True'].apply(lambda x: data_all_round[data_all_round['True'] == x]['Predicted'].mean())
    print(data_truepred)
    return data_truepred

def plot_true_vs_predicted(data_truepred):
    plt.scatter(data_truepred['True'], data_truepred['Predicted'])
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of True vs Predicted Values')
    plt.show()


if __name__ == "__main__":
    nowdir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(nowdir, '2nd_lifetime_estimation.xlsx')
    variable = 'log(Var (ΔQ0.2C-1C(V)))' # [log(Var (ΔQ0.2C-1C(V))), ΔQ/Δcycle, SOH, DCIR at low SOC(5%)]
    target = 'cycle'
    df = pd.read_excel(file_path)
    X, y = prepare_data(df, variable, target)
    alphas = alphas = [x * 10**exp for exp in range(-8, -3) for x in range(1, 10)] + [x * 10**-3 for x in range(1, 11)]
    RMSE_train, RMSE_test, RMSE_params, data_all = train_and_evaluate(X, y, alphas, 1000)
    data_truepred = process_and_round_data(data_all)
    plot_true_vs_predicted(data_truepred)
    print(f'Train Mean RMSE: {np.mean(RMSE_train)}')
    print(f'Test Mean RMSE: {np.mean(RMSE_test)}')