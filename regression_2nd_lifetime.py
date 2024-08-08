import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix

def MAPE(y_test, y_pred):
    y_test = np.power(10, y_test)
    y_pred = np.power(10, y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test) * 100)

def RMSE(y_test, y_pred):
    y_test = np.power(10, np.power(10, y_test))
    y_pred = np.power(10, np.power(10, y_pred))
    return np.sqrt(((y_test - y_pred) ** 2).mean())

def prepare_data(df, variable, target):
    X = np.log10(np.abs(df[variable].values.reshape(-1, 1)))
    target = np.log10(df[target].values.reshape(-1,1))
    y = np.log10(target)
    return X, y

def train_and_evaluate(X, y, alphas, iterations):
    data_all = pd.DataFrame()
    MAPE_train = []
    MAPE_test = []
    RMSE_train = []
    RMSE_test = []
    MAPE_params = []
    for i in range(1, iterations + 1):
        print(str(i) + f'/ {iterations}')
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=i)
        kf = KFold(n_splits=5, shuffle=True, random_state=i)
        param_grid = {'alpha': alphas}
        ridge_gsv = GridSearchCV(Ridge(), param_grid, scoring=make_scorer(MAPE, greater_is_better=False), cv=kf, refit=True)
        ridge_gsv.fit(X_trainval, y_trainval)
        y_pred_train = ridge_gsv.predict(X_trainval)
        y_pred_test = ridge_gsv.predict(X_test)
        
        y_list = np.hstack([X_test, np.power(10, y_test), np.power(10, y_pred_test)])
        df_tmp = pd.DataFrame(y_list, columns=['X_test', 'True', 'Predicted'])
        data_all = pd.concat([data_all, df_tmp])
        
        score_train = ridge_gsv.score(X_trainval, y_trainval)
        score_test = ridge_gsv.score(X_test, y_test)
        
        MAPE_train.append(score_train)
        MAPE_test.append(score_test)
        RMSE_train.append(RMSE(y_trainval, y_pred_train))
        RMSE_test.append(RMSE(y_test, y_pred_test))
        MAPE_params.append(ridge_gsv.best_params_)
        
    return MAPE_train, MAPE_test, RMSE_train, RMSE_test, MAPE_params, data_all

def process_and_round_data(data_all):
    data_all.columns = ['X_test', 'True', 'Predicted']
    data_all_round = data_all.round(4)
    print(data_all_round)

    data_truepred = data_all_round.groupby(['X_test', 'True']).mean().reset_index()
    print(data_truepred)
    return data_truepred

def plot_true_vs_predicted(data_truepred, plot_path):
    plt.scatter(data_truepred['True'], data_truepred['Predicted'])
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of True vs Predicted Values')
    plt.savefig(plot_path)
    #plt.show()

if __name__ == "__main__":
    nowdir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(nowdir, '2nd_lifetime_estimation.xlsx')
    variable = 'Var (ΔQ0.2C-1C(V))'  # [Var (ΔQ0.2C-1C(V)), ΔQ_Δcycle, SOH, DCIR at low SOC(5%)]
    target = 'cycle'
    df = pd.read_excel(file_path)
    X, y = prepare_data(df, variable, target)
    y_class = (np.power(10,np.power(10, y)) >= 550).astype(int).ravel()
    
    # Ridge Regression part
    alphas = [x * 10**exp for exp in range(-8, -3) for x in range(1, 10)] + [x * 10**-3 for x in range(1, 11)]
    MAPE_train, MAPE_test, RMSE_train, RMSE_test, MAPE_params, data_all = train_and_evaluate(X, y, alphas, 1000)
    data_truepred = process_and_round_data(data_all)
    plot_true_vs_predicted(data_truepred, os.path.join(nowdir, f'true_vs_predicted_{variable}.png'))
    print(f'Train Mean MAPE: {np.mean(MAPE_train)}')
    print(f'Test Mean MAPE: {np.mean(MAPE_test)}')
    print(f'Train Mean RMSE: {np.mean(RMSE_train)}')
    print(f'Test Mean RMSE: {np.mean(RMSE_test)}')

    # Combine all results into a single DataFrame
    results_df = pd.DataFrame({
        'True': data_truepred['True'],
        'Predicted': data_truepred['Predicted'],
        'MAPE_train': np.mean(MAPE_train),
        'MAPE_test': np.mean(MAPE_test),
        'RMSE_train': np.mean(RMSE_train),
        'RMSE_test': np.mean(RMSE_test)
    })

    # Save combined results to a single CSV file
    results_df.to_csv(os.path.join(nowdir, f'combined_results_{variable}.csv'), index=False)