# Data manipulation
import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from timeit import default_timer as timer
from hyperopt import STATUS_OK
from hyperopt import hp
from sklearn import preprocessing
import csv
from sklearn.metrics import f1_score
import numpy as np
# custom evaluation metric using f1 from sklearn to balance between recall and precision
def f_score(pred,train_data):
    ground_truth = train_data.get_label()
    pred = (pred>0.5).astype(int)
    print(pred.shape)
    print(ground_truth.shape)
    label = [0,1]
    return 'f1', f1_score(y_true= ground_truth,y_pred= pred,labels=label,average='micro'), True
N_FOLDS = 5
MAX_EVALS = 1000
global ITERATION
global train_set
space = {
    'boosting_type': hp.choice('boosting_type',
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'is_unbalance': hp.choice('is_unbalance', [True, False]),
}
def objective(hyperparameters):
    global ITERATION

    ITERATION += 1

    # Using early stopping to find number of trees trained
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']

    # Retrieve the subsample
    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type and subsample to top level keys
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
    hyperparameters['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    start = timer()
    run_time = timer() - start
    # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round=10000, nfold=N_FOLDS,
                        early_stopping_rounds=100,feval=f_score, seed=0)
    print(cv_results)
    best_score = cv_results['f1-mean'][-1]
    loss = 1 - best_score
    n_estimators = len(cv_results['f1-mean'])
    hyperparameters['n_estimators'] = n_estimators

    # Write to the csv file
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
    of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}
if __name__ == '__main__':
    trials = Trials()
    OUT_FILE = 'bayes_test_f1.csv'
    of_connection = open(OUT_FILE, 'w')
    writer = csv.writer(of_connection)
    ITERATION = 0
    train = pd.read_csv('data/credit-data.csv')
    train_label = train['default payment next month']
    features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
                'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    cat = ['SEX', 'EDUCATION', 'MARRIAGE']
    train = train.drop(columns=['default payment next month','ID'])
    train_set = lgb.Dataset(data=train, label=train_label, feature_name=features,categorical_feature=cat,free_raw_data=False)
    headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
    writer.writerow(headers)
    of_connection.close()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials,
                max_evals=MAX_EVALS)





