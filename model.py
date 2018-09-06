import numpy as np
import lightgbm as lgb
#import the f-score function from tuning
import tuning
class CreditModel:
    def __init__(self):
        """
        Instantiates the model object, creating class variables if needed.
        """
        self.param = {'boosting_type': 'gbdt',
                      'colsample_bytree': 0.9005139582685586,
                      'is_unbalance': True,
                      'learning_rate': 0.21095433542302736,
                      'min_child_samples': 50,
                      'num_leaves': 34,
                      'reg_alpha': 0.6584706652182384,
                      'reg_lambda': 0.4627822841155883,
                      'subsample_for_bin': 60000,
                      'subsample': 0.782248105670204,
                      'verbose': 1, 'n_estimators': 18}



        self.model = None
        # TODO: Initialize your model object.
        pass

    def fit(self, X_train, y_train):
        """
        Fits the model based on the given `X_train` and `y_train`.

        You should somehow manipulate and store this data to your model class
        so that you can make predictions on new testing data later on.
        """
        # since the import data does not have the column name, we should relabel them so that
        # lightgbm can know which is categorical features
        features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        cat = ['SEX', 'EDUCATION', 'MARRIAGE']
        dataset = lgb.Dataset(data= X_train,label= y_train,feature_name=features,categorical_feature=cat)
        # we will modify the number of round based on tuning result
        self.model = lgb.train(self.param, dataset, feval=tuning.f_score)

        # TODO: Fit your model based on the given X and y.
        pass

    def predict(self, X_test):
        """
        Returns `y_hat`, a prediction for a given `X_test` after fitting.

        You should make use of the data that you stored/computed in the
        fitting phase to make your prediction on this new testing data.
        """
        y_hat = self.model.predict(X_test)
        # y_hat = np.round(y_hat)
        # TODO: Predict on `X_test` based on what you learned in the fit phase.
        # return np.random.randint(2, size=len(X_test))
        return y_hat