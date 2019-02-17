"""Predictors used in the MissingnessPredictor class"""

import xgboost as xgb

def xgb_model(X, y, param=None, num_round=20):
    """XGB predictor model"""
    if param is None:
        param = {
            'max_depth': 3,
            'eta': 0.3,
            'silent': 1,
            'objective': 'multi:softprob',
            'num_class': 2
        }
    xgb_ = xgb.DMatrix(X, y)
    bst = xgb.train(param, xgb_, num_round)
    preds = bst.predict(xgb_)[:, 1]
    return preds
