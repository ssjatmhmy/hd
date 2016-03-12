from sklearn.metrics import mean_squared_error, make_scorer


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_
    
RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)
