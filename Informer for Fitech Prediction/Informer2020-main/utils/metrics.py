import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def MAPE_eps(pred, true, eps=1e-8):
    true = true.astype(np.float64)
    pred = pred.astype(np.float64)
    return float(np.mean(np.abs(pred - true) / (np.abs(true) + eps)))

def r2_score(pred, true):
    true = np.array(true).astype(np.float64).ravel()
    pred = np.array(pred).astype(np.float64).ravel()

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)

    return 1 - ss_res / (ss_tot + 1e-8)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    mape_eps = MAPE_eps(pred, true)
    r2 = r2_score(pred, true)


    
    return mae,mse,rmse,mape,mspe,mape_eps,r2