import glob
import pandas as pd
import numpy as np
from sklearn import metrics
from functools import partial
from scipy.optimize import fmin
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression


def run_training(pred_df,fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred","lr_cnt_pred","rf_svd_pred"]].values
    xvalid = valid_df[["lr_pred","lr_cnt_pred","rf_svd_pred"]].values

    scl = StandardScaler()
    xtrain = scl.fit_transform(xtrain)
    xvalid = scl.transform(xvalid)

    opt = LinearRegression()
    opt.fit(xtrain, train_df.sentiment.values)
    preds = opt.predict(xvalid)
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"{fold}, {auc}")

    return opt.coef_



if __name__ == "__main__":
    files = glob.glob("../model_pred/*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on="id", how = "left")
    targets = df.sentiment.values
    pred_cols = ["lr_pred","lr_cnt_pred","rf_svd_pred"]
    coefs = []

    for j in range(5):
        coefs.append(run_training(df, j))


    coef = np.array(coefs)
    print(coefs)
    coefs = np.mean(coefs, axis=0)
    print(coefs)


    wt_avg = (
        coefs[0] * df.lr_pred.values
        + coefs[1] * df.lr_cnt_pred.values
        + coefs[2] * df.rf_svd_pred.values
    )
    print("optimal auc after finding coefs")
    print(metrics.roc_auc_score(targets, wt_avg))