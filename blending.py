import glob
import pandas as pd
import numpy as np
from sklearn import metrics

if __name__ == "__main__":
    files = glob.glob("../model_pred/*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on="id", how = "left")
    #print(df.head(10))
    targets = df.sentiment.values

    pred_cols = ["lr_pred","lr_cnt_pred","rf_svd_pred"]

    for col in pred_cols:
        auc = metrics.roc_auc_score(targets, df[col].values)
        print(f"{col}, overall_auc={auc}")

## The first method of blending is called averaging

    print('average')
    avg_pred = np.mean(df[["lr_pred","lr_cnt_pred","rf_svd_pred"]].values, axis=1)
    print(metrics.roc_auc_score(targets,avg_pred))

## This below method is called weighted average. 
## how you choose the weight. the weight below i choosed are random values. 
## The correct way of choosing the weight is to Optimize it.

    print('weighted average')

    lr_pred = df.lr_pred.values
    lr_cnt_pred = df.lr_cnt_pred.values
    rf_svd_pred = df.rf_svd_pred.values
    avg_pred = (lr_pred +3 * lr_cnt_pred + rf_svd_pred) / 5 
    print(metrics.roc_auc_score(targets,avg_pred))

    print("rank averaging")

    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    rf_svd_pred = df.rf_svd_pred.rank().values
    avg_pred = (lr_pred +3 * lr_cnt_pred + rf_svd_pred) / 3 
    print(metrics.roc_auc_score(targets,avg_pred))


    print("weighted rank averaging")

    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    rf_svd_pred = df.rf_svd_pred.rank().values
    avg_pred = (lr_pred + 3 * lr_cnt_pred + rf_svd_pred) / 5
    print(metrics.roc_auc_score(targets,avg_pred))

    
