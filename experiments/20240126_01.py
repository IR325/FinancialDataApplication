"""LightGBM, CrossValidationを利用したベースラインモデル."""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split

# 共通設定
DATA_PATH = "../data"
RESULT_PATH = "../results"
SUMMARY_FILENAME = "summary.csv"
# 個別設定
EXPERIMENT_NAME = "20240126_01"
RESULT_FILENAME = f"{EXPERIMENT_NAME}.csv"
MEMO = "提供されているすべての特徴量を利用"


def main():
    # データ読み込み
    train_data = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), index_col=0)
    test_data = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), index_col=0)

    # 前処理
    obj_cols = train_data.select_dtypes(include=object).columns
    train_data[obj_cols] = train_data[obj_cols].astype("category")
    test_data[obj_cols] = test_data[obj_cols].astype("category")

    # 説明変数と目的変数に分ける
    X = train_data.drop(columns="MIS_Status").copy()
    y = train_data["MIS_Status"].copy()
    X_test = test_data.copy()

    # ハイパーパラメータの設定
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
    }

    # cross validation
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    scores = []
    for i, (train_eval_index, valid_index) in enumerate(kf.split(train_data)):
        # 学習データ，early_stopping用，評価用に分割
        X_train_eval = X.iloc[train_eval_index].copy()
        y_train_eval = y.iloc[train_eval_index].copy()
        X_train, X_eval, y_train, y_eval = train_test_split(X_train_eval, y_train_eval, test_size=0.25, random_state=42)
        X_valid = X.iloc[valid_index].copy()
        y_valid = y.iloc[valid_index].copy()

        # lightgbm用データセットに変換
        train_dataset = lgb.Dataset(X_train, label=y_train)
        eval_dataset = lgb.Dataset(X_eval, label=y_eval)

        # LightGBMモデルの学習
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=1000,
            valid_sets=[train_dataset, eval_dataset],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
        )

        # 評価用データの予測
        y_pred_proba = model.predict(X_valid)
        y_pred = np.where(y_pred_proba < 0.5, 0, 1)
        score = f1_score(y_valid, y_pred, average="macro")
        scores.append(score)

    # CVの結果の保存
    summary_path = os.path.join(RESULT_PATH, SUMMARY_FILENAME)
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
    else:
        df = pd.DataFrame({"experiment_name": [], "cv_score": [], "MEMO": [], "board_score": []})
    result_df = pd.DataFrame({"experiment_name": [EXPERIMENT_NAME], "cv_score": [np.mean(scores)], "MEMO": [MEMO], "board_score": [None]})
    df = pd.concat([df, result_df])
    df.drop_duplicates().to_csv(summary_path, index=False)

    # テストデータへの予測
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, random_state=42)
    train_dataset = lgb.Dataset(X_train, label=y_train)
    eval_dataset = lgb.Dataset(X_eval, label=y_eval)
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=1000,
        valid_sets=[train_dataset, eval_dataset],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
    )
    y_pred_proba = model.predict(X_test)
    y_pred = np.where(y_pred_proba < 0.5, 0, 1)
    # 結果の保存
    sample_submit = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"), index_col=0, header=None)  # 応募用サンプルファイル
    sample_submit[1] = y_pred
    sample_submit.to_csv(os.path.join(RESULT_PATH, RESULT_FILENAME), header=None)


main()
