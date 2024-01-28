"""LightGBM, CrossValidationを利用したベースラインモデル."""
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split

warnings.simplefilter("ignore")

# 共通設定
DATA_PATH = "../data"
RESULT_PATH = "../results"
SUMMARY_FILENAME = "summary.csv"
# 個別設定
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
MEMO = "損失関数にloglossではなくてAUCを利用．"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """データの前処理."""
    # 金額をobj->int型へ変換
    dollar_amount_cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
    for col in dollar_amount_cols:
        df[col] = df[col].apply(lambda x: x.replace("$", "").replace(".", "").replace(",", "")).astype(int).copy()
    # 年月関連を年と月で分ける
    ymd_cols = ["DisbursementDate", "ApprovalDate"]
    for col in ymd_cols:
        df[col + "_year"] = pd.to_datetime(df[col]).apply(lambda x: x.year)
        df[col + "_month"] = pd.to_datetime(df[col]).apply(lambda x: x.month)
        df = df.drop(columns=col)
    # category型への変換
    obj_cols = df.select_dtypes(include=object).columns
    df[obj_cols] = df[obj_cols].astype("category").copy()
    return df


def postprocess_prediction(y_pred_proba, negative_ratio):
    threshold = np.sort(y_pred_proba)[int(y_pred_proba.shape[0] * negative_ratio)]
    y_pred = np.where(y_pred_proba < threshold, 0, 1)
    return y_pred


def save_cv_result(result_df: pd.DataFrame):
    summary_path = os.path.join(RESULT_PATH, SUMMARY_FILENAME)
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
    else:
        df = pd.DataFrame({"experiment_name": [], "cv_score": [], "MEMO": [], "board_score": []})
    df = pd.concat([df, result_df]).drop_duplicates()
    df.to_csv(summary_path, index=False)


def main():
    # データ読み込み
    train_data = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), index_col=0)
    test_data = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), index_col=0)

    # frequency encoding
    encoding_target_cols = ["FranchiseCode", "RevLineCr", "LowDoc", "UrbanRural", "State", "BankState", "City", "Sector"]
    for col in encoding_target_cols:
        count_dict = dict(train_data[col].value_counts())
        train_data[f"{col}_freq_encoding"] = train_data[col].map(count_dict)
        test_data[f"{col}_freq_encoding"] = test_data[col].map(count_dict).fillna(1).astype(int)
    # 前処理
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # 説明変数と目的変数に分ける
    X = train_data.drop(columns="MIS_Status").copy()
    y = train_data["MIS_Status"].copy()
    X_test = test_data.copy()

    # ハイパーパラメータの設定
    params = {
        "objective": "binary",
        "metric": "auc",
    }

    # cross validation
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    scores = []
    negative_ratios = []
    for i, (train_eval_post_index, valid_index) in enumerate(kf.split(train_data)):
        # 学習データ，early_stopping用，評価用に分割
        X_train_eval_post = X.iloc[train_eval_post_index].copy()
        y_train_eval_post = y.iloc[train_eval_post_index].copy()
        X_train_eval, X_post, y_train_eval, y_post = train_test_split(X_train_eval_post, y_train_eval_post, test_size=0.2, random_state=42)
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

        # 後処理の閾値決定
        y_pred_proba = model.predict(X_post)
        best_score = 0
        negative_ratio = 0
        for i in np.linspace(0, 0.3, 30):
            y_pred = postprocess_prediction(y_pred_proba, i)
            score = f1_score(y_post, y_pred, average="macro")
            if score > best_score:
                negative_ratio = i
                best_score = score
        negative_ratios.append(negative_ratio)

        # 評価用データの予測
        y_pred_proba = model.predict(X_valid)
        y_pred = postprocess_prediction(y_pred_proba, negative_ratio)
        score = f1_score(y_valid, y_pred, average="macro")
        scores.append(score)

    # CVの結果の保存
    result_df = pd.DataFrame({"experiment_name": [EXPERIMENT_NAME], "cv_score": [np.mean(scores)], "MEMO": [MEMO], "board_score": [None]})
    save_cv_result(result_df)

    # テストデータへの予測
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = lgb.Dataset(X_train, label=y_train)
    eval_dataset = lgb.Dataset(X_eval, label=y_eval)
    # 学習
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=1000,
        valid_sets=[train_dataset, eval_dataset],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
    )
    # 予測
    y_pred_proba = model.predict(X_test)
    # 後処理
    best_negative_ratio = np.mean(negative_ratios)
    y_pred = postprocess_prediction(y_pred_proba, best_negative_ratio)

    # 結果の保存
    sample_submit = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"), index_col=0, header=None)  # 応募用サンプルファイル
    sample_submit[1] = y_pred
    sample_submit.to_csv(os.path.join(RESULT_PATH, f"{EXPERIMENT_NAME}.csv"), header=None)


main()
