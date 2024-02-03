"""LightGBM, CrossValidationを利用したベースラインモデル."""
import os
import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split

warnings.simplefilter("ignore")

# 共通設定
DATA_PATH = "../data"
RESULT_PATH = "../results"
SUMMARY_FILENAME = "summary.csv"
# 個別設定
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
MEMO = "20240129_02のbest_negative_ratioとbest_weightを利用．（cvは実際より高く出るはず）"


@dataclass
class Params:
    num_boost_round = 1000
    early_stopping_round = 200
    seed = 42
    lightgbm_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "random_seed": seed,
    }
    catboost_params = {
        "learning_rate": 0.05,
        "iterations": num_boost_round,
        "random_seed": seed,
        "verbose": False,
    }
    negative_ratio = 0.08
    weight = [0.3, 0.7]
    methods = ["LightGBM", "CatBoost"]
    categorical_features = ["RevLineCr", "LowDoc", "UrbanRural", "State", "Sector", "City", "BankState"]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """データの前処理."""
    # 金額をobj->int型へ変換
    dollar_amount_cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
    for col in dollar_amount_cols:
        df[col] = df[col].apply(lambda x: x.replace("$", "").replace(".", "").replace(",", "")).astype(int).copy()
    #
    # 年月関連を年と月で分ける
    ymd_cols = ["DisbursementDate", "ApprovalDate"]
    for col in ymd_cols:
        df[col + "_year"] = pd.to_datetime(df[col]).apply(lambda x: x.year)
        df[col + "_month"] = pd.to_datetime(df[col]).apply(lambda x: x.month)
        df = df.drop(columns=col)
    # null処理
    df[Params.categorical_features] = df[Params.categorical_features].fillna("Unknown")
    # category型への変換
    obj_cols = df.select_dtypes(include=object).columns
    df[obj_cols] = df[obj_cols].astype("category").copy()
    return df


def lightgbm_training(X_train, y_train, X_eval, y_eval):
    # lightgbm用データセットに変換
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=Params.categorical_features)
    eval_dataset = lgb.Dataset(X_eval, label=y_eval, categorical_feature=Params.categorical_features)

    # LightGBMモデルの学習
    model = lgb.train(
        Params.lightgbm_params,
        train_dataset,
        num_boost_round=Params.num_boost_round,
        valid_sets=[train_dataset, eval_dataset],
        callbacks=[lgb.early_stopping(stopping_rounds=Params.early_stopping_round, verbose=True)],
    )

    return model


def catboost_training(X_train, y_train, X_eval, y_eval):
    train_pool = Pool(X_train, label=y_train, cat_features=Params.categorical_features)
    eval_pool = Pool(X_eval, label=y_eval, cat_features=Params.categorical_features)
    model = CatBoostClassifier(**Params.catboost_params)
    model.fit(train_pool, eval_set=[eval_pool], early_stopping_rounds=Params.early_stopping_round, use_best_model=True)
    return model


def cv_training(method, X, y):
    def _train(X_train, y_train, X_eval, y_eval, X_valid):
        if type(X_valid) == pd.core.frame.DataFrame:
            if method == "LightGBM":
                model = lightgbm_training(X_train, y_train, X_eval, y_eval)
                pred_proba = model.predict(X_valid)
            elif method == "CatBoost":
                model = catboost_training(X_train, y_train, X_eval, y_eval)
                pred_proba = model.predict_proba(X_valid)[:, 1]
        else:
            if method == "LightGBM":
                model = lightgbm_training(X_train, y_train, X_eval, y_eval)
            elif method == "CatBoost":
                model = catboost_training(X_train, y_train, X_eval, y_eval)
            pred_proba = None
        return model, pred_proba

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    valid_pred_proba = np.zeros((X.shape[0]))
    for i, (train_eval_index, valid_index) in enumerate(kf.split(X)):
        # 学習データ，early_stopping用，評価用に分割
        X_train_eval = X.iloc[train_eval_index].copy()
        y_train_eval = y.iloc[train_eval_index].copy()
        X_train, X_eval, y_train, y_eval = train_test_split(X_train_eval, y_train_eval, test_size=0.25, random_state=42)
        X_valid = X.iloc[valid_index].copy()
        # 学習
        model, pred_proba = _train(X_train, y_train, X_eval, y_eval, X_valid)
        valid_pred_proba[valid_index] = pred_proba
    # 結果予測用モデルの学習
    X_train, X_eval, y_train, y_eval = train_test_split(X_train_eval, y_train_eval, test_size=0.2, random_state=42)
    model, _ = _train(X_train, y_train, X_eval, y_eval, X_valid=None)
    return model, valid_pred_proba


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


def Preprocessing():
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
    return X, y, X_test


def Learning(methods, X, y):
    models = []
    valid_pred_probas = np.zeros((len(methods), X.shape[0]))
    # 学習
    for i, method in enumerate(methods):
        model, valid_pred_proba = cv_training(method, X, y)
        models.append(model)
        valid_pred_probas[i] = valid_pred_proba
    # アンサンブル
    y_pred_proba = np.average(valid_pred_probas, axis=0, weights=Params.weight)
    # 後処理
    y_pred = postprocess_prediction(y_pred_proba, Params.negative_ratio)
    # CV結果の保存
    score = f1_score(y, y_pred, average="macro")
    result_df = pd.DataFrame({"experiment_name": [EXPERIMENT_NAME], "cv_score": [score], "MEMO": [MEMO], "board_score": [None]})
    save_cv_result(result_df)
    return models


def Predicting(X_test, models):
    test_pred_probas = np.zeros((len(models), X_test.shape[0]))
    for i, model in enumerate(models):
        test_pred_probas[i] = model.predict(X_test)
    # 予測
    y_pred_proba = np.average(test_pred_probas, axis=0, weights=Params.weight)
    # 後処理
    y_pred = postprocess_prediction(y_pred_proba, Params.negative_ratio)

    # 結果の保存
    sample_submit = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"), index_col=0, header=None)  # 応募用サンプルファイル
    sample_submit[1] = y_pred
    sample_submit.to_csv(os.path.join(RESULT_PATH, f"{EXPERIMENT_NAME}.csv"), header=None)


def main():
    X, y, X_test = Preprocessing()

    models = Learning(Params.methods, X, y)

    Predicting(X_test, models)


if __name__ == "__main__":
    main()
