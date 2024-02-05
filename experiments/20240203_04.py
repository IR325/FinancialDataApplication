"""LightGBM, CrossValidationを利用したベースラインモデル."""
import itertools
import os
import pickle
import random
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
MEMO = "20240203_03.pyをベース．損失関数はauc.early stoppingはf1"


def lgb_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return "f1score", f1_score(y_true, np.where(y_pred >= 0.5, 1, 0), average="macro"), True


@dataclass
class Params:
    n_splits = 5
    num_boost_round = 1000
    early_stopping_round = 200
    seed = 42
    lightgbm_params = {
        "objective": "binary",
        "metric": "auc",
        "random_seed": seed,
    }
    catboost_params = {
        "learning_rate": 0.05,
        "iterations": num_boost_round,
        "random_seed": seed,
        "verbose": False,
    }
    methods = ["LightGBM", "CatBoost"]
    drop_cols = ["ApprovalDate", "DisbursementDate", "FranchiseCode", "NewExist", "BankState", "City"]
    encoding_features = ["FranchiseCode", "RevLineCr", "LowDoc", "UrbanRural", "State", "BankState", "Sector", "City"]
    categorical_features = list(set(encoding_features) - set(drop_cols))
    weights = [np.array(comb) for comb in list(itertools.product(np.arange(0, 1.01, 0.1), repeat=len(methods))) if sum(comb) == 1]
    negative_ratios = np.arange(0, 0.2, 0.02)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """データの前処理."""
    # frequency encoding
    encoding_cols = Params.encoding_features
    df[encoding_cols] = df[encoding_cols].fillna("Unknown")
    for col in encoding_cols:
        count_dict = dict(df[col].value_counts())
        df[f"{col}_freq_encoding"] = df[col].map(count_dict)
    # 金額をobj->int型へ変換
    dollar_amount_cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
    for col in dollar_amount_cols:
        df[col] = df[col].apply(lambda x: x.replace("$", "").replace(".", "").replace(",", "")).astype(int).copy()
    # 融資支払日と承認日の差を特徴量として追加
    # df["date_diff"] = (pd.to_datetime(df["DisbursementDate"]) - pd.to_datetime(df["ApprovalDate"])).apply(lambda x: x.days)
    # 年月関連を年と月で分ける
    # ymd_cols = ["DisbursementDate", "ApprovalDate"]
    # for col in ymd_cols:
    #     df[col + "_year"] = pd.to_datetime(df[col]).apply(lambda x: x.year)
    #     df[col + "_month"] = pd.to_datetime(df[col]).apply(lambda x: x.month)
    # category型への変換
    obj_cols = df.select_dtypes(include=object).columns
    df[obj_cols] = df[obj_cols].astype("category").copy()
    # 不要なカラムを削除
    df = df.drop(columns=Params.drop_cols)
    print(f"特徴量数: {len(df.columns)}")
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
        feval=lgb_metric,
        callbacks=[lgb.early_stopping(stopping_rounds=Params.early_stopping_round, verbose=True)],
    )

    return model


def catboost_training(X_train, y_train, X_eval, y_eval):
    train_pool = Pool(X_train, label=y_train, cat_features=Params.categorical_features)
    eval_pool = Pool(X_eval, label=y_eval, cat_features=Params.categorical_features)
    model = CatBoostClassifier(**Params.catboost_params)
    model.fit(train_pool, eval_set=[eval_pool], early_stopping_rounds=Params.early_stopping_round, use_best_model=True)
    return model


def _train(method, X_train, y_train, X_eval, y_eval):
    if method == "LightGBM":
        model = lightgbm_training(X_train, y_train, X_eval, y_eval)
    elif method == "CatBoost":
        model = catboost_training(X_train, y_train, X_eval, y_eval)
    return model


def _predict(method, model, X):
    if method == "LightGBM":
        pred_proba = model.predict(X)
    elif method == "CatBoost":
        pred_proba = model.predict_proba(X)[:, 1]
    return pred_proba


class AdvancedKfold:
    def __init__(self, n_splits, random_state):
        self.n_splits = n_splits
        random.seed(random_state)

    def split(self, X: pd.DataFrame):
        indicies = list(X.index)
        indicies = random.sample(indicies, len(indicies))
        n_sample = int(len(indicies) / self.n_splits)
        for i in range(self.n_splits):
            # 抽出
            eval_indicies = indicies[n_sample * i : n_sample * (i + 1)]
            if n_sample * (i + 2) <= len(indicies):
                valid_indicies = indicies[n_sample * (i + 1) : n_sample * (i + 2)]
            else:
                valid_indicies = indicies[0:n_sample]
            train_indicies = list(set(indicies) - set(eval_indicies) - set(valid_indicies))
            yield train_indicies, eval_indicies, valid_indicies


def cv_training(methods, X, y):
    best_weights = np.zeros((Params.n_splits, len(methods)))
    best_negative_ratios = []
    scores = []
    kf = KFold(n_splits=Params.n_splits, random_state=42, shuffle=True)
    for i, (train_eval_tune_index, valid_index) in enumerate(kf.split(X)):
        # データの分割
        X_train_eval_tune = X.iloc[train_eval_tune_index].copy()
        y_train_eval_tune = y.iloc[train_eval_tune_index].copy()
        X_train_eval, X_tune, y_train_eval, y_tune = train_test_split(X_train_eval_tune, y_train_eval_tune, test_size=0.25, random_state=Params.seed)
        X_train, X_eval, y_train, y_eval = train_test_split(X_train_eval, y_train_eval, test_size=0.33, random_state=Params.seed)
        X_valid = X.iloc[valid_index].copy()
        y_valid = y.iloc[valid_index].copy()

        tune_pred_probas = np.zeros((len(methods), X_tune.shape[0]))
        valid_pred_probas = np.zeros((len(methods), X_valid.shape[0]))

        for j, method in enumerate(methods):
            # 学習
            model = _train(method, X_train, y_train, X_eval, y_eval)
            # validに対する予測
            valid_pred_probas[j] = _predict(method, model, X_valid)
            # tuneに対する予測
            tune_pred_probas[j] = _predict(method, model, X_tune)

        # モデルの重み，クラス比率の最適化
        best_score = 0
        for weight, negative_ratio in itertools.product(Params.weights, Params.negative_ratios):
            y_pred_proba = np.average(tune_pred_probas, axis=0, weights=weight)
            y_pred = postprocess_prediction(y_pred_proba, negative_ratio)
            score = f1_score(y_tune, y_pred, average="macro")
            if score > best_score:
                best_weight = weight
                best_negative_ratio = negative_ratio
                best_score = score
        best_weights[i] = best_weight
        best_negative_ratios.append(best_negative_ratio)
        # validの処理
        valid_pred_proba = np.average(valid_pred_probas, axis=0, weights=best_weight)
        valid_pred = postprocess_prediction(valid_pred_proba, best_negative_ratio)
        scores.append(f1_score(y_valid, valid_pred, average="macro"))
    return np.mean(scores), np.mean(best_weights, axis=0), np.mean(best_negative_ratios)


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
    # CV
    cv_score, cv_best_weight, cv_best_negative_ratio = cv_training(methods, X, y)
    print(f"best score: {cv_score}")
    print(f"best weight: {cv_best_weight}")
    print(f"best negative ratio: {cv_best_negative_ratio}")
    result_df = pd.DataFrame({"experiment_name": [EXPERIMENT_NAME], "cv_score": [cv_score], "MEMO": [MEMO], "board_score": [None]})
    save_cv_result(result_df)
    # 各モデルの学習
    for method in methods:
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, random_state=Params.seed)
        model = _train(method, X_train, y_train, X_eval, y_eval)
        models.append(model)
        with open(os.path.join(RESULT_PATH, f"{EXPERIMENT_NAME}_{method}_model.pickle"), mode="wb") as f:
            pickle.dump(model, f)
    return models, cv_best_weight, cv_best_negative_ratio


def Predicting(X_test, models, best_weight, best_negative_ratio):
    test_pred_probas = np.zeros((len(models), X_test.shape[0]))
    for i, model in enumerate(models):
        test_pred_probas[i] = model.predict(X_test)
    # 予測
    y_pred_proba = np.average(test_pred_probas, axis=0, weights=best_weight)
    # 後処理
    y_pred = postprocess_prediction(y_pred_proba, best_negative_ratio)

    # 結果の保存
    sample_submit = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"), index_col=0, header=None)  # 応募用サンプルファイル
    sample_submit[1] = y_pred
    sample_submit.to_csv(os.path.join(RESULT_PATH, f"{EXPERIMENT_NAME}.csv"), header=None)


def main():
    X, y, X_test = Preprocessing()

    models, best_weight, best_negative_ratio = Learning(Params.methods, X, y)

    Predicting(X_test, models, best_weight, best_negative_ratio)


if __name__ == "__main__":
    main()
