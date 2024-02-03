"""LightGBM, CrossValidationを利用したベースラインモデル."""
import itertools
import os
import pickle
import random
import time
import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import optuna
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
MEMO = "catboostのパラメータもチューニング対象に"


@dataclass
class Params:
    n_splits = 5
    n_trials = 100
    num_boost_round = 1000
    early_stopping_round = 200
    seed = 42
    methods = ["LightGBM", "CatBoost"]
    drop_cols = []
    categorical_features = ["FranchiseCode", "RevLineCr", "LowDoc", "UrbanRural", "State", "BankState", "Sector", "City", "Franchise_or_not"]
    encoding_target_cols = ["FranchiseCode", "RevLineCr", "LowDoc", "UrbanRural", "State", "BankState", "Sector", "City", "Franchise_or_not"]
    lgb_constant_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "n_estimators": num_boost_round,
        "verbosity": -1,
        "random_seed": seed,
    }
    catboost_constant_params = {
        "learning_rate": 0.05,
        "iterations": num_boost_round,
        "random_seed": seed,
        "verbose": False,
    }
    cv_best_params = None

    def get_lightgbm_params_range(self, trial):
        variable_params = {
            "feature_fraction": trial.suggest_float("lgb_feature_fraction", 0.1, 0.9),
            "bagging_fraction": trial.suggest_float("lgb_bagging_fraction", 0.1, 0.9),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 7, 62),
            "lambda_l1": trial.suggest_float("lgb_lambda_l1", 0, 5),
            "lambda_l2": trial.suggest_float("lgb_lambda_l2", 0, 5),
        }
        return self.lgb_constant_params | variable_params

    def get_catboost_params_range(self, trial):
        variable_params = {
            "depth": trial.suggest_int("catboost_depth", 1, 10),
            "subsample": trial.suggest_float("catboost_subsample", 0.05, 1.0),
            "colsample_bylevel": trial.suggest_float("catboost_colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("catboost_min_data_in_leaf", 1, 100),
            "l2_leaf_reg": trial.suggest_float("catboost_l2_leaf_reg", 0, 5),
        }
        return self.catboost_constant_params | variable_params


def _dict_average(dicts: list) -> dict:
    averaged_dict = {}
    for k, v in dicts[0].items():
        averaged_dict[k] = v
    for d in dicts[1:]:
        for k, v in d.items():
            if type(v) == int:
                averaged_dict[k] = round((averaged_dict[k] + v) / len(dicts))
            else:
                averaged_dict[k] = (averaged_dict[k] + v) / len(dicts)

    return averaged_dict


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """データの前処理."""
    # 不要なカラムを削除
    df = df.drop(columns=Params.drop_cols)
    # 非フランチャイズかフランチャイズかの2値の特徴量を追加
    df["Franchise_or_not"] = (
        df["FranchiseCode"]
        .mask(((df["FranchiseCode"] == 0) | (df["FranchiseCode"] == 1)), 0)
        .mask(((df["FranchiseCode"] != 0) & (df["FranchiseCode"] != 1)), 1)
    )
    # 金額をobj->int型へ変換
    dollar_amount_cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
    for col in dollar_amount_cols:
        df[col] = df[col].apply(lambda x: x.replace("$", "").replace(".", "").replace(",", "")).astype(int).copy()
    # 融資支払日と承認日の差を特徴量として追加
    df["date_diff"] = (pd.to_datetime(df["DisbursementDate"]) - pd.to_datetime(df["ApprovalDate"])).apply(lambda x: x.days)
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
    # frequency encoding
    for col in Params.encoding_target_cols:
        count_dict = dict(df[col].value_counts())
        df[f"{col}_freq_encoding"] = df[col].map(count_dict).astype(int)
    return df


def lightgbm_training(X_train, y_train, X_eval, y_eval, trial):
    # lightgbm用データセットに変換
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=Params.categorical_features)
    eval_dataset = lgb.Dataset(X_eval, label=y_eval, categorical_feature=Params.categorical_features)

    if trial:
        params = Params().get_lightgbm_params_range(trial)
    else:
        if Params.cv_best_params:
            best_params = {k.split("lgb_")[1]: v for k, v in Params.cv_best_params.items() if "lgb" in k}
        else:
            best_params = {k.split("lgb_")[1]: v for k, v in Params.study.best_params.items() if "lgb" in k}
        params = Params.lgb_constant_params | best_params

    # LightGBMモデルの学習
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=Params.num_boost_round,
        valid_sets=[train_dataset, eval_dataset],
        callbacks=[lgb.early_stopping(stopping_rounds=Params.early_stopping_round, verbose=True)],
    )

    return model


def catboost_training(X_train, y_train, X_eval, y_eval, trial):
    train_pool = Pool(X_train, label=y_train, cat_features=Params.categorical_features)
    eval_pool = Pool(X_eval, label=y_eval, cat_features=Params.categorical_features)

    if trial:
        params = Params().get_catboost_params_range(trial)
    else:
        if Params.cv_best_params:
            best_params = {k.split("catboost_")[1]: v for k, v in Params.cv_best_params.items() if "catboost" in k}
        else:
            best_params = {k.split("catboost_")[1]: v for k, v in Params.study.best_params.items() if "catboost" in k}
        params = Params.catboost_constant_params | best_params

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=[eval_pool], early_stopping_rounds=Params.early_stopping_round, use_best_model=True)
    return model


def _train(method, X_train, y_train, X_eval, y_eval, trial):
    if method == "LightGBM":
        model = lightgbm_training(X_train, y_train, X_eval, y_eval, trial)
    elif method == "CatBoost":
        model = catboost_training(X_train, y_train, X_eval, y_eval, trial)
    return model


def _predict(method, model, X):
    if method == "LightGBM":
        pred_proba = model.predict(X)
    elif method == "CatBoost":
        pred_proba = model.predict_proba(X)[:, 1]
    return pred_proba


def objective_with_args(methods, X_train, y_train, X_eval, y_eval, X_valid, X_tune, y_tune):
    def objective(trial):
        tune_pred_probas = np.zeros((len(methods), X_tune.shape[0]))
        valid_pred_probas = np.zeros((len(methods), X_valid.shape[0]))
        model_weights = []
        for j, method in enumerate(methods):
            # 学習
            model = _train(method, X_train, y_train, X_eval, y_eval, trial)
            # validに対する予測
            valid_pred_probas[j] = _predict(method, model, X_valid)
            # tuneに対する予測
            tune_pred_probas[j] = _predict(method, model, X_tune)
            # モデルのweight
            model_weight = trial.suggest_float(f"model_{j}_weight", 0, 1)  # TODO: ここだいぶわかりにくい書き方している
            model_weights.append(model_weight)
        # モデルの重み，クラス比率の最適化
        negative_ratio = trial.suggest_float("negative_ratio", 0, 1)
        y_pred_proba = np.average(tune_pred_probas, axis=0, weights=model_weights)
        y_pred = postprocess_prediction(y_pred_proba, negative_ratio)
        score = f1_score(y_tune, y_pred, average="macro")
        return score

    return objective


def cv_training(methods, X, y):
    best_weights = np.zeros((Params.n_splits, len(methods)))
    best_negative_ratios = []
    best_params = []
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

        # optuanaによる最適化
        Params.study = optuna.create_study(direction="maximize")
        Params.study.optimize(objective_with_args(methods, X_train, y_train, X_eval, y_eval, X_valid, X_tune, y_tune), n_trials=Params.n_trials)

        # validの予測
        best_weight = []
        valid_pred_probas = np.zeros((len(methods), X_valid.shape[0]))
        for j, method in enumerate(methods):
            # 学習
            model = _train(method, X_train, y_train, X_eval, y_eval, trial=None)
            best_weight.append(Params.study.best_params[f"model_{j}_weight"])
            valid_pred_probas[j] = _predict(method, model, X_valid)
        valid_pred_proba = np.average(valid_pred_probas, axis=0, weights=best_weight)
        valid_pred = postprocess_prediction(valid_pred_proba, Params.study.best_params["negative_ratio"])
        # 結果の格納
        scores.append(f1_score(y_valid, valid_pred, average="macro"))
        best_weights[i] = best_weight
        best_negative_ratios.append(Params.study.best_params["negative_ratio"])
        best_params.append({k: v for k, v in Params.study.best_params.items() if ((k != "negative_ratio") & ("weight" not in k))})
    return np.mean(scores), _dict_average(best_params), np.mean(best_weights, axis=0), np.mean(best_negative_ratios)


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
    cv_score, cv_best_params, cv_best_weight, cv_best_negative_ratio = cv_training(methods, X, y)
    Params.cv_best_params = cv_best_params
    print(f"best score: {cv_score}")
    print(f"best_params: {cv_best_params}")
    print(f"best weight: {cv_best_weight}")
    print(f"best negative ratio: {cv_best_negative_ratio}")
    result_df = pd.DataFrame({"experiment_name": [EXPERIMENT_NAME], "cv_score": [cv_score], "MEMO": [MEMO], "board_score": [None]})
    save_cv_result(result_df)
    # 各モデルの学習
    for method in methods:
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, random_state=Params.seed)
        model = _train(method, X_train, y_train, X_eval, y_eval, trial=None)
        models.append(model)
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
    start_time = time.time()
    main()
    print(f"{int((time.time() - start_time)/60)}分")
