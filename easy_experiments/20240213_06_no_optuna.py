"""LightGBM, CrossValidationを利用したベースラインモデル."""
import itertools
import os
import pickle
import random
import time
import warnings
from dataclasses import dataclass

import fsspec
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from catboost import CatBoostClassifier, Pool
from dateutil.relativedelta import relativedelta
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.simplefilter("ignore")

# 実行をどこで行うか
IS_EC2 = False
DO_OPTUNA = False
# 共通設定
if IS_EC2:
    BUCKET = "ryusuke-data-competition"
    DATA_PATH = os.path.join(f"s3://{BUCKET}", "data")
    RESULT_PATH = os.path.join(f"s3://{BUCKET}", "results")
    SUMMARY_FILENAME = "summary.csv"
else:
    DATA_PATH = "../data"
    RESULT_PATH = "../results"
    SUMMARY_FILENAME = "summary_easy.csv"
# 個別設定
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
IND_RESULT_PATH = os.path.join(RESULT_PATH, EXPERIMENT_NAME)
if not IS_EC2:
    os.makedirs(IND_RESULT_PATH, exist_ok=True)
MEMO = "20240213_05_no_optuna.pyをベース．State, BankStateごとの集約特徴量を作成．"


@dataclass
class Params:
    n_splits = 5
    n_trials = 100
    seed = 42
    methods = ["LightGBM", "CatBoost"]
    # 前処理関連
    drop_cols = []
    drop_cols_for_nn = ["City"]
    categorical_features = [
        "FranchiseCode",
        "RevLineCr",
        "LowDoc",
        "UrbanRural",
        "State",
        "BankState",
        "Sector",
        "City",
        "Franchise_or_not",
        "DisbursementDate_year",
        "ApprovalDate_year",
        "RepaymentDate_year",
        "DisbursementDate_month",
        "ApprovalDate_month",
        "RepaymentDate_month",
        "DisbursementDate_ym",
        "ApprovalDate_ym",
        "RepaymentDate_ym",
    ]
    encoding_target_cols = [
        "FranchiseCode",
        "RevLineCr",
        "LowDoc",
        "UrbanRural",
        "State",
        "BankState",
        "Sector",
        "City",
        "Franchise_or_not",
        "DisbursementDate_year",
        "ApprovalDate_year",
        "RepaymentDate_year",
        "DisbursementDate_month",
        "ApprovalDate_month",
        "RepaymentDate_month",
        "DisbursementDate_ym",
        "ApprovalDate_ym",
        "RepaymentDate_ym",
    ]
    # ハイパラ
    num_boost_round = 1000
    early_stopping_round = 200
    lgb_constant_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "n_estimators": num_boost_round,
        "verbosity": -1,
    }
    lgb_default_params = {
        "feature_fraction": 0.1677902074607869,
        "bagging_fraction": 0.11961649929211123,
        "num_leaves": 13,
        "lambda_l1": 0.9619150476292759,
        "lambda_l2": 0.5303505605413363,
    }
    catboost_constant_params = {
        "learning_rate": 0.05,
        "iterations": num_boost_round,
        "verbose": False,
    }
    catboost_default_params = {
        "depth": 2,
        "l2_leaf_reg": 0.7631639618350191,
        "subsample": 0.1368711550219926,
        "colsample_bylevel": 0.19528642238413346,
        "min_data_in_leaf": 13,
    }
    nn_constant_params = {}
    nn_default_params = {  # 適当
        "dropout_rate": 0.07707932119768192,  # Dropout率
        "optimizer": "adam",  # 最適化関数
        "activation": "relu",  # 活性化関数
        "n_layer": 2,  # レイヤー数
        "hidden_units": 34,  # 隠れ層のユニット数
        "epochs": 10,  # 学習済みモデルからの確認方法が不明
    }
    model_default_weights = [0.34, 0.66]
    model_default_negative_ratio = 0.078521
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

    def get_nn_params_range(self, trial):
        valiable_params = {
            "dropout_rate": trial.suggest_float("nn_dropout_rate", 0.0, 0.5),  # Dropout率
            "optimizer": trial.suggest_categorical("nn_optimizer", ["adam"]),  # 最適化関数
            "activation": trial.suggest_categorical("nn_activation", ["relu"]),  # 活性化関数
            "n_layer": trial.suggest_int("nn_n_layer", 1, 10),  # レイヤー数
            "hidden_units": trial.suggest_int("nn_hidden_units", 20, 300),  # 隠れ層のユニット数
            "epochs": trial.suggest_int("nn_epochs", 1, 20),
        }
        return self.nn_constant_params | valiable_params


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Tensorflow
    tf.random.set_seed(seed)


def _dict_average(dicts: list) -> dict:
    averaged_dict = {}
    if dicts[0]:
        for k, v in dicts[0].items():
            averaged_dict[k] = v
        for d in dicts[1:]:
            for k, v in d.items():
                if type(v) == int:
                    averaged_dict[k] = round((averaged_dict[k] + v) / len(dicts))
                elif type(v) == float:
                    averaged_dict[k] = (averaged_dict[k] + v) / len(dicts)
                elif type(v) == str:
                    averaged_dict[k] = v  # TODO: 改修必要

    return averaged_dict


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    dollar_amount_cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
    for col in dollar_amount_cols:
        df[col] = df[col].apply(lambda x: x.replace("$", "").replace(".", "").replace(",", "")).astype(int).copy()
    # 融資支払日と承認日の差を特徴量として追加
    ymd_cols = ["DisbursementDate", "ApprovalDate"]
    # Sectorをグルーピング
    df["Sector"] = (
        df["Sector"]
        .mask(df["Sector"].between(31, 33, inclusive="both"), 31)
        .mask(df["Sector"].between(44, 45, inclusive="both"), 44)
        .mask(df["Sector"].between(48, 49, inclusive="both"), 48)
    )
    # 年月関連を年と月で分ける
    for col in ymd_cols:
        df[col] = pd.to_datetime(df[col], format="%d-%b-%y")
        df[col] = df[col].fillna(max(df[col]) + relativedelta(years=1))
    return df


def add_aggregate_feature(df, groupby_col: str, agg_cols: str):
    for agg_col in agg_cols:
        average_df = df.groupby(groupby_col)[agg_col].mean().reset_index().rename(columns={agg_col: f"average_{groupby_col}_{agg_col}"})
        std_df = df.groupby(groupby_col)[agg_col].std().reset_index().rename(columns={agg_col: f"std_{groupby_col}_{agg_col}"})
        df = pd.merge(df, average_df, on=groupby_col, how="left")
        df = pd.merge(df, std_df, on=groupby_col, how="left")
        df[f"variance_{groupby_col}_{agg_col}"] = (df[agg_col] - df[f"average_{groupby_col}_{agg_col}"]) / df[f"std_{groupby_col}_{agg_col}"]
    return df


def add_feature(df: pd.DataFrame, interest_data: pd.DataFrame) -> pd.DataFrame:
    # 非フランチャイズかフランチャイズかの2値の特徴量を追加
    df["Franchise_or_not"] = (
        df["FranchiseCode"]
        .mask(((df["FranchiseCode"] == 0) | (df["FranchiseCode"] == 1)), 0)
        .mask(((df["FranchiseCode"] != 0) & (df["FranchiseCode"] != 1)), 1)
    )
    # 融資前後雇用者数
    df["NoEmp_before_after"] = df["NoEmp"] - df["RetainedJob"]
    df["NoEmp_before_after_ratio"] = df["RetainedJob"] / (df["NoEmp"] + 1)
    # 1ヶ月あたり融資額
    df["DisbursementGross_per_Month"] = df["DisbursementGross"] / (df["Term"] + 1)
    # 銀行が承認した額との差SBAが保証する額の差
    df["BankSBADiff"] = df["GrAppv"] - df["SBA_Appv"]
    df["BankSBADiff_ratio"] = df["GrAppv"] / (df["SBA_Appv"] + 1)
    df["AppvGrossDiff"] = df["GrAppv"] - df["DisbursementGross"]
    df["AppvGrossDiff_ratio"] = df["GrAppv"] / (df["DisbursementGross"] + 1)
    # 外部データを追加
    interest_data["DATE"] = pd.to_datetime(interest_data["DATE"])
    df["DATE"] = df["DisbursementDate"].apply(lambda x: x + relativedelta(day=1))
    df = pd.merge(df, interest_data, on="DATE", how="left").rename(columns={"FEDFUNDS": "Disbursement_Fedfunds"})
    df = df.drop(columns="DATE")
    df["DATE"] = df["ApprovalDate"].apply(lambda x: x + relativedelta(day=1))
    df = pd.merge(df, interest_data, on="DATE", how="left").rename(columns={"FEDFUNDS": "Approval_Fedfunds"})
    df = df.drop(columns="DATE")
    # 年月関連の特徴量を追加
    df["RepaymentDate"] = df.apply(lambda row: row["DisbursementDate"] + relativedelta(years=row["Term"] // 12, months=row["Term"] % 12), axis=1)
    ymd_cols = ["DisbursementDate", "ApprovalDate", "RepaymentDate"]
    for col in ymd_cols:
        df[col + "_year"] = pd.to_datetime(df[col]).apply(lambda x: x.year)
        df[col + "_month"] = pd.to_datetime(df[col]).apply(lambda x: x.month)
        df[col + "_ym"] = pd.to_datetime(df[col]).apply(lambda x: str(x.year) + str(x.month))
    df["date_diff"] = (pd.to_datetime(df["DisbursementDate"]) - pd.to_datetime(df["ApprovalDate"])).apply(lambda x: x.days)
    df = df.drop(columns=ymd_cols)
    # Sectorごとの集約特徴量の作成
    df[Params.categorical_features] = df[Params.categorical_features].fillna("Unknown").astype("category").copy()
    agg_cols = list(set(df.select_dtypes(["int", "float"]).columns) - set(["MIS_Status"]))
    groupby_cols = ["Sector", "State", "BankState"]
    for groupby_col in groupby_cols:
        df = add_aggregate_feature(df, groupby_col=groupby_col, agg_cols=agg_cols)
    # frequency encoding
    df[Params.categorical_features] = df[Params.categorical_features].fillna("Unknown").astype("category").copy()
    for col in Params.encoding_target_cols:
        count_dict = dict(df[col].value_counts())
        df[f"{col}_freq_encoding"] = df[col].map(count_dict).astype(int)
    print(df.shape)
    return df


def preprocess_data(df: pd.DataFrame, interest_data: pd.DataFrame) -> pd.DataFrame:
    """データの前処理."""
    # 不要なカラムを削除
    df = df.drop(columns=Params.drop_cols)
    # データを綺麗に
    df = clean_data(df)
    # 特徴量を追加
    df = add_feature(df, interest_data)
    return df


def _preprocess_unknown_city(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    # Stateと最頻Cityの対応辞書を作成
    df = train_data.groupby(["State", "City"]).size().reset_index().rename(columns={0: "count"})
    idx = df.groupby("State").idxmax().values.flatten()
    df = df.iloc[idx]
    rename_dict = dict(zip(df["State"].to_list(), df["City"].to_list()))
    # testデータにしかないCityは同じstateの最頻Cityで置換
    test_data["City"] = test_data["City"].where(test_data["City"].isin(set(train_data["City"])), test_data["State"].map(rename_dict))
    return test_data


def preprocess_city(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 表記揺れに対処
    train_data["City"] = train_data["City"].apply(lambda x: x.upper().split("(")[0].replace(" ", ""))
    test_data["City"] = test_data["City"].apply(lambda x: x.upper().split("(")[0].replace(" ", ""))
    # testデータにしかないCityは同じstateの最頻Cityで置換
    test_data = _preprocess_unknown_city(train_data, test_data)
    return train_data, test_data


def preprocess_data_for_nn(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train_for_nn = df_train.drop(columns=["MIS_Status"] + Params.drop_cols_for_nn).copy()
    df_test_for_nn = df_test.drop(columns=Params.drop_cols_for_nn).copy()

    # 数値型を標準化
    numerical_cols = df_train_for_nn.select_dtypes(["int", "float"]).columns
    ss = StandardScaler()
    ss.fit(df_train_for_nn[numerical_cols])
    train_numerical_data = ss.transform(df_train_for_nn[numerical_cols])
    df_train_numerical_for_nn = pd.DataFrame(
        data=train_numerical_data, columns=ss.get_feature_names_out() + "_nn", index=df_train_for_nn.index
    ).fillna(0)
    test_numerical_data = ss.transform(df_test_for_nn[numerical_cols])
    df_test_numerical_for_nn = pd.DataFrame(data=test_numerical_data, columns=ss.get_feature_names_out() + "_nn", index=df_test_for_nn.index).fillna(
        0
    )

    # カテゴリ変数をone hot encoding
    category_cols = df_train_for_nn.select_dtypes("category").columns
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    ohe.fit(df_train_for_nn[category_cols].astype("str"))
    train_categorical_data = ohe.transform(df_train_for_nn[category_cols].astype("str"))
    df_train_categorical_for_nn = pd.DataFrame(
        data=train_categorical_data, columns=ohe.get_feature_names_out() + "_nn", index=df_train_for_nn.index
    ).fillna(0)
    test_categorical_data = ohe.transform(df_test_for_nn[category_cols].astype("str"))
    df_test_categorical_for_nn = pd.DataFrame(
        data=test_categorical_data, columns=ohe.get_feature_names_out() + "_nn", index=df_test_for_nn.index
    ).fillna(0)

    # 結合
    df_train = pd.concat([df_train, df_train_numerical_for_nn, df_train_categorical_for_nn], axis=1)
    df_test = pd.concat([df_test, df_test_numerical_for_nn, df_test_categorical_for_nn], axis=1)
    return df_train, df_test


def get_params(method, trial):
    if method == "lightgbm":
        if DO_OPTUNA:
            if trial:  # optuna用
                params = Params().get_lightgbm_params_range(trial)
            else:
                if Params.cv_best_params:  # テストデータ予測用モデルの学習の場合
                    best_params = {k.split("lgb_")[1]: v for k, v in Params.cv_best_params.items() if "lgb" in k}
                else:  # CV用
                    best_params = {k.split("lgb_")[1]: v for k, v in Params.study.best_params.items() if "lgb" in k}
                params = Params.lgb_constant_params | best_params
        else:
            params = Params.lgb_constant_params | Params.lgb_default_params
    elif method == "catboost":
        if DO_OPTUNA:
            if trial:
                params = Params().get_catboost_params_range(trial)
            else:
                if Params.cv_best_params:
                    best_params = {k.split("catboost_")[1]: v for k, v in Params.cv_best_params.items() if "catboost" in k}
                else:
                    best_params = {k.split("catboost_")[1]: v for k, v in Params.study.best_params.items() if "catboost" in k}
                params = Params.catboost_constant_params | best_params
        else:
            params = Params.catboost_constant_params | Params.catboost_default_params
    elif method == "nn":
        if DO_OPTUNA:
            if trial:
                params = Params().get_nn_params_range(trial)
            else:
                if Params.cv_best_params:
                    best_params = {k.split("nn_")[1]: v for k, v in Params.cv_best_params.items() if "nn" in k}
                else:
                    best_params = {k.split("nn_")[1]: v for k, v in Params.study.best_params.items() if "nn" in k}
                params = Params.nn_constant_params | best_params
        else:
            params = Params.nn_constant_params | Params.nn_default_params
    return params


def lightgbm_training(X_train, y_train, X_eval, y_eval, trial):
    # lightgbm用データセットに変換
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=Params.categorical_features)
    eval_dataset = lgb.Dataset(X_eval, label=y_eval, categorical_feature=Params.categorical_features)

    params = get_params("lightgbm", trial)

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

    params = get_params("catboost", trial)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=[eval_pool], early_stopping_rounds=Params.early_stopping_round, use_best_model=True)
    return model


def nn_training(X_train, y_train, X_eval, y_eval, trial):
    params = get_params("nn", trial)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)))  # 入力層
    for i in range(params["n_layer"]):  # 隠れ層
        model.add(tf.keras.layers.Dense(params["hidden_units"], activation=params["activation"]))
        model.add(tf.keras.layers.Dropout(params["dropout_rate"]))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))  # 出力層

    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(X_train, y_train, epochs=params["epochs"])
    return model


def _extract_nn_cols(df):
    return [col for col in df.columns if "nn" in col]


def _extract_not_nn_cols(df):
    return [col for col in df.columns if "nn" not in col]


def train(method, X_train, y_train, X_eval, y_eval, trial):
    if method == "LightGBM":
        use_cols = _extract_not_nn_cols(X_train)
        model = lightgbm_training(X_train[use_cols], y_train, X_eval[use_cols], y_eval, trial)
    elif method == "CatBoost":
        use_cols = _extract_not_nn_cols(X_train)
        model = catboost_training(X_train[use_cols], y_train, X_eval[use_cols], y_eval, trial)
    elif method == "NN":
        use_cols = _extract_nn_cols(X_train)
        model = nn_training(X_train[use_cols], y_train, X_eval[use_cols], y_eval, trial)
    return model


def predict(method, model, X):
    if method == "LightGBM":
        use_cols = _extract_not_nn_cols(X)
        pred_proba = model.predict(X[use_cols])
    elif method == "CatBoost":
        use_cols = _extract_not_nn_cols(X)
        pred_proba = model.predict_proba(X[use_cols])[:, 1]
    elif method == "NN":
        use_cols = _extract_nn_cols(X)
        pred_proba = model.predict(X[use_cols]).reshape(-1)
    return pred_proba


def objective_with_args(methods, X_train, y_train, X_eval, y_eval, X_valid, X_tune, y_tune):
    def objective(trial):
        tune_pred_probas = np.zeros((len(methods), X_tune.shape[0]))
        valid_pred_probas = np.zeros((len(methods), X_valid.shape[0]))
        model_weights = []
        for j, method in enumerate(methods):
            # 学習
            model = train(method, X_train, y_train, X_eval, y_eval, trial)
            # validに対する予測
            valid_pred_probas[j] = predict(method, model, X_valid)
            # tuneに対する予測
            tune_pred_probas[j] = predict(method, model, X_tune)
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
    best_weights_arr = np.zeros((Params.n_splits, len(methods)))
    best_negative_ratios = []
    best_params_arr = []
    scores = []
    kf = KFold(n_splits=Params.n_splits, shuffle=True)
    for i, (train_eval_tune_index, valid_index) in enumerate(kf.split(X)):
        # データの分割
        X_train_eval_tune = X.iloc[train_eval_tune_index].copy()
        y_train_eval_tune = y.iloc[train_eval_tune_index].copy()
        X_train_eval, X_tune, y_train_eval, y_tune = train_test_split(X_train_eval_tune, y_train_eval_tune, test_size=0.25)
        X_train, X_eval, y_train, y_eval = train_test_split(X_train_eval, y_train_eval, test_size=0.33)
        X_valid = X.iloc[valid_index].copy()
        y_valid = y.iloc[valid_index].copy()

        # optuanaによる最適化
        if DO_OPTUNA:
            Params.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=Params.seed + i))
            Params.study.optimize(objective_with_args(methods, X_train, y_train, X_eval, y_eval, X_valid, X_tune, y_tune), n_trials=Params.n_trials)
        else:
            Params.study = None

        # validの予測
        best_weights = []
        valid_pred_probas = np.zeros((len(methods), X_valid.shape[0]))
        for j, method in enumerate(methods):
            model = train(method, X_train, y_train, X_eval, y_eval, trial=None)
            valid_pred_probas[j] = predict(method, model, X_valid)
            best_weight = Params.study.best_params[f"model_{j}_weight"] if Params.study else Params.model_default_weights[j]
            best_weights.append(best_weight)
        valid_pred_proba = np.average(valid_pred_probas, axis=0, weights=best_weights)
        best_negative_ratio = Params.study.best_params["negative_ratio"] if Params.study else Params.model_default_negative_ratio
        valid_pred = postprocess_prediction(valid_pred_proba, best_negative_ratio)
        # 結果の格納
        scores.append(f1_score(y_valid, valid_pred, average="macro"))
        best_weights_arr[i] = best_weights
        best_negative_ratios.append(best_negative_ratio)
        best_params = {k: v for k, v in Params.study.best_params.items() if ((k != "negative_ratio") & ("weight" not in k))} if Params.study else None
        best_params_arr.append(best_params)
    return scores, best_params_arr, best_weights_arr, best_negative_ratios


def postprocess_prediction(y_pred_proba, negative_ratio):
    threshold = np.sort(y_pred_proba)[int(y_pred_proba.shape[0] * negative_ratio)]
    y_pred = np.where(y_pred_proba < threshold, 0, 1)
    return y_pred


def save_cv_result(cv_scores):
    result_df = pd.DataFrame({"experiment_name": [EXPERIMENT_NAME], "cv_score": [np.mean(cv_scores)], "MEMO": [MEMO], "board_score": [None]})

    summary_path = os.path.join(RESULT_PATH, SUMMARY_FILENAME)
    try:
        df = pd.read_csv(summary_path)
    except:
        df = pd.DataFrame({"experiment_name": [], "cv_score": [], "MEMO": [], "board_score": []})
    df = pd.concat([df, result_df]).drop_duplicates()
    df.to_csv(summary_path, index=False)


def save_cv_detail(methods, cv_scores, cv_best_weights: np.ndarray, cv_best_negative_ratios: np.ndarray):
    df = pd.DataFrame(data=cv_best_weights, columns=methods)
    df = pd.concat([df, pd.DataFrame(data=cv_best_negative_ratios, columns=["negative_ratio"])], axis=1)
    df = pd.concat([df, pd.DataFrame(data=cv_scores, columns=["cv_score"])], axis=1)
    df.to_csv(os.path.join(IND_RESULT_PATH, "cv_details.csv"), index=False)


def save_model_as_pickle(model, filename):
    with fsspec.open(os.path.join(IND_RESULT_PATH, filename), "wb") as f:
        pickle.dump(model, f)


def Preprocessing():
    # データ読み込み
    train_data = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), index_col=0)
    test_data = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), index_col=0)
    interest_data = pd.read_csv(os.path.join(DATA_PATH, "federal_funds_effective_rate.csv"))

    # Cityの前処理
    train_data, test_data = preprocess_city(train_data, test_data)
    # 前処理
    train_data = preprocess_data(train_data, interest_data)
    test_data = preprocess_data(test_data, interest_data)
    # NN用前処理
    train_data, test_data = preprocess_data_for_nn(train_data, test_data)

    # 説明変数と目的変数に分ける
    X = train_data.drop(columns="MIS_Status").copy()
    y = train_data["MIS_Status"].copy()
    X_test = test_data.copy()
    return X, y, X_test


def Learning(methods, X, y):
    models = []
    # CV
    cv_scores, cv_best_params, cv_best_weights, cv_best_negative_ratios = cv_training(methods, X, y)
    save_cv_result(cv_scores)
    save_cv_detail(methods, cv_scores, cv_best_weights, cv_best_negative_ratios)
    Params.cv_best_params = _dict_average(cv_best_params)
    # 各モデルの学習
    for method in methods:
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25)
        model = train(method, X_train, y_train, X_eval, y_eval, trial=None)
        models.append(model)
        save_model_as_pickle(model, filename=f"{method}_model.pickle")
    return models, np.mean(cv_best_weights, axis=0), np.mean(cv_best_negative_ratios)


def Predicting(X_test, methods, models, best_weight, best_negative_ratio):
    test_pred_probas = np.zeros((len(models), X_test.shape[0]))
    for i, (method, model) in enumerate(zip(methods, models)):
        test_pred_probas[i] = predict(method, model, X_test)
    # 予測
    y_pred_proba = np.average(test_pred_probas, axis=0, weights=best_weight)
    # 後処理
    y_pred = postprocess_prediction(y_pred_proba, best_negative_ratio)

    # 結果の保存
    sample_submit = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"), index_col=0, header=None)  # 応募用サンプルファイル
    sample_submit[1] = y_pred
    sample_submit.to_csv(os.path.join(RESULT_PATH, f"{EXPERIMENT_NAME}.csv"), header=None)


def main():
    fix_seed(Params.seed)
    X, y, X_test = Preprocessing()

    models, best_weight, best_negative_ratio = Learning(Params.methods, X, y)

    Predicting(X_test, Params.methods, models, best_weight, best_negative_ratio)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"{int((time.time() - start_time)/60)}分")
