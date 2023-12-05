import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


def feature_engineering_turns(turns):
    turns['move_len'] = turns['move'].str.len()
    turns['rack_len'] = turns['rack'].str.len()
    turns['rack_usage'] = turns['move_len'] / turns['rack_len']
    turns_groupby = turns.groupby(['game_id', 'nickname']).agg(
        mean_points=('points', 'mean'),
        max_points=('points', 'max'),
        min_points=('points', 'min'),
        num_moves=('move', 'count'),
        mean_move_len=('move_len', 'mean'),
        max_move_len=('move_len', 'max'),
        min_move_len=('move_len', 'min'),
        mean_rack_len=('rack_len', 'mean'),
        max_rack_len=('rack_len', 'max'),
        min_rack_len=('rack_len', 'min'),
        mean_rack_usage=('rack_usage', 'mean'),
        max_rack_usage=('rack_usage', 'max'),
        min_rack_usage=('rack_usage', 'min')).reset_index()
    return turns_groupby


def feature_engineering_games(games):
    games["created_at"] = pd.to_datetime(games["created_at"])
    games["created_at_month"] = games["created_at"].dt.month
    games["created_at_day"] = games["created_at"].dt.day
    games["created_at_hour"] = games["created_at"].dt.hour
    games["created_at_day_of_week"] = games["created_at"].dt.dayofweek
    games['is_first_winner'] = games.apply(lambda x: is_first_winner(x['first'], x['winner']), axis=1)
    return games


def is_first_winner(first, winner):
    bot_names = ['BetterBot', 'STEEBot', 'HastyBot']
    if (first in bot_names and winner == 0) or (first not in bot_names and winner == 1):
        return 1
    return 0


def merge_datasets(train, test, games, turns):
    train_test_merged = pd.concat([train, test], axis=0).sort_values('game_id')
    full_df = pd.merge(pd.merge(train_test_merged, turns), games)
    full_df['score_per_move'] = full_df['score'] / full_df['num_moves']
    full_df = pd.get_dummies(full_df,
                             columns=['initial_time_seconds', 'time_control_name', 'game_end_reason', 'lexicon',
                                      'increment_seconds', 'rating_mode', 'max_overtime_minutes']).drop(
        ['nickname', 'created_at', 'first'], axis=1)

    return full_df


def split_data(data):
    train_data = data[~data['rating'].isna()].drop('game_id', axis=1)
    test_data = data[data['rating'].isna()].drop('game_id', axis=1)

    test_ids = data[data['rating'].isna()]['game_id']

    X, y = train_data.drop('rating', axis=1), train_data['rating']
    X_test = test_data.drop('rating', axis=1)

    return X, y, X_test, test_ids


def preprocess_data(train, test, games, turns):
    turns_fe = feature_engineering_turns(turns)
    games_fe = feature_engineering_games(games)
    full_df = merge_datasets(train, test, games_fe, turns_fe)
    return full_df


def evaluate_models_with_kfold(X, y, X_test, n_splits=5):
    models = {
        'LinearRegression': LinearRegression(),
        'LGBMRegressor': LGBMRegressor(verbose=-1, random_state=42),
        'CatBoostRegressor': CatBoostRegressor(verbose=0, random_state=42, iterations=100),
        'RandomForestRegressor': RandomForestRegressor(random_state=42)
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_rmse = {name: 0 for name in models}
    model_predictions = {name: [] for name in models}

    for train_index, val_index in tqdm(kf.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        for name, model in models.items():
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            rmse = mean_squared_error(y_val, val_preds, squared=False)
            model_rmse[name] += rmse / n_splits
            test_preds = model.predict(X_test)
            model_predictions[name].append(test_preds)

    for name in models:
        model_predictions[name] = np.mean(model_predictions[name], axis=0)

    best_model_predictions = model_predictions[min(model_rmse, key=model_rmse.get)]
    return model_rmse, best_model_predictions


def make_submission(test_ids, model_predictions):
    submission = pd.DataFrame()
    submission["game_id"] = test_ids
    submission["rating"] = model_predictions


    submission.to_csv("PlayerRatingSubmission.csv", index=False)


def main():
    games = pd.read_csv("data/games.csv")
    turns = pd.read_csv("data/turns.csv")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    full_df = preprocess_data(train, test, games, turns)
    X, y, X_test, test_ids = split_data(full_df)
    model_rmse, model_predictions = evaluate_models_with_kfold(X, y, X_test, n_splits=5)
    print(model_rmse)
    make_submission(test_ids, model_predictions)


if __name__ == "__main__":
    main()
