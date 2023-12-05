import pandas as pd
import numpy as np
import re

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from sklearn.inspection import permutation_importance
import shap


def categorize_location(loc):
    """
    check if a location is in a predefined category.
    :param loc: Location to check.
    :return: int: 1 if location is in a category, 0 otherwise.
    """
    for category, loc_list in locations.items():
        if loc in loc_list:
            return 1
    return 0


def is_first_winner(first, winner):
    """
    Determine if the first participant is the winner.
    :param first: Name of the first participant.
    :param winner: Indicator of the winner (0 for bot, 1 for human).
    :return: int: 1 if first is the winner, 0 otherwise.
    """
    bot_names = ['BetterBot', 'STEEBot', 'HastyBot']
    if (first in bot_names and winner == 0) or (first not in bot_names and winner == 1):
        return 1
    return 0


def swap_elements(lst):
    """
    Swap numbers and letters in each location of the list.
    :param lst: List containing string elements with letters and numbers which represent the location on the board.
    :return: List where each element has its numbers and letters swapped.
    """
    swapped = []
    for element in lst:
        # Separate numbers and letters
        numbers = ''.join(filter(str.isdigit, element))
        letters = ''.join(filter(str.isalpha, element))

        # Swap only if the letter is first in the original string
        if element[0].isalpha():
            swapped_element = numbers + letters
        else:
            swapped_element = letters + numbers
        swapped.append(swapped_element)
    return swapped


def feature_engineering_turns(turns):
    """
    Perform feature engineering on the 'turns' DataFrame.
    :param turns: turns dataframe which holds the information of the turns in each game.
    :return: A new pandas DataFrame with additional features derived from the original data.
    """
    locations = {'letter_x3': ['6B', 'B10', 'F2', 'F6', 'F10', 'F14', 'J2', 'J6', 'J10', 'J14', 'N6', 'N10'],
                 'letter_x2': ['A4', 'A12', 'C7', 'C9', 'D1', 'D8', 'D15', 'G3', 'G7', 'G9', 'G13', 'H4', 'H12', 'I3',
                               'I7', 'I9', 'I13', 'L1', 'L8', 'L15', 'M7', 'M9', 'O4', 'O12'],
                 'word_x3': ['A1', 'H1', 'O1', '8A', 'O8', '15A', 'H15', 'O15'],
                 'word_x2': ['B2', 'C3', 'D4', 'E5', 'K5', 'L4', 'M3', 'N2', 'B14', 'C13', 'D12', 'E11', 'K11', 'L12',
                             'M13', 'N14']}

    hard_letters = ['K', 'Z', 'Q', 'X', 'J']
    medium_letters = ['F', 'H', 'V', 'W', 'Y', 'B', 'C', 'M', 'P']
    easy_letters = ['D', 'G', 'E', 'A', 'I', 'O', 'N', 'R', 'T', 'L', 'S', 'U']

    for key in locations.keys():
        locations[key] += swap_elements(locations[key])

    turns['move_len'] = turns['move'].str.len()
    turns['rack_len'] = turns['rack'].str.len()
    turns["points_per_letter"] = turns["points"] / turns["move_len"]
    turns['rack_usage'] = turns['move_len'] / turns['rack_len']
    turns["move"].fillna("None", inplace=True)
    turns['hard_letters'] = turns["move"].apply(lambda x: len([letter for letter in x if letter in hard_letters]))
    turns['medium_letters'] = turns["move"].apply(lambda x: len([letter for letter in x if letter in medium_letters]))
    turns['easy_letters'] = turns["move"].apply(lambda x: len([letter for letter in x if letter in easy_letters]))
    turns['loc_category'] = turns['location'].apply(categorize_location)

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
        min_rack_usage=('rack_usage', 'min'),
        num_special_loc=('loc_category', 'sum'),
        mean_hard_letters=('hard_letters', 'mean'),
        max_hard_letters=('hard_letters', 'max'),
        sum_hard_letters=('hard_letters', 'sum'),
        mean_medium_letters=('hard_letters', 'mean'),
        min_medium_letters=('medium_letters', 'min'),
        max_medium_letters=('medium_letters', 'max'),
        sum_medium_letters=('medium_letters', 'sum'),
        mean_easy_letters=('easy_letters', 'mean'),
        min_easy_letters=('easy_letters', 'min'),
        max_easy_letters=('easy_letters', 'max'),
        sum_easy_letters=('easy_letters', 'sum')).reset_index()
    return turns_groupby


def feature_engineering_games(games):
    """
    Perform feature engineering on the 'games' DataFrame.
    :param games: turns dataframe which holds the information of each game.
    :return: A new pandas DataFrame with additional features derived from the original data.
    """
    games["created_at"] = pd.to_datetime(games["created_at"])
    games["created_at_month"] = games["created_at"].dt.month
    games["created_at_day"] = games["created_at"].dt.day
    games["created_at_hour"] = games["created_at"].dt.hour
    games["created_at_day_of_week"] = games["created_at"].dt.dayofweek
    games['is_winner_first'] = games.apply(lambda x: is_winner_first(x['first'], x['winner']), axis=1)
    return games


def merge_datasets(train, test, games, turns):
    """
    Merges all the given datasets.
    :param train: train dataset which holds the information about the train set.
    :param test: test dataset which holds the information about the test set.
    :param games: game dataset which holds the information each game.
    :param turns: train dataset which holds the information the turns in each game.
    :return: merged dataset.
    """
    train_test_merged = pd.concat([train, test], axis=0).sort_values('game_id')
    full_df = pd.merge(pd.merge(train_test_merged, turns), games)
    full_df['score_per_move'] = full_df['score'] / full_df['num_moves']
    full_df['is_first'] = full_df.apply(lambda x: x['nickname'] == x['first'], axis=1)
    full_df = pd.get_dummies(full_df,
                             columns=['initial_time_seconds', 'time_control_name', 'game_end_reason', 'lexicon',
                                      'increment_seconds', 'rating_mode', 'max_overtime_minutes']).drop(
        ['nickname', 'created_at', 'first'], axis=1)

    # after feat importance from shap
    feat_to_remove = ['game_end_reason_CONSECUTIVE_ZEROES', 'initial_time_seconds_60', 'initial_time_seconds_30',
                      'min_easy_letters', 'increment_seconds_2', 'min_medium_letters', 'max_hard_letters',
                      'increment_seconds_15', 'max_rack_len', 'initial_time_seconds_1260', 'increment_seconds_30',
                      'sum_hard_letters']
    full_df = full_df.drop(feat_to_remove, axis=1)
    return full_df


def split_data(data):
    """
    Splits the given data into training and test sets based on the presence of ratings.
    :param data:
    :return: tuple: A tuple containing four elements in the following order:
             X (DataFrame): Features for the training set.
             y (Series): Target variable ('rating') for the training set.
             X_test (DataFrame): Features for the test set.
             test_ids (Series): Game IDs corresponding to the test set.
    """
    train_data = data[~data['rating'].isna()].drop('game_id', axis=1)
    test_data = data[data['rating'].isna()].drop('game_id', axis=1)

    test_ids = data[data['rating'].isna()]['game_id']

    X, y = train_data.drop('rating', axis=1), train_data['rating']
    X_test = test_data.drop('rating', axis=1)

    return X, y, X_test, test_ids


def preprocess_data(train, test, games, turns):
    """
    Performs preprocessing on the provided datasets for data analysis or modeling.
    :param train: train dataset which holds the information about the train set.
    :param test: test dataset which holds the information about the test set.
    :param games: game dataset which holds the information each game.
    :param turns: train dataset which holds the information the turns in each game.
    :return: merged dataset after feature engineering.
    """
    turns_fe = feature_engineering_turns(turns)
    games_fe = feature_engineering_games(games)
    full_df = merge_datasets(train, test, games_fe, turns_fe)
    return full_df


def evaluate_models_with_kfold_and_hyperparameter_tuning(X, y, X_test, n_splits=5):
    """
    Evaluates multiple models with K-Fold cross-validation and hyperparameter tuning.
    :param X: Features of the training set.
    :param y: Target variable of the training set.
    :param X_test: Features of the test set.
    :param n_splits: Number of splits for K-Fold cross-validation. Default is 5.
    :return: A tuple containing three elements in the following order:
           model_rmse (dict): Dictionary of RMSE values for each model.
           best_model_predictions (Series or array): Predictions made by the best model.
           best_model (model object): The best performing model after tuning.
    """
    models = {
        'LinearRegression': LinearRegression(),
        'LGBMRegressor': LGBMRegressor(verbose=-1),
        'CatBoostRegressor': CatBoostRegressor(verbose=False, iterations=100),
        'RandomForestRegressor': RandomForestRegressor()
    }

    # Hyperparameter grids for each model
    param_grids = {
        'LinearRegression': {'normalize': [True, False]},
        'LGBMRegressor': {'num_leaves': [15, 31, 50], 'learning_rate': [0.1, 0.05], 'n_estimators': [50, 100, 300]},
        'CatBoostRegressor': {'depth': [4, 6, 8, 10], 'learning_rate': [0.1, 0.05]},
        'RandomForestRegressor': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20]}
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_rmse = {name: 0 for name in models}
    model_predictions = {name: [] for name in models}

    for train_index, val_index in tqdm(kf.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        for name, model in models.items():
            grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            val_preds = best_model.predict(X_val)
            rmse = mean_squared_error(y_val, val_preds, squared=False)
            model_rmse[name] += rmse / n_splits

            test_preds = best_model.predict(X_test)
            model_predictions[name].append(test_preds)

    for name in models:
        model_predictions[name] = np.mean(model_predictions[name], axis=0)

    best_model_predictions = model_predictions[min(model_rmse, key=model_rmse.get)]
    return model_rmse, best_model_predictions, models[min(model_rmse, key=model_rmse.get)]


def make_submission(test_ids, model_predictions):
    """
    Creates a submission file from model predictions.
    :param test_ids: Game IDs from the test dataset.
    :param model_predictions: Predicted ratings from the model.
    """
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
    model_rmse, model_predictions, best_model = evaluate_models_with_kfold_and_hyperparameter_tuning(X, y, X_test,
                                                                                                     n_splits=5)
    print(model_rmse)
    make_submission(test_ids, model_predictions)


if __name__ == "__main__":
    main()
