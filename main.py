import pandas as pd
import numpy as np
import re

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from tqdm import tqdm
import joblib
import argparse
from typing import List, Dict, Tuple, Union
from pandas.core.groupby.generic import DataFrameGroupBy


def categorize_location(locations: Dict[str, List[str]], loc: str) -> int:
    """
    check if a location is in a predefined category.
    :param locations: a dictionary with locations and their category.
    :param loc: Location to check.
    :return: int: 1 if location is in a category, 0 otherwise.
    """
    for category, loc_list in locations.items():
        if loc in loc_list:
            return 1
    return 0


def is_first_winner(first: str, winner: int) -> int:
    """
    Determine if the first participant is the winner.
    :param first: Name of the first participant.
    :param winner: Indicator of the winner
    :return: int: 1 if first is the winner, 0 otherwise.
    """
    bot_names = ['BetterBot', 'STEEBot', 'HastyBot']
    if (first in bot_names and winner == 0) or (first not in bot_names and winner == 1):
        return 1
    return 0


def swap_elements(lst: List[str]) -> List[str]:
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

        if element[0].isalpha():
            swapped_element = numbers + letters
        else:
            swapped_element = letters + numbers
        swapped.append(swapped_element)
    return swapped


def bot_encoding(gr: DataFrameGroupBy) -> pd.Series:
    """
    Encoding the bot name in each group
    :param gr: Grouped by object.
    :return: Encoded bot names.
    """
    nicknames = gr.nickname
    is_bot = nicknames.isin(['BetterBot', 'STEEBot', 'HastyBot'])
    bot_name = nicknames[is_bot]
    bot_name = bot_name.values[0]
    nicknames = nicknames.replace(nicknames[~is_bot].values[0], bot_name)
    return nicknames


def feature_engineering_turns(turns: pd.DataFrame) -> pd.DataFrame:
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
    turns['loc_category'] = turns['location'].apply(lambda x: categorize_location(locations, x))

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
        mean_special_loc=('loc_category', 'mean'),
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


def merge_datasets(train: pd.DataFrame, test: pd.DataFrame, games: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
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

    full_df["created_at"] = pd.to_datetime(full_df["created_at"])
    full_df['score_per_move'] = full_df['score'] / full_df['num_moves']

    full_df["created_at_month"] = full_df["created_at"].dt.month
    full_df["created_at_day"] = full_df["created_at"].dt.day
    full_df["created_at_hour"] = full_df["created_at"].dt.hour
    full_df["created_at_day_of_week"] = full_df["created_at"].dt.dayofweek
    full_df['is_first_winner'] = full_df.apply(lambda x: is_first_winner(x['first'], x['winner']), axis=1)
    full_df['is_first'] = full_df.apply(lambda x: x['nickname'] == x['first'], axis=1)

    bot_difficulty = full_df.groupby('game_id').apply(bot_encoding).values
    bot_difficulty = pd.Series(bot_difficulty).replace({'BetterBot': 1, 'STEEBot': 2, 'HastyBot': 3})
    full_df['bot_difficulty'] = bot_difficulty

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


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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


def preprocess_data(train: pd.DataFrame, test: pd.DataFrame, games: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """
    Performs preprocessing on the provided datasets for data analysis or modeling.
    :param train: train dataset which holds the information about the train set.
    :param test: test dataset which holds the information about the test set.
    :param games: game dataset which holds the information each game.
    :param turns: train dataset which holds the information the turns in each game.
    :return: merged dataset after feature engineering.
    """
    turns_fe = feature_engineering_turns(turns)
    full_df = merge_datasets(train, test, games, turns_fe)
    return full_df


def evaluate_models_with_kfold_and_hyperparameter_tuning(
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        print_rmse: bool = False) -> Tuple[
        Union[pd.Series, np.array], Dict[str, Union[LGBMRegressor, CatBoostRegressor]]]:
    """
    Evaluates multiple models with K-Fold cross-validation and hyperparameter tuning.
    :param X: Features of the training set.
    :param y: Target variable of the training set.
    :param X_test: Features of the test set.
    :param print_rmse: Boolean value which indicates if to print the rmse values.
    :return: A tuple containing three elements in the following order:
           best_model_predictions (Series or array): Predictions made by the best model.
           best_models (dict): Dictionary of model name and best performing model.
    """
    models = {
        'LGBMRegressor': LGBMRegressor(verbose=-1, objective='rmse'),
        'CatBoostRegressor': CatBoostRegressor(verbose=False),
    }

    param_grids = {
        'LGBMRegressor': {'num_leaves': [15, 31, 50], 'learning_rate': [0.1, 0.05, 0.01],
                          'n_estimators': [50, 100, 300, 500, 1000], 'max_depth': [3, 4]},
        'CatBoostRegressor': {'depth': [4, 6, 8, 10], 'learning_rate': [0.1, 0.05, 0.01], 'iterations': [50, 100, 200]},
    }

    model_rmse_train = {name: 0 for name in models}
    model_rmse_val = {name: 0 for name in models}
    models_predictions = {}
    best_models = {}
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=20)

    for name, model in tqdm(models.items()):
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_

        train_preds = best_models[name].predict(X_train)
        rmse_train = mean_squared_error(y_train, train_preds, squared=False)
        model_rmse_train[name] += rmse_train

        val_preds = best_models[name].predict(X_val)
        rmse_val = mean_squared_error(y_val, val_preds, squared=False)
        model_rmse_val[name] += rmse_val

        test_preds = best_models[name].predict(X_test)
        models_predictions[name] = test_preds

    best_model_predictions = models_predictions['CatBoostRegressor']
    if print_rmse:
        print(f'Train rmse {model_rmse_train}')
        print(f'Val rmse {model_rmse_val}')

    return best_model_predictions, best_models


def make_submission(path: str, test_ids: pd.Series, model_predictions: Union[pd.Series, np.array]) -> None:
    """
    Creates a submission file from model predictions.
    :param path: The file path to the save the predictions.
    :param test_ids: Game IDs from the test dataset.
    :param model_predictions: Predicted ratings from the model.
    """
    submission = pd.DataFrame()
    submission["game_id"] = test_ids
    submission["rating"] = model_predictions
    submission.to_csv(path, index=False)


def load_model_catboost(path: str) -> CatBoostRegressor:
    """
    Load a CatBoost model from the specified file path.
    :param path: The file path to the saved CatBoost model.
    :return: CatBoostRegressor: An instance of the loaded CatBoostRegressor model.
    """
    catboost = CatBoostRegressor()
    catboost.load_model(path)
    return catboost


def predict(catboost: catboost, X_test: pd.DataFrame) -> pd.Series:
    """
    Make predictions by combining the outputs of a LightGBM and a CatBoost model.
    :param catboost: The CatBoost model.
    :param X_test: The test data to make predictions on.
    :return: The combined predictions from the LightGBM and CatBoost models.
    """
    catboost_preds = catboost.predict(X_test)
    preds = catboost_preds
    return preds


def main(path_predictions: str, path_catboost: str, train_model: bool) -> None:
    print('---------- Loading Data ----------')

    games = pd.read_csv("data/games.csv")
    turns = pd.read_csv("data/turns.csv")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    print('---------- Preprocessing Data ----------')

    full_df = preprocess_data(train, test, games, turns)
    X, y, X_test, test_ids = split_data(full_df)

    if train_model:
        print('---------- Finding Model and Tuning Hyperparameters ----------')
        model_predictions, best_model = evaluate_models_with_kfold_and_hyperparameter_tuning(
            X, y, X_test)
    else:
        print('---------- Using Preload Model and Predicting ----------')
        catboost_model = load_model_catboost(path_catboost)
        model_predictions = predict(catboost_model, X_test)

    print('---------- Done ----------')
    make_submission(path_predictions, test_ids, model_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the main function with options for training and model paths.')

    parser.add_argument('--path_predictions', type=str, default='predictions/predictions.csv',
                        help='Path to save the prediction results.')
    parser.add_argument('--path_catboost', type=str, default='models/catboost_model.cbm',
                        help='Path to the pre-trained CatBoost model.')
    parser.add_argument('--path_lgbm', type=str, default='models/lgbm_model.joblib',
                        help='Path to the pre-trained LightGBM model.')
    parser.add_argument('--train_model', action='store_true',
                        help='If True, train the model. Otherwise, use pre-trained models for prediction.')

    args = parser.parse_args()
    main(args.path_predictions, args.path_catboost, args.train_model)
