import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.count_last_results import count_last_results


def data_preprocessing(csv_path, down_season, up_season):
    """
    Read csv_file, and filter data, remove unnecessary columns and adding new ones.

    Args:
    :param csv_path: Path to file
    :param season:
    :return:
    """

    # Reading data as pandas dataframe and translation every column to english
    matches_df = pd.read_csv(csv_path, sep=';')
    matches_df = matches_df.rename(columns={'Kolejka': 'matchday'})

    # Deleting walkover and cancelled matches
    walkover_mask = matches_df['IsWalkover'] == 0
    cancelled_mask = matches_df['IsCancelled'] == 0
    wrong_matches_mask = walkover_mask & cancelled_mask
    matches_df = matches_df[wrong_matches_mask]

    # Change season column type to int and date to datetime
    matches_df['season'] = matches_df['season'].str.split('/').str[0]
    matches_df['season'] = pd.to_numeric(matches_df['season'])
    matches_df['date'] = pd.to_datetime(matches_df['date'])

    # Filtering season date
    season_mask = matches_df['season'] >= down_season
    matches_df = matches_df[season_mask]

    # Deleting teams that play only one season in Ekstraklasa
    season_count = matches_df.groupby('home')['season'].nunique()
    one_season_mask = season_count == 1
    season_count = season_count[one_season_mask]
    delete_one_season_mask = ~matches_df['home'].isin(season_count)
    matches_df = matches_df[delete_one_season_mask]

    # Sort by date and reset index after deleting teams
    matches_df = matches_df.sort_values(by='date').reset_index(drop=True)

    # Unique teams encoding with LabelEncoder it will be transformed to Embedding then
    team_encoder = LabelEncoder()

    all_teams = pd.concat([matches_df['home'], matches_df['away']]).unique()
    unique_teams = team_encoder.fit(all_teams)

    unique_teams_names = team_encoder.classes_

    unique_teams_map = {}
    for number in range(len(unique_teams_names)):
        unique_teams_map[number] = unique_teams_names[number]

    matches_df['home'] = unique_teams.transform(matches_df['home'])
    matches_df['away'] = unique_teams.transform(matches_df['away'])

    # Creating new column that shows goals balance
    matches_df['goals'] = matches_df['gh'] - matches_df['ga']

    # Creating new column that show the result, 2 = win, 1 = draw, 0 = loss
    win_mask = matches_df['goals'] > 0
    draw_mask = matches_df['goals'] == 0
    conditions = [win_mask, draw_mask]
    choices = [2, 1]
    matches_df['result'] = np.select(conditions, choices, default=0)

    train_mask = matches_df['season'] < up_season
    test_mask = matches_df['season'] >= up_season

    matches_df_train = matches_df[train_mask]
    matches_df_test = matches_df[test_mask]

    last_results_train = count_last_results(matches_df_train)
    last_results_test = count_last_results(matches_df_test)

    matches_df_train = matches_df_train.sort_values(by="date").reset_index(drop=True)
    matches_df_train = pd.concat([matches_df_train, last_results_train], axis=1)

    matches_df_test = matches_df_test.sort_values(by="date").reset_index(drop=True)
    matches_df_test = pd.concat([matches_df_test, last_results_test], axis=1)

    # Delete unnecessary columns
    matches_df_train = matches_df_train.drop(columns=['Id', 'season', 'IsCancelled', 'IsWalkover', 'hour', 'note', 'gh', 'ga', 'date', 'goals'])
    matches_df_test = matches_df_test.drop(columns=['Id', 'season', 'IsCancelled', 'IsWalkover', 'hour', 'note', 'gh', 'ga', 'date', 'goals'])

    # Swapping order of columns
    swap_columns_titles = ['matchday', 'home', 'away', 'last_results_home', 'last_results_away', 'result', 'goals']
    matches_df_train = matches_df_train.reindex(columns=swap_columns_titles)
    matches_df_test = matches_df_test.reindex(columns=swap_columns_titles)

    return matches_df_train, matches_df_test, unique_teams_map
