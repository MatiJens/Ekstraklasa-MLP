import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data_from_csv(csv_path, club_name, season):

    # Reading data as pandas dataframe and translation every column to english
    matches_df = pd.read_csv(csv_path, sep = ';')
    matches_df = matches_df.rename(columns={'Kolejka': 'matchday'})

    # Deleting walkover and cancelled matches
    walkover_mask = matches_df['IsWalkover'] == 0
    cancelled_mask = matches_df['IsCancelled'] == 0
    wrong_matches_mask = walkover_mask & cancelled_mask
    matches_df = matches_df[wrong_matches_mask]

    # Change season column type to int
    matches_df['season'] = matches_df['season'].str.split('/').str[0]
    matches_df['season'] = pd.to_numeric(matches_df['season'])

    # Filtering season date
    season_mask = matches_df['season'] >= season
    matches_df = matches_df[season_mask]

    # Filtering club name and adding new column is_home, that tells if match was in home stadium
    club_home_mask = matches_df['home'] == club_name
    club_away_mask = matches_df['away'] == club_name
    club_mask = club_home_mask | club_away_mask
    matches_df = matches_df[club_mask]
    matches_df['is_home'] = club_home_mask.astype(int)

    # Adding goals result score
    goals = np.where(
        matches_df['is_home'] == 1,
        matches_df['gh'] - matches_df['ga'],
        matches_df['ga'] - matches_df['gh']
    )

    matches_df['goals'] = goals

    # Creating result masks
    win_home_mask = (matches_df['is_home'] == 1) & (matches_df['gh'] > matches_df['ga'])
    win_away_mask = (matches_df['is_home'] == 0) & (matches_df['ga'] > matches_df['gh'])
    win_mask = win_home_mask | win_away_mask
    draw_mask = matches_df['ga'] == matches_df['gh']

    # Creating new column that show the result, 2 = win, 1 = draw, 0 = loss
    conditions = [win_mask, draw_mask]
    choices = [2, 1]
    matches_df['result'] = np.select(conditions, choices, default=0)

    # Counting results of last 5 matches
    last_results_map = matches_df['result'].map({2 : 1, 1 : 0, 0 : -1})
    last_results = last_results_map.rolling(window = 5, min_periods = 1).sum().shift(1).fillna(0)
    matches_df.insert(matches_df.columns.get_loc('result'), 'last_results', last_results.astype(int))
    #matches_df['last_results'] =

    # Adding opponent as Label Encoding column it will be transformed to Embedding
    matches_df = matches_df.reset_index(drop=True)

    opponent = np.where(
        matches_df['is_home'] == 1,
        matches_df['away'],
        matches_df['home']
    )

    opponent = np.char.replace(opponent.astype(str), ' ', '')

    opponent_encoder = LabelEncoder()

    matches_df.insert(matches_df.columns.get_loc('result'), 'opponent_id', opponent_encoder.fit_transform(opponent))

    # Deleting teams that play only one season in Ekstraklasa
    season_count = matches_df.groupby('opponent_id')['season'].nunique()
    one_season_mask = season_count == 1
    season_count = season_count[one_season_mask]
    delete_one_season_mask = ~matches_df['opponent_id'].isin(season_count)
    matches_df = matches_df[delete_one_season_mask]

    matches_df = matches_df.drop(columns=['Id', 'IsCancelled', 'IsWalkover', 'hour', 'date', 'note', 'home', 'away', 'gh', 'ga', 'season'])

    # Final reset of indexes
    matches_df = matches_df.reset_index(drop=True)

    return matches_df


