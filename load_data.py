import numpy as np
import pandas as pd

def load_data_from_csv(csv_path, club_name, season):

    # Reading data as pandas dataframe and deleting unnecessary columns
    matches_df = pd.read_csv(csv_path, sep = ';')
    matches_df = matches_df.drop(columns =['Id', 'Kolejka', 'IsCancelled', 'IsWalkover', 'hour', 'date', 'note'])

    # Change season column type to int
    matches_df['season'] = matches_df['season'].str.split('/').str[0]
    matches_df['season'] = pd.to_numeric(matches_df['season'])

    # Deleting teams that play only one season in Ekstraklasa
    season_count = matches_df.groupby('home')['season'].nunique()
    one_season_mask = season_count == 1
    season_count = season_count[one_season_mask]
    delete_one_season_mask = ~matches_df['home'].isin(season_count)
    matches_df = matches_df[delete_one_season_mask]

    # Filtering season date
    season_mask = matches_df['season'] >= season
    matches_df = matches_df[season_mask]

    # Filtering club name
    club_home_mask = matches_df['home'] == club_name
    club_away_mask = matches_df['away'] == club_name
    club_mask = club_home_mask | club_away_mask
    matches_df = matches_df[club_mask]

    # Adding new column is_home, that tells if match was in home stadium
    matches_df['is_home'] = club_home_mask.astype(int)

    # Creating result masks
    win_home_mask = (matches_df['is_home'] == 1) & (matches_df['gh'] > matches_df['ga'])
    win_away_mask = (matches_df['is_home'] == 0) & (matches_df['ga'] > matches_df['gh'])
    win_mask = win_home_mask | win_away_mask
    draw_mask = matches_df['ga'] == matches_df['gh']

    # Creating new column that show the result, 2 = win, 1 = draw, 0 = loss
    conditions = [win_mask, draw_mask]
    choices = [2, 1]
    matches_df['result'] = np.select(conditions, choices, default=0)

    # Deleting goals columns, they are unnecessary now
    matches_df = matches_df.drop(columns = ['gh', 'ga'])

    # Adding opponent column
    matches_df['opponent'] = np.where(
        matches_df['is_home'] == 1,
        matches_df['away'],
        matches_df['home']
    )

    # Deleting home and away columns, they are unnecessary now
    matches_df = matches_df.drop(columns = ['home', 'away'])

    # TODO counting last 5 results of matches
    #matches_df['last_results'] =

    return matches_df
