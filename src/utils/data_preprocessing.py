import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def data_preprocessing(csv_path, season):
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
    season_mask = matches_df['season'] >= season
    matches_df = matches_df[season_mask]

    # Deleting teams that play only one season in Ekstraklasa
    season_count = matches_df.groupby('home')['season'].nunique()
    one_season_mask = season_count == 1
    season_count = season_count[one_season_mask]
    delete_one_season_mask = ~matches_df['home'].isin(season_count)
    matches_df = matches_df[delete_one_season_mask]

    # Sort by date and reset index after deleting teams
    matches_df = matches_df.sort_values(by='Id').reset_index(drop=True)

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

    # Create home and away copy, adding columns that say if the match was in home,
    # multiply away results by -1 to match them with actual team result,
    # and change home and away columns names to same
    matches_df_home_copy = matches_df[['Id', 'home', 'result']].copy()
    matches_df_home_copy['result'] = matches_df_home_copy['result'].map({2: 1, 1: 0, 0: -1})
    matches_df_home_copy = matches_df_home_copy.rename(columns={'home': 'team_id'})
    matches_df_home_copy['is_home'] = 1

    matches_df_away_copy = matches_df[['Id', 'away', 'result']].copy()
    matches_df_away_copy = matches_df_away_copy.rename(columns={'away': 'team_id'})
    matches_df_away_copy['result'] *= -1
    matches_df_away_copy['is_home'] = 0

    # Creating one df that is sum of away and home copy
    matches_df_copy = pd.concat([matches_df_away_copy, matches_df_home_copy])

    # Sorting by team_id and matchId
    matches_df_copy = matches_df_copy.sort_values(by=['team_id', 'Id']).reset_index(drop=True)

    # Rolling by result and sum to get form of every unique team
    last_results = matches_df_copy.groupby('team_id')['result'] \
        .rolling(window=5, min_periods=1) \
        .sum() \
        .shift(1) \
        .fillna(0) \
        .astype(int) \
        .reset_index(level=0, drop=True)

    # Creating new data frame with last results and concatenating it with matches_df_copy
    last_results_data = {'last_results': last_results}
    last_results_df = pd.DataFrame(last_results_data)

    matches_df_copy = pd.concat([matches_df_copy, last_results_df], axis=1)

    # Zeroing last_results of every first match of every team (It is caused by .shift(1))
    first_occurrence_team = matches_df_copy.drop_duplicates(subset=['team_id']).index
    matches_df_copy.loc[first_occurrence_team, 'last_results'] = 0

    # Sorting matches by matchId
    matches_df_copy = matches_df_copy.sort_values(by='Id')

    # Creating new columns that say last_results of home and away team in matchId
    matches_df_copy = matches_df_copy.set_index(['Id', 'is_home'])['last_results'].unstack()
    matches_df_copy = matches_df_copy.rename(columns={0: 'last_results_away', 1: 'last_results_home'})

    # Resetting indexes of matches_df and matches_df_copy and concatenating them
    matches_df_copy = matches_df_copy.reset_index(drop=True)
    matches_df = matches_df.sort_values(by="Id").reset_index(drop=True)
    matches_df = pd.concat([matches_df, matches_df_copy], axis=1)

    # Delete unnecessary columns
    matches_df = matches_df.drop(columns=['Id' ,'IsCancelled', 'IsWalkover', 'season', 'hour', 'note', 'gh', 'ga', 'date'])

    # Swapping order of columns
    swap_columns_titles = ['matchday', 'home', 'away', 'last_results_home', 'last_results_away', 'result', 'goals']
    matches_df = matches_df.reindex(columns=swap_columns_titles)

    return matches_df, unique_teams_map
