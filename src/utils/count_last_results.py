import pandas as pd

def count_last_results(matches_df):
    # Create home and away copy, adding columns that say if the match was in home,
    # multiply away results by -1 to match them with actual team result,
    # and change home and away columns names to same
    matches_df_home_copy = matches_df[['Id', 'date', 'home', 'result']].copy()
    matches_df_home_copy['result'] = matches_df_home_copy['result'].map({2: 1, 1: 0, 0: -1})
    matches_df_home_copy = matches_df_home_copy.rename(columns={'home': 'team_id'})
    matches_df_home_copy['is_home'] = 1

    matches_df_away_copy = matches_df[['Id', 'date', 'away', 'result']].copy()
    matches_df_away_copy = matches_df_away_copy.rename(columns={'away': 'team_id'})
    matches_df_away_copy['result'] *= -1
    matches_df_away_copy['is_home'] = 0

    # Creating one df that is sum of away and home copy
    matches_df_copy = pd.concat([matches_df_away_copy, matches_df_home_copy])

    # Sorting by team_id and matchId
    matches_df_copy = matches_df_copy.sort_values(by=['team_id', 'date', 'Id']).reset_index(drop=True)

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
    matches_df_copy = matches_df_copy.sort_values(by=['Id', 'date'])

    # Creating new columns that say last_results of home and away team in matchId
    matches_df_copy = matches_df_copy.set_index(['Id', 'is_home'])['last_results'].unstack()
    matches_df_copy = matches_df_copy.rename(columns={0: 'last_results_away', 1: 'last_results_home'})

    # Resetting indexes of matches_df and matches_df_copy and concatenating them
    matches_df_copy = matches_df_copy.reset_index(drop=True)

    return matches_df_copy
