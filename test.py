import pandas as pd
import numpy as np

# --- Uproszczony przykład danych (jak poprzednio) ---
# To symuluje matches_df po wczytaniu, przefiltrowaniu i obliczeniu kolumn 'result', posortowane wg 'Id'
data = {
    'Id': [100, 101, 102, 103, 104, 105],
    'home': ['A', 'B', 'A', 'B', 'A', 'B'],
    'away': ['B', 'A', 'B', 'A', 'B', 'A'],
    'result': [1, -1, 0, 1, 0, -1] # wynik z perspektywy drużyny 'home' (1=win, 0=draw, -1=loss)
}
matches_df_simple = pd.DataFrame(data)
# Symulujemy, że team_id to zakodowane A=0, B=1
team_map = {'A': 0, 'B': 1}
matches_df_simple['home_id'] = matches_df_simple['home'].map(team_map)
matches_df_simple['away_id'] = matches_df_simple['away'].map(team_map)
matches_df_simple = matches_df_simple[['Id', 'home_id', 'away_id', 'result']].copy() # Upraszczamy kolumny

print("--- Krok 0: Nasze przykładowe dane meczy (matches_df_simple) ---")
print(matches_df_simple)
print("-" * 30)

# --- Krok 1: Przygotowanie danych do rolling (jeden wiersz na występ zespołu w meczu) ---
home_results = matches_df_simple[['Id', 'home_id', 'result']].copy()
home_results = home_results.rename(columns={'home_id': 'team_id'})
home_results['is_home_flag'] = 1

away_results = matches_df_simple[['Id', 'away_id', 'result']].copy()
away_results = away_results.rename(columns={'away_id': 'team_id'})
away_results['result'] *= -1
away_results['is_home_flag'] = 0

print("--- Krok 1a: Wyniki z perspektywy gospodarzy (home_results) ---")
print(home_results)
print("-" * 30)
print("--- Krok 1b: Wyniki z perspektywy gości (away_results) ---")
print(away_results)
print("-" * 30)

# --- Krok 2: Łączymy dane 'home' i 'away' ---
team_results_for_rolling = pd.concat([home_results, away_results]).reset_index(drop=True)

print("--- Krok 2: Połączone dane (team_results_for_rolling), 2 wiersze na mecz, prosty index ---")
print(team_results_for_rolling)
print("-" * 30)

# --- Krok 3: Sortujemy dane po zespole i id meczu ---
team_results_for_rolling = team_results_for_rolling.sort_values(by=['team_id', 'Id'])

print("--- Krok 3: Połączone dane POSORTOWANE wg zespołu i id meczu ---")
print(team_results_for_rolling)
print("-" * 30)

# --- Krok 4-5: Obliczamy rolling sum wyników per zespół, shift, fillna, astype ---
# Grupowanie po team_id i obliczanie rolling
# Wynik rolling ma MultiIndex (team_id, index_z_team_results_for_rolling_przed_sortowaniem)
rolling_form_series = team_results_for_rolling.groupby('team_id')['result'] \
    .rolling(window=5, min_periods=1). \
    sum() \
    .shift(1). \
    fillna(0). \
    astype(int)

print("--- Krok 5: Wynik rolling po shift/fillna/astype, z MultiIndex ---")
print(rolling_form_series)
print("-" * 30)

# --- Krok 6: Przygotowujemy wynik rolling do połączenia z oryginalnymi danymi meczy ---

# Konwertujemy serię rolling na DataFrame. MultiIndex stanie się kolumnami.
# Domyślne nazwy kolumn to prawdopodobnie: team_id, level_1 (lub inna nazwa poziomu indexu), 'result' (nazwa oryginalnej serii)
rolling_form_df = rolling_form_series.reset_index()

print("--- Krok 6a: Wynik rolling jako DataFrame (przed zmianą nazw) ---")
print(rolling_form_df.head())
print("Kolumny po reset_index():", rolling_form_df.columns)
print("Spodziewane kolumny to: nazwa_team_id_z_indexu, nazwa_poziomu_indexu, 'result'")
print("-" * 30)

# Zmieniamy nazwy kolumn. Zmieniamy nazwę kolumny z oryginalnym indeksem
# oraz nazwę kolumny z wartością rolling, która prawdopodobnie nazywa się 'result'.
# Sprawdź wydruk z Krok 6a, aby upewnić się co do domyślnych nazw poziomów indexu ('level_0', 'level_1' itp.)
# Domyślnie pierwszy poziom to nazwa grupy (team_id), drugi to oryginalny index ('level_1').
# Wartość to 'result'.
rolling_form_df = rolling_form_df.rename(columns={
    'level_1': 'index_in_concat_df_before_sort', # Nazwa drugiego poziomu MultiIndexu
    'result': 'rolling_form_value' # Nazwa kolumny z wartością rolling
})

print("--- Krok 6b: Wynik rolling jako DataFrame (PO zmianie nazw) ---")
print(rolling_form_df.head())
print("Kolumny po zmianie nazw:", rolling_form_df.columns)
print("-" * 30)


# --- Krok 7: Łączymy wynik rolling z DataFrame, który zawiera oryginalne Id meczu i flagę is_home ---
# Używamy team_results_for_rolling (po concat i reset_index, ale PRZED sort_values) jako bazę.
# Ma on prosty index numeryczny (0, 1, 2...) i kolumny Id, team_id, result, is_home_flag.

# Odtwarzamy DataFrame 'team_results_for_rolling' sprzed sortowania, z prostym indexem numerycznym
team_results_for_rolling_before_sort = pd.concat([home_results, away_results]).reset_index(drop=True)


# Łączymy wyniki rolling ('rolling_form_df') z DataFrame zawierającym Id i is_home_flag ('team_results_for_rolling_before_sort').
# Używamy indexu lewego DF ('team_results_for_rolling_before_sort', który jest prostym indexem numerycznym) do połączenia
# z kolumną 'index_in_concat_df_before_sort' z prawego DF ('rolling_form_df').
team_results_with_rolling = pd.merge(
    team_results_for_rolling_before_sort, # Lewy DF (index to 0, 1, 2...)
    rolling_form_df[['index_in_concat_df_before_sort', 'rolling_form_value']], # Prawy DF (wybrane kolumny)
    left_index=True, # Użyj indeksu lewego DF do łączenia
    right_on='index_in_concat_df_before_sort', # Użyj tej kolumny z prawego DF do łączenia
    how='left' # Zachowaj wszystkie wiersze z lewego DF
)

# Usuwamy kolumnę 'index_in_concat_df_before_sort' używaną do łączenia (nie jest już potrzebna)
team_results_with_rolling = team_results_with_rolling.drop(columns=['index_in_concat_df_before_sort'])

print("--- Krok 7: Połączone dane z wynikiem rolling i oryginalnym Id/is_home ---")
print(team_results_with_rolling.head())
print("Sprawdź, czy kolumna 'rolling_form_value' jest obecna.")
print("-" * 30)

# --- Krok 8: Przekształcamy dane do formatu 1 wiersz na mecz (pivot_table) ---
final_rolling_forms_reshaped = team_results_with_rolling.pivot_table(
    index='Id',          # Index będzie Id meczu
    columns='is_home_flag', # 0 i 1 stają się nazwami kolumn
    values='rolling_form_value' # Wartości w tych kolumnach to obliczona forma
)

print("--- Krok 8: Wynik rolling przekształcony (pivot), 1 wiersz na mecz ---")
print(final_rolling_forms_reshaped.head())
print("-" * 30)

# --- Krok 9: Zmieniamy nazwy kolumn ---
final_rolling_forms_reshaped = final_rolling_forms_reshaped.rename(columns={0: 'last_results_away', 1: 'last_results_home'})

print("--- Krok 9: Wynik rolling ze zmienionymi nazwami kolumn ---")
print(final_rolling_forms_reshaped.head())
print("-" * 30)

# --- Krok 10: Dołączamy te kolumny z powrotem do oryginalnego DataFrame meczów ---
# Upewniamy się, że final_rolling_forms_reshaped ma Id jako kolumnę, nie index
final_rolling_forms_reshaped = final_rolling_forms_reshaped.reset_index()

matches_df_final = pd.merge(matches_df_simple, final_rolling_forms_reshaped, on='Id', how='left')

print("--- Krok 10: Ostateczny DataFrame z dołączonymi kolumnami last_results_home/away ---")
print(matches_df_final.head())
print("-" * 30)
