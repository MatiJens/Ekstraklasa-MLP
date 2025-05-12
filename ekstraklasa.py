from load_data import load_data_from_csv

csv_path = "data/poland_1.csv"
data = load_data_from_csv(csv_path, "Legia Warszawa", 2000)

print(data.head())