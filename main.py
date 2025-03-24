import pandas as pd

def filter_by_year(df, year):
    return df[df['timestamp'].dt.year == year]

df_17 = pd.DataFrame()
df_23 = pd.DataFrame()

for i in range(1, 35):
    print(i) 

    df = pd.read_json(f'part{i}.jsonl', lines=True)
    
    filtered_17 = filter_by_year(df, 2017)
    filtered_23 = filter_by_year(df, 2023)

    print(len(filtered_17.index))
    print(len(filtered_23.index))

    df_17 = pd.concat([df_17, filtered_17])
    df_23 = pd.concat([df_23, filtered_23])

df_17.to_csv('2017.csv')
df_23.to_csv('2023.csv')