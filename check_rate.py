import pandas as pd

df = pd.read_csv(r'C:\Users\nvanwieren\Downloads\open_job_ops 3-17-2026.csv')
print(df[['partNumber','operationCode','quantity','quantityCompleted','runTime','rate','setupTime']].dropna(subset=['rate']).head(20).to_string())
print('\nrate stats:', df['rate'].describe())
print('runTime stats:', df['runTime'].describe())
