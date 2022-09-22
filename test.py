import pandas as pd

from torch_geometric.data import download_url, extract_zip

df = pd.read_csv('./data/Cora/group-edges.csv', header=None, names=['id', 'types'])
print(df.head())
df['count'] = 1

print(df.groupby('types').agg({'count': 'sum'}))

print(len(set(df['types'])))


