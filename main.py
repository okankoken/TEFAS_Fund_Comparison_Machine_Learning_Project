import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


#Dataset has been created between 16.09.2019 - 16.09.2024 dates
df = pd.read_excel("datasets/TEFAS_Fund_Comparison.xlsx" , sheet_name="Meta_Data")

df
df.head(10)
df.info
df.shape
df.dtypes
df.columns
df.describe().T
def check_df(dataframe, head=20):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.info)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    #print("##################### Quantiles #####################")
    #print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

fon_türü_counts = df.groupby('Fon Türü').agg({'Fon Kodu': 'count'})

fon_türü_counts.sort_values(by='Fon Kodu', ascending=False)
