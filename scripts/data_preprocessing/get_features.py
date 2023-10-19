import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/wines_SPA.csv', delimiter = ',')
df['year'] = df['year'].replace('N.V.', np.NaN)  #заменяем в датафрейме реплейсом
df['body'] = df['body'].replace('<NA>', np.NaN)
df['acidity'] = df['acidity'].replace('<NA>', np.NaN)
df = df.dropna().reset_index(drop=True)
df = df.drop_duplicates().reset_index(drop=True)
df = df.astype({"body": "Int64"}) #изменяем тип
df_year_num = df.astype({"year": "Int64"})
df_year_num['extract'] = [2022-x for x in df_year_num["year"]] #заполняем новосозданый столбец, пробегая по значениям года
df_year_num.drop(columns = ['year'],inplace=True)  #Удаляем год
df = df_year_num
df.drop(columns=['country','year','acidity'],inplace=True)
