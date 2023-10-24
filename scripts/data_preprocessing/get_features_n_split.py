import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv('data/raw/wines_SPA.csv', delimiter = ',')
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
df.drop(columns=['country','acidity'],inplace=True)
X_Train, X_test, y_Train, y_test = train_test_split(df.drop(columns=['price']), df.price.values.ravel(), test_size=0.3, random_state=42)
os.makedirs(os.path.join("data", "stage_1"),exist_ok=True)
X_Train.to_csv('data/stage_1/X_Train.csv')
X_test.to_csv('data/stage_1/X_test.csv')
y_Train = pd.DataFrame(y_Train)
y_test = pd.DataFrame(y_test)
y_Train.to_csv('data/stage_1/y_Train.csv')
y_test.to_csv('data/stage_1/y_test.csv')