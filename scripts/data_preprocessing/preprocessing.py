import pandas as pd
import os
X_Train = pd.read_csv('data/stage_1/X_Train.csv', index_col = 0)
X_test = pd.read_csv('data/stage_1/X_test.csv', index_col = 0)
y_Train = pd.read_csv('data/stage_1/y_Train.csv', index_col = 0)
y_test = pd.read_csv('data/stage_1/y_test.csv', index_col = 0)


cat_columns = []
num_columns = []

for column_name in X_Train.columns:
    if (X_Train[column_name].dtypes == object):
        cat_columns +=[column_name]
    else:
        num_columns +=[column_name]

print('categorical columns:\t ',cat_columns, '\n len = ',len(cat_columns))

print('numerical columns:\t ',  num_columns, '\n len = ',len(num_columns))

from sklearn.impute import SimpleImputer # Объект для замены пропущенных значений
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # Импортируем нормализацию и One-Hot Encoding от scki-kit-learn
from sklearn.pipeline import Pipeline # Pipeline.Не добавить, не убавить
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок

numerical_pipe = Pipeline([
    ('scaler', MinMaxScaler())   #нормализуем числовые минмакс скалером
])

categorical_pipe = Pipeline([
    ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False)) #категориальные кодируем ванхотом
])

preprocessors = ColumnTransformer(transformers=[   #преобразуем в красоту
    ('num', numerical_pipe, num_columns),
    ('cat', categorical_pipe, cat_columns)
])

preprocessors.fit(X_Train)

X_Train = preprocessors.transform(X_Train) # преобразуем  тренировочные данные

X_test = preprocessors.transform(X_test) # преобразуем  тестовые данные

X_Train = pd.DataFrame(X_Train)
X_test = pd.DataFrame(X_test)

os.makedirs(os.path.join("data", "stage_2"),exist_ok=True)

X_Train.to_csv('data/stage_2/X_Train.csv')
X_test.to_csv('data/stage_2/X_test.csv')
y_Train.to_csv('data/stage_2/y_Train.csv')
y_test.to_csv('data/stage_2/y_test.csv')