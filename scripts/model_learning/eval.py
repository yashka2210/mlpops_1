from sklearn.metrics import mean_squared_error as mse, r2_score
import pandas as pd
import pickle
import json

SVM_reg = pickle.load(open("models/model.pkl","rb"))
X_test = pd.read_csv("data/stage_2/X_test.csv", index_col=0)
y_test = pd.read_csv("data/stage_2/y_test.csv", index_col=0)
y_predict=SVM_reg.predict(X_test)
r2 = r2_score(y_test,y_predict)
with open("evaluate/r2score.json","w") as ev:
    json.dump({"r2_score": r2}, ev)
