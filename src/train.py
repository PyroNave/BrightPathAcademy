from paths import model_path, data_path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from custom_model import CustomModel
import joblib


df = pd.read_csv(data_path)


# Data cleaning
def GPAGradeClass(GPA: float) -> int:
    if GPA < 2:
        return 4
    elif GPA < 2.5:
        return 3
    elif GPA < 3:
        return 2
    elif GPA < 3.5:
        return 1
    else:
        return 0


df['GradeClass'] = [GPAGradeClass(x) for x in df['GPA']]

X = df.drop(['GradeClass', 'GPA', 'StudentID'], axis=1)
y = df['GPA']

model = CustomModel(LinearRegression())
model.fit(X, y)

joblib.dump(model, model_path)
